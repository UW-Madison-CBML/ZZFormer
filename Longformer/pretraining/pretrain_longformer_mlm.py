import os
import gc
import yaml
import pickle
import random
import argparse
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader

from transformers import LongformerConfig, LongformerForMaskedLM
from transformers import get_linear_schedule_with_warmup

from data.dataloader import apply_mlm_mask_gpu   # reuse the exact masker you already use

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Vocab
# ============================================================
VOCAB = {
    "PAD":  0,
    "a":    1, "c": 2, "g": 3, "t": 4,
    "x":    5,
    "BOS":  6,
    "EOS":  7,
    "MASK": 8,
}
VOCAB_SIZE     = len(VOCAB)
PAD_TOKEN_ID   = VOCAB["PAD"]
BOS_TOKEN_ID   = VOCAB["BOS"]
EOS_TOKEN_ID   = VOCAB["EOS"]
MASK_TOKEN_ID  = VOCAB["MASK"]
UNK_TOKEN_ID   = VOCAB["x"]
IGNORE_INDEX   = -100


# ============================================================
# Dataset: takes either a pickle dict {seq: anything}  or a list of sequences.
# Labels are not needed for MLM — we only need the tokens.
# ============================================================
class NucleotideMLMDataset(Dataset):
    def __init__(self, seqs, max_seq_len):
        self.max_seq_len = max_seq_len

        # Accept dict ({seq: label}) or list/tuple of seqs
        if isinstance(seqs, dict):
            self.seqs = list(seqs.keys())
        else:
            self.seqs = list(seqs)
        self.seqs = [s.lower() for s in self.seqs]
        print(f"  MLM dataset: {len(self.seqs)} sequences")

        self.input_ids, self.padding_mask = self._tokenize_all()

    def _tokenize_all(self):
        N, L = len(self.seqs), self.max_seq_len
        input_ids   = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
        padding_mask = torch.ones((N, L), dtype=torch.bool)   # True = PAD (PyTorch convention)

        body_max = L - 2
        for i, seq in enumerate(self.seqs):
            body = [VOCAB.get(c, UNK_TOKEN_ID) for c in seq[:body_max]]
            ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
            input_ids[i, :len(ids)]    = torch.tensor(ids, dtype=torch.long)
            padding_mask[i, :len(ids)] = False                # real tokens
        return input_ids, padding_mask

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.padding_mask[idx]


# ============================================================
# Build the MLM model
# ============================================================
def build_longformer_mlm(cfg):
    m = cfg["model"]
    lcfg = LongformerConfig(
        attention_window             = m["attention_window"],
        vocab_size                   = m["vocab_size"],
        max_position_embeddings      = m["max_position_embeddings"],
        hidden_size                  = m["d_model"],
        num_hidden_layers            = m["num_layers"],
        num_attention_heads          = m["nhead"],
        intermediate_size            = m["dim_feedforward"],
        position_embedding_type      = m.get("position_embedding_type", "absolute"),
        return_dict                  = True,
        pad_token_id                 = PAD_TOKEN_ID,
        bos_token_id                 = BOS_TOKEN_ID,
        eos_token_id                 = EOS_TOKEN_ID,
        hidden_dropout_prob          = m["dropout"],
        attention_probs_dropout_prob = m["dropout"],
    )
    return LongformerForMaskedLM(lcfg)


# ============================================================
# Diagnostics — same shape as your old log
# ============================================================
def log_param_diagnostics(model, f):
    f.write(f"{'Parameter':<60} {'Shape':<20} {'Param Norm':>12} "
            f"{'Grad Norm':>12} {'Grad Max':>12}\n")
    f.write("-" * 120 + "\n")
    for name, p in model.named_parameters():
        if p.grad is not None:
            f.write(
                f"{name:<60} {str(list(p.shape)):<20} "
                f"{p.data.norm().item():>12.6f} "
                f"{p.grad.data.norm().item():>12.6f} "
                f"{p.grad.data.abs().max().item():>12.6e}\n"
            )


# ============================================================
# Train / eval loops — mirror your previous MLM loop
# ============================================================
def run_train(model, dataloader, optimizer, scheduler, cfg, path=None, epoch=None):
    model.train()
    total_loss = 0.0
    mask_prob          = cfg["mlm"]["mask_prob"]
    mask_ignore_ids    = tuple(cfg["mlm"]["mask_ignore_token_ids"])
    # Also keep BOS/EOS/MASK out of replacement candidates:
    mask_ignore_ids = tuple(set(mask_ignore_ids) | {BOS_TOKEN_ID, EOS_TOKEN_ID, MASK_TOKEN_ID})

    for tokens, src_key_padding_mask in dataloader:
        tokens               = tokens.to(DEVICE, non_blocking=True)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # --- Mask on GPU (same fn as your old code) ---
        masked_tokens, mlm_labels = apply_mlm_mask_gpu(
            tokens=tokens,
            mask_prob=mask_prob,
            pad_token_id=PAD_TOKEN_ID,
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            mask_ignore_token_ids=mask_ignore_ids,
            ignore_index=IGNORE_INDEX,
        )

        # --- HF conventions ---
        attention_mask        = (~src_key_padding_mask).long()   # 1 = attend
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1                           # global on BOS

        out = model(
            input_ids=masked_tokens,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=mlm_labels,                                    # -100 positions ignored
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += out.loss.item()

    avg_loss = total_loss / len(dataloader)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\nepoch {epoch}| Total train_loss | {avg_loss:.6f}\n")
            log_param_diagnostics(model, f)
    return avg_loss


@torch.no_grad()
def run_val(model, dataloader, cfg):
    model.eval()
    total_loss, total_correct, total_masked = 0.0, 0, 0
    mask_prob       = cfg["mlm"]["mask_prob"]
    mask_ignore_ids = tuple(cfg["mlm"]["mask_ignore_token_ids"])
    mask_ignore_ids = tuple(set(mask_ignore_ids) | {BOS_TOKEN_ID, EOS_TOKEN_ID, MASK_TOKEN_ID})

    for tokens, src_key_padding_mask in dataloader:
        tokens               = tokens.to(DEVICE, non_blocking=True)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)

        masked_tokens, mlm_labels = apply_mlm_mask_gpu(
            tokens=tokens,
            mask_prob=mask_prob,
            pad_token_id=PAD_TOKEN_ID,
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            mask_ignore_token_ids=mask_ignore_ids,
            ignore_index=IGNORE_INDEX,
        )
        attention_mask        = (~src_key_padding_mask).long()
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1

        out = model(
            input_ids=masked_tokens,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=mlm_labels,
        )
        total_loss += out.loss.item()

        preds   = out.logits.argmax(-1)
        active  = mlm_labels != IGNORE_INDEX
        total_correct += (preds[active] == mlm_labels[active]).sum().item()
        total_masked  += active.sum().item()

    avg_loss = total_loss / len(dataloader)
    acc      = total_correct / max(1, total_masked)
    ppl      = float(np.exp(min(20.0, avg_loss)))   # capped for safety
    print(f"  Val | loss {avg_loss:.4f} | acc {acc:.4f} | ppl {ppl:.2f}")
    return {"val_loss": avg_loss, "val_acc": acc, "val_ppl": ppl}



def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    checkpoint_path=None,
    device="cpu",
    strict=True,
):
    """
    Restore model / optimizer / scheduler state from a .pt saved by this script.

    Returns:
        model, optimizer, scheduler,
        start_epoch  (= saved_epoch + 1),
        best_loss    (or +inf if not in ckpt)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if missing:
        print(f"  ⚠️  missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  ⚠️  unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    if optimizer is not None and "optim_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
        print("  ✓ optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print("  ✓ scheduler state restored")

    saved_epoch = ckpt.get("epoch", -1)
    start_epoch = saved_epoch + 1
    best_loss   = ckpt.get("best_loss", float("inf"))

    print(f"  → resuming at epoch {start_epoch} | best_loss so far = {best_loss:.4f}")
    return model, optimizer, scheduler, start_epoch, best_loss




# ============================================================
# Main
# ============================================================
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Sanity-check vocab size
    assert cfg["model"]["vocab_size"] == VOCAB_SIZE, (
        f"config vocab_size ({cfg['model']['vocab_size']}) != script VOCAB_SIZE ({VOCAB_SIZE})"
    )

    if not args.debugging:
        wandb.init(
            name=args.run_name or "longformer_mlm",
            entity=args.wandb_team    or cfg["wandb"]["team"],
            project=args.wandb_project or cfg["wandb"]["project"],
            dir=args.wandb_dir         or cfg["wandb"]["dir"],
            config=cfg,
        )

    # --- Data ---
    with open(args.train_dir, "rb") as f:
        train_seqs = pickle.load(f)
    val_seqs = None
    if args.val_dir:
        with open(args.val_dir, "rb") as f:
            val_seqs = pickle.load(f)

    if args.debugging:
        from itertools import islice
        if isinstance(train_seqs, dict): train_seqs = dict(islice(train_seqs.items(), 200))
        if isinstance(val_seqs,   dict): val_seqs   = dict(islice(val_seqs.items(),   50))

    max_seq_len = cfg["model"]["max_seq_len"]
    train_ds = NucleotideMLMDataset(train_seqs, max_seq_len)
    val_ds   = NucleotideMLMDataset(val_seqs,   max_seq_len) if val_seqs is not None else None

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batchsize"],
                              shuffle=True,  num_workers=cfg["train"]["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batchsize"],
                              shuffle=False, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True) if val_ds is not None else None

    # --- Model / optim / scheduler ---
    model = build_longformer_mlm(cfg).to(DEVICE)
    print(f"Longformer MLM params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["train"].get("warmup_steps", 0),
        num_training_steps=total_steps,
    )

    # --- Save paths ---
    save_dir = args.save_dir or cfg["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"longformer_mlm_{args.run_name}.pt")
    log_path  = os.path.join(save_dir, f"{args.run_name}_mlm_diagnostics.txt")

    # --- Train ---
    # --- Auto-resume if a checkpoint exists at save_path ---
    start_epoch = 0
    best_loss   = float("inf")
    if os.path.isfile(save_path):
        print(f"Found existing checkpoint at {save_path} — resuming")
        model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler,
            checkpoint_path=save_path,
            device=DEVICE,
        )
        if start_epoch >= cfg["train"]["epochs"]:
            print(f"Already trained {start_epoch} epochs ≥ target {cfg['train']['epochs']}. "
                f"Nothing to do.")
            return

    # --- Train ---
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss = run_train(
            model, train_loader, optimizer, scheduler, cfg,
            path=log_path, epoch=epoch,
        )
        wandb.log({"train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch})

        gc.collect()
        torch.cuda.empty_cache()
    



    val_out = run_val(model, val_loader, cfg) if val_loader is not None else {"val_loss": float("inf"), "val_acc": 0.0}
    print(f"Epoch {epoch:03d} | train {train_loss:.4f} | "
            f"val {val_out['val_loss']:.4f} | acc {val_out['val_acc']:.4f}")

    if not args.debugging:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch":     epoch,
            "best_loss": best_loss,
            "config":    cfg,
            "vocab":     VOCAB,
        }, save_path)
        print(f"  ↳ new best (val_loss {best_loss:.4f}) saved to {save_path}")







        
# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--train_dir",  required=True, help="Pickle dict OR list of sequences for MLM train")
    p.add_argument("--val_dir",    required=False, help="Same shape for MLM val")
    p.add_argument("--save_dir",   default=None)
    p.add_argument("--run_name",   default="longformer_mlm")
    p.add_argument("--seed",       default=22, type=int)
    p.add_argument("--debugging",  action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_team",    default=None)
    p.add_argument("--wandb_dir",     default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    main(args)