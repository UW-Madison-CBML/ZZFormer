import os
import gc
import yaml
import argparse
import random
import pickle
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader

from transformers import LongformerConfig, LongformerForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Label map — FLAT (Longformer is single-label classification)
# ============================================================
ORDER_TO_SUPERFAMILIES={'LTR': ['Pao', 'Gypsy', 'Copia', 'DIRS', 'Caulimovirus', 'ERV'],
'DNA': ['Harbinger', 'CMC', 'P', 'hAT', 'TcMar', 'PiggyBac', 'Zator', 'MULE', 'Merlin', 'Kolobok', 'Maverick', 'Novosib', 'Zisupton', 'Crypton', 'Academ', 'IS3EU', 'Dada', 'Sola', 'Ginger'],
'LINE': ['R1', 'I', 'CR1', 'L1', 'RTE', 'L2', 'Dong-R4', 'R2', 'Dualen', 'CRE', 'Tad1', 'Rex-Babar', 'Proto2', 'Proto1'],
'Satellite': [],
'RC': ['Helitron'],
'SINE': ['tRNA', '5S', '7SL', 'U'],
'Structural_RNA': [],
'PLE': [],
'Other': [],
}


def build_flat_label_map(order_to_sf):
    """
    Build a flat class index over every (order) and (order/sf) label.
    Returns:
      label_to_id : {"Line": 0, "Line/CR1": 1, ...}
      id_to_label : reverse
      num_labels  : int
    """
    label_to_id = {}
    for order, sfs in order_to_sf.items():
        label_to_id[order] = len(label_to_id)
        for sf in sfs:
            label_to_id[f"{order}/{sf}"] = len(label_to_id)
    id_to_label = {i: k for k, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_to_id)







def load_pretrained_longformer_mlm(pretrained_path, classifier_model):
    """
    Transfer the Longformer backbone (embeddings + encoder) from a
    LongformerForMaskedLM checkpoint into a LongformerForSequenceClassification.

    Transferred  : longformer.embeddings.*  +  longformer.encoder.*
    Dropped      : lm_head.*                (MLM-only)
    Random init  : classifier.*             (classification head, new)
    """
    print(f"Loading pretrained MLM weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location="cpu")
    sd   = ckpt.get("model_state_dict", ckpt)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("longformer."):
            new_sd[k] = v          # entire backbone
        elif k.startswith("lm_head."):
            continue               # drop MLM head
        else:
            print(f"  skipping unrecognized key: {k}")

    missing, unexpected = classifier_model.load_state_dict(new_sd, strict=False)

    expected_missing   = [k for k in missing if k.startswith("classifier.")]
    unexpected_missing = [k for k in missing if k not in expected_missing]

    print(f"\n--- MLM → Classifier transfer ---")
    print(f"  ✓ transferred {len(new_sd)} backbone keys")
    print(f"  classifier head missing (expected, will be randomly initialized): "
          f"{len(expected_missing)}")
    for k in expected_missing:
        print(f"      {k}")

    if unexpected_missing:
        print(f"  ⚠️  UNEXPECTED missing keys ({len(unexpected_missing)}):")
        for k in unexpected_missing[:10]:
            print(f"      {k}")
    if unexpected:
        print(f"  ⚠️  unexpected keys in checkpoint ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"      {k}")

    return classifier_model







# ============================================================
# Dataset — yields HF-compatible (input_ids, attention_mask, labels)
# ============================================================
class NucleotideClassificationDataset(Dataset):
    """
    Input pickle: { seq_str : (order, superfamily_or_None) }

    Tokenizes char-by-char, prepends BOS, appends EOS, pads to max_seq_len.
    Returns dict batches HF Trainer-style.
    """
    def __init__(self, seq_dict, label_to_id, max_seq_len):
        self.label_to_id = label_to_id
        self.max_seq_len = max_seq_len  # includes BOS + EOS slots

        self.seqs, self.labels = [], []
        skipped = 0
        for seq, lbl in seq_dict.items():
            order, sf = lbl
            name = f"{order}/{sf}" if sf else order
            if name not in label_to_id:
                skipped += 1
                continue
            self.seqs.append(seq.lower())
            self.labels.append(label_to_id[name])

        print(f"  loaded {len(self.seqs)} samples (skipped {skipped} unmapped)")

        # Pre-tokenize once for speed
        self.input_ids, self.attention_masks = self._tokenize_all()
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _tokenize_all(self):
        N = len(self.seqs)
        L = self.max_seq_len
        input_ids      = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((N, L), dtype=torch.long)

        max_body = L - 2  # leave room for BOS + EOS
        for i, seq in enumerate(self.seqs):
            body = [VOCAB.get(c, UNK_TOKEN_ID) for c in seq[:max_body]]
            ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
            input_ids[i, :len(ids)]      = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        return input_ids, attention_mask

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels":         self.labels[idx],
        }


# ============================================================
# Model builder — mirrors TEClass2's get_model()
# ============================================================
def build_longformer(vocab_size, num_labels, cfg):
    m = cfg["model"]
    longformer_config = LongformerConfig(
        attention_window             = m["attention_window"],
        vocab_size                   = vocab_size,
        max_position_embeddings      = m["max_position_embeddings"],
        num_labels                   = num_labels,
        hidden_size                  = m["d_model"],
        num_hidden_layers            = m["num_layers"],
        num_attention_heads          = m["nhead"],
        intermediate_size            = m["dim_feedforward"],
        position_embedding_type      = m.get("position_embedding_type", "absolute"),
        problem_type                 = "single_label_classification",
        return_dict                  = True,
        pad_token_id                 = PAD_TOKEN_ID,
        bos_token_id                 = BOS_TOKEN_ID,
        eos_token_id                 = EOS_TOKEN_ID,
        hidden_dropout_prob          = m["dropout"],
        attention_probs_dropout_prob = m["dropout"],
    )
    return LongformerForSequenceClassification(longformer_config)


# ============================================================
# Training / validation loops
# ============================================================
def _add_global_attention(attention_mask):
    """Longformer needs global attention on at least one token.
       Convention: first token (the BOS) acts like [CLS]."""
    g = torch.zeros_like(attention_mask)
    g[:, 0] = 1
    return g


def run_train(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)
        global_attention_mask = _add_global_attention(attention_mask)

        optimizer.zero_grad()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=labels,
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out.loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def run_val(model, dataloader, id_to_label):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)
        global_attention_mask = _add_global_attention(attention_mask)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=labels,
        )
        total_loss += out.loss.item()
        all_preds.append(out.logits.argmax(-1).cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(dataloader)

    # --- Flat metrics ---
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    print(f"  Val loss {avg_loss:.4f} | Acc {acc:.4f} | P {p:.4f} R {r:.4f} F1 {f1:.4f}")

    # --- Order-level metrics (collapse leaf → order) ---
    def to_order(class_id):
        name = id_to_label[int(class_id)]
        return name.split("/")[0]
    order_true = [to_order(c) for c in labels]
    order_pred = [to_order(c) for c in preds]
    p_o, r_o, f1_o, _ = precision_recall_fscore_support(
        order_true, order_pred, average="macro", zero_division=0
    )
    acc_o = accuracy_score(order_true, order_pred)
    print(f"  Order        | Acc {acc_o:.4f} | P {p_o:.4f} R {r_o:.4f} F1 {f1_o:.4f}")

    return {
        "val_loss":  avg_loss,
        "sf_acc":  acc,  "sf_p":  p,  "sf_r":  r,  "sf_f1":  f1,
        "order_acc": acc_o,"order_p": p_o,"order_r": r_o,"order_f1": f1_o,
    }


# ============================================================
# Main
# ============================================================
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not args.debugging:
        wandb.init(
            name=args.run_name or f"longformer_fold{args.fold}",
            entity=args.wandb_team    or cfg["wandb"]["team"],
            project=args.wandb_project or cfg["wandb"]["project"],
            dir=args.wandb_dir         or cfg["wandb"]["dir"],
            config=cfg,
        )

    # --- Label space ---
    label_to_id, id_to_label, num_labels = build_flat_label_map(ORDER_TO_SUPERFAMILIES)
    print(f"Flat label space: {num_labels} classes")

    # --- Data ---
    with open(args.train_dir, "rb") as f:
        train_seqs = pickle.load(f)
    with open(args.val_dir, "rb") as f:
        val_seqs = pickle.load(f)

    if args.debugging:
        from itertools import islice
        train_seqs = dict(islice(train_seqs.items(), 100))
        val_seqs   = dict(islice(val_seqs.items(),   50))

    max_seq_len = cfg["model"]["max_seq_len"]
    train_ds = NucleotideClassificationDataset(train_seqs, label_to_id, max_seq_len)
    val_ds   = NucleotideClassificationDataset(val_seqs,   label_to_id, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batchsize"],
                              shuffle=True, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batchsize"],
                              shuffle=False, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True)

    # --- Model ---
    model = build_longformer(VOCAB_SIZE, num_labels, cfg)
    # if args.pretrained_mlm:
    model = load_pretrained_longformer_mlm(args.pretrained_mlm, model)
    for p in model.longformer.parameters():
        p.requires_grad = False


    model.to(DEVICE)

    FREEZE_EPOCHS = 3
    print(f"Backbone frozen for first {FREEZE_EPOCHS} epochs")





    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Longformer params: {n_trainable:,}")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
    )

    # --- Save dir ---
    save_dir = args.save_dir or cfg["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"longformer_fold{args.fold}_{args.run_name}.pt")

    # --- Train ---
    best_f1 = -1.0
    for epoch in range(cfg["train"]["epochs"]):
        if epoch == FREEZE_EPOCHS:
            print("--- Unfreezing Longformer backbone ---")
            for p in model.longformer.parameters():
                p.requires_grad = True
            # Add unfrozen params as a new group (typically a SMALLER LR for the backbone)
            optimizer.add_param_group({
                "params": list(model.longformer.parameters()),
                "lr": cfg["train"]["lr"] * 0.1,   # 10× smaller for fine-tuning
            })
        train_loss = run_train(model, train_loader, optimizer)
        

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} ")

        if not args.debugging:
            wandb.log({"train_loss": train_loss})

        

        gc.collect()
        torch.cuda.empty_cache()


    # if val_out["leaf_f1"] > best_f1:
            # best_f1 = val_out["leaf_f1"]
    if not args.debugging:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_leaf_f1": best_f1,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        }, save_path)
        print(f"  ↳ new best (leaf F1 {best_f1:.4f}) saved to {save_path}")
    
    val_out    = run_val(model, val_loader, id_to_label)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Longformer params: {n_trainable:,}")

    with open(os.path.join(save_dir, f"fold{args.fold}_{args.run_name}_val.yaml"), "w") as f:
        yaml.safe_dump(val_out, f)

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--fold",          required=True, type=int)
    p.add_argument("--train_dir",     required=True, help="Train pickle: {seq: (order, sf_or_None)}")
    p.add_argument("--val_dir",       required=True, help="Val   pickle: {seq: (order, sf_or_None)}")
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--run_name",      default="run")
    p.add_argument("--seed",          default=22, type=int)
    p.add_argument("--debugging",     action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_team",    default=None)
    p.add_argument("--wandb_dir",     default=None)
    p.add_argument("--pretrained_mlm", type=str, default=None,
               help="Path to LongformerForMaskedLM .pt checkpoint. "
                    "Backbone weights will be transferred; classification head is randomly initialized.")
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    main(args)