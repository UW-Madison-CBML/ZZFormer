import os
import gc
import yaml
import wandb
import argparse
import random
import numpy as np
import torch
import pickle
from functools import partial
from torch.utils.data import DataLoader
import math
# Assuming these are correctly defined in your local modules
from model.model import (
    BaselineTransformer_Vanilla_MLM,
    BaselineTransformer_BERTSTyle_MLM
)
from data.dataloader import SequenceDataset_nocollate, SequenceDataset, collate_fn, apply_mlm_mask_gpu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def log_param_diagnostics(model, f):
    # grad_norm = get_total_norm(model.parameters(), norm_type=2.0)
    """Write per-parameter stats after loss.backward(), before optimizer.step()."""
    f.write(f"{'Parameter':<60} {'Shape':<20} {'Param Norm':>12} {'Grad Norm':>12} {'Grad Max':>12}\n")
    f.write("-" * 120 + "\n")

    for name, p in model.named_parameters():
        if p.grad is not None:
            f.write(
                f"{name:<60} {str(list(p.shape)):<20} "
                f"{p.data.norm().item():>12.6f} "
                f"{p.grad.data.norm().item():>12.6f} "
                f"{p.grad.data.abs().max().item():>12.6e}\n"
            )
    # f.write(f"\nTotal grad norm (L2): {sum(p.numel() for p in model.parameters())}\n")


# =====================================================================
# TRAINING LOOP
# =====================================================================
def run_train(
    model,
    dataloader,
    optimizer=None,
    ignore_index=None,  
    PAD_TOKEN=None,
    MASK_TOKEN_ID=None,
    VOCAB_SIZE=None,
    mask_ignore_token_ids=(5,),
    path=None,
    epoch=None,
):
    total_loss = 0.0

    for tokens, src_key_padding_mask in dataloader:

        # 1. Move to GPU
        tokens = tokens.to(DEVICE, non_blocking=True)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # 2. Apply Masking instantly on the GPU
        masked_tokens, mlm_labels = apply_mlm_mask_gpu(
            tokens=tokens,
            mask_prob=0.15,
            pad_token_id=PAD_TOKEN,
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            mask_ignore_token_ids=mask_ignore_token_ids,
            ignore_index=ignore_index,
        )


        loss, logits = model(masked_tokens,src_key_padding_mask,mlm_labels)

        # ---------------- Backward ----------------
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)


    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\nepoch {epoch}| Total train_loss | {avg_loss:.6f}\n")
        log_param_diagnostics(model, f)


    return avg_loss


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device="cpu",
    load_optimizer=True
):
    """Load model (and optionally optimizer) from checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state only if requested
    if load_optimizer and optimizer is not None and "optim_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    # Extract metadata
    epoch = checkpoint.get("epoch", -1)
    best_train_loss = checkpoint.get("best_train_loss", float("inf"))
    
    print(f"Loaded checkpoint: {checkpoint_path} | Resuming from epoch {epoch+1} | Best Loss: {best_train_loss:.4f}")

    return model, optimizer, epoch, best_train_loss



def main(args):
    # Load config ONCE
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # cfg = config

    vocab = {
        "PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5
    }
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN = 0
    MASK_TOKEN_ID = VOCAB_SIZE  
    ignore_index = -100
    mask_ignore_token_ids=(vocab["X"], ) # e.g., 'X' token ID to ignore during masking

    # ---------------- Model Init ----------------
    if args.mode == "mlm":
        model = BaselineTransformer_Vanilla_MLM(
            src_vocab_size=VOCAB_SIZE,
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["nhead"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            num_layers=config["model"]["num_layers"],
            positional_encoding=config["model"]["positional_encoding"],
            max_position_embeddings=config["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            ignore_index=ignore_index
        )

    elif args.mode == "bert_mlm":
        model = BaselineTransformer_BERTSTyle_MLM(
            src_vocab_size=VOCAB_SIZE,
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["nhead"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            num_layers=config["model"]["num_layers"],
            positional_encoding=config["model"]["positional_encoding"],
            max_position_embeddings=config["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            ignore_index=ignore_index
        )

    model.to(DEVICE)

    # ---------------- WandB ----------------
    if not args.debugging:
        wandb.init(
            name=args.run_name if args.run_name else f"{args.mode}_{args.seed}",
            settings=wandb.Settings(_service_wait=300),
            entity=args.wandb_team if args.wandb_team else config["wandb"]["team"],
            project=args.wandb_project if args.wandb_project else config["wandb"]["project"],
            dir=args.wandb_dir if args.wandb_dir else config["wandb"]["dir"],
            config=config
        )

    # ---------------- Dataset ----------------
    ds_path = args.train_dir if args.train_dir else config["data"]["train_dir"]
    with open(ds_path, "rb") as f:
        all_seqs = pickle.load(f)


    train_dataset = SequenceDataset_nocollate(
        all_seqs,
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,              # ✅ works now
        persistent_workers=True,      # ✅ no pickling issues
    )


    # ---------------- Optimizer & Scheduler ----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])


    save_dir = args.save_dir if args.save_dir else config["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.mode}_best.pt")

    if os.path.exists(save_path):
        model, optimizer, init_epoch, best_train_loss = load_checkpoint(
            model, optimizer, save_path, device=DEVICE, load_optimizer=True
        )
        init_epoch += 1  # Start at the next epoch
    else:
        init_epoch = 0
        best_train_loss = float("inf")


    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{args.run_name}_training_diagnostics.txt")
    # ---------------- Training loop ----------------
    for epoch in range(init_epoch, config["train"]["epochs"]):
        print(f"Epoch {epoch:03d} starting...")

        model.train()
        train_loss = run_train(
            model,
            train_loader,
            optimizer=optimizer,
            ignore_index=ignore_index,  
            PAD_TOKEN=PAD_TOKEN,
            MASK_TOKEN_ID=MASK_TOKEN_ID,
            VOCAB_SIZE=VOCAB_SIZE,
            mask_ignore_token_ids=mask_ignore_token_ids,
            path=log_path,
            epoch=epoch,
        )

        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")


        if not args.debugging:
            wandb.log({"epoch": epoch, "train_loss": train_loss})

        gc.collect()
        torch.cuda.empty_cache()

    if not args.debugging:
        save_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(), 
            "optim_state_dict": optimizer.state_dict(),
            "best_train_loss": train_loss,  # Save this so we don't have to recalculate on resume
            "src_vocab_size": VOCAB_SIZE,
            "num_layers": config["model"]["num_layers"],
            "nhead": config["model"]["nhead"],
            "max_position_embeddings": config["model"]["max_seq_len"],
            "d_model": config["model"]["d_model"],
            "dim_feedforward": config["model"]["dim_feedforward"],
            "positional_encoding": config["model"]["positional_encoding"],
            "pad_token_id": PAD_TOKEN,
            "ignore_index": ignore_index
        }
        torch.save(save_data, save_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["none", "mlm", "bert_mlm"], required=True)
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument('--train_dir', type=str, default=None, help='Path to training data pickle file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model checkpoints')
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name override')
    parser.add_argument('--wandb_team', type=str, default=None, help='WandB team/entity name override')
    parser.add_argument('--wandb_dir', type=str, default=None, help='WandB log directory override')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()

    if not args.debugging and not (args.wandb_project and args.wandb_team and args.wandb_dir):
        print("Warning: Wandb config (project, team, dir) is incomplete. Ensure they are in the YAML or passed via CLI.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)