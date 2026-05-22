
import os
import gc
import yaml
import wandb
import argparse
import random
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import math

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric

# Import your new unified model
from model.model_transformer_hierarchical import (
    HierarchicalTransformerClassifier,
    build_classification_tree,
    # set_alphas_with_phi,
    hierarchical_predict,
    node_lineage_string,
    build_label_to_node_id,
)

from utils import _get_depth_labels,_compute_sklearn_metrics
from data.dataloader import HierarchicalSequenceDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")







def load_pretrained_mlm_weights(pretrained_mlm_path, new_model):
    """
    Transfer encoder weights from a pretrained MLM checkpoint
    (BaselineTransformer_Vanilla_MLM) to HierarchicalTransformerClassifier.

    Both models use identical naming:
        src_embed.*          → transferred
        pos_encoder.*        → transferred
        encoder_layers.N.*   → transferred (if layer index < new model's num_layers)
        sequence_head.*      → skipped (MLM head, not needed)
        output_head.*        → not in checkpoint (new, randomly initialized)
    """
    print(f"Loading pretrained MLM weights from {pretrained_mlm_path}...")

    checkpoint = torch.load(pretrained_mlm_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
    else:
        old_state_dict = checkpoint

    new_state_dict = {}

    for key, weight in old_state_dict.items():

        if key.startswith("src_embed."):
            new_state_dict[key] = weight

        elif key.startswith("pos_encoder."):
            new_state_dict[key] = weight

        elif key.startswith("encoder_layers."):
            layer_idx = int(key.split(".")[1])
            if layer_idx < len(new_model.encoder_layers):
                new_state_dict[key] = weight

        elif key.startswith("sequence_head."):
            continue  # MLM head — not needed

        else:
            print(f"  Skipping unrecognized key: {key}")

    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)

    print("\n--- Weight Transfer Complete ---")

    expected_missing = [k for k in missing_keys if "output_head" in k]
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    for k in missing_keys:
        tag = "  (expected)" if k in expected_missing else "  ⚠️  UNEXPECTED"
        print(f"  Missing key: {k}{tag}")

    if unexpected_missing:
        print(f"\n⚠️  WARNING - {len(unexpected_missing)} core keys are missing:")
        for k in unexpected_missing[:10]:
            print(f"    {k}")
    else:
        print(f"\n✅ Success! All encoder weights transferred. "
              f"{len(expected_missing)} output head keys initialized from scratch.")

    return new_model















# =============================================================================
# DIAGNOSTICS
# =============================================================================

def log_param_diagnostics(model, f):
    """Write per-parameter stats after loss.backward(), before optimizer.step()."""
    f.write(
        f"{'Parameter':<60} {'Shape':<20} {'Param Norm':>12} {'Grad Norm':>12} {'Grad Max':>12}\n"
    )
    f.write("-" * 120 + "\n")

    for name, p in model.named_parameters():
        if p.grad is not None:
            f.write(
                f"{name:<60} {str(list(p.shape)):<20} "
                f"{p.data.norm().item():>12.6f} "
                f"{p.grad.data.norm().item():>12.6f} "
                f"{p.grad.data.abs().max().item():>12.6e}\n"
            )


# =============================================================================
# TRAINING LOOP — unified, no more order/sf split
# =============================================================================

def run_train(model, dataloader, optimizer, path=None, epoch=None):
    """
    Single training loop for the hierarchical model.
    The model outputs raw logits; HierarchicalSoftmaxLoss handles the tree.
    """
    total_loss = 0.0
    model.train(True)

    for tokens, src_key_padding_mask, target_node_ids in dataloader:
        tokens = tokens.to(DEVICE, non_blocking=True)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)
        target_node_ids = target_node_ids.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Forward: model returns raw logits, loss is computed internally
        # using HierarchicalSoftmaxLoss (walks tree from target leaf → root)
        outputs = model(tokens, src_key_padding_mask, target_node_ids)
        loss = outputs["total_loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\nepoch {epoch}| Total train_loss | {avg_loss:.6f}\n")
            log_param_diagnostics(model, f)

    return avg_loss

































# =============================================================================
# VALIDATION LOOP — unified, with hierarchical metrics
# =============================================================================
def run_val(model, dataloader, classification_tree):
    """
    Validation loop that computes:
      1. Hierarchical loss (same as training)
      2. Per-depth greedy accuracy (order = depth 1, superfamily = depth 2)
      3. sklearn metrics (accuracy, macro P/R/F1) at both order and superfamily level
      4. All of the above at two thresholds: 0.7 and 0.9
    """
    all_logits = []
    all_target_ids = []

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens, src_key_padding_mask, target_node_ids in dataloader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)
            target_node_ids = target_node_ids.to(DEVICE, non_blocking=True)

            outputs = model(tokens, src_key_padding_mask, target_node_ids)
            loss = outputs["total_loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_target_ids.append(target_node_ids.cpu())

    avg_loss = total_loss / len(dataloader)
    all_logits = torch.cat(all_logits, dim=0)
    all_target_ids = torch.cat(all_target_ids, dim=0)

    # Convert raw logits → per-node probabilities (once, reused by both thresholds)
    probs = node_probabilities(all_logits, root=classification_tree)

    # ---- Compute metrics at each threshold ----
    results = {"val_loss": avg_loss}

    for threshold in [0.7, 0.9]:
        t_str = str(threshold).replace(".", "")  # "07" or "09" for dict keys

        print(f"\n{'='*60}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'='*60}")

        # Greedy predictions with this threshold
        # If confidence < threshold at any level, prediction stops at the parent node
        pred_nodes = greedy_predictions(
            probs, root=classification_tree, threshold=threshold
        )

        total_samples = len(all_target_ids)

        # ---- Order-level metrics (depth 1) ----
        true_order_raw, pred_order_raw = _get_depth_labels(
            pred_nodes, all_target_ids, classification_tree, depth=1
        )

        true_order = []
        pred_order = []
        for t, p in zip(true_order_raw, pred_order_raw):
            if t is not None:
                true_order.append(t)
                pred_order.append(p)

        n_order_skipped = total_samples - len(true_order)
        if n_order_skipped > 0:
            print(f"\n  {n_order_skipped} samples not classified at order level (stopped at root)")

        order_metrics = _compute_sklearn_metrics(
            true_order, pred_order, total_samples,
            level_name=f"Order (threshold={threshold})"
        )

        # ---- Superfamily-level metrics (depth 2) ----
        true_sf_raw, pred_sf_raw = _get_depth_labels(
            pred_nodes, all_target_ids, classification_tree, depth=2
        )

        true_sf = []
        pred_sf = []
        for t, p in zip(true_sf_raw, pred_sf_raw):
            if t is not None:
                true_sf.append(t)
                pred_sf.append(p)

        n_sf_skipped = total_samples - len(true_sf)
        print(f"\n  {n_sf_skipped} samples excluded from SF metrics:")
        print(f"    - No superfamily in ground truth (DIRS, Helitron, PLE, etc.)")
        print(f"    - Or prediction stopped at order level due to threshold")

        if len(true_sf) > 0:
            sf_metrics = _compute_sklearn_metrics(
                true_sf, pred_sf, total_samples,
                level_name=f"Superfamily (threshold={threshold})"
            )
        else:
            print(f"\n  No samples classified at superfamily level with threshold={threshold}")
            sf_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                          "n_samples": 0, "n_total_samples": total_samples,
                          "n_active_classes": 0, "n_total_classes": 0}

        results[f"sf_acc_t{t_str}"] = sf_metrics["accuracy"]
        results[f"sf_prec_t{t_str}"] = sf_metrics["precision"]
        results[f"sf_rec_t{t_str}"] = sf_metrics["recall"]
        results[f"sf_f1_t{t_str}"] = sf_metrics["f1"]
        results[f"sf_classified_t{t_str}"] = sf_metrics.get("n_samples", 0)
        results[f"sf_active_classes_t{t_str}"] = sf_metrics.get("n_active_classes", 0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  {'Metric':<25} {'t=0.7':>10} {'t=0.9':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Order Accuracy':<25} {results['order_acc_t07']:>10.4f} {results['order_acc_t09']:>10.4f}")
    print(f"  {'Order F1 (macro)':<25} {results['order_f1_t07']:>10.4f} {results['order_f1_t09']:>10.4f}")
    print(f"  {'SF Accuracy':<25} {results['sf_acc_t07']:>10.4f} {results['sf_acc_t09']:>10.4f}")
    print(f"  {'SF F1 (macro)':<25} {results['sf_f1_t07']:>10.4f} {results['sf_f1_t09']:>10.4f}")

    return results


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def load_checkpoint(model, optimizer, checkpoint_path, device="cpu", load_optimizer=True, strict=True):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if load_optimizer and optimizer is not None and "optim_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    best_val_sf_acc = checkpoint.get("best_val_sf_acc", -1.0)

    print(f"Loaded checkpoint: {checkpoint_path} | Resuming from epoch {epoch+1} | Best SF Acc: {best_val_sf_acc:.4f}")

    return model, optimizer, epoch, best_val_sf_acc


def load_model_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("transformer_encoder.layers."):
            k = k.replace("transformer_encoder.layers.", "encoder_layers.", 1)
        remapped[k] = v

    model.load_state_dict(remapped, strict=True)
    model.to(device)
    model.eval()
    return model


def load_pretrained_mlm_weights(pretrained_mlm_path, new_model):
    """
    Transfer encoder weights from a pretrained MLM checkpoint
    to the new hierarchical model. The output head is left randomly initialized.
    """
    print(f"Loading pretrained MLM weights from {pretrained_mlm_path}...")

    checkpoint = torch.load(pretrained_mlm_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
    else:
        old_state_dict = checkpoint

    new_state_dict = {}

    for key, weight in old_state_dict.items():
        key = key.replace("module.", "")

        # A. Transfer the Embedding
        if key.startswith("src_embed."):
            new_state_dict[key] = weight

        # B. Transfer Positional Encoding
        elif key.startswith("pos_encoder."):
            new_state_dict[key] = weight

        # C. Map the Transformer Layers
        elif key.startswith("transformer_encoder.layers."):
            parts = key.split(".")
            layer_idx = int(parts[2])
            if layer_idx < len(new_model.encoder_layers):
                new_key = key.replace("transformer_encoder.layers.", "encoder_layers.")
                new_state_dict[new_key] = weight

        # D. Ignore the MLM Head
        elif key.startswith("sequence_head."):
            continue

    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)

    print("\n--- Weight Transfer Complete ---")

    expected_missing = [k for k in missing_keys if "output_head" in k or "penultimate" in k or "final" in k]
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    for k in missing_keys:
        print(f"  Missing key: {k}")

    if unexpected_missing:
        print(f"⚠️  WARNING - These core keys are missing and shouldn't be:\n{unexpected_missing[:10]}")
    else:
        print(f"✅ Success! {len(expected_missing)} new head keys correctly initialized from scratch.")

    return new_model

















#########################
#VALIDATION METRICS
#########################






def create_metrics_save_path(model, save_path, device,  val_loader, classification_tree, val_metrics, metrics_save_path,wandb,log=True):


    model = load_model_weights(model, save_path, device=DEVICE)

    val_metrics = run_val(model, val_loader, classification_tree)

    


    total_params = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_p:,}")
    print(f"Non-trainable parameters: {total_params - trainable_p:,}")

    print("\nModel Run Finished. Validation results for this fold:")
    create_valmetrics(val_metrics, metrics_save_path,wandb,log=True)
    
    return total_params






def create_valmetrics(val_metrics, metrics_save_path,wandb,log=True):


    print(
        f"Val Loss: {val_metrics['val_loss']:.4f} | "

        f"Precision thresh 0.7 {val_metrics['sf_prec_t07']:.4f} | "
        f"Recall thresh 0.7 {val_metrics['sf_rec_t07']:.4f}",

        f"Precision thresh 0.9 {val_metrics['order_prec_t09']:.4f} | "
        f"Recall thresh 0.9 {val_metrics['order_rec_t09']:.4f}",
    )


    if log:
        wandb.log({
            "val_loss": val_metrics["val_loss"],
            # Threshold 0.7
            "order_acc_t07": val_metrics["order_acc_t07"],
            "order_f1_t07": val_metrics["order_f1_t07"],
            "order_prec_t07": val_metrics["order_prec_t07"],
            "order_rec_t07": val_metrics["order_rec_t07"],
            "sf_acc_t07": val_metrics["sf_acc_t07"],
            "sf_f1_t07": val_metrics["sf_f1_t07"],
            "sf_prec_t07": val_metrics["sf_prec_t07"],
            "sf_rec_t07": val_metrics["sf_rec_t07"],
            # Threshold 0.9
            "order_acc_t09": val_metrics["order_acc_t09"],
            "order_f1_t09": val_metrics["order_f1_t09"],
            "order_prec_t09": val_metrics["order_prec_t09"],
            "order_rec_t09": val_metrics["order_rec_t09"],
            "sf_acc_t09": val_metrics["sf_acc_t09"],
            "sf_f1_t09": val_metrics["sf_f1_t09"],
            "sf_prec_t09": val_metrics["sf_prec_t09"],
            "sf_rec_t09": val_metrics["sf_rec_t09"],
        })

    with open(metrics_save_path, "a") as f:
        f.write(
            f"Fold {args.fold} | "
            f"Loss {val_metrics['val_loss']:.4f} | "
            f"Order Prec(0.7) {val_metrics['order_prec_t07']:.4f} | "
            f"Order Rec(0.7) {val_metrics['order_rec_t07']:.4f} | "
            f"SF Prec(0.7) {val_metrics['sf_prec_t07']:.4f} | "
            f"SF Rec(0.7) {val_metrics['sf_rec_t07']:.4f} | "
            f"Order Prec(0.9) {val_metrics['order_prec_t09']:.4f} | "
            f"Order Rec(0.9) {val_metrics['order_rec_t09']:.4f} | "
            f"SF Prec(0.9) {val_metrics['sf_prec_t09']:.4f} | "
            f"SF Rec(0.9) {val_metrics['sf_rec_t09']:.4f}\n"

            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"

            f"Threshold 0.7 Precision    Recall     F1     Accuracy\n"
            f" {val_metrics['order_prec_t07']:.4f}    {val_metrics['order_rec_t07']:.4f}    {val_metrics['order_f1_t07']:.4f}    {val_metrics['order_acc_t07']:.4f}\n"
            f" {val_metrics['sf_prec_t07']:.4f}    {val_metrics['sf_rec_t07']:.4f}    {val_metrics['sf_f1_t07']:.4f}    {val_metrics['sf_acc_t07']:.4f}\n"
            f"==================================================================================================================================\n"
            f"Threshold 0.9 Precision    Recall     F1     Accuracy\n"
            f" {val_metrics['order_prec_t09']:.4f}    {val_metrics['order_rec_t09']:.4f}    {val_metrics['order_f1_t09']:.4f}    {val_metrics['order_acc_t09']:.4f}\n"
            f" {val_metrics['sf_prec_t09']:.4f}    {val_metrics['sf_rec_t09']:.4f}    {val_metrics['sf_f1_t09']:.4f}    {val_metrics['sf_acc_t09']:.4f}\n"
        )
    return val_metrics
