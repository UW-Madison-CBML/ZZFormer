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

    for threshold in [0.0]:
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


        results[f"order_acc_t{t_str}"] = order_metrics["accuracy"]
        results[f"order_prec_t{t_str}"] = order_metrics["precision"]
        results[f"order_rec_t{t_str}"] = order_metrics["recall"]
        results[f"order_f1_t{t_str}"] = order_metrics["f1"]
        results[f"order_classified_t{t_str}"] = order_metrics.get("n_samples", 0)
        results[f"order_active_classes_t{t_str}"] = order_metrics.get("n_active_classes", 0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  {'Metric':<25} {'t=0.0':>10} {'t=0.0':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Order Accuracy':<25} {results['order_acc_t00']:>10.4f} {results['order_acc_t00']:>10.4f}")
    print(f"  {'Order F1 (macro)':<25} {results['order_f1_t00']:>10.4f} {results['order_f1_t00']:>10.4f}")
    print(f"  {'SF Accuracy':<25} {results['sf_acc_t00']:>10.4f} {results['sf_acc_t00']:>10.4f}")
    print(f"  {'SF F1 (macro)':<25} {results['sf_f1_t00']:>10.4f} {results['sf_f1_t00']:>10.4f}")

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
# MAIN
# =============================================================================

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5}
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN = vocab["PAD"]
    ignore_index = -100

 


    # ================================================================
    # 1. BUILD THE CLASSIFICATION TREE
    # ================================================================
    # Load the order→superfamily mapping from config.
    # Your config should have a section like:
    #   hierarchy:
    #     LINE: [CR1, L1, L2, Jockey, RTE]
    #     ...
    order_to_superfamilies = config["hierarchy"]

    classification_tree = build_classification_tree(
        order_to_superfamilies,
        label_smoothing=config.get("label_smoothing", 0.0),
        gamma=config.get("gamma", 0.0),
    )

    # Apply phi weighting (Corgi-style per-depth loss weighting)
    # phi = config.get("phi", 1.0)
    # set_alphas_with_phi(classification_tree, phi=phi)

    classification_tree.render(print=True)
    print(f"Tree layer_size (model output dim): {classification_tree.layer_size}")
    print(f"Total nodes: {len(classification_tree.node_list)}")

    # Build the label mapping: "ORDER/SF" → node_id
    label_map = build_label_to_node_id(classification_tree)

    # ================================================================
    # 2. INITIALIZE MODEL — one unified model, not two
    # ================================================================
    model = HierarchicalTransformerClassifier(
        src_vocab_size=VOCAB_SIZE,
        classification_tree=classification_tree,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["nhead"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        num_layers=config["model"]["num_layers"],
        max_position_embeddings=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN,
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
    )

    # 3. Load Pretrained MLM Weights BEFORE Freezing
    if args.pretrained_mlm:
        model = load_pretrained_mlm_weights(args.pretrained_mlm, model)

    # Freeze the backbone initially
    for param in model.src_embed.parameters():
        param.requires_grad = False
    for param in model.encoder_layers.parameters():
        param.requires_grad = False

    model.to(DEVICE)

    # ================================================================
    # 4. WANDB
    # ================================================================
    if not args.debugging:
        wandb.init(
            name=args.run_name or f"hierarchical_fold{args.fold}_{args.seed}",
            settings=wandb.Settings(_service_wait=300),
            entity=args.wandb_team or config["wandb"]["team"],
            project=args.wandb_project or config["wandb"]["project"],
            dir=args.wandb_dir or config["wandb"]["dir"],
            config=config,
        )

    # ================================================================
    # 5. DATASETS — unified, no more order/sf filtering
    # ================================================================
    print(f"Loading Train Fold from: {args.train_dir}")
    with open(args.train_dir, "rb") as f:
        train_seqs = pickle.load(f)

    print(f"Loading Val Fold from: {args.val_dir}")
    with open(args.val_dir, "rb") as f:
        val_seqs = pickle.load(f)

    if args.debugging:
        from itertools import islice
        train_seqs = dict(islice(train_seqs.items(), 100))
        val_seqs = dict(islice(val_seqs.items(), 50))

    # The dataset handles both order-only and order/superfamily labels
    # via the label_map. If a sample has sf_label="", it maps to the order node.
    train_dataset = HierarchicalSequenceDataset(
        train_seqs,
        label_to_id=label_map,
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN,
        ignore_index=ignore_index,
    )

    val_dataset = HierarchicalSequenceDataset(
        val_seqs,
        label_to_id=label_map,
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN,
        ignore_index=ignore_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    # ================================================================
    # 6. SAVING & RESUMPTION SETUP
    # ================================================================
    save_dir = args.save_dir or config["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"hierarchical_fold{args.fold}_best_{args.run_name}.pt")
    metrics_save_path = os.path.join(save_dir, f"hierarchical_allfold_metrics_fold{args.fold}_nothresh.txt")
    log_path = os.path.join(save_dir, f"{args.run_name}_training_diagnostics.txt")

    # Peek at checkpoint for resumption
    init_epoch = 0
    best_val_sf_acc = -1.0
    if os.path.exists(save_path):
        temp_checkpoint = torch.load(save_path, map_location="cpu")
        init_epoch = temp_checkpoint.get("epoch", -1) + 1
        best_val_sf_acc = temp_checkpoint.get("best_val_sf_acc", -1.0)
        print(f"Found existing checkpoint for Fold {args.fold}. Resuming from epoch {init_epoch}...")

    # If resuming past the freeze phase, unfreeze now
    if init_epoch > 4:
        for param in model.parameters():
            param.requires_grad = True

    # Create optimizer with only trainable params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config["train"]["lr"])

    # ================================================================
    # 7. IF CHECKPOINT EXISTS — just validate and report
    # ================================================================
    if os.path.exists(save_path):
            model = load_model_weights(model, save_path, device=DEVICE)
            # if model["epoch"] >= config["train"]["epochs"]:

            val_metrics = run_val(model, val_loader, classification_tree)

            print("\nModel Run Finished. Validation results for this fold:")
            print(f"Val Loss:  {val_metrics['val_loss']:.4f}")

            total_params = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters:         {total_params:,}")
            print(f"Trainable parameters:     {trainable_p:,}")
            print(f"Non-trainable parameters: {total_params - trainable_p:,}")


            thresholds=[0.0]
            t_str_def=str(thresholds[0]).replace(".", "")
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Val Loss: {val_metrics['val_loss']:.4f} | "

                f"Precision thresh {thresholds[0]} {val_metrics[f'sf_prec_t{t_str_def}']:.4f} | "
                f"Recall thresh {thresholds[0]} {val_metrics[f'sf_rec_t{t_str_def}']:.4f}",

                f"Precision thresh {thresholds[0]} {val_metrics[f'order_prec_t{t_str_def}']:.4f} | "
                f"Recall thresh {thresholds[0]} {val_metrics[f'order_rec_t{t_str_def}']:.4f}",
            )

            with open(metrics_save_path, "a") as f:
                f.write(
                    f"Fold {args.fold} | "
                    f"Loss {val_metrics['val_loss']:.4f} | "
                    f"Order Prec({thresholds[0]}) {val_metrics[f'order_prec_t{t_str_def}']:.4f} | "
                    f"Order Rec({thresholds[0]}) {val_metrics[f'order_rec_t{t_str_def}']:.4f} | "
                    f"SF Prec({thresholds[0]}) {val_metrics[f'sf_prec_t{t_str_def}']:.4f} | "
                    f"SF Rec({thresholds[0]}) {val_metrics[f'sf_rec_t{t_str_def}']:.4f} | "
                    f"Order Prec({thresholds[0]}) {val_metrics[f'order_prec_t{t_str_def}']:.4f} | "
                    f"Order Rec({thresholds[0]}) {val_metrics[f'order_rec_t{t_str_def}']:.4f} | "
                    f"SF Prec({thresholds[0]}) {val_metrics[f'sf_prec_t{t_str_def}']:.4f} | "
                    f"SF Rec({thresholds[0]}) {val_metrics[f'sf_rec_t{t_str_def}']:.4f}\n"

                    f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"

                    f"Threshold {t_str_def} Precision    Recall     F1     Accuracy\n"
                    f" {val_metrics[f'order_prec_t{t_str_def}']:.4f}    {val_metrics[f'order_rec_t{t_str_def}']:.4f}    {val_metrics[f'order_f1_t{t_str_def}']:.4f}    {val_metrics[f'order_acc_t{t_str_def}']:.4f}\n"
                    f" {val_metrics[f'sf_prec_t{t_str_def}']:.4f}    {val_metrics[f'sf_rec_t{t_str_def}']:.4f}    {val_metrics[f'sf_f1_t{t_str_def}']:.4f}    {val_metrics[f'sf_acc_t{t_str_def}']:.4f}\n"
                    f"==================================================================================================================================\n"
                    f"Threshold {t_str_def} Precision    Recall     F1     Accuracy\n"
                    f" {val_metrics[f'order_prec_t{t_str_def}']:.4f}    {val_metrics[f'order_rec_t{t_str_def}']:.4f}    {val_metrics[f'order_f1_t{t_str_def}']:.4f}    {val_metrics[f'order_acc_t{t_str_def}']:.4f}\n"
                    f" {val_metrics[f'sf_prec_t{t_str_def}']:.4f}    {val_metrics[f'sf_rec_t{t_str_def}']:.4f}    {val_metrics[f'sf_f1_t{t_str_def}']:.4f}    {val_metrics[f'sf_acc_t{t_str_def}']:.4f}\n"
                )
            return

    # ================================================================
    # 8. TRAINING LOOP
    # ================================================================
    print(f"\n{'='*40}\nSTARTING TRAINING FOR FOLD {args.fold}\n{'='*40}")

    for epoch in range(init_epoch, config["train"]["epochs"]):
        # Unfreeze backbone after initial head warmup
        if epoch == 4:
            print("\n--- Unfreezing Transformer Backbone for Fine-Tuning ---")
            for param in model.parameters():
                param.requires_grad = True

            unfrozen_params = list(model.src_embed.parameters()) + list(model.encoder_layers.parameters())
            optimizer.add_param_group({"params": unfrozen_params, "lr": config["train"]["lr"]})

        # --- TRAIN ---
        train_loss = run_train(model, train_loader, optimizer, path=log_path, epoch=epoch)

        # --- LOG ---
        if not args.debugging:
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # 9. SAVE CHECKPOINT
    # ================================================================
    if not args.debugging:
        save_data = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_sf_acc": best_val_sf_acc,
            "src_vocab_size": VOCAB_SIZE,
            "d_model": config["model"]["d_model"],
            "pad_token_id": PAD_TOKEN,
            "ignore_index": ignore_index,
            "classification_tree": classification_tree,  # Save tree for inference
        }
        torch.save(save_data, save_path)

        print(
        f"Training Completed After {epoch+1} Epochs | "
        f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | ")

    # ================================================================
    # 10. FINAL VALIDATION
    # ================================================================
    val_metrics = run_val(model, val_loader, classification_tree)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Val Loss: {val_metrics['val_loss']:.4f} | "

        f"Precision thresh 0.7 {val_metrics['sf_prec_t07']:.4f} | "
        f"Recall thresh 0.7 {val_metrics['sf_rec_t07']:.4f}\n",

        f"Precision thresh 0.9 {val_metrics['order_prec_t09']:.4f} | "
        f"Recall thresh 0.9 {val_metrics['order_rec_t09']:.4f}\n",
    )


    if not args.debugging:
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
            f"Order | {val_metrics['order_prec_t07']:.4f}  |  {val_metrics['order_rec_t07']:.4f}  |  {val_metrics['order_f1_t07']:.4f}  |  {val_metrics['order_acc_t07']:.4f}\n"
            f"SF    | {val_metrics['sf_prec_t07']:.4f}  |  {val_metrics['sf_rec_t07']:.4f}  |  {val_metrics['sf_f1_t07']:.4f}  |  {val_metrics['sf_acc_t07']:.4f}\n"
            f"==================================================================================================================================\n"
            f"Threshold 0.9 Precision    Recall     F1     Accuracy\n"
            f"Order | {val_metrics['order_prec_t09']:.4f}  |  {val_metrics['order_rec_t09']:.4f}  |  {val_metrics['order_f1_t09']:.4f}  |  {val_metrics['order_acc_t09']:.4f}\n"
            f"SF    | {val_metrics['sf_prec_t09']:.4f}  |  {val_metrics['sf_rec_t09']:.4f}  |  {val_metrics['sf_f1_t09']:.4f}  |  {val_metrics['sf_acc_t09']:.4f}\n"
        )



















    total_params = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_p:,}")
    print(f"Non-trainable parameters: {total_params - trainable_p:,}")




# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # No more --mode flag! One model handles both order and superfamily.

    # Pretrained & Fold Config
    parser.add_argument("--pretrained_mlm", type=str, required=True, help="Path to trained MLM .pt file")
    parser.add_argument("--fold", type=int, required=True, help="The current fold number")

    # Data Config
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data pickle file for this fold")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data pickle file for this fold")

    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")

    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_team", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)