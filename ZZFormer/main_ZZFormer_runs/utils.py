import yaml
import argparse
import numpy as np
import torch
from torcheval.metrics.functional import binary_auroc, binary_auprc
import torch.nn.functional as F
from typing import Dict, Tuple
import os
import csv
import pandas as pd
# from torchmetrics.classification import MulticlassAveragePrecision, MulticlassAUROC, MulticlassF1Score
from torch import tensor
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score

from torcheval.metrics import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassAUPRC,
    MulticlassConfusionMatrix,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def _save_confusion_outputs_older(
    true_labels,
    pred_labels,
    total_samples,
    out_dir,
    run_name,
    level_name,      # "order" or "sf"
    threshold,
):
    """
    Saves:
      1) Raw confusion matrix CSV
      2) Row-normalized confusion matrix CSV
      3) Long-format CSV (true,pred,count)
      4) Summary txt with coverage
    """
    os.makedirs(out_dir, exist_ok=True)
    t_str = str(threshold).replace(".", "")

    if len(true_labels) == 0:
        # Save an empty marker file for traceability
        empty_path = os.path.join(
            out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_EMPTY.txt"
        )
        with open(empty_path, "w") as f:
            f.write(f"No evaluable samples for {level_name} at threshold={threshold}\n")
            f.write(f"total_samples={total_samples}\n")
        return

    labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    # ---- Wide matrix CSV (counts) ----
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv = os.path.join(out_dir, f"confusion_{run_name}_{level_name}_t{t_str}.csv")
    cm_df.to_csv(cm_csv)

    # ---- Row-normalized matrix CSV ----
    row_sums = cm_df.sum(axis=1).replace(0, 1)
    cm_norm_df = cm_df.div(row_sums, axis=0)
    cm_norm_csv = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_row_norm.csv"
    )
    cm_norm_df.to_csv(cm_norm_csv)

    # ---- Long format CSV ----
    long_csv = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_long.csv"
    )
    with open(long_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "count"])
        for i, true_lab in enumerate(labels):
            for j, pred_lab in enumerate(labels):
                c = int(cm[i, j])
                if c > 0:
                    writer.writerow([true_lab, pred_lab, c])

    # ---- Summary ----
    summary_path = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_summary.txt"
    )
    with open(summary_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"level={level_name}\n")
        f.write(f"threshold={threshold}\n")
        f.write(f"evaluated_samples={len(true_labels)}\n")
        f.write(f"total_samples={total_samples}\n")
        f.write(f"coverage={len(true_labels)/total_samples if total_samples else 0:.6f}\n")
        f.write(f"n_labels={len(labels)}\n")
        f.write(f"labels={labels}\n")


def _save_confusion_outputs(
    true_labels,
    pred_labels,
    total_samples,
    out_dir,
    run_name,
    level_name,      # "order" or "sf"
    threshold,
):
    """
    Saves:
      1) Raw confusion matrix CSV (GT labels as rows/cols)
      2) Row-normalized confusion matrix CSV
      3) Long-format CSV (true,pred,count)
      4) Summary txt with coverage + phantom prediction stats

    NOTE:
      - Matrix label space is ground-truth classes only.
      - Predictions outside GT classes are still counted as wrong via summary
        but are not added as standalone matrix rows/cols.
    """
    import os, csv
    import pandas as pd
    from collections import Counter
    from sklearn.metrics import confusion_matrix

    os.makedirs(out_dir, exist_ok=True)
    t_str = str(threshold).replace(".", "")

    if len(true_labels) == 0:
        empty_path = os.path.join(
            out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_EMPTY.txt"
        )
        with open(empty_path, "w") as f:
            f.write(f"No evaluable samples for {level_name} at threshold={threshold}\n")
            f.write(f"total_samples={total_samples}\n")
        return

    # ---- GT-only label space (fix) ----
    gt_labels = sorted(set(true_labels))
    pred_counts = Counter(pred_labels)

    # Track predictions not in GT label space
    phantom_classes = sorted(set(pred_labels) - set(gt_labels))
    n_phantom_preds = sum(pred_counts[c] for c in phantom_classes)

    # Filter out phantom-prediction samples for matrix construction only
    # (they remain documented in summary as out-of-GT predictions)
    y_true_cm = []
    y_pred_cm = []
    for t, p in zip(true_labels, pred_labels):
        if p in gt_labels:
            y_true_cm.append(t)
            y_pred_cm.append(p)

    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=gt_labels)

    # ---- Wide matrix CSV (counts) ----
    cm_df = pd.DataFrame(cm, index=gt_labels, columns=gt_labels)
    cm_csv = os.path.join(out_dir, f"confusion_{run_name}_{level_name}_t{t_str}.csv")
    cm_df.to_csv(cm_csv)

    # ---- Row-normalized matrix CSV ----
    row_sums = cm_df.sum(axis=1).replace(0, 1)
    cm_norm_df = cm_df.div(row_sums, axis=0)
    cm_norm_csv = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_row_norm.csv"
    )
    cm_norm_df.to_csv(cm_norm_csv)

    # ---- Long format CSV ----
    long_csv = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_long.csv"
    )
    with open(long_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "count"])
        for i, true_lab in enumerate(gt_labels):
            for j, pred_lab in enumerate(gt_labels):
                c = int(cm[i, j])
                if c > 0:
                    writer.writerow([true_lab, pred_lab, c])

    # ---- Summary ----
    summary_path = os.path.join(
        out_dir, f"confusion_{run_name}_{level_name}_t{t_str}_summary.txt"
    )
    with open(summary_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"level={level_name}\n")
        f.write(f"threshold={threshold}\n")
        f.write(f"evaluated_samples={len(true_labels)}\n")
        f.write(f"total_samples={total_samples}\n")
        f.write(f"coverage={len(true_labels)/total_samples if total_samples else 0:.6f}\n")
        f.write(f"matrix_samples={len(y_true_cm)}\n")
        f.write(f"dropped_from_matrix_due_to_phantom_preds={len(true_labels)-len(y_true_cm)}\n")
        f.write(f"n_gt_labels={len(gt_labels)}\n")
        f.write(f"gt_labels={gt_labels}\n")
        f.write(f"n_phantom_classes={len(phantom_classes)}\n")
        f.write(f"n_phantom_predictions={n_phantom_preds}\n")
        if phantom_classes:
            f.write(f"phantom_classes={phantom_classes}\n")





def _get_depth_labels(
    pred_nodes,
    all_target_ids,
    classification_tree,
    depth,
    order_to_superfamilies=None,  # NEW
):
    """
    Extract predicted and target label strings at a specific tree depth.

    depth=1 → order level
    depth=2 → superfamily level (order/sf)

    Returns None for samples where:
      - GT doesn't reach this depth
      - Prediction doesn't reach this depth
      - GT label at this depth is not in trained taxonomy (order_to_superfamilies)
    """
    pred_labels = []
    true_labels = []

    # ---- Build trained-label sets (from provided dict) ----
    trained_orders = set()
    trained_sfs = set()
    trained_order_sf = set()

    if order_to_superfamilies is not None:
        trained_orders = set(order_to_superfamilies.keys())
        for o, sfs in order_to_superfamilies.items():
            for sf in sfs:
                trained_sfs.add(sf)
                trained_order_sf.add(f"{o}/{sf}")

    for pred_node, target_id in zip(pred_nodes, all_target_ids):
        target_node = classification_tree.node_list[target_id.item()]
        target_path = target_node.path
        pred_path = pred_node.path

        # GT too shallow
        if len(target_path) <= depth:
            true_labels.append(None)
            pred_labels.append(None)
            continue

        # Prediction too shallow
        if len(pred_path) <= depth:
            true_labels.append(None)
            pred_labels.append(None)
            continue

        # Candidate labels
        true_label = str(target_path[depth])
        pred_label = str(pred_path[depth])

        if depth >= 2:
            true_order = str(target_path[1])
            pred_order = str(pred_path[1])
            true_label = f"{true_order}/{true_label}"
            pred_label = f"{pred_order}/{pred_label}"

        # ---- NEW: trained-taxonomy membership check on GT ----
        if order_to_superfamilies is not None:
            gt_is_trained = True

            if depth == 1:
                gt_is_trained = (true_label in trained_orders)
            elif depth >= 2:
                # strict pair check; ensures order/sf combo is trained
                gt_is_trained = (true_label in trained_order_sf)

            if not gt_is_trained:
                true_labels.append(None)
                pred_labels.append(None)
                continue

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    return true_labels, pred_labels






def _get_depth_labels_older(pred_nodes, all_target_ids, classification_tree, depth):
    """
    Extract predicted and target label strings at a specific tree depth.

    depth=1 → order level (LINE, SINE, TIR, DIRS, ...)
    depth=2 → superfamily level (LINE/CR1, SINE/ID, TIR/hAT, ...)

    Returns None for samples where:
      - The TRUE target doesn't reach the requested depth (e.g., DIRS at depth=2)
      - The PREDICTION didn't reach the requested depth (e.g., stopped at order due to threshold)
    The caller must filter these out.
    """
    pred_labels = []
    true_labels = []

    for pred_node, target_id in zip(pred_nodes, all_target_ids):
        target_node = classification_tree.node_list[target_id.item()]
        target_path = target_node.path
        pred_path = pred_node.path

        # If the true target doesn't reach this depth, skip entirely
        if len(target_path) <= depth:
            true_labels.append(None)
            pred_labels.append(None)
            continue

        # If the prediction didn't reach this depth (stopped early due to threshold),
        # this sample wasn't classified at this level — skip it
        if len(pred_path) <= depth:
            true_labels.append(None)
            pred_labels.append(None)
            continue

        true_label = str(target_path[depth])
        pred_label = str(pred_path[depth])

        # For superfamily level, prepend order to disambiguate
        if depth >= 2:
            true_order = str(target_path[1])
            pred_order = str(pred_path[1])
            true_label = f"{true_order}/{true_label}"
            pred_label = f"{pred_order}/{pred_label}"

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    return true_labels, pred_labels




def _compute_sklearn_metrics(true_labels, pred_labels, total_samples, level_name=""):
    """Compute accuracy and macro precision/recall/F1.

    Macro averaging is performed ONLY over classes that actually exist in the
    ground truth (`set(true_labels)`). Predictions are NOT filtered or remapped:
    if the model predicts a class that doesn't exist in the ground truth
    ("phantom" class), that prediction is still evaluated — it simply counts as
    a wrong prediction (the corresponding true class gets a recall miss, and
    the phantom class is just not part of the macro average).

    Args:
        true_labels:    list of true label strings (already filtered, no Nones)
        pred_labels:    list of predicted label strings (already filtered, no Nones)
        total_samples:  total number of samples BEFORE filtering (for coverage ratio)
        level_name:     name for printing
    """
    from collections import Counter

    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)

    # Classes that truly exist in the ground truth — these are what we average over.
    gt_classes = sorted(true_counts.keys())

    # Classes the model predicted that do NOT exist in the ground truth (phantoms).
    # We do NOT remove these predictions; we only report them for transparency.
    phantom_classes = sorted(set(pred_counts.keys()) - set(true_counts.keys()))

    # Ground-truth classes the model never predicted (recall = 0 for these).
    never_predicted = sorted(set(true_counts.keys()) - set(pred_counts.keys()))

    # Accuracy uses ALL samples and ALL predictions (phantom predictions count as wrong).
    acc = accuracy_score(true_labels, pred_labels)

    # Macro P/R/F1 averaged ONLY over ground-truth classes.
    # Phantom predictions stay in `pred_labels`, so they still cause:
    #   - the true class of that sample to suffer a recall miss, and
    #   - sklearn to score them as incorrect (they just aren't averaged as their own class).
    prec = precision_score(true_labels, pred_labels, labels=gt_classes, average="macro", zero_division=0)
    rec  = recall_score(   true_labels, pred_labels, labels=gt_classes, average="macro", zero_division=0)
    f1   = f1_score(       true_labels, pred_labels, labels=gt_classes, average="macro", zero_division=0)

    if level_name:
        n_evaluated = len(true_labels)
        coverage = n_evaluated / total_samples if total_samples > 0 else 0.0
        n_phantom_preds = sum(pred_counts[c] for c in phantom_classes)

        print(f"\n  --- {level_name} ---")
        print(f"  Samples evaluated:  {n_evaluated} / {total_samples} ({coverage:.1%})")
        print(f"  Ground-truth classes (averaged over): {len(gt_classes)}")
        print(f"  Accuracy:           {acc:.4f}")
        print(f"  Precision:          {prec:.4f} (macro over ground-truth classes)")
        print(f"  Recall:             {rec:.4f} (macro over ground-truth classes)")
        print(f"  F1:                 {f1:.4f} (macro over ground-truth classes)")

        if never_predicted:
            print(f"\n  ⚠️  Ground-truth classes the model NEVER predicted "
                  f"({len(never_predicted)}) — included in macro avg with recall=0:")
            print(f"  {'Class':<40} {'True Support':>15}")
            print(f"  {'-'*55}")
            for cls in never_predicted:
                print(f"  {cls:<40} {true_counts[cls]:>15}")
            n_missed = sum(true_counts[c] for c in never_predicted)
            print(f"  {'-'*55}")
            print(f"  {'TOTAL missed samples':<40} {n_missed:>15}")

        if phantom_classes:
            print(f"\n  ⚠️  Predicted classes NOT in ground truth "
                  f"({len(phantom_classes)}) — {n_phantom_preds} predictions, "
                  f"still counted as wrong, NOT averaged:")
            print(f"  {'Class':<40} {'# Predictions':>15}")
            print(f"  {'-'*55}")
            for cls in phantom_classes:
                print(f"  {cls:<40} {pred_counts[cls]:>15}")

        print(f"\n  Classification report (ground-truth classes only):")
        print(classification_report(true_labels, pred_labels, labels=gt_classes, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_samples": len(true_labels),
        "n_total_samples": total_samples,
        "n_gt_classes": len(gt_classes),
        "never_predicted_classes": never_predicted,
        "never_predicted_support": {c: true_counts[c] for c in never_predicted},
        "phantom_predicted_classes": phantom_classes,
        "phantom_prediction_counts": {c: pred_counts[c] for c in phantom_classes},
    }














def calcmetrics_torcheval_multiclass_filtered(
    y_true: torch.Tensor, 
    y_pred_logits: torch.Tensor,
    pad_token_id: int = 0,
    ignore_index: int = -100,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate AUROC, AUPRC, accuracy, F1, precision, and recall for multi-class classification.
    
    Args:
        y_true: Ground truth class labels, shape (N,) with values in [0, num_classes-1]
        y_pred_logits: Predicted logits, shape (N, num_classes)
        pad_token_id: Token ID to ignore (default: 0)
        ignore_index: Label value to ignore (default: -100)
        average: 'weighted', 'macro', or 'micro'
    
    Returns:
        Dictionary with AUROC, AUPRC, normAUPRC, baseline_auprc, accuracy, F1, precision, recall
    """
    
    # Ensure tensors are on CPU and detached
    y_true = y_true.detach().cpu().long()
    y_pred_logits = y_pred_logits.detach().cpu().float()
    
    # Create valid mask: exclude ignore_index and pad_token_id
    valid_mask = (y_true != ignore_index) & (y_true != pad_token_id)
    
    # Filter out invalid positions
    y_true_valid = y_true[valid_mask]
    y_pred_logits_valid = y_pred_logits[valid_mask]
    
    # Handle edge cases
    if y_true_valid.numel() == 0:
        print("⚠️ WARNING: No valid samples after filtering for multi-class classification")
        return {
            'AUROC': 0.5,
            'AUPRC': 0.0,
            'normAUPRC': 0.0,
            'baseline_auprc': 0.0,
            'accuracy': 0.0,
            'F1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
    
    num_classes = y_pred_logits_valid.shape[1]
    
    # Convert logits to probabilities using softmax
    y_pred_probs = F.softmax(y_pred_logits_valid, dim=1)
    
    # Calculate accuracy using argmax
    predictions = y_pred_logits_valid.argmax(dim=-1)
    accuracy = (predictions == y_true_valid).float().mean().item()
    
    # ============================================================
    # F1, PRECISION, RECALL (Macro for multi-class)
    # ============================================================


    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")
    f1_metric.update(predictions, y_true_valid)
    f1 = f1_metric.compute().item()


    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes)
    conf_matrix_metric.update(predictions, y_true_valid)
    conf_matrix = conf_matrix_metric.compute()  # (num_classes, num_classes)

    # Per-class true positives, predicted positives, actual positives
    tp = conf_matrix.diag()
    pred_positives = conf_matrix.sum(dim=0)   # column sums
    actual_positives = conf_matrix.sum(dim=1) # row sums

    # Per-class precision and recall (avoiding division by zero)
    per_class_precision = tp / pred_positives.clamp(min=1)
    per_class_recall = tp / actual_positives.clamp(min=1)

    # Macro average: only over classes that exist in y_true
    classes_present = actual_positives > 0
    if classes_present.any():
        precision = per_class_precision[classes_present].mean().item()
        recall = per_class_recall[classes_present].mean().item()
    else:
        precision = 0.0
        recall = 0.0
    
    # ============================================================
    # AUROC AND AUPRC
    # ============================================================
    try:
        auroc_metric = MulticlassAUROC(
            num_classes=num_classes,
            average=average,
        )
        # Fix: Use .update() and .compute() instead of calling the object
        auroc_metric.update(y_pred_probs, y_true_valid)
        auroc = auroc_metric.compute().item()
        
        auprc_metric = MulticlassAUPRC(num_classes=num_classes, average=average)
        auprc_metric.update(y_pred_probs, y_true_valid)
        auprc = auprc_metric.compute().item()
        

    except Exception as e:
        print(f"⚠️ Warning: AUROC/AUPRC computation failed: {e}")
        print(f"   y_pred_probs shape: {y_pred_probs.shape}")
        print(f"   y_true_valid shape: {y_true_valid.shape}, unique: {y_true_valid.unique()}")
        return {
            'AUROC': 0.5,
            'AUPRC': 0.0,
            'normAUPRC': 0.0,
            'baseline_auprc': 0.0,
            'accuracy': accuracy,
            'F1': f1,
            'precision': precision,
            'recall': recall,
        }
    
    # ============================================================
    # BASELINE AUPRC
    # ============================================================
    y_true_onehot = F.one_hot(y_true_valid.long(), num_classes=num_classes).float()
    
    baseline_per_class = []
    weights = []
    
    for class_idx in range(num_classes):
        y_true_binary = y_true_onehot[:, class_idx]
        num_pos = torch.sum(y_true_binary).item()
        
        if num_pos > 0:
            pos_rate = torch.mean(y_true_binary).item()
            baseline_per_class.append(pos_rate)
            weights.append(num_pos)
    
    if len(baseline_per_class) == 0:
        baseline_auprc = 0.0
    elif average == 'macro':
        baseline_auprc = sum(baseline_per_class) / len(baseline_per_class)
    elif average == 'weighted':
        total_weight = sum(weights)
        baseline_auprc = sum(b * w for b, w in zip(baseline_per_class, weights)) / total_weight
    else:
        num_pos_total = torch.sum(y_true_onehot).item()
        baseline_auprc = num_pos_total / y_true_onehot.numel()
    
    # ============================================================
    # NORMALIZED AUPRC
    # ============================================================
    if baseline_auprc < 1.0:
        norm_auprc = (auprc - baseline_auprc) / (1.0 - baseline_auprc)
    else:
        norm_auprc = 0.0
    
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'normAUPRC': norm_auprc,
        'baseline_auprc': baseline_auprc,
        'accuracy': accuracy,
        'F1': f1,
        'precision': precision,
        'recall': recall,
    }

