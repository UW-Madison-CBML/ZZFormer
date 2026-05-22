import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics.functional import binary_auroc, binary_auprc
from sklearn.metrics import recall_score, precision_score

import torch
from torcheval.metrics.functional import binary_auroc, binary_auprc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from typing import Dict, Tuple


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

















def calcmetrics_torchmetrics_multiclass_filtered_test(
    y_true: torch.Tensor,
    y_pred_probs: torch.Tensor,   # <-- PROBS, not logits
    pad_token_id: int = 0,
    ignore_index: int = -100,
    average: str = "weighted"
):
    y_true = y_true.detach().cpu()
    y_pred_probs = y_pred_probs.detach().cpu()

    valid_mask = (y_true != ignore_index) & (y_true != pad_token_id)
    y_true_valid = y_true[valid_mask]
    y_pred_probs_valid = y_pred_probs[valid_mask]
    print("In targets: ", y_pred_probs)
    print("Valid targets: ", y_pred_probs_valid," \n", valid_mask)
    print("probabilities after validifying: ", y_pred_probs_valid)

    if y_true_valid.numel() == 0:
        return dict(AUROC=0.5, AUPRC=0.0, normAUPRC=0.0, baseline_auprc=0.0, accuracy=0.0)

    num_classes = y_pred_probs_valid.shape[1]
    print("num_classes: ",y_pred_probs_valid.shape[1])



    auroc = MulticlassAUROC(
        num_classes=num_classes,
        average=average
    )(y_pred_probs_valid, y_true_valid).item()

    auprc = MulticlassAveragePrecision(
        num_classes=num_classes,
        average=average
    )(y_pred_probs_valid, y_true_valid).item()

    y_pred_probs2 = F.softmax(y_pred_probs_valid)
    acc = (y_pred_probs2.argmax(dim=1) == y_true_valid).float().mean().item()

    baseline = torch.bincount(y_true_valid, minlength=num_classes).float()
    baseline = baseline[baseline > 0].sum() / y_true_valid.numel()

    norm_auprc = (auprc - baseline) / (1 - baseline)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "normAUPRC": norm_auprc,
        "baseline_auprc": baseline.item(),
        "accuracy": acc
    }



























def calculate_auroc_torch(y_true:  torch.Tensor, y_pred_probs: torch.Tensor) -> float:
    """
    Calculate AUROC using torch (binary classification)
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_pred_probs: Predicted probabilities for class 1, shape (N,)
    
    Returns:
        AUROC score (float)
    """
    # Sort by prediction probabilities in descending order
    sorted_indices = torch.argsort(y_pred_probs, descending=True)
    sorted_labels = y_true[sorted_indices]
    
    # Calculate TPR and FPR at different thresholds
    num_pos = torch.sum(y_true).float()
    num_neg = torch.numel(y_true) - num_pos
    
    if num_pos == 0 or num_neg == 0:
        return 0.5  # Undefined case
    
    # Cumulative TP and FP
    tp = torch.cumsum(sorted_labels. float(), dim=0)
    fp = torch.arange(1, len(sorted_labels) + 1, device=y_true.device).float() - tp
    
    # TPR and FPR
    tpr = tp / num_pos
    fpr = fp / num_neg
    
    # Add (0,0) point for AUROC calculation
    tpr = torch.cat([torch.tensor([0.0], device=y_true.device), tpr])
    fpr = torch.cat([torch.tensor([0.0], device=y_true.device), fpr])
    
    # Calculate AUROC using trapezoidal rule
    auroc = torch.trapz(tpr, fpr)
    
    return auroc. item()


def calculate_auprc_torch(y_true: torch. Tensor, y_pred_probs: torch.Tensor) -> float:
    """
    Calculate AUPRC using torch (binary classification)
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_pred_probs: Predicted probabilities for class 1, shape (N,)
    
    Returns:
        AUPRC score (float)
    """
    # Sort by prediction probabilities in descending order
    sorted_indices = torch.argsort(y_pred_probs, descending=True)
    sorted_labels = y_true[sorted_indices]
    
    num_pos = torch.sum(y_true).float()
    
    if num_pos == 0:
        return 0.0
    
    # Cumulative TP
    tp = torch.cumsum(sorted_labels.float(), dim=0)
    # Cumulative predictions (all 1's since we're going through sorted list)
    fp = torch.arange(1, len(sorted_labels) + 1, device=y_true. device).float() - tp
    
    # Precision and Recall
    precision = tp / (tp + fp)
    recall = tp / num_pos
    
    # Add (0, 1) point (at threshold infinity, no predictions are positive)
    precision = torch.cat([torch.tensor([1.0], device=y_true.device), precision])
    recall = torch.cat([torch.tensor([0.0], device=y_true.device), recall])
    
    # Calculate AUPRC using trapezoidal rule
    auprc = torch.trapz(precision, recall)
    
    return auprc.item()


def calculate_metrics_torch_binary(y_true: torch. Tensor, y_pred_probs: torch.Tensor) -> Dict[str, float]:
    """
    Calculate AUROC, AUPRC, and normalized AUPRC for binary classification
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_pred_probs: Predicted probabilities for class 1, shape (N,)
    
    Returns:
        Dictionary with AUROC, AUPRC, and normalized AUPRC
    """
    # Ensure tensors are on CPU and detached
    y_true = y_true.detach().cpu()
    y_pred_probs = y_pred_probs. detach().cpu()
    
    # Calculate AUROC
    auroc = calculate_auroc_torch(y_true, y_pred_probs)
    
    # Calculate AUPRC
    auprc = calculate_auprc_torch(y_true, y_pred_probs)
    
    # Calculate baseline AUPRC (random classifier)
    pos_rate = torch.mean(y_true. float()).item()
    baseline_auprc = pos_rate
    
    # Normalize AUPRC
    if baseline_auprc < 1.0:
        norm_auprc = (auprc - baseline_auprc) / (1.0 - baseline_auprc)
    else:
        norm_auprc = 0.0
    
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'normAUPRC': norm_auprc,
        'baseline_auprc': baseline_auprc
    }


#Useful for our case - multi-class classification (like MLM)
def calculate_metrics_torch_multiclass(y_true: torch.Tensor, 
                                       y_pred_logits: torch.Tensor,
                                       average: str = 'weighted') -> Dict[str, float]: 
    """
    Calculate AUROC and AUPRC for multi-class classification (like MLM)
    
    Args:
        y_true: Ground truth class labels, shape (N,) with values in [0, num_classes-1]
        y_pred_logits: Predicted logits or probabilities, shape (N, num_classes)
        average: 'weighted', 'macro', or 'micro'
    
    Returns:
        Dictionary with AUROC, AUPRC, and normalized AUPRC
    """
    # Ensure tensors are on CPU and detached
    y_true = y_true.detach().cpu()
    y_pred_logits = y_pred_logits. detach().cpu()
    
    # Convert logits to probabilities
    y_pred_probs = F.softmax(y_pred_logits, dim=1)
    
    num_classes = y_pred_probs.shape[1]
    
    # One-hot encode labels
    y_true_onehot = F.one_hot(y_true. long(), num_classes=num_classes).float()
    
    if average == 'micro':
        # Flatten for micro-averaging
        y_true_flat = y_true_onehot.view(-1)
        y_pred_flat = y_pred_probs. view(-1)
        
        auroc = calculate_auroc_torch(y_true_flat, y_pred_flat)
        auprc = calculate_auprc_torch(y_true_flat, y_pred_flat)
        
    elif average == 'macro' or average == 'weighted': 
        # Calculate per-class metrics
        auroc_per_class = []
        auprc_per_class = []
        weights = []
        
        for class_idx in range(num_classes):
            y_true_binary = y_true_onehot[:, class_idx]
            y_pred_binary = y_pred_probs[:, class_idx]
            
            num_pos = torch.sum(y_true_binary).item()
            
            if num_pos == 0:
                continue  # Skip classes with no positive examples
            
            auroc_cls = calculate_auroc_torch(y_true_binary, y_pred_binary)
            auprc_cls = calculate_auprc_torch(y_true_binary, y_pred_binary)
            
            auroc_per_class.append(auroc_cls)
            auprc_per_class.append(auprc_cls)
            
            if average == 'weighted':
                weights.append(num_pos)
        
        if len(auroc_per_class) == 0:
            return {
                'AUROC': 0.5,
                'AUPRC': 0.0,
                'normAUPRC': 0.0,
                'baseline_auprc': 0.0
            }
        
        if average == 'macro':
            auroc = sum(auroc_per_class) / len(auroc_per_class)
            auprc = sum(auprc_per_class) / len(auprc_per_class)
        else:  # weighted
            total_weight = sum(weights)
            auroc = sum(a * w for a, w in zip(auroc_per_class, weights)) / total_weight
            auprc = sum(a * w for a, w in zip(auprc_per_class, weights)) / total_weight
    
    # Calculate baseline AUPRC
    pos_rates = torch.mean(y_true_onehot, dim=0)
    baseline_auprc = torch.mean(pos_rates).item()
    
    # Normalize AUPRC
    if baseline_auprc < 1.0:
        norm_auprc = (auprc - baseline_auprc) / (1.0 - baseline_auprc)
    else:
        norm_auprc = 0.0
    
    return {
        'AUROC':  auroc,
        'AUPRC': auprc,
        'normAUPRC':  norm_auprc,
        'baseline_auprc': baseline_auprc
    }













def calculate_metrics_torch_multiclass_filtered(y_true:  torch.Tensor, 
                                                y_pred_logits: torch.Tensor,
                                                pad_token_id: int = 0,
                                                ignore_index: int = -100,
                                                average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate AUROC, AUPRC, and normalized AUPRC for multi-class classification (like MLM)
    Filters out pad_token_id and ignore_index positions
    
    Args:
        y_true: Ground truth class labels, shape (B, L) with values in [0, num_classes-1]
        y_pred_logits: Predicted logits or probabilities, shape (B, L, num_classes)
        pad_token_id: Token ID to ignore (default: 0)
        ignore_index: Label value to ignore (default: -100)
        average: 'weighted', 'macro', or 'micro'
    
    Returns:
        Dictionary with AUROC, AUPRC, normalized AUPRC, and accuracy
    """
    # Ensure tensors are on CPU and detached
    y_true = y_true.detach().cpu()
    y_pred_logits = y_pred_logits.detach().cpu()
    
    # Create valid mask:  exclude ignore_index and pad_token_id
    valid_mask = (y_true != ignore_index) & (y_true != pad_token_id)
    
    # Filter out invalid positions
    y_true_valid = y_true[valid_mask]
    y_pred_logits_valid = y_pred_logits[valid_mask]  # Shape: (num_valid, num_classes)
    
    # Handle edge cases
    if y_true_valid.numel() == 0:
        return {
            'AUROC': 0.5,
            'AUPRC': 0.0,
            'normAUPRC': 0.0,
            'baseline_auprc': 0.0,
            'accuracy': 0.0
        }
    
    # Convert logits to probabilities
    y_pred_probs = F.softmax(y_pred_logits_valid, dim=1)
    
    num_classes = y_pred_probs.shape[1]
    
    # One-hot encode labels
    y_true_onehot = F. one_hot(y_true_valid. long(), num_classes=num_classes).float()
    
    # Calculate accuracy
    predictions = y_pred_logits_valid.argmax(dim=-1)
    accuracy = (predictions == y_true_valid).float().mean().item()
    
    if average == 'micro':
        # Flatten for micro-averaging
        y_true_flat = y_true_onehot.view(-1)
        y_pred_flat = y_pred_probs. view(-1)
        
        auroc = calculate_auroc_torch(y_true_flat, y_pred_flat)
        auprc = calculate_auprc_torch(y_true_flat, y_pred_flat)
        
    elif average == 'macro' or average == 'weighted': 
        # Calculate per-class metrics
        auroc_per_class = []
        auprc_per_class = []
        weights = []
        
        for class_idx in range(num_classes):
            y_true_binary = y_true_onehot[:, class_idx]
            y_pred_binary = y_pred_probs[:, class_idx]
            
            num_pos = torch.sum(y_true_binary).item()
            
            if num_pos == 0:
                continue  # Skip classes with no positive examples
            
            auroc_cls = calculate_auroc_torch(y_true_binary, y_pred_binary)
            auprc_cls = calculate_auprc_torch(y_true_binary, y_pred_binary)
            
            auroc_per_class.append(auroc_cls)
            auprc_per_class.append(auprc_cls)
            
            if average == 'weighted':
                weights.append(num_pos)
        
        if len(auroc_per_class) == 0:
            return {
                'AUROC': 0.5,
                'AUPRC': 0.0,
                'normAUPRC': 0.0,
                'baseline_auprc': 0.0,
                'accuracy': accuracy
            }
        
        if average == 'macro':
            auroc = sum(auroc_per_class) / len(auroc_per_class)
            auprc = sum(auprc_per_class) / len(auprc_per_class)
        else:  # weighted
            total_weight = sum(weights)
            auroc = sum(a * w for a, w in zip(auroc_per_class, weights)) / total_weight
            auprc = sum(a * w for a, w in zip(auprc_per_class, weights)) / total_weight
    
    # Calculate baseline AUPRC
    pos_rates = torch.mean(y_true_onehot, dim=0)
    baseline_auprc = torch.mean(pos_rates).item()
    
    # Normalize AUPRC
    if baseline_auprc < 1.0:
        norm_auprc = (auprc - baseline_auprc) / (1.0 - baseline_auprc)
    else:
        norm_auprc = 0.0
    
    return {
        'AUROC':  auroc,
        'AUPRC': auprc,
        'normAUPRC':  norm_auprc,
        'baseline_auprc': baseline_auprc,
        'accuracy': accuracy
    }















####################################################### Old methods ###########################################################
class TokenLevelEvaluator:
    def __init__(self, device="cpu"):
        self.device = device
        self.reset()

    def reset(self):
        # store all predicted scores and binary labels
        self.all_scores = []
        self.all_binary = []

    def update(self, logits, true_ids, mask):
        """
        logits: (B, L, V) model output scores
        true_ids: (B, L) true token ids 
        mask: (B, L) boolean, True where sample is valid for evaluation
              (e.g. mlm_mask for MLM, reconstruction mask for tasked)
        """
        B, L, V = logits.shape

        # convert to probabilities for the true class
        probs = torch.softmax(logits, dim=-1)  # (B,L,V)
        # gather prob assigned to the true token
        scores = probs[torch.arange(B)[:,None], torch.arange(L)[None,:], true_ids]

        # flatten
        scores_flat = scores.view(-1)
        labels_flat = true_ids.view(-1)
        mask_flat = mask.view(-1)

        # select only valid positions
        valid_scores = scores_flat[mask_flat]
        # binary label: 1 if predicted label matches true label (positive) else 0
        predicted_labels = (valid_scores == labels_flat[mask_flat]).long()

        self.all_scores.append(valid_scores.detach().cpu())
        self.all_binary.append(predicted_labels.detach().cpu())

    def compute(self):
        """
        Returns: (auroc, auprc, norm_auprc)
        norm_auprc = auprc - positive_label_fraction
        """
        if not self.all_scores:
            return None, None, None

        scores_cat = torch.cat(self.all_scores)
        binary_cat = torch.cat(self.all_binary)

        roc = binary_auroc(scores_cat, binary_cat)
        prc = binary_auprc(scores_cat, binary_cat)
        baseline = binary_cat.float().mean()

        return float(roc), float(prc), float(prc - baseline)


#-----------------------------MLM Model-----------------------------

'''evaluator = TokenLevelEvaluator(device=device)
model.eval()

with torch.no_grad():
    for batch in val_loader_mlm:
        tokens = batch["tokens"].to(device)
        labels = batch["mlm_labels"].to(device)
        mask = labels != -100   # only masked positions matter

        logits, _ = model(tokens, mlm_labels=None)  # forward without mlm loss

        evaluator.update(logits, labels, mask)

roc, prc, norm_prc = evaluator.compute()
print(f"MLM AUROC={roc:.4f}, AUPRC={prc:.4f}, NormAUPRC={norm_prc:.4f}")'''



#-----------------------------Shifted Seq Task Model-----------------------------
'''evaluator = TokenLevelEvaluator(device=device)
model.eval()

with torch.no_grad():
    for batch in val_loader_tasked:
        tokens = batch["tokens"].to(device)
        targets = batch["target_tokens"].to(device)
        mask = targets >= 0  # valid reconstruction positions

        logits, _ = model(tokens, target_tokens=None)

        evaluator.update(logits, targets, mask)

roc, prc, norm_prc = evaluator.compute()
print(f"Task AUROC={roc:.4f}, AUPRC={prc:.4f}, NormAUPRC={norm_prc:.4f}")'''













############################################################## Adapted from previous code ##############################################################

def get_masked(logits, labels, mask):
    """Mask logits and labels using a given mask"""
    if isinstance(mask, np.ndarray):
        masked_logits = logits[mask.astype(bool)]
        masked_labels = labels[mask.astype(bool)]
    else:
        masked_logits = logits[mask.bool()].view(-1)
        masked_labels = labels[mask.bool()].view(-1)

    return masked_logits, masked_labels

def get_auroc(masked_logits, masked_labels):
    """Evaluate AUROC, AUPRC, and norm AUPRC for given logits, labels"""

    if isinstance(masked_labels, np.ndarray):
        masked_labels = torch.tensor(masked_labels)
    if isinstance(masked_logits, np.ndarray):
        masked_logits = torch.tensor(masked_logits)
    
    roc = binary_auroc(masked_logits, masked_labels)
    prc = binary_auprc(masked_logits, masked_labels)
    
    baseline = masked_labels.sum()/len(masked_labels)
    
    return abs(roc), abs(prc), abs(prc)-baseline




def compute_auroc_for_mlm(logits, mlm_labels):
    # flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = mlm_labels.view(-1)

    # mask
    mask = labels_flat != -100
    if not mask.any():
        return None

    probs = torch.softmax(logits_flat, dim=1)
    true_scores = probs[torch.arange(labels_flat.size(0)), labels_flat]

    masked_logits, masked_labels = get_masked(true_scores, true_scores, mask)

    binary_labels = (masked_labels > 0).long()
    return get_auroc(masked_logits, binary_labels)


#Usage example:
#-----------------------------MLM Model-----------------------------
# logits: (B, L, V) from model
# mlm_labels: (B, L), -100 for non-mask, else true token index

# flatten
# logits_flat = logits.view(-1, logits.size(-1))
# labels_flat = mlm_labels.view(-1)

# # mask out unmasked and padding positions
# mask = labels_flat != -100

# # predicted scores: use probability of the true class
# probs = torch.softmax(logits_flat, dim=1)
# true_scores = probs[torch.arange(labels_flat.size(0)), labels_flat]
# masked_logits, masked_labels = get_masked(true_scores, labels_flat, mask)


# binary_label = 1 if predicted token is true token else 0
# binary_labels = (masked_logits.argmax(dim=-1) == masked_labels).long()
# roc, prc, norm_prc = get_auroc(masked_logits, binary_labels)


#-----------------------------Tasked Model-----------------------------
# logits = model.sequence_head(h)  # shape (B,L,V)
# target = batch["target_tokens"]  # true token ids

# logits_flat = logits.view(-1, logits.size(-1))
# target_flat = target.view(-1)

# # mask out paddings
# mask = target_flat >= 0

# # predicted probability of true class
# probs = torch.softmax(logits_flat, dim=1)
# true_scores = probs[torch.arange(target_flat.size(0)), target_flat]

# masked_logits, masked_labels = get_masked(true_scores, target_flat, mask)

# # binary labels: correct or not
# binary_labels = (masked_logits.argmax(dim=-1) == masked_labels).long()

# roc, prc, norm_prc = get_auroc(masked_logits, binary_labels)
