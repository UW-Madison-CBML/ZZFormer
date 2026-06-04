import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "/staging/kkumari/TERRIER"
DATASET_NAME = sys.argv[2] if len(sys.argv) > 2 else "repbase"
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "Drosophila_melanogaster"
THRESHOLD = sys.argv[4] if len(sys.argv) > 4 else "0.0"

CM_EXT = ".CSV"
NUM_FOLDS = 5
MODEL_NAME = "Terrier"
OTHER_LABEL = "__OTHER__"


def restrict_to_gt_classes_bucket_pred_only(cm_df: pd.DataFrame):
    """
    Keep only GT-present classes (row sum > 0) as evaluation classes.
    Bucket predictions into non-GT classes into OTHER_LABEL so they are penalized.
    """
    # Align rows/cols to the same universe
    all_labels = sorted(set(cm_df.index) | set(cm_df.columns))
    cm_df = cm_df.reindex(index=all_labels, columns=all_labels, fill_value=0)

    row_sums = cm_df.sum(axis=1)
    gt_classes = row_sums[row_sums > 0].index.tolist()

    if not gt_classes:
        return cm_df.iloc[0:0, 0:0], []

    # Keep only GT rows
    cm_rows = cm_df.loc[gt_classes, :]

    # Bucket predicted-only columns
    pred_only_cols = [c for c in cm_rows.columns if c not in gt_classes]
    if pred_only_cols:
        cm_rows[OTHER_LABEL] = cm_rows[pred_only_cols].sum(axis=1)
        cm_rows = cm_rows.drop(columns=pred_only_cols)

    # Ensure all GT columns exist (+ OTHER if present)
    cols = gt_classes + ([OTHER_LABEL] if OTHER_LABEL in cm_rows.columns else [])
    cm_rows = cm_rows.reindex(columns=cols, fill_value=0)

    return cm_rows, gt_classes


def metrics_from_confusion_matrix(cm_path):
    cm_df = pd.read_csv(cm_path, index_col=0)

    cm_eval, gt_classes = restrict_to_gt_classes_bucket_pred_only(cm_df)
    if not gt_classes:
        print(f"  [WARNING] No ground-truth classes in {cm_path}")
        return None

    # Reconstruct y_true/y_pred
    y_true, y_pred = [], []
    row_labels = list(cm_eval.index)
    col_labels = list(cm_eval.columns)

    for i, t in enumerate(row_labels):
        for j, p in enumerate(col_labels):
            count = int(cm_eval.iloc[i, j])
            if count <= 0:
                continue
            y_true.extend([t] * count)
            y_pred.extend([p] * count)

    macro_precision = precision_score(y_true, y_pred, labels=gt_classes, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=gt_classes, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=gt_classes, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    other_count = int(cm_eval[OTHER_LABEL].sum()) if OTHER_LABEL in cm_eval.columns else 0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "num_gt_classes": len(gt_classes),
        "other_pred_count": other_count,
    }


def evaluate_model_across_folds(level):
    fold_metrics = []
    available_folds = []

    for fold in range(NUM_FOLDS):
        fold_dir = os.path.join(BASE_DIR, DATASET_NAME, f"fold_{fold}")
        outdir = os.path.join(fold_dir, "inference_eval", OUTNAME)

        if level == "superfamily":
            cm_path = os.path.join(outdir, f"fold_{fold}.confusion_matrix.superfamily.threshold{THRESHOLD}{CM_EXT}")
        elif level == "order":
            cm_path = os.path.join(outdir, f"fold_{fold}.confusion_matrix.order_only.threshold{THRESHOLD}{CM_EXT}")
        else:
            raise ValueError(f"Unknown level: {level}")

        if not os.path.exists(cm_path):
            print(f"  [WARNING] Missing: {cm_path}")
            continue

        metrics = metrics_from_confusion_matrix(cm_path)
        if metrics is None:
            print(f"  [WARNING] No usable classes in fold {fold}, skipping.")
            continue

        fold_metrics.append(metrics)
        available_folds.append(fold)

    return fold_metrics, available_folds


def print_model_results(model_name, level, fold_metrics, available_folds):
    print(f"\n  Model: {model_name} | Level: {level}")
    print(f"  Available folds: {available_folds}")

    if not fold_metrics:
        print("  No folds found.\n")
        return

    for fold_idx, m in zip(available_folds, fold_metrics):
        print(
            f"    Fold {fold_idx}: "
            f"Macro-P={m['macro_precision']:.4f}  "
            f"Macro-R={m['macro_recall']:.4f}  "
            f"Macro-F1={m['macro_f1']:.4f}  "
            f"Accuracy={m['accuracy']:.4f}  "
            f"(GT-classes={m['num_gt_classes']}, OTHER-preds={m['other_pred_count']})"
        )

    print(f"  ── Mean ± Std over {len(available_folds)} folds ──")
    for k, label in [("macro_precision", "Macro-P"),
                     ("macro_recall", "Macro-R"),
                     ("macro_f1", "Macro-F1"),
                     ("accuracy", "Accuracy")]:
        vals = [m[k] for m in fold_metrics]
        print(f"    {label:9s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


def main():
    summary_rows = []

    # ORDER
    print("=" * 70)
    print("ORDER-LEVEL RESULTS (averaged across 5 folds)")
    print("=" * 70)

    fold_metrics, available_folds = evaluate_model_across_folds("order")
    print_model_results(MODEL_NAME, "order", fold_metrics, available_folds)
    if fold_metrics:
        summary_rows.append({
            "Level": "Order",
            "Model": MODEL_NAME,
            "Folds": len(available_folds),
            "Macro-P": np.mean([m["macro_precision"] for m in fold_metrics]),
            "Macro-P_std": np.std([m["macro_precision"] for m in fold_metrics]),
            "Macro-R": np.mean([m["macro_recall"] for m in fold_metrics]),
            "Macro-R_std": np.std([m["macro_recall"] for m in fold_metrics]),
            "Macro-F1": np.mean([m["macro_f1"] for m in fold_metrics]),
            "Macro-F1_std": np.std([m["macro_f1"] for m in fold_metrics]),
            "Accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "Accuracy_std": np.std([m["accuracy"] for m in fold_metrics]),
        })

    # SUPERFAMILY
    print("\n" + "=" * 70)
    print("SUPERFAMILY-LEVEL RESULTS (averaged across 5 folds)")
    print("=" * 70)

    fold_metrics, available_folds = evaluate_model_across_folds("superfamily")
    print_model_results(MODEL_NAME, "superfamily", fold_metrics, available_folds)
    if fold_metrics:
        summary_rows.append({
            "Level": "Superfamily",
            "Model": MODEL_NAME,
            "Folds": len(available_folds),
            "Macro-P": np.mean([m["macro_precision"] for m in fold_metrics]),
            "Macro-P_std": np.std([m["macro_precision"] for m in fold_metrics]),
            "Macro-R": np.mean([m["macro_recall"] for m in fold_metrics]),
            "Macro-R_std": np.std([m["macro_recall"] for m in fold_metrics]),
            "Macro-F1": np.mean([m["macro_f1"] for m in fold_metrics]),
            "Macro-F1_std": np.std([m["macro_f1"] for m in fold_metrics]),
            "Accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "Accuracy_std": np.std([m["accuracy"] for m in fold_metrics]),
        })

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).round(4)
        print(summary_df.to_string(index=False))
        output_path = os.path.join(
            BASE_DIR,
            f"{DATASET_NAME}_{OUTNAME}_thresh_{THRESHOLD}_summary_metrics.csv",
        )
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to {output_path}")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()