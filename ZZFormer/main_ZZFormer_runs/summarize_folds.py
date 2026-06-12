#!/usr/bin/env python3
"""
Aggregate per-fold YAML validation metrics into a mean ± std CSV.

Expects files named like:  fold{N}_*_val.yaml   in --metrics_dir
Each YAML has keys: order_acc, order_p, order_r, order_f1,
                    sf_acc,    sf_p,    sf_r,    sf_f1, val_loss

Usage:
    python summarize_longformer_folds.py \
        --metrics_dir /staging/kkumari/longformer_runs/repbase \
        --model_name  Longformer \
        --out         repbase_summary.csv
"""

import argparse
import glob
import os
import csv
import re
import yaml
import numpy as np


FOLD_RE = re.compile(r"fold(\d+)_.*_val\.yaml$")


def load_fold_yamls(metrics_dir):
    """Return {fold_id: metrics_dict}, sorted by fold."""
    out = {}
    for path in glob.glob(os.path.join(metrics_dir, "fold*_val.yaml")):
        m = FOLD_RE.search(os.path.basename(path))
        if not m:
            continue
        fold = int(m.group(1))
        with open(path) as f:
            out[fold] = yaml.safe_load(f)
    return dict(sorted(out.items()))


def summarize(metrics_by_fold, model_name, out_path):
    folds = list(metrics_by_fold.keys())
    n = len(folds)
    if n == 0:
        raise RuntimeError("No fold YAMLs found.")

    print(f"Found {n} folds: {folds}")

    def stack(key):
        return np.array([metrics_by_fold[f][key] for f in folds], dtype=float)

    levels = [
        # (display name, prec_key,  rec_key,  f1_key,   acc_key)
        ("Order",        "order_p", "order_r", "order_f1", "order_acc"),
        ("Superfamily",  "sf_p",    "sf_r",    "sf_f1",    "sf_acc"),
    ]

    header = [
        "Level", "Model", "Folds",
        "Macro-P", "Macro-P_std",
        "Macro-R", "Macro-R_std",
        "Macro-F1", "Macro-F1_std",
        "Accuracy", "Accuracy_std",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        print(",".join(header))

        for display, pk, rk, f1k, ak in levels:
            p, r, f1, acc = stack(pk), stack(rk), stack(f1k), stack(ak)
            row = [
                display, model_name, n,
                f"{p.mean():.4f}",   f"{p.std(ddof=0):.4f}",
                f"{r.mean():.4f}",   f"{r.std(ddof=0):.4f}",
                f"{f1.mean():.4f}",  f"{f1.std(ddof=0):.4f}",
                f"{acc.mean():.4f}", f"{acc.std(ddof=0):.4f}",
            ]
            w.writerow(row)
            print(",".join(map(str, row)))

    print(f"\n✅ Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True,
                    help="Directory containing fold*_val.yaml files")
    ap.add_argument("--model_name",  required=True,
                    help="e.g. Longformer, TERL, ZZFormer")
    ap.add_argument("--out",         default="summary.csv")
    args = ap.parse_args()

    metrics = load_fold_yamls(args.metrics_dir)
    summarize(metrics, args.model_name, args.out)

'''
python summarize_folds.py \
    --metrics_dir /staging/kkumari/terrsystem/longformer_runs_wprtrn/repetdb \
    --model_name  Longformer_prtrn \
    --out         /staging/kkumari/terrsystem/longformer_runs_wprtrn/summary_longformer_wprtrn/cross5val/thresh00/5crossVAL_repetdb_nothresh_thresh_0.0_summary_metrics.csv
    
'''