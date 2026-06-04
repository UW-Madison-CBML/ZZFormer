#!/usr/bin/env python3
"""
Aggregate per-fold evaluation reports into one CSV per dataset (model name).

Expected layout:
    <metrics_root>/Order/<fold>/<dataset>_metrics_order.txt
    <metrics_root>/SF/<fold>/<dataset>_metrics_superfamily.txt

Output (one file per dataset, written to --out-dir):
    <dataset>.csv
        Level,Model,Folds,Macro-P,Macro-P_std,Macro-R,Macro-R_std,
        Macro-F1,Macro-F1_std,Accuracy,Accuracy_std
        Order,<dataset>,5,...
        Superfamily,<dataset>,5,...
"""

import argparse, csv, glob, os, re, statistics, sys
from collections import defaultdict

KEYS = ["Macro-Precision", "Macro-Recall", "Macro-F1", "Accuracy"]

# --- patterns matching the evaluate_predictions.py report format ---
_RE_ACC = re.compile(r"^\s*Accuracy\s*[:=]\s*([0-9.eE+-]+)")
_RE_MACRO = re.compile(
    r"^\s*macro\s+P\s*=\s*([0-9.eE+-]+)\s+"
    r"R\s*=\s*([0-9.eE+-]+)\s+"
    r"F1?\s*=\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
_RE_OLD = re.compile(
    r"^\s*(Macro-Precision|Macro-Recall|Macro-F1|Accuracy)\s*[:\t ]+([0-9.eE+-]+)"
)
# pull "<dataset>" out of "<dataset>_metrics_order.txt" / "<dataset>_metrics_superfamily.txt"
_RE_FNAME = re.compile(r"^(?P<dataset>.+)_metrics_(?:order|superfamily)\.txt$")


def parse_metrics_file(path):
    vals = {}
    with open(path) as f:
        for line in f:
            m = _RE_ACC.match(line)
            if m:
                vals["Accuracy"] = float(m.group(1))
                continue
            m = _RE_MACRO.match(line)
            if m:
                vals["Macro-Precision"] = float(m.group(1))
                vals["Macro-Recall"] = float(m.group(2))
                vals["Macro-F1"] = float(m.group(3))
                continue
            m = _RE_OLD.match(line)
            if m:
                vals[m.group(1)] = float(m.group(2))
    return vals


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def collect(metrics_root, level_dir, folds):
    """
    Returns: dict[dataset][fold] = {metric_key: value}
    Also returns a flat list of (path, dataset, fold) actually found,
    for diagnostics.
    """
    by_dataset = defaultdict(dict)
    found = []
    for fold in folds:
        d = os.path.join(metrics_root, level_dir, str(fold))
        if not os.path.isdir(d):
            print(f"[warn] missing directory: {d}", file=sys.stderr)
            continue
        for fp in sorted(glob.glob(os.path.join(d, "*_metrics_*.txt"))):
            fname = os.path.basename(fp)
            m = _RE_FNAME.match(fname)
            if not m:
                print(f"[warn] skipping unrecognized filename: {fp}", file=sys.stderr)
                continue
            dataset = m.group("dataset")
            vals = parse_metrics_file(fp)
            if not vals:
                print(f"[warn] no metrics parsed from: {fp}", file=sys.stderr)
            by_dataset[dataset][fold] = vals
            found.append((fp, dataset, fold))
    return by_dataset, found


def aggregate_level(by_dataset, dataset, folds):
    """Return per-key list of fold values for one dataset."""
    collected = {k: [] for k in KEYS}
    n = 0
    for fold in folds:
        vals = by_dataset.get(dataset, {}).get(fold)
        if not vals:
            continue
        n += 1
        for k in KEYS:
            if k in vals:
                collected[k].append(vals[k])
    return collected, n


HEADER = [
    "Level", "Model", "Folds",
    "Macro-P", "Macro-P_std",
    "Macro-R", "Macro-R_std",
    "Macro-F1", "Macro-F1_std",
    "Accuracy", "Accuracy_std",
]


def fmt(v, digits=4):
    return "nan" if v != v else f"{v:.{digits}f}"


def make_row(level_name, dataset, collected, n_folds, digits=4):
    p_m, p_s = mean_std(collected["Macro-Precision"])
    r_m, r_s = mean_std(collected["Macro-Recall"])
    f_m, f_s = mean_std(collected["Macro-F1"])
    a_m, a_s = mean_std(collected["Accuracy"])
    return [
        level_name, dataset, str(n_folds),
        fmt(p_m, digits), fmt(p_s, digits),
        fmt(r_m, digits), fmt(r_s, digits),
        fmt(f_m, digits), fmt(f_s, digits),
        fmt(a_m, digits), fmt(a_s, digits),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", required=True,
                    help="root that contains Order/<fold>/... and SF/<fold>/...")
    ap.add_argument("--folds", required=True, help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--out-dir", required=True, help="directory to write <dataset>.csv files")
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    order_by_ds, order_found = collect(args.metrics_root, "Order", folds)
    sf_by_ds,    sf_found    = collect(args.metrics_root, "SF",    folds)

    print(f"Order  reports found: {len(order_found)}", file=sys.stderr)
    print(f"SF     reports found: {len(sf_found)}",    file=sys.stderr)
    if not order_found and not sf_found:
        sys.exit(
            f"[error] no metric files found under {args.metrics_root!r} "
            f"for folds {folds}. Check --metrics-root and your cwd."
        )

    datasets = sorted(set(order_by_ds) | set(sf_by_ds))
    print(f"Datasets discovered  : {datasets}", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)

    for ds in datasets:
        rows = [HEADER]

        # Order row
        coll_o, n_o = aggregate_level(order_by_ds, ds, folds)
        rows.append(make_row("Order", ds, coll_o, n_o, args.digits))

        # Superfamily row
        coll_s, n_s = aggregate_level(sf_by_ds, ds, folds)
        rows.append(make_row("Superfamily", ds, coll_s, n_s, args.digits))

        out_path = os.path.join(args.out_dir, f"{ds}.csv")
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerows(rows)
        print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()