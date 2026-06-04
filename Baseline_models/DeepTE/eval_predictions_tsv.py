#!/usr/bin/env python3
"""
Evaluate TSV predictions against a ground-truth pickle (md5(seq.upper()) -> labels).

- Predictions TSV is produced by your inference script:
    id    pred_label    pred_index    confidence
- Ground-truth pickle is produced by build_groundtruth_pickle.py
  using headers like: >seqid#ORDER/SUPERFAMILY
- Trained label map JSON is produced at training time:
    {"Copia": 0, "Gypsy": 1, ...}

Policy:
- Closed-set evaluation: any sample whose GT label (after alias) is NOT in
  the model's trained label set is DROPPED. We are not doing out-of-set
  classification, so scoring the model on labels it could never output is
  meaningless.
- accuracy over remaining samples
- macro P/R/F1 averaged over the intersection of {trained labels} and
  {labels present in y_true after filtering}
- if --level superfamily and predicted label has no superfamily -> SKIP that sample
"""

import argparse, csv, hashlib, inspect, json, os, pickle, sys
from collections import Counter

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from alias import load_alias_map, apply_alias


def seq_key(seq: str) -> str:
    return hashlib.md5(seq.upper().encode()).hexdigest()


def read_fasta(path):
    sid = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(chunks)
                sid = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if sid is not None:
            yield sid, "".join(chunks)


def read_pred_tsv(path):
    preds = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit("Pred TSV has no header row.")
        need = {"id", "pred_label"}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"Pred TSV must contain {sorted(need)}; got {r.fieldnames}")
        for row in r:
            preds[row["id"]] = row["pred_label"]
    return preds


def sklearn_kw(fn):
    return {'zero_division': 0} if 'zero_division' in inspect.signature(fn).parameters else {}


# def pred_to_level(pred_label: str, level: str):
#     if pred_label is None:
#         return None
#     pred_label = pred_label.strip()
#     if not pred_label:
#         return None

#     if level == "order":
#         return pred_label.split("_", 1)[0]

#     if level == "superfamily":
#         if "_" not in pred_label:
#             return None
#         sf = pred_label.split("_", 1)[1]
#         return sf if sf != "" else None

#     raise ValueError(level)




def pred_to_level(pred_label: str, level: str, trained_labels: set):
    """
    Normalize a raw predicted label to the requested taxonomic level.

    Cases:
      1. Model was trained directly at this level (label-map contains the
         predicted token as-is) -> return pred_label unchanged.
      2. Model was trained on compound 'ORDER_SUPERFAMILY' labels:
           - level=order       -> first token
           - level=superfamily -> rest after first '_'
           - if no '_' and not in trained_labels -> None
    """
    if pred_label is None:
        return None
    pred_label = pred_label.strip()
    if not pred_label:
        return None

    # Case 1: model is already at the requested level.
    if pred_label in trained_labels:
        return pred_label

    # Case 2: compound label, split it.
    if level == "order":
        # return pred_label.split("_", 1)[0]
        if "_" in pred_label:
            order= pred_label.split("_", 1)[0]
        order = pred_label.strip()
        return order if order != "" else None

    if level == "superfamily":
        if "_" in pred_label:
            sf= pred_label.split("_", 1)[1]
        sf = pred_label.strip()
        return sf if sf != "" else None

    raise ValueError(level)








# def load_trained_labels(path, pred_alias):
#     """
#     Load the {label: id} JSON saved at training time and return a set of
#     alias-normalized label strings. We normalize with the SAME alias map
#     used for predictions, because trained-label strings come from the same
#     label-space the model emits.
#     """
#     with open(path) as f:
#         mp = json.load(f)
#     if not isinstance(mp, dict):
#         raise SystemExit(f"--label-map must be a JSON dict {{label: id}}; got {type(mp).__name__}")
#     return {apply_alias(k, pred_alias) for k in mp.keys()}


def load_trained_labels(path, pred_alias):
    with open(path) as f:
        mp = json.load(f)
    if not isinstance(mp, dict):
        raise SystemExit(f"--label-map must be a JSON dict {{label: id}}; got {type(mp).__name__}")
    raw = set(mp.keys())
    normalized = {apply_alias(k, pred_alias) for k in raw}
    return raw, normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-tsv", required=True, help="predictions.tsv from inference")
    ap.add_argument("--fasta", required=True, help="FASTA used to generate the predictions.tsv")
    ap.add_argument("-g", "--gt-pickle", required=True, help="ground truth pickle")
    ap.add_argument("--label-map", required=True,
                    help="JSON label map saved at training time ({label: id}). "
                         "Used to restrict evaluation to the model's trained label set.")
    ap.add_argument("--level", choices=["order", "superfamily"], required=True)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--pred-alias", default=None)
    ap.add_argument("--gt-alias", default=None)
    args = ap.parse_args()

    pred_alias = load_alias_map(args.pred_alias)
    gt_alias = load_alias_map(args.gt_alias)

    trained_labels_raw, trained_labels = load_trained_labels(args.label_map, pred_alias)
    
    if not trained_labels:
        raise SystemExit("Trained label set is empty.")
    
    print("Trained labels:", sorted(trained_labels))
    

    preds_by_id = read_pred_tsv(args.pred_tsv)
    with open(args.gt_pickle, "rb") as f:
        gt = pickle.load(f)

    print("contains underscore?  :", any('_' in t for t in trained_labels_raw))
    print("sample raw preds      :", list(preds_by_id.values())[:10])
    print("trained_labels_raw    :", sorted(trained_labels_raw))

    y_true, y_pred = [], []
    n_fasta = n_no_pred = n_no_gt = n_gt_missing_level = 0
    n_skip_pred_missing_level = 0
    n_dropped_gt_oos = 0          # NEW: GT label not in trained labels
    dropped_oos_counter = Counter()  # NEW: which OOS GT labels were dropped
    unmatched_examples = []

    for sid, seq in read_fasta(args.fasta):
        n_fasta += 1

        raw_pred = preds_by_id.get(sid)
        if raw_pred is None:
            n_no_pred += 1
            continue

        pred_level = pred_to_level(raw_pred, args.level,trained_labels_raw)
        if args.level == "superfamily" and pred_level is None:
            n_skip_pred_missing_level += 1
            continue
        pred_level = apply_alias(pred_level, pred_alias)

        entry = gt.get(seq_key(seq))
        if entry is None:
            n_no_gt += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append((sid, seq[:40]))
            continue

        true_level = entry.get(args.level)
        if true_level is None:
            n_gt_missing_level += 1
            continue
        true_level = apply_alias(true_level, gt_alias)

        # ---- NEW: closed-set filter ----
        if true_level not in trained_labels:
            n_dropped_gt_oos += 1
            dropped_oos_counter[true_level] += 1
            continue

        y_true.append(true_level)
        y_pred.append(pred_level)


    print("y_pred",sorted(set(y_pred)))
    print("y_true", sorted(set(y_true)))
    # ---- reporting counts ----
    print(f"Trained label set size                 : {len(trained_labels)}")
    print(f"FASTA records read                     : {n_fasta}")
    print(f"Missing prediction rows                : {n_no_pred}")
    print(f"Seqs not found in GT (md5 mismatch)    : {n_no_gt}")
    print(f"GT entries missing {args.level}        : {n_gt_missing_level}")
    if args.level == "superfamily":
        print(f"Skipped (no predicted superfamily)     : {n_skip_pred_missing_level}")
    print(f"Dropped (GT label not in trained set)  : {n_dropped_gt_oos}")
    if dropped_oos_counter:
        print(f"  -> top dropped OOS GT labels         : {dropped_oos_counter.most_common(10)}")
    print(f"Matched & evaluated                    : {len(y_true)}")

    if unmatched_examples:
        print("First few GT-unmatched examples (sid, seq[:40]):")
        for ex in unmatched_examples:
            print(" ", ex)

    if not y_true:
        print("Nothing to evaluate.")
        sys.exit(1)

    # Macro averaging is restricted to (trained labels) ∩ (labels present in y_true).
    # - We don't include trained labels with zero support in this test set
    #   (they'd just contribute 0s and depress macro scores meaninglessly).
    # - We don't include GT labels outside the trained set (filtered above).
    labels = sorted(set(y_true) & trained_labels)

    pkw, crkw = sklearn_kw(precision_score), sklearn_kw(classification_report)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=labels, average="macro", **pkw)
    macro_r = recall_score(y_true, y_pred, labels=labels, average="macro", **pkw)
    macro_f = f1_score(y_true, y_pred, labels=labels, average="macro", **pkw)

    lines = []
    lines.append("=" * 79)
    lines.append(f"Evaluation level : {args.level}")
    lines.append(f"N samples        : {len(y_true)}")
    lines.append(f"Trained labels   : {len(trained_labels)}  "
                 f"(evaluated against {len(labels)} present in y_true)")
    lines.append("=" * 79)
    # lines.append(f"Accuracy         : {acc:.{args.digits}f}")
    # lines.append(f"macro     P={macro_p:.{args.digits}f}  "
    #              f"R={macro_r:.{args.digits}f}  F1={macro_f:.{args.digits}f}")

    lines.append(f"Accuracy         : {acc:.{args.digits}f}")
    lines.append(f"Macro-Precision\t{macro_p:.{args.digits}f}")
    lines.append(f"Macro-Recall\t{macro_r:.{args.digits}f}")
    lines.append(f"Macro-F1\t{macro_f:.{args.digits}f}")
    # (keep the human-friendly summary line too if you want)
    lines.append(f"macro     P={macro_p:.{args.digits}f}  R={macro_r:.{args.digits}f}  F1={macro_f:.{args.digits}f}")

    # Diagnostic: predicted labels that aren't in the trained set
    # (shouldn't happen if --pred-alias is consistent, but worth flagging)
    pred_oos = [p for p in y_pred if p not in trained_labels]
    if pred_oos:
        lines.append("")
        lines.append(f"WARNING: model produced labels not in trained set: "
                     f"{Counter(pred_oos).most_common(10)}")

    lines.append("")
    lines.append("Per-class report (trained ∩ present-in-GT labels):")
    lines.append(classification_report(
        y_true, y_pred,
        labels=labels, target_names=labels,
        digits=args.digits, **crkw
    ))

    lines.append("Confusion matrix (rows=true, cols=pred); class order:")
    lines.append(", ".join(labels))
    lines.append(str(confusion_matrix(y_true, y_pred, labels=labels)))

    lines.append("")
    lines.append("True label distribution : " + str(Counter(y_true).most_common()))
    lines.append("Pred label distribution : " + str(Counter(y_pred).most_common()))

    if n_dropped_gt_oos:
        lines.append("")
        lines.append(f"Dropped (closed-set filter) : {n_dropped_gt_oos}")
        lines.append(f"  Top OOS GT labels         : {dropped_oos_counter.most_common(20)}")

    report = "\n".join(lines) + "\n"
    print(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()