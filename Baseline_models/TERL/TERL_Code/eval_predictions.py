#!/usr/bin/env python3
"""
Evaluate TERL predicted FASTAs against a ground-truth pickle.

Usage:
    python eval_predictions.py -p preds_dir -g ground_truth.pkl --level order
    python eval_predictions.py -p preds_dir -g ground_truth.pkl --level superfamily \
        --pred-alias 'PIF-Harbinger=PIF,Dirs=DIRS,Helitron=RC,Mariner=TcMar'
"""
import argparse, glob, hashlib, inspect, os, pickle, sys
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from alias import load_alias_map, apply_alias

# Inversion of TERL's `classification` dict (terl_test.py): printed leaf -> canonical training class
PRINTED_TO_CANONICAL = {
    'Copia':'Copia','Gypsy':'Gypsy','Bel-Pao':'Bel-Pao','Retrovirus':'Retrovirus','ERV':'ERV',
    'Dirs':'Dirs','Ngaro':'Ngaro','VIPER':'VIPER','Penelope':'Penelope',
    'R2':'R2','RTE':'RTE','Jockey':'Jockey','L1':'L1','I':'I',
    'tRNA':'tRNA','7SL':'7SL','5S':'5S',
    'Tc1-Mariner':'TcMar','hAT':'hAT','Mutator':'Mutator','Merlin':'Merlin',
    'Transib':'Transib','P':'P','PiggyBac':'PiggyBac','PIF-Harbinger':'PIF-Harbinger',
    'CACTA':'CACTA','Crypton':'Crypton','Helitron':'Helitron','Maverick':'Maverick',
    'LTR':'LTR','DIRS':'DIRS','PLE':'PLE','LINE':'LINE','SINE':'SINE','TIR':'TIR',
    'Subclass 1':'Subclass 1','Subclass 2':'Subclass 2',
    'Class I':'Class I','Class II':'Class II',
    'TRIM':'TRIM','LARD':'LARD','MITE':'MITE','SNAC':'SNAC',
    'NonTE':'Random',
}

# Predicted leaves we never want to score (too coarse: not at order/superfamily level)
# ALWAYS_SKIP = {'Class I','Class II','Subclass 1','Subclass 2'}


def seq_key(seq):
    return hashlib.md5(seq.upper().encode()).hexdigest()

def iter_fastas(path):
    if os.path.isdir(path):
        for ext in ('*.fa','*.fasta','*.fna'):
            yield from sorted(glob.glob(os.path.join(path,'**',ext), recursive=True))
    else:
        yield path

def extract_pred_label(header):
    h = header.lstrip('>').rstrip('\n')
    parts = h.split('\t')
    if len(parts) < 2:
        only = h.strip()
        # if not only or only in ALWAYS_SKIP: return None
        return PRINTED_TO_CANONICAL.get(only, only)
    leaf = parts[-2].strip()
    # if leaf in ALWAYS_SKIP: return None
    return PRINTED_TO_CANONICAL.get(leaf, leaf)

def parse_pred_fasta(path):
    hdr, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if hdr is not None: yield hdr, ''.join(chunks)
                hdr, chunks = line, []
            else:
                chunks.append(line.strip())
        if hdr is not None: yield hdr, ''.join(chunks)

def sklearn_kw(fn):
    return {'zero_division': 0} if 'zero_division' in inspect.signature(fn).parameters else {}





# ---- new helper, put near the top ----
def load_label_set(spec):
    """Load a set of labels from a file (one per line) or a comma-separated string.
    Returns None when spec is None (meaning: do not filter)."""
    if not spec:
        return None
    if os.path.isfile(spec):
        with open(spec) as fh:
            items = [line.split('#', 1)[0].strip() for line in fh]
            return {x for x in items if x}
    return {tok.strip() for tok in spec.split(',') if tok.strip()}








def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--preds', required=True)
    ap.add_argument('-g','--gt-pickle', required=True)
    ap.add_argument('--level', choices=['order','superfamily'], required=True)
    ap.add_argument('-o','--out', default=None)
    ap.add_argument('--digits', type=int, default=4)
    ap.add_argument('--pred-alias', default=None,
                    help='Alias map (TSV file or "a=b,c=d") applied to predicted leaves.')
    ap.add_argument('--gt-alias', default=None,
                    help='Alias map applied to GT labels at eval time.')

    ap.add_argument('--model-classes', default=None,
                help='Restrict evaluation to GT samples whose (aliased) true '
                     'label is in this set. Pass a file with one class per line '
                     'or a comma-separated string. Use the model class names '
                     '(post-pred-alias) for the chosen --level. '
                     'Sequences whose GT label is not in this set are skipped.')




    args = ap.parse_args()

    pred_alias = load_alias_map(args.pred_alias)
    gt_alias   = load_alias_map(args.gt_alias)
    model_classes = load_label_set(args.model_classes)   # None => no filter






    with open(args.gt_pickle, 'rb') as f:
        gt = pickle.load(f)


    y_true, y_pred = [], []
    n_records = n_matched = n_no_gt = n_unlabeled = n_skipped_pred = 0
    n_skipped_gt_oov = 0                       # NEW
    skipped_pred_examples = Counter()
    skipped_gt_examples   = Counter()          # NEW
    unmatched_examples = []

    for fp in iter_fastas(args.preds):
        for raw_header, seq in parse_pred_fasta(fp):
            n_records += 1
            if not seq:
                continue

            canonical_pred = extract_pred_label(raw_header)
            if canonical_pred is None:
                n_skipped_pred += 1
                skipped_pred_examples[raw_header.strip()] += 1
                continue
            canonical_pred = apply_alias(canonical_pred, pred_alias)

            entry = gt.get(seq_key(seq))
            if entry is None:
                n_no_gt += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append((os.path.basename(fp), seq[:40]))
                continue

            true_label = entry.get(args.level)
            if true_label is None:
                n_unlabeled += 1
                continue
            true_label = apply_alias(true_label, gt_alias)

            # NEW: drop samples whose GT label the model cannot possibly predict
            if model_classes is not None and true_label not in model_classes:
                n_skipped_gt_oov += 1
                skipped_gt_examples[true_label] += 1
                continue

            y_true.append(true_label)
            y_pred.append(canonical_pred)
            n_matched += 1

    print(f'Predicted records read           : {n_records}')
    print(f'Matched & evaluated              : {n_matched}')
    print(f'Pred too coarse (skipped)        : {n_skipped_pred}')
    print(f'GT label out of model vocab      : {n_skipped_gt_oov}')
    print(f'Pred seqs not found in GT        : {n_no_gt}')
    print(f'GT entries missing {args.level} : {n_unlabeled}')
    if skipped_pred_examples:
        print('Top skipped predicted headers    :',
            skipped_pred_examples.most_common(10))
    if skipped_gt_examples:
        print('Top dropped GT labels (OOV)      :',
            skipped_gt_examples.most_common(10))

    # y_true, y_pred = [], []
    # n_records = n_matched = n_no_gt = n_unlabeled = n_skipped_pred = 0
    # skipped_pred_examples = Counter()
    # unmatched_examples = []

    # for fp in iter_fastas(args.preds):
    #     for raw_header, seq in parse_pred_fasta(fp):
    #         n_records += 1
    #         if not seq:
    #             continue

    #         canonical_pred = extract_pred_label(raw_header)
    #         if canonical_pred is None:
    #             n_skipped_pred += 1
    #             skipped_pred_examples[raw_header.strip()] += 1
    #             continue
    #         canonical_pred = apply_alias(canonical_pred, pred_alias)

    #         entry = gt.get(seq_key(seq))
    #         if entry is None:
    #             n_no_gt += 1
    #             if len(unmatched_examples) < 5:
    #                 unmatched_examples.append((os.path.basename(fp), seq[:40]))
    #             continue

    #         true_label = entry.get(args.level)
    #         if true_label is None:
    #             n_unlabeled += 1
    #             continue
    #         true_label = apply_alias(true_label, gt_alias)

    #         y_true.append(true_label)
    #         y_pred.append(canonical_pred)
    #         n_matched += 1

    print(f'Predicted records read           : {n_records}')
    print(f'Matched & evaluated              : {n_matched}')
    print(f'Pred too coarse (skipped)        : {n_skipped_pred}')
    print(f'Pred seqs not found in GT        : {n_no_gt}')
    print(f'GT entries missing {args.level} : {n_unlabeled}')
    if skipped_pred_examples:
        print('Top skipped predicted headers    :',
              skipped_pred_examples.most_common(10))

    if not y_true:
        print('Nothing to evaluate.')
        if unmatched_examples:
            print('First few unmatched (file, seq[:40]):')
            for ex in unmatched_examples: print(' ', ex)
        sys.exit(1)

    # ---------------------------------------------------------------------
    # CORRECT "macro over GT-present classes + penalize predicted-only labels"
    # ---------------------------------------------------------------------
    # In this y_true/y_pred setting, the correct policy is simply:
    #   labels_for_macro = set(y_true)
    # because any label not in y_true has support=0 and should not be averaged.
    # Predicted-only labels will still be penalized naturally:
    # - they count as wrong in accuracy
    # - they contribute FP for that predicted label and FN for the true label
    labels = sorted(set(y_true))

    pkw, crkw = sklearn_kw(precision_score), sklearn_kw(classification_report)

    lines = []
    lines.append('=' * 79)
    lines.append(f'Evaluation level : {args.level}')
    lines.append(f'N samples        : {len(y_true)}')
    lines.append('=' * 79)

    # Accuracy: penalizes any predicted label not equal to true label (including "predicted-only")
    lines.append(f'Accuracy         : {accuracy_score(y_true, y_pred):.{args.digits}f}')

    for avg in ('macro', 'micro', 'weighted'):
        p = precision_score(y_true, y_pred, labels=labels, average=avg, **pkw)
        r = recall_score(y_true, y_pred, labels=labels, average=avg, **pkw)
        f = f1_score(y_true, y_pred, labels=labels, average=avg, **pkw)
        lines.append(f'{avg:8s}  P={p:.{args.digits}f}  R={r:.{args.digits}f}  F1={f:.{args.digits}f}')

    # Optional but useful: how often model predicted labels not in GT label set
    pred_only = [p for p in y_pred if p not in set(labels)]
    if pred_only:
        lines.append('')
        lines.append(f'Predicted-only labels (not in GT) count: {len(pred_only)}')
        lines.append('Top predicted-only labels               : ' +
                     str(Counter(pred_only).most_common(10)))

    lines.append('')
    lines.append('Per-class report (GT-present labels only):')
    lines.append(classification_report(
        y_true, y_pred, labels=labels, target_names=labels,
        digits=args.digits, **crkw))

    lines.append('Confusion matrix (rows=true, cols=pred); class order (GT-present only):')
    lines.append(', '.join(labels))
    lines.append(str(confusion_matrix(y_true, y_pred, labels=labels)))

    lines.append('')
    lines.append('True label distribution : ' + str(Counter(y_true).most_common()))
    lines.append('Pred label distribution : ' + str(Counter(y_pred).most_common()))

    report = '\n'.join(lines) + '\n'
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f: f.write(report)
        print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()