#!/usr/bin/env python3
"""
Aggregate per-fold metrics produced by eval_predictions.py.

Expected layout (any depth, but fold dir name must be the integer fold):

    <root>/<fold>/<species>_metrics_<level>.txt
    e.g.  SF/1/Drosophila_melanogaster_metrics_superfamily.txt

Usage:
    python aggregate_folds.py --root SF  --level superfamily  --out SF_summary.tsv
    python aggregate_folds.py --root Order --level order       --out Order_summary.tsv
"""
import argparse, glob, os, re, sys
from collections import defaultdict
import statistics as st

# Lines we care about look like:
#   "Accuracy         : 0.2300"
#   "macro     P=0.0423  R=0.0794  F1=0.0339"
#   "micro     P=0.2300  R=0.2300  F1=0.2300"
#   "weighted  P=0.3190  R=0.2300  F1=0.2631"
ACC_RE = re.compile(r'^Accuracy\s*:\s*([\d.]+)')
AVG_RE = re.compile(r'^(macro|micro|weighted)\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)')

def parse_report(path):
    """Return dict like {'accuracy':0.23, 'macro':{'P':...,'R':...,'F1':...}, 'micro':..., 'weighted':...}"""
    out = {}
    with open(path) as f:
        for line in f:
            m = ACC_RE.match(line)
            if m:
                out['accuracy'] = float(m.group(1)); continue
            m = AVG_RE.match(line)
            if m:
                avg, p, r, fone = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
                out[avg] = {'P': p, 'R': r, 'F1': fone}
    return out

def find_files(root, level):
    """Return list of (species, fold, path)."""
    pat = os.path.join(root, '*', f'*_metrics_{level}.txt')
    files = sorted(glob.glob(pat))
    out = []
    for fp in files:
        fold_dir = os.path.basename(os.path.dirname(fp))
        try:
            fold = int(fold_dir)
        except ValueError:
            print(f'WARN: skipping (fold dir not an int): {fp}', file=sys.stderr); continue
        base = os.path.basename(fp)
        species = base.split(f'_metrics_{level}.txt')[0]
        out.append((species, fold, fp))
    return out

def mean_std(xs):
    if not xs: return (float('nan'), float('nan'))
    if len(xs) == 1: return (xs[0], 0.0)
    return (st.mean(xs), st.pstdev(xs))   # population stdev (n in denominator)
    # use st.stdev(xs) for sample stdev (n-1) if you prefer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Top dir containing fold subdirs (e.g. SF or Order).')
    ap.add_argument('--level', required=True, choices=['order','superfamily'])
    ap.add_argument('--out', default=None, help='Optional TSV output path.')
    ap.add_argument('--digits', type=int, default=4)
    args = ap.parse_args()

    triples = find_files(args.root, args.level)
    if not triples:
        print(f'No files found under {args.root}/<fold>/*_metrics_{args.level}.txt'); sys.exit(1)

    # species -> list of parsed dicts
    bucket = defaultdict(list)
    for species, fold, fp in triples:
        try:
            bucket[species].append((fold, parse_report(fp)))
        except Exception as e:
            print(f'WARN: failed to parse {fp}: {e}', file=sys.stderr)

    # Print + collect rows
    cols = ['species','n_folds',
            'acc_mean','acc_std',
            'macro_P_mean','macro_P_std','macro_R_mean','macro_R_std','macro_F1_mean','macro_F1_std',
            'micro_F1_mean','micro_F1_std',
            'weighted_F1_mean','weighted_F1_std']
    rows = [cols]

    fmt = lambda v: f'{v:.{args.digits}f}'
    print('\n=== Aggregate across folds ({} samples per species) ==='.format(args.level))
    header = '{:30s} {:>3s} {:>12s} {:>16s} {:>16s} {:>16s} {:>16s}'.format(
        'species','N','acc','macro P','macro R','macro F1','weighted F1')
    print(header)
    print('-' * len(header))

    for species in sorted(bucket):
        items = sorted(bucket[species])  # by fold
        accs = [d.get('accuracy', float('nan')) for _, d in items]
        mps  = [d.get('macro',{}).get('P', float('nan')) for _, d in items]
        mrs  = [d.get('macro',{}).get('R', float('nan')) for _, d in items]
        mfs  = [d.get('macro',{}).get('F1', float('nan')) for _, d in items]
        mif  = [d.get('micro',{}).get('F1', float('nan')) for _, d in items]
        wf   = [d.get('weighted',{}).get('F1', float('nan')) for _, d in items]

        a_m, a_s = mean_std(accs)
        p_m, p_s = mean_std(mps)
        r_m, r_s = mean_std(mrs)
        f_m, f_s = mean_std(mfs)
        mi_m, mi_s = mean_std(mif)
        w_m, w_s = mean_std(wf)

        print('{:30s} {:3d} {:>7s}±{:<5s} {:>7s}±{:<5s} {:>7s}±{:<5s} {:>7s}±{:<5s} {:>7s}±{:<5s}'.format(
            species, len(items),
            fmt(a_m), fmt(a_s),
            fmt(p_m), fmt(p_s),
            fmt(r_m), fmt(r_s),
            fmt(f_m), fmt(f_s),
            fmt(w_m), fmt(w_s),
        ))

        rows.append([species, str(len(items)),
                     fmt(a_m), fmt(a_s),
                     fmt(p_m), fmt(p_s),
                     fmt(r_m), fmt(r_s),
                     fmt(f_m), fmt(f_s),
                     fmt(mi_m), fmt(mi_s),
                     fmt(w_m), fmt(w_s)])

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            for r in rows: f.write('\t'.join(r) + '\n')
        print(f'\nWrote: {args.out}')

if __name__ == '__main__':
    main()