#!/usr/bin/env python3
"""
Summarize TERL training reports across folds for one dataset.

Expected layout (created by terl_train.py with -sr):

    <save_dir>/<dataset>/<level_dir>/<fold>/Report_fold<fold>_<...>.txt

where <level_dir> is "SF" (superfamily) or "order".

Usage:
    python summarize_terl_folds.py -s /staging/kkumari/TERL -d mntedb -m TERL \
        -o mntedb_summary.csv

Produces a CSV like:

    Level,Model,Folds,Macro-P,Macro-P_std,Macro-R,Macro-R_std,Macro-F1,Macro-F1_std,Accuracy,Accuracy_std
    Order,TERL,5,0.8207,0.1221,...
    Superfamily,TERL,5,0.8717,0.0833,...
"""
import argparse, glob, os, re, statistics as st, sys, csv

# Lines from the SKLEARN block we care about:
#   "Accuracy:        0.9680"
#   "macro          0.6712     0.6118     0.6352"
ACC_RE   = re.compile(r'^Accuracy:\s*([\d.]+)')
MACRO_RE = re.compile(r'^macro\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')

# Map directory name on disk -> label that goes in the CSV "Level" column
DIR_TO_LEVEL = {
    'SF':          'Superfamily',
    'sf':          'Superfamily',
    'superfamily': 'Superfamily',
    'order':       'Order',
    'Order':       'Order',
}

def parse_report(path):
    """Return (accuracy, macroP, macroR, macroF1) parsed from one report file.
    We restrict parsing to AFTER the 'SKLEARN CLASSIFICATION REPORT' marker
    so we don't accidentally pick up TERL's own (different) macro numbers."""
    acc = mp = mr = mf = None
    in_sklearn = False
    with open(path) as fh:
        for line in fh:
            if 'SKLEARN CLASSIFICATION REPORT' in line:
                in_sklearn = True
                continue
            if not in_sklearn:
                continue
            m = ACC_RE.match(line)
            if m and acc is None:
                acc = float(m.group(1)); continue
            m = MACRO_RE.match(line)
            if m and mp is None:
                mp, mr, mf = map(float, m.groups()); continue
            if acc is not None and mp is not None:
                break
    return acc, mp, mr, mf

def collect_level(save_dir, dataset, level_dirs):
    """Return list of (fold:int, acc, P, R, F1) for one level (Order or SF)."""
    rows = []
    for ld in level_dirs:
        # <save_dir>/<dataset>/<level_dir>/<fold>/Report_fold*.txt
        pat = os.path.join(save_dir, dataset, ld, '*', 'Report_fold*.txt')
        for fp in sorted(glob.glob(pat)):
            fold_dir = os.path.basename(os.path.dirname(fp))
            try:
                fold = int(fold_dir)
            except ValueError:
                print(f'WARN: skipping (fold dir not int): {fp}', file=sys.stderr); continue
            acc, p, r, f = parse_report(fp)
            if None in (acc, p, r, f):
                print(f'WARN: could not parse SKLEARN block in {fp}', file=sys.stderr); continue
            rows.append((fold, acc, p, r, f, fp))
    # If multiple report files exist per fold (re-runs), keep the latest by mtime.
    by_fold = {}
    for row in rows:
        fold = row[0]
        if fold not in by_fold or os.path.getmtime(row[5]) > os.path.getmtime(by_fold[fold][5]):
            by_fold[fold] = row
    return sorted(by_fold.values())

def mean_std(xs):
    if not xs:           return (float('nan'), float('nan'))
    if len(xs) == 1:     return (xs[0], 0.0)
    return (st.mean(xs), st.pstdev(xs))   # population stddev; swap to st.stdev for sample stddev

def fmt(v, digits):
    return f'{v:.{digits}f}' if v == v else ''   # NaN-safe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s','--save-dir', required=True,
                    help='Top dir under which <dataset>/{SF,order}/<fold>/Report_*.txt live.')
    ap.add_argument('-d','--dataset',  required=True,
                    help='Dataset name (the second path component, e.g. mntedb / repetdb / repbase).')
    ap.add_argument('-m','--model',    default='TERL', help='Model label for the CSV (default: TERL).')
    ap.add_argument('-o','--out',      required=True, help='Output CSV path.')
    ap.add_argument('--digits', type=int, default=4)
    args = ap.parse_args()

    levels = [
        ('Order',       ['order', 'Order']),
        ('Superfamily', ['SF', 'sf', 'superfamily']),
    ]

    csv_rows = [['Level','Model','Folds',
                 'Macro-P','Macro-P_std',
                 'Macro-R','Macro-R_std',
                 'Macro-F1','Macro-F1_std',
                 'Accuracy','Accuracy_std']]

    print(f'\n=== {args.dataset} ===')
    for level_name, dirs in levels:
        rows = collect_level(args.save_dir, args.dataset, dirs)
        if not rows:
            print(f'  {level_name:12s} : no reports found under '
                  f'{os.path.join(args.save_dir, args.dataset, "{"+",".join(dirs)+"}")}/<fold>/Report_*.txt')
            continue

        accs = [r[1] for r in rows]
        ps   = [r[2] for r in rows]
        rs   = [r[3] for r in rows]
        fs   = [r[4] for r in rows]
        a_m,a_s = mean_std(accs)
        p_m,p_s = mean_std(ps)
        r_m,r_s = mean_std(rs)
        f_m,f_s = mean_std(fs)
        n = len(rows)

        print(f'  {level_name:12s} : N={n} folds=' + ','.join(str(r[0]) for r in rows))
        print(f'    Macro P  = {p_m:.{args.digits}f} ± {p_s:.{args.digits}f}')
        print(f'    Macro R  = {r_m:.{args.digits}f} ± {r_s:.{args.digits}f}')
        print(f'    Macro F1 = {f_m:.{args.digits}f} ± {f_s:.{args.digits}f}')
        print(f'    Accuracy = {a_m:.{args.digits}f} ± {a_s:.{args.digits}f}')

        csv_rows.append([level_name, args.model, n,
                         fmt(p_m,args.digits), fmt(p_s,args.digits),
                         fmt(r_m,args.digits), fmt(r_s,args.digits),
                         fmt(f_m,args.digits), fmt(f_s,args.digits),
                         fmt(a_m,args.digits), fmt(a_s,args.digits)])

    if len(csv_rows) == 1:
        print('No data parsed; nothing to write.')
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    print(f'\nWrote {args.out}')

if __name__ == '__main__':
    main()