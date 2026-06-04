#!/usr/bin/env python3
"""
Summarize DEEPTE per-fold reports into the same CSV format as the TERL summarizer.

Expected layout:

    <save_dir>/<dataset>/<fold>/<level_dir>/<report_file>

where:
    <fold>      is an integer-named directory (0..N-1)
    <level_dir> is one of: SF / sf / superfamily   OR   order / Order
    <report_file> contains lines like:
        Macro-Precision   0.7402
        Macro-Recall      0.6575
        Macro-F1          0.6905
        Accuracy          0.9787

Usage:
    python summarize_deepte_folds.py -s /staging/kkumari/DEEPTE -d mntedb \
        -m DEEPTE -o mntedb_deepte_summary.csv
"""
import argparse, glob, os, re, statistics as st, sys, csv

# Lines we care about (whitespace OR ':' between key and value):
PAT = {
    'P':   re.compile(r'^Macro[-_ ]?Precision\s*[:\s]\s*([\d.]+)', re.I),
    'R':   re.compile(r'^Macro[-_ ]?Recall\s*[:\s]\s*([\d.]+)',    re.I),
    'F1':  re.compile(r'^Macro[-_ ]?F1\s*[:\s]\s*([\d.]+)',         re.I),
    'ACC': re.compile(r'^Accuracy\s*[:\s]\s*([\d.]+)',              re.I),
}

LEVEL_DIRS = {
    'Order':       ['order', 'Order'],
    'Superfamily': ['SF', 'sf', 'superfamily'],
}

def parse_report(path):
    """Return (acc, P, R, F1) or Nones if missing."""
    found = {}
    with open(path) as fh:
        for line in fh:
            for k, pat in PAT.items():
                if k in found: continue
                m = pat.match(line.strip())
                if m:
                    found[k] = float(m.group(1))
            if len(found) == 4:
                break
    return found.get('ACC'), found.get('P'), found.get('R'), found.get('F1')


def collect_level(save_dir, dataset, level_dirs):
    """Walk <save_dir>/<dataset>/<fold>/<level_dir>/* and parse the newest report per fold."""
    rows_by_fold = {}
    fold_root = os.path.join(save_dir, dataset)
    if not os.path.isdir(fold_root):
        return []

    for fold_dir in sorted(os.listdir(fold_root)):
        try:
            fold = int(fold_dir)
        except ValueError:
            continue
        for ld in level_dirs:
            level_path = os.path.join(fold_root, fold_dir, ld)
            if not os.path.isdir(level_path): continue

            # Accept .txt / .tsv / .log report-ish files. Take the newest.
            candidates = []
            for ext in ('*.txt', '*.tsv', '*.log', '*.report'):
                candidates += glob.glob(os.path.join(level_path, '**', ext), recursive=True)
            if not candidates: continue
            candidates.sort(key=os.path.getmtime, reverse=True)

            for fp in candidates:
                acc, p, r, f = parse_report(fp)
                if None in (acc, p, r, f):
                    continue                       # try next file in this level dir
                rows_by_fold[fold] = (fold, acc, p, r, f, fp)
                break                              # done for this fold/level
    return [rows_by_fold[k] for k in sorted(rows_by_fold)]


def mean_std(xs):
    if not xs:        return (float('nan'), float('nan'))
    if len(xs) == 1:  return (xs[0], 0.0)
    return (st.mean(xs), st.pstdev(xs))   # population std; swap to st.stdev for sample std


def fmt(v, d):
    return f'{v:.{d}f}' if v == v else ''   # NaN-safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s','--save-dir', required=True,
                    help='Top dir under which <dataset>/<fold>/{SF,order}/<report>.txt live.')
    ap.add_argument('-d','--dataset',  required=True,
                    help='Dataset name (e.g. mntedb / repetdb / repbase).')
    ap.add_argument('-m','--model',    default='DEEPTE',
                    help='Model label for the CSV (default: DEEPTE).')
    ap.add_argument('-o','--out',      required=True, help='Output CSV path.')
    ap.add_argument('--digits', type=int, default=4)
    args = ap.parse_args()

    csv_rows = [['Level','Model','Folds',
                 'Macro-P','Macro-P_std',
                 'Macro-R','Macro-R_std',
                 'Macro-F1','Macro-F1_std',
                 'Accuracy','Accuracy_std']]

    print(f'\n=== {args.dataset} ===')
    for level_name, dirs in LEVEL_DIRS.items():
        rows = collect_level(args.save_dir, args.dataset, dirs)
        if not rows:
            print(f'  {level_name:12s} : no reports parsed under '
                  f'{os.path.join(args.save_dir, args.dataset, "<fold>", "{"+",".join(dirs)+"}")}')
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
        print('No data parsed; nothing to write.'); sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()