#!/usr/bin/env python3
"""
Build a ground-truth pickle from FASTA files whose headers look like:
    >rnd-1_family-134#LINE/CR1
i.e., ><seq_id>#<Order>/<Superfamily>

Usage:
    python build_gt_pickle.py -i gt_dir1 [gt_dir2 ...] -o ground_truth.pkl
"""
import argparse, glob, hashlib, os, pickle, sys

from alias import load_alias_map, apply_alias



def iter_fastas(paths):
    for p in paths:
        if os.path.isdir(p):
            for ext in ('*.fa', '*.fasta', '*.fna'):
                yield from sorted(glob.glob(os.path.join(p, '**', ext), recursive=True))
        else:
            yield p

def parse_header(h):
    """'>rnd-1_family-134#LINE/CR1 extra' -> ('rnd-1_family-134', 'LINE', 'CR1')"""
    h = h.lstrip('>').strip().split()[0]   # drop comments after whitespace
    if '#' not in h:
        return h, None, None
    sid, lab = h.split('#', 1)
    if '/' in lab:
        order, sfam = lab.split('/', 1)
    else:
        order, sfam = lab, None
    return sid, order, sfam

def seq_key(seq):
    return hashlib.md5(seq.upper().encode()).hexdigest()

def parse_fasta(path):
    sid = order = sfam = None
    seq_chunks = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if sid is not None:
                    yield sid, order, sfam, ''.join(seq_chunks)
                sid, order, sfam = parse_header(line)
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if sid is not None:
            yield sid, order, sfam, ''.join(seq_chunks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', nargs='+', required=True,
                    help='Ground-truth FASTA files or directories.')
    ap.add_argument('-o', '--output', required=True,
                    help='Output pickle path.')
    ap.add_argument('--alias', default=None,
                help='Alias map (TSV file or inline "a=b,c=d") applied to GT '
                     'order/superfamily labels before storing.')

    args = ap.parse_args()
    alias = load_alias_map(args.alias)

    gt = {}
    n_total = n_skipped = 0
    for fp in iter_fastas(args.input):
        for sid, order, sfam, seq in parse_fasta(fp):
            n_total += 1
            if not seq:
                n_skipped += 1; continue
            k = seq_key(seq)
            order = apply_alias(order, alias)
            sfam  = apply_alias(sfam,  alias)
            
            gt[k] = {'seq_id': sid, 'order': order, 'superfamily': sfam,
                     'source_file': os.path.basename(fp)}
    with open(args.output, 'wb') as f:
        pickle.dump(gt, f)
    print(f'Wrote {len(gt)} GT entries from {n_total} sequences '
          f'({n_skipped} skipped) -> {args.output}')

if __name__ == '__main__':
    main()