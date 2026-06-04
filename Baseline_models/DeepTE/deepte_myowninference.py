import argparse
import json
import numpy as np
from keras.models import load_model
import itertools

BASES = ['A','C','G','T']
K = 7
VEC_LEN = len(BASES) ** K  # 16384

def word_seq(seq, k, stride=1):
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]

def generate_kmer_dic(k):
    # fixed ordering depends on itertools.product order
    return {''.join(p): 0 for p in itertools.product(BASES, repeat=k)}

def generate_mat(words_list, kmer_dic):
    for w in words_list:
        if w in kmer_dic:           # ignore unexpected kmers
            kmer_dic[w] += 1
    return [kmer_dic[k] for k in kmer_dic]

def clean_seq_training_style(seq: str) -> str:
    """
    Match your training-time cleaning (4-letter model):
      - uppercase
      - remove tabs/spaces
      - map ambiguity codes to A/C/G/T (not to N)
      - map N -> T
    """
    seq = seq.upper().replace("\t", "").replace(" ", "")
    seq = seq.replace("Y","C")
    seq = seq.replace("D","G")
    seq = seq.replace("S","C")
    seq = seq.replace("R","G")
    seq = seq.replace("V","A")
    seq = seq.replace("K","G")
    seq = seq.replace("N","T")
    seq = seq.replace("H","A")
    seq = seq.replace("W","A")
    seq = seq.replace("M","C")
    seq = seq.replace("X","G")
    seq = seq.replace("B","C")
    # Optional: handle gaps if your FASTA contains them
    seq = seq.replace("-", "T")
    return seq

def seq_to_vec(seq, k=K, stride=1):
    seq = clean_seq_training_style(seq)
    words = word_seq(seq, k, stride=stride)
    kmer_dic = generate_kmer_dic(k)
    return generate_mat(words, kmer_dic)

def read_fasta(path):
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

def invert_dict(d):
    # d: class_name -> index
    return {int(v): k for k, v in d.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label-map", required=True, help="JSON saved from training (label->index)")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", default="predictions.tsv")
    args = ap.parse_args()

    model = load_model(args.model)

    with open(args.label_map) as f:
        label_to_idx = json.load(f)
    inv_map = invert_dict(label_to_idx)

    recs = read_fasta(args.fasta)
    if not recs:
        raise SystemExit("No FASTA records found.")

    X = np.asarray([seq_to_vec(seq) for _, seq in recs], dtype="float64")
    X = X.reshape(X.shape[0], 1, VEC_LEN, 1)

    probs = model.predict(X, verbose=0)

    # Sanity checks to prevent wrong label mapping
    n_classes_model = probs.shape[1]
    n_classes_map = len(inv_map)
    if n_classes_model != n_classes_map:
        raise SystemExit(
            f"Class count mismatch: model outputs {n_classes_model} classes "
            f"but label map has {n_classes_map}.\n"
            f"Did you pass the correct *_label_map.json for this model?"
        )

    pred_idx = np.argmax(probs, axis=1)
    pred_label = [inv_map[int(i)] for i in pred_idx]
    pred_conf = probs[np.arange(len(pred_idx)), pred_idx].astype(float)

    with open(args.out, "w") as out:
        out.write("id\tpred_label\tpred_index\tconfidence\n")
        for (rid, _), lab, idx, conf in zip(recs, pred_label, pred_idx, pred_conf):
            out.write(f"{rid}\t{lab}\t{int(idx)}\t{conf:.6f}\n")

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()