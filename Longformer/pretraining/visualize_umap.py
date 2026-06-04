# from ALL_SCRIPTS.pretrain_longformer_mlm import VOCAB
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

from transformers import LongformerConfig, LongformerForMaskedLM

import argparse
import pickle
import yaml
import umap  # pip install umap-learn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = {
    "PAD":  0,
    "a":    1, "c": 2, "g": 3, "t": 4,
    "x":    5,
    "BOS":  6,
    "EOS":  7,
    "MASK": 8,
}
VOCAB_SIZE     = len(VOCAB)
PAD_TOKEN_ID   = VOCAB["PAD"]
BOS_TOKEN_ID   = VOCAB["BOS"]
EOS_TOKEN_ID   = VOCAB["EOS"]
MASK_TOKEN_ID  = VOCAB["MASK"]
UNK_TOKEN_ID   = VOCAB["x"]
IGNORE_INDEX   = -100


superfamily_colors = {
    # LTR-like / retroviral group: blues
    "Pao": "#eff3ff",          # old Bel-Pao
    "Copia": "#bdd7e7",
    "ERV": "#6baed6",
    "Gypsy": "#2171b5",
    "DIRS": "#ffff33",         # kept from old order_colors
    "Caulimovirus": "#084594",

    # DNA / former TIR-like group: oranges/browns
    "CMC": "#feedde",          # old CACTA-like color
    "TcMar": "#fdbe85",
    "hAT": "#fd8d3c",
    "MULE": "#e6550d",         # old MuLE
    "Harbinger": "#a63603",    # old PIF
    
    # New DNA superfamilies: related orange/brown shades
    "P": "#fdd0a2",
    "PiggyBac": "#fdae6b",
    "Zator": "#e34a33",
    "Merlin": "#b30000",
    "Kolobok": "#7f2704",
    "Maverick": "#d94801",
    "Novosib": "#8c2d04",
    "Zisupton": "#cc4c02",
    "Crypton": "#993404",
    "Academ": "#ec7014",
    "IS3EU": "#fe9929",
    "Dada": "#d95f0e",
    "Sola": "#f16913",
    "Ginger": "#a63603",

    # LINE group: purples
    "CR1": "#f2f0f7",
    "I": "#dadaeb",
    "L1": "#9e9ac8",
    "R2": "#807dba",
    "RTE": "#6a51a3",
    "R1": "#cbc9e2",
    "L2": "#756bb1",
    "Dong-R4": "#54278f",
    "Dualen": "#3f007d",
    "CRE": "#bcbddc",
    "Tad1": "#9e9ac8",
    "Rex-Babar": "#4a1486",
    "Proto2": "#6a51a3",
    "Proto1": "#807dba",

    # SINE group: greens
    "tRNA": "#74c476",         # old SINE2/tRNA
    "5S": "#238b45",           # old SINE3/5S
    "7SL": "#bae4b3",          # old SINE1/7SL
    "U": "#edf8e9",

    # RC / Helitron
    "Helitron": "#e41a1c",

    # Empty / non-superfamily groups
    "No superfamily": "gray",
}

order_colors = {
    "LTR": "#377eb8",
    "DNA": "#ff7f00",           # old TIR color
    "LINE": "#984ea3",
    "SINE": "#4daf4a",
    "RC": "#e41a1c",            # old Helitron color
    "PLE": "#a65628",

    # New broad categories
    "Satellite": "#999999",
    "Structural_RNA": "#66c2a5",
    "Other": "#bdbdbd",
}

# vocab = {"PAD": 0, "a": 1, "c": 2, "g": 3, "t": 4, "x": 5}

vocab = {
    "PAD":  0,
    "a":    1, "c": 2, "g": 3, "t": 4,
    "x":    5,
    "BOS":  6,
    "EOS":  7,
    "MASK": 8,
}
VOCAB_SIZE     = len(vocab)
PAD_TOKEN_ID   = vocab["PAD"]
BOS_TOKEN_ID   = vocab["BOS"]
EOS_TOKEN_ID   = vocab["EOS"]
MASK_TOKEN_ID  = vocab["MASK"]
UNK_TOKEN_ID   = vocab["x"]
IGNORE_INDEX   = -100

# ============================================================
# Build the MLM model
# ============================================================
def build_longformer_mlm(cfg):
    m = cfg["model"]
    lcfg = LongformerConfig(
        attention_window             = m["attention_window"],
        vocab_size                   = m["vocab_size"],
        max_position_embeddings      = m["max_position_embeddings"],
        hidden_size                  = m["d_model"],
        num_hidden_layers            = m["num_layers"],
        num_attention_heads          = m["nhead"],
        intermediate_size            = m["dim_feedforward"],
        position_embedding_type      = m.get("position_embedding_type", "absolute"),
        return_dict                  = True,
        pad_token_id                 = PAD_TOKEN_ID,
        bos_token_id                 = BOS_TOKEN_ID,
        eos_token_id                 = EOS_TOKEN_ID,
        hidden_dropout_prob          = m["dropout"],
        attention_probs_dropout_prob = m["dropout"],
    )
    return LongformerForMaskedLM(lcfg)



def encode_sequence(seq: str, max_seq_len: int):
    """Encodes a string sequence to tensor with padding."""
    ids = [vocab.get(c, vocab["x"]) for c in seq[:max_seq_len]]
    pad_len = max_seq_len - len(ids)
    if pad_len > 0:
        ids = ids + [vocab["PAD"]] * pad_len
    return torch.tensor(ids, dtype=torch.long)



@torch.no_grad()
def extract_embeddings_longformer(
    model,
    sequences,
    max_seq_len,
    batch_size,
    device,
    pool="mean",            # "mean" (over real tokens) or "bos" (the BOS/[CLS] token)
):
    """
    Run sequences through LongformerForMaskedLM and pull hidden states from the
    final layer of the backbone. Returns (N, hidden_size) numpy array.
    """
    model.eval()
    model.to(device)

    all_emb = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]

        # ---- Tokenize (with BOS/EOS so it matches MLM pretraining) ----
        N, L = len(batch), max_seq_len
        input_ids      = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((N, L), dtype=torch.long)
        body_max = L - 2
        for r, seq in enumerate(batch):
            body = [vocab.get(c, UNK_TOKEN_ID) for c in seq.lower()[:body_max]]
            ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
            input_ids[r, :len(ids)]      = torch.tensor(ids, dtype=torch.long)
            attention_mask[r, :len(ids)] = 1

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1   # global on BOS (matches MLM pretrain)

        # ---- Forward through the BACKBONE only (skip the MLM head) ----
        out = model.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state            # (B, L, hidden_size)

        # ---- Pool ----
        if pool == "bos":
            emb = hidden[:, 0, :]                 # BOS token (acts like [CLS])
        elif pool == "mean":
            # Masked mean over real tokens only
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(f"unknown pool: {pool}")

        all_emb.append(emb.cpu())

    return torch.cat(all_emb, dim=0).numpy()


def visualize_latent_space_umap(model, sequence_dict, max_seq_len=512,
                                batch_size=64, save_dir=None, run_name=None,
                                DPI=800, device=DEVICE, pool="mean"):
    sequences = list(sequence_dict.keys())
    all_orders        = [sequence_dict[s][0] for s in sequences]
    all_superfamilies = [sequence_dict[s][1] for s in sequences]

    emb_path = f"{save_dir}/{run_name}_only_embeddings.npy"
    if os.path.exists(emb_path):
        print("Found cached embeddings; loading…")
        X = np.load(emb_path)
    else:
        print(f"Extracting embeddings from {len(sequences)} sequences "
              f"(pool='{pool}')…")
        X = extract_embeddings_longformer(
            model, sequences,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
            pool=pool,
        )
        np.save(emb_path, X)
        with open(f"{save_dir}/{run_name}_seq_and_embeddings.pkl", "wb") as f:
            pickle.dump({s: X[i] for i, s in enumerate(sequences)}, f)
        print(f"Saved embeddings → {emb_path}")

    # === Everything below is unchanged from your script ===
    print(f"Running UMAP on {X.shape[0]}×{X.shape[1]}…")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(X_2d[:, 0], X_2d[:, 1],
                c=[order_colors.get(o, "gray") for o in all_orders],
                s=15, alpha=0.5, edgecolors="black", linewidth=0.09)
    ax1.set_title("UMAP Latent Space by Order", fontsize=16)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(handles=[mpatches.Patch(color=c, label=l)
                        for l, c in order_colors.items()],
               loc="best", title="Orders", fontsize=10)

    ax2.scatter(X_2d[:, 0], X_2d[:, 1],
                c=[superfamily_colors.get(s, "gray") for s in all_superfamilies],
                s=15, alpha=0.5, edgecolors="black", linewidth=0.15)
    ax2.set_title("UMAP Latent Space by Superfamily", fontsize=16)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.legend(handles=[mpatches.Patch(color=c, label=l)
                        for l, c in superfamily_colors.items()],
               loc="center left", bbox_to_anchor=(1, 0.5),
               title="Superfamilies", fontsize=9, ncol=2)

    plt.tight_layout()
    out_png = f"{save_dir}/{run_name}_umap_visualization.png"
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot → {out_png}")













def visualize_latent_space_umap_leviver(model, sequence_dict, max_seq_len=512,
                                batch_size=64, save_dir=None, run_name=None,
                                DPI=800, device=DEVICE, pool="mean"):
    sequences = list(sequence_dict.keys())
    all_orders        = [sequence_dict[s][0] for s in sequences]
    all_superfamilies = [sequence_dict[s][1] for s in sequences]

    emb_path = f"{save_dir}/{run_name}_only_embeddings.npy"
    if os.path.exists(emb_path):
        print("Found cached embeddings; loading…")
        X = np.load(emb_path)
    else:
        print(f"Extracting embeddings from {len(sequences)} sequences "
              f"(pool='{pool}')…")
        X = extract_embeddings_longformer(
            model, sequences,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
            pool=pool,
        )
        np.save(emb_path, X)
        with open(f"{save_dir}/{run_name}_seq_and_embeddings.pkl", "wb") as f:
            pickle.dump({s: X[i] for i, s in enumerate(sequences)}, f)
        print(f"Saved embeddings → {emb_path}")

    # === Everything below is unchanged from your script ===
    print(f"Running UMAP on {X.shape[0]}×{X.shape[1]}…")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=350)

    # --- Plot 1: By Order ---
    order_c = [order_colors.get(o, "gray") for o in all_orders]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=order_c, s=15, alpha=0.5, edgecolors='black', linewidth=0.09)
    ax1.set_title("Order", fontsize=26)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Custom Legend for Order
    order_patches = [mpatches.Patch(color=color, label=label) for label, color in order_colors.items()]
    # ax1.legend(handles=order_patches, loc='best', title="Orders", fontsize=10)

    # --- Plot 2: By Superfamily ---
    superfamily_c = [superfamily_colors.get(sf, "gray") for sf in all_superfamilies]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=superfamily_c, s=15, alpha=0.5, edgecolors='black', linewidth=0.15)
    ax2.set_title("Superfamily", fontsize=26)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Custom Legend for Superfamily (Formatted in columns outside the plot)
    sf_patches = [mpatches.Patch(color=color, label=label) for label, color in superfamily_colors.items()]
    # ax2.legend(handles=sf_patches, loc='center left', bbox_to_anchor=(1, 0.5), 
    #             title="Superfamilies", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{run_name}_umap_vis.png",  dpi=DPI, bbox_inches="tight")
    plt.close()








def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Build model (must match the MLM pretraining config!) ---
    model = build_longformer_mlm(config).to(DEVICE)
    ckpt  = torch.load(args.model_dir, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded {args.model_dir} (epoch {ckpt.get('epoch', '?')})")

    # --- Load sequences ---
    ds_path = args.seq_file or config["data"]["train_dir"]
    with open(ds_path, "rb") as f:
        all_seqs = pickle.load(f)
    print(f"Loaded {len(all_seqs)} sequences for visualization.")

    os.makedirs(args.save_dir, exist_ok=True)
    visualize_latent_space_umap_leviver(
        model, all_seqs,
        max_seq_len = config["model"]["max_seq_len"],   # use config, not hardcoded 512
        batch_size  = 128,
        save_dir    = args.save_dir,
        run_name    = args.run_name,
        DPI         = args.DPI,
        device      = DEVICE,
        pool        = args.pool,                         # NEW
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--seq_file",  default=None)
    parser.add_argument("--model_dir", required=True, help="Path to trained MLM .pt")
    parser.add_argument("--save_dir",  required=True)
    parser.add_argument("--run_name",  required=True)
    parser.add_argument("--DPI",       type=int, default=800)
    parser.add_argument("--pool",      choices=["mean", "bos"], default="mean",
                        help="How to pool token-level hidden states into a sequence vector.")
    args = parser.parse_args()
    main(args)


