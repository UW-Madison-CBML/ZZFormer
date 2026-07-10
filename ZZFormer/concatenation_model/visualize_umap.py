import os
import argparse
import pickle
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
from data.dataloader_cnn import TopoDataset, load_pi_lookups
from torch.utils.data import Dataset, DataLoader
from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
import numpy as np
import torch
import wandb

# from model.model_transformer_hierarchical import build_classification_tree

from model.zzformer_concatenate import HierarchicalLongformerClassifier_Concat,build_classification_tree


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


# ---------------------------------------------------------------------------
# Label hierarchy
# ---------------------------------------------------------------------------
ORDER_TO_SUPERFAMILIES = {
    'LTR': ['Pao','Gypsy','Copia','DIRS','Caulimovirus','ERV'],
    'DNA': ['Harbinger','CMC','P','hAT','TcMar','PiggyBac','Zator','MULE','Merlin',
            'Kolobok','Maverick','Novosib','Zisupton','Crypton','Academ','IS3EU',
            'Dada','Sola','Ginger'],
    'LINE': ['R1','I','CR1','L1','RTE','L2','Dong-R4','R2','Dualen','CRE','Tad1',
             'Rex-Babar','Proto2','Proto1'],
    'Satellite': [], 'RC': ['Helitron'], 'SINE': ['tRNA','5S','7SL','U'],
    'Structural_RNA': [], 'PLE': [], 'Other': [],
}




def build_label_to_node_id(root: SoftmaxNode):
    """
    Returns {"LINE": node_idx, "LINE/L1": node_idx, ...}
    where node_idx is the position in root.node_list (what
    HierarchicalSoftmaxLoss expects as the target id).
    """
    root.set_indexes_if_unset()
    label_to_id = {}
    for idx, node in enumerate(root.node_list):
        if node is root:
            continue
        if node.parent is root:
            name = node.name                              # e.g. "LINE", "DIRS"
        else:
            name = f"{node.parent.name}/{node.name}"      # e.g. "LINE/L1"
        label_to_id[name] = idx
    return label_to_id





# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_batch(seqs, max_seq_len):
    """Char-level tokenize with BOS/EOS, pad to max_seq_len."""
    N, L = len(seqs), max_seq_len
    input_ids      = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((N, L), dtype=torch.long)
    body_max = L - 2
    for r, seq in enumerate(seqs):
        body = [VOCAB.get(c, UNK_TOKEN_ID) for c in seq.lower()[:body_max]]
        ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
        input_ids[r, :len(ids)]      = torch.tensor(ids, dtype=torch.long)
        attention_mask[r, :len(ids)] = 1
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Embedding extraction — uses the model's own pooled representation
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, sequences, max_seq_len, batch_size, device,
                       global_attention_mode="bos",topology_images=None):
    """
    Pull the same pooled representation that the classification head sees,
    so the UMAP reflects what the *classifier* learned.
    """
    model.eval().to(device)
    all_emb = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        input_ids, attention_mask = tokenize_batch(batch, max_seq_len)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Use the SAME mask convention as training (BOS-global by default)
        global_attention_mask = torch.zeros_like(attention_mask)
        if "bos" in global_attention_mode:
            global_attention_mask[:, 0] = 1
        if "eos" in global_attention_mode:
            global_attention_mask |= (input_ids == EOS_TOKEN_ID).long()

        # emb = model.get_latent_embeddings(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     global_attention_mask=global_attention_mask,
        # )
        emb = model.get_latent_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            topology_images=topology_images,  # required
            global_attention_mask=global_attention_mask,  # optional/ignored
        )
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0).numpy()




@torch.no_grad()
def extract_embeddings_from_loader(model, loader, device):
    model.eval().to(device)
    all_emb, all_orders, all_superfamilies, sequences = [], [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        topology_images = [x.to(device) for x in batch["topology_images"]]

        emb = model.get_latent_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            topology_images=topology_images,
        )
        all_emb.append(emb.cpu())

        # labels is batched by default_collate
        labels = batch["labels"]
        for lab in labels:
            order_sup= lab.split("/")
            # expected lab: (order, superfamily)
            order = order_sup[0]
            sf = order_sup[1] if len(order_sup) > 1 and order_sup[1] is not None else "No superfamily"
            all_orders.append(order)
            all_superfamilies.append(sf)

        sequences.extend(batch["sequence"])

    X = torch.cat(all_emb, dim=0).numpy()
    return X, all_orders, all_superfamilies, sequences

# ---------------------------------------------------------------------------
# UMAP plot
# ---------------------------------------------------------------------------
def visualize_latent_space_umap(model, sequence_dict, max_seq_len, batch_size,
                                save_dir, run_name, DPI, device,
                                global_attention_mode="bos", topology_images=None,loader=None):
    # sequences         = list(sequence_dict.keys())
    # all_orders        = [sequence_dict[s][0] for s in sequences]
    # all_superfamilies = [sequence_dict[s][1] if sequence_dict[s][1] is not None
    #                      else "No superfamily" for s in sequences]

    os.makedirs(save_dir, exist_ok=True)
    # emb_path = os.path.join(save_dir, f"{run_name}_only_embeddings.npy")

    
    # if os.path.exists(emb_path):
    #     print("Found cached embeddings; loading…")
    #     X, all_orders, all_superfamilies, sequences = np.load(emb_path, allow_pickle=True)
    # else:
        

    #     X, all_orders, all_superfamilies, sequences= extract_embeddings_from_loader(model, loader, device)
    #     print(f"Extracting embeddings from {len(sequences)} sequences "
    #           f"(global_attention_mode='{global_attention_mode}')…")



    #     np.save(emb_path, (X, all_orders, all_superfamilies, sequences))
    #     with open(os.path.join(save_dir,
    #               f"{run_name}_seq_and_embeddings.pkl"), "wb") as f:
    #         pickle.dump({s: X[i] for i, s in enumerate(sequences)}, f)
    #     print(f"Saved embeddings → {emb_path}")

    cache_path = os.path.join(save_dir, f"{run_name}_emb_cache.pkl")

    if os.path.exists(cache_path):
        print("Found cached embeddings; loading…")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        X = cache["X"]
        all_orders = cache["all_orders"]
        all_superfamilies = cache["all_superfamilies"]
        sequences = cache["sequences"]
    else:
        X, all_orders, all_superfamilies, sequences = extract_embeddings_from_loader(model, loader, device)

        print(f"Extracted embeddings from {len(sequences)} sequences")

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "X": X,
                    "all_orders": all_orders,
                    "all_superfamilies": all_superfamilies,
                    "sequences": sequences,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved cache → {cache_path}")








    print(f"Running UMAP on {X.shape[0]}×{X.shape[1]}…")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=350)

    ax1.scatter(X_2d[:, 0], X_2d[:, 1],
                c=[order_colors.get(o, "gray") for o in all_orders],
                s=15, alpha=0.5, edgecolors="black", linewidth=0.09)
    ax1.set_title("Order", fontsize=26); ax1.set_xticks([]); ax1.set_yticks([])

    ax2.scatter(X_2d[:, 0], X_2d[:, 1],
                c=[superfamily_colors.get(sf, "gray") for sf in all_superfamilies],
                s=15, alpha=0.5, edgecolors="black", linewidth=0.15)
    ax2.set_title("Superfamily", fontsize=26); ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{run_name}_umap_Jun26.png")
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Build the SAME tree used at training time ----
    classification_tree = build_classification_tree(
        ORDER_TO_SUPERFAMILIES,
        label_smoothing=cfg.get("label_smoothing", 0.0),
        gamma=cfg.get("gamma", 0.0),
    )


    model = HierarchicalLongformerClassifier_Concat(
        classification_tree     = classification_tree,
        vocab_size              = VOCAB_SIZE,
        d_model                 = cfg["model"]["d_model"],
        n_heads                 = cfg["model"]["nhead"],
        # n_heads_cross           = cfg["model"]["nhead_crossatten"],
        num_layers              = cfg["model"]["num_layers"],
        dim_feedforward         = cfg["model"]["dim_feedforward"],
        dropout                 = cfg["model"]["dropout"],
        max_position_embeddings = cfg["model"]["max_position_embeddings"],
        attention_window        = cfg["model"]["attention_window"],
        pad_token_id            = PAD_TOKEN_ID,
        bos_token_id            = BOS_TOKEN_ID,
        eos_token_id            = EOS_TOKEN_ID,
        classifier_hidden_dim   = cfg["model"].get("classifier_hidden_dim", 256),
        # ---- topology cross-attention ----
        topology_latent_dim     = cfg["model"].get("topology_latent_dim", 512),
        k_mers                  = tuple(cfg["model"].get("k_mers", (4, 8, 14, 20))),
    ).to(DEVICE)


    ckpt = torch.load(args.model_dir, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded {args.model_dir} (epoch {ckpt.get('epoch', '?')})")

    # ---- Aggregate sequences from train/test pickles ----
    all_seqs = {}
    if args.train_file:
        with open(args.train_file, "rb") as f:
            d = pickle.load(f)
        all_seqs.update(d)
        print(f"Loaded {len(d)} TRAIN sequences.")
    if args.test_file:
        with open(args.test_file, "rb") as f:
            d = pickle.load(f)
        all_seqs.update(d)
        print(f"Loaded {len(d)} TEST sequences.")
    if args.seq_file and not all_seqs:
        with open(args.seq_file, "rb") as f:
            d = pickle.load(f)
        all_seqs.update(d)
        print(f"Loaded {len(d)} sequences (single file).")

    assert all_seqs, "No sequences loaded — pass --train_file/--test_file/--seq_file."
    print(f"Total unique sequences for visualization: {len(all_seqs)}")



    # ---- Load persistence images ONCE (shared between train + val) ----
    # Expects pi_dir/4mer/*.tar.gz, pi_dir/8mer/*.tar.gz, etc.
    pi_lookups = load_pi_lookups(args.pi_dir, k_mers=(4, 8, 14, 20))

    max_seq_len = cfg["model"]["max_seq_len"]

    # --- Tree + label map ---
    classification_tree = build_classification_tree(
        ORDER_TO_SUPERFAMILIES,
        label_smoothing = cfg.get("label_smoothing", 0.0),
        gamma           = cfg.get("gamma", 0.0),
    )
    label_to_node_id = build_label_to_node_id(classification_tree)

    # --- Datasets ---
    train_ds = TopoDataset(
        data_dict   = all_seqs,
        max_seq_len = max_seq_len,
        pi_lookups  = pi_lookups,
        mask        = False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["train"]["batchsize"],
        shuffle     = False,
        num_workers = cfg["train"]["num_workers"],
        pin_memory  = True,
    )

    visualize_latent_space_umap(
        model, all_seqs,
        max_seq_len           = cfg["model"]["max_seq_len"],
        batch_size            = args.batch_size,
        save_dir              = args.save_dir,
        run_name              = args.run_name,
        DPI                   = args.DPI,
        device                = DEVICE,
        global_attention_mode = cfg["model"].get("global_attention_mode", "bos"),
        topology_images       = None,
        loader=train_loader,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--seq_file",   default=None)
    p.add_argument("--train_file", default=None)
    p.add_argument("--test_file",  default=None)
    p.add_argument("--model_dir",  required=True,
                   help="Path to trained HierarchicalLongformer .pt")
    p.add_argument("--save_dir",   required=True)
    p.add_argument("--run_name",   required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--pi_dir",     required=True,
                   help="Path to directory containing persistence image .tar.gz files")
    p.add_argument("--DPI",        type=int, default=800)
    args = p.parse_args()
    main(args)