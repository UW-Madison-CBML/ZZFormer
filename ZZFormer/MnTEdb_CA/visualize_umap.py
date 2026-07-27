import argparse
import os
import pickle
import numpy as np
import yaml
import re
import glob
import gc

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

import os
import gc
import yaml
import argparse
import random
import pickle
import sys
from collections import defaultdict
import glob
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
import umap  # pip install umap-learn

# Import model components and dataset helper
from model.ZZFormer_CAatend import HierarchicalLongformerClassifier
from model.helper_functions import *
from model.dataloader_cnn import TopoDataset  # Ensure TopoDataset is imported from your dataset module

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

ORDER_TO_SUPERFAMILIES = {
    'LTR': ['Pao', 'Gypsy', 'Copia', 'DIRS', 'Caulimovirus', 'ERV'],
    'DNA': ['Harbinger', 'CMC', 'P', 'hAT', 'TcMar', 'PiggyBac', 'Zator', 'MULE', 'Merlin', 'Kolobok', 'Maverick', 'Novosib', 'Zisupton', 'Crypton', 'Academ', 'IS3EU', 'Dada', 'Sola', 'Ginger'],
    'LINE': ['R1', 'I', 'CR1', 'L1', 'RTE', 'L2', 'Dong-R4', 'R2', 'Dualen', 'CRE', 'Tad1', 'Rex-Babar', 'Proto2', 'Proto1'],
    'Satellite': [],
    'RC': ['Helitron'],
    'SINE': ['tRNA', '5S', '7SL', 'U'],
    'Structural_RNA': [],
    'PLE': [],
    'Other': [],
}

superfamily_colors = {
    # LTR-like / retroviral group: blues
    "Pao": "#eff3ff",
    "Copia": "#bdd7e7",
    "ERV": "#6baed6",
    "Gypsy": "#2171b5",
    "DIRS": "#ffff33",
    "Caulimovirus": "#084594",

    # DNA / former TIR-like group: oranges/browns
    "CMC": "#feedde",
    "TcMar": "#fdbe85",
    "hAT": "#fd8d3c",
    "MULE": "#e6550d",
    "Harbinger": "#a63603",
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
    "tRNA": "#74c476",
    "5S": "#238b45",
    "7SL": "#bae4b3",
    "U": "#edf8e9",

    # RC / Helitron
    "Helitron": "#e41a1c",

    # Empty / non-superfamily groups
    "No superfamily": "gray",
}

order_colors = {
    "LTR": "#377eb8",
    "DNA": "#ff7f00",
    "LINE": "#984ea3",
    "SINE": "#4daf4a",
    "RC": "#e41a1c",
    "PLE": "#a65628",
    "Satellite": "#999999",
    "Structural_RNA": "#66c2a5",
    "Other": "#bdbdbd",
}


# def load_npz(filepath, load_meta=False):
#     """Loads NPZ arrays and optional metadata safely."""
#     data = np.load(filepath, allow_pickle=True)
#     images = data["images"]
#     if load_meta:
#         metadata = data["metadata"]
#         return images, metadata
#     return images


def extract_chunk_ids(filename):
    match = re.search(r"chunk_(\d+)_", filename)
    return int(match.group(1)) if match else 0


def build_hierarchical_classifier(cfg, classification_tree):
    """Instantiates the HierarchicalLongformerClassifier from config dict."""
    m = cfg["model"]
    k_mers = tuple(m.get("k_mers", [4, 8, 14, 20]))
    
    model = HierarchicalLongformerClassifier(
        classification_tree      = classification_tree,
        vocab_size               = len(VOCAB),
        d_model                  = m["d_model"],
        n_heads                  = m["nhead"],
        num_layers               = m["num_layers"],
        dim_feedforward          = m["dim_feedforward"],
        dropout                  = m.get("dropout", 0.1),
        max_position_embeddings = m.get("max_position_embeddings", 1026),
        attention_window         = m.get("attention_window", 256),
        pad_token_id             = PAD_TOKEN_ID,
        bos_token_id             = BOS_TOKEN_ID,
        eos_token_id             = EOS_TOKEN_ID,
        classifier_hidden_dim   = m.get("classifier_hidden_dim", 256),
        topology_latent_dim      = m.get("topology_latent_dim", 256),
        topo_channels            = m.get("topo_channels", 3),
        topo_filters             = m.get("topo_filters", 16),
        topo_reduced_persistence = m.get("topo_reduced_persistence", 16),
        k_mers                   = k_mers,
    )
    return model


@torch.no_grad()
def extract_embeddings_hierarchical(
    model,
    dataloader,
    device,
    k_mers=(4, 8, 14, 20),
):
    """
    Runs batches through HierarchicalLongformerClassifier and extracts the BOS-pooled
    hidden states from the backbone output tensor `h`.
    
    Returns:
        np.ndarray of shape (N, d_model)
    """
    model.eval()
    model.to(device)

    all_emb = []
    num_layers = len(k_mers)

    for batch in dataloader:
        input_ids       = batch["input_ids"].to(device, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(device, non_blocking=True)
        target_node_ids = batch["target_node_ids"].to(device, non_blocking=True)

        # Build list of topology image tensors across requested k-mers
        topology_images = [
            batch[f"{k}mer_image"].to(device, non_blocking=True) for k in k_mers
        ]

        # Forward pass returning ({'total_loss': ..., 'logits': ...}, h)
        _, h = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            target_node_ids = target_node_ids,
            topology_images = topology_images,
        )

        # BOS pooled vector is at sequence index 0: h[:, 0, :] -> Shape: (B, d_model)
        bos_embeddings = h[:, 0, :]
        all_emb.append(bos_embeddings.cpu())

    return torch.cat(all_emb, dim=0).numpy()


def visualize_latent_space_umap_leviver(
    model,
    dataset_dict,
    config,
    batch_size=64,
    save_dir=None,
    run_name=None,
    DPI=350,
    device=DEVICE,
):
    sequences = dataset_dict.get("sequences", [])
    all_orders = [
        meta[0] if isinstance(meta, (list, tuple)) else "Other"
        for meta in dataset_dict.get("labels", [])
    ]
    all_superfamilies = [
        meta[1] if isinstance(meta, (list, tuple)) and len(meta) > 1 else "No superfamily"
        for meta in dataset_dict.get("labels", [])
    ]

    k_mers = tuple(config["model"].get("k_mers", [4, 8, 14, 20]))
    emb_path = os.path.join(save_dir, f"{run_name}_only_embeddings.npy")

    if os.path.exists(emb_path):
        print(f"Found cached embeddings; loading from {emb_path}…")
        X = np.load(emb_path)
    else:
        print(f"Extracting embeddings from {len(sequences)} samples via TopoDataset…")

        dataset = TopoDataset(
            data_dict   = dataset_dict,
            max_seq_len = config["model"]["max_seq_len"],
            k_mers      = k_mers,
            mask        = False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 4,
            pin_memory  = True,
        )

        X = extract_embeddings_hierarchical(
            model      = model,
            dataloader = dataloader,
            device     = device,
            k_mers     = k_mers,
        )

        X = X.reshape(X.shape[0], -1) #flatten last 2 dimensions

        np.save(emb_path, X)
        with open(os.path.join(save_dir, f"{run_name}_seq_and_embeddings.pkl"), "wb") as f:
            pickle.dump({seq: X[i] for i, seq in enumerate(sequences)}, f)
        print(f"Saved embeddings → {emb_path}")

    # === UMAP Visualization ==
    print(f"Running UMAP on {X.shape[0]} samples × {X.shape[1]} dims…")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )
    X_2d = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=DPI)

    # --- Plot 1: By Order ---
    order_c = [order_colors.get(o, "gray") for o in all_orders]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=order_c, s=15, alpha=0.5, edgecolors="black", linewidth=0.09)
    ax1.set_title("Order", fontsize=26)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- Plot 2: By Superfamily ---
    superfamily_c = [superfamily_colors.get(sf, "gray") for sf in all_superfamilies]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=superfamily_c, s=15, alpha=0.5, edgecolors="black", linewidth=0.15)
    ax2.set_title("Superfamily", fontsize=26)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{run_name}_umap_vis.png")
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out_png}")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Build or Load Classification Tree Structure ---
    # tree_path = config["data"].get("tree_file", "classification_tree.pkl")
    # if os.path.exists(tree_path):
    #     with open(tree_path, "rb") as f:
    #         classification_tree = pickle.load(f)
    # else:
    classification_tree = build_classification_tree(
        ORDER_TO_SUPERFAMILIES,
        label_smoothing=config.get("label_smoothing", 0.0),
        gamma=config.get("gamma", 0.0),
    )

    label_map = build_label_to_node_id(classification_tree)

    # --- Load Data Dictionary ---
    if args.seq_file and os.path.exists(args.seq_file):
        print(f"Loading pickled dataset from {args.seq_file}…")
        with open(args.seq_file, "rb") as f:
            dataset_dict = pickle.load(f)
    elif args.pi_dir:
        print(f"Gathering NPZ files from {args.pi_dir}…")
        mer4 = sorted(glob.glob(f"{args.pi_dir}/*_4mer.npz"), key=extract_chunk_ids)
        mer8 = sorted(glob.glob(f"{args.pi_dir}/*_8mer.npz"), key=extract_chunk_ids)
        mer14 = sorted(glob.glob(f"{args.pi_dir}/*_14mer.npz"), key=extract_chunk_ids)
        mer20 = sorted(glob.glob(f"{args.pi_dir}/*_20mer.npz"), key=extract_chunk_ids)

        torch.cuda.empty_cache()
        gc.collect()

        dataset_dict = {
            '4mer': [],
            '8mer': [],
            '14mer': [],
            '20mer': [],
            'sequences': [],
            'labels': [],      # Formatted as (order, superfamily) tuples for UMAP
            'label_ids': [],
            'order': [],
            'superfamily': [],
        }

        for a, b, c, d in zip(mer4, mer8, mer14, mer20):
            arr4 = load_npz(a, load_meta=False)
            arr8 = load_npz(b, load_meta=False)
            arr14 = load_npz(c, load_meta=False)
            arr20, metadata = load_npz(d, load_meta=True)

            seqs = [m[0] for m in metadata]
            raw_labels = [m[1] for m in metadata]
            label_ids = [label_map.get(m[1]) for m in metadata]
            order = [m[2] for m in metadata]
            superfamily = [m[3] for m in metadata]

            labels_tuples = list(zip(order, superfamily))

            dataset_dict['4mer'].extend(arr4)
            dataset_dict['8mer'].extend(arr8)
            dataset_dict['14mer'].extend(arr14)
            dataset_dict['20mer'].extend(arr20)

            dataset_dict['sequences'].extend(seqs)
            dataset_dict['labels'].extend(labels_tuples)
            dataset_dict['label_ids'].extend(label_ids)
            dataset_dict['order'].extend(order)
            dataset_dict['superfamily'].extend(superfamily)
    else:
        raise ValueError("Either --seq_file or --pi_dir must be provided!")

    print(f"Loaded {len(dataset_dict.get('sequences', []))} sequences for visualization.")

    # --- Build Hierarchical Longformer Classifier ---
    model = build_hierarchical_classifier(config, classification_tree).to(DEVICE)
    ckpt = torch.load(args.model_dir, map_location=DEVICE)
    
    # Handle state_dict key matching
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {args.model_dir} (epoch {ckpt.get('epoch', '?')})")

    os.makedirs(args.save_dir, exist_ok=True)

    visualize_latent_space_umap_leviver(
        model        = model,
        dataset_dict = dataset_dict,
        config       = config,
        batch_size   = args.batch_size,
        save_dir     = args.save_dir,
        run_name     = args.run_name,
        DPI          = args.DPI,
        device       = DEVICE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Path to config yaml file")
    parser.add_argument("--seq_file",   default=None, help="Path to input .pkl dataset dict")
    parser.add_argument("--pi_dir",     default=None, help="Directory containing NPZ chunk files")
    parser.add_argument("--model_dir",  required=True, help="Path to trained classifier checkpoint (.pt)")
    parser.add_argument("--save_dir",   required=True, help="Directory to save generated outputs")
    parser.add_argument("--run_name",   required=True, help="Run name identifier prefix")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--DPI",        type=int, default=350)
    args = parser.parse_args()
    main(args)
