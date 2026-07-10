import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import pickle
import yaml
import umap
import os
import colorsys

from model.model_transformer_hierarchical import (
    HierarchicalTransformerClassifier,
    build_classification_tree,
    build_label_to_node_id,
    node_lineage_string,
)
from data.datalaoder_infer2 import HierarchicalFASTADataset
from data.dataloader_inference import HierarchicalSequenceDataset

# =========================================================================
# KNOWN COLORS — your existing palettes. Anything not here gets auto-colored.
# =========================================================================

KNOWN_ORDER_COLORS = {
    "TIR": "#ff7f00", "LTR": "#377eb8", "SINE": "#4daf4a", "LINE": "#984ea3",
    "Helitron": "#e41a1c", "DIRS": "#ffff33", "PLE": "#a65628", "RC": "#e41a1c",
}

KNOWN_SF_COLORS = {
    "Bel-Pao": "#eff3ff", "Copia": "#bdd7e7", "ERV": "#6baed6", "Gypsy": "#2171b5",
    "CACTA": "#feedde", "TcMar": "#fdbe85", "hAT": "#fd8d3c", "MuLE": "#e6550d",
    "MULE": "#e6550d", "PIF": "#a63603",
    "CR1": "#f2f0f7", "I": "#dadaeb", "Jockey": "#bcbddc", "L1": "#9e9ac8",
    "R2": "#807dba", "RTE": "#6a51a3", "Rex1": "#4a1486",
    "ID": "#edf8e9", "SINE1/7SL": "#bae4b3", "SINE2/tRNA": "#74c476",
    "SINE3/5S": "#238b45",
}






def get_known_sets_from_tree_and_palettes(classification_tree, known_order_colors, known_sf_colors):
    # From palettes
    known_orders = set(known_order_colors.keys())
    known_sfs = set(known_sf_colors.keys())

    # From tree node names (best-effort: use node_list + depth)
    if hasattr(classification_tree, "node_list"):
        for node in classification_tree.node_list:
            if getattr(node, "is_root", False):
                continue
            # In your tree: depth 1 = order, depth 2 = superfamily
            d = getattr(node, "depth", None)
            name = str(node)
            if d == 1:
                known_orders.add(name)
            elif d == 2:
                known_sfs.add(name)

    return known_orders, known_sfs

UNKNOWN_COLOR = "#817a7a"  # grey

def relabel_unknowns(orders, superfamilies, known_orders, known_sfs):
    new_orders = []
    new_sfs = []

    for o, sf in zip(orders, superfamilies):
        # order
        if o in known_orders:
            new_orders.append(o)
        else:
            new_orders.append("Unknown")

        # superfamily
        if sf == "No superfamily":
            new_sfs.append(sf)
        else:
            # accept either full sf or its short token after last slash (your current palette logic)
            sf_short = sf.split("/")[-1]
            if (sf in known_sfs) or (sf_short in known_sfs):
                new_sfs.append(sf)
            else:
                new_sfs.append("Unknown")

    return new_orders, new_sfs












def generate_distinct_colors(n, seed=42):
    """
    Generate n visually distinct colors using golden-ratio hue spacing.
    Returns a list of hex color strings.
    """
    colors = []
    rng = np.random.RandomState(seed)
    # Golden ratio conjugate for hue spacing
    golden_ratio = 0.618033988749895
    hue = rng.random()
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        # Vary saturation and lightness slightly for distinctness
        sat = 0.55 + rng.random() * 0.35   # 0.55–0.90
        light = 0.45 + rng.random() * 0.25  # 0.45–0.70
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def build_color_map(labels, known_colors):
    """
    Build a complete label → hex color mapping.
    Uses known_colors for labels that match, generates new colors for unknown labels.

    Args:
        labels:        list of label strings (all unique labels in the dataset)
        known_colors:  dict of {label_str: hex_color}

    Returns:
        dict of {label_str: hex_color} covering every unique label
    """
    unique_labels = sorted(set(labels))

    # Separate known and unknown
    known = {}
    unknown = []
    for label in unique_labels:
        if label in known_colors:
            known[label] = known_colors[label]
        else:
            # Also try matching just the superfamily part (after "/")
            short = label.split("/")[-1] if "/" in label else label
            if short in known_colors:
                known[label] = known_colors[short]
            else:
                unknown.append(label)

    # Generate colors for unknown labels
    if unknown:
        new_colors = generate_distinct_colors(len(unknown))
        for label, color in zip(unknown, new_colors):
            known[label] = color

    return known


# =========================================================================
# EMBEDDING EXTRACTION — uses the updated get_latent_embeddings(tokens, mask)
# =========================================================================

def extract_embeddings_for_umap(model, dataloader, classification_tree, device="cuda"):
    all_embeddings = []
    all_node_ids = []

    model.eval()
    with torch.no_grad():
        for tokens, src_key_padding_mask, target_node_ids in dataloader:
            tokens = tokens.to(device, non_blocking=True)
            src_key_padding_mask = src_key_padding_mask.to(device, non_blocking=True)

            embeddings = model.get_latent_embeddings(tokens, src_key_padding_mask)
            all_embeddings.append(embeddings.cpu())
            all_node_ids.append(target_node_ids.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_node_ids = torch.cat(all_node_ids, dim=0)

    ds = dataloader.dataset

    # --- Debug: verify root IDs really exist ---
    root_count = int((all_node_ids == 0).sum().item())
    print(f"[DEBUG] target_node_id==0 count (root fallback): {root_count}")

    # Build labels using stored flags
    orders = []
    superfamilies = []
    for i, nid in enumerate(all_node_ids):
        node = classification_tree.node_list[nid.item()]
        path = node.path  # (root, order) or (root, order, superfamily)

        # Order: if dataset says unknown order, force "Unknown" for plotting
        if hasattr(ds, "order_was_unknown") and ds.order_was_unknown[i]:
            order = "Unknown"
        else:
            order = str(path[1]) if len(path) > 1 else "Unknown"

        # Superfamily:
        # - if there is no SF in header: "No superfamily"
        # - if SF existed but couldn't be mapped: "Unknown"
        # - else if tree path contains SF: use it
        if hasattr(ds, "had_sfs") and not ds.had_sfs[i]:
            sf = "No superfamily"
        elif hasattr(ds, "sf_was_unknown") and ds.sf_was_unknown[i]:
            sf = "Unknown"
        else:
            sf = str(path[2]) if len(path) > 2 else "No superfamily"

        orders.append(order)
        superfamilies.append(sf)

    print("Unique orders:", sorted(set(orders)))
    print("Unique superfamilies:", sorted(set(superfamilies)))

    return all_embeddings, orders, superfamilies








# =========================================================================
# UMAP VISUALIZATION
# =========================================================================
def visualize_umap(
    embeddings,
    orders,
    superfamilies,
    save_dir,
    run_name,
    DPI=800,
    n_neighbors=15,
    min_dist=0.1,
):
    X = embeddings

    UNKNOWN_LABEL = "Unknown"
    UNKNOWN_COLOR = "#5d5454"  # grey

    # --------------------------
    # 0) Input sanity prints
    # --------------------------
    print("\n[DEBUG] ---- visualize_umap() ----")
    print(f"[DEBUG] N embeddings: {X.shape[0]}, dim: {X.shape[1]}")
    print(f"[DEBUG] N orders: {len(orders)}, N superfamilies: {len(superfamilies)}")
    if len(orders) != X.shape[0] or len(superfamilies) != X.shape[0]:
        print("[DEBUG][WARNING] Label list lengths do not match embeddings rows!")

    # Normalize whitespace (important if labels came from files)
    orders = [str(o).strip() for o in orders]
    superfamilies = [str(sf).strip() for sf in superfamilies]

    print(f"[DEBUG] Unique raw orders (first 50): {sorted(set(orders))[:50]}")
    print(f"[DEBUG] Unique raw superfamilies (first 50): {sorted(set(superfamilies))[:50]}")

    # --------------------------
    # 1) Relabel unknowns
    # --------------------------
    orders_clean = [o if o in KNOWN_ORDER_COLORS else UNKNOWN_LABEL for o in orders]

    superfamilies_clean = []
    for sf in superfamilies:
        if sf == "No superfamily":
            superfamilies_clean.append(sf)
            continue
        short = sf.split("/")[-1].strip() if "/" in sf else sf
        if (sf in KNOWN_SF_COLORS) or (short in KNOWN_SF_COLORS):
            superfamilies_clean.append(sf)
        else:
            superfamilies_clean.append(UNKNOWN_LABEL)

    # Debug prints: counts + examples
    n_unknown_orders = sum(o == UNKNOWN_LABEL for o in orders_clean)
    n_unknown_sfs = sum(sf == UNKNOWN_LABEL for sf in superfamilies_clean)
    n_no_sf = sum(sf == "No superfamily" for sf in superfamilies_clean)

    unknown_order_raw = sorted({o for o in orders if o not in KNOWN_ORDER_COLORS})
    unknown_sf_raw = sorted({
        sf for sf in superfamilies
        if sf != "No superfamily"
        and sf not in KNOWN_SF_COLORS
        and (sf.split("/")[-1].strip() not in KNOWN_SF_COLORS)
    })

    print(f"[DEBUG] Unknown orders after relabeling: {n_unknown_orders}/{len(orders_clean)}")
    if n_unknown_orders:
        print(f"[DEBUG] Raw order labels that became Unknown ({len(unknown_order_raw)}): {unknown_order_raw[:50]}")
    else:
        print("[DEBUG] No orders were relabeled to Unknown.")

    print(f"[DEBUG] Unknown superfamilies after relabeling: {n_unknown_sfs}/{len(superfamilies_clean)}")
    if n_unknown_sfs:
        print(f"[DEBUG] Raw SF labels that became Unknown ({len(unknown_sf_raw)}): {unknown_sf_raw[:50]}")
    else:
        print("[DEBUG] No superfamilies were relabeled to Unknown.")

    print(f"[DEBUG] No superfamily count: {n_no_sf}/{len(superfamilies_clean)}")

    # --------------------------
    # 2) Build color maps
    # --------------------------
    order_known = dict(KNOWN_ORDER_COLORS)
    order_known[UNKNOWN_LABEL] = UNKNOWN_COLOR

    sf_known = dict(KNOWN_SF_COLORS)
    sf_known[UNKNOWN_LABEL] = UNKNOWN_COLOR
    if "No superfamily" not in sf_known:
        sf_known["No superfamily"] = "#96d8d6"

    order_cmap = build_color_map(orders_clean, order_known)
    sf_cmap = build_color_map(superfamilies_clean, sf_known)

    # Confirm Unknown is in the cmap if it exists in labels
    print(f"[DEBUG] Unknown in unique_orders_clean? {UNKNOWN_LABEL in set(orders_clean)}")
    print(f"[DEBUG] Unknown in order_cmap? {UNKNOWN_LABEL in order_cmap}")
    print(f"[DEBUG] Unknown in unique_sfs_clean? {UNKNOWN_LABEL in set(superfamilies_clean)}")
    print(f"[DEBUG] Unknown in sf_cmap? {UNKNOWN_LABEL in sf_cmap}")

    # --------------------------
    # 3) Run UMAP
    # --------------------------
    print(f"Running UMAP on {X.shape[0]} sequences ({X.shape[1]} dimensions)...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    X_2d = reducer.fit_transform(X)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{run_name}_umap_coords.npy"), X_2d)

    # --------------------------
    # 4) Plot + legend debug
    # --------------------------
    print("Plotting UMAP Latent Space...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: By Order ---
    order_c = [order_cmap[o] for o in orders_clean]
    ax1.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=order_c, s=15, alpha=0.5,
        edgecolors="black", linewidth=0.09
    )
    ax1.set_title("UMAP Latent Space by Order", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])

    unique_orders = sorted(set(orders_clean))
    print(f"[DEBUG] unique_orders used for legend ({len(unique_orders)}): {unique_orders}")

    # Put Unknown at the end if present
    if UNKNOWN_LABEL in unique_orders:
        unique_orders = [o for o in unique_orders if o != UNKNOWN_LABEL] + [UNKNOWN_LABEL]

    order_patches = [mpatches.Patch(color=order_cmap[o], label=o) for o in unique_orders]
    ax1.legend(handles=order_patches, loc="best", title="Orders", fontsize=10)

    # --- Plot 2: By Superfamily ---
    sf_c = [sf_cmap[sf] for sf in superfamilies_clean]
    ax2.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=sf_c, s=15, alpha=0.5,
        edgecolors="black", linewidth=0.15
    )
    ax2.set_title("UMAP Latent Space by Superfamily", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    unique_sfs = sorted(set(superfamilies_clean))
    print(f"[DEBUG] unique_sfs used for legend ({len(unique_sfs)}): {unique_sfs}")

    tail = [x for x in ["No superfamily", UNKNOWN_LABEL] if x in unique_sfs]
    unique_sfs = [s for s in unique_sfs if s not in set(tail)] + tail

    sf_patches = [mpatches.Patch(color=sf_cmap[sf], label=sf) for sf in unique_sfs]
    ax2.legend(
        handles=sf_patches,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Superfamilies",
        fontsize=9,
        ncol=2,
    )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{run_name}_umap_visualization.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()

    print("[DEBUG] ---- done visualize_umap() ----\n")



# =========================================================================
# MAIN
# =========================================================================

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    vocab = {
        "PAD": 0,
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "X": 5
    }
    VOCAB_SIZE = len(vocab)
    vocab_size = VOCAB_SIZE  # PAD, A, C, G, T, X
    PAD_TOKEN = vocab["PAD"]
    ignore_index = -100


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Check for cached embeddings ----
    emb_cache_path = os.path.join(args.save_dir, f"{args.run_name}_only_embeddings.npy")
    meta_cache_path = os.path.join(args.save_dir, f"{args.run_name}_meta.pkl")

    if os.path.exists(emb_cache_path) and os.path.exists(meta_cache_path):
        print("Found cached embeddings. Loading from disk...")
        X = np.load(emb_cache_path)
        with open(meta_cache_path, "rb") as f:
            meta = pickle.load(f)
        orders = meta["orders"]
        superfamilies = meta["superfamilies"]

    else:
        # ---- Load checkpoint ----
        print(f"Loading checkpoint from: {args.model_dir}")
        checkpoint = torch.load(args.model_dir, map_location="cpu")

        # ---- Rebuild tree ----
        if "classification_tree" in checkpoint:
            classification_tree = checkpoint["classification_tree"]
            classification_tree.set_indexes_if_unset()
        else:
            order_to_superfamilies = config["hierarchy"]
            classification_tree = build_classification_tree(
                order_to_superfamilies,
                label_smoothing=config.get("label_smoothing", 0.0),
                gamma=config.get("gamma", 0.0),
            )

        label_map = build_label_to_node_id(classification_tree)

        # ---- Rebuild model ----
        model = HierarchicalTransformerClassifier(
            src_vocab_size=checkpoint.get("src_vocab_size", vocab_size),
            classification_tree=classification_tree,
            d_model=checkpoint.get("d_model", config["model"]["d_model"]),
            n_heads=config["model"]["nhead"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            num_layers=config["model"]["num_layers"],
            max_position_embeddings=config["model"]["max_seq_len"],
            pad_token_id=checkpoint.get("pad_token_id", PAD_TOKEN),
            classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(DEVICE)
        model.eval()

        # ---- Load dataset ----
        seq_path = args.seq_file

        if seq_path.endswith(".pkl") or seq_path.endswith(".pickle"):
            print(f"Loading pickle data from: {seq_path}")
            with open(seq_path, "rb") as f:
                all_seqs = pickle.load(f)
            print(f"Loaded {len(all_seqs)} sequences")

            dataset = HierarchicalSequenceDataset(
                all_seqs,
                label_to_id=label_map,
                max_seq_len=config["model"]["max_seq_len"],
                pad_token_id=PAD_TOKEN,
                ignore_index=ignore_index,
            )
        else:
            print(f"Loading FASTA data from: {seq_path}")

            dataset = HierarchicalFASTADataset(
                fasta_paths=seq_path,
                label_to_id=label_map,
                max_seq_len=config["model"]["max_seq_len"],
                pad_token_id=PAD_TOKEN,
                map_rules_str=args.map_rules,
                min_seq_len=args.min_seq_len,
                vocab=vocab,     
            )




        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=True,
        )

        # ---- Extract embeddings ----
        print("Extracting embeddings from model...")
        X, orders, superfamilies = extract_embeddings_for_umap(
            model, dataloader, classification_tree, device=DEVICE
        )

        # ---- Cache for reuse ----
        np.save(emb_cache_path, X)
        with open(meta_cache_path, "wb") as f:
            pickle.dump({"orders": orders, "superfamilies": superfamilies}, f)
        print(f"Cached embeddings to {emb_cache_path}")
        print(f"Cached metadata to {meta_cache_path}")

    # ---- Visualize ----
    print(f"\nEmbedding shape: {X.shape}")
    print(f"Unique orders: {sorted(set(orders))}")
    print(f"Unique superfamilies: {sorted(set(superfamilies))}")

    visualize_umap(
        X, orders, superfamilies,
        save_dir=args.save_dir,
        run_name=args.run_name,
        DPI=args.DPI,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP visualization of hierarchical model latent space")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained checkpoint .pt file")
    parser.add_argument("--seq_file", type=str, required=True,
                        help="Path to data: .pkl pickle or .fa/.fasta FASTA file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--run_name", type=str, required=True, help="Name prefix for output files")
    parser.add_argument("--DPI", type=int, default=800, help="DPI resolution of saved figure")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding extraction")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter")
    parser.add_argument("--map_rules", type=str, default="",
                        help="Comma-separated label remapping rules for FASTA input")
    parser.add_argument("--min_seq_len", type=int, default=0,
                        help="Minimum sequence length for FASTA input")

    args = parser.parse_args()
    main(args)

































'''









def extract_embeddings_for_umap(model, dataloader, classification_tree, device="cuda"):
    """
    Extract mean-pooled encoder embeddings + order/superfamily labels for every sample.

    Returns:
        embeddings: (N, d_model) numpy array
        orders:     list of order strings
        superfamilies: list of superfamily strings (or "No superfamily")
        accessions: list of accession strings (if available from dataset)
    """
    all_embeddings = []
    all_node_ids = []

    model.eval()
    with torch.no_grad():
        for tokens, src_key_padding_mask, target_node_ids in dataloader:
            tokens = tokens.to(device, non_blocking=True)
            src_key_padding_mask = src_key_padding_mask.to(device, non_blocking=True)

            embeddings = model.get_latent_embeddings(tokens, src_key_padding_mask)

            all_embeddings.append(embeddings.cpu())
            all_node_ids.append(target_node_ids.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_node_ids = torch.cat(all_node_ids, dim=0)

    # Extract order and superfamily labels from tree
    orders = []
    superfamilies = []
    for nid in all_node_ids:
        node = classification_tree.node_list[nid.item()]
        path = node.path  # (root, order) or (root, order, superfamily)

        if len(path) > 1:
            order = str(path[1])
        else:
            order = "Unknown"

        if len(path) > 2:
            sf = str(path[2])
        else:
            sf = "No superfamily"

        orders.append(order)
        superfamilies.append(sf)

    return all_embeddings, orders, superfamilies








def visualize_umap_older(
    embeddings,
    orders,
    superfamilies,
    save_dir,
    run_name,
    DPI=800,
    n_neighbors=15,
    min_dist=0.1,
):
    """
    Reduce embeddings to 2D via UMAP and plot colored by order and superfamily.
    Automatically generates colors for any labels not in the known palettes.
    """
    X = embeddings

    # Build color maps (auto-generates for unknown labels)
    order_cmap = build_color_map(orders, KNOWN_ORDER_COLORS)
    sf_cmap = build_color_map(superfamilies, KNOWN_SF_COLORS)

    # Print what was auto-generated
    auto_orders = [k for k in order_cmap if k not in KNOWN_ORDER_COLORS]
    auto_sfs = [k for k in sf_cmap if k not in KNOWN_SF_COLORS and k.split("/")[-1] not in KNOWN_SF_COLORS]
    if auto_orders:
        print(f"  Auto-generated colors for {len(auto_orders)} unknown orders: {auto_orders}")
    if auto_sfs:
        print(f"  Auto-generated colors for {len(auto_sfs)} unknown superfamilies: {auto_sfs}")

    # Run UMAP
    print(f"Running UMAP on {X.shape[0]} sequences ({X.shape[1]} dimensions)...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    X_2d = reducer.fit_transform(X)

    # Save UMAP coordinates
    np.save(os.path.join(save_dir, f"{run_name}_umap_coords.npy"), X_2d)

    # ---- Plot ----
    print("Plotting UMAP Latent Space...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: By Order ---
    order_c = [order_cmap[o] for o in orders]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=order_c, s=15, alpha=0.5,
                edgecolors="black", linewidth=0.09)
    ax1.set_title("UMAP Latent Space by Order", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])



    # Legend: only include orders that appear in the data
    unique_orders = sorted(set(orders))
    order_patches = [mpatches.Patch(color=order_cmap[o], label=o) for o in unique_orders]
    ax1.legend(handles=order_patches, loc="best", title="Orders", fontsize=10)

    # --- Plot 2: By Superfamily ---
    sf_c = [sf_cmap[sf] for sf in superfamilies]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=sf_c, s=15, alpha=0.5,
                edgecolors="black", linewidth=0.15)
    ax2.set_title("UMAP Latent Space by Superfamily", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    unique_sfs = sorted(set(superfamilies))
    sf_patches = [mpatches.Patch(color=sf_cmap[sf], label=sf) for sf in unique_sfs]
    ax2.legend(handles=sf_patches, loc="center left", bbox_to_anchor=(1, 0.5),
               title="Superfamilies", fontsize=9, ncol=2)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{run_name}_umap_visualization.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()
'''