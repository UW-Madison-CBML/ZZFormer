import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

import argparse
import pickle
import yaml
import umap  # pip install umap-learn
import os

# Your provided color dictionaries
superfamily_colors = {
    "Bel-Pao": "#eff3ff", "Copia": "#bdd7e7", "ERV": "#6baed6", "Gypsy": "#2171b5",
    "CACTA": "#feedde", "TcMar": "#fdbe85", "hAT": "#fd8d3c", "MuLE": "#e6550d", "PIF": "#a63603",
    "CR1": "#f2f0f7", "I": "#dadaeb", "Jockey": "#bcbddc", "L1": "#9e9ac8", "R2": "#807dba", "RTE": "#6a51a3", "Rex1": "#4a1486",
    "SINE": "#edf8e9", "SINE1/7SL": "#bae4b3", "SINE2/tRNA": "#74c476", "SINE3/5S": "#238b45",
    "No superfamily": "gray"
}

order_colors = {
    "TIR": "#ff7f00", "LTR": "#377eb8", "SINE": "#4daf4a", "LINE": "#984ea3",
    "Helitron": "#e41a1c", "DIRS": "#ffff33", "PLE": "#a65628"
}

vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5}

def encode_sequence(seq: str, max_seq_len: int):
    """Encodes a string sequence to tensor with padding."""
    ids = [vocab.get(c, vocab["X"]) for c in seq[:max_seq_len]]
    pad_len = max_seq_len - len(ids)
    if pad_len > 0:
        ids = ids + [vocab["PAD"]] * pad_len
    return torch.tensor(ids, dtype=torch.long)

def visualize_latent_space_umap(model, sequence_dict, max_seq_len=512, batch_size=64, save_dir=None,run_name=None,DPI=800,  device="cuda"):
    """
    Extracts embeddings, reduces dimensions via UMAP, and plots the latent space.
    sequence_dict: dict of { "ATCG...": ("Order", "Superfamily") }
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_orders = []
    all_superfamilies = []
    
    sequences = list(sequence_dict.keys())

    if os.path.exists(f"{save_dir}/{run_name}_only_embeddings.npy"):
        print("Found existing embeddings. Loading from disk...")
        X = np.load(f"{save_dir}/{run_name}_only_embeddings.npy")
        # with open(f"{save_dir}/{run_name}_seq_and_embeddings.pkl", "rb") as f:
        #     seq_to_embedding = pickle.load(f)
        # Extract orders and superfamilies for plotting
        for seq in sequences:
            order, superfamily = sequence_dict[seq]
            all_orders.append(order)
            all_superfamilies.append(superfamily)
    else:
        print("Extracting embeddings from model...")
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]
                
                # Encode sequences into padded tensors
                tokens = torch.stack([encode_sequence(s, max_seq_len) for s in batch_seqs]).to(device)
                src_key_padding_mask = (tokens == vocab["PAD"])
                
                batch_dict = {
                    "tokens": tokens,
                    "src_key_padding_mask": src_key_padding_mask
                }

                # Get mean-pooled embeddings (Requires get_latent_embeddings in your model class)
                embeddings = model.get_latent_embeddings(batch_dict)
                all_embeddings.append(embeddings.cpu())
                
                # Store labels for plotting later
                for s in batch_seqs:
                    order, superfamily = sequence_dict[s]
                    all_orders.append(order)
                    all_superfamilies.append(superfamily)

        # Concatenate all batches into a single numpy array
        X = torch.cat(all_embeddings, dim=0).numpy()
        # Save embeddings
        np.save(f"{save_dir}/{run_name}_only_embeddings.npy", X)
        print(f"Saved embeddings to {save_dir}/{run_name}_embeddings.npy")
        
        # Create dict: {sequence: embedding}
        seq_to_embedding = {seq: X[i] for i, seq in enumerate(sequences)}
        
        # Save as pickle
        with open(f"{save_dir}/{run_name}_seq_and_embeddings.pkl", "wb") as f:
            pickle.dump(seq_to_embedding, f)
        print(f"Saved embeddings dict to {save_dir}/{run_name}_seq_and_embeddings.pkl")
        

    # ---------------------------------------------------------
    # 2. Dimensionality Reduction (UMAP)
    # ---------------------------------------------------------
    print(f"Running UMAP on {X.shape[0]} sequences ({X.shape[1]} dimensions)...")
    
    # UMAP Parameters:
    # n_neighbors: Larger values see more global structure, smaller see local. (default: 15)
    # min_dist: Controls how tightly UMAP packs points together. (default: 0.1)
    # metric: 'cosine' often works best for high-dimensional neural network embeddings
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='cosine', 
        random_state=42
    )
    
    X_2d = reducer.fit_transform(X)
    
    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    print("Plotting UMAP Latent Space...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- Plot 1: By Order ---
    order_c = [order_colors.get(o, "gray") for o in all_orders]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=order_c, s=15, alpha=0.5, edgecolors='black', linewidth=0.09)
    ax1.set_title("UMAP Latent Space by Order", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Custom Legend for Order
    order_patches = [mpatches.Patch(color=color, label=label) for label, color in order_colors.items()]
    ax1.legend(handles=order_patches, loc='best', title="Orders", fontsize=10)

    # --- Plot 2: By Superfamily ---
    superfamily_c = [superfamily_colors.get(sf, "gray") for sf in all_superfamilies]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=superfamily_c, s=15, alpha=0.5, edgecolors='black', linewidth=0.15)
    ax2.set_title("UMAP Latent Space by Superfamily", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Custom Legend for Superfamily (Formatted in columns outside the plot)
    sf_patches = [mpatches.Patch(color=color, label=label) for label, color in superfamily_colors.items()]
    ax2.legend(handles=sf_patches, loc='center left', bbox_to_anchor=(1, 0.5), 
               title="Superfamilies", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{run_name}_umap_visualization.png", dpi=DPI, bbox_inches='tight')
    print("Saved plot to 'latent_space_umap_visualization.png'")
    # plt.show()


# ==========================================
# HOW TO RUN IT
# ==========================================
def main(args):
    vocab = {
        "PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5
    }
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN = 0
    ignore_index = -100
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name= args.model_name if args.model_name else "model"
    if model_name=="model_noscale":
        from model.model_noscal import BaselineTransformer_Vanilla_MLM
    elif model_name=="model_nowtinit_noscal":
        from model.model_nowtinit_noscal import BaselineTransformer_Vanilla_MLM
    elif model_name=="model_nowtinit":
        from model.model_nowtinit import BaselineTransformer_Vanilla_MLM
    else:
        from model.model import BaselineTransformer_Vanilla_MLM

    # 1. Load your model
    model = BaselineTransformer_Vanilla_MLM(
            src_vocab_size=VOCAB_SIZE,
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["nhead"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            num_layers=config["model"]["num_layers"],
            positional_encoding=config["model"]["positional_encoding"],
            max_position_embeddings=config["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            ignore_index=ignore_index
        )

    
    ckpt_path=args.model_dir
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    ds_path = args.seq_file if args.seq_file else config["data"]["train_dir"]
    with open(ds_path, "rb") as f:
        all_seqs = pickle.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # from itertools import islice

    # all_seqs=dict(islice(all_seqs.items(), 10))
    print(f"Loaded {len(all_seqs)} sequences for visualization.")
    # 3. Call the function
    visualize_latent_space_umap(model, all_seqs, max_seq_len=512, batch_size=128, save_dir=args.save_dir,run_name=args.run_name,DPI=args.DPI, device=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--mode", choices=["none", "mlm", "bert_mlm"], required=True)
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument('--seq_file', type=str, default=None, help='Path to training seq_file with labels pickle file')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to trained model  file')
    parser.add_argument('--model_name', type=str, default=None, help='what type of model?')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save the image')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--DPI', type=int, default=800, help='DPI resolution of fig saved')


    
    args = parser.parse_args()

    main(args)