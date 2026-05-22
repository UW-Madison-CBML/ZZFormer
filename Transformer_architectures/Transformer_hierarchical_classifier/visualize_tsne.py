import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from model.model import (
    BaselineTransformer_Vanilla_MLM,
    BaselineTransformer_BERTSTyle_MLM,
    BaselineTransformer_MLM
)
import argparse
import pickle
import yaml
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

def visualize_latent_space(model, sequence_dict, max_seq_len=512, batch_size=64, save_dir=None,run_name=None,DPI=800,  device="cuda"):
    """
    Extracts embeddings, reduces dimensions via t-SNE, and plots the latent space.
    sequence_dict: dict of { "ATCG...": ("Order", "Superfamily") }
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_orders = []
    all_superfamilies = []
    
    sequences = list(sequence_dict.keys())
    
    print("Extracting embeddings from model...")
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            
            # Encode
            tokens = torch.stack([encode_sequence(s, max_seq_len) for s in batch_seqs]).to(device)
            src_key_padding_mask = (tokens == vocab["PAD"])
            
            batch_dict = {
                "tokens": tokens,
                "src_key_padding_mask": src_key_padding_mask
            }
            
            # Get mean-pooled embeddings
            embeddings = model.get_latent_embeddings(batch_dict)
            all_embeddings.append(embeddings.cpu())
            
            # Store labels
            for s in batch_seqs:
                order, superfamily = sequence_dict[s]
                all_orders.append(order)
                all_superfamilies.append(superfamily)

    # Concatenate all batches
    X = torch.cat(all_embeddings, dim=0).numpy()
    
    # ---------------------------------------------------------
    # 2. Dimensionality Reduction (t-SNE)
    # ---------------------------------------------------------
    print(f"Running t-SNE on {X.shape[0]} sequences ({X.shape[1]} dimensions)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_2d = tsne.fit_transform(X)
    
    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- Plot 1: By Order ---
    order_c = [order_colors.get(o, "gray") for o in all_orders] # default to gray if missing
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=order_c, s=15, alpha=0.8, edgecolors='none')
    ax1.set_title("Latent Space by Order", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Custom Legend for Order
    order_patches = [mpatches.Patch(color=color, label=label) for label, color in order_colors.items()]
    ax1.legend(handles=order_patches, loc='best', title="Orders", fontsize=10)

    # --- Plot 2: By Superfamily ---
    superfamily_c = [superfamily_colors.get(sf, "gray") for sf in all_superfamilies]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=superfamily_c, s=15, alpha=0.8, edgecolors='black', linewidth=0.2)
    ax2.set_title("Latent Space by Superfamily", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Custom Legend for Superfamily (Can be quite large, so we format it in columns)
    sf_patches = [mpatches.Patch(color=color, label=label) for label, color in superfamily_colors.items()]
    ax2.legend(handles=sf_patches, loc='center left', bbox_to_anchor=(1, 0.5), 
               title="Superfamilies", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{run_name}_tsne_visualization.png", dpi=DPI, bbox_inches='tight')
    print("Saved plot to 'latent_space_visualization.png'")
    # plt.show()

# ==========================================
# HOW TO RUN IT
# ==========================================
def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    cfg = config
    # 1. Load your model
    model = BaselineTransformer_Vanilla_MLM(
            src_vocab_size=VOCAB_SIZE,
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            num_layers=cfg["model"]["num_layers"],
            positional_encoding=cfg["model"]["positional_encoding"],
            max_position_embeddings=cfg["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            ignore_index=ignore_index
        )
    vocab = {
        "PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5
    }
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN = 0
    ignore_index = -100
    
    ckpt_path=args.model_dir
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    ds_path = args.seq_file if args.train_dir else config["data"]["train_dir"]
    with open(ds_path, "rb") as f:
        all_seqs = pickle.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Call the function
    visualize_latent_space(model, all_seqs, max_seq_len=512, batch_size=128, save_dir=args.save_dir,run_name=args.run_name,DPI=args.DPI, device=DEVICE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--mode", choices=["none", "mlm", "bert_mlm"], required=True)
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument('--seq_file', type=str, default=None, help='Path to training seq_file with labels pickle file')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to trained model  file')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save the image')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--DPI', type=str, default=800, help='DPI resolution of fig saved')


    
    args = parser.parse_args()

    main(args)