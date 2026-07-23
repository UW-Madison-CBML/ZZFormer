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

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

from hierarchicalsoftmax import SoftmaxNode, HierarchicalSoftmaxLoss
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import LongformerModel, LongformerConfig

# Load my functions and classes
from model.dataloader_cnn import TopoDataset,LazyTopoDataset # Dataloader
from model.ZZFormer_CAatend import HierarchicalLongformerClassifier,TopologyEncoder# Model, CNN
from model.utils import * # Metrics
from model.helper_functions import * # all other helper functions

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


# ============================================================
# Label map — FLAT (Longformer is single-label classification)
# ============================================================
ORDER_TO_SUPERFAMILIES={'LTR': ['Pao', 'Gypsy', 'Copia', 'DIRS', 'Caulimovirus', 'ERV'],
'DNA': ['Harbinger', 'CMC', 'P', 'hAT', 'TcMar', 'PiggyBac', 'Zator', 'MULE', 'Merlin', 'Kolobok', 'Maverick', 'Novosib', 'Zisupton', 'Crypton', 'Academ', 'IS3EU', 'Dada', 'Sola', 'Ginger'],
'LINE': ['R1', 'I', 'CR1', 'L1', 'RTE', 'L2', 'Dong-R4', 'R2', 'Dualen', 'CRE', 'Tad1', 'Rex-Babar', 'Proto2', 'Proto1'],
'Satellite': [],
'RC': ['Helitron'],
'SINE': ['tRNA', '5S', '7SL', 'U'],
'Structural_RNA': [],
'PLE': [],
'Other': [],
}


# ============================================================
# Train / Val
# ============================================================
def run_train(model, dataloader, optimizer, model_cfg):
    model.train()
    total_loss = 0.0
    # Grab k_mers list from config, default to dataset standard
    k_mers = model_cfg.get("k_mers", (4, 8, 14, 20))
    for batch_idx, batch in enumerate(dataloader):
        input_ids       = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        target_node_ids = batch["target_node_ids"].to(DEVICE, non_blocking=True)
        images = [
            batch[f"{k}mer_images"].to(DEVICE, non_blocking=True) 
            for k in k_mers
        ]
        optimizer.zero_grad()
        # Forward pass
        out, _ = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            target_node_ids = target_node_ids,
            topology_images = images,
        )
        loss = out["total_loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader) #epoch loss


@torch.no_grad()
def run_val(model, dataloader, classification_tree, model_cfg, threshold=0.0):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    embeddings = []
    for batch in dataloader:
        input_ids       = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        target_node_ids = batch["target_node_ids"].to(DEVICE, non_blocking=True)
        # topology_images = _move_topology_images(batch["topology_images"], DEVICE)
        images = [
            batch[f"{k}mer_images"].to(DEVICE, non_blocking=True) 
            for k in model_cfg.get("k_mers", (4, 8, 14, 20))
        ]
        out, h = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            target_node_ids = target_node_ids,
            topology_images = images,
        )
        total_loss += out["total_loss"].item()
        all_logits.append(out["logits"].cpu())
        all_targets.append(target_node_ids.cpu())
        embeddings.append(h.cpu())
    avg_loss    = total_loss / len(dataloader)
    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    embeddings = torch.cat(embeddings, dim=0)
    # Hierarchical decoding → per-sample predicted node
    probs      = node_probabilities(all_logits, root=classification_tree)
    pred_nodes = greedy_predictions(probs, root=classification_tree,
                                    threshold=threshold)
    true_nodes = [classification_tree.node_list[int(i)] for i in all_targets]
    def split(node):
        if node is classification_tree:
            return (None, None)
        if node.parent is classification_tree:
            return (node.name, None)       # order-only leaf (DIRS, PLE, ...)
        return (node.parent.name, node.name)
    def macro(true, pred):
        p, r, f1, _ = precision_recall_fscore_support(
            true, pred, average="macro", zero_division=0
        )
        return {"p": float(p), "r": float(r), "f1": float(f1), "acc": float(accuracy_score(true, pred))}
    # ---- Order metrics ----
    o_true = [split(n)[0] for n in true_nodes]
    o_pred = [split(n)[0] for n in pred_nodes]
    order_m = macro(o_true, o_pred)
    # ---- Superfamily metrics (skip samples with no true SF) ----
    sf_pairs = [(split(t)[1], split(p)[1])
                for t, p in zip(true_nodes, pred_nodes)
                if split(t)[1] is not None]
    if sf_pairs:
        sf_true, sf_pred = zip(*sf_pairs)
        sf_pred = [p if p is not None else "__STOPPED__" for p in sf_pred]
        sf_m = macro(list(sf_true), list(sf_pred))
    else:
        sf_m = {"p": 0., "r": 0., "f1": 0., "acc": 0.}
    print(f"  Val loss {avg_loss:.4f}")
    print(f"  Order  | acc {order_m['acc']:.4f} | P {order_m['p']:.4f} "
          f"R {order_m['r']:.4f} F1 {order_m['f1']:.4f}")
    print(f"  SF     | acc {sf_m['acc']:.4f} | P {sf_m['p']:.4f} "
          f"R {sf_m['r']:.4f} F1 {sf_m['f1']:.4f}")
    metrics = {
        "val_loss":  avg_loss,
        "order_acc": order_m["acc"], "order_p": order_m["p"],
        "order_r":   order_m["r"],   "order_f1": order_m["f1"],
        "sf_acc":    sf_m["acc"],    "sf_p":    sf_m["p"],
        "sf_r":      sf_m["r"],      "sf_f1":   sf_m["f1"],
    }
    results = {"embeddings": embeddings, "o_true":o_true, "o_pred":o_pred, "sf_true":sf_true, "sf_pred":sf_pred}
    return metrics, results


# ============================================================
# Main
# ============================================================
def _slice_data_dict(d, n):
    """Take the first n entries of every parallel list in a TopoDataset dict."""
    return {k: v[:n] for k, v in d.items()}

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    wandb_cfg = cfg["wandb"]
    run_tag = args.run_name or "longformer_topo"
    fold = args.fold

    if not args.debugging:
        wandb.init(
            name=args.run_name or f"longformer_fold{args.fold}",
            entity=args.wandb_team    or cfg["wandb"]["team"],
            project=args.wandb_project or cfg["wandb"]["project"],
            dir=args.wandb_dir         or cfg["wandb"]["dir"],
            config=cfg,
        )

    classification_tree = build_classification_tree(
        ORDER_TO_SUPERFAMILIES,
        label_smoothing = cfg.get("label_smoothing", 0.0),
        gamma           = cfg.get("gamma", 0.0),
    )

    # Build tree
    label_map = build_label_to_node_id(classification_tree)

    # --- Data ---
    # Load npz
    all_data = defaultdict(list)

    mer4 = sorted(glob.glob(f"{args.pi_dir}/*_4mer.npz"), key=extract_chunk_ids)
    mer8 = sorted(glob.glob(f"{args.pi_dir}/*_8mer.npz"), key=extract_chunk_ids)
    mer14 = sorted(glob.glob(f"{args.pi_dir}/*_14mer.npz"), key=extract_chunk_ids)
    mer20 = sorted(glob.glob(f"{args.pi_dir}/*_20mer.npz"), key=extract_chunk_ids)

    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()

    file_quads = list(zip(mer4, mer8, mer14, mer20))

    train_dataset = LazyTopoDataset(
        file_quads  = file_quads,
        label_map   = label_map,
        max_seq_len = model_cfg['max_seq_len'],
        fold_idx    = args.fold,
        split       = 'train',
        k_mers      = (4, 8, 14, 20),
    )

    test_dataset = LazyTopoDataset(
        file_quads  = file_quads,
        label_map   = label_map,
        max_seq_len = model_cfg['max_seq_len'],
        fold_idx    = args.fold,
        split       = 'test',
        k_mers      = (4, 8, 14, 20),
    )


    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg["batchsize"], 
        shuffle=True, 
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg["batchsize"], 
        shuffle=False, 
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )


    if args.debugging:
        train_data = _slice_data_dict(train_data, 100)
        test_data   = _slice_data_dict(test_data,    50)

    print("We are using model - HierarchicalLongformerClassifier_Concat")
    # --- Model (Longformer + per-k-mer CNN topology cross-attention) ---
    model = HierarchicalLongformerClassifier(
        classification_tree     = classification_tree,
        vocab_size              = VOCAB_SIZE,
        d_model                 = model_cfg["d_model"],
        n_heads                 = model_cfg["nhead"],
        num_layers              = model_cfg["num_layers"],
        dim_feedforward         = model_cfg["dim_feedforward"],
        dropout                 = model_cfg["dropout"],
        max_position_embeddings = model_cfg["max_position_embeddings"],
        attention_window        = model_cfg["attention_window"],
        pad_token_id            = PAD_TOKEN_ID,
        bos_token_id            = BOS_TOKEN_ID,
        eos_token_id            = EOS_TOKEN_ID,
        classifier_hidden_dim   = model_cfg.get("classifier_hidden_dim", 256),
        topology_latent_dim     = model_cfg.get("topology_latent_dim", 256), # Dimension fed to CrossAttention
        k_mers                  = tuple(model_cfg.get("k_mers", (4, 8, 14, 20))),
        # ---- Topology Encoder Hyperparameters ----
        topo_channels           = model_cfg["topology_in_channels"], 
        topo_filters            = model_cfg["topology_cnn_filters"], 
        topo_reduced_persistence= 16,
    )

    # --- Pre-trained MLM backbone ---
    if args.pretrained_mlm:
        model = load_pretrained_longformer_mlm(args.pretrained_mlm, model, DEVICE)

    # Freeze ONLY the Longformer backbone — the new per-k-mer topology CNNs,
    # k-mer projections, BOS cross-attention layers, and head stay trainable.
    for p in model.longformer.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    FREEZE_EPOCHS = 3
    print(f"Backbone frozen for first {FREEZE_EPOCHS} epochs")


    max_id_in_data = max(max(train_data['label_ids']), max(test_data['label_ids']))
    n_nodes        = len(classification_tree.node_list)
    assert max_id_in_data < n_nodes, (max_id_in_data, n_nodes)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params before unfreezing backbone: {n_trainable:,}")

    # model.parameters() already contains self.topology_encoder
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01)
    )

    # --- Save dir ---
    save_dir  = args.save_dir or cfg["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_tag}_{fold}.pt")

    # --- Train ---
    for epoch in range(train_cfg["epochs"]):
        if epoch == FREEZE_EPOCHS:
            print("--- Unfreezing Longformer backbone ---")
            for p in model.longformer.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.longformer.parameters()),
                "lr":     train_cfg["lr"] * 0.1,   # 10x smaller for fine-tuning
            })

        train_loss = run_train(model, train_loader, optimizer, model_cfg)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f}")

        # if not args.debugging:
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

    # --- Save ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch":            epoch,
        "label_to_node_id": label_map,
    }, save_path)
    print(f"  ↳ new model at epoch {epoch} saved to {save_path}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreezing: {n_trainable:,}")

    # --- Validate ---
    val_out, res = run_val(model, val_loader, classification_tree, model_cfg)
    with open(os.path.join(save_dir, f"{run_tag}_{fold}_val.yaml"), "w") as f:
        yaml.safe_dump(val_out, f)

    pickle_path = os.path.join(save_dir, f"{run_tag}_{fold}_val.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(res, f)













# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--fold",          required=True, type=int)
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--run_name",      default="run")
    p.add_argument("--seed",          default=22, type=int)
    p.add_argument("--debugging",     action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_team",    default=None)
    p.add_argument("--wandb_dir",     default=None)
    p.add_argument("--pi_dir",        required=True, help="Dir containing precomputed persistence images")
    p.add_argument("--pretrained_mlm", type=str, default=None,
               help="Path to LongformerForMaskedLM .pt checkpoint. "
                    "Backbone weights will be transferred; classification head is randomly initialized.")
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    main(args)



    # p.add_argument("--train_dir",     required=True, help="Train pickle: {seq: (order, sf_or_None)}")
    # p.add_argument("--val_dir",       required=True, help="Val   pickle: {seq: (order, sf_or_None)}")