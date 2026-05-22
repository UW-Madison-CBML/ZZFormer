
import os
import gc

import pandas as pd
import yaml
import wandb
import argparse
import random
import numpy as np
import torch
import pickle
from functools import partial
from torch.utils.data import DataLoader
import math
# Assuming these are correctly defined in your local modules
from model.model_CE_wAE import HierarchicalFFNTransformerClassifier
import glob
import tarfile


from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions

from data.dataloader_CA import HierarchicalPersistenceDataset, hierarchical_collate
from utils import calcmetrics_torcheval_multiclass_filtered


def write_fold_metrics(metrics_save_path, fold, val_out, tag="final"):
    """
    Append a formatted block of per-fold val metrics.
    `val_out` is the dict returned by run_val(...):
        {'loss', 'order': {...}, 'superfamily': {...}}
    """
    o = val_out["order"]
    s = val_out["superfamily"]
    loss = val_out["loss"]

    sep = "%" * 130 + "\n"
    bar = "=" * 130 + "\n"

    with open(metrics_save_path, "a") as f:
        f.write(
            f"[{tag}] Fold {fold} | Loss {loss:.4f} | "
            f"Order  F1 {o['F1']:.4f}  Prec {o['precision']:.4f}  Rec {o['recall']:.4f}  "
            f"AUROC {o['AUROC']:.4f}  AUPRC {o['AUPRC']:.4f}  normAUPRC {o['normAUPRC']:.4f} | "
            f"SF     F1 {s['F1']:.4f}  Prec {s['precision']:.4f}  Rec {s['recall']:.4f}  "
            f"AUROC {s['AUROC']:.4f}  AUPRC {s['AUPRC']:.4f}  normAUPRC {s['normAUPRC']:.4f}\n"
        )
        f.write(sep)
        f.write(
            f"{'Level':<6} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | "
            f"{'AUROC':>8} | {'AUPRC':>8} | {'normAUPRC':>10} | {'baseAUPRC':>10}\n"
        )
        f.write(bar)
        f.write(
            f"{'Order':<6} | {o['accuracy']:8.4f} | {o['precision']:8.4f} | "
            f"{o['recall']:8.4f} | {o['F1']:8.4f} | {o['AUROC']:8.4f} | "
            f"{o['AUPRC']:8.4f} | {o['normAUPRC']:10.4f} | {o['baseline_auprc']:10.4f}\n"
        )
        f.write(
            f"{'SF':<6} | {s['accuracy']:8.4f} | {s['precision']:8.4f} | "
            f"{s['recall']:8.4f} | {s['F1']:8.4f} | {s['AUROC']:8.4f} | "
            f"{s['AUPRC']:8.4f} | {s['normAUPRC']:10.4f} | {s['baseline_auprc']:10.4f}\n"
        )
        f.write(bar + "\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# TRAINING / VALIDATION  (hierarchical, single head)
# ==============================================================================

def _move_batch(batch, device):
    """Move a hierarchical_collate batch onto `device`."""
    out = {
        "tokens":               batch["tokens"].to(device, non_blocking=True),
        "src_key_padding_mask": batch["src_key_padding_mask"].to(device, non_blocking=True),
        "target_node_ids":      batch["target_node_ids"].to(device, non_blocking=True),
        "topology_latent_stack": [
            t.to(device, non_blocking=True) for t in batch["topology_latent_stack"]
        ],
        # topology_mask is either None or a list of (None | bool tensor)
        "topology_mask": (
            None if batch["topology_mask"] is None
            else [None if m is None else m.to(device, non_blocking=True)
                  for m in batch["topology_mask"]]
        ),
    }
    return out


def run_train(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        b = _move_batch(batch, DEVICE)

        optimizer.zero_grad()
        outputs = model(
            tokens=b["tokens"],
            src_key_padding_mask=b["src_key_padding_mask"],
            target_node_ids=b["target_node_ids"],
            topology_latent_stack=b["topology_latent_stack"],
            topology_mask=b["topology_mask"],
        )
        loss = outputs["total_loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def run_val(model, dataloader, root, ignore_index=-100):
    """
    Returns:
      {
        'loss': float,
        'order':       <calcmetrics_torcheval_multiclass_filtered output>,
        'superfamily': <calcmetrics_torcheval_multiclass_filtered output>,
      }
    """
    model.eval()
    total_loss = 0.0

    all_logits = []
    all_target_ids = []

    for batch in dataloader:
        b = _move_batch(batch, DEVICE)

        outputs = model(
            tokens=b["tokens"],
            src_key_padding_mask=b["src_key_padding_mask"],
            target_node_ids=b["target_node_ids"],
            topology_latent_stack=b["topology_latent_stack"],
            topology_mask=b["topology_mask"],
        )
        total_loss += outputs["total_loss"].item()

        all_logits.append(outputs["logits"].detach().cpu())
        all_target_ids.append(b["target_node_ids"].detach().cpu())

    avg_loss = total_loss / len(dataloader)
    all_logits     = torch.cat(all_logits,     dim=0)   # (N, root.layer_size) RAW logits
    all_target_ids = torch.cat(all_target_ids, dim=0)   # (N,)

    # ------------------------------------------------------------------
    # Build per-level "logits" + targets from the hierarchical output.
    # ------------------------------------------------------------------
    # node_probabilities returns P(node) for EVERY node in the tree,
    # already including the parent-chain product (so depth-2 entries
    # are joint probabilities P(order, superfamily)).
    # Shape: (N, len(root.node_list))
    node_probs = node_probabilities(all_logits, root=root)

    # ---- Build index lookups over the tree ----
    order_children = list(root.children)                          # depth-1 nodes
    order_to_idx   = {n: i for i, n in enumerate(order_children)}
    sf_nodes       = [c for o in order_children for c in o.children]   # depth-2 nodes
    sf_to_idx      = {n: i for i, n in enumerate(sf_nodes)}

    # Tree-wide node-id → position in node_list (for slicing node_probs)
    nodelist_pos = {node: i for i, node in enumerate(root.node_list)}

    def _order_ancestor(node):
        cur = node
        while cur.parent is not None and cur.parent is not root:
            cur = cur.parent
        return cur  # depth-1 node

    # ---- Per-sample true nodes (from node ids) ----
    true_nodes = [root.node_list[int(i)] for i in all_target_ids]

    # ============================== ORDER ==============================
    # logits: log P(order_i) for each order, derived from node_probs.
    order_cols = torch.tensor(
        [nodelist_pos[o] for o in order_children], dtype=torch.long
    )
    order_probs = node_probs[:, order_cols].clamp_min(1e-12)      # (N, num_orders)
    order_logits_for_metric = order_probs.log()                   # softmax(log p) == p

    y_true_order = torch.tensor(
        [order_to_idx[_order_ancestor(n)] for n in true_nodes],
        dtype=torch.long,
    )

    order_metrics = calcmetrics_torcheval_multiclass_filtered(
        y_true=y_true_order,
        y_pred_logits=order_logits_for_metric,
        pad_token_id=-1,             # disable pad filtering (no pad class at order level)
        ignore_index=ignore_index,
        average="macro",
    )

    # ============================ SUPERFAMILY ==========================
    # Only samples whose true label is itself a superfamily contribute.
    # Samples labeled at the order level (no SF) get ignore_index.
    sf_cols = torch.tensor(
        [nodelist_pos[s] for s in sf_nodes], dtype=torch.long
    )
    if len(sf_cols) > 0:
        sf_probs = node_probs[:, sf_cols].clamp_min(1e-12)        # (N, num_sf)
        sf_logits_for_metric = sf_probs.log()

        y_true_sf = torch.tensor(
            [sf_to_idx[n] if n in sf_to_idx else ignore_index
             for n in true_nodes],
            dtype=torch.long,
        )

        sf_metrics = calcmetrics_torcheval_multiclass_filtered(
            y_true=y_true_sf,
            y_pred_logits=sf_logits_for_metric,
            pad_token_id=-1,
            ignore_index=ignore_index,
            average="macro",
        )
    else:
        sf_metrics = {k: 0.0 for k in
            ["AUROC", "AUPRC", "normAUPRC", "baseline_auprc",
             "accuracy", "F1", "precision", "recall"]}

    # ---- Pretty print ----
    print(f"Val Loss: {avg_loss:.4f}")
    for name, m in (("Order", order_metrics), ("Superfamily", sf_metrics)):
        print(
            f"  {name:<11}| Acc {m['accuracy']:.4f}  "
            f"P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['F1']:.4f}  "
            f"AUROC {m['AUROC']:.4f}  AUPRC {m['AUPRC']:.4f}  "
            f"normAUPRC {m['normAUPRC']:.4f}"
        )

    return {
        "loss": avg_loss,
        "order": order_metrics,
        "superfamily": sf_metrics,
    }






# ====================================================================================
# CHECKPOINT LOADING
# ====================================================================================
def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device="cpu",
    load_optimizer=True,
    strict=True
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if load_optimizer and optimizer is not None and "optim_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    best_val_f1 = checkpoint.get("best_val_f1", -1.0)
    
    print(f"Loaded checkpoint: {checkpoint_path} | Resuming from epoch {epoch+1} | Best F1: {best_val_f1:.4f}")
    
    return model, optimizer, epoch, best_val_f1




def load_pretrained_mlm_weights(pretrained_mlm_path, new_model):
    print(f"Loading pretrained MLM weights from {pretrained_mlm_path}...")
    
    # 1. Load the checkpoint
    checkpoint = torch.load(pretrained_mlm_path, map_location=DEVICE)
    
    # -> THE FIX: Check if it's a checkpoint dict, and extract just the weights
    if "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
    else:
        old_state_dict = checkpoint

    new_state_dict = {}
    
    for key, weight in old_state_dict.items():
        # (Optional safety step): If the old model used DataParallel, remove "module."
        key = key.replace("module.", "")
        
        # A. Transfer the Embedding
        if key.startswith('src_embed.'):
            new_state_dict[key] = weight
            
        # B. Transfer Positional Encoding
        elif key.startswith('pos_encoder.'):
            new_state_dict[key] = weight
            
        # C. Map the Transformer Layers
        elif key.startswith('transformer_encoder.layers.'):
            # Extract the layer index
            parts = key.split('.')
            layer_idx = int(parts[2])
            
            # If the new model has fewer layers than the pretrained one, we drop the extra layers
            if layer_idx < len(new_model.encoder_layers):
                new_key = key.replace('transformer_encoder.layers.', 'encoder_layers.')
                new_state_dict[new_key] = weight
                
        # D. Ignore the MLM Head
        elif key.startswith('sequence_head.'):
            continue

    # 2. Load into the new model with strict=False
    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
    
    print("\n--- Weight Transfer Complete ---")
    
    # Filter the missing keys to show a cleaner log
    expected_missing = [k for k in missing_keys if 'classifier' in k or 'kmer_projections' in k or 'logit_scale' in k]
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    for k in missing_keys:
        print(f"Missing key: {k}")
    
    if unexpected_missing:
        print(f"⚠️ WARNING - These core keys are missing and shouldn't be:\n{unexpected_missing[:10]}")
    else:
        print(f"✅ Success! {len(expected_missing)} new head keys correctly initialized from scratch.")
        
    return new_model















def build_classification_tree(
    order_to_superfamilies: dict,
    label_smoothing: float = 0.0,
    gamma: float = 0.0,
) -> SoftmaxNode:
    """
    Builds a 2-level hierarchical softmax tree.

    Args:
        order_to_superfamilies: e.g. {
            "LINE": ["CR1", "L1", "L2", "Jockey", "RTE"],
            "SINE": ["Alu", "MIR", "tRNA"],
            "DNA":  ["hAT", "TcMar", "Merlin"],
            ...
        }
    Returns:
        root: The root SoftmaxNode with set_indexes() already called.
    """
    root = SoftmaxNode(
        "root",
        label_smoothing=label_smoothing,
        gamma=gamma,
    )
    for order_name, superfamily_list in order_to_superfamilies.items():
        order_node = SoftmaxNode(
            order_name,
            parent=root,
            label_smoothing=label_smoothing,
            gamma=gamma,
        )
        for sf_name in superfamily_list:
            SoftmaxNode(
                sf_name,
                parent=order_node,
                label_smoothing=label_smoothing,
                gamma=gamma,
            )
    root.set_indexes()
    return root

def build_label_to_node_id(root: SoftmaxNode) -> dict:
    """
    Builds a mapping from node name strings → node_id integers.
    These integer IDs are what you pass as target_node_ids during training.
    They index into root.node_list, which is what HierarchicalSoftmaxLoss uses.
    """
    root.set_indexes_if_unset()
    label_to_id = {}
    for node_id, node in enumerate(root.node_list):
        # Full path (e.g., "LINE/CR1")
        if node.parent and not node.parent.is_root:
            full_name = "/".join(
                [str(n) for n in node.ancestors[1:]] + [str(node)]
            )
        else:
            full_name = str(node)
        label_to_id[full_name] = node_id
        # Short name if unambiguous
        short_name = str(node)
        if short_name not in label_to_id:
            label_to_id[short_name] = node_id
    return label_to_id


def node_lineage_string(node) -> str:
    """Convert a SoftmaxNode to its full lineage path string."""
    if node.is_root:
        return "Unknown"
    return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])

def get_PI(path):
    all_pkl = {}
    image_files = glob.glob(f"./{path}/*")
    # print(image_files)
    for file in image_files:
        with tarfile.open(file, "r:gz") as tar:
            all_files = tar.getnames()
            pkl_path = next((f for f in all_files if f.endswith('.pkl')), None)
            if pkl_path:
                member = tar.getmember(pkl_path)
                f = tar.extractfile(member)
                data = pickle.load(f)
                all_pkl = all_pkl | data
    return all_pkl

def update_metadata(lookup, all_pkl):
    for seq, metadata in lookup.items():
        if seq in all_pkl:
            all_pkl[seq].update(metadata)
        else:
            all_pkl[seq] = {
                **metadata,
                'persistence_image': np.zeros((128,128,5)) #add 0 if none
            }
    return all_pkl



# ------------------------------------------------------------------
# CLI:
#   --fold 0..4                 which CV fold to run
#   --mer4_dir / --mer8_dir / --mer14_dir / --mer20_dir   PI dirs
#   --labels_path               TSV with seq_x, labels, fold_{i}, dataset
#   --config / --pretrained_mlm / --run_name / --seed / ...
# ------------------------------------------------------------------

def build_fold_dicts(labels_df, pi_paths, fold, label_map):
    """
    Returns (train_dict, val_dict) shaped for HierarchicalPersistenceDataset:
        {
          'seq_x':[...], 'Label':[...], 'label_id':[...], 'dataset':[...],
          <pi_path_0>: [...], <pi_path_1>: [...], ...
        }

    Single hierarchical label_id per sample — no order/superfamily split.
    """
    # Map labels to tree node ids ONCE
    labels_df = labels_df.copy()
    labels_df["label_id"] = labels_df["labels"].map(label_map)

    # OPTIONAL: drop rows whose label couldn't be mapped (e.g. ambiguous "LTR"
    # with no superfamily, if you decided to exclude those)
    labels_df = labels_df.dropna(subset=["label_id"])
    labels_df["label_id"] = labels_df["label_id"].astype(int)

    # unmapped = labels_df[labels_df["label_id"].isna()]["labels"].unique()
    # assert len(unmapped) == 0, f"Labels not in tree: {unmapped}"

    fold_col = f"fold_{fold}"
    lookup = (
        labels_df.drop_duplicates("seq_x")
                 .set_index("seq_x")[["label_id", "labels", fold_col, "dataset"]]
                 .to_dict("index")
    )

    # Load + align PIs for every k-mer
    pi_per_kmer = {}
    for p in pi_paths:
        pkl = get_PI(p)
        pkl = update_metadata(lookup, pkl)   # fills zeros where PI missing
        pi_per_kmer[p] = pkl

    # Split by fold
    def _split(role):
        rows = labels_df[labels_df[fold_col] == role].drop_duplicates("seq_x")
        seqs = rows["seq_x"].tolist()
        out = {
            "seq_x":    seqs,
            "Label":    rows["labels"].tolist(),
            "label_id": rows["label_id"].tolist(),
            "dataset":  rows["dataset"].tolist(),
        }
        for p, pkl in pi_per_kmer.items():
            out[p] = [pkl[s]["persistence_image"] for s in seqs]
        return out

    return _split("train"), _split("test")   # "test" = val for this fold


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vocab        = {"PAD": 0, "a": 1, "c": 2, "g": 3, "t": 4, "x": 5}
    VOCAB_SIZE   = len(vocab)
    PAD_TOKEN    = vocab["PAD"]
    IGNORE_INDEX = -100

    ORDER_TO_SUPERFAMILIES={'DIRS': [],
    'Helitron': [],
    'Line': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
    'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
    'PLE': [],
    'Sine': ['SINE', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
    'TIR': ['CACTA', 'MuLE', 'PIF', 'TcMar', 'hAT']}

    # ---- K-mer wiring (must match #PI streams) ----
    pi_paths, K_MERS = [], []
    for d, k in [(args.mer4_dir, 4), (args.mer8_dir, 8),
                 (args.mer14_dir, 14), (args.mer20_dir, 20)]:
        if d:
            pi_paths.append(d)
            K_MERS.append(k)
    assert len(pi_paths) > 0, "Need at least one --merN_dir"

    # ---- Hierarchical tree ----
    root = build_classification_tree(ORDER_TO_SUPERFAMILIES)
    label_map = build_label_to_node_id(root)

    # ---- Labels & per-fold data ----
    labels = pd.read_csv(args.labels_path, sep="\t")
    train_dict, val_dict = build_fold_dicts(labels, pi_paths, args.fold, label_map)

    assert len(vocab) == config["model"]["vocab_size"]

    # ---- Model ----
    model = HierarchicalFFNTransformerClassifier(
        src_vocab_size=VOCAB_SIZE,
        classification_tree=root,                 # ← replaces num_orders
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["nhead"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        max_position_embeddings=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN,
        ignore_index=IGNORE_INDEX,
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        k_mers=K_MERS,
        topology_latent_dim=config["topology"]["topological_embedding_dim"],
        context_mode="pi_tokens",                 # PersistenceImageEncoder5Tokens
    )

    # ---- Pretrained MLM ----
    if args.pretrained_mlm:
        print(f"Loading pretrained MLM weights from {args.pretrained_mlm}...")
        model = load_pretrained_mlm_weights(args.pretrained_mlm, model)

    # Freeze backbone initially
    for p in model.src_embed.parameters():
        p.requires_grad = False
    for p in model.encoder_layers.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    # ---- WandB ----
    if not args.debugging:
        wandb.init(
            name=args.run_name or f"fold{args.fold}_{args.seed}",
            settings=wandb.Settings(_service_wait=300),
            entity=args.wandb_team    or config["wandb"]["team"],
            project=args.wandb_project or config["wandb"]["project"],
            dir=args.wandb_dir         or config["wandb"]["dir"],
            config=config,
        )

    # ---- Datasets / Loaders (defined ONCE) ----
    MAX_SEQ_LEN = config["model"]["max_seq_len"]

    train_dataset = HierarchicalPersistenceDataset(
        train_dict, pi_keys=pi_paths, vocab=vocab, unk_token="x", max_seq_len=MAX_SEQ_LEN,
    )
    val_dataset = HierarchicalPersistenceDataset(
        val_dict,   pi_keys=pi_paths, vocab=vocab, unk_token="x", max_seq_len=MAX_SEQ_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=hierarchical_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=False,                            # ← val should not shuffle
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=hierarchical_collate,
    )




    # ---------------- Resumption & Saving Setup ----------------
    save_dir = args.save_dir or config["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    save_path         = os.path.join(save_dir, f"hier_fold{args.fold}_best_{args.run_name}.pt")
    metrics_save_path = os.path.join(save_dir, "hier_allfold_metrics.txt")

    # 1. Peek at checkpoint
    init_epoch  = 0
    best_val_f1 = -1.0
    if os.path.exists(save_path):
        tmp = torch.load(save_path, map_location="cpu")
        init_epoch  = tmp.get("epoch", -1) + 1
        best_val_f1 = tmp.get("best_val_f1", -1.0)
        print(f"Found checkpoint for fold {args.fold}; resuming from epoch {init_epoch}.")

    # 2. Past freeze phase? Unfreeze before building the optimizer.
    FREEZE_EPOCHS = 4
    if init_epoch > FREEZE_EPOCHS:
        for p in model.parameters():
            p.requires_grad = True
        head_params = [p for n, p in model.named_parameters()
                    if not (n.startswith("src_embed") or n.startswith("encoder_layers"))]
        backbone_params = (list(model.src_embed.parameters())
                        + list(model.encoder_layers.parameters()))
        optimizer = torch.optim.AdamW(
            [{"params": head_params,     "lr": config["train"]["lr"]},
            {"params": backbone_params, "lr": config["train"]["lr"]}],
        )
    else:
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=config["train"]["lr"])

    # 3. Optimizer
    # trainable = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.AdamW(trainable, lr=config["train"]["lr"])

    # 4. Restore weights / optimizer state if resuming
    if os.path.exists(save_path):
        model, optimizer, _, _ = load_checkpoint(
            model, optimizer, save_path, device=DEVICE,
            load_optimizer=True, strict=True,
        )

        # Validate the loaded checkpoint once, log to file
        val_out = run_val(model, val_loader, root=root, ignore_index=IGNORE_INDEX)
        # with open(metrics_save_path, "a") as f:
        #     f.write(
        #         f"[resume] Fold {args.fold} | loss {val_out['loss']:.4f} | "
        #         f"order F1 {val_out['order']['F1']:.4f} | "
        #         f"sf F1 {val_out['superfamily']['F1']:.4f}\n"
        #     )
        write_fold_metrics(metrics_save_path, args.fold, val_out, tag="resume")

    # ---------------- Training loop ----------------
    print(f"\n{'='*40}\nSTARTING FOLD {args.fold}\n{'='*40}")

    for epoch in range(init_epoch, config["train"]["epochs"]):
        # Unfreeze backbone at FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS:
            print("\n--- Unfreezing transformer backbone ---")
            for p in model.parameters():
                p.requires_grad = True
            unfrozen = (list(model.src_embed.parameters())
                        + list(model.encoder_layers.parameters()))
            optimizer.add_param_group(
                {"params": unfrozen, "lr": config["train"]["lr"]}
            )

        # ---- Train ----
        train_loss = run_train(model, train_loader, optimizer=optimizer)

        # ---- Validate ----
        val_out = run_val(model, val_loader, root=root, ignore_index=IGNORE_INDEX)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | "
            f"val {val_out['loss']:.4f} | "
            f"order F1 {val_out['order']['F1']:.4f} | "
            f"sf F1 {val_out['superfamily']['F1']:.4f} | LR {cur_lr:.6f}"
        )

        # ---- WandB ----
        if not args.debugging:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss":   val_out["loss"],
                # Order
                "val_order_acc":        val_out["order"]["accuracy"],
                "val_order_prec":       val_out["order"]["precision"],
                "val_order_rec":        val_out["order"]["recall"],
                "val_order_f1":         val_out["order"]["F1"],
                "val_order_auroc":      val_out["order"]["AUROC"],
                "val_order_auprc":      val_out["order"]["AUPRC"],
                "val_order_normauprc":  val_out["order"]["normAUPRC"],
                # Superfamily
                "val_sf_acc":           val_out["superfamily"]["accuracy"],
                "val_sf_prec":          val_out["superfamily"]["precision"],
                "val_sf_rec":           val_out["superfamily"]["recall"],
                "val_sf_f1":            val_out["superfamily"]["F1"],
                "val_sf_auroc":         val_out["superfamily"]["AUROC"],
                "val_sf_auprc":         val_out["superfamily"]["AUPRC"],
                "val_sf_normauprc":     val_out["superfamily"]["normAUPRC"],
                "lr": cur_lr,
            })

        # ---- Checkpoint best ----
        # Track by superfamily F1 (deepest level) — change if you prefer order F1
        score = val_out["superfamily"]["F1"]
        if score > best_val_f1:
            best_val_f1 = score
            if not args.debugging:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "src_vocab_size": VOCAB_SIZE,
                    "d_model": config["model"]["d_model"],
                    "pad_token_id": PAD_TOKEN,
                    "ignore_index": IGNORE_INDEX,
                }, save_path)
                print(f"  ↳ new best (sf F1 {best_val_f1:.4f}) saved to {save_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # ---- Final fold summary ----
    if 'val_out' in locals():
        write_fold_metrics(metrics_save_path, args.fold, val_out, tag=f"final best_sf_F1={best_val_f1:.4f}")
    print("Done.")










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--mode", choices=["classify_order", "classify_sf"], required=True) 
    
    # Pretrained & Fold Config
    parser.add_argument("--pretrained_mlm", type=str, required=True, help="Path to trained MLM .pt file")
    parser.add_argument("--fold", type=int, required=True, help="The current fold number (e.g., 1, 2, 3)")
    parser.add_argument("--labels_path",  type=str, required=True, help="TSV with columns: seq_x, labels, dataset, fold_0..fold_4.")
    
    # Data Config
    # ---- Persistence-image directories (at least one required) ----
    parser.add_argument("--mer4_dir",     type=str, default=None, help="Dir of 4-mer persistence-image .tar.gz files.")
    parser.add_argument("--mer8_dir",     type=str, default=None, help="Dir of 8-mer persistence-image .tar.gz files.")
    parser.add_argument("--mer14_dir",    type=str, default=None, help="Dir of 14-mer persistence-image .tar.gz files.")
    parser.add_argument("--mer20_dir",    type=str, default=None, help="Dir of 20-mer persistence-image .tar.gz files.")

    # parser.add_argument('--missing_lookup_dir', type=str, required=True, help='Path to pickle file containing the missing lookup dictionary')

    parser.add_argument('--save_dir', type=str, default="/staging/kkumari/", help='Directory to save model checkpoints')
    
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name override')
    parser.add_argument('--wandb_team', type=str, default=None, help='WandB team/entity name override')
    parser.add_argument('--wandb_dir', type=str, default=None, help='WandB log directory override')
    parser.add_argument('--run_name', type=str, default="zzformer", help='WandB run name')
    
    args = parser.parse_args()


    # Sanity: at least one PI dir
    if not any([args.mer4_dir, args.mer8_dir, args.mer14_dir, args.mer20_dir]):
        parser.error("At least one of --mer4_dir / --mer8_dir / --mer14_dir / --mer20_dir is required.")


    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)


