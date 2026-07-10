import os
import gc
import yaml
import argparse
import random
import pickle

from model.zzformer_concatenate import HierarchicalLongformerClassifier_Concat,build_classification_tree
from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data.dataloader_cnn import TopoDataset, load_pi_lookups

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












def load_pretrained_longformer_mlm(pretrained_path, classifier_model):
    """
    Transfer the Longformer backbone (embeddings + encoder) from a
    LongformerForMaskedLM checkpoint into a LongformerForSequenceClassification.

    Transferred  : longformer.embeddings.*  +  longformer.encoder.*
    Dropped      : lm_head.*                (MLM-only)
    Random init  : classifier.*             (classification head, new)
    """
    print(f"Loading pretrained MLM weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=DEVICE)
    sd   = ckpt.get("model_state_dict", ckpt)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("longformer."):
            new_sd[k] = v          # entire backbone
        elif k.startswith("lm_head."):
            continue               # drop MLM head
        else:
            print(f"  skipping unrecognized key: {k}")

    missing, unexpected = classifier_model.load_state_dict(new_sd, strict=False)

    # expected_missing   = [k for k in missing if k.startswith(("output_head.", "hierarchical_loss."))]
    expected_missing = [
                        k for k in missing if k.startswith((
                            "output_head.",
                            "hierarchical_loss.",
                            # new modules — randomly initialised when loading an MLM checkpoint:
                            "topology_encoders.",
                            "kmer_projections.",
                            "bos_cross_attn.",
                        ))
                    ]
    
    unexpected_missing = [k for k in missing if k not in expected_missing]

    print(f"\n--- MLM → Classifier transfer ---")
    print(f"  ✓ transferred {len(new_sd)} backbone keys")
    print(f"  classifier head missing (expected, will be randomly initialized): "
          f"{len(expected_missing)}")
    for k in expected_missing:
        print(f"      {k}")

    if unexpected_missing:
        print(f"  ⚠️  UNEXPECTED missing keys ({len(unexpected_missing)}):")
        for k in unexpected_missing[:10]:
            print(f"      {k}")
    if unexpected:
        print(f"  ⚠️  unexpected keys in checkpoint ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"      {k}")

    return classifier_model




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









# ============================================================
# Train / Val
# ============================================================
def _move_topology_images(batch_topo, device):
    """
    `topology_images` is a list[Tensor] of length num_layers, one per k-mer,
    each shaped (B, C, H, W). Move them all to `device`.
    """
    return [t.to(device, non_blocking=True) for t in batch_topo]


def run_train(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids       = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        target_node_ids = batch["target_node_ids"].to(DEVICE, non_blocking=True)
        topology_images = _move_topology_images(batch["topology_images"], DEVICE)

        optimizer.zero_grad()
        out = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            target_node_ids = target_node_ids,
            topology_images = topology_images,
        )
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out["total_loss"].item()
    return total_loss / len(dataloader)


@torch.no_grad()
def run_val(model, dataloader, classification_tree, threshold=0.0):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    for batch in dataloader:
        input_ids       = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
        target_node_ids = batch["target_node_ids"].to(DEVICE, non_blocking=True)
        topology_images = _move_topology_images(batch["topology_images"], DEVICE)

        out = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            target_node_ids = target_node_ids,
            topology_images = topology_images,
        )
        total_loss += out["total_loss"].item()
        all_logits.append(out["logits"].cpu())
        all_targets.append(target_node_ids.cpu())

    avg_loss    = total_loss / len(dataloader)
    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)

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
        return {"p": p, "r": r, "f1": f1, "acc": accuracy_score(true, pred)}

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

    return {
        "val_loss":  avg_loss,
        "order_acc": order_m["acc"], "order_p": order_m["p"],
        "order_r":   order_m["r"],   "order_f1": order_m["f1"],
        "sf_acc":    sf_m["acc"],    "sf_p":    sf_m["p"],
        "sf_r":      sf_m["r"],      "sf_f1":   sf_m["f1"],
    }
# ============================================================
# Main
# ============================================================
def _slice_data_dict(d, n):
    """Take the first n entries of every parallel list in a TopoDataset dict."""
    return {k: v[:n] for k, v in d.items()}

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_tag = args.run_name or "longformer_topo"

    if not args.debugging:
        wandb.init(
            name=args.run_name or f"longformer_fold{args.fold}",
            entity=args.wandb_team    or cfg["wandb"]["team"],
            project=args.wandb_project or cfg["wandb"]["project"],
            dir=args.wandb_dir         or cfg["wandb"]["dir"],
            config=cfg,
        )

    # --- Data ---
    # train.pkl / val.pkl are pre-split TopoDataset data_dicts:
    #   {'sequence': [...], 'labels': [...], 'label_id': [...]}
    # Persistence images are loaded separately from args.pi_dir and matched
    # by sequence string inside TopoDataset, so no fold logic lives here.
    with open(args.train_dir, "rb") as f:
        train_dict = pickle.load(f)
    with open(args.val_dir, "rb") as f:
        val_dict = pickle.load(f)

    # ---- Load persistence images ONCE (shared between train + val) ----
    # Expects pi_dir/4mer/*.tar.gz, pi_dir/8mer/*.tar.gz, etc.
    pi_lookups = load_pi_lookups(args.pi_dir, k_mers=(4, 8, 14, 20))

    if args.debugging:
        train_dict = _slice_data_dict(train_dict, 100)
        val_dict   = _slice_data_dict(val_dict,    50)

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
        data_dict   = train_dict,
        max_seq_len = max_seq_len,
        pi_lookups  = pi_lookups,
        mask        = False,
    )
    val_ds = TopoDataset(
        data_dict   = val_dict,
        max_seq_len = max_seq_len,
        pi_lookups  = pi_lookups,
        mask        = False,
    )
    print("We are using model - HierarchicalLongformerClassifier_Concat")
    # --- Model (Longformer + per-k-mer CNN topology cross-attention) ---
    model = HierarchicalLongformerClassifier_Concat(
        classification_tree     = classification_tree,
        vocab_size              = VOCAB_SIZE,
        d_model                 = cfg["model"]["d_model"],
        n_heads                 = cfg["model"]["nhead"],
        n_heads_cross           = cfg["model"]["nhead_crossatten"],
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
    )

    # --- Pre-trained MLM backbone ---
    if args.pretrained_mlm:
        model = load_pretrained_longformer_mlm(args.pretrained_mlm, model)

    # Freeze ONLY the Longformer backbone — the new per-k-mer topology CNNs,
    # k-mer projections, BOS cross-attention layers, and head stay trainable.
    for p in model.longformer.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    FREEZE_EPOCHS = 3
    print(f"Backbone frozen for first {FREEZE_EPOCHS} epochs")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["train"]["batchsize"],
        shuffle     = True,
        num_workers = cfg["train"]["num_workers"],
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["train"]["batchsize"],
        shuffle     = False,
        num_workers = cfg["train"]["num_workers"],
        pin_memory  = True,
    )

    max_id_in_data = max(max(train_dict['label_id']), max(val_dict['label_id']))
    n_nodes        = len(classification_tree.node_list)
    assert max_id_in_data < n_nodes, (max_id_in_data, n_nodes)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params before unfreezing backbone: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = cfg["train"]["lr"],
    )

    # --- Save dir ---
    save_dir  = args.save_dir or cfg["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_tag}.pt")

    # --- Train ---
    for epoch in range(cfg["train"]["epochs"]):
        if epoch == FREEZE_EPOCHS:
            print("--- Unfreezing Longformer backbone ---")
            for p in model.longformer.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.longformer.parameters()),
                "lr":     cfg["train"]["lr"] * 0.1,   # 10x smaller for fine-tuning
            })

        train_loss = run_train(model, train_loader, optimizer)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f}")

        # if not args.debugging:
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        gc.collect()
        torch.cuda.empty_cache()

    # --- Save ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch":            epoch,
        "label_to_node_id": label_to_node_id,
    }, save_path)
    print(f"  ↳ new model at epoch {epoch} saved to {save_path}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreezing: {n_trainable:,}")

    # --- Validate ---
    val_out = run_val(model, val_loader, classification_tree)
    with open(os.path.join(save_dir, f"{run_tag}_val.yaml"), "w") as f:
        yaml.safe_dump(val_out, f)














# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--fold",          required=True, type=int)
    p.add_argument("--train_dir",     required=True, help="Train pickle: {seq: (order, sf_or_None)}")
    p.add_argument("--val_dir",       required=True, help="Val   pickle: {seq: (order, sf_or_None)}")
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