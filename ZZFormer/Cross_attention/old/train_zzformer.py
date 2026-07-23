import os
import gc
import yaml
import argparse
import random
import pickle
import system as sys

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# from data.dataloader_cnn import TopoDataset, load_pi_lookups
from data.dataloader_cnn import TopoDataset # Dataloader
from model.topology_encoder import TopologyEncoder # CNN
from model.ZZFormer_CAatend import HierarchicalLongformerClassifier,build_classification_tree # ZZFormer CA model


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

# Test set
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = sys.argv[1]
save_dir = f"/staging/s/svaren/072026/cross_attention/{dataset}/"


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

##############################################################################
# Helper functions
def extract_chunk_ids(filepath):
    match = re.search(r"chunk_(\d+)_(\d+)", filepath)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

def load_npz(npz_path, load_meta=False):
    loaded = np.load(npz_path, allow_pickle=True)
    arrays = [loaded[key] for key in sorted(loaded.files) if key.startswith("array_")]
    if load_meta:
        metadata = loaded["metadata"].tolist()
        return arrays, metadata
    return arrays


# Load config file
with open("longformer_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
train_cfg = cfg["train"]
wandb_cfg = cfg["wandb"]

##############################################################################
# Classification tree
classification_tree = build_classification_tree(
        ORDER_TO_SUPERFAMILIES,
        label_smoothing = cfg.get("label_smoothing", 0.0),
        gamma           = cfg.get("gamma", 0.0),
    )

# Build tree
# root = build_classification_tree(ORDER_TO_SUPERFAMILIES)
label_map = build_label_to_node_id(classification_tree)

##############################################################################
# Load npz
all_data = defaultdict(list)

mer4 = sorted(glob.glob(f"{pi_dir}/*_4mer.npz"), key=extract_chunk_ids)
mer8 = sorted(glob.glob(f"{pi_dir}/*_8mer.npz"), key=extract_chunk_ids)
mer14 = sorted(glob.glob(f"{pi_dir}/*_14mer.npz"), key=extract_chunk_ids)
mer20 = sorted(glob.glob(f"{pi_dir}/*_20mer.npz"), key=extract_chunk_ids)

# Load whold dataset
# Metadata: sequence, labels, order, superfamily, fold_0, fold_1, fold_2, fold_3, fold_4
for a, b, c, d in zip(mer4, mer8, mer14, mer20):
    arr4 = load_npz(a, load_meta=False)
    arr8 = load_npz(b, load_meta=False)
    arr14 = load_npz(c, load_meta=False)
    arr20, metadata = load_npz(d, load_meta=True) #only need 1 metadata
    sequences = [m[0] for m in metadata]
    labels = [m[1] for m in metadata]
    label_ids = [label_map.get(m[1]) for m in metadata]
    order = [m[2] for m in metadata]
    superfamily = [m[3] for m in metadata]
    fold_0 = [m[4] for m in metadata]
    fold_1 = [m[5] for m in metadata]
    fold_2 = [m[6] for m in metadata]
    fold_3 = [m[7] for m in metadata]
    fold_4 = [m[8] for m in metadata]
    all_data['4mer'].append(arr4)
    all_data['8mer'].append(arr8)
    all_data['14mer'].append(arr14)
    all_data['20mer'].append(arr20)
    all_data['sequences'].append(sequences)
    all_data['labels'].append(labels)
    all_data['label_ids'].append(label_ids)
    all_data['order'].append(order)
    all_data['superfamily'].append(superfamily)
    all_data['fold_0'].append(fold_0)
    all_data['fold_1'].append(fold_1)
    all_data['fold_2'].append(fold_2)
    all_data['fold_3'].append(fold_3)
    all_data['fold_4'].append(fold_4)


##############################################################################
# ZZFormer with cross attention
# Train/test
for fold in range(1):
    train_data = {
        '4mer': [], '8mer': [], '14mer': [], '20mer': [],
        'sequences': [], 'label_ids': [], 'labels': [], 'order': [], 'superfamily': []
    }
    test_data = {
        '4mer': [], '8mer': [], '14mer': [], '20mer': [],
        'sequences': [], 'label_ids': [], 'labels': [], 'order': [], 'superfamily': []
    }
    num_chunks = len(all_data['sequences'])
    all_results = {}
    print(f"Fold {fold}")

    # Metadata: sequence, labels, order, superfamily, fold_0, fold_1, fold_2, fold_3, fold_4
    # Train
    for chunk_idx in range(num_chunks):
        fold_assignments = all_data[f'fold_{fold}'][chunk_idx]
        
        train_indices = [i for i, assignment in enumerate(fold_assignments) if assignment == "train"]
        test_indices  = [i for i, assignment in enumerate(fold_assignments) if assignment != "train"]
        
        if train_indices:
            train_data['4mer'].append(all_data['4mer'][chunk_idx][train_indices])
            train_data['8mer'].append(all_data['8mer'][chunk_idx][train_indices])
            train_data['14mer'].append(all_data['14mer'][chunk_idx][train_indices])
            train_data['20mer'].append(all_data['20mer'][chunk_idx][train_indices])
            
            train_data['sequences'].append([all_data['sequences'][chunk_idx][i] for i in train_indices])
            train_data['labels'].append([all_data['labels'][chunk_idx][i] for i in train_indices])
            train_data['label_ids'].append([all_data['label_ids'][chunk_idx][i] for i in train_indices])
            train_data['order'].append([all_data['order'][chunk_idx][i] for i in train_indices])
            train_data['superfamily'].append([all_data['superfamily'][chunk_idx][i] for i in train_indices])
            
        if test_indices:
            test_data['4mer'].append(all_data['4mer'][chunk_idx][test_indices])
            test_data['8mer'].append(all_data['8mer'][chunk_idx][test_indices])
            test_data['14mer'].append(all_data['14mer'][chunk_idx][test_indices])
            test_data['20mer'].append(all_data['20mer'][chunk_idx][test_indices])
            
            test_data['sequences'].append([all_data['sequences'][chunk_idx][i] for i in test_indices])
            test_data['labels'].append([all_data['labels'][chunk_idx][i] for i in test_indices])
            test_data['label_ids'].append([all_data['label_ids'][chunk_idx][i] for i in test_indices])
            test_data['order'].append([all_data['order'][chunk_idx][i] for i in test_indices])
            test_data['superfamily'].append([all_data['superfamily'][chunk_idx][i] for i in test_indices])


    train_dataset = TopoDataset(
        train_data, 
        max_seq_len=max_seq_len, 
        k_mers=k_mers_list, 
        mask=False
    )
    test_dataset = TopoDataset(
        test_data, 
        max_seq_len=max_seq_len, 
        k_mers=k_mers_list, 
        mask=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg["batchsize"], 
        shuffle=True, 
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg["batchsize"], 
        shuffle=False, 
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )

    # Load model
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
        # pool                    = cfg["model"].get("pool", "bos"),
        # ---- topology cross-attention ----
        topology_latent_dim     = model_cfg.get("topology_latent_dim", 64), #check this??
        k_mers                  = tuple(model_cfg.get("k_mers", (4, 8, 14, 20))),
    )

    # --- Pre-trained MLM backbone ---
    if args.pretrained_mlm:
        model = load_pretrained_longformer_mlm(args.pretrained_mlm, model) # 

    # Freeze ONLY the Longformer backbone — the new k-mer projections,
    # cross-attention layers, BOS-global blocks, and head all stay trainable.
    for p in model.longformer.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params before unfreezing backbone: {n_trainable:,}")

    # Loss defined in the model class
    # criterion = HierarchicalSoftmaxLoss(root=classification_tree) 

    # AdamW to help prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.01) # Might lower lr because finetuning

    # WandB
    wandb.init(
        entity=wandb_cfg["team"],
        project=wandb_cfg["project"],
        dir=wandb_cfg["dir"],
        group=f"fold_{fold}",
        config=cfg
    )

    for epoch in range(train_cfg['epochs']):
        # Freezing weights
        if epoch == FREEZE_EPOCHS:
            print("--- Unfreezing Longformer backbone ---")
            for p in model.longformer.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.longformer.parameters()),
                "lr": cfg["train"]["lr"] * 0.1,   # 10× smaller for fine-tuning
            })

        # run_train.py
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_node_ids = batch["target_node_ids"].to(device)
            topology_images = [img.to(device) for img in batch["topology_images"]]
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_node_ids=target_node_ids,
                topology_images=topology_images
            )
            
            loss = outputs["total_loss"]
            loss.backward()
            
            # Mitigates exploding gradients, helps stability
            # Scale down gradients of parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            running_loss += loss.item()

        wandb.log({"train_loss": train_loss, "epoch": epoch})
        
            
        epoch_loss = running_loss / len(train_loader)
        torch.cuda.empty_cache()

    wandb.finish()

    run_tag = f"{dataset}_{fold}" or f"longformer_topo_{fold}"
    save_path = os.path.join(save_dir, f"{run_tag}.pt")

    print(f"Saving model for fold {fold}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch":            epoch,
        "label_to_node_id": label_to_node_id,
    }, save_path)
    print(f"  ↳ new model at epoch {epoch} saved to {save_path}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreezing: {n_trainable:,}")

    # Test
    val_out = run_val(model, test_loader, classification_tree)
    with open(os.path.join(save_dir, f"{run_tag}_val.yaml"), "w") as f:
        yaml.safe_dump(val_out, f)