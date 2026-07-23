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
from model.dataloader_cnn import TopoDataset # Dataloader
from model.topology_encoder import TopologyEncoder # CNN
from model.helper_functions import * # all other helper functions



# from data.dataloader_cnn import TopoDataset, load_pi_lookups


from model.ZZFormer_CAatend import HierarchicalLongformerClassifier,build_classification_tree # ZZFormer CA model



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "MnTEdb"
save_dir = f"/staging/s/svaren/072026/cross_attention/{dataset}/"
# Make the directory if it hasn't been made yet


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
# Load config file
with open("longformer_config.yml", "r") as f:
    cfg = yaml.safe_load(f)



model_cfg = cfg["model"]
train_cfg = cfg["train"]
wandb_cfg = cfg["wandb"]

##############################################################################
# Classification tree
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
pi_dir = "/staging/s/svaren/072026/mntedb"

mer4 = sorted(glob.glob(f"{pi_dir}/*_4mer.npz"), key=extract_chunk_ids)
mer8 = sorted(glob.glob(f"{pi_dir}/*_8mer.npz"), key=extract_chunk_ids)
mer14 = sorted(glob.glob(f"{pi_dir}/*_14mer.npz"), key=extract_chunk_ids)
mer20 = sorted(glob.glob(f"{pi_dir}/*_20mer.npz"), key=extract_chunk_ids)



torch.cuda.empty_cache()
gc.collect()
gc.collect()

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
# for fold in range(1):
fold = 0
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
    # Train and test set idx
    train_indices = [i for i, assignment in enumerate(fold_assignments) if assignment == "train"]
    test_indices  = [i for i, assignment in enumerate(fold_assignments) if assignment != "train"]
    # Split into train and test
    if train_indices:
        train_data['4mer'].extend([all_data['4mer'][chunk_idx][i] for i in train_indices])
        train_data['8mer'].extend([all_data['8mer'][chunk_idx][i] for i in train_indices])
        train_data['14mer'].extend([all_data['14mer'][chunk_idx][i] for i in train_indices])
        train_data['20mer'].extend([all_data['20mer'][chunk_idx][i] for i in train_indices])
        # Metadata
        train_data['sequences'].extend([all_data['sequences'][chunk_idx][i] for i in train_indices])
        train_data['labels'].extend([all_data['labels'][chunk_idx][i] for i in train_indices])
        train_data['label_ids'].extend([all_data['label_ids'][chunk_idx][i] for i in train_indices])
        train_data['order'].extend([all_data['order'][chunk_idx][i] for i in train_indices])
        train_data['superfamily'].extend([all_data['superfamily'][chunk_idx][i] for i in train_indices])
    if test_indices:
        test_data['4mer'].extend([all_data['4mer'][chunk_idx][i] for i in test_indices])
        test_data['8mer'].extend([all_data['8mer'][chunk_idx][i] for i in test_indices])
        test_data['14mer'].extend([all_data['14mer'][chunk_idx][i] for i in test_indices])
        test_data['20mer'].extend([all_data['20mer'][chunk_idx][i] for i in test_indices])
        # Metadata
        test_data['sequences'].extend([all_data['sequences'][chunk_idx][i] for i in test_indices])
        test_data['labels'].extend([all_data['labels'][chunk_idx][i] for i in test_indices])
        test_data['label_ids'].extend([all_data['label_ids'][chunk_idx][i] for i in test_indices])
        test_data['order'].extend([all_data['order'][chunk_idx][i] for i in test_indices])
        test_data['superfamily'].extend([all_data['superfamily'][chunk_idx][i] for i in test_indices])



# Dataloaders
train_dataset = TopoDataset(
    train_data, 
    max_seq_len=model_cfg['max_seq_len'], 
    k_mers=model_cfg['k_mers'], 
    mask=False
)
test_dataset = TopoDataset(
    test_data, 
    max_seq_len=model_cfg['max_seq_len'], 
    k_mers=model_cfg['k_mers'],
    mask=False
)
# Dataloaders
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
    topology_latent_dim     = model_cfg.get("topology_latent_dim", 128), #check this??
    k_mers                  = tuple(model_cfg.get("k_mers", (4, 8, 14, 20))),
)#.to(DEVICE)

# Load topo encoder
topology_encoder = TopologyEncoder(
    n_channels=model_cfg["topology_in_channels"], 
    n_filters=model_cfg["topology_cnn_filters"], 
    model_dim=model_cfg["classifier_hidden_dim"], 
    reduced_persistence=16).to(DEVICE)

# dummy_images = torch.randn(32, 3, 128, 1024).to(device)
# topo_out = topology_encoder(dummy_images) #torch.Size([32, 1024, 256])
# dummy_text_tokens = torch.randn(32, 1024, 256).to(device)
# full_out = model()








# --- Pre-trained MLM backbone ---
pretrained_mlm = "/staging/groups/bhaskar_group/zzformer_hash/weights/longformer_mlm_pretraining_reponly_May28.pt"
# if args.pretrained_mlm:
    # model = load_pretrained_longformer_mlm(args.pretrained_mlm, model) # 

if pretrained_mlm:
    model = load_pretrained_longformer_mlm(pretrained_mlm, model, DEVICE)


# Freeze ONLY the Longformer backbone — the new k-mer projections,
# cross-attention layers, BOS-global blocks, and head all stay trainable.
for p in model.longformer.parameters():
    p.requires_grad = False
# Move model to device
model.to(DEVICE)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params before unfreezing backbone: {n_trainable:,}")
# Loss defined in the model class
# criterion = HierarchicalSoftmaxLoss(root=classification_tree) 
# AdamW to help prevent overfitting



# optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.01) # Might lower lr because finetuning
trainable_params = [p for p in model.parameters() if p.requires_grad] + list(topology_encoder.parameters())
optimizer = torch.optim.AdamW(
    trainable_params, 
    lr=cfg["train"]["lr"],
    weight_decay=train_cfg.get("weight_decay", 0.01)
)

# # WandB
# wandb.init(
#     entity=wandb_cfg["team"],
#     project=wandb_cfg["project"],
#     dir=wandb_cfg["dir"],
#     group=f"fold_{fold}",
#     config=cfg
# )

# def _move_topology(batch_topo, device):
#     """topology_latent_stack is a list[Tensor]; move each to device."""
#     return [t.to(device, non_blocking=True) for t in batch_topo]

# batch.keys(): dict_keys(['input_ids', 'attention_mask', 'target_node_ids', 'topology_images', 'labels', 'sequence'])
# dict_keys(['input_ids', 'attention_mask', 'target_node_ids', 'labels', 'sequence', '4mer_image', '8mer_image', '14mer_image', '20mer_image'])


# Training loop
FREEZE_EPOCHS = 3
for epoch in range(5): #train_cfg['epochs']
    # Freezing weights
    if epoch == FREEZE_EPOCHS:
        print("--- Unfreezing Longformer backbone ---")
        for p in model.longformer.parameters():
            p.requires_grad = True
        optimizer.add_param_group({
            "params": list(model.longformer.parameters()),
            "lr": cfg["train"]["lr"] * 0.1,   # 10× smaller for fine-tuning
        })
    ##### run_train.py #####
    model.train()
    topology_encoder.train()
    total_loss = 0.0
    # Batch
    for batch_idx, batch in enumerate(train_loader):#SWITCH BACK TO TEST_LOADER
        input_ids             = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask        = batch["attention_mask"].to(DEVICE, non_blocking=True)
        target_node_ids       = batch["target_node_ids"].to(DEVICE, non_blocking=True)
        # img4 = batch["4mer_image"].to(device) #torch.Size([195, 3, 128, 1024])
        images = [
            batch[f"{k}mer_image"].to(DEVICE, non_blocking=True) 
            for k in model_cfg.get("k_mers", (4, 8, 14, 20))
        ]
        # Zero grad
        optimizer.zero_grad()
        topology_latent_stack = [topology_encoder(img) for img in images] #[topology_encoder(images[i]) for i in range(len(images))]
        # topo_z = topology_encoder(img4) #torch.Size([195, 1024, 256])
        # topology_latent_stack = [topo_z, topo_z, topo_z, topo_z]
        out = model(
            input_ids             = input_ids,
            attention_mask        = attention_mask,
            target_node_ids       = target_node_ids,
            topology_latent_stack = topology_latent_stack,
        )
        loss = out["total_loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # wandb.log({"train_loss": train_loss, "epoch": epoch})
    epoch_loss = total_loss / len(train_loader)#SWITCH BACK TO TEST_LOADER
    # Save epoch_loss
    torch.cuda.empty_cache()

# wandb.finish()
# Save model
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
# Get number trainable parameters after unfreezing
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params after unfreezing: {n_trainable:,}")
# Validation set and save
val_out = run_val(model, test_loader, classification_tree)
with open(os.path.join(save_dir, f"{run_tag}_val.yaml"), "w") as f:
    yaml.safe_dump(val_out, f)