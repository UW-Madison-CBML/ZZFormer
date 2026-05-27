import os
import gc
from turtle import pd
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
# from data.dataloader import SequenceDataset_with_topology, SequenceDataset_with_topology_missingflagged

from data.dataloader_CA import HierarchicalPersistenceDataset, hierarchical_collate
from utils import calcmetrics_torcheval_multiclass_filtered
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================================================================================================================
# TRAINING / VALIDATION LOOP
# ==============================================================================================================================================================================

def run_train_order(
    model,
    dataloader,
    optimizer=None,
    ):

    total_loss = 0.0

    model.train(True)
    # torch.set_grad_enabled(True)

    for tokens, src_key_padding_mask, order_labels, _, mer2, missing2, mer4, missing4, mer8, missing8  in dataloader:
        tokens = tokens.to(DEVICE, non_blocking=True)                    # (B, L)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)  # (B, L)
        order_labels = order_labels.to(DEVICE, non_blocking=True)        # (B,)
        # mer2: [32, 512], mer4: [32, 512], mer8: [32, 512]
        # Stack into the list the model expects
        topology_latent_stack = [
            mer2.to(DEVICE, non_blocking=True),
            mer4.to(DEVICE, non_blocking=True),
            mer8.to(DEVICE, non_blocking=True),
        ]
        topology_mask_stack = [
            missing2.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing4.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing8.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            ]


        optimizer.zero_grad()
        # ---------------- Forward ----------------
        outputs = model(tokens, src_key_padding_mask, order_labels, topology_latent_stack=topology_latent_stack, topology_mask=topology_mask_stack)
        loss = outputs['total_loss']

        # ---------------- Backward (Train only) ----------------
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(dataloader)



    return avg_loss




def run_train_sf(
    model,
    dataloader,
    optimizer=None,
    ):

    total_loss = 0.0

    model.train(True)
    # torch.set_grad_enabled(True)

    for tokens, src_key_padding_mask, _, superfamily_labels, mer2, missing2, mer4, missing4, mer8, missing8  in dataloader:
        tokens = tokens.to(DEVICE, non_blocking=True)                    # (B, L)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)  # (B, L)
        superfamily_labels = superfamily_labels.to(DEVICE, non_blocking=True)      # (B,)
        # mer2: [32, 512], mer4: [32, 512], mer8: [32, 512]
        # Stack into the list the model expects
        topology_latent_stack = [
            mer2.to(DEVICE, non_blocking=True),
            mer4.to(DEVICE, non_blocking=True),
            mer8.to(DEVICE, non_blocking=True),
        ]
        topology_mask_stack = [
            missing2.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing4.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing8.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
        ]



        optimizer.zero_grad()
        # ---------------- Forward ----------------
        outputs = model(tokens, src_key_padding_mask, superfamily_labels,topology_latent_stack=topology_latent_stack, topology_mask=topology_mask_stack)
        loss = outputs['total_loss']

        # ---------------- Backward (Train only) ----------------
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(dataloader)



    return avg_loss








def run_val_order(
    model,
    dataloader,
    ignore_index=-100,
    logits_key=None,
    ):
    
    # Store raw tensors, not numpy arrays!
    all_logits = []
    all_labels = []
    
    model.eval()
    total_loss = 0.0
    # class_loss = 0.0
    with torch.no_grad():
        for tokens, src_key_padding_mask, order_labels, _, mer2, missing2, mer4, missing4, mer8, missing8  in dataloader:
            tokens = tokens.to(DEVICE, non_blocking=True)                    # (B, L)
            src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)  # (B, L)
            order_labels = order_labels.to(DEVICE, non_blocking=True)        # (B,)
            # mer2: [32, 512], mer4: [32, 512], mer8: [32, 512]
            # Stack into the list the model expects
            topology_latent_stack = [
                mer2.to(DEVICE, non_blocking=True),
                mer4.to(DEVICE, non_blocking=True),
                mer8.to(DEVICE, non_blocking=True),
            ]
            topology_mask_stack = [
            missing2.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing4.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing8.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            ]

            # ---------------- Forward ----------------
            outputs = model(tokens, src_key_padding_mask, order_labels, topology_latent_stack=topology_latent_stack, topology_mask=topology_mask_stack)
            loss = outputs['total_loss']
            # class_loss += outputs[loss_key].item()

            logits = outputs[logits_key]

            total_loss += loss.item()

            labels = order_labels

                
            
            # Append the raw PyTorch tensors (moved to CPU to save GPU memory)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    
    print(f"Validation Total Loss: {avg_loss:.4f}") #| Validation Class Loss: {class_loss / len(dataloader):.4f}")

    # Now torch.cat will work perfectly because they are lists of PyTorch Tensors
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate metrics using your custom torcheval function
    class_metrics = calcmetrics_torcheval_multiclass_filtered(
        y_true=all_labels,
        y_pred_logits=all_logits,
        ignore_index=ignore_index, average='macro'
    )

    # Assuming your function returns a dictionary with these keys:
    acc = class_metrics['accuracy']
    prec = class_metrics['precision']
    rec = class_metrics['recall']
    f1 = class_metrics['F1']

    return avg_loss, acc, prec, rec, f1







def run_val_sf(
    model,
    dataloader,
    ignore_index=-100,
    logits_key=None,
    ):
    
    # Store raw tensors, not numpy arrays!
    all_logits = []
    all_labels = []
    
    model.eval()
    total_loss = 0.0
    # class_loss = 0.0
    with torch.no_grad():
        for tokens, src_key_padding_mask, _, superfamily_labels, mer2, missing2, mer4, missing4, mer8, missing8  in dataloader:
            tokens = tokens.to(DEVICE, non_blocking=True)                    # (B, L)
            src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)  # (B, L)
            superfamily_labels = superfamily_labels.to(DEVICE, non_blocking=True)      # (B,)
            # mer2: [32, 512], mer4: [32, 512], mer8: [32, 512]
            # Stack into the list the model expects
            topology_latent_stack = [
                mer2.to(DEVICE, non_blocking=True),
                mer4.to(DEVICE, non_blocking=True),
                mer8.to(DEVICE, non_blocking=True),
            ]
            topology_mask_stack = [
            missing2.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing4.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            missing8.unsqueeze(1).to(DEVICE, non_blocking=True),  # (B,) → (B, 1)
            ]


            # ---------------- Forward ----------------
            outputs = model(tokens, src_key_padding_mask, superfamily_labels,topology_latent_stack=topology_latent_stack, topology_mask=topology_mask_stack)
            loss = outputs['total_loss']
            # class_loss += outputs[loss_key].item()

            logits = outputs[logits_key]

            total_loss += loss.item()
            labels = superfamily_labels
                
            
            # Append the raw PyTorch tensors (moved to CPU to save GPU memory)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    
    print(f"Validation Total Loss: {avg_loss:.4f}") #| Validation Class Loss: {class_loss / len(dataloader):.4f}")

    # Now torch.cat will work perfectly because they are lists of PyTorch Tensors
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate metrics using your custom torcheval function
    class_metrics = calcmetrics_torcheval_multiclass_filtered(
        y_true=all_labels,
        y_pred_logits=all_logits,
        ignore_index=ignore_index, average='macro'
    )

    # Assuming your function returns a dictionary with these keys:
    acc = class_metrics['accuracy']
    prec = class_metrics['precision']
    rec = class_metrics['recall']
    f1 = class_metrics['F1']

    return avg_loss, acc, prec, rec, f1





def run_train_sf_test(model, dataloader, optimizer=None):
    total_loss = 0.0
    model.train(True)

    for batch_idx, (tokens, src_key_padding_mask, _, superfamily_labels,
        mer2, missing2, mer4, missing4, mer8, missing8) in enumerate(dataloader):

        tokens = tokens.to(DEVICE, non_blocking=True)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)
        superfamily_labels = superfamily_labels.to(DEVICE, non_blocking=True)

        topology_latent_stack = [
            mer2.to(DEVICE, non_blocking=True),
            mer4.to(DEVICE, non_blocking=True),
            mer8.to(DEVICE, non_blocking=True),
        ]
        topology_mask_stack = [
            missing2.unsqueeze(1).to(DEVICE, non_blocking=True),
            missing4.unsqueeze(1).to(DEVICE, non_blocking=True),
            missing8.unsqueeze(1).to(DEVICE, non_blocking=True),
        ]

        # ============== DEBUG PRINTS (first 3 batches) ==============
        if batch_idx < 3:
            print(f"\n{'='*60}")
            print(f"BATCH {batch_idx}")
            print(f"{'='*60}")
            print(f"tokens:    shape={tokens.shape}, dtype={tokens.dtype}")
            print(f"pad_mask:  shape={src_key_padding_mask.shape}, dtype={src_key_padding_mask.dtype}")
            print(f"sf_labels: shape={superfamily_labels.shape}, dtype={superfamily_labels.dtype}")
            print(f"sf_labels unique: {superfamily_labels.unique().tolist()}")

            for i, (vec, mask, name) in enumerate(zip(
                topology_latent_stack, topology_mask_stack, ['mer2', 'mer4', 'mer8']
            )):
                num_missing = mask.sum().item()
                num_total = mask.shape[0]
                print(f"\n  {name}:")
                print(f"    vec:     shape={vec.shape}, dtype={vec.dtype}")
                print(f"    mask:    shape={mask.shape}, dtype={mask.dtype}")
                print(f"    missing: {num_missing}/{num_total} ({100*num_missing/num_total:.1f}%)")
                print(f"    vec min={vec.min().item():.4f}, max={vec.max().item():.4f}, mean={vec.mean().item():.4f}")
                print(f"    vec has nan: {torch.isnan(vec).any().item()}")
                print(f"    vec has inf: {torch.isinf(vec).any().item()}")
                # Check if "present" samples are all zeros (would indicate bad loading)
                present_mask = ~mask.squeeze(1)  # (B,) True = present
                if present_mask.any():
                    present_vecs = vec[present_mask]
                    all_zero_present = (present_vecs.abs().sum(dim=1) == 0).sum().item()
                    print(f"    present samples that are all-zero: {all_zero_present}/{present_mask.sum().item()}")
                else:
                    print(f"    ALL samples missing for this k-mer!")

            # Check if all 3 masks are True for any sample (fully missing topology)
            all_missing = (missing2 & missing4 & missing8)
            print(f"\n  Samples missing ALL 3 k-mers: {all_missing.sum().item()}/{len(all_missing)}")
        # ============== END DEBUG ==============

        optimizer.zero_grad()
        outputs = model(tokens, src_key_padding_mask, superfamily_labels,
                        topology_latent_stack=topology_latent_stack,
                        topology_mask=topology_mask_stack)
        loss = outputs['total_loss']

        # ============== DEBUG: Check loss ==============
        if batch_idx < 3:
            print(f"\n  loss = {loss.item()}")
            print(f"  loss is nan: {torch.isnan(loss).item()}")
            logits = outputs['sf_logits']
            print(f"  logits: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            print(f"  logits has nan: {torch.isnan(logits).any().item()}")

        if torch.isnan(loss):
            print(f"\n  !!! NAN DETECTED AT BATCH {batch_idx} — stopping !!!")
            # Extra debug: check intermediate values
            with torch.no_grad():
                h = model.src_embed(tokens) * (model.d_model ** 0.5)
                h = model.pos_encoder(h)
                print(f"  after embed: nan={torch.isnan(h).any().item()}, min={h.min().item():.4f}, max={h.max().item():.4f}")

                for layer_idx, (self_attn, cross_attn, proj, topo_latent, topo_mask) in enumerate(zip(
                    model.encoder_layers, model.cross_attn_layers,
                    model.kmer_projections, topology_latent_stack, topology_mask_stack
                )):
                    h = self_attn(h, src_key_padding_mask=src_key_padding_mask)
                    print(f"  after self_attn[{layer_idx}]: nan={torch.isnan(h).any().item()}, min={h.min().item():.4f}, max={h.max().item():.4f}")

                    context = proj(topo_latent).unsqueeze(1)
                    print(f"  context[{layer_idx}]: nan={torch.isnan(context).any().item()}, min={context.min().item():.4f}, max={context.max().item():.4f}")
                    print(f"  mask[{layer_idx}] all True: {topo_mask.all().item()}, any True: {topo_mask.any().item()}")

                    h = cross_attn(h, context, context_key_padding_mask=topo_mask)
                    print(f"  after cross_attn[{layer_idx}]: nan={torch.isnan(h).any().item()}, min={h.min().item():.4f}, max={h.max().item():.4f}")
            break
        # ============== END DEBUG ==============

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss




















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


def split_data(all_pkl, lookup):
    filtered_pkl = {}

    for seq, metadata in lookup.items():
        if seq in all_pkl:
            # Create a shallow copy so the original all_pkl stays untouched
            entry = all_pkl[seq].copy()
            entry.update(metadata)
            filtered_pkl[seq] = entry
        else:
            # Create a brand new entry for sequences not in all_pkl
            filtered_pkl[seq] = {
                **metadata,
                'persistence_image': np.zeros((128, 128, 5))
            }
    return filtered_pkl

def prep_for_dataloader(all_pkl, n):
    # Get data ready for dataloader
    train = []
    label = []
    label_id = []
    sequences = []
    dataset = []

    for key, values in all_pkl.items():
        train.append(values['persistence_image'])
        label.append(values['labels'])
        label_id.append(values['label_id'])
        sequences.append(key)
        dataset.append(values['dataset'])

    dataset = {f'x{n}': train, 'label_id':label_id, 'Label':label, 'seq_x':sequences, 'dataset':dataset}
    return dataset

def get_train_test(all_pkl, train_lookup, test_lookup):
    train = split_data(all_pkl, train_lookup)
    test = split_data(all_pkl, test_lookup)
    return train, test

def dataloader_ready(train, test, n):
    train = prep_for_dataloader(train, n)
    test = prep_for_dataloader(test, n)
    return train, test

def get_test_train_split(all_pkl, labels, i):
    train = labels[labels[f'fold_{i}'] == 'train']
    test = labels[labels[f'fold_{i}'] == 'test']

    train_lookup = train.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels',f'fold_{i}', 'dataset']].to_dict('index')
    test_lookup = test.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels',f'fold_{i}', 'dataset']].to_dict('index')

    train, test = get_train_test(all_pkl, train_lookup, test_lookup)

    train, test = dataloader_ready(train, test, i)

    return train, test







# ====================================================================================
# MAIN
# ====================================================================================
def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "X": 5}
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN = vocab["PAD"]
    ignore_index = -100
    k_mers=[]
    if args.mer2_dir:
        k_mers.append(2)
    if args.mer4_dir:
        k_mers.append(4)
    if args.mer8_dir:
        k_mers.append(8)

    # 1. Initialize Model

    model = HierarchicalFFNTransformerClassifier(
        src_vocab_size=VOCAB_SIZE, 
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["nhead"], 
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"], 
        num_layers=config["model"]["num_layers"],
        # positional_encoding=config["model"]["positional_encoding"], 
        max_position_embeddings=config["model"]["max_seq_len"],
        pad_token_id=PAD_TOKEN, ignore_index=ignore_index,
        num_orders=config["data"]["num_orders"], 
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        k_mers=k_mers,
        topology_latent_dim=config["topology"]["topological_embedding_dim"],
        # clip_weight=config["topology"]["clip_weight"],
    )


    # 2. Load Pretrained MLM Weights BEFORE Freezing
    if args.pretrained_mlm:
        print(f"Loading pretrained MLM weights from {args.pretrained_mlm}...")
        # checkpoint = torch.load(args.pretrained_mlm, map_location=DEVICE)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model=load_pretrained_mlm_weights(args.pretrained_mlm, model)

    # Freeze the backbone initially
    for param in model.src_embed.parameters():
        param.requires_grad = False
    for param in model.encoder_layers.parameters():
        param.requires_grad = False

    





    model.to(DEVICE)

    # ---------------- WandB ----------------
    if not args.debugging:
        wandb.init(
            name=args.run_name if args.run_name else f"{args.mode}_fold{args.fold}_{args.seed}",
            settings=wandb.Settings(_service_wait=300),
            entity=args.wandb_team if args.wandb_team else config["wandb"]["team"],
            project=args.wandb_project if args.wandb_project else config["wandb"]["project"],
            dir=args.wandb_dir if args.wandb_dir else config["wandb"]["dir"],
            config=config
        )









    labels_path = args.labels_path # 'MnTEdb_labels.tsv'
    pi_dir = args.pi_dir #'MnTEdb/'
    save_name = args.save_name #'MnTEdb'



    # Create tree and id mappings
    ORDER_TO_SUPERFAMILIES={'DIRS': [],
    'Helitron': [],
    'Line': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
    'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
    'PLE': [],
    'Sine': ['SINE', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
    'TIR': ['CACTA', 'MuLE', 'PIF', 'TcMar', 'hAT']}

    # Build tree
    root = build_classification_tree(ORDER_TO_SUPERFAMILIES)
    label_map = build_label_to_node_id(root)

    # Load labels
    labels = pd.read_csv(labels_path, sep='\t')
    labels['label_id'] = labels['labels'].map(label_map)





    # Load labels
    labels = pd.read_csv(labels_path, sep='\t')
    labels['label_id'] = labels['labels'].map(label_map)
    lookup = labels.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels', 'dataset']].to_dict('index')

    pi_paths = [f'{pi_dir}/4mer', f'{pi_dir}/8mer', f'{pi_dir}/14mer', f'{pi_dir}/20mer']

    all_pkl = {}
    train_images = []

    for p in pi_paths:
        pkl_file = get_PI(p)
        pkl_file = update_metadata(lookup, pkl_file)
        
        folds = {}
        for i in range(5):
            train, test = get_test_train_split(pkl_file, labels, i=i)
            folds[f'split_{i}'] = {'train': train, 'test': test}

        all_pkl[p] = folds # for given database, all the splits for all kmers

    datasets = list(all_pkl.keys())
    train_dict = {}
    test_dict = {}

    all_results = {} # To save all results






    PI_KEYS = pi_paths   # ['./PI/4mer', './PI/8mer', './PI/14mer', './PI/20mer']
    K_MERS  = [4, 8, 14, 20]
    MAX_SEQ_LEN = 1024   # whatever you use



    train_dataset = HierarchicalPersistenceDataset(
                    train_dict, pi_keys=PI_KEYS, vocab=vocab, max_seq_len=MAX_SEQ_LEN,
                )  # track_missing=False by default

    train_loader = DataLoader(
                    train_dataset, batch_size=32, shuffle=True,
                    collate_fn=hierarchical_collate,   # mask_missing=False by default
                )
    


    val_dataset = HierarchicalPersistenceDataset(
                    val_dict, pi_keys=PI_KEYS, vocab=vocab, max_seq_len=MAX_SEQ_LEN,
                )  # track_missing=False by default


    val_loader = DataLoader(
                    val_dataset, batch_size=32, shuffle=True,
                    collate_fn=hierarchical_collate,   # mask_missing=False by default
                )

    train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["train"]["batchsize"],
                    shuffle=True,
                    num_workers=config["train"]["num_workers"],
                    pin_memory=True,
                    persistent_workers=True,
                    )
    
    val_loader = DataLoader(
                    val_dataset,
                    batch_size=config["train"]["batchsize"],
                    shuffle=True,
                    num_workers=config["train"]["num_workers"],
                    pin_memory=True,
                    persistent_workers=True,
                    )









    # ---------------- Resumption & Saving Setup ----------------
    save_dir = args.save_dir if args.save_dir else config["dir"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    



    # Save file now includes the fold number
    save_path = os.path.join(save_dir, f"{args.mode}_fold{args.fold}_best_{args.run_name}_transformeronly.pt")
    metrics_save_path = os.path.join(save_dir, f"{args.mode}_allfold_metrics.txt")

    logit_key= "order_logits" if args.mode == "classify_order" else "sf_logits"

    # 1. Peek at the checkpoint to get the starting epoch
    init_epoch = 0
    best_val_f1 = -1.0
    if os.path.exists(save_path):
        temp_checkpoint = torch.load(save_path, map_location="cpu")
        init_epoch = temp_checkpoint.get("epoch", -1) + 1
        best_val_f1 = temp_checkpoint.get("best_val_f1", -1.0)
        print(f"Found existing checkpoint for Fold {args.fold}. Preparing to resume from epoch {init_epoch}...")

    # 2. If resuming past the freeze phase, unfreeze the backbone now!
    if init_epoch > 4:
        for param in model.parameters():
            param.requires_grad = True

    # 3. Create the optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config["train"]["lr"])





    if args.mode == "classify_sf":
        label_key= "superfamily_labels"
        loss_key="sf_loss"
    else:
        label_key= "order_labels"
        loss_key="order_loss"

    



    # 4. Now safely load the actual model weights and optimizer states
    if os.path.exists(save_path):
        model, optimizer, _, _ = load_checkpoint(
            model, optimizer, save_path, device=DEVICE, load_optimizer=True, strict=True
        )

        if label_key == "order_labels":
            # --- VALIDATE ---
            val_loss, val_acc, val_prec, val_rec, val_f1 = run_val_order(model, val_loader, ignore_index=ignore_index,logits_key=logit_key) #, loss_key=loss_key)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Val Acc:  {val_acc:.4f}| Val Prec:  {val_prec:.4f}  | Val Rec:  {val_rec:.4f}  | Val F1:  {val_f1:.4f}")

            print("Model Run Finished.\n Validation results for this fold:")
            print(f"Val Loss: | Val Acc:  | Val Prec:  | Val Rec:  | Val F1: ")
            print(f"{val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
            with open(metrics_save_path, "a") as f:
                f.write(f"Fold {args.fold} | {val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}\n")

        else:
            # --- VALIDATE ---
            val_loss, val_acc, val_prec, val_rec, val_f1 = run_val_sf(model, val_loader, ignore_index=ignore_index,logits_key=logit_key) #, loss_key=loss_key)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Val Acc:  {val_acc:.4f}| Val Prec:  {val_prec:.4f}  | Val Rec:  {val_rec:.4f}  | Val F1:  {val_f1:.4f}")

            print("Model Run Finished.\n Validation results for this fold:")
            print(f"Val Loss: | Val Acc:  | Val Prec:  | Val Rec:  | Val F1: ")
            print(f"{val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
            with open(metrics_save_path, "a") as f:
                f.write(f"Fold {args.fold} | {val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}\n")













    # ---------------- Training loop ----------------
    print(f"\n{'='*40}\nSTARTING TRAINING FOR FOLD {args.fold}\n{'='*40}")
    if label_key == "order_labels" and not os.path.exists(save_path):
        for epoch in range(init_epoch, config["train"]["epochs"]):
            # ... Train for 2-3 epochs so the Classifier & Projections converge a bit ...
            if epoch == 4:
                print("\n--- Unfreezing Transformer Backbone for Fine-Tuning ---")
                for param in model.parameters():
                    param.requires_grad = True
                    
                # We MUST add the newly unfrozen layers to the optimizer!
                unfrozen_params = list(model.src_embed.parameters()) + list(model.encoder_layers.parameters())
                # Optional but highly recommended: use a smaller learning rate for the pretrained backbone
                optimizer.add_param_group({'params': unfrozen_params, 'lr': config["train"]["lr"]})

            

            # --- TRAIN ---
            train_loss = run_train_order(model, train_loader, optimizer=optimizer)

            # --- LOGGING ---
            if not args.debugging:
                wandb.log({
                    # "epoch": epoch,
                    "train_loss": train_loss,
                })

            gc.collect()
            torch.cuda.empty_cache()

        if not args.debugging:
            save_data = {
                "model_state_dict": model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "src_vocab_size": VOCAB_SIZE,
                "d_model": config["model"]["d_model"],
                "pad_token_id": PAD_TOKEN,
                "ignore_index": ignore_index
            }
            torch.save(save_data, save_path)

        # --- VALIDATE ---
        val_loss, val_acc, val_prec, val_rec, val_f1 = run_val_order(model, val_loader, ignore_index=ignore_index,logits_key=logit_key) #, loss_key=loss_key)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        print(f"Val Acc:  {val_acc:.4f}| Val Prec:  {val_prec:.4f}  | Val Rec:  {val_rec:.4f}  | Val F1:  {val_f1:.4f}")


        # --- LOGGING ---
        if not args.debugging:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
            })

        print("Model Run Finished.\n Validation results for this fold:")
        print(f"Val Loss: | Val Acc:  | Val Prec:  | Val Rec:  | Val F1: ")
        print(f"{val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
        with open(metrics_save_path, "a") as f:
            f.write(f"Fold {args.fold} | {val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}\n")



    elif os.path.exists(save_path)==False:
        for epoch in range(init_epoch, config["train"]["epochs"]):
            # ... Train for 2-3 epochs so the Classifier & Projections converge a bit ...
            if epoch == 4:
                print("\n--- Unfreezing Transformer Backbone for Fine-Tuning ---")
                for param in model.parameters():
                    param.requires_grad = True
                    
                # We MUST add the newly unfrozen layers to the optimizer!
                unfrozen_params = list(model.src_embed.parameters()) + list(model.encoder_layers.parameters())
                # Optional but highly recommended: use a smaller learning rate for the pretrained backbone
                optimizer.add_param_group({'params': unfrozen_params, 'lr': config["train"]["lr"]})

            

            # --- TRAIN ---
            train_loss = run_train_sf(model, train_loader, optimizer=optimizer)

            # --- LOGGING ---
            if not args.debugging:
                wandb.log({
                    "train_loss": train_loss,
                })





            gc.collect()
            torch.cuda.empty_cache()
        







        if not args.debugging:
            save_data = {
                "model_state_dict": model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "src_vocab_size": VOCAB_SIZE,
                "d_model": config["model"]["d_model"],
                "pad_token_id": PAD_TOKEN,
                "ignore_index": ignore_index
            }
            torch.save(save_data, save_path)

        # --- VALIDATE ---
        val_loss, val_acc, val_prec, val_rec, val_f1 = run_val_sf(model, val_loader, ignore_index=ignore_index,logits_key=logit_key) #, loss_key=loss_key)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        print(f"Val Acc:  {val_acc:.4f}| Val Prec:  {val_prec:.4f}  | Val Rec:  {val_rec:.4f}  | Val F1:  {val_f1:.4f}")


        # --- LOGGING ---
        if not args.debugging:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
            })

        print("Model Run Finished.\n Validation results for this fold:")
        print(f"Val Loss: | Val Acc:  | Val Prec:  | Val Rec:  | Val F1: ")
        print(f"{val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
        with open(metrics_save_path, "a") as f:
            f.write(f"Fold {args.fold} | {val_loss:.4f} | {val_acc:.4f} | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}\n")



























if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["classify_order", "classify_sf"], required=True) 
    
    # Pretrained & Fold Config
    parser.add_argument("--pretrained_mlm", type=str, required=True, help="Path to trained MLM .pt file")
    parser.add_argument("--fold", type=int, required=True, help="The current fold number (e.g., 1, 2, 3)")
    
    # Data Config
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data pickle file for this fold')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data pickle file for this fold')
    parser.add_argument('--mer2_dir', type=str, required=False, help='Path to 2mer topological embeddings')
    parser.add_argument('--mer4_dir', type=str, required=False, help='Path to 4mer topological embeddings')
    parser.add_argument('--mer8_dir', type=str, required=False, help='Path to 8mer topological embeddings')
    parser.add_argument('--missing_lookup_dir', type=str, required=True, help='Path to pickle file containing the missing lookup dictionary')

    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model checkpoints')
    
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name override')
    parser.add_argument('--wandb_team', type=str, default=None, help='WandB team/entity name override')
    parser.add_argument('--wandb_dir', type=str, default=None, help='WandB log directory override')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)



'''

fold=0
testfile="/Users/kkumari/Desktop/TDA/Sequence_embeddings/CREATE/pickles/mntedb/fold_0_test_seqlabels.pkl"
trainfile="/Users/kkumari/Desktop/TDA/Sequence_embeddings/CREATE/pickles/mntedb/fold_0_train_seqlabels.pkl"

python retrain_ffn.py \
    --config "./config/ffn_config.yml" \
    --mode "classify_order" \
    --pretrained_mlm "./tmp/mlm_best.pt" \
    --fold $fold \
    --train_dir $trainfile \
    --val_dir $testfile \
    --mer2_dir "./data/2mer_embeddings_100epochs_mntedb.tsv" \
    --mer4_dir "./data/4mer_embeddings_100epochs_mntedb.tsv" \
    --mer8_dir "./data/8mer_embeddings_100epochs_mntedb.tsv" \
    --save_dir "./tmp/" \
    --wandb_project "zzformer" \
    --wandb_team 'kkumari-university-of-wisconsin-madison' \
    --run_name "ZZTEST_sf_fold0"


    





Some old ways of loading data-


    # # Now pass the clean lists into the Dataset wrapper
    # train_dataset = SequenceDataset(train_seqs, 
    #                                 max_seq_len=config["model"]["max_seq_len"], 
    #                                 mode=args.mode,        
    #                                 mer2_path=args.mer2_dir,  # <-- ADD THIS
    #                                 mer4_path=args.mer4_dir,  # <-- ADD THIS
    #                                 mer8_path=args.mer8_dir,   # <-- ADD THIS )
    #                                 ignore_index=ignore_index,
    #                             )

    # val_dataset = SequenceDataset(val_seqs, 
    #                               max_seq_len=config["model"]["max_seq_len"], 
    #                               mode=args.mode,        
    #                               mer2_path=args.mer2_dir,  # <-- ADD THIS
    #                               mer4_path=args.mer4_dir,  # <-- ADD THIS
    #                               mer8_path=args.mer8_dir,   # <-- ADD THIS )
    #                               ignore_index=ignore_index,
    #                             )



    # train_loader = DataLoader(
    #     train_dataset, batch_size=config["train"]["batchsize"], shuffle=True,
    #     num_workers=config["train"]["num_workers"], pin_memory=True,  # Fast CPU-to-GPU
    #     collate_fn=partial(collate_fn, pad_token_id=PAD_TOKEN, ignore_index=ignore_index) 
    # )
    
    # val_loader = DataLoader(
    #     val_dataset, batch_size=config["train"]["batchsize"], shuffle=False, # Shuffle not needed for val
    #     num_workers=config["train"]["num_workers"],pin_memory=True,  # Fast CPU-to-GPU
    #     collate_fn=partial(collate_fn, pad_token_id=PAD_TOKEN, ignore_index=ignore_index) 
    # )


    # Now pass the clean lists into the Dataset wrapper
    # train_dataset = SequenceDataset_noCollate(train_seqs, 
    #                                 max_seq_len=config["model"]["max_seq_len"], 
    #                                 mode=args.mode,        
    #                                 mer2_path=args.mer2_dir,  # <-- ADD THIS
    #                                 mer4_path=args.mer4_dir,  # <-- ADD THIS
    #                                 mer8_path=args.mer8_dir,   # <-- ADD THIS )
    #                                 ignore_index=ignore_index,
    #                             )
    # val_dataset = SequenceDataset_noCollate(val_seqs, 
    #                               max_seq_len=config["model"]["max_seq_len"], 
    #                               mode=args.mode,        
    #                               mer2_path=args.mer2_dir,  # <-- ADD THIS
    #                               mer4_path=args.mer4_dir,  # <-- ADD THIS
    #                               mer8_path=args.mer8_dir,   # <-- ADD THIS )
    #                               ignore_index=ignore_index,
    #                             )

    # # Create standard dataloader
    # raw_train_loader = DataLoader(train_dataset, batch_size=config["train"]["batchsize"], shuffle=True, num_workers=config["train"]["num_workers"])

    # # Wrap it in the Device loader! 
    # train_loader = DeviceDataLoader(raw_train_loader, DEVICE)


    # # Create standard dataloader
    # raw_val_loader = DataLoader(val_dataset, batch_size=config["train"]["batchsize"], shuffle=True, num_workers=config["train"]["num_workers"])

    # # Wrap it in the Device loader! 
    # val_loader = DeviceDataLoader(raw_val_loader, DEVICE)


'''
