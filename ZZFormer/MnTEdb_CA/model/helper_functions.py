import os
import gc
import yaml
import argparse
import random
import pickle
import sys
import re
import wandb
import numpy as np

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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

def _move_topology_images(batch_topo, device):
    """
    `topology_images` is a list[Tensor] of length num_layers, one per k-mer,
    each shaped (B, C, H, W). Move them all to `device`.
    """
    return [t.to(device, non_blocking=True) for t in batch_topo]


def load_pretrained_longformer_mlm(pretrained_path, classifier_model, device):
    """
    Transfer the Longformer backbone (embeddings + encoder) from a
    LongformerForMaskedLM checkpoint into a LongformerForSequenceClassification.
    Transferred  : longformer.embeddings.*  +  longformer.encoder.*
    Dropped      : lm_head.*                (MLM-only)
    Random init  : classifier.*             (classification head, new)
    """
    print(f"Loading pretrained MLM weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=device)
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
