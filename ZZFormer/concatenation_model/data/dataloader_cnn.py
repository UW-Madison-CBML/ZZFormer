"""
Dataset for `HierarchicalLongformerClassifier`.

Returns RAW persistence images per k-mer instead of pre-computed topology
latents. The per-k-mer CNN heads inside the model now turn the images into
latents on the fly. Images are loaded from on-disk tar.gz archives keyed by
sequence string and matched against `data_dict['sequence']`.
"""

import os
import glob
import pickle
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset

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













# ----------------------------------------------------------------------
# Loader helpers (same convention as the CNN pipeline)
# ----------------------------------------------------------------------
def get_PI(path):
    """Load all `*.pkl` payloads from `*.tar.gz` archives in `path`."""
    all_pkl = {}
    image_files = glob.glob(f"{path}/*tar.gz")
    for file in image_files:
        with tarfile.open(file, "r:gz") as tar:
            all_files = tar.getnames()
            pkl_path = next((f for f in all_files if f.endswith('.pkl')), None)
            if pkl_path:
                print(f"Loading {pkl_path} from {file}...")
                member = tar.getmember(pkl_path)
                f = tar.extractfile(member)
                data = pickle.load(f)
                all_pkl = all_pkl | data
    return all_pkl


def load_pi_lookups(pi_dir, k_mers=(4, 8, 14, 20)):
    """
    Load persistence images for each k-mer from `{pi_dir}/{k}mer/*.tar.gz`.

    Returns a list aligned with `k_mers`. Each entry is a dict
        { sequence_string : { 'persistence_image': np.ndarray(H, W, C), ... } }.
    Load these ONCE outside your fold loop and pass them in via `pi_lookups`
    to avoid re-reading the tar.gz archives 5x.
    """
    print(f"Loading persistence images from:")
    print([os.path.join(pi_dir, f'{k}mer') for k in k_mers])
    return [get_PI(os.path.join(pi_dir, f"{k}mer")) for k in k_mers]


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class TopoDataset(Dataset):
    """
    Hierarchical dataset for `HierarchicalLongformerClassifier`.

    Expects `data_dict` with:
        'sequence' : list[str]      raw nucleotide strings (also the PI key)
        'labels'   : list           human-readable labels (passthrough)
        'label_id' : list[int]      LEAF node ids into the SoftmaxNode tree

    Persistence images are supplied via EITHER:
        - `pi_lookups` : pre-loaded list of 4 dicts                 (recommended)
        - `pi_dir`     : base dir with `{k}mer/` subdirs of tar.gz   (one-shot)

    Per-sample output:
        input_ids        : (L,)        long
        attention_mask   : (L,)        long, 1=valid, 0=pad
        target_node_ids  : ()          long
        topology_images  : list of 4 tensors, each (C, H, W) float32,
                           in k-mer order [4, 8, 14, 20]
        labels           : passthrough
        sequence         : passthrough
        topo_mask (opt)  : (4,) float, 1=non-zero PI, 0=zero fallback

    Sequences without a matching PI are zero-filled with `pi_shape`,
    matching the CNN-side `PersistenceDataset` convention.
    """

    def __init__(
        self,
        data_dict,
        max_seq_len,
        pi_dir=None,
        pi_lookups=None,
        k_mers=(4, 8, 14, 20),
        pi_shape=(128, 128, 5),
        mask=False,
    ):
        self.sequences   = data_dict['sequence']
        self.labels      = data_dict['labels']
        self.label_ids   = data_dict['label_id']
        self.max_seq_len = max_seq_len
        self.mask        = mask
        self.k_mers      = list(k_mers)
        self.pi_shape    = tuple(pi_shape)

        assert (
            len(self.sequences) == len(self.label_ids) == len(self.labels)
        ), "sequence / labels / label_id must all have the same length."

        # ---- Resolve per-k-mer PI lookups ----
        if pi_lookups is None:
            assert pi_dir is not None, (
                "Provide either `pi_lookups` (pre-loaded) or `pi_dir`."
            )
            pi_lookups = load_pi_lookups(pi_dir, self.k_mers)
        assert len(pi_lookups) == len(self.k_mers), (
            f"Expected {len(self.k_mers)} PI lookups (one per k-mer), "
            f"got {len(pi_lookups)}."
        )
        self.pi_lookups = pi_lookups

        # ---- Tokenise once ----
        self.input_ids, self.attention_masks = self._tokenize_all()
        self.targets = torch.tensor(self.label_ids, dtype=torch.long)

        # ---- Quick report on missing PIs ----
        for k, lookup in zip(self.k_mers, self.pi_lookups):
            missing = sum(
                1 for s in self.sequences
                if s not in lookup or 'persistence_image' not in lookup[s]
            )
            if missing:
                print(
                    f"  TopoDataset: {missing}/{len(self.sequences)} "
                    f"sequences missing {k}-mer PI -> zero-filled."
                )
        print(
            f"  TopoDataset: {len(self.sequences)} samples | "
            f"L={max_seq_len} | k-mers={self.k_mers} | mask={mask}"
        )

    # ------------------------------------------------------------------
    # Tokenisation (unchanged convention)
    # ------------------------------------------------------------------
    def _tokenize_all(self):
        N, L = len(self.sequences), self.max_seq_len
        input_ids      = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((N, L), dtype=torch.long)
        body_max = L - 2  # reserve BOS / EOS

        for i, seq in enumerate(self.sequences):
            seq  = seq.lower()
            body = [VOCAB.get(c, UNK_TOKEN_ID) for c in seq[:body_max]]
            ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
            input_ids[i, : len(ids)]      = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # PI lookup -> (C, H, W) tensor
    # ------------------------------------------------------------------
    def _get_pi_tensor(self, lookup, seq):
        entry = lookup.get(seq)
        if entry is not None and 'persistence_image' in entry:
            pi = entry['persistence_image']            # (H, W, C)
        else:
            pi = np.zeros(self.pi_shape, dtype=np.float32)
        # Match the CNN-side PersistenceDataset: (H, W, C) -> (C, H, W)
        return (
            torch.as_tensor(pi, dtype=torch.float32)
                 .permute(2, 0, 1)
                 .contiguous()
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # One persistence image per k-mer, in self.k_mers order
        images = [self._get_pi_tensor(lookup, seq) for lookup in self.pi_lookups]

        item = {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_masks[idx],
            "target_node_ids": self.targets[idx],
            # default_collate stacks element-wise:
            # per-sample list[(C,H,W)]*4   --batch-->   list[(B,C,H,W)]*4
            "topology_images": images,
            "labels":          self.labels[idx],
            "sequence":        seq,
        }

        if self.mask:
            item["topo_mask"] = torch.tensor(
                [1.0 if torch.any(img != 0) else 0.0 for img in images],
                dtype=torch.float32,
            )

        return item
    

















