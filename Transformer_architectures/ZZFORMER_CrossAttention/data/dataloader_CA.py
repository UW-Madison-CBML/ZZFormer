import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

vocab        = {"PAD": 0, "a": 1, "c": 2, "g": 3, "t": 4, "x": 5}

VOCAB_SIZE = len(vocab)
PAD_TOKEN = vocab["PAD"]
MASK_TOKEN_ID = VOCAB_SIZE  


LABEL_MAPPINGS ={'Class': {'ClassI': 0, 'ClassII': 1}, 
    'Subclass': {'LTR': 0, 'Non-LTR': 1, 'Sub1': 2, 'Sub2': 3}, 
    'Order': {'DIRS': 0, 'Helitron': 1, 'LINE': 2, 'Line':2, 'LTR': 3, 'PLE': 4, 'SINE': 5,'Sine': 5, 'TIR': 6}, 
    'Superfamily': {'Bel-Pao': 0, 'CACTA': 1, 'CR1': 2, 'Copia': 3, 'DIRS': 4, 'ERV': 5, 'Gypsy': 6, 'Helitron': 7, 'I': 8, 'ID': 9, 'Jockey': 10, 'L1': 11, 'MULE': 12,'MuLE': 12, 'PIF': 13, 'PLE': 14, 'R2': 15, 'RTE': 16, 'Rex1': 17, 'SINE1/7SL': 18, 'SINE2/tRNA': 19, 'SINE3/5S': 20, 'TcMar': 21, 'hAT': 22,"SINE": 23}}

HIERARCHY_LEVELS = ['Class', 'Subclass', 'Order', 'Superfamily']

def read_tsv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'Line': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'Sine': ['SINE', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MuLE', 'PIF', 'TcMar', 'hAT']}

class HierarchicalPersistenceDataset(Dataset):
    """
    See module docstring. By default, no missing-tracking is done — every PI
    (including all-zero placeholders) is treated as a valid context.

    Set `track_missing=True` if you want per-sample missing flags returned,
    so a collate fn can build a `topology_mask` that hides those tokens.
    """

    def __init__(
        self,
        data_dict,
        pi_keys,
        vocab,
        max_seq_len,
        pad_token_id=0,
        unk_token="x",
        track_missing: bool = False,
        zero_tol: float = 1e-12,
    ):
        self.pi_keys = list(pi_keys)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.unk_token = unk_token
        self.track_missing = track_missing
        self.zero_tol = zero_tol

        self.seq_x    = list(data_dict["seq_x"])
        self.label    = list(data_dict["Label"])
        self.label_id = list(data_dict["label_id"])
        self.dataset  = list(data_dict["dataset"])

        N = len(self.seq_x)
        for k in self.pi_keys:
            assert len(data_dict[k]) == N, f"{k} length mismatch ({len(data_dict[k])} vs {N})"
        self.pi_arrays = [data_dict[k] for k in self.pi_keys]

        self.all_tokens = self._encode_all(self.seq_x)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_label_ids = torch.tensor(self.label_id, dtype=torch.long)

    # ------------------------------------------------------------------ utils
    def _encode_all(self, sequences):
        encoded = []
        unk_id = self.vocab.get(self.unk_token, 0)
        for seq in sequences:
            ids = [self.vocab.get(c, unk_id) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _to_pi_tensor(self, arr):
        """(128,128,5) numpy → (5,128,128) float tensor."""
        t = torch.as_tensor(arr, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 5:   # HWC → CHW
            t = t.permute(2, 0, 1).contiguous()
        return t

    # ------------------------------------------------------------------ API
    def __len__(self):
        return len(self.seq_x)

    def __getitem__(self, idx):
        pi_list = [self._to_pi_tensor(arr[idx]) for arr in self.pi_arrays]

        item = {
            "tokens": self.all_tokens[idx],
            "src_key_padding_mask": self.all_padding_masks[idx],
            "target_node_id": self.all_label_ids[idx],
            "pi_list": pi_list,
            "seq_x":   self.seq_x[idx],
            "label":   self.label[idx],
            "dataset": self.dataset[idx],
        }

        if self.track_missing:
            item["pi_missing"] = [
                torch.tensor(bool(pi.abs().max() < self.zero_tol))
                for pi in pi_list
            ]

        return item

















def hierarchical_collate(batch, mask_missing: bool = False, ctx_tokens: int = 5):
    """
    Default (`mask_missing=False`): `topology_mask` is None — all PI tokens,
    including all-zero placeholders, are attended to normally.

    `mask_missing=True`: builds a per-k-mer (B, ctx_tokens) bool mask that
    hides tokens for samples flagged as missing. Requires the dataset to be
    constructed with `track_missing=True`. To avoid all-True rows (which would
    NaN out softmax), one key position is kept "alive" per sample.
    """
    tokens               = torch.stack([b["tokens"]              for b in batch], dim=0)
    src_key_padding_mask = torch.stack([b["src_key_padding_mask"] for b in batch], dim=0)
    target_node_ids      = torch.stack([b["target_node_id"]      for b in batch], dim=0)

    num_kmers = len(batch[0]["pi_list"])
    topology_latent_stack = [
        torch.stack([b["pi_list"][k] for b in batch], dim=0)
        for k in range(num_kmers)
    ]

    if not mask_missing:
        topology_mask = None
    else:
        assert "pi_missing" in batch[0], (
            "mask_missing=True requires the dataset to be built with "
            "track_missing=True."
        )
        topology_mask = []
        for k in range(num_kmers):
            missing = torch.stack([b["pi_missing"][k] for b in batch], dim=0)  # (B,)
            if not missing.any():
                topology_mask.append(None)
                continue
            B = missing.shape[0]
            mask = missing.view(B, 1).expand(B, ctx_tokens).clone()  # True = ignore
            mask[:, 0] = False  # keep one alive token to prevent all-masked rows
            topology_mask.append(mask)

    return {
        "tokens": tokens,
        "src_key_padding_mask": src_key_padding_mask,
        "target_node_ids": target_node_ids,
        "topology_latent_stack": topology_latent_stack,
        "topology_mask": topology_mask,
        "seq_x":   [b["seq_x"]   for b in batch],
        "label":   [b["label"]   for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


# def hierarchical_collate(batch):
#     """
#     Stacks per-sample lists into per-k-mer batched tensors,
#     matching what HierarchicalFFNTransformerClassifier.forward expects:

#         topology_latent_stack: list of length num_cross_layers,
#                                each tensor (B, 5, 128, 128)
#         topology_mask:         list of length num_cross_layers,
#                                each tensor (B,) bool   OR  None
#     """
#     tokens               = torch.stack([b["tokens"]              for b in batch], dim=0)
#     src_key_padding_mask = torch.stack([b["src_key_padding_mask"] for b in batch], dim=0)
#     target_node_ids      = torch.stack([b["target_node_id"]      for b in batch], dim=0)

#     num_kmers = len(batch[0]["pi_list"])
#     topology_latent_stack = [
#         torch.stack([b["pi_list"][k] for b in batch], dim=0)        # (B,5,128,128)
#         for k in range(num_kmers)
#     ]
#     topology_missing_per_kmer = [
#         torch.stack([b["pi_missing"][k] for b in batch], dim=0)     # (B,)
#         for k in range(num_kmers)
#     ]

#     # Build a `topology_mask` per k-mer in the shape MultiheadAttention wants:
#     # (B, S_ctx) bool, True = padding/ignore.
#     # PersistenceImageEncoder5Tokens emits 5 tokens, so broadcast the
#     # per-sample missing flag across those 5 positions.
#     # NOTE: rows that are ALL True will produce NaNs in attention — so we
#     # instead leave the mask = None for any k-mer where no sample is missing,
#     # and for missing samples we keep one "alive" token to avoid all-masked rows.
#     topology_mask = []
#     S_CTX = 5  # PersistenceImageEncoder5Tokens
#     for missing in topology_missing_per_kmer:
#         if not missing.any():
#             topology_mask.append(None)
#             continue
#         B = missing.shape[0]
#         mask = missing.view(B, 1).expand(B, S_CTX).clone()  # (B, 5) True if missing
#         # Keep first token "alive" for missing rows to avoid all-True softmax NaN
#         mask[:, 0] = False
#         topology_mask.append(mask)

#     return {
#         "tokens": tokens,
#         "src_key_padding_mask": src_key_padding_mask,
#         "target_node_ids": target_node_ids,
#         "topology_latent_stack": topology_latent_stack,
#         "topology_mask": topology_mask,
#         # metadata (kept as plain lists)
#         "seq_x":   [b["seq_x"]   for b in batch],
#         "label":   [b["label"]   for b in batch],
#         "dataset": [b["dataset"] for b in batch],
#     }