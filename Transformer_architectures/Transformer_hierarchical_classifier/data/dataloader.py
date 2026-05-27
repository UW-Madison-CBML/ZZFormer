import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 

import math

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)



LABEL_MAPPINGS ={'Class': {'ClassI': 0, 'ClassII': 1}, 
    'Subclass': {'LTR': 0, 'Non-LTR': 1, 'Sub1': 2, 'Sub2': 3}, 
    'Order': {'DIRS': 0, 'Helitron': 1, 'LINE': 2, 'Line':2, 'LTR': 3, 'PLE': 4, 'SINE': 5,'Sine': 5, 'TIR': 6}, 
    'Superfamily': {'Bel-Pao': 0, 'CACTA': 1, 'CR1': 2, 'Copia': 3, 'DIRS': 4, 'ERV': 5, 'Gypsy': 6, 'Helitron': 7, 'I': 8, 'ID': 9, 'Jockey': 10, 'L1': 11, 'MULE': 12,'MuLE': 12, 'PIF': 13, 'PLE': 14, 'R2': 15, 'RTE': 16, 'Rex1': 17, 'SINE1/7SL': 18, 'SINE2/tRNA': 19, 'SINE3/5S': 20, 'TcMar': 21, 'hAT': 22,"SINE": 23}}

HIERARCHY_LEVELS = ['Class', 'Subclass', 'Order', 'Superfamily']



vocab = {
        "PAD": 0,
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "X": 5
    }
VOCAB_SIZE = len(vocab)
PAD_TOKEN = 0
MASK_TOKEN_ID = VOCAB_SIZE  



ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'LINE': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'SINE': ['ID', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MULE', 'PIF', 'TcMar', 'hAT']}



# MAP_RULES="/I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC,Helitron=RC"



# =============================================================================
# DATASET — adapted for hierarchical targets
# =============================================================================

class HierarchicalSequenceDataset(Dataset):
    """
    Drop-in replacement for SequenceDataset_nocollate_nomer that produces
    hierarchical node IDs instead of separate order/superfamily labels.

    Pickle format expected: {sequence_string: (order_str, superfamily_str)}
    where superfamily_str can be "" for orders without superfamilies.

    Returns per sample: (tokens, src_key_padding_mask, target_node_id)
    """

    def __init__(
        self,
        sequence_dict,
        label_to_id: dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
    ):
        assert ignore_index is not None

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        # ---- Filter and build valid sequence list ----
        valid_sequences = []
        valid_node_ids = []
        skipped = 0

        for seq, (order_str, sf_str) in sequence_dict.items():
            # Build full hierarchical name
            if sf_str and sf_str != "":
                full_name = f"{order_str}/{sf_str}"
            else:
                # Orders without superfamilies (DIRS, Helitron, PLE)
                full_name = order_str

            if full_name in label_to_id:
                node_id = label_to_id[full_name]
            elif order_str in label_to_id:
                node_id = label_to_id[order_str]
            else:
                skipped += 1
                continue

            valid_sequences.append(seq)
            valid_node_ids.append(node_id)

        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} sequences with labels not in classification tree")

        # ---- Pre-encode everything once ----
        self.all_tokens = self._encode_all(valid_sequences)                 # (N, L)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)     # (N, L)
        self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)  # (N,)

        print(f"  Dataset created: {len(self.all_node_ids)} samples")

    def _encode_all(self, sequence_list):
        """Encode every sequence upfront into a single (N, max_seq_len) tensor."""
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],          # (L,)
            self.all_padding_masks[idx],   # (L,)
            self.all_node_ids[idx],        # scalar
        )



























class SequenceDataset_nocollate_nomer(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
    ):
        assert ignore_index is not None

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        # ---- Pre-encode everything once ----
        sequences = list(sequence_dict.keys())
        self.all_tokens = self._encode_all(sequences)                          # (N, L)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)        # (N, L)
        self.all_order_labels = self._encode_labels(sequences, sequence_dict, label_idx=0, label_type='Order')          # (N,)
        self.all_superfamily_labels = self._encode_labels(sequences, sequence_dict, label_idx=1, label_type='Superfamily')  # (N,)

    def _encode_all(self, sequence_list):
        """Encode every sequence upfront into a single (N, max_seq_len) tensor."""
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_labels(self, sequences, sequence_dict, label_idx, label_type):
        """Encode all labels for a given label type into a (N,) tensor."""
        labels = []
        for seq in sequences:
            raw_label = sequence_dict[seq][label_idx]
            encoded = LABEL_MAPPINGS[label_type].get(raw_label, self.ignore_index)
            labels.append(encoded)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        # Returns: (tokens, src_key_padding_mask, order_label, superfamily_label)
        return (
            self.all_tokens[idx],              # (L,)
            self.all_padding_masks[idx],       # (L,)
            self.all_order_labels[idx],        # scalar
            self.all_superfamily_labels[idx],  # scalar
        )
























class SequenceDataset(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        mode: str = "none",
        ignore_index=-100,
    ):
        assert ignore_index is not None

        self.ignore_index = ignore_index
        self.sequence_dict = sequence_dict
        self.sequence_list = list(self.sequence_dict.keys())
        self.max_seq_len = max_seq_len
        self.mode = mode


    def __len__(self):
        return len(self.sequence_list)

    def encode(self, seq: str):
        ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
        pad_len = self.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [PAD_TOKEN] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.sequence_list[idx]
        tokens = self.encode(seq)
        
        # Tuple unpacking
        order_label, superfamily_label = self.sequence_dict[seq]

        out = {
            "tokens": tokens, 
            "order_labels": torch.tensor(LABEL_MAPPINGS['Order'].get(order_label, self.ignore_index), dtype=torch.long),
            "superfamily_labels": torch.tensor(LABEL_MAPPINGS['Superfamily'].get(superfamily_label, self.ignore_index), dtype=torch.long),
        }
            
        return out


def collate_fn(batch, pad_token_id=PAD_TOKEN):
    assert pad_token_id is not None

    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    superfamily_labels = torch.stack([b["superfamily_labels"] for b in batch], dim=0)
    order_labels = torch.stack([b["order_labels"] for b in batch], dim=0)
    # FIXED: Removed duplicate 'order_labels' declaration here

    src_key_padding_mask = (tokens == pad_token_id)

    out = {
        "tokens": tokens,
        "src_key_padding_mask": src_key_padding_mask,
        "superfamily_labels": superfamily_labels,
        "order_labels": order_labels,
    }

    return out











from torch.utils.data import Dataset


def read_tsv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


class SequenceDataset_noCollate(Dataset):
    def __init__(self, sequence_dict, max_seq_len, mode="none", 
                 mer2_path=None, mer4_path=None, mer8_path=None, ignore_index=-100):
        
        self.ignore_index = ignore_index
        self.sequence_dict = sequence_dict
        self.sequence_list = list(self.sequence_dict.keys())
        self.max_seq_len = max_seq_len
        self.mode = mode
        
        # Consolidate dynamic k-mers into a unified dictionary to remove 'if' blocks later
        self.active_mers = {}
        paths = {"mer2": mer2_path, "mer4": mer4_path, "mer8": mer8_path}
        
        for name, path in paths.items():
            if path is not None:
                data = read_tsv(path)
                mer_dict = {row[0].lower(): torch.tensor([float(num) for num in row[3:]], dtype=torch.float) for row in data[1:]}
                mer_dim = len(next(iter(mer_dict.values())))
                self.active_mers[name] = (mer_dict, mer_dim)

    def __len__(self):
        return len(self.sequence_list)

    def encode(self, seq: str):
        ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
        pad_len = self.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [PAD_TOKEN] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.sequence_list[idx]
        tokens = self.encode(seq)
        order_label, superfamily_label = self.sequence_dict[seq]

        out = {
            "tokens": tokens, 
            "order_labels": torch.tensor(LABEL_MAPPINGS['Order'].get(order_label, self.ignore_index), dtype=torch.long),
            "superfamily_labels": torch.tensor(LABEL_MAPPINGS['Superfamily'].get(superfamily_label, self.ignore_index), dtype=torch.long),
        }
        
        # Zero 'if' statements: Dynamically unpacks only the k-mers that were provided in __init__
        out.update({
            mer_name: mer_dict.get(seq.lower(), torch.zeros(dim, dtype=torch.float))
            for mer_name, (mer_dict, dim) in self.active_mers.items()
        })
            
        return out

# NOTE: YOU CAN NOW DELETE `collate_fn` COMPLETELY!


class DeviceDataLoader:
    """Wraps a dataloader to seamlessly move batches to the specified device."""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        # Yields batches mapped directly to the GPU using PyTorch's native tree_map (Zero manual loops)
        for batch in self.dataloader:
            yield torch.utils._pytree.tree_map(lambda t: t.to(self.device, non_blocking=True), batch)

    def __len__(self):
        return len(self.dataloader)










# =====================================================================
# GPU MASKING FUNCTION
# =====================================================================
def apply_mlm_mask_gpu(
    tokens,
    mask_prob=0.15,
    pad_token_id=0,
    mask_token_id=6, # Adjust to your VOCAB_SIZE
    vocab_size=6,    # Number of valid tokens (0-5)
    mask_ignore_token_ids=(5,)  # e.g., 'X'
):
    """
    Applies standard BERT MLM masking ON THE GPU.
    Expects `tokens` to already be on the target DEVICE.
    """
    device = tokens.device
    B, L = tokens.shape

    masked_tokens = tokens.clone()
    mlm_labels = torch.full_like(tokens, -100)

    # 1. Build "can be masked" boolean mask
    can_mask = tokens != pad_token_id
    for tid in mask_ignore_token_ids:
        can_mask &= (tokens != tid)

    # 2. Decide which tokens are selected for MLM
    rand = torch.rand((B, L), device=device)
    mlm_mask = (rand < mask_prob) & can_mask

    # Labels: only masked positions have targets
    mlm_labels[mlm_mask] = tokens[mlm_mask]

    # 3. Replacement strategy (80 / 10 / 10)
    replace_rand = torch.rand((B, L), device=device)

    # 80% → [MASK]
    mask_replace = mlm_mask & (replace_rand < 0.8)
    masked_tokens[mask_replace] = mask_token_id

    # 10% → random token (valid only)
    random_replace = mlm_mask & (replace_rand >= 0.8) & (replace_rand < 0.9)

    if random_replace.any():
        invalid_ids = set(mask_ignore_token_ids) | {pad_token_id, mask_token_id}
        
        # Pre-build valid replacement set
        valid_token_ids = torch.tensor(
            [i for i in range(vocab_size) if i not in invalid_ids],
            device=device,
            dtype=tokens.dtype
        )

        rand_idx = torch.randint(
            0, len(valid_token_ids), size=(B, L), device=device
        )
        random_tokens = valid_token_ids[rand_idx]
        masked_tokens[random_replace] = random_tokens[random_replace]

    return masked_tokens, mlm_labels

