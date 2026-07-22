import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
import math
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader





DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

vocab = {
        "PAD": 0,
        "a": 1,
        "c": 2,
        "g": 3,
        "t": 4,
        "x": 5
    }
VOCAB_SIZE = len(vocab)
PAD_TOKEN = 0
MASK_TOKEN_ID = VOCAB_SIZE  



class SequenceDataset_nocollate(Dataset):
    def __init__(
        self,
        sequence_list,
        max_seq_len,
        pad_token_id: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # ---- Pre-encode ALL sequences once ----
        self.all_tokens = self._encode_all(sequence_list)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)

    def _encode_all(self, sequence_list):
        """Encode every sequence upfront into a single (N, max_seq_len) tensor."""
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["x"]) for c in seq[:self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        # Returns: (tokens, src_key_padding_mask, mlm_labels)
        return (
            self.all_tokens[idx],                          # (L,)
            self.all_padding_masks[idx],                   # (L,)
            # torch.full((self.max_seq_len,), -100, dtype=torch.long),  # (L,)
        )







































class SequenceDataset(Dataset):
    def __init__(
        self,
        sequence_list,
        max_seq_len,
        mode: str = "none",   # "none", "tasked", "mlm"
    ):
        """
        hf_dataset: HuggingFace Dataset split
        max_seq_len: int
        mode:
          - "none"   : encoder-only
          - "tasked" : full-sequence reconstruction
          - "mlm"    : masked language modeling
        """
        assert mode in {"none", "tasked", "mlm"}
        self.sequence_list = sequence_list
        self.max_seq_len = max_seq_len
        self.mode = mode

    def __len__(self):
        return len(self.sequence_list)

    def encode(self, seq: str):
        ids = [vocab.get(c, vocab["x"]) for c in seq[:self.max_seq_len]]
        pad_len = self.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [PAD_TOKEN] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.sequence_list[idx]
        tokens = self.encode(seq)

        out = {"tokens": tokens}

        if self.mode == "tasked":
            out["target_tokens"] = tokens.clone()

        # NOTE: MLM handled in collate_fn, not here
        return out


def collate_fn(
    batch,
    mode="none",
    mlm_prob=0.15,
    vocab_size=None,
    mask_token_id=None,
    pad_token_id=PAD_TOKEN,
    mask_ignore_token_ids=(5,),  # x
):
    """
    mode:
      - "none"   : encoder-only
      - "tasked" : reconstruction
      - "mlm"    : masked language modeling
    """

    assert pad_token_id is not None
    assert mask_token_id is not None
    assert vocab_size is not None
    assert mask_ignore_token_ids is not None

    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    src_key_padding_mask = (tokens == pad_token_id)

    out = {
        "tokens": tokens,
        "src_key_padding_mask": src_key_padding_mask
    }

    if mode in ["mlm", "bert_mlm"]:
        # Just return the raw tokens and padding mask. 
        # The GPU will handle the random masking!
        pass 

    return out







# =====================================================================
# GPU MASKING FUNCTION
# =====================================================================
def apply_mlm_mask_gpu(
    tokens,
    mask_prob=0.15,
    pad_token_id=0,
    mask_token_id=6, # Adjust to your VOCAB_SIZE
    vocab_size=6,    # Number of valid tokens (0-5)
    mask_ignore_token_ids=(5,),  # e.g., 'X'
    ignore_index=-100,

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

