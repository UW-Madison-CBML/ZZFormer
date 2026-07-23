import torch
import torch.nn as nn
import torch.nn.functional as F
import math


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

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

