import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 
import math
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss
from hierarchicalsoftmax.inference import (
    greedy_predictions,
    node_probabilities,
)



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




 



class SimpleDenseClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout=0.1):
        super(SimpleDenseClassifier, self).__init__()
        # A 3-layer MLP is much more stable and trains faster than a 6-layer one.
        # Added Dropout and LayerNorm to prevent overfitting!
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output Layer (Logits)
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape expected: (Batch, embedding_dim)
        return self.network(x)


class PositionalEncoding(nn.Module):
    """
    Standard PyTorch Positional Encoding module.
    Adapted for batch_first=True (B, L, D).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape for batch_first=True: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)







# ──────────────────────────────────────────────
# 1. Cross-Attention Layer
# ──────────────────────────────────────────────
class CrossAttentionLayer_FFN(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, context, context_key_padding_mask=None):
        # Sub-layer 1: Cross-Attention + its own residual + norm
        attn_out, _ = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=context_key_padding_mask,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: FFN + its own residual + norm
        x = self.norm2(x + self.ffn(x))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class PersistenceImageEncoder5Tokens(nn.Module):
    """
    Input:  imgs [B, 5, 128, 128]
    Output: context [B, 5, d_model]  (one token per channel)
    """
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        # shared CNN applied to each channel-image independently
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 32 -> 16
            nn.ReLU(),
        )
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, imgs):
        B, C, H, W = imgs.shape  # C=5
        x = imgs.reshape(B * C, 1, H, W)      # [B*5, 1, 128, 128]
        feats = self.cnn(x)                   # [B*5, 128, 16, 16]
        feats = feats.mean(dim=(2, 3))        # GAP -> [B*5, 128]
        tok = self.proj(feats)                # [B*5, d_model]
        tok = self.ln(tok)
        tok = self.dropout(tok)
        context = tok.reshape(B, C, -1)       # [B, 5, d_model]
        return context
    

class CrossAttentionLayer_2persimg(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_key_padding_mask=None):
        # context: [B, N_ctx, d_model]
        attn_out, _ = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=context_key_padding_mask,  # [B, N_ctx] where True=mask
            need_weights=False,
        )

        out = self.norm(x + self.dropout(attn_out))
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_key_padding_mask=None):
        attn_out, _ = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=context_key_padding_mask,
        )
        out = self.norm(x + self.dropout(attn_out))
        return out











class HierarchicalFFNTransformerClassifier(nn.Module):
    """
    Hierarchical (order → superfamily) version of OrderFFNtransformer_Classifier /
    SuperfamilyFFNtransformer_Classifier.

    Encoder stack is identical to the FFN models:
        for each k-mer:
            self-attention  →  build topology context  →  cross-attention

    Classification head is identical to HierarchicalTransformerClassifier:
        mean-pool  →  output_head → raw logits (B, root.layer_size)
        HierarchicalSoftmaxLoss handles the tree internally.

    Args:
        classification_tree: SoftmaxNode root containing the order→superfamily tree.
        context_mode:
            "pi_tokens"  – use PersistenceImageEncoder5Tokens (like the Order model,
                           uses CrossAttentionLayer_2persimg).
            "proj_token" – project topo_latent → (B, 1, d_model) (like the Superfamily
                           model, uses CrossAttentionLayer).
    """

    def __init__(
        self,
        src_vocab_size: int,
        classification_tree,                 # SoftmaxNode
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        classifier_hidden_dim: int = 256,
        topology_latent_dim: int = 512,
        k_mers=(2, 4, 8),
        context_mode: str = "pi_tokens",     # "pi_tokens" or "proj_token"
    ):
        super().__init__()
        assert context_mode in ("pi_tokens", "proj_token")

        self.input_vocab_size = src_vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.dropout = dropout
        self.k_mers = list(k_mers)
        self.num_cross_layers = len(self.k_mers)
        self.context_mode = context_mode

        # ---- Hierarchical tree ----
        self.classification_tree = classification_tree
        self.classification_tree.set_indexes_if_unset()
        self.output_dim = self.classification_tree.layer_size

        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, self.dropout, max_len=max_position_embeddings
        )

        # ---- 2. Self-attention encoder blocks (one per k-mer) ----
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=self.dropout,
                norm_first=True,
                batch_first=True,
            )
            for _ in range(self.num_cross_layers)
        ])

        # ---- 3. Cross-attention layers (one per k-mer) ----
        if context_mode == "pi_tokens":
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionLayer_2persimg(d_model, n_heads, dropout=self.dropout)
                for _ in range(self.num_cross_layers)
            ])
            self.pi_encoders = nn.ModuleList([
                PersistenceImageEncoder5Tokens(d_model=d_model, dropout=self.dropout)
                for _ in range(self.num_cross_layers)
            ])
            self.kmer_projections = None
        else:  # "proj_token"
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionLayer(d_model, n_heads, dropout=self.dropout)
                for _ in range(self.num_cross_layers)
            ])
            self.kmer_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(topology_latent_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(self.num_cross_layers)
            ])
            self.pi_encoders = None

        # ---- 4. Hierarchical output head ----
        # Same shape as HierarchicalTransformerClassifier.output_head.
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_hidden_dim // 2, self.output_dim),
        )

        # ---- 5. Hierarchical loss ----
        self.hierarchical_loss = HierarchicalSoftmaxLoss(
            root=self.classification_tree
        )

    # ------------------------------------------------------------------ utils
    def _mean_pooling(self, h, src_key_padding_mask):
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode(self, tokens, src_key_padding_mask,
                topology_latent_stack=None, topology_mask=None):
        """Run the full encoder (self-attn + cross-attn over k-mers)."""
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        # If no topology is provided, just run the self-attn stack
        # (useful for `get_latent_embeddings`).
        if topology_latent_stack is None:
            for self_attn in self.encoder_layers:
                h = self_attn(h, src_key_padding_mask=src_key_padding_mask)
            return h

        assert len(topology_latent_stack) == self.num_cross_layers, (
            f"Expected {self.num_cross_layers} topology latents, "
            f"got {len(topology_latent_stack)}"
        )
        if topology_mask is None:
            topology_mask = [None] * self.num_cross_layers

        for i, (self_attn, cross_attn) in enumerate(
            zip(self.encoder_layers, self.cross_attn_layers)
        ):
            h = self_attn(h, src_key_padding_mask=src_key_padding_mask)

            topo_latent = topology_latent_stack[i]
            ctx_mask = topology_mask[i]

            if self.context_mode == "pi_tokens":
                context = self.pi_encoders[i](topo_latent)
            else:
                context = self.kmer_projections[i](topo_latent).unsqueeze(1)

            h = cross_attn(h, context, context_key_padding_mask=ctx_mask)
        return h

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        tokens,
        src_key_padding_mask,
        target_node_ids,                # (B,) leaf (superfamily) node IDs
        topology_latent_stack,          # list of len num_cross_layers
        topology_mask=None,             # list of len num_cross_layers or None
    ):
        """
        TRAINING / EVAL forward pass.

        Args:
            tokens:                (B, L)
            src_key_padding_mask:  (B, L) True where padded
            target_node_ids:       (B,) int node IDs into root.node_list.
                                   For order→superfamily, pass the SUPERFAMILY
                                   node id; the hierarchical loss walks up to
                                   the order automatically.
                                   Pass None at inference.
            topology_latent_stack: list of (B, latent_dim) tensors, one per k-mer.
            topology_mask:         list of masks (or None) for each context.

        Returns:
            dict with:
                'total_loss': scalar (0.0 if target_node_ids is None)
                'logits':     (B, root.layer_size) RAW logits — no softmax applied
        """
        h = self._encode(
            tokens, src_key_padding_mask,
            topology_latent_stack=topology_latent_stack,
            topology_mask=topology_mask,
        )
        seq_z = self._mean_pooling(h, src_key_padding_mask)

        logits = self.output_head(seq_z)  # (B, root.layer_size)

        total_loss = torch.tensor(0.0, device=tokens.device)
        if target_node_ids is not None:
            total_loss = self.hierarchical_loss(logits, target_node_ids)

        return {
            "total_loss": total_loss,
            "logits": logits,
        }

    # ----------------------------------------------------------- embeddings
    def get_latent_embeddings(self, batch):
        """
        Mean-pooled encoder representation. If `topology_latent_stack` is in
        the batch, cross-attention is applied; otherwise only self-attention.
        """
        tokens = batch["tokens"]
        src_key_padding_mask = batch.get("src_key_padding_mask", None)
        topology_latent_stack = batch.get("topology_latent_stack", None)
        topology_mask = batch.get("topology_mask", None)

        h = self._encode(
            tokens, src_key_padding_mask,
            topology_latent_stack=topology_latent_stack,
            topology_mask=topology_mask,
        )
        return self._mean_pooling(h, src_key_padding_mask)














































# class OrderFFNtransformer_Classifier(nn.Module):
#     def __init__(self, src_vocab_size, d_model=256, n_heads=8, 
#                  dim_feedforward=1024, dropout=0.1, max_position_embeddings=512,
#                  pad_token_id=0, ignore_index=-100, num_orders=7, 
#                  classifier_hidden_dim=64, num_layers=4, topology_latent_dim=512,k_mers=[2, 4, 8],):
#         super().__init__()
#         self.input_vocab_size = src_vocab_size + 1
#         self.d_model = d_model
#         self.pad_token_id = pad_token_id
#         self.ignore_index = ignore_index
#         self.num_layers = num_layers
#         self.num_orders = num_orders
#         self.dropout=dropout
#         self.k_mers = list(k_mers)  
        
#         self.num_cross_layers = len(self.k_mers)     # 3 layers, one per k-mer

#         # ---- 1. Embedding & Positional Encoding ----
#         self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, self.dropout, max_len=max_position_embeddings)

#         # ---- 2. Self-Attention Encoder Blocks (one per k-mer) ----
#         self.encoder_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=n_heads,
#                 dim_feedforward=dim_feedforward,
#                 dropout=self.dropout,
#                 norm_first=True,
#                 batch_first=True,
#             ) for _ in range(self.num_cross_layers)
#         ])

#         # ---- 3. Cross-Attention Layers (one per k-mer) ----
#         self.cross_attn_layers = nn.ModuleList([
#             CrossAttentionLayer_2persimg(d_model, n_heads, dropout=self.dropout)
#             for _ in range(self.num_cross_layers)
#         ])

#         # ---- 4. Per-kmer projection: topology_latent_dim → d_model ----
#         self.kmer_projections = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(topology_latent_dim, d_model),
#                     nn.LayerNorm(d_model),
#                 )
#                 for _ in range(self.num_cross_layers)
#             ])

#         # ---- 5. Classifier ----
#         self.order_classifier = SimpleDenseClassifier(
#             d_model, classifier_hidden_dim, self.num_orders, dropout=self.dropout
#         )

#         self.pi_encoder =nn.ModuleList([
#             PersistenceImageEncoder5Tokens(d_model=d_model, dropout=self.dropout)
#             for _ in range(self.num_cross_layers)
#         ])



#     def _mean_pooling(self, h, src_key_padding_mask):
#         """Helper function to mean-pool the sequence embeddings."""
#         if src_key_padding_mask is not None:
#             valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
#         else:
#             valid_mask = torch.ones_like(h)
            
#         sum_embeddings = torch.sum(h * valid_mask, dim=1)
#         sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
#         return sum_embeddings / sum_mask

#     def forward(self, tokens, src_key_padding_mask, order_label,
#                 topology_latent_stack, topology_mask):
#         """
#         Args:
#             tokens:                [batch, seq_len]
#             src_key_padding_mask:  [batch, seq_len]
#             order_label:           [batch]
#             topology_latent_stack: list of 3 tensors, each [batch, latent_dim]
#                                    ordered as [mer2_vec, mer4_vec, mer8_vec]
#             topology_mask:         not needed for single-vector contexts (None)
#         """
#         assert len(topology_latent_stack) == self.num_cross_layers, \
#             f"Expected {self.num_cross_layers} topology latents, got {len(topology_latent_stack)}"

#         # ---- Embedding + Positional Encoding ----
#         h = self.src_embed(tokens) * math.sqrt(self.d_model) # This scaling is from Attention is All you Need (Vaswani et al.)
#         h = self.pos_encoder(h)

#         # ---- Layer-wise: Self-Attn → Project k-mer → Cross-Attn ----
#         for self_attn, cross_attn, proj, topo_latent, topology_mask_1 in zip(
#             self.encoder_layers,
#             self.cross_attn_layers,
#             self.kmer_projections,
#             topology_latent_stack,
#             topology_mask,):
#             h = self_attn(h, src_key_padding_mask=src_key_padding_mask)
#             # topo_latent: [batch, latent_dim] → project → [batch, 1, d_model]
#             # context = proj(topo_latent).unsqueeze(1)
#             context = self.pi_encoder(topo_latent)
#             h = cross_attn(h, context, context_key_padding_mask=topology_mask_1)

#         seq_z = self._mean_pooling(h, src_key_padding_mask)

#         # ---- Classification ----
#         order_logits = self.order_classifier(seq_z)

#         order_loss = 0.0
#         if order_label is not None:
#             order_loss = F.cross_entropy(
#                 order_logits, order_label.long(), ignore_index=self.ignore_index
#             )

#         return {
#             'total_loss': order_loss,
#             'order_logits': order_logits,
#         }


#     def get_latent_embeddings(self, batch):
#         """
#         Passes tokens through the encoder and computes a mean-pooled 
#         representation across the sequence, ignoring PAD tokens.
#         """
#         tokens = batch['tokens']
#         src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
#         # 1. Embeddings
#         h = self.src_embed(tokens) * math.sqrt(self.d_model)

#         h = self.pos_encoder(h)

#         for encoderlayer in self.encoder_layers:
#             # 1. Forward through this specific transformer layer
#             h = encoderlayer(h, src_key_padding_mask=src_key_padding_mask)
  
#         # 3. Mean Pooling (Ignore padding tokens)
#         if src_key_padding_mask is not None:
#             # src_key_padding_mask is True for padding. We want True for valid tokens.
#             valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
#         else:
#             valid_mask = torch.ones_like(h)
            
#         sum_embeddings = torch.sum(h * valid_mask, dim=1)
#         sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9) # Avoid division by zero
#         mean_pooled = sum_embeddings / sum_mask
        
#         return mean_pooled






















































# class SuperfamilyFFNtransformer_Classifier(nn.Module):
#     def __init__(self, src_vocab_size, d_model=256, n_heads=8, 
#                  dim_feedforward=1024, dropout=0.1, max_position_embeddings=512,
#                  pad_token_id=0, ignore_index=-100, num_superfamilies=24,
#                  classifier_hidden_dim=64, num_layers=4, topology_latent_dim=512,k_mers=[2, 4, 8],):
#         super().__init__()
#         self.input_vocab_size = src_vocab_size + 1
#         self.d_model = d_model
#         self.pad_token_id = pad_token_id
#         self.ignore_index = ignore_index
#         self.num_layers = num_layers
#         self.num_superfamilies = num_superfamilies
#         self.dropout=dropout
#         self.k_mers = list(k_mers)  
        
#         self.num_cross_layers = len(self.k_mers)     # 3 layers, one per k-mer

#         # ---- 1. Embedding & Positional Encoding ----
#         self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, self.dropout, max_len=max_position_embeddings)

#         # ---- 2. Self-Attention Encoder Blocks (one per k-mer) ----
#         self.encoder_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=n_heads,
#                 dim_feedforward=dim_feedforward,
#                 dropout=self.dropout,
#                 norm_first=True,
#                 batch_first=True,
#             ) for _ in range(self.num_cross_layers)
#         ])

#         # ---- 3. Cross-Attention Layers (one per k-mer) ----
#         self.cross_attn_layers = nn.ModuleList([
#             CrossAttentionLayer(d_model, n_heads, dropout=self.dropout)
#             for _ in range(self.num_cross_layers)
#         ])

#         # ---- 4. Per-kmer projection: topology_latent_dim → d_model ----
#         self.kmer_projections = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(topology_latent_dim, d_model),
#                     nn.LayerNorm(d_model),
#                 )
#                 for _ in range(self.num_cross_layers)
#             ])
            
#         # ---- 5. Classifier ----
#         self.sf_classifier = SimpleDenseClassifier(
#             d_model, classifier_hidden_dim,  self.num_superfamilies, dropout=self.dropout
#         )




#     def _mean_pooling(self, h, src_key_padding_mask):
#         """Helper function to mean-pool the sequence embeddings."""
#         if src_key_padding_mask is not None:
#             valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
#         else:
#             valid_mask = torch.ones_like(h)
            
#         sum_embeddings = torch.sum(h * valid_mask, dim=1)
#         sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
#         return sum_embeddings / sum_mask

#     def forward(self, tokens, src_key_padding_mask, superfamily_label,
#                 topology_latent_stack, topology_mask):
#         """
#         Args:
#             tokens:                [batch, seq_len]
#             src_key_padding_mask:  [batch, seq_len]
#             order_label:           [batch]
#             topology_latent_stack: list of 3 tensors, each [batch, latent_dim]
#                                    ordered as [mer2_vec, mer4_vec, mer8_vec]
#             topology_mask:         not needed for single-vector contexts (None)
#         """
#         assert len(topology_latent_stack) == self.num_cross_layers, \
#             f"Expected {self.num_cross_layers} topology latents, got {len(topology_latent_stack)}"

#         # ---- Embedding + Positional Encoding ----
#         h = self.src_embed(tokens) * math.sqrt(self.d_model) # This scaling is from Attention is All you Need (Vaswani et al.)
#         h = self.pos_encoder(h)

#         # ---- Layer-wise: Self-Attn → Project k-mer → Cross-Attn ----
#         for self_attn, cross_attn, proj, topo_latent, topology_mask_1 in zip(
#             self.encoder_layers,
#             self.cross_attn_layers,
#             self.kmer_projections,
#             topology_latent_stack,
#             topology_mask,):
#             h = self_attn(h, src_key_padding_mask=src_key_padding_mask)
#             # topo_latent: [batch, latent_dim] → project → [batch, 1, d_model]
#             context = proj(topo_latent).unsqueeze(1)
#             h = cross_attn(h, context, context_key_padding_mask=topology_mask_1)

#         seq_z = self._mean_pooling(h, src_key_padding_mask)

#         # ---- Classification ----
#         sf_logits = self.sf_classifier(seq_z)

#         # ---- Superfamily Classification Loss ----
#         sf_loss = 0.0
#         if superfamily_label is not None:
#             sf_loss = F.cross_entropy(sf_logits, superfamily_label.long(), ignore_index=self.ignore_index)

#         return {
#             'total_loss': sf_loss,
#             'sf_logits': sf_logits,
#         }


#     def get_latent_embeddings(self, batch):
#         """
#         Passes tokens through the encoder and computes a mean-pooled 
#         representation across the sequence, ignoring PAD tokens.
#         """
#         tokens = batch['tokens']
#         src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
#         # 1. Embeddings
#         h = self.src_embed(tokens) * math.sqrt(self.d_model)

#         h = self.pos_encoder(h)

#         for encoderlayer in self.encoder_layers:
#             # 1. Forward through this specific transformer layer
#             h = encoderlayer(h, src_key_padding_mask=src_key_padding_mask)
  
#         # 3. Mean Pooling (Ignore padding tokens)
#         if src_key_padding_mask is not None:
#             # src_key_padding_mask is True for padding. We want True for valid tokens.
#             valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
#         else:
#             valid_mask = torch.ones_like(h)
            
#         sum_embeddings = torch.sum(h * valid_mask, dim=1)
#         sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9) # Avoid division by zero
#         mean_pooled = sum_embeddings / sum_mask
        
#         return mean_pooled


