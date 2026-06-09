import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerConfig

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss





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




class CrossAttentionLayer(nn.Module):
    """Cross-attention block (residual + post-LN), single-token-context friendly."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_key_padding_mask=None):
        attn_out, _ = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )
        return self.norm(x + self.dropout(attn_out))


















class BosCrossAttention(nn.Module):
    """
    Cross-attention where BOS (query) attends to the single per-layer
    topology context token (key/value). Only the BOS slot is updated;
    all other positions pass through unchanged. Residual + post-LN on BOS.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, context):
        # hidden:  (B, L, d_model)
        # context: (B, 1, d_model)
        bos = hidden[:, 0:1, :]                              # (B, 1, d_model)
        attn_out, _ = self.cross_attn(
            query=bos, key=context, value=context,
            need_weights=False,
        )
        bos_updated = self.norm(bos + self.dropout(attn_out))
        return torch.cat([bos_updated, hidden[:, 1:, :]], dim=1)


class HierarchicalLongformerClassifier(nn.Module):
    """
    Standard Longformer encoder (local self-attention + global attention on
    BOS, native HF behaviour) with a BOS-to-topology cross-attention block
    after every Longformer layer.

    Per layer i:
        1. Longformer layer  — local + global attention on BOS
        2. BOS cross-attends to topology_latent_stack[i] (k-mer k_mers[i])
           -> only the BOS slot is updated

    Pooling is always BOS (h[:, 0, :]).
    """

    def __init__(
        self,
        classification_tree: SoftmaxNode,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 1026,
        attention_window: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 6,
        eos_token_id: int = 7,
        classifier_hidden_dim: int = 256,
        topology_latent_dim: int = 64,
        k_mers=(4, 8, 14, 20),
    ):
        super().__init__()

        assert len(k_mers) == num_layers, (
            f"Expected one k-mer per Longformer layer: "
            f"num_layers={num_layers}, len(k_mers)={len(k_mers)}."
        )

        self.pad_token_id        = pad_token_id
        self.classification_tree = classification_tree
        self.classification_tree.set_indexes_if_unset()
        self.output_dim          = self.classification_tree.layer_size

        self.k_mers              = list(k_mers)
        self.num_layers          = num_layers
        self.topology_latent_dim = topology_latent_dim

        # ---- 1. Longformer backbone ----
        longformer_config = LongformerConfig(
            attention_window             = attention_window,
            vocab_size                   = vocab_size,
            max_position_embeddings      = max_position_embeddings,
            hidden_size                  = d_model,
            num_hidden_layers            = num_layers,
            num_attention_heads          = n_heads,
            intermediate_size            = dim_feedforward,
            hidden_dropout_prob          = dropout,
            attention_probs_dropout_prob = dropout,
            pad_token_id                 = pad_token_id,
            bos_token_id                 = bos_token_id,
            eos_token_id                 = eos_token_id,
            return_dict                  = True,
        )
        # Attribute name stays `longformer` so MLM-checkpoint keys still load.
        self.longformer = LongformerModel(longformer_config, add_pooling_layer=False)

        # ---- 2. Per-layer topology projection + BOS cross-attention ----
        self.kmer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(topology_latent_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_layers)
        ])
        self.bos_cross_attn = nn.ModuleList([
            BosCrossAttention(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- 3. Hierarchical output head ----
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, self.output_dim),
        )

        # ---- 4. Loss ----
        self.hierarchical_loss = HierarchicalSoftmaxLoss(root=self.classification_tree)

    # ----------------------------------------------------------
    # Manual encoder loop:
    #   Longformer layer (local + global-on-BOS)  ->  BOS x topology cross-attn
    # ----------------------------------------------------------
    def _run_longformer_interleaved(
        self, input_ids, valid_mask, topology_latent_stack
    ):
        lf  = self.longformer
        cfg = lf.config

        # Native Longformer global attention on BOS (position 0) only.
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        # Merge to {0=pad, 1=local, 2=global} per HF convention.
        merged = lf._merge_to_attention_mask(valid_mask, global_attention_mask)

        # Pad to multiple of attention_window (padding entries become 0).
        (padding_len, input_ids, merged,
         token_type_ids, position_ids, _) = lf._pad_to_window_size(
            input_ids     = input_ids,
            attention_mask= merged,
            token_type_ids= torch.zeros_like(input_ids),
            position_ids  = None,
            inputs_embeds = None,
            pad_token_id  = cfg.pad_token_id,
        )

        embedding_output = lf.embeddings(
            input_ids     = input_ids,
            position_ids  = position_ids,
            token_type_ids= token_type_ids,
            inputs_embeds = None,
        )
        dtype = embedding_output.dtype

        # LongformerEncoder/Layer expects:
        #   < 0 -> masked (padding)
        #   = 0 -> local attention
        #   > 0 -> global attention
        extended_attention_mask = torch.zeros_like(merged, dtype=dtype)
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged == 0, torch.finfo(dtype).min                  # padding
        )
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged == 2, torch.finfo(dtype).max                  # global (BOS)
        )

        is_index_masked      = extended_attention_mask < 0
        is_index_global_attn = extended_attention_mask > 0
        is_global_attn       = is_index_global_attn.flatten().any().item()

        hidden_states = embedding_output
        for i, layer_module in enumerate(lf.encoder.layer):
            # 1. Standard Longformer layer: local + global attention on BOS
            hidden_states = layer_module(
                hidden_states,
                attention_mask       = extended_attention_mask,
                is_index_masked      = is_index_masked,
                is_index_global_attn = is_index_global_attn,
                is_global_attn       = is_global_attn,
                output_attentions    = False,
            )[0]

            # 2. BOS x topology cross-attention. (B, 64) -> (B, 1, d_model)
            context = self.kmer_projections[i](
                topology_latent_stack[i]
            ).unsqueeze(1)
            hidden_states = self.bos_cross_attn[i](hidden_states, context)

        # Undo window padding (no-op when padding_len == 0).
        hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
        return hidden_states

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(
        self,
        input_ids,
        attention_mask,
        target_node_ids,
        topology_latent_stack,
    ):
        assert len(topology_latent_stack) == self.num_layers, (
            f"Expected {self.num_layers} topology latents "
            f"(one per Longformer layer / k-mer {self.k_mers}), "
            f"got {len(topology_latent_stack)}."
        )

        h = self._run_longformer_interleaved(
            input_ids             = input_ids,
            valid_mask            = attention_mask,
            topology_latent_stack = topology_latent_stack,
        )

        pooled     = h[:, 0, :]                                  # BOS pool
        logits     = self.output_head(pooled)
        total_loss = self.hierarchical_loss(logits, target_node_ids)

        return {"total_loss": total_loss, "logits": logits}