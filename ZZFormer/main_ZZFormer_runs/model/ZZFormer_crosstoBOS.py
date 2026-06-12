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


class BosGlobalAttention(nn.Module):
    """
    'Global attention to BOS' (residual + post-LN).
    BOS queries all valid tokens; only the BOS slot is updated.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, key_padding_mask=None):
        bos = hidden[:, 0:1, :]                     # (B, 1, d_model)
        attn_out, _ = self.attn(
            query=bos, key=hidden, value=hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        bos_updated = self.norm(bos + self.dropout(attn_out))
        return torch.cat([bos_updated, hidden[:, 1:, :]], dim=1)



class HierarchicalLongformerClassifier(nn.Module):
    """
    Longformer backbone with per-layer k-mer topology cross-attention and
    BOS global attention scheduled AFTER the cross-attention.

    One k-mer per encoder layer:
        for i in range(num_layers):
            1. Local-only Longformer self-attention (layer i)
            2. Cross-attn to topology_latent_stack[i] (k-mer k_mers[i])
            3. BOS global attention (BOS attends to all valid tokens)

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

        # One k-mer per encoder layer.
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
        # Keep attribute name `longformer` so MLM-checkpoint keys still load.
        self.longformer = LongformerModel(longformer_config, add_pooling_layer=False)

        # ---- 2-4. One per encoder layer, all indexed by the same `i` ----
        self.kmer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(topology_latent_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.bos_global_attn = nn.ModuleList([
            BosGlobalAttention(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- 5. Hierarchical output head ----
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

        # ---- 6. Loss ----
        self.hierarchical_loss = HierarchicalSoftmaxLoss(root=self.classification_tree)

    # ----------------------------------------------------------
    # Manual encoder loop:
    #   local self-attn  ->  cross-attn(k_mers[i])  ->  BOS global attn
    # ----------------------------------------------------------
    def _run_longformer_interleaved(
        self, input_ids, valid_mask, topology_latent_stack
    ):
        lf  = self.longformer
        cfg = lf.config

        # Pad up to attention_window. `valid_mask` stays in raw HF convention
        # ({0=pad, 1=token}) because the Longformer layers run in local-only
        # mode; global attention is handled by us below.
        (padding_len, input_ids, valid_mask,
         token_type_ids, position_ids, _) = lf._pad_to_window_size(
            input_ids     = input_ids,
            attention_mask= valid_mask,
            token_type_ids= torch.zeros_like(input_ids),
            position_ids  = None,
            inputs_embeds = None,
            pad_token_id  = cfg.pad_token_id,
        )

        # Embeddings
        embedding_output = lf.embeddings(
            input_ids     = input_ids,
            position_ids  = position_ids,
            token_type_ids= token_type_ids,
            inputs_embeds = None,
        )
        dtype = embedding_output.dtype

        # Local-only extended mask (no '>0' entries -> no global attention).
        extended_attention_mask = torch.zeros_like(valid_mask, dtype=dtype)
        extended_attention_mask = extended_attention_mask.masked_fill(
            valid_mask == 0, torch.finfo(dtype).min      # mask out padding
        )

        is_index_masked      = extended_attention_mask < 0
        is_index_global_attn = torch.zeros_like(valid_mask, dtype=torch.bool)
        is_global_attn       = False

        # nn.MultiheadAttention convention: True = mask this key.
        key_padding_mask = (valid_mask == 0)

        hidden_states = embedding_output
        for i, layer_module in enumerate(lf.encoder.layer):
            # 1. Local-only Longformer self-attention
            hidden_states = layer_module(
                hidden_states,
                attention_mask       = extended_attention_mask,
                is_index_masked      = is_index_masked,
                is_index_global_attn = is_index_global_attn,
                is_global_attn       = is_global_attn,
                output_attentions    = False,
            )[0]

            # 2. Cross-attention to topology k-mer token i  (B, 64) -> (B, 1, d)
            context = self.kmer_projections[i](
                topology_latent_stack[i]
            ).unsqueeze(1)
            hidden_states = self.cross_attn_layers[i](hidden_states, context)

            # 3. Global attention to BOS (AFTER cross-attention)
            hidden_states = self.bos_global_attn[i](
                hidden_states, key_padding_mask=key_padding_mask
            )

        # Undo window padding. When padding_len == 0 this is a full-length
        # slice (no-op), so it's safe to run unconditionally.
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

        pooled     = h[:, 0, :]                     # BOS pool (only mode)
        logits     = self.output_head(pooled)
        total_loss = self.hierarchical_loss(logits, target_node_ids)

        return {"total_loss": total_loss, "logits": logits}
