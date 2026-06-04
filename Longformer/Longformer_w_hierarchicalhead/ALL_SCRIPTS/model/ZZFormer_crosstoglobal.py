import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerConfig

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss


class CrossAttentionLayer(nn.Module):
    """Single-token-context-friendly cross-attention (residual + post-LN)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_key_padding_mask=None):
        # x:       (B, L,     d_model)
        # context: (B, L_ctx, d_model)   -- here L_ctx == 1
        attn_out, _ = self.cross_attn(
            query=x, key=context, value=context,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )
        return self.norm(x + self.dropout(attn_out))


class HierarchicalLongformerClassifier(nn.Module):
    """
    Longformer encoder backbone with **interleaved** k-mer topology
    cross-attention + hierarchical softmax classification head.

    Per layer i in `cross_attn_after_layers`, a single context token is
    built from the 64-d topology latent of the corresponding k-mer
    (linear -> LayerNorm -> unsqueeze) and inserted as a cross-attn
    between Longformer layer i and layer i+1.
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
        pool: str = "bos",                     # "bos" or "mean"
        # ---- Topology cross-attention ----
        topology_latent_dim: int = 64,         # 64-d embedding per k-mer
        k_mers=(4, 8, 14, 20),
        cross_attn_after_layers=None,          # list[int]; default = one-to-one with k_mers
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.pool         = pool
        self.classification_tree = classification_tree
        self.classification_tree.set_indexes_if_unset()
        self.output_dim   = self.classification_tree.layer_size

        self.k_mers              = list(k_mers)
        self.num_cross_layers    = len(self.k_mers)
        self.topology_latent_dim = topology_latent_dim
        self.num_layers          = num_layers

        # ---- Decide which Longformer layers get a cross-attn after them ----
        if cross_attn_after_layers is None:
            if num_layers == self.num_cross_layers:
                cross_attn_after_layers = list(range(num_layers))
            else:
                if self.num_cross_layers > num_layers:
                    raise ValueError(
                        f"More k-mers ({self.num_cross_layers}) than "
                        f"Longformer layers ({num_layers})."
                    )
                # Evenly space, ending at the last layer.
                cross_attn_after_layers = [
                    int(round((i + 1) * num_layers / self.num_cross_layers)) - 1
                    for i in range(self.num_cross_layers)
                ]
        else:
            cross_attn_after_layers = list(cross_attn_after_layers)
            assert len(cross_attn_after_layers) == self.num_cross_layers, (
                f"cross_attn_after_layers must have length {self.num_cross_layers}, "
                f"got {len(cross_attn_after_layers)}"
            )
            for idx in cross_attn_after_layers:
                assert 0 <= idx < num_layers, (
                    f"cross_attn_after_layers index {idx} out of range "
                    f"[0, {num_layers})"
                )

        self.cross_attn_after_layers = cross_attn_after_layers
        # layer_idx -> position in self.k_mers / self.cross_attn_layers
        self._cross_attn_index_map = {
            layer_idx: ca_idx
            for ca_idx, layer_idx in enumerate(cross_attn_after_layers)
        }

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
        # Keep attribute name `longformer` so MLM-checkpoint keys
        # (longformer.embeddings.* / longformer.encoder.*) still load.
        self.longformer = LongformerModel(longformer_config, add_pooling_layer=False)

        # ---- 2. K-mer projections: 64 -> d_model (one per k-mer) ----
        self.kmer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(topology_latent_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(self.num_cross_layers)
        ])

        # ---- 3. Cross-attention layers (one per k-mer) ----
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout=dropout)
            for _ in range(self.num_cross_layers)
        ])

        # ---- 4. Hierarchical output head ----
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

        # ---- 5. Loss ----
        self.hierarchical_loss = HierarchicalSoftmaxLoss(root=self.classification_tree)

    # ----------------------------------------------------------
    # Pooling helpers
    # ----------------------------------------------------------
    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    # ----------------------------------------------------------
    # Topology helpers
    # ----------------------------------------------------------
    def _normalize_topo_latent(self, topo_latent, i):
        """Accept (B, 64), (B, 64, 1) or (B, 1, 64) -> (B, 64)."""
        if topo_latent.dim() == 3:
            if topo_latent.shape[-1] == 1:
                topo_latent = topo_latent.squeeze(-1)
            elif topo_latent.shape[1] == 1:
                topo_latent = topo_latent.squeeze(1)
            else:
                raise ValueError(
                    f"k-mer {self.k_mers[i]} latent has shape "
                    f"{tuple(topo_latent.shape)}; expected "
                    f"(B, {self.topology_latent_dim}), "
                    f"(B, {self.topology_latent_dim}, 1) or "
                    f"(B, 1, {self.topology_latent_dim})."
                )
        assert topo_latent.shape[-1] == self.topology_latent_dim, (
            f"k-mer {self.k_mers[i]} latent last-dim is "
            f"{topo_latent.shape[-1]}, expected {self.topology_latent_dim}."
        )
        return topo_latent

    # ----------------------------------------------------------
    # Manual encoder loop with interleaved cross-attention
    # ----------------------------------------------------------
    def _run_longformer_interleaved(
        self,
        input_ids,
        attention_mask,
        global_attention_mask,
        topology_latent_stack,
    ):
        """
        Reproduces LongformerModel.forward but loops the encoder layers
        manually, inserting a CrossAttentionLayer to the k-mer topology
        context token between the configured layers.
        """
        lf  = self.longformer
        cfg = lf.config

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if global_attention_mask is None:
            # Match HF default: global attention on BOS / position 0.
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        # Merge to {0: no-attn, 1: local, 2: global}
        merged_attention_mask = lf._merge_to_attention_mask(
            attention_mask, global_attention_mask
        )

        # Pad up to attention_window
        (padding_len, input_ids, merged_attention_mask,
         token_type_ids, position_ids, _) = lf._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=merged_attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=None,
            inputs_embeds=None,
            pad_token_id=cfg.pad_token_id,
        )

        # Embeddings
        embedding_output = lf.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )
        dtype = embedding_output.dtype

        # 2D extended_attention_mask expected by LongformerEncoder:
        #   < 0  -> masked,  == 0 -> local attn,  > 0 -> global attn
        extended_attention_mask = torch.zeros_like(
            merged_attention_mask, dtype=dtype
        )
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged_attention_mask == 0, torch.finfo(dtype).min
        )
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged_attention_mask == 2, torch.finfo(dtype).max
        )

        is_index_masked       = extended_attention_mask < 0
        is_index_global_attn  = extended_attention_mask > 0
        is_global_attn        = is_index_global_attn.flatten().any().item()

        # Manual layer loop with interleaved cross-attn
        hidden_states = embedding_output
        for idx, layer_module in enumerate(lf.encoder.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

            if (topology_latent_stack is not None
                    and idx in self._cross_attn_index_map):
                ca_idx = self._cross_attn_index_map[idx]
                topo_latent = self._normalize_topo_latent(
                    topology_latent_stack[ca_idx], ca_idx
                )                                              # (B, 64)
                context = self.kmer_projections[ca_idx](
                    topo_latent
                ).unsqueeze(1)                                 # (B, 1, d_model)
                hidden_states = self.cross_attn_layers[ca_idx](
                    hidden_states, context
                )

        # Undo padding so the returned tensor matches the original length
        if padding_len > 0:
            hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]

        return hidden_states

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(
        self,
        input_ids,
        attention_mask,
        target_node_ids=None,
        global_attention_mask=None,
        topology_latent_stack=None,    # list of (B, 64) tensors, one per k-mer
    ):
        if topology_latent_stack is not None:
            assert len(topology_latent_stack) == self.num_cross_layers, (
                f"Expected {self.num_cross_layers} topology latents "
                f"(one per k-mer {self.k_mers}), "
                f"got {len(topology_latent_stack)}."
            )
            h = self._run_longformer_interleaved(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                topology_latent_stack=topology_latent_stack,
            )
        else:
            # No topology -> fall back to stock LongformerModel.
            out = self.longformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                return_dict=True,
            )
            h = out.last_hidden_state

        if self.pool == "bos":
            pooled = h[:, 0, :]
        elif self.pool == "mean":
            pooled = self._mean_pool(h, attention_mask)
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        logits = self.output_head(pooled)

        total_loss = torch.tensor(0.0, device=input_ids.device)
        if target_node_ids is not None:
            total_loss = self.hierarchical_loss(logits, target_node_ids)
        return {"total_loss": total_loss, "logits": logits}