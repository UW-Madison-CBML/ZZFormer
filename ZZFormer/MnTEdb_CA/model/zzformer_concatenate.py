import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerConfig

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss
from model.topology_encoder import TopologyEncoder




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









class HierarchicalLongformerClassifier_Concat(nn.Module):
    """
    Standard Longformer encoder (local self-attention + global attention on
    BOS) with a BOS-to-topology cross-attention block after every Longformer
    layer. Topology latents are produced on the fly by a per-k-mer CNN from
    persistence images.

    Per layer i (k = k_mers[i]):
        1. Longformer layer  — local + global attention on BOS
        2. topology_encoders[i] maps the k-mer's persistence image
           (B, C, 128, 128) -> (B, topology_latent_dim)
        3. The BOS token (h[:, 0, :]) and the topology latent are concatenated and passed through the classifier head.

    Pooling is always BOS (h[:, 0, :]).
    """

    def __init__(
        self,
        classification_tree,                  # SoftmaxNode
        vocab_size: int,
        d_model:                int = 256,
        n_heads:                int = 4,
        n_heads_cross:          int = 4,
        num_layers:             int = 4,
        dim_feedforward:        int = 1024,
        dropout:                float = 0.1,
        max_position_embeddings:int = 1026,
        attention_window:       int = 256,
        pad_token_id:           int = 0,
        bos_token_id:           int = 6,
        eos_token_id:           int = 7,
        classifier_hidden_dim:  int = 256,
        topology_latent_dim:    int = 64,
        k_mers                       = (4, 8, 14, 20),
        # ---- new: per-k-mer CNN hyperparameters ----
        topology_in_channels:   int = 5,
        topology_cnn_filters:   int = 16,
    ):
        super().__init__()

        self.pad_token_id        = pad_token_id
        self.classification_tree = classification_tree
        self.classification_tree.set_indexes_if_unset()
        self.output_dim          = self.classification_tree.layer_size

        self.k_mers              = list(k_mers)
        self.num_layers          = num_layers
        self.topology_latent_dim = topology_latent_dim
        self.n_heads_cross       = n_heads_cross

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

        # ---- 2a. Per-k-mer CNN topology encoders ----
        #   topology_encoders[i] handles the persistence image for k_mers[i].
        self.topology_encoders = nn.ModuleList([
            TopologyEncoder(
                n_channels=topology_in_channels,
                n_filters=topology_cnn_filters,
                latent_dim=topology_latent_dim,
                return_sequence=False,
            )
            for _ in range(len(k_mers))
        ])

        fused_dim = d_model + len(k_mers) * topology_latent_dim

        # ---- 3. Hierarchical output head ----
        self.output_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, classifier_hidden_dim),
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
    #   Longformer layer (local + global-on-BOS)
    #     -> per-k-mer CNN topology encoder
    #     -> BOS x topology cross-attn
    # ----------------------------------------------------------
    def _run_longformer(self, input_ids, valid_mask):
        lf  = self.longformer
        cfg = lf.config

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        merged = lf._merge_to_attention_mask(valid_mask, global_attention_mask)

        (padding_len, input_ids, merged,
        token_type_ids, position_ids, _) = lf._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=merged,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=None,
            inputs_embeds=None,
            pad_token_id=cfg.pad_token_id,
        )

        embedding_output = lf.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )
        dtype = embedding_output.dtype

        extended_attention_mask = torch.zeros_like(merged, dtype=dtype)
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged == 0, torch.finfo(dtype).min
        )
        extended_attention_mask = extended_attention_mask.masked_fill(
            merged == 2, torch.finfo(dtype).max
        )

        is_index_masked      = extended_attention_mask < 0
        is_index_global_attn = extended_attention_mask > 0
        is_global_attn       = is_index_global_attn.flatten().any().item()

        hidden_states = embedding_output
        for layer_module in lf.encoder.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=False,
            )[0]

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
        topology_images,
    ):
        """
        topology_images : Sequence[Tensor] of length == num_layers.
            topology_images[i] is the persistence-image batch for k_mers[i],
            shape (B, topology_in_channels, H, W). Order MUST match
            self.k_mers, e.g. [x4, x8, x14, x20] for k_mers=(4, 8, 14, 20).
        """
        assert len(topology_images) == len(self.k_mers), (
            f"Expected {len(self.k_mers)} persistence-image batches "
            f"(one per k-mer {self.k_mers}), got {len(topology_images)}."
        )

        h = self._run_longformer(input_ids=input_ids, valid_mask=attention_mask)
        bos = h[:, 0, :]  # (B, d_model)

        topo_latents = [
            self.topology_encoders[i](topology_images[i])  # each (B, topology_latent_dim)
            for i in range(len(self.k_mers))
        ]
        topo_concat = torch.cat(topo_latents, dim=1)       # (B, k_mers * topology_latent_dim)

        fused = torch.cat([bos, topo_concat], dim=1)       # (B, fused_dim)

        logits = self.output_head(fused)
        total_loss = self.hierarchical_loss(logits, target_node_ids)

        return {"total_loss": total_loss, "logits": logits}
    

    # ----------------------------------------------------------
    # Embedding extraction (for UMAP, etc.)
    # ----------------------------------------------------------
    @torch.no_grad()
    def get_latent_embeddings(
        self,
        input_ids,
        attention_mask,
        topology_images,
        global_attention_mask=None,  # kept for API compatibility; optional
    ):
        h = self._run_longformer(input_ids=input_ids, valid_mask=attention_mask)
        bos = h[:, 0, :]  # (B, d_model)

        topo_latents = []
        for i in range(len(self.k_mers)):
            z = self.topology_encoders[i](topology_images[i])
            # ensure (B, D)
            if z.dim() == 3:
                z = z.mean(dim=1)  # if sequence tokens
            topo_latents.append(z)

        topo_concat = torch.cat(topo_latents, dim=1)  # (B, K*D)
        fused = torch.cat([bos, topo_concat], dim=1)  # (B, fused_dim)
        return fused


