import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerConfig

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss



class CrossAttentionFullQ(nn.Module):
    """
    Cross-attention where full transformer sequence (queries) attends to
    full topology sequence (keys/values).
    hidden:  (B, Lq, d_model)   e.g. (B, 1026, 256)
    topo:    (B, Lk, d_model)   e.g. (B, 1024, 256) after projection if needed
    output:  (B, Lq, d_model)
    """
    def __init__(self, d_model, n_heads, kdim=256, vdim=256, dropout=0.1, norm_first=True):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=n_heads, kdim=256, vdim=256, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
    def forward(self, hidden, topo, topo_key_padding_mask=None):
        # hidden: (B, Lq, d_model)
        # topo:   (B, Lk, d_model)
        # topo_key_padding_mask: (B, Lk) with True for PAD positions (optional)
        if self.norm_first:
            q = self.norm(hidden)
            attn_out, _ = self.cross_attn(
                query=q,
                key=topo,
                value=topo,
                key_padding_mask=topo_key_padding_mask,
                need_weights=False,
            )
            out = hidden + self.dropout(attn_out)
        else:
            attn_out, _ = self.cross_attn(
                query=hidden,
                key=topo,
                value=topo,
                key_padding_mask=topo_key_padding_mask,
                need_weights=False,
            )
            out = self.norm(hidden + self.dropout(attn_out))
        return out

class TopologyEncoder(nn.Module):
    def __init__(self, n_channels=3, n_filters=16, model_dim=256, reduced_persistence=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        self.proj = nn.Linear(n_filters * 4 * reduced_persistence, model_dim)
    def forward(self, x):
        x = self.encoder(x)              # (B, C', H', W)
        x = x.permute(0, 3, 1, 2)        # (B, W, C', H')
        x = x.flatten(start_dim=2)       # (B, W, C'*H')
        x = self.proj(x)                 # (B, W, model_dim)
        return x #Need (B, 1024, C) for cross attention


class HierarchicalLongformerClassifier(nn.Module):
    """
    Standard Longformer encoder (local self-attention + global attention on
    BOS, native HF behaviour) with a BOS-to-topology cross-attention block
    after every Longformer layer.
    
    Per layer i:
        1. Longformer layer — local + global attention on BOS
        2. BOS cross-attends directly to topology_latent_stack[i] (k-mer k_mers[i])
           -> PyTorch handles projection from topology_latent_dim via kdim/vdim
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
        topology_latent_dim: int = 256,
        k_mers=(4, 8, 14, 20),
        # ---- new: per-k-mer CNN hyperparameters ----
        topology_in_channels:   int = 5,
        topology_cnn_filters:   int = 16,
        reduced_persistence: int = 16,
        model_dim: int = 256,
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
        self.kdim                = topology_latent_dim
        self.vdim                = topology_latent_dim
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
        # ---- 2. Per-layer BOS cross-attention ----
        # Passes kdim/vdim directly so CrossAttentionFullQ handles initial feature alignment
        self.full_cross_attn = nn.ModuleList([
            CrossAttentionFullQ(
                d_model, 
                n_heads, 
                kdim=self.kdim, 
                vdim=self.vdim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])


        # ---- 2a. Per-k-mer CNN topology encoders ----
        #   topology_encoders[i] handles the persistence image for k_mers[i].
        self.topology_encoders = nn.ModuleList([
            TopologyEncoder(
                n_channels=topology_in_channels,
                n_filters=topology_cnn_filters,
                model_dim=model_dim,
                reduced_persistence=reduced_persistence,
            )
            for _ in range(len(k_mers))
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
    #   Longformer layer (local + global-on-BOS) -> BOS x topology cross-attn
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

            # After each Longformer layer, every sequence token cross-attends to
            # the topology-token sequence for the corresponding k-mer scale.
            
            context = topology_latent_stack[i]#.unsqueeze(1)
            hidden_states = self.full_cross_attn[i](hidden_states, context)
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
        topology_images,
    ):
        assert len(topology_images) == self.num_layers, (
            f"Expected {self.num_layers} topology images "
            f"(one per Longformer layer / k-mer {self.k_mers}), "
            f"got {len(topology_images)}."
        )

        topology_latent_stack = [
                            self.topology_encoders[i](topology_images[i])  # each (B, topology_latent_dim)
                            for i in range(len(self.k_mers))
                        ]
        
        h = self._run_longformer_interleaved(
            input_ids             = input_ids,
            valid_mask            = attention_mask,
            topology_latent_stack = topology_latent_stack,
        )

        
        
        pooled     = h[:, 0, :]                                  # BOS pool
        logits     = self.output_head(pooled)
        total_loss = self.hierarchical_loss(logits, target_node_ids)
        return {"total_loss": total_loss, "logits": logits}, h