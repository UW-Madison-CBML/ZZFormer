import math
import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerConfig

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss




class HierarchicalLongformerClassifier(nn.Module):
    """
    Longformer encoder backbone + hierarchical softmax classification head.

    - Backbone is `LongformerModel` (same submodule name as inside
      LongformerForSequenceClassification and LongformerForMaskedLM),
      so MLM-pretrained weights transfer with the SAME loader you already have.
    - Head is the MLP → raw logits over root.layer_size, fed into
      HierarchicalSoftmaxLoss (Terrier-style).
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
        pool: str = "bos",                   # "bos" (global-attended) or "mean"
        # global_attention_mode: str = "bos",          # one of: bos | bos_eos | none | all | stride:N
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.pool         = pool
        self.classification_tree = classification_tree
        self.classification_tree.set_indexes_if_unset()
        self.output_dim   = self.classification_tree.layer_size
        # self.global_attention_mode = global_attention_mode

        # ---- 1. Longformer backbone (the encoder) ----
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
        # IMPORTANT: attribute name MUST be `longformer` so that the
        # MLM-checkpoint keys (longformer.embeddings.* / longformer.encoder.*)
        # load with strict=False on this module.
        self.longformer = LongformerModel(longformer_config, add_pooling_layer=False)

        # ---- 2. Hierarchical output head ----
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

        # ---- 3. Loss ----
        self.hierarchical_loss = HierarchicalSoftmaxLoss(root=self.classification_tree)

    # ----------------------------------------------------------
    # Pooling helpers
    # ----------------------------------------------------------
    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, input_ids, attention_mask, target_node_ids=None,
            global_attention_mask=None):
        """
        If `global_attention_mask` is provided by the caller, use it as-is.
        Otherwise fall back to the model's own `global_attention_mode` setting.
        """
        

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


    # ----------------------------------------------------------
    # Embedding extraction (for UMAP, etc.)
    # ----------------------------------------------------------
    @torch.no_grad()
    def get_latent_embeddings(self, input_ids, attention_mask,
                            global_attention_mask=None):
        """
        Extract pooled sequence embeddings (before the classifier head).
        If `global_attention_mask` is not provided, defaults to BOS-only.
        """
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1   # BOS

        out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        h = out.last_hidden_state
        if self.pool == "bos":
            return h[:, 0, :]
        return self._mean_pool(h, attention_mask)