import torch
import torch.nn as nn
import math

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss
from hierarchicalsoftmax.inference import (
    greedy_predictions,
    node_probabilities,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


# =============================================================================
# 1. BUILD THE CLASSIFICATION TREE
# =============================================================================

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


# =============================================================================
# 2. POSITIONAL ENCODING (unchanged from your original)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard Positional Encoding for batch_first=True (B, L, D)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 3. THE UNIFIED HIERARCHICAL MODEL
# =============================================================================

class HierarchicalTransformerClassifier(nn.Module):
    """
    A single Transformer encoder model with a hierarchical softmax output.

    TRAINING:
        The model outputs raw logits of shape (B, root.layer_size).
        HierarchicalSoftmaxLoss receives these raw logits + target node IDs.
        The loss internally walks the tree from each target leaf up to root,
        slicing the logits at each level and computing local cross-entropy.
        No softmax or tree-walking happens in the model forward pass.

    INFERENCE:
        The raw logits are post-processed outside the model:
        1. node_probabilities() applies local softmax at each tree node
           and multiplies by parent probabilities.
        2. greedy_predictions() walks the tree top-down, optionally
           stopping early if below a confidence threshold.
    """

    def __init__(
        self,
        src_vocab_size: int,
        classification_tree: SoftmaxNode,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        pad_token_id: int = 0,
        num_layers: int = 4,
        classifier_hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_vocab_size = src_vocab_size + 1
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.classification_tree = classification_tree

        # Make sure indexes are set
        self.classification_tree.set_indexes_if_unset()

        # Output dim is determined by the tree — total slots for all
        # local softmax groups across all internal nodes.
        self.output_dim = self.classification_tree.layer_size

        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=max_position_embeddings
        )

        # ---- 2. Transformer Encoder ----
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=True,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # ---- 3. Output Head ----
        # Maps from d_model → root.layer_size (raw logits for the tree).
        # This replaces both your old SimpleDenseClassifier heads.
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

        # ---- 4. Loss Function ----
        # This is the ONLY place the tree structure matters during training.
        # It receives raw logits and target node IDs — no softmax needed.
        self.hierarchical_loss = HierarchicalSoftmaxLoss(
            root=self.classification_tree
        )

    def _mean_pooling(self, h, src_key_padding_mask):
        """Mean-pool the sequence embeddings, ignoring PAD tokens."""
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, tokens, src_key_padding_mask, target_node_ids=None):
        """
        TRAINING forward pass — mirrors how Terrier/Corgi works:
        model outputs raw logits, loss function handles the tree.

        Args:
            tokens:               (B, L) token IDs
            src_key_padding_mask: (B, L) True where padded
            target_node_ids:      (B,) integer node IDs into root.node_list.
                                  Pass None at inference time.

        Returns:
            dict with:
                'total_loss': scalar (0.0 if target_node_ids is None)
                'logits':     (B, root.layer_size) RAW logits — no softmax applied
        """
        # ---- Encode ----
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        seq_z = self._mean_pooling(h, src_key_padding_mask)

        # ---- Raw logits ----
        # Just a linear projection. NO softmax. NO tree-walking.
        # Each slice of this vector corresponds to a local softmax group,
        # but the model doesn't need to know that.
        logits = self.output_head(seq_z)  # (B, root.layer_size)

        # ---- Loss (training only) ----
        # HierarchicalSoftmaxLoss walks the tree internally:
        # For each sample, it goes from the target leaf up to root,
        # slicing logits[parent.start:parent.end] at each level and
        # computing cross_entropy against the child index.
        total_loss = torch.tensor(0.0, device=tokens.device)
        if target_node_ids is not None:
            total_loss = self.hierarchical_loss(logits, target_node_ids)

        return {
            "total_loss": total_loss,
            "logits": logits,
        }

    def get_latent_embeddings(self, tokens, src_key_padding_mask=None):
        """
        Extract d_model-dimensional mean-pooled representations.
        These are the encoder outputs BEFORE the classification head.

        Args:
            tokens:               (B, L) token IDs
            src_key_padding_mask: (B, L) True where padded. Optional.

        Returns:
            (B, d_model) mean-pooled embeddings
        """
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        return self._mean_pooling(h, src_key_padding_mask)


# =============================================================================
# 4. INFERENCE UTILITIES (called OUTSIDE the model, similar to the Terrier does)
# =============================================================================

def hierarchical_predict(
    model: HierarchicalTransformerClassifier,
    tokens,
    src_key_padding_mask,
    threshold: float = 0.7,
):
    """
    Inference function — separate from model.forward(), just like Terrier's
    output_results() is separate from its training loop.

    Steps:
        1. Get raw logits from model (no softmax)
        2. node_probabilities() applies local softmax at each tree node
           and multiplies by parent probability
        3. greedy_predictions() walks tree top-down with threshold

    Args:
        model: trained HierarchicalTransformerClassifier
        tokens: (B, L) token IDs
        src_key_padding_mask: (B, L) True where padded
        threshold: confidence threshold (default 0.7, like Terrier)

    Returns:
        prediction_nodes: list of SoftmaxNode per sample
                          (could be order-level OR superfamily-level)
        probabilities:    (B, root.layer_size) per-node probabilities
    """
    model.eval()
    with torch.no_grad():
        outputs = model(tokens, src_key_padding_mask, target_node_ids=None)
        logits = outputs["logits"]

        # Step 2: Convert raw logits → per-node probabilities
        # This is what Terrier does in output_results():
        #   classification_probabilities = inference.node_probabilities(
        #       results[0], root=self.classification_tree
        #   )
        probs = node_probabilities(logits, root=model.classification_tree)

        # Step 3: Walk tree greedily with threshold
        # This is what Terrier does in output_results():
        #   greedy_predictions = inference.greedy_predictions(
        #       classification_probabilities, root=..., threshold=threshold
        #   )
        prediction_nodes = greedy_predictions(
            probs, root=model.classification_tree, threshold=threshold
        )

    return prediction_nodes, probs


def node_lineage_string(node) -> str:
    """Convert a SoftmaxNode to its full lineage path string."""
    if node.is_root:
        return "Unknown"
    return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])


# =============================================================================
# 5. HELPER: CONVERT YOUR EXISTING LABELS TO NODE IDS
# =============================================================================

def build_label_to_node_id(root: SoftmaxNode) -> dict:
    """
    Builds a mapping from node name strings → node_id integers.

    These integer IDs are what you pass as target_node_ids during training.
    They index into root.node_list, which is what HierarchicalSoftmaxLoss uses.
    """
    root.set_indexes_if_unset()
    label_to_id = {}
    for node_id, node in enumerate(root.node_list):
        # Full path (e.g., "LINE/CR1")
        if node.parent and not node.parent.is_root:
            full_name = "/".join(
                [str(n) for n in node.ancestors[1:]] + [str(node)]
            )
        else:
            full_name = str(node)
        label_to_id[full_name] = node_id

        # Short name if unambiguous
        short_name = str(node)
        if short_name not in label_to_id:
            label_to_id[short_name] = node_id

    return label_to_id




ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'LINE': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'SINE': ['ID', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MULE', 'PIF', 'TcMar', 'hAT']}




if __name__ == "__main__":
    # --- Build tree ---
    root = build_classification_tree(ORDER_TO_SUPERFAMILIES)
    root.render(print=True)
    print(f"Tree layer_size (output dim): {root.layer_size}")

    # --- Create model ---
    model = HierarchicalTransformerClassifier(
        src_vocab_size=1000,
        classification_tree=root,
        d_model=256,
        n_heads=8,
        num_layers=4,
        classifier_hidden_dim=256,
    ).to(DEVICE)
    print(f"Model output dim: {model.output_dim}")

    # --- Build label mapping ---
    label_map = build_label_to_node_id(root)

    # --- Simulate TRAINING step ---
    B, L = 4, 128
    tokens = torch.randint(0, 1000, (B, L)).to(DEVICE)
    pad_mask = torch.zeros(B, L, dtype=torch.bool).to(DEVICE)

    target_names = ["LINE/L1", "SINE/Alu", "DNA/hAT", "LTR/Gypsy"]
    target_ids = torch.tensor(
        [label_map[name] for name in target_names], dtype=torch.long
    ).to(DEVICE)

    # Training: model.forward() returns raw logits, loss handles the tree
    outputs = model(tokens, pad_mask, target_node_ids=target_ids)
    print(f"\nTraining loss:  {outputs['total_loss'].item():.4f}")
    print(f"Logits shape:   {outputs['logits'].shape}")  # (4, root.layer_size)

    # --- Simulate INFERENCE step (separate from forward!) ---
    pred_nodes, probs = hierarchical_predict(
        model, tokens, pad_mask, threshold=0.7
    )
    for i, node in enumerate(pred_nodes):
        lineage = node_lineage_string(node)
        depth = node.depth
        print(f"  Sample {i}: predicted '{lineage}' (depth {depth})")