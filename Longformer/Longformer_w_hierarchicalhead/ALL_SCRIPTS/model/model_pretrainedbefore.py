import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 
import math


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Baselines



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



class BaselineTransformer_Vanilla_MLM(nn.Module):
    """
    Simple baseline: Embedding → 1 Transformer Encoder Layer → Order Classifier.
    No k-mers, no topology, no pooling/unpooling.
    """
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, num_layers=8,
                 dim_feedforward=1024, dropout=0.1,
                 positional_encoding="sinusoidal", max_position_embeddings=512,
                 pad_token_id=0, ignore_index=None):
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.positional_encoding=positional_encoding 
        # Output only needs to predict real tokens (0 through 5)
        self.output_vocab_size = src_vocab_size
        self.dropout = nn.Dropout(dropout)

        self.src_vocab_size = src_vocab_size + 1  # +1 for MASK
        self.d_model = d_model
        
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        
        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)
        
        # ---- 2. Transformer Encoder Blocks ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # ---- 3. Output Head ----
        self.sequence_head = nn.Linear(d_model, self.output_vocab_size)

    def _create_sinusoidal_positions(self, num_positions, dim):
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) *
                             (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, batch):
        """
        tokens: (B, L) — may contain MASK_TOKEN_ID
        mlm_labels: (B, L) with -100 for non-masked (ignored) positions
        """


        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        mlm_labels = batch['mlm_labels']
        B, seq_len = tokens.shape
        
        # ---- Embedding + Positional Encoding ----
        h = self.src_embed(tokens) * math.sqrt(self.d_model)

        h = self.pos_encoder(h)

        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        mlm_loss = None
        logits = self.sequence_head(h)  # (B,L,V)

        if mlm_labels is not None:
            # flatten to (B*L, V), (B*L,)
            logits_flat = logits.view(-1, self.output_vocab_size)
            labels_flat = mlm_labels.view(-1)

            # compute loss only on masked positions
            mlm_loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,  # ignore non-mask positions
                reduction="mean"
            )

        return mlm_loss,logits
    def get_latent_embeddings(self, batch):
        """
        Passes tokens through the encoder and computes a mean-pooled 
        representation across the sequence, ignoring PAD tokens.
        """
        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
        # 1. Embeddings
        h = self.src_embed(tokens) * math.sqrt(self.d_model)

        h = self.pos_encoder(h)

        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
                
        # 3. Mean Pooling (Ignore padding tokens)
        if src_key_padding_mask is not None:
            # src_key_padding_mask is True for padding. We want True for valid tokens.
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9) # Avoid division by zero
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled




class BertMLMHead(nn.Module):
    """
    Standard BERT-style Masked Language Modeling Head.
    Includes a dense layer, GELU activation, LayerNorm, and the final vocabulary projection.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class BaselineTransformer_BERTSTyle_MLM(nn.Module):
    """
    Simple baseline: Embedding → 3 Transformer Encoder Layers → BERT MLM Head.
    """
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, num_layers=3, # <-- Added num_layers
                 dim_feedforward=1024, dropout=0.1,
                 positional_encoding="sinusoidal", max_position_embeddings=512,
                 pad_token_id=0, ignore_index=-100):
        super().__init__()
        self.src_vocab_size = src_vocab_size + 1  # +1 for MASK token
        self.d_model = d_model
        self.positional_encoding_type = positional_encoding
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        
        # ---- Embedding ----
        self.src_embed = nn.Embedding(self.src_vocab_size, d_model)
        
        if positional_encoding == "learned":
            self.pos_embed = nn.Embedding(max_position_embeddings, d_model)
        else:
            self.register_buffer(
                "pos_enc",
                self._create_sinusoidal_positions(max_position_embeddings, d_model),
                persistent=False
            )
            self.pos_embed = None
        
        self.dropout = nn.Dropout(dropout)
        
        # ---- 3 Encoder Layers ----
        # 1. Define the single layer architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        
        # 2. Stack it num_layers times (e.g., 3)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # ---- BERT-style Sequence Head ----
        self.sequence_head = BertMLMHead(d_model, self.src_vocab_size)
        
        # Weight Tying (Optional but recommended for BERT)
        self.sequence_head.decoder.weight = self.src_embed.weight

    def _create_sinusoidal_positions(self, num_positions, dim):
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, batch):
        """
        batch expects:
            'tokens': (B, L)
            'src_key_padding_mask': (B, L) boolean tensor (True for padding)
            'mlm_labels': (B, L) with -100 for non-masked positions
        """
        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        mlm_labels = batch.get('mlm_labels', None)
        B, seq_len = tokens.shape
        
        # ---- Embedding + Positional Encoding ----
        h = self.src_embed(tokens)
        
        if self.positional_encoding_type == "learned":
            pos_indices = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0).expand(B, seq_len)
            h = h + self.pos_embed(pos_indices)
        else:
            h = h + self.pos_enc[:seq_len, :].unsqueeze(0)
        
        h = self.dropout(h)

        # ---- 3 Transformer Layers ----
        # Pass the hidden states through the stacked encoder
        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        # ---- MLM Prediction ----
        logits = self.sequence_head(h)  # (B, L, V)
        mlm_loss = None

        if mlm_labels is not None:
            # Flatten to (B*L, V) and (B*L,)
            logits_flat = logits.view(-1, self.src_vocab_size)
            labels_flat = mlm_labels.view(-1)

            # Compute loss only on masked positions using ignore_index
            mlm_loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=self.ignore_index,
                reduction="mean"
            )

        return mlm_loss, logits
    def get_latent_embeddings(self, batch):
        """
        Passes tokens through the encoder and computes a mean-pooled 
        representation across the sequence, ignoring PAD tokens.
        """
        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
        # 1. Embeddings
        h = self.src_embed(tokens)
        B, seq_len = tokens.shape
        if self.positional_encoding_type == "learned":
            pos_indices = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0).expand(B, seq_len)
            h = h + self.pos_embed(pos_indices)
        else:
            h = h + self.pos_enc[:seq_len, :].unsqueeze(0)
        h = self.dropout(h)
        
        # 2. Transformer
        if hasattr(self, 'transformer_encoder'):
            h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # Fallback if using manual layer loop
            for layer in getattr(self, 'layers', getattr(self, 'encoder_layer', [])):
                h = layer(h, src_key_padding_mask=src_key_padding_mask)
                
        # 3. Mean Pooling (Ignore padding tokens)
        if src_key_padding_mask is not None:
            # src_key_padding_mask is True for padding. We want True for valid tokens.
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9) # Avoid division by zero
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled



















class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # (L,1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer → not trainable, moves with device
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(max_seq_len, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens) * self.scale                  # (B,L,D)
        # pos = torch.arange(L, device=tokens.device).unsqueeze(0) # (1,L)
        # x = x + self.pos_emb(pos)                                # (B,L,D)
        x = self.pos_emb(x)
        return x
    
class TransformerEmbedding_learnable(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        # self.pos_emb = SinusoidalPositionalEmbedding(max_seq_len, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens) * self.scale                  # (B,L,D)
        pos = torch.arange(L, device=tokens.device).unsqueeze(0) # (1,L)
        x = x + self.pos_emb(pos)                                # (B,L,D)
        # x = self.pos_emb(x)
        return x






#--------------------------------------- Baseline 3 -----------------------------------

# MLM Baseline
import torch.nn.functional as F

class BaselineTransformer_MLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size + 1  # +1 for MASK
        self.embed = TransformerEmbedding(vocab_size + 1, d_model, max_seq_len)
        # print("Vocab size for MLM:", self.vocab_size)
        # print("Vocab size for MLM+1:", self.vocab_size+1)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.sequence_head = nn.Linear(d_model, self.vocab_size)

    def forward(self, tokens, mlm_labels=None, src_key_padding_mask=None):
        """
        tokens: (B, L) — may contain MASK_TOKEN_ID
        mlm_labels: (B, L) with -100 for non-masked (ignored) positions
        """
        h = self.embed(tokens)

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        mlm_loss = None
        logits = self.sequence_head(h)  # (B,L,V)

        if mlm_labels is not None:
            # flatten to (B*L, V), (B*L,)
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = mlm_labels.view(-1)

            # compute loss only on masked positions
            mlm_loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,  # ignore non-mask positions
                reduction="mean"
            )

        return mlm_loss,logits

    # to extract embeddings from the last layer's hiddden states
    def get_embeddings(self, x, src_key_padding_mask=None):
        """
        Get embeddings from the last transformer layer
        Useful for extracting representations for visualization/downstream tasks
        
        Args: 
            x: Token indices (B, L)
            src_key_padding_mask: Attention mask where True means "ignore this position"
        
        Returns:
            h: Hidden states (B, L, d_model)
        """
        # Embedding layer
        h = self.embed(x)
        with torch.no_grad():
            # Pass through all transformer encoder layers
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=src_key_padding_mask)
        
        return h
























# get hidden states
def get_embeddings(self, x, attention_mask=None):
    """
    Get embeddings from the last transformer layer without prediction head
    Returns: (batch_size, seq_len, d_model)
    """
    # Get transformer output (before any classification heads)
    hidden_states = self.transformer(x, src_key_padding_mask=attention_mask)
    return hidden_states  # Shape: [batch_size, seq_len, d_model]








#--------------------------------------- Baseline 2 -----------------------------------
# Tasked Baseline
class BaselineTransformer_Tasked(nn.Module):
    """
    Transformer encoder with a sequence reconstruction task head.
    No topology, no contrastive loss.
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.embed = TransformerEmbedding(vocab_size, d_model, max_seq_len)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Task head (readout, NOT a decoder)
        self.sequence_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, target_tokens=None, src_key_padding_mask=None):
        """
        tokens: (B, L)
        target_tokens: (B, L) or None

        returns:
            loss (or None),
            h: (B, L, D),
            recon_loss (or None)
        """
        h = self.embed(tokens)

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        recon_loss = None
        logits = self.sequence_head(h)  # (B, L, V)
        if target_tokens is not None:
            

            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = target_tokens.view(-1)

            pad_mask_flat = src_key_padding_mask.view(-1)
            valid = (targets_flat >= 0) & (~pad_mask_flat)
            if valid.any():
                recon_loss = F.cross_entropy(
                    logits_flat[valid],
                    targets_flat[valid],
                    reduction="mean",
                )
            else:
                recon_loss = torch.tensor(0.0, device=h.device)

        return recon_loss,logits
    























#--------------------------------------- Baseline 1 -----------------------------------
## True baseline
class BaselineTransformer(nn.Module):
    """
    Plain Transformer encoder baseline.
    No task head, no loss, no topology.
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()

        # self.embed = TransformerEmbedding_learnable(vocab_size, d_model, max_seq_len)
        self.vocab_size = vocab_size
        self.embed = TransformerEmbedding(vocab_size, d_model, max_seq_len)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.sequence_head = nn.Linear(d_model, vocab_size)
    def forward(self, tokens, src_key_padding_mask=None):
        """
        tokens: (B, L)
        returns:
            h: (B, L, D) final encoder representations
        """
        h = self.embed(tokens)

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        # We will see the logits for reconstruction of original tokens as the baseline

        recon_loss = None
        if tokens is not None:
            logits = self.sequence_head(h)  # (B, L, V)

            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = tokens.view(-1)


            pad_mask_flat = src_key_padding_mask.view(-1)
            valid = (targets_flat >= 0) & (~pad_mask_flat)
            if valid.any():
                recon_loss = F.cross_entropy(
                    logits_flat[valid],
                    targets_flat[valid],
                    reduction="mean",
                )
            else:
                recon_loss = torch.tensor(0.0, device=h.device)

        return recon_loss,logits
















#-----------------------------------------------------------get hidden states ---------------------------------------------
# Add this method to your model classes
def get_embeddings(self, x, attention_mask=None):
    """
    Get embeddings from the last transformer layer without prediction head
    Returns: (batch_size, seq_len, d_model)
    """
    # Get transformer output (before any classification heads)
    hidden_states = self.transformer(x, src_key_padding_mask=attention_mask)
    return hidden_states  # Shape: [batch_size, seq_len, d_model]