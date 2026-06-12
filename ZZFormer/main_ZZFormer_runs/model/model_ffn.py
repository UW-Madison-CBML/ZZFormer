import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 
import math


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


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



class OrderFFNtransformer_Classifier(nn.Module):
    """
    Simple baseline: Embedding → 1 Transformer Encoder Layer → Order Classifier.
    No k-mers, no topology, no pooling/unpooling.
    """
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, num_layers=8,
                 dim_feedforward=1024, dropout=0.1,
                 positional_encoding="sinusoidal", max_position_embeddings=512,
                 pad_token_id=0, ignore_index=None, num_orders=7, classifier_hidden_dim=64, k_mers=[2,4,8],
                 concatenation_hidden_dim=512,):
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.positional_encoding=positional_encoding 
        # Output only needs to predict real tokens (0 through 5)
        self.output_vocab_size = src_vocab_size
        # self.dropout = nn.Dropout(dropout)

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

        # ---- Classifier ----
        self.order_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, num_orders, dropout=dropout)


    def forward(self, batch):
        """
        tokens: (B, L) — may contain MASK_TOKEN_ID
        mlm_labels: (B, L) with -100 for non-masked (ignored) positions
        """


        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        order_label = batch.get('order_labels', None)

        B, seq_len = tokens.shape
        
        # ---- Embedding + Positional Encoding ----
        # h = self.src_embed(tokens)
        h = self.src_embed(tokens) * math.sqrt(self.d_model)

        h = self.pos_encoder(h)

        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        # ---- Mean Pooling (CRITICAL FIX) ----
        # We must reduce (Batch, Seq_Len, d_model) -> (Batch, d_model)
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask # Shape: (Batch, d_model)

        # ---- Classification ----
        # Pass only the pooled vector (No padding mask needed here)
        order_logits = self.order_classifier(mean_pooled)


        # ---- Loss ----
        order_loss = None

        if order_label is not None:
            order_loss = F.cross_entropy(order_logits, order_label.long(), ignore_index=self.ignore_index)

        
        return {
            'total_loss': order_loss,
            'order_logits': order_logits,
        }



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











class SuperfamilyFFNtransformer_Classifier(nn.Module):
    """
    Simple baseline: Embedding → 1 Transformer Encoder Layer → Order Classifier.
    No k-mers, no topology, no pooling/unpooling.
    """
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, num_layers=8,
                 dim_feedforward=1024, dropout=0.1,
                 positional_encoding="sinusoidal", max_position_embeddings=512,
                 pad_token_id=0, ignore_index=None, num_superfamilies=23, classifier_hidden_dim=64, k_mers=[2,4,8],
                 concatenation_hidden_dim=512,):
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.positional_encoding=positional_encoding 
        # Output only needs to predict real tokens (0 through 5)
        self.output_vocab_size = src_vocab_size
        # self.dropout = nn.Dropout(dropout)


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


        # ---- Classifier ----
        self.sf_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, num_superfamilies, dropout=dropout)
        

    def forward(self, batch):
        """
        tokens: (B, L) — may contain MASK_TOKEN_ID
        mlm_labels: (B, L) with -100 for non-masked (ignored) positions
        """


        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        superfamily_label = batch.get('superfamily_labels', None)
        B, seq_len = tokens.shape
        
        # ---- Embedding + Positional Encoding ----
        # h = self.src_embed(tokens)
        h = self.src_embed(tokens) * math.sqrt(self.d_model)

        h = self.pos_encoder(h)


        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        # ---- Classification ----

        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)

        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask # Shape: (Batch, d_model)


        sf_logits = self.sf_classifier(mean_pooled)
        
        # ---- Loss ----
        sf_loss = None

        if superfamily_label is not None:
            sf_loss = F.cross_entropy(sf_logits, superfamily_label.long(), ignore_index=self.ignore_index)

        
        return {
            'total_loss': sf_loss,
            'sf_logits': sf_logits,
        }
    


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
