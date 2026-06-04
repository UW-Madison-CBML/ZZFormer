import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 
import math
import numpy as np

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
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, 
                 dim_feedforward=1024, dropout=0.1, max_position_embeddings=512,
                 pad_token_id=0, ignore_index=-100, num_orders=7, 
                 classifier_hidden_dim=64, num_layers=4):
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.num_layers = num_layers
        self.num_orders = num_orders
        
        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)
        
        # ---- 2. Transformer Encoder Blocks ----
        # FIX 1: Use range() for the integer self.num_layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            ) for _ in range(self.num_layers)
        ])
        
        # ---- 3. Classifier ----
        self.order_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, self.num_orders, dropout=dropout)

    def _mean_pooling(self, h, src_key_padding_mask):
        """Helper function to mean-pool the sequence embeddings."""
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, tokens, src_key_padding_mask, order_label):

        # ---- Embedding + Positional Encoding ----
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        
        # ---- Layer-wise Processing ----
        # FIX 2: Iterate directly over the module list instead of enumerate(self.num_layers)
        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)
            
        seq_z = self._mean_pooling(h, src_key_padding_mask)

        # ---- Final Classification ----
        order_logits = self.order_classifier(seq_z)

        # ---- Order Classification Loss ----
        # Using a tensor here so outputs['order_loss'].item() won't crash if label is None
        order_loss = 0.0
        if order_label is not None:
            order_loss = F.cross_entropy(order_logits, order_label.long(), ignore_index=self.ignore_index)

        # FIX 3: Since this is ablation and there's no CLIP loss, total loss is just order loss.
        # (We remove self.clip_weight since it's no longer initialized)
        # total_loss = order_loss
        
        # Keep total_clip_loss as a zero tensor to prevent .item() crashes downstream in train.py
        # total_clip_loss = torch.tensor(0.0, device=tokens.device)

        return {
            'total_loss': order_loss,
            # 'order_loss': order_loss,
            # 'clip_loss': total_clip_loss,
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

        for encoderlayer in self.encoder_layers:
            # 1. Forward through this specific transformer layer
            h = encoderlayer(h, src_key_padding_mask=src_key_padding_mask)
  
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
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, 
                 dim_feedforward=1024, dropout=0.1, max_position_embeddings=512,
                 pad_token_id=0, ignore_index=-100, num_superfamilies=24,
                 classifier_hidden_dim=64, num_layers=4,):
        """
        kmer_dims: A dictionary mapping the k-mer value to the dimension of its topological embedding.
                   Example: {2: 16, 4: 256, 8: 65536}
        """
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.num_layers = num_layers
        self.num_superfamilies = num_superfamilies
        
        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)
        
        # ---- 2. Transformer Encoder Blocks ----
        # FIX 1: Use range() for the integer self.num_layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            ) for _ in range(self.num_layers)
        ])

        # ---- 4. Classifier ----
        self.sf_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, self.num_superfamilies, dropout=dropout)

    def _mean_pooling(self, h, src_key_padding_mask):
        """Helper function to mean-pool the sequence embeddings."""
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, tokens, src_key_padding_mask, superfamily_label):
        

        # ---- Embedding + Positional Encoding ----
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        
        # Will hold the final sequence embedding after the loop
        seq_z = None 
        
        # FIX 2: Iterate directly over the module list instead of enumerate(self.num_layers)
        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)
            
        seq_z = self._mean_pooling(h, src_key_padding_mask)


        # ---- Final Classification ----
        # FIX 1: We already calculated mean pooling for the last layer (it is stored in seq_z)
        sf_logits = self.sf_classifier(seq_z)

        # ---- Superfamily Classification Loss ----
        sf_loss = 0.0
        if superfamily_label is not None:
            sf_loss = F.cross_entropy(sf_logits, superfamily_label.long(), ignore_index=self.ignore_index)

        return {
            'total_loss': sf_loss,
            # 'sf_loss': sf_loss,
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

        for encoderlayer in self.encoder_layers:
            # 1. Forward through this specific transformer layer
            h = encoderlayer(h, src_key_padding_mask=src_key_padding_mask)

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
