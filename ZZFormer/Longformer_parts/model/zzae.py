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





# --- (Keep your SimpleDenseClassifier and PositionalEncoding definitions here) ---

class PersistenceAutoencoder(nn.Module):
    def __init__(self):
        super(PersistenceAutoencoder, self).__init__()

        # --- ENCODER ---
        # Input: [Batch, 5, 128, 128]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),        # Output: [16, 64, 64]

            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),        # Output: [8, 32, 32]
            nn.Flatten()            # Output: [8192]
        )

        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        # --- DECODER ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        latent = self.encoder_fc(x)
        x = self.decoder_fc(latent)
        x = x.view(-1, 8, 32, 32)  
        reconstructed = self.decoder_conv(x)
        return reconstructed

    def get_embeddings(self, x):
        return self.encoder_fc(self.encoder_conv(x))


class OrderFFNtransformer_Classifier(nn.Module):
    def __init__(self, src_vocab_size, d_model=256, n_heads=8, 
                 dim_feedforward=1024, dropout=0.1, max_position_embeddings=512,
                 pad_token_id=0, ignore_index=-100, num_orders=7, 
                 classifier_hidden_dim=64, k_mers=[2, 4, 8], 
                 concatenation_hidden_dim=512, clip_weight=0.1, shared_clip_dim=512):
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.k_mers = k_mers
        self.clip_weight = clip_weight
        self.concatenation_hidden_dim = concatenation_hidden_dim
        self.shared_clip_dim = shared_clip_dim
        
        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)
        
        # ---- 2. Transformer Encoder Blocks ----
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_first=True, batch_first=True,
            ) for _ in self.k_mers
        ])
        
        # ---- 3. Topological CNN Encoders ----
        # Create a dictionary holding a CNN autoencoder for each k-mer
        self.topo_encoders = nn.ModuleDict({
            str(k): PersistenceAutoencoder() for k in self.k_mers
        })

        # ---- 4. CLIP Projection Heads (PROJECTING BOTH TO A SHARED SPACE) ----
        # Project sequence (256 -> 512)
        self.seq_projections = nn.ModuleList([
            nn.Linear(d_model, self.shared_clip_dim) for _ in self.k_mers
        ])
        
        # Project topology (512 -> 512)
        self.kmer_projections = nn.ModuleDict({
            str(k): nn.Linear(self.concatenation_hidden_dim, self.shared_clip_dim) for k in self.k_mers
        })
        
        # Learnable temperature parameter for CLIP
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- 5. Classifier ----
        self.order_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, num_orders, dropout=dropout)

    def _mean_pooling(self, h, src_key_padding_mask):
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, batch):
        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        order_label = batch.get('order_labels', None)

        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        # Safety fix: Always use a tensor for loss to prevent .backward() crashes
        total_clip_loss = 0.0 #torch.tensor(0.0, device=tokens.device)
        seq_z = None 
        
        for i, k in enumerate(self.k_mers):
            # 1. Forward through transformer layer
            h = self.encoder_layers[i](h, src_key_padding_mask=src_key_padding_mask)
            seq_z = self._mean_pooling(h, src_key_padding_mask)
            
            mer_key = f'mer{k}'
            if mer_key in batch and batch[mer_key] is not None:
                # 2. Extract RAW topological input [Batch, 5, 128, 128]
                raw_topo = batch[mer_key]
                
                # 3. Feed forward through the CNN Encoder to get 512-dim embedding
                kmer_emb = self.topo_encoders[str(k)].get_embeddings(raw_topo)
                
                # 4. Project BOTH sequence and topology into shared 512-dim space
                seq_z_proj = self.seq_projections[i](seq_z)
                kmer_z_proj = self.kmer_projections[str(k)](kmer_emb)
                
                # 5. L2 Normalize
                seq_z_norm = F.normalize(seq_z_proj, p=2, dim=-1)
                kmer_z_norm = F.normalize(kmer_z_proj, p=2, dim=-1)
                
                # 6. Calculate Cosine Similarity Logits
                logit_scale = self.logit_scale.exp()
                logits_per_seq = logit_scale * seq_z_norm @ kmer_z_norm.t()  
                logits_per_kmer = logits_per_seq.t()  # Symmetric transpose
                
                # 7. TRUE SYMMETRIC CLIP LOSS
                batch_size = seq_z_norm.shape[0]
                labels = torch.arange(batch_size, device=seq_z_norm.device)
                
                loss_seq = F.cross_entropy(logits_per_seq, labels)
                loss_kmer = F.cross_entropy(logits_per_kmer, labels)
                
                # Average the bidirectional loss and backpropagate it through both encoders
                clip_loss_k = (loss_seq + loss_kmer) / 2
                total_clip_loss += clip_loss_k

        # ---- Final Classification ----
        order_logits = self.order_classifier(seq_z)

        order_loss = 0.0 #torch.tensor(0.0, device=tokens.device)
        if order_label is not None:
            order_loss = F.cross_entropy(order_logits, order_label.long(), ignore_index=self.ignore_index)

        total_loss = order_loss + (self.clip_weight * total_clip_loss)

        return {
            'total_loss': total_loss,
            'order_loss': order_loss,
            'clip_loss': total_clip_loss,
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
                 classifier_hidden_dim=64, k_mers=[2, 4, 8], 
                 concatenation_hidden_dim=512, clip_weight=0.1, shared_clip_dim=512):
        """
        kmer_dims: A dictionary mapping the k-mer value to the dimension of its topological embedding.
                   Example: {2: 16, 4: 256, 8: 65536}
        """
        super().__init__()
        self.input_vocab_size = src_vocab_size + 1
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.k_mers = k_mers
        self.clip_weight = clip_weight
        self.concatenation_hidden_dim = concatenation_hidden_dim
        self.shared_clip_dim = shared_clip_dim
        
        # ---- 1. Embedding & Positional Encoding ----
        self.src_embed = nn.Embedding(self.input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)
        
        
        # ---- 2. Transformer Encoder Blocks (One per k-mer) ----
        # We use a ModuleList so we can extract the output after EACH layer
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            ) for _ in self.k_mers
        ])



        # ---- 3. Topological CNN Encoders ----
        # Create a dictionary holding a CNN autoencoder for each k-mer
        self.topo_encoders = nn.ModuleDict({
            str(k): PersistenceAutoencoder() for k in self.k_mers
        })

        # ---- 4. CLIP Projection Heads (PROJECTING BOTH TO A SHARED SPACE) ----
        # Project sequence (256 -> 512)
        self.seq_projections = nn.ModuleList([
            nn.Linear(d_model, self.shared_clip_dim) for _ in self.k_mers
        ])
        
        # Project topology (512 -> 512)
        self.kmer_projections = nn.ModuleDict({
            str(k): nn.Linear(self.concatenation_hidden_dim, self.shared_clip_dim) for k in self.k_mers
        })
        
        
        # Learnable temperature parameter for CLIP loss (initialized to standard 1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- 4. Classifier ----
        self.sf_classifier = SimpleDenseClassifier(d_model, classifier_hidden_dim, num_superfamilies, dropout=dropout)

    def _mean_pooling(self, h, src_key_padding_mask):
        """Helper function to mean-pool the sequence embeddings."""
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(h)
            
        sum_embeddings = torch.sum(h * valid_mask, dim=1)
        sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, batch):
        tokens = batch['tokens']
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        superfamily_label = batch.get('superfamily_labels', None)

        # ---- Embedding + Positional Encoding ----
        h = self.src_embed(tokens) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        # Safety fix: Always use a tensor for loss to prevent .backward() crashes
        total_clip_loss = 0.0 #torch.tensor(0.0, device=tokens.device)
        seq_z = None 
        
        for i, k in enumerate(self.k_mers):
            # 1. Forward through transformer layer
            h = self.encoder_layers[i](h, src_key_padding_mask=src_key_padding_mask)
            seq_z = self._mean_pooling(h, src_key_padding_mask)
            
            mer_key = f'mer{k}'
            if mer_key in batch and batch[mer_key] is not None:
                # 2. Extract RAW topological input [Batch, 5, 128, 128]
                raw_topo = batch[mer_key]
                
                # 3. Feed forward through the CNN Encoder to get 512-dim embedding
                kmer_emb = self.topo_encoders[str(k)].get_embeddings(raw_topo)
                
                # 4. Project BOTH sequence and topology into shared 512-dim space
                seq_z_proj = self.seq_projections[i](seq_z)
                kmer_z_proj = self.kmer_projections[str(k)](kmer_emb)
                
                # 5. L2 Normalize
                seq_z_norm = F.normalize(seq_z_proj, p=2, dim=-1)
                kmer_z_norm = F.normalize(kmer_z_proj, p=2, dim=-1)
                
                # 6. Calculate Cosine Similarity Logits
                logit_scale = self.logit_scale.exp()
                logits_per_seq = logit_scale * seq_z_norm @ kmer_z_norm.t()  
                logits_per_kmer = logits_per_seq.t()  # Symmetric transpose
                
                # 7. TRUE SYMMETRIC CLIP LOSS
                batch_size = seq_z_norm.shape[0]
                labels = torch.arange(batch_size, device=seq_z_norm.device)
                
                loss_seq = F.cross_entropy(logits_per_seq, labels)
                loss_kmer = F.cross_entropy(logits_per_kmer, labels)
                
                # Average the bidirectional loss and backpropagate it through both encoders
                clip_loss_k = (loss_seq + loss_kmer) / 2
                total_clip_loss += clip_loss_k

        # ---- Final Classification ----
        # FIX 1: We already calculated mean pooling for the last layer (it is stored in seq_z)
        sf_logits = self.sf_classifier(seq_z)

        # ---- Superfamily Classification Loss ----
        sf_loss = 0.0 #torch.tensor(0.0, device=tokens.device)
        if superfamily_label is not None:
            sf_loss = F.cross_entropy(sf_logits, superfamily_label.long(), ignore_index=self.ignore_index)

        # FIX 3: Loss Scaling. You can make 0.1 a hyperparameter passed in __init__
        # This prevents the auxiliary task from destroying the classification task
        clip_weight = self.clip_weight 
        total_loss = sf_loss + (clip_weight * total_clip_loss)

        return {
            'total_loss': total_loss,
            'sf_loss': sf_loss,
            'clip_loss': total_clip_loss,
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
