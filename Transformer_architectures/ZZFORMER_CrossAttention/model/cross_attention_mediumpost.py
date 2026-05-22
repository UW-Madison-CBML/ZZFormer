import torch
import torch.nn.functional as F

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
class CustomAttention(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="self"):
        super(CustomAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_type = attention_type

        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, target, source=None, mask=None):
        if self.attention_type == "self":
            Q = self.query(target)
            K = self.key(target)
            V = self.value(target)
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)

        # Perform attention calculation (self or cross)
        out, _ = cross_attention(Q, K, V, mask)
        return self.fc_out(out)
class CustomAttentionWithResidual(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="self"):
        super(CustomAttentionWithResidual, self).__init__()
        self.attention = CustomAttention(embed_size, num_heads, attention_type)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, target, source=None, mask=None):
        attention_out = self.attention(target, source, mask)
        # Add residual connection and layer normalization
        out = self.norm(target + self.dropout(attention_out))
        return out