import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, n_channels=5, n_filters=16):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # (B,5,128,1024) --> (B,16,64,1024)
            nn.Conv2d(n_channels, n_filters, kernel_size=(3,1), stride=(2, 1), padding=(1,0)),
            nn.ReLU(),
            # (B,16,64,1024) --> (B,32,32,1024)
            nn.Conv2d(n_filters, n_filters*2, kernel_size=(3,1), stride=(2, 1), padding=(1,0)),
            nn.ReLU(),
            # (B,32,32,1024) --> (B,64,16,1024) 
            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=(3,1), stride=(2, 1), padding=(1,0)),
            nn.ReLU(),

            nn.Flatten(start_dim=1, end_dim=2)
        )
    def forward(self, x):
        return self.encoder(x) # (B,64,16,1024) 


class TopologyEncoder(nn.Module):
    def __init__(self, root_node, n_channels=5, n_filters=16):
        super(HierarchicalCNN, self).__init__()
        self.enc4 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc8 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc14 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc20 = Encoder(n_channels=n_channels, n_filters=n_filters)
        # self.flatten = nn.Flatten()
        # combined_size = 64*16*1024
        # self.prediction_head = nn.Sequential(
        #     nn.Linear(combined_size, combined_size // 2), 
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        # )
        # self.hierarchical_layer = HierarchicalSoftmaxLinear(
        #     in_features=512,
        #     root=root_node
        # )
    def forward(self, x4, x8, x14, x20):
        # Encoder: (B, 64,16,16)
        z4 = self.enc4(x4)
        z8 = self.enc8(x8)
        z14 = self.enc14(x14)
        z20 = self.enc20(x20)

        # # Concatenate along channel
        # emb = torch.cat((z4, z8, z14, z20), dim=1)
        # # Prediction
        # classification = self.prediction_head(emb)
        # return self.hierarchical_layer(classification), emb
        return z4, z8, z14, z20


# class TopologyEncoder(nn.Module):
#     """
#     CNN that maps one k-mer's persistence image (B, C, H, W) to a
#     sequence of topology tokens (B, N_tokens, latent_dim).

#     Each output token = one spatial cell of the final conv feature map,
#     linearly projected to `latent_dim`. With three 2x2 poolings on a
#     128x128 input, the feature map is 16x16 = 256 tokens.

#     Set `pooling_stages=4` to halve that to 8x8 = 64 tokens, etc.
#     """

#     def __init__(
#         self,
#         n_channels:     int = 5,
#         n_filters:      int = 16,
#         latent_dim:     int = 512,
#         pooling_stages: int = 3,           # 3 -> 256 tokens, 4 -> 64 tokens, ...
#         return_sequence: bool = True,      # False keeps the old (B, latent_dim) output
#     ):
#         super().__init__()
#         assert pooling_stages >= 1

#         # ----- Conv stack: F, 2F, 4F, 8F, ... -----
#         layers = []
#         in_c = n_channels
#         out_c = n_filters
#         for _ in range(pooling_stages):
#             layers += [
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#             ]
#             in_c  = out_c
#             out_c = out_c * 2

#         self.conv = nn.Sequential(*layers)
#         self.final_channels = in_c                          # channels of the conv output
#         self.return_sequence = return_sequence

#         # ----- Per-cell projection: final_channels -> latent_dim -----
#         # Implemented as a 1x1 conv so it's applied to every spatial cell.
#         self.token_proj = nn.Sequential(
#             nn.Conv2d(self.final_channels, latent_dim, kernel_size=1),
#             nn.GroupNorm(1, latent_dim),                    # = LayerNorm over channels
#             nn.GELU(),
#         )

#         # Compatibility path: still expose a single-vector output if needed.
#         if not return_sequence:
#             self.pool = nn.AdaptiveAvgPool2d(1)             # (B, latent_dim, 1, 1)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         feat = self.conv(x)                                 # (B, final_channels, H', W')
#         tok  = self.token_proj(feat)                        # (B, latent_dim,    H', W')

#         if not self.return_sequence:
#             return self.pool(tok).flatten(1)                # (B, latent_dim)

#         # Spatial -> sequence: (B, latent_dim, H', W') -> (B, H'*W', latent_dim)
#         B, D, H, W = tok.shape
#         return tok.flatten(2).transpose(1, 2)               # (B, H*W, latent_dim)

  

