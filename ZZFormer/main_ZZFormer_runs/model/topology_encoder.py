import torch.nn as nn


class TopologyEncoder(nn.Module):
    """
    CNN that maps one k-mer's persistence image (B, C, 128, 128)
    to a topology latent (B, latent_dim).

    Same architecture as the original HierarchicalCNN.Encoder, but the
    final projection size is configurable so it can plug straight into
    the Longformer's topology cross-attention path.
    """

    def __init__(
        self,
        n_channels: int = 5,
        n_filters:  int = 16,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # (B, C, 128, 128) -> (B, n_filters, 64, 64)
            nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # -> (B, 2F, 32, 32)
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # -> (B, 4F, 16, 16)   (4F * 16 * 16 = 16384 when F=16)
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)
    

