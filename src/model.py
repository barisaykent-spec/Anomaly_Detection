import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Anomali tespiti için Autoencoder mimarisi.

    Mimari: 29 → 16 → 8 → 4 (bottleneck) → 8 → 16 → 29

    Args:
        input_dim (int): Girdi feature sayısı. Default: 29
    """

    def __init__(self, input_dim: int = 29):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        """Forward pass: encode → decode"""
        return self.decoder(self.encoder(x))

    def encode(self, x):
        """Sadece encoder — bottleneck temsilini döndürür"""
        return self.encoder(x)