import torch
import torch.nn as nn

class CNNSubModule(nn.Module):
    """
    CNN Sub-module — Section 2.3.2 du papier
    
    2 couches Conv1D + BatchNorm + ReLU + connexion résiduelle
    Opère sur la dimension temporelle pour extraire des patterns locaux
    
    Input  : (B, T, C)
    Output : (B, T, C)  — même shape grâce au résidu
    """
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2   # conserve la dimension temporelle

        self.conv1 = nn.Conv1d(in_channels, in_channels,
                               kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(in_channels, in_channels,
                               kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(in_channels)

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C)
        """
        # (B, T, C) → (B, C, T) pour Conv1D
        out = x.permute(0, 2, 1)
        residual = out

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        # connexion résiduelle — évite le vanishing gradient
        out = self.relu(out + residual)

        # (B, C, T) → (B, T, C)
        return out.permute(0, 2, 1)