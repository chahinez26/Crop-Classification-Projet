import torch
import torch.nn as nn
import math

class ALPEModule(nn.Module):
    """
    Attention-based Learnable Positional Encoding (ALPE)
    Section 2.3.1 du papier — gère les données manquantes
    
    Input:
        x    : (B, T, d_model) — features temporelles
        mask : (B, T)          — 1=valide, 0=manquant
    Output:
        (B, T, d_model) — encodage positionnel enrichi
    """
    def __init__(self, d_model: int, max_len: int = 36):
        super().__init__()
        self.d_model = d_model

        # encodage positionnel sinusoïdal absolu (base)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)   # (T, d_model)

        # conv1D + ECA pour apprendre l'importance de chaque position
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=1
        )

        # ECA — Efficient Channel Attention (Wang et al. 2020)
        self.eca_pool = nn.AdaptiveAvgPool1d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.eca_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (B, T) — 1=valide, 0=manquant
        """
        B, T, _ = x.shape

        # 1. encodage positionnel absolu → (B, T, d_model)
        pe = self.pe[:T, :].unsqueeze(0).expand(B, -1, -1)

        # 2. masquage des positions manquantes
        mask_expanded = mask.unsqueeze(-1).float()   # (B, T, 1)
        pe_masked = pe * mask_expanded               # positions manquantes = 0

        # 3. Conv1D sur la dimension temporelle
        # (B, T, d_model) → (B, d_model, T) pour Conv1D
        pe_conv = self.conv(pe_masked.permute(0, 2, 1))   # (B, d_model, T)

        # 4. ECA — calcule l'importance de chaque canal
        eca_weights = self.eca_pool(pe_conv)              # (B, d_model, 1)
        eca_weights = self.eca_conv(
            eca_weights.permute(0, 2, 1)                  # (B, 1, d_model)
        )
        eca_weights = self.eca_sigmoid(eca_weights)       # (B, 1, d_model)

        # 5. application des poids → (B, T, d_model)
        alpe_out = pe_conv.permute(0, 2, 1) * eca_weights

        return alpe_out