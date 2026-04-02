import torch
import torch.nn as nn
from .cnn_module import CNNSubModule
from .transformer_module import TransformerSubModule

class CTFusionModule(nn.Module):
    """
    CNN-Transformer Fusion Module — Figure 3 du papier
    
    Les deux sous-modules travaillent en parallèle,
    leurs sorties sont concaténées puis réduites par Max Pooling
    
    Input  : (B, T, d_model)
    Output : (B, T//2, d_model*2) après pooling
    """
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        kernel_size: int,
        use_alpe:    bool = False,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.cnn = CNNSubModule(d_model, kernel_size)
        self.transformer = TransformerSubModule(
            d_model, n_heads,
            use_alpe=use_alpe,
            dropout=dropout
        )

        # projection après concaténation CNN + Transformer
        # (B, T, d_model*2) → (B, T, d_model*2)
        self.fusion_proj = nn.Linear(d_model * 2, d_model * 2)

        # Max Pooling temporel — réduit T de moitié
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (B, T)
        """
        # branches parallèles
        cnn_out         = self.cnn(x)                    # (B, T, d_model)
        transformer_out = self.transformer(x, mask)      # (B, T, d_model)

        # concaténation
        fused = torch.cat([cnn_out, transformer_out], dim=-1)  # (B, T, d_model*2)
        fused = self.fusion_proj(fused)

        # max pooling temporel : (B, T, C) → (B, C, T) → pool → (B, C, T//2)
        fused = fused.permute(0, 2, 1)
        fused = self.pool(fused)
        fused = fused.permute(0, 2, 1)   # (B, T//2, d_model*2)

        return fused