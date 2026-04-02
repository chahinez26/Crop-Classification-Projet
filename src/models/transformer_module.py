import torch
import torch.nn as nn
from .alpe import ALPEModule

class TransformerSubModule(nn.Module):
    """
    Transformer Sub-module — Section 2.3.1 du papier
    
    Encoder Transformer standard + ALPE pour le premier stage
    Capture les dépendances temporelles longues
    
    Input  : (B, T, d_model)
    Output : (B, T, d_model)
    """
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        max_len:     int = 36,
        use_alpe:    bool = True,   # True seulement pour le 1er stage
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.use_alpe = use_alpe

        # ALPE — uniquement dans le 1er stage (comme dans le papier)
        if use_alpe:
            self.alpe = ALPEModule(d_model, max_len)
        else:
            # encodage positionnel standard pour les stages suivants
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_len, d_model) * 0.01
            )

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B, T, C) au lieu de (T, B, C)
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (B, T) — 1=valide, 0=manquant (utilisé seulement si use_alpe)
        """
        # 1. encodage positionnel
        if self.use_alpe and mask is not None:
            pos_enc = self.alpe(x, mask)
        else:
            pos_enc = self.pos_embedding[:, :x.size(1), :]

        x = x + pos_enc

        # 2. Multi-Head Self-Attention + résidu
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # 3. Feed-Forward + résidu
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x