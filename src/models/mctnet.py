import torch
import torch.nn as nn
from .ctfusion import CTFusionModule

class MCTNet(nn.Module):
    """
    Multi-stage CNN-Transformer Network — Section 2.3 du papier
    Wang et al. (2024)
    
    Architecture :
        Input projection → CTFusion × n_stages → Global MaxPool → MLP
    
    Input :
        x    : (B, T, n_bands)  — séries temporelles spectrales
        mask : (B, T)           — 1=valide, 0=manquant
    Output :
        (B, n_classes)          — probabilités par classe
    """
    def __init__(
        self,
        n_bands:    int = 10,
        n_classes:  int = 5,
        n_stages:   int = 3,
        d_model:    int = 64,
        n_heads:    int = 5,
        kernel_size: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_stages = n_stages

        # projection initiale : n_bands → d_model
        self.input_proj = nn.Linear(n_bands, d_model)

        # n_stages CTFusion modules
        # le 1er utilise ALPE, les suivants non (comme dans le papier)
        self.stages = nn.ModuleList()
        current_dim = d_model

        for stage_idx in range(n_stages):
            use_alpe = (stage_idx == 0)   # ALPE seulement au stage 0
            self.stages.append(
                CTFusionModule(
                    d_model=current_dim,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                    use_alpe=use_alpe,
                    dropout=dropout,
                )
            )
            current_dim = current_dim * 2   # double après chaque CTFusion

        # MLP classifier
        # après n_stages : current_dim = d_model * 2^n_stages
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, n_classes),
            # pas de Softmax ici — CrossEntropyLoss l'inclut déjà
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x    : (B, T, n_bands)
        mask : (B, T)
        """
        # projection initiale
        x = self.input_proj(x)              # (B, T, d_model)

        # passage dans les n stages
        for stage_idx, stage in enumerate(self.stages):
            mask_stage = mask if stage_idx == 0 else None
            x = stage(x, mask_stage)        # (B, T//2^(i+1), d_model*2^(i+1))

        # Global Max Pooling sur la dimension temporelle
        x = x.max(dim=1).values             # (B, current_dim)

        # classification
        logits = self.classifier(x)         # (B, n_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)