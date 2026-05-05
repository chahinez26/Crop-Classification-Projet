"""
STEP 2 - Improved Part 3 model for California
=============================================
"""

import math
import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(self, n_features=15, reduction=3):
        super().__init__()
        hidden = max(n_features // reduction, 8)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        weights = self.net(pooled).unsqueeze(1)
        return x * weights


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x.transpose(1, 2))
        y = self.conv(y.transpose(1, 2))
        y = self.sigmoid(y)
        return x * y


class ALPE(nn.Module):
    def __init__(self, n_timesteps=36, d_model=36, kernel_size=3):
        super().__init__()
        pe = torch.zeros(n_timesteps, d_model)
        position = torch.arange(0, n_timesteps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.eca = ECA(d_model)

    def forward(self, x, mask):
        bsz = x.size(0)
        pe = self.pe.unsqueeze(0).expand(bsz, -1, -1)
        pe = pe * (1.0 - mask.unsqueeze(-1))
        pe = self.conv(pe.transpose(1, 2)).transpose(1, 2)
        pe = self.eca(pe)
        return x + pe


class TransformerSubmodule(nn.Module):
    def __init__(self, d_model=36, n_head=6, use_alpe=False, kernel_size=3, n_timesteps=36, dropout=0.12):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(n_timesteps=n_timesteps, d_model=d_model, kernel_size=kernel_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            x = self.alpe(x, mask)

        key_padding_mask = mask.bool() if mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class CNNSubmodule(nn.Module):
    def __init__(self, d_model=36, kernel_size=3, dropout=0.12):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = x.transpose(1, 2)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out.transpose(1, 2)
        return self.relu(out + residual)


class CTFusion(nn.Module):
    def __init__(self, d_model=36, n_head=6, kernel_size=3, use_alpe=False, n_timesteps=36, dropout=0.12):
        super().__init__()
        self.cnn = CNNSubmodule(d_model=d_model, kernel_size=kernel_size, dropout=dropout)
        self.transformer = TransformerSubmodule(
            d_model=d_model,
            n_head=n_head,
            use_alpe=use_alpe,
            kernel_size=kernel_size,
            n_timesteps=n_timesteps,
            dropout=dropout,
        )
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, x, mask=None):
        cnn_out = self.cnn(x)
        trans_out = self.transformer(x, mask)
        fused = torch.cat([cnn_out, trans_out], dim=-1)
        return self.fusion(fused)


class Part3CaliforniaNet(nn.Module):
    def __init__(
        self,
        n_features=15,
        n_timesteps=36,
        n_classes=6,
        n_stage=3,
        n_head=6,
        kernel_size=3,
        d_model=36,
        dropout=0.12,
    ):
        super().__init__()
        self.n_stage = n_stage
        self.feature_gate = FeatureAttention(n_features=n_features)
        self.input_embedding = nn.Linear(n_features, d_model)

        self.stages = nn.ModuleList()
        t = n_timesteps
        for s in range(n_stage):
            self.stages.append(
                CTFusion(
                    d_model=d_model,
                    n_head=n_head,
                    kernel_size=kernel_size,
                    use_alpe=(s == 0),
                    n_timesteps=t,
                    dropout=dropout,
                )
            )
            t = t // 2

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x, mask):
        out = self.feature_gate(x)
        out = self.input_embedding(out)

        current_mask = mask
        for s, stage in enumerate(self.stages):
            stage_mask = current_mask if s == 0 else None
            out = stage(out, stage_mask)
            if s < self.n_stage - 1:
                out = self.pool(out.transpose(1, 2)).transpose(1, 2)
                if s == 0:
                    current_mask = current_mask[:, ::2]

        out = out.max(dim=1).values
        return self.classifier(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Part3CaliforniaNet().to(device)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")
