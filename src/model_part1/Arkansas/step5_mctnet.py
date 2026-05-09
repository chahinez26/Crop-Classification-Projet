import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=k_size // 2, bias=False) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        
        y = self.avg_pool(x.transpose(1, 2))   
        y = self.conv(y.transpose(1, 2))         
        y = self.sigmoid(y)                      
        return x * y                             





class ALPE(nn.Module):
    def __init__(self, n_timesteps=36, d_model=30, kernel_size=3):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.d_model     = d_model

        
        pe = torch.zeros(n_timesteps, d_model)
        position = torch.arange(0, n_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)   

        
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.eca  = ECA(d_model)

    def forward(self, x, mask):
        """
        x    : [B, T, d_model] # d_model : nbr de neuronnes pour représenter chaque timestep
        mask : [B, T]  — 1=missing, 0=présent
        """
        B, T, C = x.shape
        pe = self.pe.unsqueeze(0).expand(B, -1, -1)   

        
        mask_2d  = mask.unsqueeze(-1).expand_as(pe)
        pe_masked = pe * (1 - mask_2d)

        
        pe_conv = self.conv(pe_masked.transpose(1, 2)).transpose(1, 2) 
        pe_final = self.eca(pe_conv) 

        return x + pe_final





class TransformerSubmodule(nn.Module):
    def __init__(self, d_model=30, n_head=5, use_alpe=False,
                 kernel_size=3, n_timesteps=36, dropout=0.1):
        super().__init__()
        self.use_alpe = use_alpe

        if use_alpe:
            self.alpe = ALPE(n_timesteps, d_model, kernel_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            x = self.alpe(x, mask)

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x





class CNNSubmodule(nn.Module):
    def __init__(self, d_model=30, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(d_model)
        self.relu  = nn.ReLU()

    def forward(self, x):
        residual = x
        out = x.transpose(1, 2)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out = out.transpose(1, 2)
        return self.relu(out + residual)





class CTFusion(nn.Module):
    def __init__(self, d_model=30, n_head=5, kernel_size=3,
                 use_alpe=False, n_timesteps=36, dropout=0.1):
        super().__init__()
        self.cnn         = CNNSubmodule(d_model, kernel_size)
        self.transformer = TransformerSubmodule(
            d_model, n_head, use_alpe, kernel_size, n_timesteps, dropout
        )
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, x, mask=None):
        cnn_out   = self.cnn(x)
        trans_out = self.transformer(x, mask)
        fused = torch.cat([cnn_out, trans_out], dim=-1)
        return self.fusion(fused)














class MCTNet(nn.Module):
    def __init__(self,
                 n_bands=10,
                 n_timesteps=36,
                 n_classes=5,
                 n_stage=3,
                 n_head=5,
                 kernel_size=3,
                 d_model=30,     
                 dropout=0.1):
        super().__init__()

        self.n_stage = n_stage

        self.input_embedding = nn.Linear(n_bands, d_model)

        
        self.stages = nn.ModuleList()
        t = n_timesteps
        for s in range(n_stage):
            use_alpe = (s == 0)
            self.stages.append(
                CTFusion(d_model, n_head, kernel_size,
                         use_alpe=use_alpe, n_timesteps=t, dropout=dropout)
            )
            t = t // 2

        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask):
        """
        x    : [B, T, n_bands]
        mask : [B, T]
        """
        
        out = self.input_embedding(x)   

        for s, stage in enumerate(self.stages):
            m = mask if s == 0 else None
            out = stage(out, m)

            if s < self.n_stage - 1:
                out  = self.pool(out.transpose(1, 2)).transpose(1, 2)
                if s == 0:
                    mask = mask[:, ::2]

        
        out = out.max(dim=1).values   

        return self.classifier(out)   





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(device='cpu'):
    model = MCTNet(
        n_bands=10, n_timesteps=36, n_classes=5,
        n_stage=3, n_head=5, kernel_size=3, d_model=30, dropout=0.1
    ).to(device)

    n_params = count_parameters(model)
    print(f"MCTNet Arkansas — {n_params:,} paramètres entraînables")
    print(f"  (papier : 55 059 paramètres)")
    if abs(n_params - 55059) / 55059 < 0.01:
        print(f"  ✅ Quasi-identique au papier (< 1% d'écart)")
    elif abs(n_params - 55059) / 55059 < 0.05:
        print(f"  ✅ Très proche du papier (< 5% d'écart)")
    else:
        print(f"  ⚠  Différence notable : {n_params - 55059:+d}")
    return model


def test_forward_pass(device='cpu'):
    print("\n── Test forward pass ───────────────────────────────")
    model = build_model(device)
    model.eval()

    B = 4
    x    = torch.randn(B, 36, 10).to(device)
    mask = torch.zeros(B, 36).to(device)
    mask[:, 3]  = 1
    mask[:, 15] = 1

    with torch.no_grad():
        logits = model(x, mask)

    print(f"  Input  x    : {list(x.shape)}")
    print(f"  Input  mask : {list(mask.shape)}")
    print(f"  Output logits : {list(logits.shape)}  (attendu : [{B}, 5])")
    probs = F.softmax(logits, dim=-1)
    print(f"  Probs (somme) : {probs.sum(dim=-1).tolist()}  (attendu : [1.0, ...])")
    print(f"  ✅ Forward pass OK\n")
    return model


if __name__ == "__main__":
    print("=" * 55)
    print("Étape 5 — Architecture MCTNet (v2 corrigée)")
    print("=" * 55 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    model = test_forward_pass(device)

    print("Architecture MCTNet :")
    print(model)
    print(f"\n✅ Modèle prêt — lancer : python step6_train.py")