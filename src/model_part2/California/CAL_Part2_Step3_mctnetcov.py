"""
PART 2 — ÉTAPE 3 : Architecture MCTNetCov — California
=======================================================
MCTNet (Part 1) + Late Fusion des covariables environnementales.

Late Fusion :
  S2 branch  : MCTNet backbone → GlobalMaxPool → [B, 30]
  Cov branch : MLP(n_cov → 16) → [B, 16]
  Fusion     : Concat → Linear(46, 64) → ReLU → Linear(64, 6)

6 classes California :
  Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step5_mctnet import MCTNet, CTFusion, count_parameters


class CovariateEncoder(nn.Module):
    def __init__(self, n_cov, d_cov=16, dropout=0.1):
        super().__init__()
        hidden = max(n_cov * 4, 16)
        self.net = nn.Sequential(
            nn.Linear(n_cov, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_cov),
            nn.ReLU(),
        )

    def forward(self, cov):
        return self.net(cov)


class MCTNetCov(nn.Module):
    """
    MCTNet + Late Fusion des covariables — California (n_classes=6).
    n_cov : 2 (topo), 3 (clim ou sol), 8 (all)
    """
    def __init__(self, n_bands=10, n_timesteps=36, n_classes=6,
                 n_stage=3, n_head=5, kernel_size=3, d_model=30,
                 dropout=0.1, n_cov=8, d_cov=16):
        super().__init__()
        self.n_cov = n_cov

        # Backbone MCTNet
        self.input_embedding = nn.Linear(n_bands, d_model)
        self.stages = nn.ModuleList()
        t = n_timesteps
        for s in range(n_stage):
            self.stages.append(
                CTFusion(d_model, n_head, kernel_size,
                         use_alpe=(s == 0), n_timesteps=t, dropout=dropout)
            )
            t = t // 2
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encodeur covariables
        self.cov_encoder = CovariateEncoder(n_cov, d_cov, dropout)

        # Classificateur fusion
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_cov, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, mask, cov):
        out = self.input_embedding(x)
        for s, stage in enumerate(self.stages):
            out = stage(out, mask if s == 0 else None)
            if s < len(self.stages) - 1:
                out  = self.pool(out.transpose(1, 2)).transpose(1, 2)
                if s == 0:
                    mask = mask[:, ::2]
        feat_s2  = out.max(dim=1).values
        feat_cov = self.cov_encoder(cov)
        return self.classifier(torch.cat([feat_s2, feat_cov], dim=-1))


def build_model(n_cov, d_cov=16, device='cpu'):
    """Construit MCTNet (n_cov=0) ou MCTNetCov (n_cov>0) — California."""
    if n_cov == 0:
        return MCTNet(n_bands=10, n_timesteps=36, n_classes=6,
                      n_stage=3, n_head=5, kernel_size=3,
                      d_model=30, dropout=0.1).to(device)
    return MCTNetCov(n_bands=10, n_timesteps=36, n_classes=6,
                     n_stage=3, n_head=5, kernel_size=3,
                     d_model=30, dropout=0.1,
                     n_cov=n_cov, d_cov=d_cov).to(device)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 4
    print("=" * 55)
    print("Part 2 — Étape 3 : MCTNetCov — California (6 classes)")
    print("=" * 55 + "\n")
    for n_cov, name in [(0,'C1 S2only'),(3,'C2 Climat'),
                         (3,'C3 Sol'),(2,'C4 Topo'),(8,'C5 Tout')]:
        m = build_model(n_cov, device=device)
        m.eval()
        x, mask = torch.randn(B,36,10).to(device), torch.zeros(B,36).to(device)
        with torch.no_grad():
            out = m(x, mask, torch.rand(B,n_cov).to(device)) if n_cov > 0 \
                  else m(x, mask)
        print(f"✅ {name:12s} n_cov={n_cov} params={count_parameters(m):,} "
              f"out={list(out.shape)}  (attendu: [{B}, 6])")
    print("\n→ Lancer : python CAL_Part2_Step4_ablation_train.py")
