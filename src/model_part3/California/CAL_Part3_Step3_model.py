"""
PART 3 — ÉTAPE 3 (CALIFORNIA) : Architecture CBAM-CNN-BiLSTM
=============================================================
Identique à Arkansas mais adapté à :
  • N_FEATURES = 13  (10 S2 + 3 Sol)
  • N_CLASSES  = 6   (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)

Pipeline :
  Input (B, 36, 13)
    ↓ permute → (B, 13, 36)
    ↓ CNNBlock1 : Conv1D(13→128, k=7) → BN → ReLU → CBAM1D → Pool → (B, 128, 18)
    ↓ CNNBlock2 : Conv1D(128→64, k=5) → BN → ReLU → CBAM1D → Pool → (B, 64, 9)
    ↓ permute → (B, 9, 64)
    ↓ BiLSTM(hidden=128) → Dropout → (B, 9, 256)
    ↓ BiLSTM(hidden=64)  → Dropout → (B, 9, 128) → last step → (B, 128)
    ↓ Dense(128→128) → BN → ReLU → Drop → Dense(128→64) → Dense(64→6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODULE 1 : Channel Attention 1D
# "Quelles features (bandes S2 + Sol) sont les plus discriminantes ?"
# =============================================================================
class ChannelAttention1D(nn.Module):
    """
    Mc = sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    F' = Mc ⊗ F
    x : (B, C, T) → (B, C, T)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_w = self.shared_mlp(self.avg_pool(x))
        max_w = self.shared_mlp(self.max_pool(x))
        Mc    = self.sigmoid(avg_w + max_w)
        return x * Mc.unsqueeze(-1)


# =============================================================================
# MODULE 2 : Temporal Attention 1D
# "Quels timesteps sont phénologiquement importants en California ?"
# (cultures méditerranéennes : vigne, amandiers → phénologies spécifiques)
# =============================================================================
class TemporalAttention1D(nn.Module):
    """
    Mt = sigmoid( Conv1D( [AvgPool_C ; MaxPool_C] ) )
    F''= Mt ⊗ F'
    x : (B, C, T) → (B, C, T)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv1d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_t  = x.mean(dim=1, keepdim=True)
        max_t, _ = x.max(dim=1, keepdim=True)
        concat = torch.cat([avg_t, max_t], dim=1)
        Mt     = self.sigmoid(self.conv(concat))
        return x * Mt


# =============================================================================
# MODULE 3 : CBAM 1D
# =============================================================================
class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_att  = ChannelAttention1D(channels, reduction)
        self.temporal_att = TemporalAttention1D(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.temporal_att(x)
        return x


# =============================================================================
# MODULE 4 : CNN Block avec CBAM
# =============================================================================
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7,
                 cbam_reduction=8, cbam_kernel=7, dropout=0.25):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM1D(out_ch, cbam_reduction, cbam_kernel)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

        self.skip      = nn.Conv1d(in_ch, out_ch, 1, bias=False) \
                         if in_ch != out_ch else nn.Identity()
        self.skip_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = self.skip_pool(self.skip(x))
        out = self.relu(self.bn(self.conv(x)))
        out = self.cbam(out)
        out = self.pool(out)
        out = self.drop(out)
        return self.relu(out + identity)


# =============================================================================
# MODULE 5 : Architecture complète CBAM-CNN-BiLSTM — California
# =============================================================================
class CropCBAMCNNLSTM(nn.Module):
    """
    Classificateur de cultures — California
      n_features = 13  (10 S2 + 3 Sol)
      n_classes  = 6   (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)

    Paramètres par défaut identiques à Arkansas pour comparabilité.
    """
    def __init__(self, n_features=13, n_classes=6,
                 cnn_filters=(128, 64),
                 lstm_units=(128, 64),
                 dropout=0.3):
        super().__init__()

        # ── CNN Block 1 ───────────────────────────────────────────────────────
        self.cnn_block1 = CNNBlock(
            in_ch=n_features, out_ch=cnn_filters[0],
            kernel_size=7, cbam_reduction=8, cbam_kernel=7,
            dropout=dropout * 0.8
        )
        # ── CNN Block 2 ───────────────────────────────────────────────────────
        self.cnn_block2 = CNNBlock(
            in_ch=cnn_filters[0], out_ch=cnn_filters[1],
            kernel_size=5, cbam_reduction=8, cbam_kernel=7,
            dropout=dropout * 0.8
        )
        # Après 2× MaxPool(/2) : T=36 → 18 → 9

        # ── BiLSTM 1 : (B, 9, 64) → (B, 9, 256) ─────────────────────────────
        self.bilstm1 = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_units[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)

        # ── BiLSTM 2 : (B, 9, 256) → (B, 9, 128) ────────────────────────────
        self.bilstm2 = nn.LSTM(
            input_size=lstm_units[0] * 2,
            hidden_size=lstm_units[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout * 0.7)

        # ── Classifier : (B, 128) → (B, 6) ───────────────────────────────────
        lstm_out_dim = lstm_units[-1] * 2   # 128
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name: nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name: nn.init.orthogonal_(p.data)
                elif 'bias' in name: nn.init.zeros_(p.data)

    def forward(self, x):
        """x : (B, T=36, F=13)"""
        x = x.permute(0, 2, 1)          # (B, 13, 36)
        x = self.cnn_block1(x)           # (B, 128, 18)
        x = self.cnn_block2(x)           # (B, 64, 9)
        x = x.permute(0, 2, 1)          # (B, 9, 64)

        x, _ = self.bilstm1(x)           # (B, 9, 256)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)           # (B, 9, 128)
        x = self.drop2(x)

        x = x[:, -1, :]                  # (B, 128) — dernier timestep
        return self.classifier(x)         # (B, 6)


# =============================================================================
# UTILITAIRES
# =============================================================================
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(n_features=13, n_classes=6, device='cpu',
                cnn_filters=(128, 64), lstm_units=(128, 64), dropout=0.3):
    model = CropCBAMCNNLSTM(
        n_features=n_features, n_classes=n_classes,
        cnn_filters=cnn_filters, lstm_units=lstm_units, dropout=dropout,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"\n  CropCBAMCNNLSTM — California")
    print(f"  {trainable:,} paramètres entraînables")
    print(f"  (Arkansas avait 18 features → California : 13 features, 6 classes)")
    return model


def test_forward_pass(device='cpu'):
    print("\n── Test forward pass ───────────────────────────────────")
    model = build_model(device=device)
    model.eval()

    B = 8
    x = torch.randn(B, 36, 13).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=-1)

    print(f"  Input         : {list(x.shape)}")
    print(f"  Output logits : {list(logits.shape)}  (attendu: [{B}, 6])")
    print(f"  Probs (sum)   : {probs.sum(dim=-1).tolist()}")
    print(f"  ✅ Forward pass OK\n")

    # Dimensions intermédiaires
    print("── Dimensions intermédiaires ───────────────────────────")
    xc = x.permute(0, 2, 1)
    xc = model.cnn_block1(xc); print(f"  Après CNNBlock1 : {list(xc.shape)}")
    xc = model.cnn_block2(xc); print(f"  Après CNNBlock2 : {list(xc.shape)}")
    xc = xc.permute(0, 2, 1)
    xl, _ = model.bilstm1(xc); print(f"  Après BiLSTM1   : {list(xl.shape)}")
    xl, _ = model.bilstm2(xl); print(f"  Après BiLSTM2   : {list(xl.shape)}")
    print(f"  Dernier step  : {list(xl[:, -1, :].shape)}")
    return model


def print_architecture(model):
    print("\n── Architecture CBAM-CNN-BiLSTM — California ──────────")
    print("""
  Input         : (B, T=36, F=13)  ← 10 S2 + 3 Sol
                   ↓ permute → (B, 13, 36)
  ┌─────────────────────────────────────────────────────┐
  │  CNNBlock1 : Conv1D(13→128, k=7) + BN + ReLU        │
  │              → CBAM1D(128) [ch_attn + temp_attn]    │
  │              → MaxPool1D(/2) + Dropout               │
  │              Sortie : (B, 128, 18)                  │
  ├─────────────────────────────────────────────────────┤
  │  CNNBlock2 : Conv1D(128→64, k=5) + BN + ReLU        │
  │              → CBAM1D(64)  [ch_attn + temp_attn]    │
  │              → MaxPool1D(/2) + Dropout               │
  │              Sortie : (B, 64, 9)                    │
  └─────────────────────────────────────────────────────┘
                   ↓ permute → (B, 9, 64)
  ┌─────────────────────────────────────────────────────┐
  │  BiLSTM1 : hidden=128 → sortie (B, 9, 256)          │
  │  Dropout(0.3)                                       │
  ├─────────────────────────────────────────────────────┤
  │  BiLSTM2 : hidden=64  → sortie (B, 9, 128)          │
  │  Dropout(0.21) → last step → (B, 128)               │
  └─────────────────────────────────────────────────────┘
  Classifier : Linear(128→128) → BN → ReLU → Drop(0.3)
               Linear(128→64)  → ReLU
               Linear(64→6)    ← 6 classes California
""")
    total, trainable = count_parameters(model)
    print(f"  Paramètres entraîn.  : {trainable:,}")
    print(f"  (vs Arkansas 18→13 features : légèrement moins de params CNN)")


if __name__ == "__main__":
    print("=" * 58)
    print("Part 3 — Étape 3 : Architecture CBAM-CNN-BiLSTM — California")
    print("=" * 58)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice : {device}")
    model = test_forward_pass(device)
    print_architecture(model)

    print("\n✅ Architecture validée")
    print("   → Lancer : python CAL_Part3_Step4_train.py")
