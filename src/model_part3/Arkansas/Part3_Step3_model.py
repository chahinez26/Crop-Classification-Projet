"""
PART 3 — ÉTAPE 3 : Architecture CBAM-CNN-BiLSTM
=================================================
Architecture originale pour la classification de cultures.

Inspiration :
  • CBAM (Woo et al., 2018) — attention canal + spatiale
  • CNN 1D pour l'extraction de features spectrales/temporelles
  • BiLSTM pour la modélisation des dépendances temporelles long-terme

Adaptation 1D (données satellite temps-série) :
  ┌─────────────────────────────────────────────────────────┐
  │ ChannelAttention1D : "Quelles features comptent ?"      │
  │    AvgPool + MaxPool sur T → MLP partagé → sigmoid      │
  │                                                         │
  │ TemporalAttention1D : "Quels timesteps comptent ?"      │
  │    AvgPool + MaxPool sur C → Conv1D → sigmoid           │
  └─────────────────────────────────────────────────────────┘

Pipeline complet :
  Input (B, 36, 18)
    ↓ permute → (B, 18, 36)
    ↓ CNNBlock1 : Conv1D(18→128, k=7) → BN → ReLU → CBAM → Pool → (B, 128, 18)
    ↓ CNNBlock2 : Conv1D(128→64, k=5) → BN → ReLU → CBAM → Pool → (B, 64, 9)
    ↓ permute → (B, 9, 64)
    ↓ BiLSTM(64→256, hidden=128) → Dropout(0.3)
    ↓ BiLSTM(256→128, hidden=64) → Dropout(0.2)
    ↓ last timestep → (B, 128)
    ↓ Dense(128→128, ReLU) → Dropout(0.3)
    ↓ Dense(128→64, ReLU)
    ↓ Dense(64→5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODULE 1 : Channel Attention 1D
# "Quelles features (bandes S2 + covariables) sont les plus discriminantes ?"
# =============================================================================
class ChannelAttention1D(nn.Module):
    """
    Calcule un poids d'importance pour chaque canal (feature dimension).
    x : (B, C, T) → M_c : (B, C, 1)

    Mc = sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    F' = Mc ⊗ F
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)   # (B, C, T) → (B, C, 1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)   # (B, C, T) → (B, C, 1)
        # MLP partagé entre avg et max (comme dans le papier CBAM)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, T)
        avg_w = self.shared_mlp(self.avg_pool(x))   # (B, C)
        max_w = self.shared_mlp(self.max_pool(x))   # (B, C)
        Mc    = self.sigmoid(avg_w + max_w)          # (B, C)
        return x * Mc.unsqueeze(-1)                  # (B, C, T) — F'


# =============================================================================
# MODULE 2 : Temporal Attention 1D  (équivalent Spatial Attention du CBAM)
# "Quels timesteps sont les plus informatifs pour la phénologie ?"
# =============================================================================
class TemporalAttention1D(nn.Module):
    """
    Calcule un poids d'importance pour chaque timestep.
    x : (B, C, T) → M_t : (B, 1, T)

    Mt = sigmoid( Conv1D( [AvgPool_C ; MaxPool_C] ) )
    F''= Mt ⊗ F'
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size doit être impair"
        self.conv = nn.Conv1d(
            in_channels=2, out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, T)
        avg_t  = x.mean(dim=1, keepdim=True)      # (B, 1, T) — moyenne sur C
        max_t, _ = x.max(dim=1, keepdim=True)     # (B, 1, T) — max sur C
        concat = torch.cat([avg_t, max_t], dim=1) # (B, 2, T)
        Mt     = self.sigmoid(self.conv(concat))   # (B, 1, T)
        return x * Mt                              # (B, C, T) — F''


# =============================================================================
# MODULE 3 : CBAM 1D
# Séquence : Channel Attention → Temporal Attention
# =============================================================================
class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module adapté aux séries temporelles 1D.

    F  → ChannelAttn  → F'  (raffinage spectral / feature)
    F' → TemporalAttn → F'' (raffinage temporel / phénologique)
    """
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_att  = ChannelAttention1D(channels, reduction)
        self.temporal_att = TemporalAttention1D(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)    # F'  : raffinage des features
        x = self.temporal_att(x)   # F'' : raffinage temporel
        return x


# =============================================================================
# MODULE 4 : CNN Block avec CBAM intégré
# Conv1D → BN → ReLU → CBAM → MaxPool
# =============================================================================
class CNNBlock(nn.Module):
    """
    Un bloc convolutionnel avec attention CBAM et connexion résiduelle légère.

    Connexion résiduelle : si in_ch == out_ch → skip connection
                           sinon              → projection 1×1
    """
    def __init__(self, in_ch, out_ch, kernel_size=7,
                 cbam_reduction=8, cbam_kernel=7, dropout=0.25):
        super().__init__()
        pad = kernel_size // 2

        self.conv  = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.cbam  = CBAM1D(out_ch, cbam_reduction, cbam_kernel)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout(dropout)

        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
        self.skip_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x : (B, in_ch, T)
        identity = self.skip_pool(self.skip(x))   # projection résiduelle

        out = self.relu(self.bn(self.conv(x)))     # Conv → BN → ReLU
        out = self.cbam(out)                        # CBAM (ch + temporal attn)
        out = self.pool(out)                        # MaxPool1D /2
        out = self.drop(out)

        return self.relu(out + identity)            # skip connection


# =============================================================================
# MODULE 5 : Architecture complète CBAM-CNN-BiLSTM
# =============================================================================
class CropCBAMCNNLSTM(nn.Module):
    """
    Classificateur de cultures : CNN(CBAM) + BiLSTM

    Args:
        n_features  : nombre de features d'entrée (défaut=18 pour S2+Tout)
        n_classes   : nombre de classes (défaut=5)
        cnn_filters : canaux des 2 blocs CNN (défaut=[128, 64])
        lstm_units  : hidden size des 2 BiLSTM (défaut=[128, 64])
        dropout     : taux de dropout global (défaut=0.3)

    Flow :
        (B, T=36, F=18)
          → CNN blocks : (B, F, T) → (B, 64, 9)
          → BiLSTM     : (B, 9, 64) → (B, 9, 128) → (B, 128)
          → Classifier : (B, 5)
    """
    def __init__(self, n_features=18, n_classes=5,
                 cnn_filters=(128, 64),
                 lstm_units=(128, 64),
                 dropout=0.3):
        super().__init__()

        # ── Bloc CNN 1 ────────────────────────────────────────────────────────
        self.cnn_block1 = CNNBlock(
            in_ch=n_features, out_ch=cnn_filters[0],
            kernel_size=7, cbam_reduction=8, cbam_kernel=7, dropout=dropout * 0.8
        )
        # ── Bloc CNN 2 ────────────────────────────────────────────────────────
        self.cnn_block2 = CNNBlock(
            in_ch=cnn_filters[0], out_ch=cnn_filters[1],
            kernel_size=5, cbam_reduction=8, cbam_kernel=7, dropout=dropout * 0.8
        )
        # Après 2× MaxPool(/2) : T=36 → 18 → 9

        # ── BiLSTM 1 ──────────────────────────────────────────────────────────
        # entrée : cnn_filters[-1] = 64
        # sortie : lstm_units[0]*2 = 256  (bidirectionnel)
        self.bilstm1 = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_units[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.drop1 = nn.Dropout(dropout)

        # ── BiLSTM 2 ──────────────────────────────────────────────────────────
        # entrée : lstm_units[0]*2 = 256
        # sortie : lstm_units[1]*2 = 128  (bidirectionnel)
        self.bilstm2 = nn.LSTM(
            input_size=lstm_units[0] * 2,
            hidden_size=lstm_units[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.drop2 = nn.Dropout(dropout * 0.7)

        # ── Classifier ────────────────────────────────────────────────────────
        lstm_out_dim = lstm_units[-1] * 2   # *2 bidirectionnel → 128
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

        # ── Initialisation des poids ──────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias' in name:
                    nn.init.zeros_(p.data)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)

    def forward(self, x):
        """
        x : (B, T=36, F=18)  — early fusion features
        """
        # ── CNN + CBAM ────────────────────────────────────────────────────────
        x = x.permute(0, 2, 1)          # (B, F=18, T=36) pour Conv1D
        x = self.cnn_block1(x)           # (B, 128, 18)
        x = self.cnn_block2(x)           # (B, 64, 9)
        x = x.permute(0, 2, 1)          # (B, 9, 64) pour LSTM

        # ── BiLSTM ────────────────────────────────────────────────────────────
        x, _ = self.bilstm1(x)           # (B, 9, 256)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)           # (B, 9, 128)
        x = self.drop2(x)

        # Prendre le dernier timestep (résumé de la séquence)
        x = x[:, -1, :]                  # (B, 128)

        # ── Classification ────────────────────────────────────────────────────
        return self.classifier(x)         # (B, n_classes)

    def get_attention_weights(self, x):
        """
        Retourne les cartes d'attention CBAM pour visualisation.
        Utile pour l'interprétabilité.
        """
        x = x.permute(0, 2, 1)
        attn_maps = []

        for block in [self.cnn_block1, self.cnn_block2]:
            # Forward jusqu'au CBAM
            out = F.relu(block.bn(block.conv(x)))
            ch_map  = block.cbam.channel_att.sigmoid(
                block.cbam.channel_att.shared_mlp(
                    block.cbam.channel_att.avg_pool(out)) +
                block.cbam.channel_att.shared_mlp(
                    block.cbam.channel_att.max_pool(out))
            )  # (B, C)
            attn_maps.append({'channel': ch_map.detach()})
            out = block.cbam(out)
            out = block.pool(out)
            x   = out

        return attn_maps


# =============================================================================
# UTILITAIRES
# =============================================================================
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(n_features=18, n_classes=5, device='cpu',
                cnn_filters=(128, 64), lstm_units=(128, 64), dropout=0.3):
    model = CropCBAMCNNLSTM(
        n_features=n_features,
        n_classes=n_classes,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        dropout=dropout,
    ).to(device)

    total, trainable = count_parameters(model)
    mctnet_params = 55059   # référence Part 1

    print(f"\n  CropCBAMCNNLSTM — {trainable:,} paramètres entraînables")
    print(f"  (MCTNet Part 1 : {mctnet_params:,} | rapport : ×{trainable/mctnet_params:.1f})")
    return model


def test_forward_pass(device='cpu'):
    print("\n── Test forward pass ───────────────────────────────────")
    model = build_model(n_features=18, n_classes=5, device=device)
    model.eval()

    B = 8
    x = torch.randn(B, 36, 18).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=-1)

    print(f"  Input        : {list(x.shape)}")
    print(f"  Output logits: {list(logits.shape)}  (attendu: [{B}, 5])")
    print(f"  Probs (sum)  : {probs.sum(dim=-1).tolist()}")
    print(f"  ✅ Forward pass OK\n")

    # Vérification des dimensions intermédiaires
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
    print("\n── Architecture CBAM-CNN-BiLSTM ────────────────────────")
    print("""
  Input         : (B, T=36, F=18)
                   ↓ permute → (B, 18, 36)
  ┌─────────────────────────────────────────────────────┐
  │  CNNBlock1 : Conv1D(18→128, k=7) + BN + ReLU        │
  │              → CBAM1D(128) [ch_attn + temp_attn]    │
  │              → MaxPool1D(/2) + Dropout(0.24)        │
  │              → Skip(proj 1×1) + Pool + ReLU         │
  │              Sortie : (B, 128, 18)                  │
  ├─────────────────────────────────────────────────────┤
  │  CNNBlock2 : Conv1D(128→64, k=5) + BN + ReLU        │
  │              → CBAM1D(64) [ch_attn + temp_attn]     │
  │              → MaxPool1D(/2) + Dropout(0.24)        │
  │              → Skip(identité) + Pool + ReLU         │
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
               Linear(64→5)
""")
    total, trainable = count_parameters(model)
    print(f"  Paramètres total     : {total:,}")
    print(f"  Paramètres entraîn.  : {trainable:,}")


if __name__ == "__main__":
    print("=" * 58)
    print("Part 3 — Étape 3 : Architecture CBAM-CNN-BiLSTM")
    print("=" * 58)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice : {device}")

    model = test_forward_pass(device)
    print_architecture(model)

    print("\n✅ Architecture validée")
    print("   → Lancer : python Part3_Step4_train.py")
