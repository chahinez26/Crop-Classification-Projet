"""
PART 3 — ÉTAPE 4 (CALIFORNIA) : Entraînement CBAM-CNN-BiLSTM
=============================================================
Entrée  : data/merged_npz/california/Part3_CAL_dataset.npz
Sortie  : outputs/results/california/part3/

Différences vs Arkansas :
  • N_FEATURES = 13  (S2 + Sol seulement)
  • N_CLASSES  = 6   (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)
  • Déséquilibre de classes plus prononcé → class_weights plus importants

Même stratégie d'entraînement :
  • Class weights (inverse freq.)
  • CosineAnnealingWarmRestarts (T0=20, Tmult=2)
  • Early stopping (patience=40)
  • Gradient clipping (max_norm=1.0)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import os, time, json

from CAL_Part3_Step3_model import build_model, count_parameters

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_FILE    = r"data\merged_npz\california\Part3_CAL_dataset.npz"
MODEL_DIR   = r"outputs\results\california\part3\models"
RESULTS_DIR = r"outputs\results\california\part3\results"
FIG_DIR     = r"outputs\figures\california\part3"

BATCH_SIZE   = 64
EPOCHS       = 200
LR           = 0.001
PATIENCE     = 40
WEIGHT_DECAY = 1e-4
CLIP_GRAD    = 1.0
SEED         = 42

N_FEATURES  = 13
N_CLASSES   = 6
CNN_FILTERS = (128, 64)
LSTM_UNITS  = (128, 64)
DROPOUT     = 0.3

CLASS_NAMES  = ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others']
CLASS_COLORS = ['#9400D3', '#2196F3', '#FF9800', '#8B4513', '#4CAF50', '#9E9E9E']

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)


# ── Chargement ────────────────────────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(NPZ_FILE):
        raise FileNotFoundError(
            f"Fichier introuvable : {NPZ_FILE}\n"
            "→ Lancer d'abord : python CAL_Part3_Step2_preprocessing.py"
        )
    d = np.load(NPZ_FILE, allow_pickle=True)

    X         = torch.tensor(d['X'],    dtype=torch.float32)   # (N, 36, 13)
    y         = torch.tensor(d['y'],    dtype=torch.long)       # (N,)
    train_idx = d['train_idx']
    val_idx   = d['val_idx']
    test_idx  = d['test_idx']

    splits = {}
    for sp, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[sp] = TensorDataset(X[idx], y[idx])

    print(f"✅ Dataset chargé : X={X.shape}  y={y.shape}")
    print(f"   train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return splits, y[train_idx]


def make_loaders(splits):
    return {
        'train': DataLoader(splits['train'], batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=True),
        'val'  : DataLoader(splits['val'],   batch_size=256, shuffle=False),
        'test' : DataLoader(splits['test'],  batch_size=256, shuffle=False),
    }


# ── Poids de classes ──────────────────────────────────────────────────────────
def compute_class_weights(y_train, device):
    counts  = torch.bincount(y_train, minlength=N_CLASSES).float()
    weights = counts.sum() / (N_CLASSES * counts)
    weights = weights / weights.sum() * N_CLASSES
    print(f"\n── Class weights (inverse freq.) — California ──────")
    for i, (n, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"   {n:12s} : n={counts[i].int():5,}  w={w:.3f}")
    return weights.to(device)


# ── Métriques ─────────────────────────────────────────────────────────────────
def compute_metrics(preds, trues, nc=N_CLASSES):
    yt, yp = np.array(trues), np.array(preds)
    oa  = float((yt == yp).mean())
    N   = len(yt)
    cm  = np.zeros((nc, nc), dtype=np.int64)
    for t, p in zip(yt, yp): cm[t, p] += 1

    num   = N * np.trace(cm) - (cm.sum(1) * cm.sum(0)).sum()
    den   = N**2 - (cm.sum(1) * cm.sum(0)).sum()
    kappa = float(num / den) if den != 0 else 0.0

    f1s = []
    for i in range(nc):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        pr = tp/(tp+fp) if (tp+fp) > 0 else 0.
        rc = tp/(tp+fn) if (tp+fn) > 0 else 0.
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.)

    return {
        'OA'   : round(oa, 4),
        'Kappa': round(kappa, 4),
        'F1'   : round(float(np.mean(f1s)), 4),
        'cm'   : cm,
        'f1_per_class': [round(f, 4) for f in f1s],
    }


# ── Epoch ─────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_ok = tot_n = 0
    all_preds, all_trues = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if train: optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                optimizer.step()
            preds = logits.argmax(1)
            tot_loss += loss.item() * len(y_b)
            tot_ok   += (preds == y_b).sum().item()
            tot_n    += len(y_b)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y_b.cpu().numpy())

    return tot_loss/tot_n, tot_ok/tot_n, all_preds, all_trues


# ── Entraînement ──────────────────────────────────────────────────────────────
def train_model(model, loaders, device, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc','lr']}
    best_val_loss = float('inf')
    patience_cnt  = 0
    best_epoch    = 0
    model_path    = os.path.join(MODEL_DIR, 'best_cbam_cnn_bilstm_cal.pth')

    _, trainable = count_parameters(model)
    print(f"\n  Modèle : {trainable:,} paramètres entraînables")
    print(f"  Device : {device}")
    print(f"  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  "
          f"LR : {LR}  |  Patience : {PATIENCE}")
    print(f"  Scheduler : CosineAnnealingWarmRestarts (T0=20, Tmult=2)\n")

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'LR':>8}  {'t':>4}")
    print(f"  {'-'*63}")

    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(
            model, loaders['train'], criterion, optimizer, device, train=True)
        va_loss, va_acc, _, _ = run_epoch(
            model, loaders['val'],   criterion, None,      device, train=False)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['lr'].append(current_lr)

        mark = ''
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = ep
            patience_cnt  = 0
            torch.save({
                'epoch'      : ep,
                'state_dict' : model.state_dict(),
                'val_loss'   : va_loss,
                'val_acc'    : va_acc,
                'hyperparams': {
                    'n_features' : N_FEATURES, 'n_classes': N_CLASSES,
                    'cnn_filters': CNN_FILTERS, 'lstm_units': LSTM_UNITS,
                    'dropout'    : DROPOUT, 'lr': LR
                }
            }, model_path)
            mark = ' ←'
        else:
            patience_cnt += 1

        t_ep = time.time() - t_ep
        if ep % 10 == 0 or mark or ep == 1:
            print(f"  {ep:>4}  {tr_loss:8.4f}  {tr_acc*100:6.2f}%  "
                  f"{va_loss:8.4f}  {va_acc*100:6.2f}%  "
                  f"{current_lr:.2e}  {t_ep:.1f}s{mark}")

        if patience_cnt >= PATIENCE:
            print(f"\n  ⏹ Early stopping à l'epoch {ep} (meilleur: {best_epoch})")
            break

    print(f"\n  ⏱ Total : {time.time()-t_start:.0f}s | "
          f"Meilleur val_loss={best_val_loss:.4f} (ep {best_epoch})")
    return history, model_path, best_epoch


# ── Figures ────────────────────────────────────────────────────────────────────
def plot_learning_curves(history, best_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep, history['train_loss'], label='Train', color='steelblue', lw=1.8)
    axes[0].plot(ep, history['val_loss'],   label='Val',   color='tomato',    lw=1.8)
    axes[0].axvline(best_epoch, color='gray', linestyle='--', lw=1.2,
                    label=f'Best ({best_epoch})')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CrossEntropy Loss')

    axes[1].plot(ep, [a*100 for a in history['train_acc']],
                 label='Train', color='steelblue', lw=1.8)
    axes[1].plot(ep, [a*100 for a in history['val_acc']],
                 label='Val',   color='tomato',    lw=1.8)
    axes[1].axvline(best_epoch, color='gray', linestyle='--', lw=1.2)
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')

    axes[2].plot(ep, history['lr'], color='purple', lw=1.8)
    axes[2].set_title('Learning Rate (CosineWarmRestarts)')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
    axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)

    plt.suptitle('Courbes d\'apprentissage — CBAM-CNN-BiLSTM — California 2021',
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'train_01_learning_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


def plot_loss_zoom(history, best_epoch):
    n  = min(50, len(history['train_loss']))
    ep = range(1, n + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, history['train_loss'][:n], label='Train', color='steelblue', lw=2)
    ax.plot(ep, history['val_loss'][:n],   label='Val',   color='tomato',    lw=2)
    if best_epoch <= n:
        ax.axvline(best_epoch, color='gray', linestyle='--', lw=1.2,
                   label=f'Best ({best_epoch})')
    ax.set_title('Convergence (50 premières epochs) — California', fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'train_02_loss_zoom.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


def main():
    print("="*60)
    print("Part 3 — Étape 4 : Entraînement CBAM-CNN-BiLSTM — California")
    print("="*60 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    splits, y_train = load_dataset()
    loaders = make_loaders(splits)
    class_weights = compute_class_weights(y_train, device)

    print(f"\n── Construction du modèle ──────────────────────────")
    model = build_model(N_FEATURES, N_CLASSES, device, CNN_FILTERS, LSTM_UNITS, DROPOUT)

    print(f"\n── Entraînement ────────────────────────────────────")
    history, model_path, best_epoch = train_model(model, loaders, device, class_weights)

    print(f"\n── Évaluation sur test (meilleur modèle) ───────────")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    _, _, preds, trues = run_epoch(
        model, loaders['test'],
        nn.CrossEntropyLoss(weight=class_weights),
        None, device, train=False
    )
    metrics = compute_metrics(preds, trues)
    print(f"\n  OA    = {metrics['OA']:.4f}")
    print(f"  Kappa = {metrics['Kappa']:.4f}")
    print(f"  F1    = {metrics['F1']:.4f}")
    print(f"\n  Papier MCTNet : OA=0.852 | Kappa=0.806 | F1=0.829")

    results = {
        'model'     : 'CBAM-CNN-BiLSTM',
        'state'     : 'California 2021',
        'config'    : 'S2+Sol (early fusion)',
        'best_epoch': best_epoch,
        'n_params'  : count_parameters(model)[1],
        **{k: v for k, v in metrics.items() if k not in ('cm', 'f1_per_class')},
    }
    res_path = os.path.join(RESULTS_DIR, 'train_results.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)

    hist_path = os.path.join(RESULTS_DIR, 'train_history.npz')
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})

    print(f"\n── Figures ─────────────────────────────────────────")
    plot_learning_curves(history, best_epoch)
    plot_loss_zoom(history, best_epoch)

    print(f"\n✅ Modèle : {model_path}")
    print(f"✅ Résultats : {res_path}")
    print(f"\n   → Lancer : python CAL_Part3_Step5_evaluation.py")


if __name__ == "__main__":
    main()
