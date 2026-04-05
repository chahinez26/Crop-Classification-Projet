"""
ÉTAPE 6 — Entraînement MCTNet — California
==========================================
Différences vs Arkansas :
  - n_classes=6
  - DATA_FILE → CAL_dataset_preprocessed.npz
  - Importe CAL_Step5_mctnet (n_classes=6 par défaut)
  - Métriques cibles : OA=0.852, Kappa=0.806, F1=0.829
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os, time

from CAL_Step5_mctnet import MCTNet, count_parameters

# ── Configuration ──────────────────────────────────────────────
DATA_FILE    = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset_preprocessed.npz"
MODEL_FILE   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\model\best_model_cal.pth"
HISTORY_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\training_history_cal.npz"
FIG_DIR      = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\figures"

# Hyperparamètres papier Table 3 — California (identiques à Arkansas)
BATCH_SIZE   = 32
EPOCHS       = 200
LR           = 0.001
N_STAGE      = 3
N_HEAD       = 5
KERNEL_SIZE  = 3
D_MODEL      = 30
N_CLASSES    = 6      # ← 6 classes pour California
DROPOUT      = 0.1
PATIENCE     = 40
RANDOM_SEED  = 42
# ───────────────────────────────────────────────────────────────

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data():
    data = np.load(DATA_FILE)

    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    X_train = to_tensor(data['X_train'])
    y_train = to_tensor(data['y_train'], dtype=torch.long)
    m_train = to_tensor(data['mask_train'])

    X_val   = to_tensor(data['X_val'])
    y_val   = to_tensor(data['y_val'], dtype=torch.long)
    m_val   = to_tensor(data['mask_val'])

    X_test  = to_tensor(data['X_test'])
    y_test  = to_tensor(data['y_test'], dtype=torch.long)
    m_test  = to_tensor(data['mask_test'])

    train_ds = TensorDataset(X_train, m_train, y_train)
    val_ds   = TensorDataset(X_val,   m_val,   y_val)
    test_ds  = TensorDataset(X_test,  m_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train  : {len(X_train)} samples  ({len(train_loader)} batches de {BATCH_SIZE})")
    print(f"Val    : {len(X_val)} samples")
    print(f"Test   : {len(X_test)} samples")
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for X_b, m_b, y_b in loader:
        X_b, m_b, y_b = X_b.to(device), m_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b, m_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(y_b)
        total_correct += (logits.argmax(1) == y_b).sum().item()
        total_n       += len(y_b)
    return total_loss / total_n, total_correct / total_n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for X_b, m_b, y_b in loader:
            X_b, m_b, y_b = X_b.to(device), m_b.to(device), y_b.to(device)
            logits = model(X_b, m_b)
            loss   = criterion(logits, y_b)
            total_loss    += loss.item() * len(y_b)
            total_correct += (logits.argmax(1) == y_b).sum().item()
            total_n       += len(y_b)
    return total_loss / total_n, total_correct / total_n


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs  = range(1, len(history['train_loss']) + 1)
    best_ep = int(np.argmin(history['val_loss'])) + 1

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train', color='#1f77b4')
    ax.plot(epochs, history['val_loss'],   label='Val',   color='#d62728')
    ax.axvline(best_ep, color='gray', linestyle='--', linewidth=1,
               label=f'Best epoch {best_ep}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss — California'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], label='Train', color='#1f77b4')
    ax.plot(epochs, [a*100 for a in history['val_acc']],   label='Val',   color='#d62728')
    ax.axvline(best_ep, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy — California'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('MCTNet — Entraînement California', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_training_cal.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def main():
    print("=" * 55)
    print("Étape 6 — Entraînement MCTNet — California")
    print("=" * 55 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    train_loader, val_loader, test_loader = load_data()

    model = MCTNet(n_bands=10, n_timesteps=36, n_classes=N_CLASSES,
                   n_stage=N_STAGE, n_head=N_HEAD,
                   kernel_size=KERNEL_SIZE, d_model=D_MODEL,
                   dropout=DROPOUT).to(device)
    n_params = count_parameters(model)
    print(f"\nParamètres : {n_params:,}  (papier California : 55 140)")
    print(f"Écart : {n_params - 55140:+d} ({abs(n_params-55140)/55140*100:.1f}%)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_loss  = float('inf')
    patience_count = 0
    best_epoch     = 0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("-" * 60)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        marker  = ''

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_loss':            val_loss,
                'val_acc':             val_acc,
                'train_loss':          train_loss,
                'train_acc':           train_acc,
                'd_model':             D_MODEL,
                'n_classes':           N_CLASSES,
            }, MODEL_FILE)
            marker = '  ← best'
        else:
            patience_count += 1

        if epoch % 5 == 0 or marker or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
                  f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  "
                  f"{elapsed:>5.1f}s{marker}")

        if patience_count >= PATIENCE:
            print(f"\n⏹  Early stopping epoch {epoch}  (best={best_epoch})")
            break

    total_time = time.time() - t_start
    print(f"\nEntraînement terminé en {total_time:.1f}s")
    print(f"Meilleur modèle : epoch {best_epoch}  "
          f"val_loss={best_val_loss:.4f}  "
          f"val_acc={history['val_acc'][best_epoch-1]*100:.2f}%")

    np.savez(HISTORY_FILE, **{k: np.array(v) for k, v in history.items()})
    print(f"✅ Historique sauvé : {HISTORY_FILE}")

    plot_training(history)
    print(f"\n  → Lancer maintenant : python CAL_Step7_evaluate.py")


if __name__ == "__main__":
    main()