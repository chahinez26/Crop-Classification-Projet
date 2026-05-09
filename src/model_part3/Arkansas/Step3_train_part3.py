
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Step2_model_part3 import Part3ArkansasNet, count_parameters


ROOT_DIR = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"
DATA_FILE = os.path.join(ROOT_DIR, "part_3", "npz", "ARK_part3_dataset.npz")
MODEL_DIR = os.path.join(ROOT_DIR, "part_3", "model")
FIG_DIR = os.path.join(ROOT_DIR, "part_3", "figures")

BATCH_SIZE = 32
EPOCHS = 180
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40
RANDOM_SEED = 777

N_STAGE = 3
N_HEAD = 6
KERNEL_SIZE = 3
D_MODEL = 36
DROPOUT = 0.12
N_CLASSES = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def resolve_paths(tag):
    suffix = f"_{tag}" if tag else ""
    model_file = os.path.join(MODEL_DIR, f"best_model_part3{suffix}.pth")
    history_file = os.path.join(ROOT_DIR, "part_3", "npz", f"training_history_part3{suffix}.npz")
    figure_file = os.path.join(FIG_DIR, f"fig_training_part3{suffix}.png")
    return model_file, history_file, figure_file


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def macro_f1_from_numpy(y_true, y_pred, n_classes=N_CLASSES):
    f1_scores = []
    for i in range(n_classes):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def make_loaders():
    data = np.load(DATA_FILE)

    x_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    m_train = torch.tensor(data["mask_train"], dtype=torch.float32)

    x_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.long)
    m_val = torch.tensor(data["mask_val"], dtype=torch.float32)

    x_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)
    m_test = torch.tensor(data["mask_test"], dtype=torch.float32)

    train_ds = TensorDataset(x_train, m_train, y_train)
    val_ds = TensorDataset(x_val, m_val, y_val)
    test_ds = TensorDataset(x_test, m_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val  : {len(val_ds)} samples")
    print(f"Test : {len(test_ds)} samples")
    print(f"Feature count: {x_train.shape[-1]}")

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    preds_all = []
    y_all = []

    with torch.set_grad_enabled(training):
        for x_batch, m_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            y_batch = y_batch.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(x_batch, m_batch)
            loss = criterion(logits, y_batch)

            if training:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            total_correct += (preds == y_batch).sum().item()
            total_n += len(y_batch)
            preds_all.append(preds.detach().cpu().numpy())
            y_all.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(preds_all)
    macro_f1 = macro_f1_from_numpy(y_true, y_pred)
    return total_loss / total_n, total_correct / total_n, macro_f1


def plot_history(history, figure_file):
    os.makedirs(FIG_DIR, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#d62728")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, np.array(history["train_acc"]) * 100.0, label="Train", color="#1f77b4")
    axes[1].plot(epochs, np.array(history["val_acc"]) * 100.0, label="Val", color="#d62728")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, np.array(history["train_f1"]) * 100.0, label="Train", color="#1f77b4")
    axes[2].plot(epochs, np.array(history["val_f1"]) * 100.0, label="Val", color="#d62728")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1 (%)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.suptitle("Part 3 Arkansas - training curves v3", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {figure_file}")


def main():
    args = parse_args()
    model_file, history_file, figure_file = resolve_paths(args.tag)

    print("=" * 60)
    print("Step 3 - Train Part 3 Arkansas model")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    print(f"Seed: {args.seed}")
    print(f"Tag : {args.tag or 'default'}\n")

    train_loader, val_loader, test_loader = make_loaders()

    model = Part3ArkansasNet(
        n_features=15,
        n_timesteps=36,
        n_classes=N_CLASSES,
        n_stage=N_STAGE,
        n_head=N_HEAD,
        kernel_size=KERNEL_SIZE,
        d_model=D_MODEL,
        dropout=DROPOUT,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    print(
        f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Train F1':>9}  "
        f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}  {'Time':>6}"
    )
    print("-" * 92)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        marker = ""

        score = 0.7 * val_acc + 0.3 * val_f1
        improved = (score > best_score + 1e-4) or (
            abs(score - best_score) <= 1e-4 and val_loss < best_val_loss
        )
        if improved:
            best_score = score
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "selection_score": score,
                    "config": {
                        "n_features": 15,
                        "n_timesteps": 36,
                        "n_classes": N_CLASSES,
                        "n_stage": N_STAGE,
                        "n_head": N_HEAD,
                        "kernel_size": KERNEL_SIZE,
                        "d_model": D_MODEL,
                        "dropout": DROPOUT,
                    },
                },
                model_file,
            )
            marker = "  <- best"
        else:
            patience_count += 1

        if epoch == 1 or epoch % 5 == 0 or marker:
            print(
                f"{epoch:>6d}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  {train_f1*100:>8.2f}%  "
                f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  {val_f1*100:>7.2f}%  {elapsed:>5.1f}s{marker}"
            )

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best epoch={best_epoch})")
            break

    total_time = time.time() - t_start
    print(f"\nTraining finished in {total_time:.1f}s")
    print(
        f"Best epoch: {best_epoch} | best score: {best_score:.4f} "
        f"| best val loss: {best_val_loss:.4f}"
    )

    np.savez(history_file, **{k: np.array(v) for k, v in history.items()})
    print(f"Saved: {history_file}")

    plot_history(history, figure_file)
    print("\nNext step: run Step4_evaluate_part3.py")


if __name__ == "__main__":
    main()
