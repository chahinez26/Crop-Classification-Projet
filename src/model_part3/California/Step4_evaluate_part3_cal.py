
import json
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Step2_model_part3_cal import Part3CaliforniaNet


ROOT_DIR = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"
DATA_FILE = os.path.join(ROOT_DIR, "California", "part_3", "npz", "CAL_part3_dataset.npz")
RESULTS_DIR = os.path.join(ROOT_DIR, "California", "part_3", "results")
BASELINE_RESULTS = os.path.join(
    ROOT_DIR, "California", "part_1", "MCTNet_California_v2", "results", "results_CAL.json"
)

CLASS_NAMES = ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def resolve_paths(tag):
    suffix = f"_{tag}" if tag else ""
    model_file = os.path.join(ROOT_DIR, "California", "part_3", "model", f"best_model_part3_cal{suffix}.pth")
    results_file = os.path.join(RESULTS_DIR, f"results_CAL_part3{suffix}.json")
    return model_file, results_file


def confusion_matrix_fn(y_true, y_pred, n_classes=6):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def overall_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def cohen_kappa(y_true, y_pred, n_classes=6):
    n = len(y_true)
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    diag_sum = np.trace(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = float((row_sums * col_sums).sum())
    return float((n * diag_sum - expected) / (n**2 - expected))


def macro_f1(y_true, y_pred, n_classes=6):
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    f1_scores = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)), [float(v) for v in f1_scores]


def per_class_metrics(y_true, y_pred, n_classes=6):
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    metrics = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append(
            {
                "class": CLASS_NAMES[i],
                "n_test": int(cm[i, :].sum()),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }
        )
    return metrics


def load_model_and_data(device, model_file):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint = torch.load(model_file, map_location=device)
    cfg = checkpoint["config"]

    model = Part3CaliforniaNet(
        n_features=cfg["n_features"],
        n_timesteps=cfg["n_timesteps"],
        n_classes=cfg["n_classes"],
        n_stage=cfg["n_stage"],
        n_head=cfg["n_head"],
        kernel_size=cfg["kernel_size"],
        d_model=cfg["d_model"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(DATA_FILE)
    x_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)
    m_test = torch.tensor(data["mask_test"], dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(x_test, m_test, y_test), batch_size=256, shuffle=False)
    return model, test_loader, data["y_test"], checkpoint


def predict(model, loader, device):
    all_preds = []
    with torch.no_grad():
        for x_batch, m_batch, _ in loader:
            logits = model(x_batch.to(device), m_batch.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def main():
    args = parse_args()
    model_file, results_file = resolve_paths(args.tag)

    print("=" * 60)
    print("Step 4 - Evaluate Part 3 California model")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tag   : {args.tag or 'default'}\n")

    model, test_loader, y_true, checkpoint = load_model_and_data(device, model_file)
    print(
        f"Loaded checkpoint: epoch={checkpoint['epoch']} "
        f"val_loss={checkpoint['val_loss']:.4f} "
        f"val_acc={checkpoint['val_acc']*100:.2f}% "
        f"val_f1={checkpoint.get('val_f1', 0.0)*100:.2f}%"
    )

    y_pred = predict(model, test_loader, device)
    oa = overall_accuracy(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    f1_macro, class_f1 = macro_f1(y_true, y_pred)
    per_class = per_class_metrics(y_true, y_pred)
    cm = confusion_matrix_fn(y_true, y_pred)

    print("\nGlobal metrics:")
    print(f"  OA       : {oa:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  F1 macro : {f1_macro:.4f}")

    print("\nPer-class metrics:")
    for row in per_class:
        print(
            f"  {row['class']:12s} n={row['n_test']:4d} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
        )

    baseline = None
    if os.path.exists(BASELINE_RESULTS):
        with open(BASELINE_RESULTS, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print("\nComparison with Part 1 baseline:")
        print(f"  OA       : {baseline['OA']:.4f} -> {oa:.4f} ({oa - baseline['OA']:+.4f})")
        print(f"  Kappa    : {baseline['Kappa']:.4f} -> {kappa:.4f} ({kappa - baseline['Kappa']:+.4f})")
        print(f"  F1 macro : {baseline['F1_macro']:.4f} -> {f1_macro:.4f} ({f1_macro - baseline['F1_macro']:+.4f})")

    results = {
        "dataset": "California",
        "model": "Part3CaliforniaNet",
        "n_test": int(len(y_true)),
        "OA": round(float(oa), 6),
        "Kappa": round(float(kappa), 6),
        "F1_macro": round(float(f1_macro), 6),
        "per_class_metrics": per_class,
        "class_f1_raw": [round(float(v), 6) for v in class_f1],
        "confusion_matrix": cm.tolist(),
    }
    if baseline is not None:
        results["baseline_part1"] = {
            "OA": baseline["OA"],
            "Kappa": baseline["Kappa"],
            "F1_macro": baseline["F1_macro"],
        }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
