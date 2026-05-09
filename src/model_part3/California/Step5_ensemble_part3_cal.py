"""
STEP 5 - Ensemble Part 1 + Part 3 for California
================================================
"""

import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


PART1_DIR = os.path.join(ROOT_DIR, "California", "part_1", "MCTNet_California_v2")
PART3_DIR = os.path.join(ROOT_DIR, "California", "part_3")

sys.path.insert(0, PART1_DIR)
from CAL_Step5_mctnet import MCTNet  
from Step2_model_part3_cal import Part3CaliforniaNet  


PART1_DATA_FILE = os.path.join(PART1_DIR, "npz", "CAL_dataset_preprocessed.npz")
PART3_DATA_FILE = os.path.join(PART3_DIR, "npz", "CAL_part3_dataset.npz")
RESULTS_DIR = os.path.join(PART3_DIR, "results")

CLASS_NAMES = ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part3-tag", type=str, default="")
    return parser.parse_args()


def resolve_paths(part3_tag=""):
    suffix3 = f"_{part3_tag}" if part3_tag else ""
    part1_model_file = os.path.join(PART1_DIR, "model", "best_model_cal.pth")
    part3_model_file = os.path.join(PART3_DIR, "model", f"best_model_part3_cal{suffix3}.pth")
    results_file = os.path.join(RESULTS_DIR, f"results_CAL_part3_ensemble{suffix3}.json")
    return part1_model_file, part3_model_file, results_file


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
    rows = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "class": CLASS_NAMES[i],
                "n_test": int(cm[i, :].sum()),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }
        )
    return rows


def load_part1_model(device, model_file):
    ckpt = torch.load(model_file, map_location=device)
    d_model = ckpt.get("d_model", 30)
    n_classes = ckpt.get("n_classes", 6)
    model = MCTNet(
        n_bands=10,
        n_timesteps=36,
        n_classes=n_classes,
        n_stage=3,
        n_head=5,
        kernel_size=3,
        d_model=d_model,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_part3_model(device, model_file):
    ckpt = torch.load(model_file, map_location=device)
    cfg = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_datasets():
    part1 = np.load(PART1_DATA_FILE)
    part3 = np.load(PART3_DATA_FILE)
    return {
        "val": DataLoader(
            TensorDataset(
                torch.tensor(part1["X_val"], dtype=torch.float32),
                torch.tensor(part1["mask_val"], dtype=torch.float32),
                torch.tensor(part3["X_val"], dtype=torch.float32),
                torch.tensor(part3["mask_val"], dtype=torch.float32),
                torch.tensor(part1["y_val"], dtype=torch.long),
            ),
            batch_size=256,
            shuffle=False,
        ),
        "test": DataLoader(
            TensorDataset(
                torch.tensor(part1["X_test"], dtype=torch.float32),
                torch.tensor(part1["mask_test"], dtype=torch.float32),
                torch.tensor(part3["X_test"], dtype=torch.float32),
                torch.tensor(part3["mask_test"], dtype=torch.float32),
                torch.tensor(part1["y_test"], dtype=torch.long),
            ),
            batch_size=256,
            shuffle=False,
        ),
    }


def collect_probs(model1, model3, loader, device):
    probs1_all = []
    probs3_all = []
    y_all = []
    with torch.no_grad():
        for x1, m1, x3, m3, y in loader:
            logits1 = model1(x1.to(device), m1.to(device))
            logits3 = model3(x3.to(device), m3.to(device))
            probs1_all.append(F.softmax(logits1, dim=-1).cpu().numpy())
            probs3_all.append(F.softmax(logits3, dim=-1).cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(probs1_all), np.concatenate(probs3_all), np.concatenate(y_all)


def score_predictions(y_true, y_pred):
    oa = overall_accuracy(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    f1_macro, _ = macro_f1(y_true, y_pred)
    return {
        "OA": oa,
        "Kappa": kappa,
        "F1_macro": f1_macro,
        "selection_score": 0.5 * oa + 0.5 * f1_macro,
    }


def search_best_weight(y_true, probs1, probs3):
    best = None
    print("Validation search for fusion weight:")
    print(f"{'w_part3':>8s}  {'OA':>8s}  {'Kappa':>8s}  {'F1':>8s}  {'Score':>8s}")
    print("-" * 52)
    for weight in np.linspace(0.0, 1.0, 21):
        fused = (1.0 - weight) * probs1 + weight * probs3
        pred = fused.argmax(axis=1)
        scores = score_predictions(y_true, pred)
        print(
            f"{weight:8.2f}  {scores['OA']:8.4f}  {scores['Kappa']:8.4f}  "
            f"{scores['F1_macro']:8.4f}  {scores['selection_score']:8.4f}"
        )
        if best is None or scores["selection_score"] > best["selection_score"]:
            best = {"weight_part3": float(weight), **scores}
    return best


def main():
    args = parse_args()
    part1_model_file, part3_model_file, results_file = resolve_paths(args.part3_tag)

    print("=" * 60)
    print("Step 5 - Ensemble Part 1 + Part 3 California")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Part3 tag: {args.part3_tag or 'default'}\n")

    model1, ckpt1 = load_part1_model(device, part1_model_file)
    model3, ckpt3 = load_part3_model(device, part3_model_file)
    data = load_datasets()

    print(
        f"Loaded Part 1 checkpoint: epoch={ckpt1['epoch']} "
        f"val_acc={ckpt1['val_acc']*100:.2f}%"
    )
    print(
        f"Loaded Part 3 checkpoint: epoch={ckpt3['epoch']} "
        f"val_acc={ckpt3['val_acc']*100:.2f}% "
        f"val_f1={ckpt3.get('val_f1', 0.0)*100:.2f}%"
    )

    val_probs1, val_probs3, y_val = collect_probs(model1, model3, data["val"], device)
    best = search_best_weight(y_val, val_probs1, val_probs3)

    print("\nBest validation fusion:")
    print(
        f"  weight for Part 3 = {best['weight_part3']:.2f}\n"
        f"  OA={best['OA']:.4f}  Kappa={best['Kappa']:.4f}  F1={best['F1_macro']:.4f}"
    )

    test_probs1, test_probs3, y_test = collect_probs(model1, model3, data["test"], device)
    test_fused = (1.0 - best["weight_part3"]) * test_probs1 + best["weight_part3"] * test_probs3
    y_pred = test_fused.argmax(axis=1)

    oa = overall_accuracy(y_test, y_pred)
    kappa = cohen_kappa(y_test, y_pred)
    f1_macro, class_f1 = macro_f1(y_test, y_pred)
    cm = confusion_matrix_fn(y_test, y_pred)
    per_class = per_class_metrics(y_test, y_pred)

    print("\nTest metrics:")
    print(f"  OA       : {oa:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  F1 macro : {f1_macro:.4f}")

    print("\nPer-class metrics:")
    for row in per_class:
        print(
            f"  {row['class']:12s} n={row['n_test']:4d} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
        )

    results = {
        "dataset": "California",
        "model": "Ensemble(Part1_MCTNet, Part3CaliforniaNet)",
        "weight_part3": round(best["weight_part3"], 4),
        "n_test": int(len(y_test)),
        "OA": round(float(oa), 6),
        "Kappa": round(float(kappa), 6),
        "F1_macro": round(float(f1_macro), 6),
        "per_class_metrics": per_class,
        "class_f1_raw": [round(float(v), 6) for v in class_f1],
        "confusion_matrix": cm.tolist(),
        "validation_best": {
            "OA": round(float(best["OA"]), 6),
            "Kappa": round(float(best["Kappa"]), 6),
            "F1_macro": round(float(best["F1_macro"]), 6),
            "selection_score": round(float(best["selection_score"]), 6),
        },
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
