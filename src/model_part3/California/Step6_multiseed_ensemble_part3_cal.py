"""
STEP 6 - Multi-seed ensemble for California
===========================================
Usage:
  python Step6_multiseed_ensemble_part3_cal.py --tags s42 s123 s777
"""

import argparse
import json
import os
import sys
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"
PART1_DIR = os.path.join(ROOT_DIR, "California", "part_1", "MCTNet_California_v2")
PART3_DIR = os.path.join(ROOT_DIR, "California", "part_3")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if PART1_DIR not in sys.path:
    sys.path.insert(0, PART1_DIR)

from CAL_Step5_mctnet import MCTNet  # noqa: E402
from Step2_model_part3_cal import Part3CaliforniaNet  # noqa: E402


PART1_DATA_FILE = os.path.join(PART1_DIR, "npz", "CAL_dataset_preprocessed.npz")
PART3_DATA_FILE = os.path.join(PART3_DIR, "npz", "CAL_part3_dataset.npz")
RESULTS_DIR = os.path.join(PART3_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results_CAL_part3_multiseed_ensemble.json")

CLASS_NAMES = ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    return parser.parse_args()


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


def load_part1_model(device):
    ckpt = torch.load(os.path.join(PART1_DIR, "model", "best_model_cal.pth"), map_location=device)
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
    return model


def load_part3_model(device, tag):
    ckpt = torch.load(os.path.join(PART3_DIR, "model", f"best_model_part3_cal_{tag}.pth"), map_location=device)
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
    return model


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


def collect_part1_probs(model, loader, device):
    probs_all = []
    y_all = []
    with torch.no_grad():
        for x1, m1, _, _, y in loader:
            logits = model(x1.to(device), m1.to(device))
            probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(y_all)


def collect_part3_probs(models, loader, device):
    probs_by_tag = {}
    y_all = None
    for tag, model in models.items():
        probs_all = []
        labels = []
        with torch.no_grad():
            for _, _, x3, m3, y in loader:
                logits = model(x3.to(device), m3.to(device))
                probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())
                labels.append(y.numpy())
        probs_by_tag[tag] = np.concatenate(probs_all)
        y_all = np.concatenate(labels)
    return probs_by_tag, y_all


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


def average_subset(probs_by_tag, subset):
    return np.stack([probs_by_tag[tag] for tag in subset], axis=0).mean(axis=0)


def search_best_subset_and_weight(y_val, val_part1_probs, val_probs_by_tag, tags):
    best = None
    for r in range(1, len(tags) + 1):
        for subset in itertools.combinations(tags, r):
            subset_probs = average_subset(val_probs_by_tag, subset)
            for weight in np.linspace(0.0, 1.0, 21):
                fused = (1.0 - weight) * val_part1_probs + weight * subset_probs
                pred = fused.argmax(axis=1)
                scores = score_predictions(y_val, pred)
                if best is None or scores["selection_score"] > best["selection_score"]:
                    best = {"subset": list(subset), "weight_part3": float(weight), **scores}
    return best


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 6 - Multi-seed ensemble California")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tags  : {', '.join(args.tags)}\n")

    loaders = load_datasets()
    part1_model = load_part1_model(device)
    part3_models = {tag: load_part3_model(device, tag) for tag in args.tags}

    val_part1_probs, y_val = collect_part1_probs(part1_model, loaders["val"], device)
    val_probs_by_tag, _ = collect_part3_probs(part3_models, loaders["val"], device)
    best = search_best_subset_and_weight(y_val, val_part1_probs, val_probs_by_tag, args.tags)

    print("Best validation configuration:")
    print(f"  Part 3 subset   : {best['subset']}")
    print(f"  Part 3 weight   : {best['weight_part3']:.2f}")
    print(f"  OA              : {best['OA']:.4f}")
    print(f"  Kappa           : {best['Kappa']:.4f}")
    print(f"  F1 macro        : {best['F1_macro']:.4f}\n")

    test_part1_probs, y_test = collect_part1_probs(part1_model, loaders["test"], device)
    test_probs_by_tag, _ = collect_part3_probs(part3_models, loaders["test"], device)
    part3_test_probs = average_subset(test_probs_by_tag, best["subset"])
    fused_test = (1.0 - best["weight_part3"]) * test_part1_probs + best["weight_part3"] * part3_test_probs
    y_pred = fused_test.argmax(axis=1)

    oa = overall_accuracy(y_test, y_pred)
    kappa = cohen_kappa(y_test, y_pred)
    f1_macro, class_f1 = macro_f1(y_test, y_pred)
    cm = confusion_matrix_fn(y_test, y_pred)
    per_class = per_class_metrics(y_test, y_pred)

    print("Test metrics:")
    print(f"  OA       : {oa:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  F1 macro : {f1_macro:.4f}")

    results = {
        "dataset": "California",
        "model": "MultiSeedEnsemble(Part1_MCTNet, Part3CaliforniaNet)",
        "part3_subset": best["subset"],
        "weight_part3": round(float(best["weight_part3"]), 4),
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

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
