"""
ÉTAPE 7 — Évaluation complète (v2 corrigée)
============================================
Correction : charge d_model depuis le checkpoint sauvegardé par step6
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json, os

from step5_mctnet import MCTNet

# ── Configuration ──────────────────────────────────────────────
DATA_FILE    = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset_preprocessed.npz"
MODEL_FILE   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\model\best_model.pth"
RESULTS_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\results\results_ARK.json"
FIG_DIR      = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

CLASS_NAMES  = ['Corn', 'Cotton', 'Rice', 'Soybean', 'Others']
COLORS       = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']

PAPER_RESULTS = {
    'MCTNet'    : {'OA': 0.968, 'Kappa': 0.951, 'F1': 0.933},
    'GL-TAE'    : {'OA': 0.952, 'Kappa': 0.928, 'F1': 0.905},
    'Transformer': {'OA': 0.956, 'Kappa': 0.934, 'F1': 0.909},
    'ResNet'    : {'OA': 0.945, 'Kappa': 0.917, 'F1': 0.894},
    'LTAE'      : {'OA': 0.941, 'Kappa': 0.912, 'F1': 0.884},
    'RF'        : {'OA': 0.936, 'Kappa': 0.904, 'F1': 0.887},
    'SVM'       : {'OA': 0.910, 'Kappa': 0.867, 'F1': 0.846},
    'LSTM'      : {'OA': 0.907, 'Kappa': 0.862, 'F1': 0.843},
    'TAE'       : {'OA': 0.911, 'Kappa': 0.869, 'F1': 0.854},
}
# ───────────────────────────────────────────────────────────────


def confusion_matrix_fn(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def overall_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def cohen_kappa(y_true, y_pred, n_classes=5):
    N  = len(y_true)
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    diag_sum = np.trace(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = float((row_sums * col_sums).sum())
    return float((N * diag_sum - expected) / (N**2 - expected))


def macro_f1(y_true, y_pred, n_classes=5):
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    f1_scores = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)), f1_scores


def per_class_metrics(y_true, y_pred, n_classes=5):
    cm = confusion_matrix_fn(y_true, y_pred, n_classes)
    results = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p + r) > 0 else 0.0
        results.append({
            'class':     CLASS_NAMES[i],
            'n_test':    int(cm[i, :].sum()),
            'precision': round(float(p),  4),
            'recall':    round(float(r),  4),
            'f1':        round(float(f1), 4),
            'OA':        round(float((tp + tn) / cm.sum()), 4),
        })
    return results


def load_model_and_data(device):
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    # Récupérer d_model depuis le checkpoint (sauvegardé par step6 v2)
    d_model = checkpoint.get('d_model', 30)
    print(f"d_model chargé depuis checkpoint : {d_model}")

    model = MCTNet(n_bands=10, n_timesteps=36, n_classes=5,
                   n_stage=3, n_head=5, kernel_size=3,
                   d_model=d_model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modèle chargé : epoch {checkpoint['epoch']}  "
          f"val_loss={checkpoint['val_loss']:.4f}  "
          f"val_acc={checkpoint['val_acc']*100:.2f}%")

    data    = np.load(DATA_FILE)
    X_test  = torch.tensor(data['X_test'],    dtype=torch.float32)
    y_test  = torch.tensor(data['y_test'],    dtype=torch.long)
    m_test  = torch.tensor(data['mask_test'], dtype=torch.float32)

    test_loader = DataLoader(TensorDataset(X_test, m_test, y_test),
                             batch_size=256, shuffle=False)
    return model, test_loader, data['y_test']


def predict(model, loader, device):
    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, m_batch, _ in loader:
            logits = model(X_batch.to(device), m_batch.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


def plot_confusion_matrix(cm):
    cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel('Prédiction', fontsize=11)
    ax.set_ylabel('Vérité terrain (CDL)', fontsize=11)
    ax.set_title('Matrice de confusion normalisée — Arkansas', fontsize=12)
    thresh = cm_plot.max() / 2.0
    for i in range(5):
        for j in range(5):
            val = cm_plot[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color='white' if val > thresh else 'black')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def plot_results_vs_paper(our_results):
    models  = list(PAPER_RESULTS.keys())
    metrics = ['OA', 'Kappa', 'F1']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        values     = [PAPER_RESULTS[m][metric] for m in models]
        colors_bar = ['#d62728' if m == 'MCTNet' else '#aec7e8' for m in models]
        bars = ax.bar(range(len(models)), values, color=colors_bar, alpha=0.85, edgecolor='white')
        our_val = our_results[metric]
        ax.axhline(our_val, color='#2ca02c', linewidth=2.5,
                   linestyle='--', label=f'Notre : {our_val:.3f}')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(max(0, min(min(values), our_val) - 0.05), 1.01)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Comparaison Table 5 du papier — Arkansas', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_results_vs_paper.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def plot_per_class_f1(per_class):
    names   = [r['class']     for r in per_class]
    f1s     = [r['f1']        for r in per_class]
    precs   = [r['precision'] for r in per_class]
    recalls = [r['recall']    for r in per_class]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w, precs,   w, label='Precision', color='#1f77b4', alpha=0.85)
    ax.bar(x,   recalls, w, label='Recall',    color='#ff7f0e', alpha=0.85)
    ax.bar(x+w, f1s,     w, label='F1',        color='#2ca02c', alpha=0.85)
    for i, (p, r, f) in enumerate(zip(precs, recalls, f1s)):
        ax.text(i-w, p+0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i,   r+0.005, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i+w, f+0.005, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_title('Métriques par classe — Test set Arkansas', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_per_class_metrics.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def main():
    print("=" * 55)
    print("Étape 7 — Évaluation MCTNet (v2 corrigée)")
    print("=" * 55 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    model, test_loader, y_true = load_model_and_data(device)

    print("\n── Inférence sur test set ──────────────────────────")
    y_pred, y_probs = predict(model, test_loader, device)

    print("\n── Métriques globales ──────────────────────────────")
    oa           = overall_accuracy(y_true, y_pred)
    kappa        = cohen_kappa(y_true, y_pred)
    f1_macro, _  = macro_f1(y_true, y_pred)
    cm           = confusion_matrix_fn(y_true, y_pred)

    print(f"  {'Métrique':8s}  {'Notre':>8s}  {'Papier':>8s}  {'Diff':>8s}  Statut")
    print(f"  {'-'*52}")
    for metric, our, paper in [('OA', oa, 0.968), ('Kappa', kappa, 0.951), ('F1', f1_macro, 0.933)]:
        diff   = our - paper
        status = '✅' if abs(diff) < 0.03 else ('⚠ ' if abs(diff) < 0.08 else '❌')
        print(f"  {metric:8s}  {our:8.4f}  {paper:8.3f}  {diff:+8.4f}  {status}")

    print("\n── Métriques par classe ────────────────────────────")
    per_class = per_class_metrics(y_true, y_pred)
    print(f"  {'Classe':8s}  {'N test':>7s}  {'Précision':>9s}  {'Rappel':>7s}  {'F1':>6s}  {'OA':>6s}")
    for r in per_class:
        print(f"  {r['class']:8s}  {r['n_test']:7d}  "
              f"{r['precision']:9.4f}  {r['recall']:7.4f}  "
              f"{r['f1']:6.4f}  {r['OA']:6.4f}")

    print("\n── Génération des figures ──────────────────────────")
    plot_confusion_matrix(cm)
    plot_results_vs_paper({'OA': oa, 'Kappa': kappa, 'F1': f1_macro})
    plot_per_class_f1(per_class)

    results = {
        'dataset': 'Arkansas', 'model': 'MCTNet',
        'n_test':  int(len(y_true)),
        'OA':      round(oa, 6), 'Kappa': round(kappa, 6), 'F1_macro': round(f1_macro, 6),
        'paper_targets': {'OA': 0.968, 'Kappa': 0.951, 'F1': 0.933},
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Résultats sauvés : {RESULTS_FILE}")

    print("\n" + "=" * 55)
    print("RÉSUMÉ FINAL — MCTNet Arkansas (v2)")
    print("=" * 55)
    print(f"  OA    : {oa:.4f}  (papier : 0.968)")
    print(f"  Kappa : {kappa:.4f}  (papier : 0.951)")
    print(f"  F1    : {f1_macro:.4f}  (papier : 0.933)")
    gap = abs(oa - 0.968)
    if gap < 0.03:
        print("  ✅ Résultats proches du papier — reproduction réussie")
    elif gap < 0.08:
        print("  ⚠  Résultats acceptables — essayer plus d'epochs ou GPU")
    else:
        print("  ❌ Écart important — vérifier le preprocessing des données")
    print("=" * 55)


if __name__ == "__main__":
    main()