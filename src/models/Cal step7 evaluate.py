"""
ÉTAPE 7 — Évaluation complète — California
==========================================
Entrée  : best_model_cal.pth + CAL_dataset_preprocessed.npz
Sortie  : results_CAL.json + figures

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
Métriques cibles (papier Table 5) : OA=0.852, Kappa=0.806, F1=0.829
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json, os

from CAL_Step5_mctnet import MCTNet

# ── Configuration ──────────────────────────────────────────────
DATA_FILE    = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset_preprocessed.npz"
MODEL_FILE   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\model\best_model_cal.pth"
RESULTS_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\results\results_CAL.json"
FIG_DIR      = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

N_CLASSES   = 6
CLASS_NAMES = ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others']
COLORS      = ['#9400D3','#2196F3','#FF9800','#8B4513','#4CAF50','#9E9E9E']

# Résultats papier Table 5 — California
PAPER_RESULTS = {
    'MCTNet'     : {'OA': 0.852, 'Kappa': 0.806, 'F1': 0.829},
    'GL-TAE'     : {'OA': 0.843, 'Kappa': 0.796, 'F1': 0.820},
    'Transformer': {'OA': 0.813, 'Kappa': 0.759, 'F1': 0.789},
    'ResNet'     : {'OA': 0.837, 'Kappa': 0.787, 'F1': 0.821},
    'LTAE'       : {'OA': 0.838, 'Kappa': 0.788, 'F1': 0.805},
    'RF'         : {'OA': 0.823, 'Kappa': 0.770, 'F1': 0.793},
    'SVM'        : {'OA': 0.786, 'Kappa': 0.724, 'F1': 0.762},
    'LSTM'       : {'OA': 0.799, 'Kappa': 0.739, 'F1': 0.766},
    'TAE'        : {'OA': 0.787, 'Kappa': 0.728, 'F1': 0.767},
}
# ───────────────────────────────────────────────────────────────


def confusion_matrix_fn(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def overall_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def cohen_kappa(y_true, y_pred):
    N  = len(y_true)
    cm = confusion_matrix_fn(y_true, y_pred)
    diag_sum = np.trace(cm)
    expected = float((cm.sum(axis=1) * cm.sum(axis=0)).sum())
    return float((N * diag_sum - expected) / (N**2 - expected))


def macro_f1(y_true, y_pred):
    cm = confusion_matrix_fn(y_true, y_pred)
    f1_scores = []
    for i in range(N_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2*p*r/(p+r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1_scores)), f1_scores


def per_class_metrics(y_true, y_pred):
    cm = confusion_matrix_fn(y_true, y_pred)
    results = []
    for i in range(N_CLASSES):
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
    d_model    = checkpoint.get('d_model',   30)
    n_classes  = checkpoint.get('n_classes', 6)
    print(f"d_model={d_model}  n_classes={n_classes}")

    model = MCTNet(n_bands=10, n_timesteps=36, n_classes=n_classes,
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
        for X_b, m_b, _ in loader:
            logits = model(X_b.to(device), m_b.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


def plot_confusion_matrix(cm):
    cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Prédiction', fontsize=11)
    ax.set_ylabel('Vérité terrain (CDL)', fontsize=11)
    ax.set_title('Matrice de confusion normalisée — California', fontsize=12)
    thresh = cm_plot.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_plot[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if val > thresh else 'black')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_confusion_matrix_cal.png')
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
    plt.suptitle('Comparaison Table 5 du papier — California', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_results_vs_paper_cal.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def plot_per_class_f1(per_class):
    names   = [r['class']     for r in per_class]
    f1s     = [r['f1']        for r in per_class]
    precs   = [r['precision'] for r in per_class]
    recalls = [r['recall']    for r in per_class]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-w, precs,   w, label='Precision', color='#1f77b4', alpha=0.85)
    ax.bar(x,   recalls, w, label='Recall',    color='#ff7f0e', alpha=0.85)
    ax.bar(x+w, f1s,     w, label='F1',        color='#2ca02c', alpha=0.85)
    for i, (p, r, f) in enumerate(zip(precs, recalls, f1s)):
        ax.text(i-w, p+0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i,   r+0.005, f'{r:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i+w, f+0.005, f'{f:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_title('Métriques par classe — California', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_per_class_metrics_cal.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def main():
    print("=" * 55)
    print("Étape 7 — Évaluation MCTNet — California")
    print("=" * 55 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    model, test_loader, y_true = load_model_and_data(device)

    print("\n── Inférence sur test set ──────────────────────────")
    y_pred, y_probs = predict(model, test_loader, device)

    print("\n── Métriques globales ──────────────────────────────")
    oa          = overall_accuracy(y_true, y_pred)
    kappa       = cohen_kappa(y_true, y_pred)
    f1_macro, _ = macro_f1(y_true, y_pred)
    cm          = confusion_matrix_fn(y_true, y_pred)

    print(f"  {'Métrique':8s}  {'Notre':>8s}  {'Papier':>8s}  {'Diff':>8s}  Statut")
    print(f"  {'-'*52}")
    for metric, our, paper in [('OA', oa, 0.852), ('Kappa', kappa, 0.806), ('F1', f1_macro, 0.829)]:
        diff   = our - paper
        status = '✅' if abs(diff) < 0.03 else ('⚠ ' if abs(diff) < 0.08 else '❌')
        print(f"  {metric:8s}  {our:8.4f}  {paper:8.3f}  {diff:+8.4f}  {status}")

    print("\n── Métriques par classe ────────────────────────────")
    per_class = per_class_metrics(y_true, y_pred)
    print(f"  {'Classe':12s}  {'N test':>7s}  {'Précision':>9s}  {'Rappel':>7s}  {'F1':>6s}  {'OA':>6s}")
    for r in per_class:
        print(f"  {r['class']:12s}  {r['n_test']:7d}  "
              f"{r['precision']:9.4f}  {r['recall']:7.4f}  "
              f"{r['f1']:6.4f}  {r['OA']:6.4f}")

    print("\n── Génération des figures ──────────────────────────")
    plot_confusion_matrix(cm)
    plot_results_vs_paper({'OA': oa, 'Kappa': kappa, 'F1': f1_macro})
    plot_per_class_f1(per_class)

    results = {
        'dataset':   'California', 'model': 'MCTNet',
        'n_test':    int(len(y_true)),
        'OA':        round(oa, 6),
        'Kappa':     round(kappa, 6),
        'F1_macro':  round(f1_macro, 6),
        'paper_targets': {'OA': 0.852, 'Kappa': 0.806, 'F1': 0.829},
        'per_class_metrics': per_class,
        'confusion_matrix':  cm.tolist(),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Résultats sauvés : {RESULTS_FILE}")

    print("\n" + "=" * 55)
    print("RÉSUMÉ FINAL — MCTNet California")
    print("=" * 55)
    print(f"  OA    : {oa:.4f}  (papier : 0.852)")
    print(f"  Kappa : {kappa:.4f}  (papier : 0.806)")
    print(f"  F1    : {f1_macro:.4f}  (papier : 0.829)")
    gap = abs(oa - 0.852)
    if gap < 0.03:
        print("  ✅ Résultats proches du papier — reproduction réussie")
    elif gap < 0.08:
        print("  ⚠  Résultats acceptables")
    else:
        print("  ❌ Écart important — vérifier le preprocessing")
    print("=" * 55)


if __name__ == "__main__":
    main()