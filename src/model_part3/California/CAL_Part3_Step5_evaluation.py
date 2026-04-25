"""
PART 3 — ÉTAPE 5 (CALIFORNIA) : Évaluation complète — CBAM-CNN-BiLSTM
=======================================================================
Entrée  : meilleur modèle + Part3_CAL_dataset.npz
          (+ ablation_results.json de Part 2 pour comparaison)
Sorties : outputs/figures/california/part3/eval_*.png
          outputs/results/california/part3/evaluation_report.json

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
Référence papier : OA=0.852 | Kappa=0.806 | F1=0.829
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import json, os

from CAL_Part3_Step3_model import build_model, count_parameters
from CAL_Part3_Step4_train  import run_epoch, compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_FILE    = r"data\merged_npz\california\Part3_CAL_dataset.npz"
MODEL_PATH  = r"outputs\results\california\part3\models\best_cbam_cnn_bilstm_cal.pth"
RESULTS_DIR = r"outputs\results\california\part3\results"
FIG_DIR     = r"outputs\figures\california\part3"
PART2_RES   = r"outputs\results\california\part2\results\ablation_results.json"

CLASS_NAMES  = ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others']
CLASS_COLORS = ['#9400D3', '#2196F3', '#FF9800', '#8B4513', '#4CAF50', '#9E9E9E']
N_CLASSES    = 6

PAPER = {'OA': 0.852, 'Kappa': 0.806, 'F1': 0.829,
         'label': 'Papier MCTNet', 'color': 'black'}

N_FEATURES  = 13
CNN_FILTERS = (128, 64)
LSTM_UNITS  = (128, 64)
DROPOUT     = 0.3

os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────
def load_model_and_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model  = build_model(N_FEATURES, N_CLASSES, device, CNN_FILTERS, LSTM_UNITS, DROPOUT)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"✅ Modèle chargé (epoch {ckpt['epoch']} | "
          f"val_loss={ckpt['val_loss']:.4f} | val_acc={ckpt['val_acc']*100:.2f}%)")

    d        = np.load(NPZ_FILE, allow_pickle=True)
    X        = torch.tensor(d['X'], dtype=torch.float32)
    y        = torch.tensor(d['y'], dtype=torch.long)
    test_idx = d['test_idx']
    X_test   = X[test_idx]; y_test = y[test_idx]
    print(f"✅ Test set : {len(y_test):,} points — 6 classes")
    return model, X_test, y_test, device, ckpt['epoch']


# ── Inférence ─────────────────────────────────────────────────────────────────
def run_inference(model, X_test, y_test, device, batch_size=256):
    model.eval()
    all_preds, all_probs, all_trues = [], [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            Xb = X_test[i:i+batch_size].to(device)
            yb = y_test[i:i+batch_size]
            logits = model(Xb)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_trues.extend(yb.numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_trues)


# ── Métriques par classe ──────────────────────────────────────────────────────
def compute_per_class_metrics(cm):
    results = []
    for i in range(N_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.
        iou  = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0.
        results.append({
            'class'    : CLASS_NAMES[i],
            'n_true'   : int(cm[i].sum()),
            'Precision': round(float(prec), 4),
            'Recall'   : round(float(rec),  4),
            'F1'       : round(float(f1),   4),
            'IoU'      : round(float(iou),  4),
        })
    return results


# ── Figure 1 : Matrices de confusion ─────────────────────────────────────────
def plot_confusion_matrix(cm, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Normalisée
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_norm, ax=axes[0],
                annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, linewidths=0.5, annot_kws={'size': 11})
    axes[0].set_xlabel('Prédit', fontsize=12); axes[0].set_ylabel('Réel', fontsize=12)
    axes[0].set_title(f'Matrice de confusion normalisée\n'
                      f'OA={metrics["OA"]:.4f}  Kappa={metrics["Kappa"]:.4f}  '
                      f'F1={metrics["F1"]:.4f}', fontsize=11)
    axes[0].tick_params(axis='x', rotation=30)

    # Brute
    sns.heatmap(cm, ax=axes[1],
                annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, annot_kws={'size': 10})
    axes[1].set_xlabel('Prédit', fontsize=12); axes[1].set_ylabel('Réel', fontsize=12)
    axes[1].set_title('Matrice de confusion — Counts', fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle('CBAM-CNN-BiLSTM — California 2021 (S2+Sol — 6 classes)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_01_confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 2 : Métriques par classe ──────────────────────────────────────────
def plot_per_class_metrics(per_class):
    metrics_m = ['Precision', 'Recall', 'F1', 'IoU']
    colors_m  = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
    x = np.arange(N_CLASSES); w = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (m, c) in enumerate(zip(metrics_m, colors_m)):
        vals = [pc[m] for pc in per_class]
        bars = ax.bar(x + (i-1.5)*w, vals, w, color=c, alpha=0.85,
                      edgecolor='white', label=m)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{pc['class']}\n(n={pc['n_true']:,})" for pc in per_class], fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Métriques par classe — CBAM-CNN-BiLSTM — California 2021\n'
                 '(6 cultures : Grapes, Rice, Alfalfa, Almonds, Pistachios, Others)',
                 fontsize=11)
    ax.legend(fontsize=10, ncol=4)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.9, color='gray', linestyle='--', lw=0.8, alpha=0.7)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_02_per_class_metrics.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 3 : Confiance vs précision ────────────────────────────────────────
def plot_confidence_curve(preds, probs, trues):
    thresholds = np.arange(0.3, 1.0, 0.02)
    oas, coverages = [], []
    for thr in thresholds:
        conf     = probs.max(axis=1)
        mask     = conf >= thr
        coverage = mask.mean()
        oas.append((preds[mask] == trues[mask]).mean() if mask.sum() >= 10 else np.nan)
        coverages.append(coverage)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    l1, = ax1.plot(thresholds, oas, 'b-o', markersize=4, lw=2,
                   label='OA (conf ≥ seuil)')
    l2, = ax2.plot(thresholds, [c*100 for c in coverages],
                   'r--s', markersize=4, lw=2, label='Couverture (%)')
    ax1.axhline(PAPER['OA'], color='gray', linestyle=':', lw=1.5, label='Papier (0.852)')
    ax1.set_xlabel('Seuil de confiance', fontsize=11)
    ax1.set_ylabel('Overall Accuracy', fontsize=11, color='blue')
    ax2.set_ylabel('Couverture (%)',    fontsize=11, color='red')
    ax1.set_ylim(0.65, 1.0); ax2.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(handles=[l1, l2,
               plt.Line2D([0],[0],color='gray',linestyle=':',lw=1.5,label='Papier')],
               fontsize=9, loc='lower right')
    ax1.set_title('OA vs seuil de confiance — CBAM-CNN-BiLSTM — California', fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_03_confidence_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 4 : Comparaison des modèles ───────────────────────────────────────
def plot_comparison(metrics_p3, part2_results=None):
    models = []
    if part2_results:
        for c, r in part2_results.items():
            models.append({'label': r['label'], 'OA': r['OA'],
                           'Kappa': r['Kappa'], 'F1': r['F1'],
                           'color': r['color']})
    models.append({
        'label': 'CBAM-CNN-BiLSTM\n(Part 3)',
        'OA': metrics_p3['OA'], 'Kappa': metrics_p3['Kappa'],
        'F1': metrics_p3['F1'], 'color': '#e6194b'
    })
    models.append({**PAPER})

    metric_keys   = ['OA', 'Kappa', 'F1']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, mk, mc in zip(axes, metric_keys, metric_colors):
        vals   = [m[mk] for m in models]
        colors = [m['color'] for m in models]
        bars   = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.axhline(PAPER[mk], color='gray', linestyle='--', lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m['label'] for m in models],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylim(max(0, min(vals) - 0.06), 1.02)
        ax.set_title(mk, fontsize=14, fontweight='bold', color=mc)
        ax.grid(axis='y', alpha=0.3)
        # Highlight Part 3
        bars[len(models)-2].set_edgecolor('#e6194b')
        bars[len(models)-2].set_linewidth(2)

    plt.suptitle('Comparaison des modèles — California 2021\n'
                 '(S2+Sol | trait pointillé = papier MCTNet)', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_04_model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 5 : Analyse des erreurs ────────────────────────────────────────────
def plot_error_analysis(cm, preds, probs, trues):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Erreurs hors diagonale
    cm_off = cm.astype(float).copy()
    np.fill_diagonal(cm_off, 0)
    cm_off_norm = cm_off / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    sns.heatmap(cm_off_norm, ax=axes[0],
                annot=True, fmt='.1f', cmap='Reds',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, annot_kws={'size': 10})
    axes[0].set_xlabel('Prédit', fontsize=12)
    axes[0].set_ylabel('Réel (Confusion %)', fontsize=12)
    axes[0].set_title('Taux d\'erreur entre classes (%) — California\n'
                      '(hors diagonale)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=30)

    # Confiance correct vs incorrect
    conf    = probs.max(axis=1)
    correct = (preds == trues)
    axes[1].hist(conf[correct],  bins=40, alpha=0.6, color='forestgreen',
                 label='Correct', density=True)
    axes[1].hist(conf[~correct], bins=40, alpha=0.6, color='tomato',
                 label='Incorrect', density=True)
    axes[1].set_xlabel('Confiance (max proba)', fontsize=11)
    axes[1].set_ylabel('Densité', fontsize=11)
    axes[1].set_title('Distribution de confiance — Correct vs Incorrect', fontsize=11)
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)
    axes[1].axvline(0.5, color='gray', linestyle='--', lw=1)

    plt.suptitle('Analyse des erreurs — CBAM-CNN-BiLSTM — California 2021',
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_05_error_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 6 : Radar comparatif ───────────────────────────────────────────────
def plot_radar_comparison(metrics_p3, part2_results=None):
    cats   = ['OA', 'Kappa', 'F1']
    angles = [n/3*2*np.pi for n in range(3)] + [0]
    vmin, vmax = 0.70, 1.0

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    def norm(v): return (v - vmin) / (vmax - vmin)

    if part2_results:
        for c, r in part2_results.items():
            vals = [norm(r[m]) for m in cats] + [norm(r[cats[0]])]
            ax.plot(angles, vals, 'o-', lw=1.5, color=r['color'],
                    label=r['label'], alpha=0.6)
            ax.fill(angles, vals, alpha=0.04, color=r['color'])

    # Notre modèle
    vals3 = [norm(metrics_p3[m]) for m in cats] + [norm(metrics_p3[cats[0]])]
    ax.plot(angles, vals3, 'D-', lw=2.5, color='#e6194b',
            label='CBAM-CNN-BiLSTM (Part 3)')
    ax.fill(angles, vals3, alpha=0.15, color='#e6194b')

    # Papier
    vp = [norm(PAPER[m]) for m in cats] + [norm(PAPER[cats[0]])]
    ax.plot(angles, vp, 's--', lw=1.8, color='black',
            label='Papier MCTNet', alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=14, fontweight='bold')
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 5)], size=8)
    ax.set_title(f'Radar — Comparaison modèles — California\n[{vmin:.2f}, {vmax:.2f}]',
                 fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eval_06_radar_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Rapport final ─────────────────────────────────────────────────────────────
def print_final_report(metrics, per_class, best_epoch, n_params, part2_results=None):
    print("\n" + "="*65)
    print("RAPPORT FINAL — Part 3 — CBAM-CNN-BiLSTM — California 2021")
    print("="*65)
    print(f"""
  Modèle       : CBAM-CNN-BiLSTM (architecture originale Part 3)
  Données      : S2+Sol (Early Fusion) → (N, 36, 13)
  Paramètres   : {n_params:,}
  Best epoch   : {best_epoch}
  Classes (6)  : {', '.join(CLASS_NAMES)}
""")
    print(f"  ── Métriques globales ──────────────────────────────")
    print(f"  {'Modèle':32s}  {'OA':>7}  {'Kappa':>7}  {'F1':>7}")
    print(f"  {'-'*57}")

    if part2_results:
        for c, r in part2_results.items():
            print(f"  {r['label']:32s}  {r['OA']:7.4f}  "
                  f"{r['Kappa']:7.4f}  {r['F1']:7.4f}")
        print(f"  {'-'*57}")

    print(f"  {'CBAM-CNN-BiLSTM (Part 3)':32s}  "
          f"{metrics['OA']:7.4f}  {metrics['Kappa']:7.4f}  "
          f"{metrics['F1']:7.4f}  ← Notre modèle")
    print(f"  {'Papier MCTNet':32s}  {PAPER['OA']:7.3f}  "
          f"{PAPER['Kappa']:7.3f}  {PAPER['F1']:7.3f}  ← Référence")

    delta_oa = metrics['OA'] - PAPER['OA']
    print(f"\n  ΔOA vs Papier : {delta_oa:+.4f}  "
          f"({'≥ papier ✅' if delta_oa >= 0 else 'en dessous du papier'})")

    print(f"\n  ── Métriques par classe ────────────────────────────")
    print(f"  {'Classe':12s}  {'n':>6}  {'Prec':>7}  {'Rec':>7}  "
          f"{'F1':>7}  {'IoU':>7}")
    print(f"  {'-'*57}")
    for pc in per_class:
        print(f"  {pc['class']:12s}  {pc['n_true']:6,}  "
              f"{pc['Precision']:7.4f}  {pc['Recall']:7.4f}  "
              f"{pc['F1']:7.4f}  {pc['IoU']:7.4f}")

    print(f"\n  ── Spécificités California ─────────────────────────")
    print("""
  • 6 classes vs 5 (Arkansas) : tâche plus difficile
  • S2+Sol retenu (vs S2+Tout pour Arkansas) : covariables
    édaphiques (pH, OC, Texture) plus discriminantes en
    California qu'en Arkansas (diversité pédologique plus grande)
  • Cultures méditerranéennes (Vigne, Amandiers, Pistachiers) :
    phénologies distinctes bien capturées par le BiLSTM
  • Le CBAM temporal attend : floraison précoce (fév-mars)
    et feuillaison tardive comme marqueurs discriminants
""")
    print("="*65)
    print("✅ Part 3 — California terminée !")
    print("="*65)


def main():
    print("="*60)
    print("Part 3 — Étape 5 : Évaluation — California")
    print("="*60 + "\n")

    model, X_test, y_test, device, best_epoch = load_model_and_data()

    print("\n── Inférence sur le test set ───────────────────────")
    preds, probs, trues = run_inference(model, X_test, y_test, device)

    metrics = compute_metrics(preds.tolist(), trues.tolist(), nc=N_CLASSES)
    cm = metrics.pop('cm')
    metrics.pop('f1_per_class')
    print(f"  OA    = {metrics['OA']:.4f}")
    print(f"  Kappa = {metrics['Kappa']:.4f}")
    print(f"  F1    = {metrics['F1']:.4f}")

    per_class = compute_per_class_metrics(cm)

    part2_results = None
    if os.path.exists(PART2_RES):
        with open(PART2_RES) as f:
            part2_results = json.load(f)
        print(f"\n✅ Résultats Part 2 California chargés")
    else:
        print(f"\n⚠ Résultats Part 2 non trouvés → comparaison vs papier seulement")

    print("\n── Génération des figures ──────────────────────────")
    plot_confusion_matrix(cm, metrics)
    plot_per_class_metrics(per_class)
    plot_confidence_curve(preds, probs, trues)
    plot_comparison(metrics, part2_results)
    plot_error_analysis(cm, preds, probs, trues)
    plot_radar_comparison(metrics, part2_results)

    n_total, n_train = count_parameters(model)
    report = {
        'model'           : 'CBAM-CNN-BiLSTM',
        'state'           : 'California 2021',
        'config'          : 'S2+Sol (early fusion, 13 features)',
        'n_classes'       : N_CLASSES,
        'class_names'     : CLASS_NAMES,
        'best_epoch'      : best_epoch,
        'n_params'        : n_train,
        'n_test_samples'  : int(len(trues)),
        **metrics,
        'per_class'       : per_class,
        'confusion_matrix': cm.tolist(),
        'paper_reference' : PAPER,
    }
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Rapport sauvegardé : {report_path}")

    print_final_report(metrics, per_class, best_epoch, n_train, part2_results)


if __name__ == "__main__":
    main()
