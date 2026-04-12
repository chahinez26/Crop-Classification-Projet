"""
ÉTAPE 2 — Exploration des données (EDA)
========================================
Entrée  : ARK_dataset.npz
Sorties : figures/  (PNG)
  - fig_ndvi_timeseries.png   ← reproduit Fig. 2(a) du papier
  - fig_class_distribution.png
  - fig_missing_heatmap.png
  - fig_band_correlation.png

Installer : pip install matplotlib seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Configuration ──────────────────────────────────────────────
DATASET_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\ARK_dataset.npz"
FIG_DIR      = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\figures"
BAND_NAMES   = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
CLASS_NAMES  = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'}
CLASS_COLORS = {0:'#2ca02c', 1:'#d62728', 2:'#1f77b4',
                3:'#ff7f0e', 4:'#9467bd'}
# Dates de début de chaque fenêtre de 10 jours (pour l'axe X)
TIMESTEP_LABELS = [
    'Jan-1','Jan-11','Jan-21','Feb-1','Feb-11','Feb-21',
    'Mar-1','Mar-11','Mar-21','Apr-1','Apr-11','Apr-21',
    'May-1','May-11','May-21','Jun-1','Jun-11','Jun-21',
    'Jul-1','Jul-11','Jul-21','Aug-1','Aug-11','Aug-21',
    'Sep-1','Sep-11','Sep-21','Oct-1','Oct-11','Oct-21',
    'Nov-1','Nov-11','Nov-21','Dec-1','Dec-11','Dec-21',
]
# ───────────────────────────────────────────────────────────────

os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    data = np.load(DATASET_FILE)
    X, y, mask = data['X'], data['y'], data['mask']
    print(f"Dataset chargé : X{X.shape}  y{y.shape}  mask{mask.shape}")
    print(f"Missing global : {100*mask.mean():.1f}%\n")
    return X, y, mask


def compute_ndvi(X, mask):
    """NDVI = (B08 - B04) / (B08 + B04). Pixels missing → NaN."""
    b8  = X[:, :, 6].copy()   # B08 = NIR (index 6)
    b4  = X[:, :, 2].copy()   # B04 = Red (index 2)
    denom = b8 + b4
    ndvi = np.where((denom > 0) & (mask == 0), (b8 - b4) / denom, np.nan)
    return ndvi


# ── Figure 1 : Courbes NDVI par classe (reproduit Fig. 2a du papier) ─────────
def plot_ndvi_timeseries(X, y, mask):
    ndvi = compute_ndvi(X, mask)
    t_axis = np.arange(36)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#f8f8f8')

    for lbl, name in CLASS_NAMES.items():
        idx   = np.where(y == lbl)[0]
        ndvi_class = ndvi[idx, :]                    # [N_class, 36]
        mean  = np.nanmean(ndvi_class, axis=0)
        std   = np.nanstd(ndvi_class, axis=0)
        color = CLASS_COLORS[lbl]

        ax.plot(t_axis, mean, color=color, linewidth=2, label=name, zorder=3)
        ax.fill_between(t_axis, mean - std, mean + std,
                        color=color, alpha=0.12, zorder=2)

    # Repères saisonniers
    for x, season in [(0,'Winter'), (9,'Spring'), (18,'Summer'), (27,'Autumn')]:
        ax.axvline(x, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.text(x + 0.3, 0.82, season, fontsize=8, color='gray', va='top')

    ax.set_xticks(t_axis[::3])
    ax.set_xticklabels(TIMESTEP_LABELS[::3], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Timestep (10-day intervals)', fontsize=11)
    ax.set_ylabel('NDVI', fontsize=11)
    ax.set_title('NDVI time-series profiles — Arkansas (reproduces Fig. 2a)', fontsize=12)
    ax.set_ylim(-0.1, 0.9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ndvi_timeseries.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


# ── Figure 2 : Distribution des classes ──────────────────────────────────────
def plot_class_distribution(y):
    paper = {0:1522, 1:762, 2:2423, 3:4677, 4:616}
    names = [CLASS_NAMES[i] for i in range(5)]
    our_counts   = [int((y == i).sum()) for i in range(5)]
    paper_counts = [paper[i] for i in range(5)]
    colors       = [CLASS_COLORS[i] for i in range(5)]

    x    = np.arange(5)
    w    = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, our_counts,   w, label='Notre dataset',    color=colors, alpha=0.9)
    ax.bar(x + w/2, paper_counts, w, label='Papier (Table 2)', color=colors, alpha=0.45,
           edgecolor='black', linewidth=0.8)

    for i, (oc, pc) in enumerate(zip(our_counts, paper_counts)):
        ax.text(i - w/2, oc + 40, str(oc), ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, pc + 40, str(pc), ha='center', va='bottom', fontsize=9,
                color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Nombre de points', fontsize=11)
    ax.set_title('Distribution des classes — Arkansas', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(paper_counts) * 1.18)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_class_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


# ── Figure 3 : Heatmap des valeurs manquantes ────────────────────────────────
def plot_missing_heatmap(mask, y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gauche : taux de missing par timestep (courbe)
    miss_pct = mask.mean(axis=0) * 100
    ax = axes[0]
    ax.bar(np.arange(36), miss_pct,
           color=['#d62728' if v > 30 else '#1f77b4' for v in miss_pct],
           alpha=0.8)
    ax.set_xticks(np.arange(0, 36, 3))
    ax.set_xticklabels(TIMESTEP_LABELS[::3], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('% pixels manquants', fontsize=11)
    ax.set_title('Taux de missing par timestep', fontsize=12)
    ax.axhline(miss_pct.mean(), color='black', linestyle='--',
               linewidth=1, label=f'Moyenne : {miss_pct.mean():.1f}%')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Droite : heatmap missing par classe × timestep
    ax2 = axes[1]
    miss_by_class = np.zeros((5, 36))
    for lbl in range(5):
        idx = np.where(y == lbl)[0]
        miss_by_class[lbl, :] = mask[idx, :].mean(axis=0) * 100

    sns.heatmap(miss_by_class,
                ax=ax2,
                xticklabels=[TIMESTEP_LABELS[i] if i % 3 == 0 else ''
                             for i in range(36)],
                yticklabels=[CLASS_NAMES[i] for i in range(5)],
                cmap='RdYlGn_r',
                vmin=0, vmax=100,
                cbar_kws={'label': '% missing'})
    ax2.set_title('Missing par classe × timestep', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_missing_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


# ── Figure 4 : Moyennes spectrales par classe à T18 (été) ────────────────────
def plot_spectral_signatures(X, y, mask):
    T_ETE = 17   # T18 = juillet = pleine saison
    fig, ax = plt.subplots(figsize=(10, 5))

    x_bands = np.arange(10)
    for lbl, name in CLASS_NAMES.items():
        idx = np.where((y == lbl) & (mask[:, T_ETE] == 0))[0]
        if len(idx) == 0:
            continue
        mean_bands = X[idx, T_ETE, :].mean(axis=0) / 10000  # réflectance 0-1
        ax.plot(x_bands, mean_bands, 'o-', color=CLASS_COLORS[lbl],
                linewidth=2, markersize=5, label=f'{name} (n={len(idx)})')

    ax.set_xticks(x_bands)
    ax.set_xticklabels(BAND_NAMES, fontsize=9)
    ax.set_xlabel('Bande Sentinel-2', fontsize=11)
    ax.set_ylabel('Réflectance de surface', fontsize=11)
    ax.set_title('Signatures spectrales par classe — T18 (Juillet)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_spectral_signatures.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def print_summary_stats(X, y, mask):
    print("── Statistiques globales ───────────────────────────")
    print(f"  Total pixels     : {len(y)}")
    print(f"  Missing global   : {100*mask.mean():.2f}%")
    print(f"  Présents global  : {100*(1-mask.mean()):.2f}%")

    # Timestep avec le plus/moins de missing
    miss_per_t = mask.mean(axis=0) * 100
    t_worst  = int(miss_per_t.argmax())
    t_best   = int(miss_per_t.argmin())
    print(f"  Timestep le + nuageux : T{t_worst+1:02d} ({TIMESTEP_LABELS[t_worst]}) "
          f"→ {miss_per_t[t_worst]:.1f}% missing")
    print(f"  Timestep le + clair   : T{t_best+1:02d} ({TIMESTEP_LABELS[t_best]})  "
          f"→ {miss_per_t[t_best]:.1f}% missing")

    # Stats par bande à T18
    print(f"\n  Statistiques bandes à T18 (juillet) :")
    idx_present = np.where(mask[:, 17] == 0)[0]
    print(f"  {'Bande':5s}  {'Moyenne':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for b, bname in enumerate(BAND_NAMES):
        vals = X[idx_present, 17, b]
        print(f"  {bname:5s}  {vals.mean():8.0f}  {vals.std():8.0f}  "
              f"{vals.min():8.0f}  {vals.max():8.0f}")
    print()


def main():
    print("=" * 55)
    print("Étape 2 — Exploration des données (EDA)")
    print("=" * 55 + "\n")

    X, y, mask = load_data()

    print_summary_stats(X, y, mask)

    print("── Génération des figures ──────────────────────────")
    plot_ndvi_timeseries(X, y, mask)
    plot_class_distribution(y)
    plot_missing_heatmap(mask, y)
    plot_spectral_signatures(X, y, mask)

    print(f"\n✅ Toutes les figures dans : {FIG_DIR}/")
    print("   → Lancer maintenant : python step3_preprocess.py")


if __name__ == "__main__":
    main()