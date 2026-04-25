"""
PART 3 — ÉTAPE 1 : EDA sur les données fusionnées (S2 + Tout)
==============================================================
Entrée  : data/merged_npz/arkansas/ARK_covariates.npz
Sorties : outputs/figures/arkansas/part3/eda_*.png

Early fusion : X_all (N,36,10) + cov répétées → (N,36,18)
Classes      : Corn, Cotton, Rice, Soybean, Others
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import os

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_FILE = r"data\merged_npz\arkansas\ARK_covariates.npz"
FIG_DIR  = r"outputs\figures\arkansas\part3"
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES  = ['Corn', 'Cotton', 'Rice', 'Soybean', 'Others']
CLASS_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
N_CLASSES    = 5

S2_BANDS = ['B2 Blue','B3 Green','B4 Red','B5 RE1',
            'B6 RE2','B7 RE3','B8 NIR','B8A RE4','B11 SWIR1','B12 SWIR2']

COV_NAMES = ['T° moy.(°C)', 'Précip.(mm)', 'Rayon.(W/m²)',
             'pH Sol', 'OC Sol(g/kg)', 'Texture Sol',
             'Élévation(m)', 'Landforms']

ALL_FEAT_NAMES = S2_BANDS + COV_NAMES   # 18 features

TIMESTEP_LABELS = [
    'Jan-1','Jan-11','Jan-21','Feb-1','Feb-11','Feb-21',
    'Mar-1','Mar-11','Mar-21','Apr-1','Apr-11','Apr-21',
    'May-1','May-11','May-21','Jun-1','Jun-11','Jun-21',
    'Jul-1','Jul-11','Jul-21','Aug-1','Aug-11','Aug-21',
    'Sep-1','Sep-11','Sep-21','Oct-1','Oct-11','Oct-21',
    'Nov-1','Nov-11','Nov-21','Dec-1','Dec-11','Dec-21',
]

# ── Chargement ────────────────────────────────────────────────────────────────
def load_data():
    d = np.load(NPZ_FILE, allow_pickle=True)
    X_s2   = d['X_all']        # (N, 36, 10)
    mask   = d['mask_all']     # (N, 36)
    y      = d['y_all']        # (N,)
    c_clim = d['cov_clim']     # (N, 3)
    c_soil = d['cov_soil']     # (N, 3)
    c_topo = d['cov_topo']     # (N, 2)

    N, T, _ = X_s2.shape

    # Early fusion : répéter covariables statiques sur T=36
    c_all = np.concatenate([c_clim, c_soil, c_topo], axis=1)   # (N, 8)
    c_rep = np.repeat(c_all[:, np.newaxis, :], T, axis=1)       # (N, 36, 8)
    X_fus = np.concatenate([X_s2, c_rep], axis=2)               # (N, 36, 18)

    train_idx = d['train_idx']
    test_idx  = d['test_idx']

    print(f"✅ Chargé | N={N} | X_s2={X_s2.shape} | X_fus={X_fus.shape}")
    print(f"   Classes : {dict(zip(CLASS_NAMES, [int((y==i).sum()) for i in range(N_CLASSES)]))}")
    return X_s2, X_fus, mask, y, train_idx, test_idx


# ── Figure 1 : Distribution des classes ──────────────────────────────────────
def plot_class_distribution(y):
    counts = [(y == i).sum() for i in range(N_CLASSES)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Barres
    bars = ax1.bar(CLASS_NAMES, counts, color=CLASS_COLORS, alpha=0.85,
                   edgecolor='white', linewidth=1.2)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                 f'{cnt:,}\n({cnt/len(y)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title('Distribution des classes — Arkansas 2021', fontsize=12)
    ax1.set_ylabel('Nombre de points', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Camembert
    wedges, texts, autotexts = ax2.pie(
        counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title('Proportions des classes', fontsize=12)

    plt.suptitle(f'Répartition des classes (N={len(y):,} points)', fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_01_class_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 2 : Profils temporels S2 par classe ────────────────────────────────
def plot_s2_temporal_profiles(X_s2, mask, y):
    # Masquer les pixels nuageux avant d'agréger
    X_masked = X_s2.copy()
    for b in range(10):
        X_masked[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)

    # Sélection de 4 bandes clés
    key_bands = [2, 6, 7, 8]   # Red, NIR, RE4, SWIR1
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for ai, bi in enumerate(key_bands):
        ax = axes[ai]
        for lbl, name in enumerate(CLASS_NAMES):
            vals = X_masked[y == lbl, :, bi]  # (n_cls, 36)
            mean = np.nanmean(vals, axis=0)
            std  = np.nanstd(vals, axis=0)
            ax.plot(range(36), mean, color=CLASS_COLORS[lbl],
                    label=name, lw=2)
            ax.fill_between(range(36), mean - std, mean + std,
                            color=CLASS_COLORS[lbl], alpha=0.12)
        ax.set_title(f'{S2_BANDS[bi]} — profil temporel',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(range(0, 36, 3))
        ax.set_xticklabels(TIMESTEP_LABELS[::3], rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Réflectance (norm.)', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    plt.suptitle('Profils temporels S2 par classe — Arkansas 2021\n'
                 '(moyenne ± std, nuages masqués)', fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_02_s2_temporal_profiles.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 3 : Profils temporels S2 — toutes les 10 bandes ──────────────────
def plot_s2_all_bands(X_s2, mask, y):
    X_masked = X_s2.copy()
    for b in range(10):
        X_masked[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()

    for bi in range(10):
        ax = axes[bi]
        for lbl, name in enumerate(CLASS_NAMES):
            vals = X_masked[y == lbl, :, bi]
            mean = np.nanmean(vals, axis=0)
            ax.plot(range(36), mean, color=CLASS_COLORS[lbl], label=name, lw=1.8)
        ax.set_title(S2_BANDS[bi], fontsize=9, fontweight='bold')
        ax.set_xticks(range(0, 36, 6))
        ax.set_xticklabels(TIMESTEP_LABELS[::6], rotation=30, ha='right', fontsize=6)
        ax.grid(alpha=0.3)

    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    plt.suptitle('Profils temporels — 10 bandes S2 — Arkansas 2021', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_03_s2_all_bands.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 4 : NDVI temporel par classe ──────────────────────────────────────
def plot_ndvi(X_s2, mask, y):
    """NDVI = (NIR - Red) / (NIR + Red)  bandes 6 et 2"""
    nir = np.where(mask == 0, X_s2[:, :, 6], np.nan)
    red = np.where(mask == 0, X_s2[:, :, 2], np.nan)
    ndvi = (nir - red) / (nir + red + 1e-8)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lbl, name in enumerate(CLASS_NAMES):
        v = ndvi[y == lbl]
        m = np.nanmean(v, axis=0)
        s = np.nanstd(v, axis=0)
        ax.plot(range(36), m, color=CLASS_COLORS[lbl], label=name, lw=2.2)
        ax.fill_between(range(36), m - s, m + s,
                        color=CLASS_COLORS[lbl], alpha=0.12)

    ax.set_xticks(range(0, 36, 2))
    ax.set_xticklabels(TIMESTEP_LABELS[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('NDVI (moyenne ± std)', fontsize=11)
    ax.set_title('Profil NDVI temporel par classe — Arkansas 2021\n'
                 '(nuages masqués)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_04_ndvi_temporal.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 5 : Covariables statiques par classe ───────────────────────────────
def plot_covariates_by_class(X_fus, y):
    """Distributions des 8 covariables (issues de la fusion)"""
    # On prend le timestep 0 (toutes identiques pour les statiques)
    cov = X_fus[:, 0, 10:]   # (N, 8) — les 8 covariables

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for fi in range(8):
        ax = axes[fi]
        for lbl, name in enumerate(CLASS_NAMES):
            vals = cov[y == lbl, fi]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=30, alpha=0.55, density=True,
                    color=CLASS_COLORS[lbl], label=name)
        ax.set_title(COV_NAMES[fi], fontsize=10, fontweight='bold')
        ax.set_ylabel('Densité', fontsize=8)
        ax.grid(alpha=0.3)

        # ANOVA
        groups = [cov[y == lbl, fi][np.isfinite(cov[y == lbl, fi])]
                  for lbl in range(N_CLASSES)]
        try:
            _, p = sp_stats.f_oneway(*groups)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else 'ns'))
            ax.text(0.98, 0.97, f'ANOVA {sig}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='navy')
        except Exception:
            pass

    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.suptitle('Distributions des 8 covariables par classe — Arkansas 2021',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_05_covariates_by_class.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 6 : Matrice de corrélation — features fusionnées ──────────────────
def plot_correlation_matrix(X_fus):
    """Corrélation sur les 18 features (S2 agrégé + covariables)"""
    # Agréger S2 : moyenne temporelle par bande
    s2_mean = np.nanmean(X_fus[:, :, :10], axis=1)   # (N, 10)
    cov     = X_fus[:, 0, 10:]                         # (N, 8)
    X_flat  = np.concatenate([s2_mean, cov], axis=1)   # (N, 18)

    valid = np.all(np.isfinite(X_flat), axis=1)
    corr  = np.corrcoef(X_flat[valid].T)

    labels = [f'S2_{b.split()[0]}' for b in S2_BANDS] + \
             ['Clim_T', 'Clim_P', 'Clim_R',
              'Sol_pH', 'Sol_OC', 'Sol_Tex',
              'Topo_E', 'Topo_L']

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.4, square=True)

    # Séparateur S2 / covariables
    ax.axhline(10, color='black', linewidth=2)
    ax.axvline(10, color='black', linewidth=2)
    ax.set_title('Corrélation — 10 bandes S2 (moy. temporelle) + 8 covariables\n'
                 'Arkansas 2021', fontsize=12, pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_06_correlation_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 7 : PCA sur l'espace fusionné ─────────────────────────────────────
def plot_pca_fused(X_fus, y):
    """PCA 2D sur les 18 features (S2 moy. + covariables)"""
    s2_mean = np.nanmean(X_fus[:, :, :10], axis=1)
    cov     = X_fus[:, 0, 10:]
    X_flat  = np.concatenate([s2_mean, cov], axis=1)

    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(y), min(3000, len(y)), replace=False)
    sub  = X_flat[idx]; sub_y = y[idx]
    valid = np.all(np.isfinite(sub), axis=1)
    sub  = sub[valid]; sub_y = sub_y[valid]

    # Standardiser avant PCA
    mu = sub.mean(0); std = sub.std(0) + 1e-8
    sub_z = (sub - mu) / std

    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(sub_z)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, name in enumerate(CLASS_NAMES):
        m = (sub_y == lbl)
        ax.scatter(Xp[m, 0], Xp[m, 1], c=CLASS_COLORS[lbl],
                   label=name, alpha=0.45, s=12, edgecolors='none')
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'PCA — 18 features fusionnées (S2+Tout)\n'
                 f'n={valid.sum()} pts — Arkansas 2021', fontsize=12)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_07_pca_fused.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 8 : Statistiques de la donnée d'entrée ─────────────────────────────
def plot_data_stats(X_s2, mask, y):
    """Stats sur la qualité des données S2 (couverture nuageuse, etc.)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. % pixels nuageux par timestep
    cloud_pct = mask.mean(axis=0) * 100  # (36,)
    axes[0].bar(range(36), cloud_pct, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(0, 36, 3))
    axes[0].set_xticklabels(TIMESTEP_LABELS[::3], rotation=40, ha='right', fontsize=7)
    axes[0].set_ylabel('% pixels masqués', fontsize=10)
    axes[0].set_title('Couverture nuageuse par timestep', fontsize=11)
    axes[0].grid(alpha=0.3)

    # 2. Distribution du nombre de nuages par point
    cloud_per_pt = mask.sum(axis=1)   # (N,)
    axes[1].hist(cloud_per_pt, bins=37, color='steelblue', alpha=0.8,
                 edgecolor='white')
    axes[1].set_xlabel('Nombre de timesteps masqués', fontsize=10)
    axes[1].set_ylabel('Nombre de points', fontsize=10)
    axes[1].set_title('Distribution des masques par point', fontsize=11)
    axes[1].axvline(cloud_per_pt.mean(), color='red', linestyle='--',
                    label=f'Moyenne = {cloud_per_pt.mean():.1f}')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # 3. Statistiques de réflectance par bande
    X_clean = X_s2.copy()
    for b in range(10):
        X_clean[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)
    band_means = np.nanmean(X_clean, axis=(0, 1))   # (10,)
    band_stds  = np.nanstd(X_clean,  axis=(0, 1))   # (10,)
    x_pos = np.arange(10)
    axes[2].bar(x_pos, band_means, yerr=band_stds, color='forestgreen',
                alpha=0.8, capsize=4, edgecolor='white')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([b.split()[0] for b in S2_BANDS],
                             rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('Réflectance normalisée (÷10000)', fontsize=10)
    axes[2].set_title('Moyenne ± std des 10 bandes S2', fontsize=11)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Qualité et statistiques des données — Arkansas 2021', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_08_data_stats.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Résumé console ────────────────────────────────────────────────────────────
def print_summary(X_s2, X_fus, mask, y, train_idx, test_idx):
    n = len(y)
    val_idx = np.setdiff1d(np.arange(n), np.concatenate([train_idx, test_idx]))
    print("\n" + "="*60)
    print("RÉSUMÉ EDA — Part 3 — Arkansas 2021")
    print("="*60)
    print(f"\n  Données S2          : {X_s2.shape}  (N, T=36, B=10)")
    print(f"  Données fusionnées  : {X_fus.shape}  (N, T=36, F=18)")
    print(f"  Masque nuageux      : {mask.shape}  | moy. {mask.mean()*100:.1f}% masqués")
    print(f"\n  Splits :")
    print(f"    Train : {len(train_idx):,}  ({len(train_idx)/n*100:.0f}%)")
    print(f"    Val   : {len(val_idx):,}   ({len(val_idx)/n*100:.0f}%)")
    print(f"    Test  : {len(test_idx):,}  ({len(test_idx)/n*100:.0f}%)")
    print(f"\n  Distribution des classes :")
    for i, name in enumerate(CLASS_NAMES):
        cnt = (y == i).sum()
        print(f"    {name:10s} : {cnt:5,}  ({cnt/n*100:.1f}%)")
    print(f"\n  Features fusionnées (18) :")
    print(f"    S2  (10) : {', '.join(S2_BANDS[:3])}... [temporelles]")
    print(f"    Clim (3) : T°, Précip., Rayonnement  [répétées sur T]")
    print(f"    Sol  (3) : pH, OC, Texture            [répétées sur T]")
    print(f"    Topo (2) : Élévation, Landforms        [répétées sur T]")
    print(f"\n  → Lancer : python Part3_Step2_preprocessing.py")
    print("="*60)


def main():
    print("="*60)
    print("Part 3 — Étape 1 : EDA données fusionnées — Arkansas")
    print("="*60 + "\n")

    if not os.path.exists(NPZ_FILE):
        print(f"❌ Fichier introuvable : {NPZ_FILE}")
        print("   Vérifier le chemin vers ARK_covariates.npz")
        return

    X_s2, X_fus, mask, y, train_idx, test_idx = load_data()

    print("\n── Génération des figures ─────────────────────────────")
    plot_class_distribution(y)
    plot_s2_temporal_profiles(X_s2, mask, y)
    plot_s2_all_bands(X_s2, mask, y)
    plot_ndvi(X_s2, mask, y)
    plot_covariates_by_class(X_fus, y)
    plot_correlation_matrix(X_fus)
    plot_pca_fused(X_fus, y)
    plot_data_stats(X_s2, mask, y)

    print_summary(X_s2, X_fus, mask, y, train_idx, test_idx)
    print(f"\n✅ 8 figures sauvegardées dans : {FIG_DIR}")


if __name__ == "__main__":
    main()
