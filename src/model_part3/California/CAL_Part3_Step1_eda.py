"""
PART 3 — ÉTAPE 1 (CALIFORNIA) : EDA sur les données fusionnées (S2 + Sol)
==========================================================================
Entrée  : data/merged_npz/california/CAL_covariates.npz
Sorties : outputs/figures/california/part3/eda_*.png

Meilleure configuration (ablation Part 2) :
  S2+Sol → OA=0.848 | Kappa=0.802 | F1=0.831 (dépasse le papier sur F1 !)

Early fusion : X_all (N,36,10) + cov_soil répétée → (N,36,13)

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import os

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_FILE = r"data\merged_npz\california\CAL_covariates.npz"
FIG_DIR  = r"outputs\figures\california\part3"
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES  = ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others']
CLASS_COLORS = ['#9400D3', '#2196F3', '#FF9800', '#8B4513', '#4CAF50', '#9E9E9E']
N_CLASSES    = 6

S2_BANDS = ['B2 Blue','B3 Green','B4 Red','B5 RE1',
            'B6 RE2','B7 RE3','B8 NIR','B8A RE4','B11 SWIR1','B12 SWIR2']

# Sol uniquement (meilleure config California)
SOL_NAMES      = ['pH Sol', 'OC Sol(g/kg)', 'Texture Sol']
ALL_FEAT_NAMES = S2_BANDS + SOL_NAMES   # 13 features

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
    if not os.path.exists(NPZ_FILE):
        raise FileNotFoundError(f"❌ Fichier introuvable : {NPZ_FILE}")

    d = np.load(NPZ_FILE, allow_pickle=True)
    X_s2     = d['X_all'].astype(np.float32)      # (N, 36, 10)
    mask     = d['mask_all'].astype(np.float32)    # (N, 36)
    y        = d['y_all'].astype(np.int64)         # (N,)
    cov_soil = d['cov_soil'].astype(np.float32)   # (N, 3) — Sol uniquement
    train_idx = d['train_idx']
    test_idx  = d['test_idx']

    N, T, _ = X_s2.shape

    # Early fusion : S2 + Sol répété sur T=36
    cov_rep = np.repeat(cov_soil[:, np.newaxis, :], T, axis=1)  # (N, 36, 3)
    X_fus   = np.concatenate([X_s2, cov_rep], axis=2)            # (N, 36, 13)

    print(f"✅ Chargé | N={N} | X_s2={X_s2.shape} | X_fus={X_fus.shape}")
    print(f"   Config retenue : S2+Sol → 13 features")
    print(f"   (Meilleure ablation Part 2 : OA=0.848, F1=0.831 > papier)")
    counts = {CLASS_NAMES[i]: int((y==i).sum()) for i in range(N_CLASSES)}
    print(f"   Classes : {counts}")
    return X_s2, X_fus, mask, y, cov_soil, train_idx, test_idx


# ── Figure 1 : Distribution des classes ──────────────────────────────────────
def plot_class_distribution(y):
    counts = [(y == i).sum() for i in range(N_CLASSES)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    bars = ax1.bar(CLASS_NAMES, counts, color=CLASS_COLORS, alpha=0.85, edgecolor='white')
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{cnt:,}\n({cnt/len(y)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title('Distribution des classes — California 2021', fontsize=12)
    ax1.set_ylabel('Nombre de points', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title('Proportions des classes', fontsize=12)

    plt.suptitle(f'Répartition des 6 classes — California (N={len(y):,} points)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_01_class_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 2 : Profils temporels S2 par classe ────────────────────────────────
def plot_s2_temporal_profiles(X_s2, mask, y):
    X_masked = X_s2.copy()
    for b in range(10):
        X_masked[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)

    key_bands = [2, 6, 7, 8]   # Red, NIR, RE4, SWIR1
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ai, bi in enumerate(key_bands):
        ax = axes[ai]
        for lbl, name in enumerate(CLASS_NAMES):
            vals = X_masked[y == lbl, :, bi]
            mean = np.nanmean(vals, axis=0)
            std  = np.nanstd(vals, axis=0)
            ax.plot(range(36), mean, color=CLASS_COLORS[lbl], label=name, lw=2)
            ax.fill_between(range(36), mean-std, mean+std,
                            color=CLASS_COLORS[lbl], alpha=0.12)
        ax.set_title(f'{S2_BANDS[bi]} — profil temporel', fontsize=11, fontweight='bold')
        ax.set_xticks(range(0, 36, 3))
        ax.set_xticklabels(TIMESTEP_LABELS[::3], rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Réflectance (norm.)', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    plt.suptitle('Profils temporels S2 — California 2021 (nuages masqués)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_02_s2_temporal_profiles.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 3 : Toutes les 10 bandes S2 ───────────────────────────────────────
def plot_s2_all_bands(X_s2, mask, y):
    X_masked = X_s2.copy()
    for b in range(10):
        X_masked[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
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
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    plt.suptitle('Profils temporels — 10 bandes S2 — California 2021', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_03_s2_all_bands.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 4 : NDVI temporel par classe ──────────────────────────────────────
def plot_ndvi(X_s2, mask, y):
    nir  = np.where(mask == 0, X_s2[:, :, 6], np.nan)
    red  = np.where(mask == 0, X_s2[:, :, 2], np.nan)
    ndvi = (nir - red) / (nir + red + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 6))
    for lbl, name in enumerate(CLASS_NAMES):
        v = ndvi[y == lbl]
        m = np.nanmean(v, axis=0)
        s = np.nanstd(v, axis=0)
        ax.plot(range(36), m, color=CLASS_COLORS[lbl], label=name, lw=2.2)
        ax.fill_between(range(36), m-s, m+s, color=CLASS_COLORS[lbl], alpha=0.12)

    ax.set_xticks(range(0, 36, 2))
    ax.set_xticklabels(TIMESTEP_LABELS[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('NDVI (moyenne ± std)', fontsize=11)
    ax.set_title('Profil NDVI temporel par classe — California 2021\n'
                 '(nuages masqués) — 6 cultures', fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_04_ndvi_temporal.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 5 : Covariables Sol par classe ─────────────────────────────────────
def plot_soil_by_class(cov_soil, y):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for fi, (ax, name) in enumerate(zip(axes, SOL_NAMES)):
        for lbl, cname in enumerate(CLASS_NAMES):
            vals = cov_soil[y == lbl, fi]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=30, alpha=0.55, density=True,
                    color=CLASS_COLORS[lbl], label=cname)
        ax.set_title(name, fontsize=11, fontweight='bold', color='#8c564b')
        ax.set_ylabel('Densité', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        groups = [cov_soil[y==lbl, fi][np.isfinite(cov_soil[y==lbl, fi])]
                  for lbl in range(N_CLASSES)]
        try:
            _, p = sp_stats.f_oneway(*groups)
            sig  = '***' if p < 0.001 else ('**' if p < 0.01 else
                   ('*' if p < 0.05 else 'ns'))
            ax.text(0.98, 0.97, f'ANOVA {sig}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, color='navy')
        except Exception:
            pass

    plt.suptitle('Covariables Sol par classe — California 2021\n'
                 '(sélectionnées comme meilleure config ablation)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_05_soil_by_class.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 6 : Corrélation — S2(moy) + Sol ───────────────────────────────────
def plot_correlation_matrix(X_fus):
    s2_mean = np.nanmean(X_fus[:, :, :10], axis=1)   # (N, 10)
    cov     = X_fus[:, 0, 10:]                         # (N, 3)
    X_flat  = np.concatenate([s2_mean, cov], axis=1)   # (N, 13)

    valid = np.all(np.isfinite(X_flat), axis=1)
    corr  = np.corrcoef(X_flat[valid].T)

    labels = [f'S2_{b.split()[0]}' for b in S2_BANDS] + \
             ['Sol_pH', 'Sol_OC', 'Sol_Tex']

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 7.5},
                linewidths=0.4, square=True)
    # Séparateur S2 / Sol
    ax.axhline(10, color='black', linewidth=2)
    ax.axvline(10, color='black', linewidth=2)
    ax.set_title('Corrélation — 10 bandes S2 (moy. temporelle) + 3 covariables Sol\n'
                 'California 2021', fontsize=12, pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_06_correlation_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 7 : PCA sur l'espace fusionné (13 features) ───────────────────────
def plot_pca_fused(X_fus, y):
    s2_mean = np.nanmean(X_fus[:, :, :10], axis=1)
    cov     = X_fus[:, 0, 10:]
    X_flat  = np.concatenate([s2_mean, cov], axis=1)   # (N, 13)

    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(y), min(3000, len(y)), replace=False)
    sub   = X_flat[idx]; sub_y = y[idx]
    valid = np.all(np.isfinite(sub), axis=1)
    sub   = sub[valid]; sub_y = sub_y[valid]

    mu    = sub.mean(0); std = sub.std(0) + 1e-8
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
    ax.set_title(f'PCA — 13 features fusionnées (S2+Sol)\n'
                 f'n={valid.sum()} pts — California 2021 — 6 classes', fontsize=12)
    ax.legend(fontsize=10, markerscale=3, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_07_pca_fused.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Figure 8 : Qualité des données S2 ────────────────────────────────────────
def plot_data_stats(X_s2, mask, y):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Couverture nuageuse par timestep
    cloud_pct = mask.mean(axis=0) * 100
    axes[0].bar(range(36), cloud_pct, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(0, 36, 3))
    axes[0].set_xticklabels(TIMESTEP_LABELS[::3], rotation=40, ha='right', fontsize=7)
    axes[0].set_ylabel('% pixels masqués', fontsize=10)
    axes[0].set_title('Couverture nuageuse par timestep — California', fontsize=11)
    axes[0].grid(alpha=0.3)

    # Nuages par point
    cloud_per_pt = mask.sum(axis=1)
    axes[1].hist(cloud_per_pt, bins=37, color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].set_xlabel('Nombre de timesteps masqués', fontsize=10)
    axes[1].set_ylabel('Nombre de points', fontsize=10)
    axes[1].set_title('Distribution des masques par point', fontsize=11)
    axes[1].axvline(cloud_per_pt.mean(), color='red', linestyle='--',
                    label=f'Moy.={cloud_per_pt.mean():.1f}')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Stats réflectance par bande
    X_clean = X_s2.copy()
    for b in range(10):
        X_clean[:, :, b] = np.where(mask == 0, X_s2[:, :, b], np.nan)
    band_means = np.nanmean(X_clean, axis=(0, 1))
    band_stds  = np.nanstd(X_clean,  axis=(0, 1))
    x_pos = np.arange(10)
    axes[2].bar(x_pos, band_means, yerr=band_stds, color='forestgreen',
                alpha=0.8, capsize=4, edgecolor='white')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([b.split()[0] for b in S2_BANDS],
                             rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('Réflectance normalisée (÷10000)', fontsize=10)
    axes[2].set_title('Moyenne ± std — 10 bandes S2', fontsize=11)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Qualité et statistiques des données — California 2021', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'eda_08_data_stats.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ {out}")


# ── Résumé console ────────────────────────────────────────────────────────────
def print_summary(X_s2, X_fus, mask, y, train_idx, test_idx):
    n = len(y)
    val_idx = np.setdiff1d(np.arange(n), np.concatenate([train_idx, test_idx]))
    print("\n" + "="*60)
    print("RÉSUMÉ EDA — Part 3 — California 2021")
    print("="*60)
    print(f"\n  Données S2          : {X_s2.shape}  (N, T=36, B=10)")
    print(f"  Données fusionnées  : {X_fus.shape}  (N, T=36, F=13)")
    print(f"  Masque nuageux      : {mask.shape} | moy. {mask.mean()*100:.1f}% masqués")
    print(f"\n  Splits :")
    print(f"    Train : {len(train_idx):,}  ({len(train_idx)/n*100:.0f}%)")
    print(f"    Val   : {len(val_idx):,}   ({len(val_idx)/n*100:.0f}%)")
    print(f"    Test  : {len(test_idx):,}  ({len(test_idx)/n*100:.0f}%)")
    print(f"\n  Distribution des 6 classes :")
    for i, name in enumerate(CLASS_NAMES):
        cnt = (y == i).sum()
        print(f"    {name:12s} : {cnt:5,}  ({cnt/n*100:.1f}%)")
    print(f"\n  Features fusionnées (13) :")
    print(f"    S2   (10) : 10 bandes (temporelles, T=36)")
    print(f"    Sol  (3)  : pH, OC, Texture  [répétées sur T=36]")
    print(f"\n  Justification S2+Sol (ablation Part 2) :")
    print(f"    OA=0.848 | Kappa=0.802 | F1=0.831")
    print(f"    → F1 dépasse le papier (0.829) !")
    print(f"    → S2+Tout (0.824) et S2+Topo (0.830) moins bons")
    print(f"\n  → Lancer : python CAL_Part3_Step2_preprocessing.py")
    print("="*60)


def main():
    print("="*60)
    print("Part 3 — Étape 1 : EDA données fusionnées — California")
    print("="*60 + "\n")

    X_s2, X_fus, mask, y, cov_soil, train_idx, test_idx = load_data()

    print("\n── Génération des 8 figures ────────────────────────────")
    plot_class_distribution(y)
    plot_s2_temporal_profiles(X_s2, mask, y)
    plot_s2_all_bands(X_s2, mask, y)
    plot_ndvi(X_s2, mask, y)
    plot_soil_by_class(cov_soil, y)
    plot_correlation_matrix(X_fus)
    plot_pca_fused(X_fus, y)
    plot_data_stats(X_s2, mask, y)

    print_summary(X_s2, X_fus, mask, y, train_idx, test_idx)
    print(f"\n✅ 8 figures sauvegardées dans : {FIG_DIR}")


if __name__ == "__main__":
    main()
