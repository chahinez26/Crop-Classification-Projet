"""
PART 2 — ÉTAPE 2 : Analyse exploratoire des covariables (EDA)
==============================================================
Entrée  : ARK_covariates.npz
Sorties : figures/part2/
  fig_cov_distributions.png  — histogrammes par classe
  fig_cov_boxplots.png        — boxplots par classe
  fig_cov_correlation.png     — matrice de corrélation
  fig_cov_pca.png             — PCA 2D colorée par classe
  fig_cov_importance.png      — importance Random Forest
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy import stats as sp_stats

# ── Configuration ──────────────────────────────────────────────────────────
COV_FILE = r"data\merged_npz\arkansas\ARK_covariates.npz"
FIG_DIR  = r"outputs\figures\arkansas\part2"

os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES  = ['Corn', 'Cotton', 'Rice', 'Soybean', 'Others']
CLASS_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']

# Noms et labels d'affichage pour les 8 features
FEATURE_INFO = {
    'temp_mean'    : ('Température moy.\n(°C)',        'Climat'),
    'precip_total' : ('Précipitations\ntotales (mm)', 'Climat'),
    'solar_mean'   : ('Rayonnement\nsolaire (W/m²)',  'Climat'),
    'soil_ph'      : ('pH Sol',                        'Sol'),
    'soil_oc'      : ('Carbone\norganique (g/kg)',     'Sol'),
    'soil_texture' : ('Texture Sol\n(USDA 1-12)',      'Sol'),
    'elevation'    : ('Élévation\n(m)',                'Topo'),
    'landforms'    : ('Landforms\n(Weiss 21-42)',      'Topo'),
}
GROUP_COLORS = {'Climat': '#1f77b4', 'Sol': '#8c564b', 'Topo': '#7f7f7f'}
# ───────────────────────────────────────────────────────────────────────────


def load_data():
    data = np.load(COV_FILE, allow_pickle=True)
    cov_clim_raw = data['cov_clim_raw']
    cov_soil_raw = data['cov_soil_raw']
    cov_topo_raw = data['cov_topo_raw']
    cov_all_norm = data['cov_all']
    y            = data['y_all']
    train_idx    = data['train_idx']

    all_raw = np.concatenate([cov_clim_raw, cov_soil_raw, cov_topo_raw], axis=1)
    all_features = list(FEATURE_INFO.keys())

    print(f"Chargé : {len(y)} points  |  {all_raw.shape[1]} covariables")
    return all_raw, cov_all_norm, y, train_idx, all_features


# ── Figure 1 : Distributions par classe ──────────────────────────────────
def plot_distributions(raw, y, features):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    for f, feat in enumerate(features):
        ax = axes[f]
        for lbl, name in enumerate(CLASS_NAMES):
            vals = raw[y == lbl, f]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=35, alpha=0.55,
                    color=CLASS_COLORS[lbl], label=name, density=True)
        label, group = FEATURE_INFO[feat]
        ax.set_title(label, fontsize=10, fontweight='bold',
                     color=GROUP_COLORS[group])
        ax.set_ylabel('Densité', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    # Légende groupes
    gpatches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
    fig.legend(handles=gpatches, loc='upper right', fontsize=9,
               title='Groupe', bbox_to_anchor=(1.0, 1.0))

    plt.suptitle('Distributions des covariables par classe — Arkansas',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 2 : Boxplots par classe ───────────────────────────────────────
def plot_boxplots(raw, y, features):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    for f, feat in enumerate(features):
        ax = axes[f]
        data_by_class = [raw[y == lbl, f][np.isfinite(raw[y == lbl, f])]
                         for lbl in range(5)]

        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.8},
                        flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3},
                        whiskerprops={'linewidth': 1.2},
                        boxprops={'linewidth': 1.2})
        for patch, color in zip(bp['boxes'], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels([n[:4] for n in CLASS_NAMES], fontsize=8)
        label, group = FEATURE_INFO[feat]
        ax.set_title(label, fontsize=10, fontweight='bold',
                     color=GROUP_COLORS[group])
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', alpha=0.3)

        # ANOVA p-value
        try:
            f_stat, p_val = sp_stats.f_oneway(*data_by_class)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
                  ('*' if p_val < 0.05 else 'ns'))
            ax.text(0.98, 0.97, f'ANOVA {sig}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.5, color='navy')
        except:
            pass

    plt.suptitle('Boxplots des covariables par classe — Arkansas',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_boxplots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 3 : Matrice de corrélation ────────────────────────────────────
def plot_correlation(raw, features):
    # Inclure NDVI moyen (chargé depuis X_all si disponible)
    labels_display = [FEATURE_INFO[f][0].replace('\n', ' ') for f in features]

    valid = np.all(np.isfinite(raw), axis=1)
    corr = np.corrcoef(raw[valid].T)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax,
                xticklabels=labels_display,
                yticklabels=labels_display,
                cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                linewidths=0.5, square=True)
    ax.set_title('Matrice de corrélation — 8 covariables', fontsize=12, pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_correlation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 4 : PCA 2D colorée par classe ────────────────────────────────
def plot_pca(cov_norm, y):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), min(3000, len(y)), replace=False)
    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(cov_norm[idx])
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in enumerate(CLASS_NAMES):
        m = (y[idx] == lbl)
        ax.scatter(Xp[m, 0], Xp[m, 1],
                   c=CLASS_COLORS[lbl], label=name,
                   alpha=0.45, s=12, edgecolors='none')

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title('PCA — 8 covariables normalisées (n=3000 pts)', fontsize=12)
    ax.legend(fontsize=9, markerscale=2.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_pca.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 5 : Importance Random Forest ──────────────────────────────────
def plot_importance(cov_norm, y, train_idx, features):
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(cov_norm[train_idx], y[train_idx])

    imp    = rf.feature_importances_
    labels = [FEATURE_INFO[f][0].replace('\n', ' ') for f in features]
    colors = [GROUP_COLORS[FEATURE_INFO[f][1]] for f in features]
    order  = np.argsort(imp)[::-1]

    # Score test
    data = np.load(COV_FILE, allow_pickle=True)
    test_idx = data['test_idx']
    score_rf = rf.score(cov_norm[test_idx], y[test_idx])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(features)), imp[order],
           color=[colors[i] for i in order], alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([labels[i] for i in order], rotation=40,
                       ha='right', fontsize=9)
    ax.set_ylabel('Importance (Gini)', fontsize=11)
    ax.set_title('Importance des covariables — Random Forest (train set)',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    gp = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
    ax.legend(handles=gp, fontsize=9)
    ax.text(0.99, 0.97,
            f'RF OA (test, cov. seules) : {score_rf:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_importance.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")
    print(f"   RF OA avec covariables seules : {score_rf:.4f}")

    # Résumé importance
    print(f"\n   Classement importance :")
    for rank, i in enumerate(order):
        print(f"     {rank+1}. {labels[i]:30s} {imp[i]:.4f}  "
              f"({FEATURE_INFO[features[i]][1]})")


def print_stats_table(raw, y, features):
    print("\n── Statistiques par classe ────────────────────────────────────────")
    print(f"  {'Feature':15s} {'Corn':>7s} {'Cotton':>7s} {'Rice':>7s} "
          f"{'Soybean':>8s} {'Others':>7s} {'ANOVA':>8s}")
    print(f"  {'-'*65}")
    for j, feat in enumerate(features):
        vals_cls = [raw[y == lbl, j] for lbl in range(5)]
        means    = [v[np.isfinite(v)].mean() if len(v) > 0 else 0 for v in vals_cls]
        try:
            _, p = sp_stats.f_oneway(*[v[np.isfinite(v)] for v in vals_cls])
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*'  if p < 0.05 else 'ns'))
        except:
            sig = '?'
        print(f"  {feat:15s} " +
              " ".join(f"{m:7.2f}" for m in means) + f"  {sig:>8s}")


def main():
    print("=" * 60)
    print("Part 2 — Étape 2 : EDA covariables")
    print("=" * 60 + "\n")

    raw, norm, y, train_idx, features = load_data()

    print_stats_table(raw, y, features)

    print("\n── Génération figures ─────────────────────────────────────")
    plot_distributions(raw, y, features)
    plot_boxplots(raw, y, features)
    plot_correlation(raw, features)
    plot_pca(norm, y)
    plot_importance(norm, y, train_idx, features)

    print(f"\n✅ Figures dans : {FIG_DIR}")
    print("   → Lancer : python Part2_Step3_mctnetcov.py")


if __name__ == "__main__":
    main()
