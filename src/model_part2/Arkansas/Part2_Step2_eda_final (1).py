"""
PART 2 — ÉTAPE 2 : Analyse exploratoire des covariables (EDA)
==============================================================
Entrée  : ARK_covariates_v2.npz
Sorties : outputs/figures/arkansas/part2/
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
COV_FILE = r"data\merged_npz\arkansas\ARK_covariates_v2.npz"
FIG_DIR  = r"outputs\figures\arkansas\part2"

os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES  = ['Corn', 'Cotton', 'Rice', 'Soybean', 'Others']
N_CLASSES    = len(CLASS_NAMES)
CLASS_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']

CLIM_COLS   = ['temp_mean', 'precip_total', 'solar_mean']
SOIL_COLS   = ['soil_ph', 'soil_oc', 'soil_texture']
TOPO_COLS   = ['elevation', 'landforms']
STATIC_COLS = SOIL_COLS + TOPO_COLS

# ── FEATURE_INFO : clé → (label affiché, groupe) ──────────────────────────
FEATURE_INFO = {
    'temp_mean'    : ('T° moyenne (°C)',     'Climat'),
    'precip_total' : ('Précip. totale (mm)', 'Climat'),
    'solar_mean'   : ('Rayonnement (W/m²)',  'Climat'),
    'soil_ph'      : ('pH Sol',              'Sol'),
    'soil_oc'      : ('OC Sol (g/kg)',       'Sol'),
    'soil_texture' : ('Texture USDA',        'Sol'),
    'elevation'    : ('Élévation (m)',       'Topo'),
    'landforms'    : ('Landforms',           'Topo'),
}

GROUP_COLORS = {'Climat': '#1f77b4', 'Sol': '#8c564b', 'Topo': '#7f7f7f'}

TIMESTEP_LABELS = [
    'Jan-1','Jan-11','Jan-21','Feb-1','Feb-11','Feb-21',
    'Mar-1','Mar-11','Mar-21','Apr-1','Apr-11','Apr-21',
    'May-1','May-11','May-21','Jun-1','Jun-11','Jun-21',
    'Jul-1','Jul-11','Jul-21','Aug-1','Aug-11','Aug-21',
    'Sep-1','Sep-11','Sep-21','Oct-1','Oct-11','Oct-21',
    'Nov-1','Nov-11','Nov-21','Dec-1','Dec-11','Dec-21',
]
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    d = np.load(COV_FILE, allow_pickle=True)

    X_clim_raw = d['X_clim_raw']   # [N, 36, 3]
    mask_clim  = d['mask_clim']    # [N, 36]
    X_soil_raw = d['X_soil_raw']   # [N, 3] brut
    X_topo_raw = d['X_topo_raw']   # [N, 2] brut
    X_soil     = d['X_soil']       # [N, 3] normalisé
    X_topo     = d['X_topo']       # [N, 2] normalisé
    y          = d['y_all']
    train_idx  = d['train_idx']
    test_idx   = d['test_idx']

    # Moyennes climatiques annuelles [N, 3] (en ignorant les masques nuageux)
    clim_raw = np.stack([
        np.where(mask_clim == 0, X_clim_raw[:, :, b], np.nan).mean(axis=1)
        for b in range(3)
    ], axis=1)  # [N, 3]

    static_raw  = np.concatenate([X_soil_raw, X_topo_raw], axis=1)   # [N, 5]
    static_norm = np.concatenate([X_soil,     X_topo],     axis=1)   # [N, 5]

    # Matrice complète [N, 8] : 3 clim + 5 static
    raw  = np.concatenate([clim_raw,  static_raw],  axis=1)
    norm = np.concatenate([clim_raw,  static_norm], axis=1)

    features = CLIM_COLS + STATIC_COLS   # liste ordonnée des 8 features

    print(f"Chargé : N={len(y)}  clim={clim_raw.shape}  static={static_raw.shape}  "
          f"total={raw.shape}")
    return raw, norm, y, train_idx, test_idx, features


# ── Figure 1 : Distributions par classe ──────────────────────────────────────
def plot_distributions(raw, y, features):
    n_feat = len(features)
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
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

    # Cacher les axes vides
    for ax in axes[n_feat:]:
        ax.set_visible(False)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=N_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('Distributions des covariables par classe — Arkansas',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 2 : Boxplots + ANOVA ──────────────────────────────────────────────
def plot_boxplots(raw, y, features):
    n_feat = len(features)
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()

    for f, feat in enumerate(features):
        ax = axes[f]
        data_by_class = [raw[y == lbl, f][np.isfinite(raw[y == lbl, f])]
                         for lbl in range(N_CLASSES)]

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

        try:
            _, p_val = sp_stats.f_oneway(*data_by_class)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
                  ('*'  if p_val < 0.05 else 'ns'))
            ax.text(0.98, 0.97, f'ANOVA {sig}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.5, color='navy')
        except Exception:
            pass

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    plt.suptitle('Boxplots des covariables par classe — Arkansas',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_boxplots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 3 : Matrice de corrélation ────────────────────────────────────────
def plot_correlation(raw, features):
    labels_display = [FEATURE_INFO[f][0] for f in features]

    valid = np.all(np.isfinite(raw), axis=1)
    corr  = np.corrcoef(raw[valid].T)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax,
                xticklabels=labels_display,
                yticklabels=labels_display,
                cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                linewidths=0.5, square=True)
    ax.set_title('Matrice de corrélation — 8 covariables — Arkansas',
                 fontsize=12, pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_correlation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 4 : PCA 2D colorée par classe ─────────────────────────────────────
def plot_pca(norm, y):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), min(3000, len(y)), replace=False)

    # Ignorer les lignes avec NaN avant la PCA
    sub_norm = norm[idx]
    valid    = np.all(np.isfinite(sub_norm), axis=1)
    sub_norm = sub_norm[valid]
    sub_y    = y[idx][valid]

    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(sub_norm)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in enumerate(CLASS_NAMES):
        m = (sub_y == lbl)
        ax.scatter(Xp[m, 0], Xp[m, 1],
                   c=CLASS_COLORS[lbl], label=name,
                   alpha=0.45, s=12, edgecolors='none')

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'PCA — 8 covariables normalisées (n={valid.sum()} pts) — Arkansas',
                 fontsize=12)
    ax.legend(fontsize=9, markerscale=2.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cov_pca.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


# ── Figure 5 : Importance Random Forest ──────────────────────────────────────
def plot_importance(norm, y, train_idx, test_idx, features):
    # Filtrer les NaN pour le RF
    valid_train = np.all(np.isfinite(norm[train_idx]), axis=1)
    valid_test  = np.all(np.isfinite(norm[test_idx]),  axis=1)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(norm[train_idx][valid_train], y[train_idx][valid_train])
    score_rf = rf.score(norm[test_idx][valid_test], y[test_idx][valid_test])

    imp    = rf.feature_importances_
    labels = [FEATURE_INFO[f][0] for f in features]
    colors = [GROUP_COLORS[FEATURE_INFO[f][1]] for f in features]
    order  = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(features)), imp[order],
           color=[colors[i] for i in order], alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([labels[i] for i in order], rotation=40,
                       ha='right', fontsize=9)
    ax.set_ylabel('Importance (Gini)', fontsize=11)
    ax.set_title('Importance des covariables — Random Forest — Arkansas',
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
    print(f"\n   Classement importance :")
    for rank, i in enumerate(order):
        print(f"     {rank+1}. {labels[i]:30s} {imp[i]:.4f}  "
              f"({FEATURE_INFO[features[i]][1]})")


# ── Stats console ─────────────────────────────────────────────────────────────
def print_stats_table(raw, y, features):
    print("\n── Statistiques par classe ────────────────────────────────────────")
    header = f"  {'Feature':15s} " + " ".join(f"{n[:6]:>7s}" for n in CLASS_NAMES)
    print(header + f"  {'ANOVA':>8s}")
    print(f"  {'-'*75}")
    for j, feat in enumerate(features):
        vals_cls = [raw[y == lbl, j] for lbl in range(N_CLASSES)]
        means    = [v[np.isfinite(v)].mean() if len(v[np.isfinite(v)]) > 0 else 0
                    for v in vals_cls]
        try:
            _, p = sp_stats.f_oneway(*[v[np.isfinite(v)] for v in vals_cls])
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*'  if p < 0.05 else 'ns'))
        except Exception:
            sig = '?'
        print(f"  {feat:15s} " +
              " ".join(f"{m:7.2f}" for m in means) + f"  {sig:>8s}")
    print()


def main():
    print("=" * 60)
    print("Part 2 — Étape 2 : EDA covariables — Arkansas")
    print("=" * 60 + "\n")

    if not os.path.exists(COV_FILE):
        print(f"❌ Fichier introuvable : {COV_FILE}")
        print("   → Lancer d'abord Part2_Step1_merge.py")
        return

    raw, norm, y, train_idx, test_idx, features = load_data()

    print_stats_table(raw, y, features)

    print("\n── Génération figures ─────────────────────────────────────")
    plot_distributions(raw, y, features)
    plot_boxplots(raw, y, features)
    plot_correlation(raw, features)
    plot_pca(norm, y)
    plot_importance(norm, y, train_idx, test_idx, features)

    print(f"\n✅ Toutes les figures dans : {FIG_DIR}")
    print("   → Lancer : python Part2_Step3_mctnetcov.py")


if __name__ == "__main__":
    main()