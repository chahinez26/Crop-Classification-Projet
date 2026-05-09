
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


ROOT_DIR = r"C:\Users\chahi\Desktop\CNN_Transformer_Project\crop-classification"

REGION_CFG = {
    "ark": {
        "label"       : "Arkansas",
        "data_file"   : os.path.join(ROOT_DIR, "data", "merged_npz","arkansas", "Part3_ARK_dataset.npz"),
        "fig_dir"     : os.path.join(ROOT_DIR, "outputs", "figures", "eda_part3", "arkansas"),
        "class_names" : ["Corn", "Cotton", "Rice", "Soybean", "Others"],
        "class_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    },
    "cal": {
        "label"       : "California",
        "data_file"   : os.path.join(ROOT_DIR, "data", "merged_npz","california", "Part3_CAL_dataset.npz"),
        "fig_dir"     : os.path.join(ROOT_DIR, "outputs", "figures", "eda_part3", "california"),
        "class_names" : ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"],
        "class_colors": ["#e377c2", "#2ca02c", "#bcbd22", "#8c564b", "#17becf", "#7f7f7f"],
    },
}

BAND_NAMES  = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDVI", "EVI", "NDWI", "GNDVI", "NDRE"]
FEATURE_NAMES = BAND_NAMES + INDEX_NAMES
N_TIMESTEPS = 36
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def savefig(fig, path, tight=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def style_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if grid:
        ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_class_data(X_all, y_all, mask_all, class_idx):
    idx = y_all == class_idx
    return X_all[idx], mask_all[idx]


def fig_class_distribution(data, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    n_classes   = len(class_names)
    splits      = ["train", "val", "test"]
    split_colors = ["#4C72B0", "#DD8452", "#55A868"]

    counts = {}
    for s in splits:
        y = data[f"y_{s}"]
        counts[s] = [int((y == c).sum()) for c in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{cfg['label']} — Distribution des classes", fontsize=13, fontweight="bold")


    x    = np.arange(n_classes)
    w    = 0.26
    ax   = axes[0]
    for i, (s, sc) in enumerate(zip(splits, split_colors)):
        bars = ax.bar(x + (i - 1) * w, counts[s], width=w, label=SPLIT_LABELS[s],
                      color=sc, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, counts[s]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        str(val), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.legend(fontsize=8)
    style_ax(ax, "Nombre d'échantillons par split", ylabel="Nombre d'échantillons")


    ax2   = axes[1]
    total = np.array(counts["test"])
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    ax2.pie(total, labels=class_names, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops=wedge_props, textprops={"fontsize": 9})
    ax2.set_title("Répartition sur le jeu de test", fontsize=11, fontweight="bold")

    return fig


def fig_missing_data(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    n_classes   = len(class_names)
    ts          = np.arange(1, N_TIMESTEPS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{cfg['label']} — Analyse des données manquantes (masque)", fontsize=13, fontweight="bold")


    missing_global = mask_all.mean(axis=0) * 100  
    axes[0].bar(ts, missing_global, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].set_xticks(ts[::4])
    style_ax(axes[0], "Taux de données manquantes global par timestep",
             xlabel="Timestep (intervalle de 10 jours)", ylabel="% manquant")


    for c, (cname, col) in enumerate(zip(class_names, colors)):
        mask_c = mask_all[y_all == c].mean(axis=0) * 100
        axes[1].plot(ts, mask_c, label=cname, color=col, linewidth=1.8, marker="o",
                     markersize=3, alpha=0.85)
    axes[1].set_xticks(ts[::4])
    axes[1].legend(fontsize=8, ncol=2)
    style_ax(axes[1], "Taux de données manquantes par classe",
             xlabel="Timestep", ylabel="% manquant")

    return fig


def fig_temporal_profiles_bands(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    ts          = np.arange(1, N_TIMESTEPS + 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"{cfg['label']} — Profils temporels moyens : Bandes spectrales",
                 fontsize=13, fontweight="bold")

    for b, (bname, ax) in enumerate(zip(BAND_NAMES, axes.flatten())):
        for c, (cname, col) in enumerate(zip(class_names, colors)):
            idx   = y_all == c
            x_c   = X_all[idx, :, b]
            m_c   = mask_all[idx]
            valid = m_c == 0
            mean_ts = np.array([x_c[:, t][valid[:, t]].mean() if valid[:, t].any() else np.nan
                                 for t in range(N_TIMESTEPS)])
            std_ts  = np.array([x_c[:, t][valid[:, t]].std() if valid[:, t].any() else np.nan
                                 for t in range(N_TIMESTEPS)])
            ax.plot(ts, mean_ts, color=col, label=cname, linewidth=1.6)
            ax.fill_between(ts, mean_ts - std_ts, mean_ts + std_ts, alpha=0.12, color=col)
        ax.set_xticks(ts[::8])
        style_ax(ax, bname, xlabel="Timestep", ylabel="Réfl. norm.")

    handles = [mpatches.Patch(color=col, label=cn)
               for col, cn in zip(colors, class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(class_names),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    return fig


def fig_temporal_profiles_indices(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    ts          = np.arange(1, N_TIMESTEPS + 1)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(f"{cfg['label']} — Profils temporels moyens : Indices de végétation",
                 fontsize=13, fontweight="bold")

    for i, (iname, ax) in enumerate(zip(INDEX_NAMES, axes)):
        feat_idx = 10 + i
        for c, (cname, col) in enumerate(zip(class_names, colors)):
            idx   = y_all == c
            x_c   = X_all[idx, :, feat_idx]
            m_c   = mask_all[idx]
            valid = m_c == 0
            mean_ts = np.array([x_c[:, t][valid[:, t]].mean() if valid[:, t].any() else np.nan
                                 for t in range(N_TIMESTEPS)])
            std_ts  = np.array([x_c[:, t][valid[:, t]].std() if valid[:, t].any() else np.nan
                                 for t in range(N_TIMESTEPS)])
            ax.plot(ts, mean_ts, color=col, label=cname, linewidth=1.8)
            ax.fill_between(ts, mean_ts - std_ts, mean_ts + std_ts, alpha=0.13, color=col)
        ax.set_xticks(ts[::8])
        style_ax(ax, iname, xlabel="Timestep", ylabel="Valeur norm.")

    handles = [mpatches.Patch(color=col, label=cn)
               for col, cn in zip(colors, class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(class_names),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    return fig


def fig_ndvi_zoom(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    ts          = np.arange(1, N_TIMESTEPS + 1)
    months      = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Aoû","Sep","Oct","Nov","Déc"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{cfg['label']} — Calendrier phénologique NDVI par classe",
                 fontsize=13, fontweight="bold")

    ndvi_idx = 10  
    for c, (cname, col) in enumerate(zip(class_names, colors)):
        idx   = y_all == c
        x_c   = X_all[idx, :, ndvi_idx]
        m_c   = mask_all[idx]
        valid = m_c == 0
        mean_ts = np.array([x_c[:, t][valid[:, t]].mean() if valid[:, t].any() else np.nan
                             for t in range(N_TIMESTEPS)])
        std_ts  = np.array([x_c[:, t][valid[:, t]].std() if valid[:, t].any() else np.nan
                             for t in range(N_TIMESTEPS)])
        ax.plot(ts, mean_ts, color=col, label=cname, linewidth=2.2, marker="o", markersize=3)
        ax.fill_between(ts, mean_ts - std_ts, mean_ts + std_ts, alpha=0.10, color=col)


    month_ticks = np.arange(2, 37, 3)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(months, fontsize=8)
    ax.set_xlim(1, N_TIMESTEPS)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=9, ncol=2)
    style_ax(ax, ylabel="NDVI normalisé")

    return fig


def fig_boxplots_indices(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    n_classes   = len(class_names)

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    fig.suptitle(f"{cfg['label']} — Boxplots des indices de végétation par classe",
                 fontsize=13, fontweight="bold")

    for i, (iname, ax) in enumerate(zip(INDEX_NAMES, axes)):
        feat_idx = 10 + i
        data_per_class = []
        for c in range(n_classes):
            idx   = y_all == c
            x_c   = X_all[idx, :, feat_idx]
            m_c   = mask_all[idx]
            vals  = x_c[m_c == 0]  
            data_per_class.append(vals)

        bplot = ax.boxplot(data_per_class, patch_artist=True,
                           medianprops={"color": "black", "linewidth": 2},
                           whiskerprops={"linewidth": 1.2},
                           flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
        for patch, col in zip(bplot["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        style_ax(ax, iname, ylabel="Valeur norm.")

    return fig


def fig_kde_indices(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]
    n_classes   = len(class_names)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"{cfg['label']} — Distributions KDE des indices de végétation par classe",
                 fontsize=13, fontweight="bold")

    for i, (iname, ax) in enumerate(zip(INDEX_NAMES, axes)):
        feat_idx = 10 + i
        for c, (cname, col) in enumerate(zip(class_names, colors)):
            idx  = y_all == c
            vals = X_all[idx, :, feat_idx][mask_all[idx] == 0]
            if len(vals) < 10:
                continue

            xgrid = np.linspace(vals.min(), vals.max(), 300)
            try:
                kde = gaussian_kde(vals, bw_method=0.15)
                ax.fill_between(xgrid, kde(xgrid), alpha=0.18, color=col)
                ax.plot(xgrid, kde(xgrid), color=col, label=cname, linewidth=1.8)
            except Exception:
                pass
        style_ax(ax, iname, xlabel="Valeur norm.", ylabel="Densité")

    handles = [mpatches.Patch(color=col, label=cn)
               for col, cn in zip(colors, class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(class_names),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    return fig


def fig_correlation_matrix(X_all, mask_all, cfg):


    N, T, F = X_all.shape
    means = np.full((N, F), np.nan)
    for n in range(N):
        for f in range(F):
            valid = mask_all[n] == 0
            if valid.any():
                means[n, f] = X_all[n, valid, f].mean()

    corr = np.corrcoef(means.T)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle(f"{cfg['label']} — Matrice de corrélation entre les 15 features",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Corrélation de Pearson")

    ax.set_xticks(range(15))
    ax.set_yticks(range(15))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)

    for i in range(15):
        for j in range(15):
            val = corr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(val) > 0.6 else "black")


    for pos in [9.5]:
        ax.axhline(pos, color="black", linewidth=1.5)
        ax.axvline(pos, color="black", linewidth=1.5)

    ax.text(4.5, -1.2, "Bandes spectrales", ha="center", fontsize=9, fontweight="bold")
    ax.text(12.0, -1.2, "Indices", ha="center", fontsize=9, fontweight="bold")

    return fig


def fig_pca_2d(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    colors      = cfg["class_colors"]


    N, T, F = X_all.shape
    feats = np.zeros((N, F))
    for n in range(N):
        valid = mask_all[n] == 0
        feats[n] = X_all[n, valid, :].mean(axis=0) if valid.any() else X_all[n].mean(axis=0)

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(feats_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{cfg['label']} — ACP 2D sur les 15 features (moyenne temporelle)",
                 fontsize=13, fontweight="bold")


    for c, (cname, col) in enumerate(zip(class_names, colors)):
        idx = y_all == c
        axes[0].scatter(pc[idx, 0], pc[idx, 1], c=col, label=cname,
                        s=8, alpha=0.35, edgecolors="none")
    axes[0].legend(fontsize=8, markerscale=3)
    style_ax(axes[0],
             f"PC1 vs PC2 — tous splits",
             f"PC1 ({var_exp[0]:.1f}%)",
             f"PC2 ({var_exp[1]:.1f}%)")


    for c, (cname, col) in enumerate(zip(class_names, colors)):
        idx  = y_all == c
        x_c, y_c = pc[idx, 0], pc[idx, 1]
        mu_x, mu_y = x_c.mean(), y_c.mean()
        sx, sy = x_c.std() * 1.5, y_c.std() * 1.5
        theta = np.linspace(0, 2 * np.pi, 100)
        axes[1].plot(mu_x + sx * np.cos(theta), mu_y + sy * np.sin(theta),
                     color=col, linewidth=2, label=cname)
        axes[1].scatter(mu_x, mu_y, color=col, s=80, zorder=5, marker="X")
    axes[1].legend(fontsize=8)
    style_ax(axes[1], "Ellipses de dispersion (±1.5σ) par classe",
             f"PC1 ({var_exp[0]:.1f}%)", f"PC2 ({var_exp[1]:.1f}%)")

    return fig


def fig_temporal_variance_heatmap(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    n_classes   = len(class_names)


    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6), sharey=True)
    if n_classes == 1:
        axes = [axes]
    fig.suptitle(f"{cfg['label']} — Variance temporelle par feature et par classe",
                 fontsize=13, fontweight="bold")

    vmin, vmax = 0.0, None
    heat_data = []
    for c in range(n_classes):
        idx = y_all == c
        x_c = X_all[idx]
        m_c = mask_all[idx]
        var_ft = np.zeros((15, N_TIMESTEPS))
        for f in range(15):
            for t in range(N_TIMESTEPS):
                vals = x_c[:, t, f][m_c[:, t] == 0]
                var_ft[f, t] = vals.var() if len(vals) > 1 else 0.0
        heat_data.append(var_ft)
        if vmax is None:
            vmax = var_ft.max()
        else:
            vmax = max(vmax, var_ft.max())

    for c, (cname, hd) in enumerate(zip(class_names, heat_data)):
        im = axes[c].imshow(hd, cmap="YlOrRd", aspect="auto",
                            vmin=vmin, vmax=vmax,
                            extent=[1, N_TIMESTEPS, 14.5, -0.5])
        axes[c].set_title(cname, fontsize=10, fontweight="bold")
        axes[c].set_xlabel("Timestep", fontsize=8)
        if c == 0:
            axes[c].set_yticks(range(15))
            axes[c].set_yticklabels(FEATURE_NAMES, fontsize=8)

        axes[c].axhline(9.5, color="white", linewidth=1.5, linestyle="--")

    plt.colorbar(im, ax=axes[-1], label="Variance")
    return fig


def fig_stats_table(X_all, y_all, mask_all, cfg):
    class_names = cfg["class_names"]
    n_classes   = len(class_names)

    rows = []
    for f, fname in enumerate(FEATURE_NAMES):
        vals_all = X_all[:, :, f][mask_all == 0]
        row = [fname, f"{vals_all.mean():.4f}", f"{vals_all.std():.4f}",
               f"{vals_all.min():.4f}", f"{vals_all.max():.4f}"]
        for c in range(n_classes):
            idx  = y_all == c
            vals = X_all[idx, :, f][mask_all[idx] == 0]
            row.append(f"{vals.mean():.3f}")
        rows.append(row)

    col_labels = ["Feature", "Mean", "Std", "Min", "Max"] + [f"μ {cn}" for cn in class_names]
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.4), 8))
    ax.axis("off")
    fig.suptitle(f"{cfg['label']} — Statistiques descriptives par feature",
                 fontsize=12, fontweight="bold")

    col_widths = [0.09] + [0.07] * 4 + [0.075] * n_classes
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)


    for j in range(n_cols):
        table[(0, j)].set_facecolor("#2E75B6")
        table[(0, j)].set_text_props(color="white", fontweight="bold")


    for i in range(1, 16):
        bg = "#EBF5FB" if i % 2 == 0 else "white"
        if i > 10:  
            bg = "#FFF3CD" if i % 2 == 0 else "#FFEAA7"
        for j in range(n_cols):
            table[(i, j)].set_facecolor(bg)

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, choices=["ark", "cal"])
    args = parser.parse_args()

    if args.region is None:
        args.region = input("Choisir région (ark/cal) : ").strip().lower()

    return args


def main():
    args = parse_args()
    cfg  = REGION_CFG[args.region]

    print("=" * 65)
    print(f"Step 0 — EDA Part 3 : {cfg['label']}")
    print("=" * 65)


    print(f"\nChargement des données : {cfg['data_file']}")
    data     = np.load(cfg["data_file"])
    X_all    = data["X_all"]
    y_all    = data["y_all"]
    mask_all = data["mask_all"].astype(np.uint8)

    print(f"  X_all    : {X_all.shape}")
    print(f"  y_all    : {y_all.shape} — classes {np.unique(y_all).tolist()}")
    print(f"  mask_all : {mask_all.shape} — missing {mask_all.mean()*100:.1f}%")

    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\nFigures enregistrées dans : {fig_dir}\n")


    print("[1/11] Distribution des classes...")
    fig = fig_class_distribution(data, cfg)
    savefig(fig, os.path.join(fig_dir, "fig01_class_distribution.png"))


    print("[2/11] Données manquantes...")
    fig = fig_missing_data(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig02_missing_data.png"))


    print("[3/11] Profils temporels — bandes spectrales...")
    fig = fig_temporal_profiles_bands(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig03_temporal_bands.png"))


    print("[4/11] Profils temporels — indices de végétation...")
    fig = fig_temporal_profiles_indices(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig04_temporal_indices.png"))


    print("[5/11] Calendrier phénologique NDVI...")
    fig = fig_ndvi_zoom(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig05_ndvi_phenology.png"))


    print("[6/11] Boxplots indices de végétation par classe...")
    fig = fig_boxplots_indices(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig06_boxplots_indices.png"))


    print("[7/11] Distributions KDE des indices...")
    fig = fig_kde_indices(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig07_kde_indices.png"))


    print("[8/11] Matrice de corrélation...")
    fig = fig_correlation_matrix(X_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig08_correlation_matrix.png"))


    print("[9/11] ACP 2D...")
    fig = fig_pca_2d(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig09_pca_2d.png"))


    print("[10/11] Heatmap variance temporelle...")
    fig = fig_temporal_variance_heatmap(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig10_temporal_variance.png"))


    print("[11/11] Tableau des statistiques descriptives...")
    fig = fig_stats_table(X_all, y_all, mask_all, cfg)
    savefig(fig, os.path.join(fig_dir, "fig11_stats_table.png"))

    print("\n" + "=" * 65)
    print(f"EDA terminée — {11} figures sauvegardées dans :")
    print(f"  {fig_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
