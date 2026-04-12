"""
PART 2 — ÉTAPE 5 : Visualisation des résultats d'ablation — California
=======================================================================
Entrée  : ablation_results.json + ablation_histories.npz
Sorties : figures/part2/california/ — 4 figures
  fig_ablation_metrics.png   — barplot OA/Kappa/F1 par config
  fig_ablation_delta.png     — gain vs baseline
  fig_ablation_learning.png  — courbes d'apprentissage
  fig_ablation_radar.png     — radar OA/Kappa/F1

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
Métriques papier Table 5 : OA=0.852, Kappa=0.806, F1=0.829
"""

import numpy as np
import matplotlib.pyplot as plt
import json, os

RESULTS_FILE = r"outputs\results\california\part2\results\ablation_results.json"
HIST_FILE    = r"outputs\results\california\part2\results\ablation_histories.npz"
FIG_DIR      = r"outputs\figures\california\part2"
PAPER        = {'OA': 0.852, 'Kappa': 0.806, 'F1': 0.829}
os.makedirs(FIG_DIR, exist_ok=True)


def load():
    with open(RESULTS_FILE) as f:
        res = json.load(f)
    hist = np.load(HIST_FILE)
    print(f"Résultats chargés : {len(res)} configurations\n")
    for c, r in res.items():
        print(f"  {r['label']:25s}  OA={r['OA']:.4f}  "
              f"Kappa={r['Kappa']:.4f}  F1={r['F1']:.4f}")
    return res, hist


def plot_metrics(res):
    configs = list(res.keys())
    labels  = [r['label'] for r in res.values()]
    colors  = [r['color'] for r in res.values()]
    metrics, mcols = ['OA', 'Kappa', 'F1'], ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(configs))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, metric, mc in zip(axes, metrics, mcols):
        vals = [res[c][metric] for c in configs]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(PAPER[metric], color='gray', linestyle='--', linewidth=1.5,
                   label=f'Papier ({PAPER[metric]:.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('S2 + ', 'S2+\n') for l in labels], fontsize=9)
        ax.set_title(metric, fontsize=14, fontweight='bold', color=mc)
        ax.set_ylim(max(0, min(vals) - 0.05), 1.02)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study — OA / Kappa / F1 par configuration\n'
                 'California 2021', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ablation_metrics.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(); print(f"✅ {out}")


def plot_delta(res):
    configs = [c for c in res if c != 'C1_S2only']
    labels  = [res[c]['label'].replace('S2 + ', '') for c in configs]
    x, w    = np.arange(len(configs)), 0.22
    metrics, mcols = ['OA', 'Kappa', 'F1'], ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color='black', linewidth=1.5)
    for i, (metric, mc) in enumerate(zip(metrics, mcols)):
        deltas = [(res[c][metric] - res['C1_S2only'][metric]) * 100 for c in configs]
        bars = ax.bar(x + (i - 1) * w, deltas, w, color=mc, alpha=0.8,
                      edgecolor='white', label=metric)
        for bar, d in zip(bars, deltas):
            if abs(d) > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.05 if d >= 0 else -0.25),
                        f'{d:+.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Δ vs S2 only (points %)', fontsize=11)
    ax.set_title('Gain/perte vs baseline S2-only — California', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ablation_delta.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(); print(f"✅ {out}")


def plot_learning(res, hist):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for c, r in res.items():
        try:
            vl = hist[f'{c}_vl']; va = hist[f'{c}_va']
            ep = range(1, len(vl) + 1)
            a1.plot(ep, vl,                  color=r['color'], lw=1.8, label=r['label'])
            a2.plot(ep, [a*100 for a in va], color=r['color'], lw=1.8, label=r['label'])
        except KeyError:
            pass
    for ax, t, y in [(a1,'Validation Loss','Loss'),
                      (a2,'Validation Accuracy','Accuracy (%)')]:
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(y, fontsize=11)
        ax.set_title(t, fontsize=12); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle('Courbes d\'apprentissage — 5 configurations — California',
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ablation_learning.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(); print(f"✅ {out}")


def plot_radar(res):
    cats   = ['OA', 'Kappa', 'F1']
    angles = [n / 3 * 2 * np.pi for n in range(3)] + [0]
    vmin, vmax = 0.70, 1.0

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c, r in res.items():
        vals = [(r[m]-vmin)/(vmax-vmin) for m in cats] + \
               [(r[cats[0]]-vmin)/(vmax-vmin)]
        ax.plot(angles, vals, 'o-', lw=2, color=r['color'], label=r['label'])
        ax.fill(angles, vals, alpha=0.08, color=r['color'])
    pv = [(PAPER[m]-vmin)/(vmax-vmin) for m in cats] + \
         [(PAPER[cats[0]]-vmin)/(vmax-vmin)]
    ax.plot(angles, pv, 's--', lw=1.5, color='gray', label='Papier', alpha=0.7)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=14, fontweight='bold')
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 5)], size=8)
    ax.set_title(f'Radar — [{vmin:.2f}, {vmax:.2f}] — California',
                 fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ablation_radar.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(); print(f"✅ {out}")


def print_summary(res):
    print("\n" + "="*65)
    print("RÉSUMÉ FINAL — Part 2 : Covariables environnementales — California")
    print("="*65)
    print("\n  Covariables (8) :")
    print("    Climat : temp_mean(°C)  precip_total(mm)  solar_mean(W/m²)")
    print("    Sol    : soil_ph(pH)    soil_oc(g/kg)     soil_texture(USDA)")
    print("    Topo   : elevation(m)   landforms(Weiss)")
    base   = res['C1_S2only']['OA']
    best_k = max(res, key=lambda c: res[c]['OA'])
    print(f"\n  {'Configuration':25s}  {'OA':>7}  {'Kappa':>7}  {'F1':>7}  {'ΔOA':>8}")
    print(f"  {'-'*57}")
    for c, r in res.items():
        mk = ' ★' if c == best_k else ''
        print(f"  {r['label']:25s}  {r['OA']:7.4f}  {r['Kappa']:7.4f}  "
              f"{r['F1']:7.4f}  {r['OA']-base:+8.4f}{mk}")
    print(f"  {'Papier MCTNet':25s}  {PAPER['OA']:7.3f}  "
          f"{PAPER['Kappa']:7.3f}  {PAPER['F1']:7.3f}")
    print(f"\n  ★ Best : {res[best_k]['label']}  "
          f"Δ = {(res[best_k]['OA']-base)*100:+.2f} pp")
    print("\n" + "="*65)
    print("✅ Part 2 terminée — California → passer à Part 3")
    print("="*65)


def main():
    print("="*58)
    print("Part 2 — Étape 5 : Résultats d'ablation — California")
    print("="*58 + "\n")
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Fichier non trouvé : {RESULTS_FILE}")
        print("   Lancer d'abord : python CAL_Part2_Step4_ablation_train.py")
        return
    res, hist = load()
    print("\n── Génération des figures ─────────────────────────────────")
    plot_metrics(res)
    plot_delta(res)
    plot_learning(res, hist)
    plot_radar(res)
    print_summary(res)


if __name__ == "__main__":
    main()
