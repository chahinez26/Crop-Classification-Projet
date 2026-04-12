"""
ÉTAPE 4 — Vérification du split et résumé final
=================================================
Entrée  : ARK_dataset_preprocessed.npz
Sortie  : console + figures/fig_split_summary.png

Vérifie que tout est prêt pour entraîner MCTNet :
  Shapes correctes
  Distribution des classes dans chaque split
  Absence de fuite train/test
  Valeurs normalisées dans [0, 1]
  Masque cohérent avec les zéros dans X
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset_preprocessed.npz"
FIG_DIR     = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\figures"
CLASS_NAMES = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'}
COLORS      = ['#2ca02c','#d62728','#1f77b4','#ff7f0e','#9467bd']
# ───────────────────────────────────────────────────────────────

os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    data = np.load(INPUT_FILE)
    splits = {}
    for split in ['train', 'val', 'test']:
        splits[split] = {
            'X': data[f'X_{split}'],
            'y': data[f'y_{split}'],
            'm': data[f'mask_{split}'],
        }
    return splits


def check_shapes(splits):
    print("── 1. Vérification des shapes ──────────────────────")
    expected = {
        'train': (1200, 36, 10),
        'val':   (300,  36, 10),
        'test':  (8500, 36, 10),
    }
    all_ok = True
    for split, sp in splits.items():
        X, y, m = sp['X'], sp['y'], sp['m']
        exp = expected[split]
        ok_X = X.shape == exp
        ok_y = y.shape == (exp[0],)
        ok_m = m.shape == (exp[0], 36)
        status = "✅" if (ok_X and ok_y and ok_m) else "❌"
        print(f"  {status} {split:5s} : X{X.shape}  y{y.shape}  mask{m.shape}")
        if not (ok_X and ok_y and ok_m):
            all_ok = False
    if all_ok:
        print("  → Toutes les shapes sont correctes\n")
    else:
        print("  ⚠  Certaines shapes ne correspondent pas au papier\n")
    return all_ok


def check_class_distribution(splits):
    print("── 2. Distribution des classes ─────────────────────")
    paper = {
        'train': {0:240, 1:240, 2:240, 3:240, 4:240},
        'val':   {0:60,  1:60,  2:60,  3:60,  4:60},
        'test':  {0:1222,1:462, 2:2123,3:4377,4:316},
    }
    print(f"  {'Classe':8s}", end='')
    for split in ['train','val','test']:
        print(f"  {split.capitalize():>8s}", end='')
    print(f"  {'Papier(test)':>14s}")

    for lbl, name in CLASS_NAMES.items():
        print(f"  {name:8s}", end='')
        for split in ['train','val','test']:
            y = splits[split]['y']
            n = int((y == lbl).sum())
            ok = "✓" if n == paper[split][lbl] else "≈"
            print(f"  {n:>6d}{ok:1s}", end='')
        print(f"  {paper['test'][lbl]:>14d}")

    print()
    for split in ['train','val','test']:
        y = splits[split]['y']
        print(f"  Total {split:5s} : {len(y)}")
    print()


def check_no_leakage(splits):
    print("── 3. Absence de fuite train/test ──────────────────")
    # Vérifier que les indices ne se chevauchent pas
    data = np.load(INPUT_FILE)
    train_idx = set(data['train_idx'].tolist())
    val_idx   = set(data['val_idx'].tolist())
    test_idx  = set(data['test_idx'].tolist())

    overlap_tv = train_idx & val_idx
    overlap_tt = train_idx & test_idx
    overlap_vt = val_idx   & test_idx

    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("  ✅ Aucun chevauchement entre train / val / test\n")
    else:
        if overlap_tv: print(f"  ❌ Train ∩ Val   : {len(overlap_tv)} pixels")
        if overlap_tt: print(f"  ❌ Train ∩ Test  : {len(overlap_tt)} pixels")
        if overlap_vt: print(f"  ❌ Val   ∩ Test  : {len(overlap_vt)} pixels")
        print()


def check_normalization(splits):
    print("── 4. Vérification normalisation ───────────────────")
    for split, sp in splits.items():
        X, m = sp['X'], sp['m']
        present = X[m == 0]
        missing = X[m == 1]
        ok_range = (present.min() >= 0) and (present.max() <= 1)
        ok_miss  = np.allclose(missing, 0)
        status_r = "✅" if ok_range else "❌"
        status_m = "✅" if ok_miss  else "❌"
        print(f"  {split:5s} : {status_r} valeurs∈[{present.min():.3f},{present.max():.3f}]  "
              f"{status_m} missing=0  (n_présents={len(present):,})")
    print()


def check_mask_consistency(splits):
    print("── 5. Cohérence masque / valeurs zéro ─────────────")
    for split, sp in splits.items():
        X, m = sp['X'], sp['m']
        # Pixels déclarés missing : toutes bandes doivent être 0
        idx_miss = np.where(m == 1)
        all_zero = np.all(X[idx_miss[0], idx_miss[1], :] == 0)
        status = "✅" if all_zero else "❌"
        n_miss = int(m.sum())
        print(f"  {split:5s} : {status} {n_miss:6d} timesteps missing → "
              f"toutes bandes = 0 : {all_zero}")
    print()


def plot_split_summary(splits):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = ['Train (n=1200)', 'Validation (n=300)', 'Test (n=8500)']

    for ax, (split, sp), title in zip(axes, splits.items(), titles):
        y = sp['y']
        counts = [int((y == lbl).sum()) for lbl in range(5)]
        labels = [CLASS_NAMES[i] for i in range(5)]

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            colors=COLORS,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            textprops={'fontsize': 8}
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title(title, fontsize=11, fontweight='bold')

    plt.suptitle('Distribution des classes par split — Arkansas', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_split_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def print_final_checklist(splits):
    print("\n" + "=" * 55)
    print("RÉCAPITULATIF — Prêt pour MCTNet")
    print("=" * 55)
    print()
    print("  Fichier d'entrée modèle : ARK_dataset_preprocessed.npz")
    print()
    print("  Données :")
    for split, sp in splits.items():
        X, y, m = sp['X'], sp['y'], sp['m']
        miss = 100 * m.mean()
        print(f"    {split:5s} : X={X.shape}  y={y.shape}  missing={miss:.1f}%")

    print()
    print("  Hyperparamètres MCTNet (Arkansas, Table 3 du papier) :")
    print("    n_stage    = 3")
    print("    n_head     = 5")
    print("    kernel_size= 3")
    print("    lr         = 0.001")
    print("    optimizer  = Adam")
    print("    batch_size = 32")
    print("    epochs     = 200")
    print()
    print("  Métriques cibles (Table 5 du papier) :")
    print("    OA     = 0.968")
    print("    Kappa  = 0.951")
    print("    F1     = 0.933")
    print()
    print("  → Lancer maintenant : python step5_train_mctnet.py")
    print("=" * 55)


def main():
    print("=" * 55)
    print("Étape 4 — Vérification du split")
    print("=" * 55 + "\n")

    splits = load_data()

    all_ok = check_shapes(splits)
    check_class_distribution(splits)
    check_no_leakage(splits)
    check_normalization(splits)
    check_mask_consistency(splits)
    plot_split_summary(splits)
    print_final_checklist(splits)

    if all_ok:
        print("\n✅ Tout est correct — tu peux lancer l'entraînement !")
    else:
        print("\n⚠  Des problèmes ont été détectés — vérifier les étapes 1-3")


if __name__ == "__main__":
    main()