"""
ÉTAPE 4 — Vérification du split — California
=============================================
Entrée  : CAL_dataset_preprocessed.npz
Sortie  : console + figures_california/fig_split_summary.png

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset_preprocessed.npz"
FIG_DIR     = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\figures"
N_CLASSES   = 6
CLASS_NAMES = {0:'Grapes', 1:'Rice', 2:'Alfalfa',
               3:'Almonds', 4:'Pistachios', 5:'Others'}
COLORS      = ['#9400D3','#2196F3','#FF9800','#8B4513','#4CAF50','#9E9E9E']
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
    return splits, data


def check_shapes(splits):
    print("── 1. Vérification des shapes ──────────────────────")
    # 6 classes × 240 train + 6 × 60 val = 1440 train, 360 val
    expected_train = (6 * 240, 36, 10)
    expected_val   = (6 * 60,  36, 10)

    all_ok = True
    for split, sp in splits.items():
        X, y, m = sp['X'], sp['y'], sp['m']
        print(f"  {split:5s} : X{X.shape}  y{y.shape}  mask{m.shape}")

        if split == 'train' and X.shape[0] != expected_train[0]:
            print(f"    ⚠  Attendu : {expected_train[0]} samples")
            all_ok = False
        if split == 'val' and X.shape[0] != expected_val[0]:
            print(f"    ⚠  Attendu : {expected_val[0]} samples")
            all_ok = False

    if all_ok:
        print("  → Shapes cohérentes\n")
    return all_ok


def check_class_distribution(splits):
    print("── 2. Distribution des classes ─────────────────────")
    print(f"  {'Classe':12s}", end='')
    for split in ['train', 'val', 'test']:
        print(f"  {split.capitalize():>8s}", end='')
    print()

    for lbl, name in CLASS_NAMES.items():
        print(f"  {name:12s}", end='')
        for split in ['train', 'val', 'test']:
            y = splits[split]['y']
            n = int((y == lbl).sum())
            print(f"  {n:8d}", end='')
        print()

    print()
    for split in ['train', 'val', 'test']:
        print(f"  Total {split:5s} : {len(splits[split]['y'])}")
    print()


def check_no_leakage(data):
    print("── 3. Absence de fuite train/test ──────────────────")
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
              f"{status_m} missing=0")
    print()


def plot_split_summary(splits):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_train = len(splits['train']['y'])
    n_val   = len(splits['val']['y'])
    n_test  = len(splits['test']['y'])
    titles  = [f'Train (n={n_train})', f'Validation (n={n_val})', f'Test (n={n_test})']

    for ax, (split, sp), title in zip(axes, splits.items(), titles):
        y      = sp['y']
        counts = [int((y == lbl).sum()) for lbl in range(N_CLASSES)]
        labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]

        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=COLORS,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.75, textprops={'fontsize': 7}
        )
        for at in autotexts:
            at.set_fontsize(6)
        ax.set_title(title, fontsize=11, fontweight='bold')

    plt.suptitle('Distribution des classes par split — California', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_split_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvé : {out}")


def print_final_checklist(splits):
    print("\n" + "=" * 55)
    print("RÉCAPITULATIF — Prêt pour MCTNet California")
    print("=" * 55)
    for split, sp in splits.items():
        X, y, m = sp['X'], sp['y'], sp['m']
        print(f"    {split:5s} : X={X.shape}  y={y.shape}  missing={100*m.mean():.1f}%")
    print()
    print("  Hyperparamètres MCTNet (California, Table 3 du papier) :")
    print("    n_stage=3, n_head=5, kernel_size=3, lr=0.001, Adam")
    print("    n_classes=6, d_model=30, batch_size=32, epochs=200")
    print()
    print("  Métriques cibles (Table 5 du papier) :")
    print("    OA=0.852  Kappa=0.806  F1=0.829")
    print()
    print("  → Lancer maintenant : python CAL_Step6_train.py")
    print("=" * 55)


def main():
    print("=" * 55)
    print("Étape 4 — Vérification split — California")
    print("=" * 55 + "\n")

    splits, data = load_data()

    check_shapes(splits)
    check_class_distribution(splits)
    check_no_leakage(data)
    check_normalization(splits)
    plot_split_summary(splits)
    print_final_checklist(splits)


if __name__ == "__main__":
    main()