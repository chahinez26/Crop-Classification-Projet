"""
ÉTAPE 3 — Prétraitement et normalisation — California
======================================================
Entrée  : CAL_dataset.npz
Sortie  : CAL_dataset_preprocessed.npz

Différences California vs Arkansas :
  - 6 classes (Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5)
  - N total ≈ 10 020 points (zones asymétriques)
  - Même protocole papier : 300 pts/classe → 240 train + 60 val | reste → test
    SAUF Others : beaucoup plus de points test (~3212)
"""

import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset.npz"
OUTPUT_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset_preprocessed.npz"
RANDOM_SEED = 42
BAND_NAMES  = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
N_CLASSES   = 6
CLASS_NAMES = {0:'Grapes', 1:'Rice', 2:'Alfalfa',
               3:'Almonds', 4:'Pistachios', 5:'Others'}

# Papier : 300 pts/classe pour train+val (240 train + 60 val)
N_TRAIN_VAL_PER_CLASS = 300
TRAIN_RATIO           = 0.8   # 240 train, 60 val
# ───────────────────────────────────────────────────────────────


def load_data():
    data = np.load(INPUT_FILE)
    X    = data['X']
    y    = data['y']
    mask = data['mask']
    print(f"Chargé : X{X.shape}  y{y.shape}  mask{mask.shape}\n")
    return X, y, mask


def make_split(y, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    print("── Split train / val / test ────────────────────────")
    print(f"  {'Classe':12s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}  {'Total':>6s}")

    for lbl, name in CLASS_NAMES.items():
        idx_class = np.where(y == lbl)[0]
        n = len(idx_class)

        perm      = rng.permutation(n)
        idx_class = idx_class[perm]

        n_tv  = min(N_TRAIN_VAL_PER_CLASS, n)
        n_tr  = int(n_tv * TRAIN_RATIO)
        n_val = n_tv - n_tr

        train_idx.extend(idx_class[:n_tr].tolist())
        val_idx.extend(idx_class[n_tr:n_tv].tolist())
        test_idx.extend(idx_class[n_tv:].tolist())

        print(f"  {name:12s}  {n_tr:6d}  {n_val:6d}  {n-n_tv:6d}  {n:6d}")

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx   = np.array(val_idx,   dtype=np.int64)
    test_idx  = np.array(test_idx,  dtype=np.int64)

    print(f"  {'TOTAL':12s}  {len(train_idx):6d}  {len(val_idx):6d}  "
          f"{len(test_idx):6d}  {len(y):6d}")
    print(f"\n  → Attendu papier : train=1440, val=360, test≈8220")
    return train_idx, val_idx, test_idx


def normalize(X, train_idx, mask):
    print("\n── Normalisation min-max ───────────────────────────")
    X_norm     = np.zeros_like(X, dtype=np.float32)
    band_stats = []

    X_train    = X[train_idx]
    mask_train = mask[train_idx]

    for b in range(10):
        vals_train = X_train[:, :, b][mask_train == 0]
        b_min = float(vals_train.min())
        b_max = float(vals_train.max())
        band_stats.append({'band': BAND_NAMES[b], 'min': b_min, 'max': b_max})

        denom = b_max - b_min if b_max > b_min else 1.0
        X_norm[:, :, b] = np.where(
            mask == 0,
            np.clip((X[:, :, b] - b_min) / denom, 0, 1),
            0.0
        )
        print(f"  {BAND_NAMES[b]:4s} : min={b_min:7.0f}  max={b_max:7.0f}  "
              f"→ normalisé [0, 1]")

    return X_norm, band_stats


def main():
    print("=" * 55)
    print("Étape 3 — Prétraitement California")
    print("=" * 55 + "\n")

    X, y, mask = load_data()

    train_idx, val_idx, test_idx = make_split(y)

    X_norm, band_stats = normalize(X, train_idx, mask)

    # Vérification post-normalisation
    present = X_norm[mask == 0]
    missing = X_norm[mask == 1]
    print(f"\n── Vérification ────────────────────────────────────")
    print(f"  Présents : min={present.min():.4f}  max={present.max():.4f}")
    print(f"  Missing  : valeur unique = {np.unique(missing)}")

    input2  = mask.astype(np.float32)

    X_train = X_norm[train_idx]
    X_val   = X_norm[val_idx]
    X_test  = X_norm[test_idx]
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]
    m_train = input2[train_idx]
    m_val   = input2[val_idx]
    m_test  = input2[test_idx]

    print(f"\n── Shapes finales ──────────────────────────────────")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}     y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train, mask_train=m_train,
        X_val=X_val,     y_val=y_val,     mask_val=m_val,
        X_test=X_test,   y_test=y_test,   mask_test=m_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        X_all=X_norm, y_all=y, mask_all=input2,
    )
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("\n  → Lancer maintenant : python CAL_Step4_verify_split.py")


if __name__ == "__main__":
    main()