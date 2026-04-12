"""
ÉTAPE 3 — Prétraitement et normalisation
=========================================
Entrée  : ARK_dataset.npz
Sortie  : ARK_dataset_preprocessed.npz

Opérations (conformes au papier) :
  1. Normalisation min-max par bande spectrale
     (calculée sur les pixels présents du train set uniquement
      pour éviter la fuite d'information test→train)
  2. Construction du masque binaire Input 2 du modèle MCTNet
     mask[i,t] = 1 si timestep t manquant, 0 sinon (déjà dans dataset.npz)
  3. Split train / val / test selon Table 2 du papier
     300 pts/classe → 240 train + 60 val | reste → test
  4. Sauvegarde des indices de split pour reproductibilité
"""

import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\ARK_dataset.npz"
OUTPUT_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset_preprocessed.npz"
RANDOM_SEED = 42
BAND_NAMES  = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
CLASS_NAMES = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'}

# Papier Table 2 : 300 pts/classe pour train+val (240 train + 60 val)
N_TRAIN_VAL_PER_CLASS = 300
TRAIN_RATIO           = 0.8    # 240 train, 60 val
# ───────────────────────────────────────────────────────────────


def load_data():
    data = np.load(INPUT_FILE)
    X    = data['X']       # [N, 36, 10]  float32
    y    = data['y']       # [N]          int32
    mask = data['mask']    # [N, 36]      uint8  1=missing
    print(f"Chargé : X{X.shape}  y{y.shape}  mask{mask.shape}\n")
    return X, y, mask


# ── Split train / val / test ──────────────────────────────────
def make_split(y, seed=RANDOM_SEED):
    """
    Papier : "300 samples per crop type were randomly selected to compose
    the training and validation datasets (with a rate of 8:2), and
    the rest samples were used for testing."
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    print("── Split train / val / test ────────────────────────")
    print(f"  {'Classe':8s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}  {'Total':>6s}")

    for lbl, name in CLASS_NAMES.items():
        idx_class = np.where(y == lbl)[0]
        n = len(idx_class)

        # Mélanger
        perm = rng.permutation(n)
        idx_class = idx_class[perm]

        # Prendre min(N_TRAIN_VAL_PER_CLASS, n) pour train+val
        n_tv  = min(N_TRAIN_VAL_PER_CLASS, n)
        n_tr  = int(n_tv * TRAIN_RATIO)     # 240
        n_val = n_tv - n_tr                 # 60

        train_idx.extend(idx_class[:n_tr].tolist())
        val_idx.extend(idx_class[n_tr:n_tv].tolist())
        test_idx.extend(idx_class[n_tv:].tolist())

        print(f"  {name:8s}  {n_tr:6d}  {n_val:6d}  {n-n_tv:6d}  {n:6d}")

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx   = np.array(val_idx,   dtype=np.int64)
    test_idx  = np.array(test_idx,  dtype=np.int64)

    print(f"  {'TOTAL':8s}  {len(train_idx):6d}  {len(val_idx):6d}  "
          f"{len(test_idx):6d}  {len(y):6d}")
    print(f"\n  → Attendu papier : train=1200, val=300, test=8500")

    return train_idx, val_idx, test_idx


# ── Normalisation min-max ─────────────────────────────────────
def normalize(X, train_idx, mask):
    """
    Normalisation min-max par bande, calculée UNIQUEMENT sur le train set
    et sur les pixels présents (mask=0) pour éviter la fuite d'information.

    Formule : X_norm = (X - X_min) / (X_max - X_min)
    Valeurs manquantes (X=0 et mask=1) → restent à 0 après normalisation.
    """
    print("\n── Normalisation min-max ───────────────────────────")
    X_norm = np.zeros_like(X, dtype=np.float32)
    band_stats = []   # pour sauvegarde et inspection

    # Pixels du train set avec données présentes
    X_train = X[train_idx]            # [N_train, 36, 10]
    mask_train = mask[train_idx]       # [N_train, 36]

    for b in range(10):
        # Valeurs de la bande b sur tous les timesteps présents du train
        vals_train = X_train[:, :, b][mask_train == 0]
        b_min = float(vals_train.min())
        b_max = float(vals_train.max())
        band_stats.append({'band': BAND_NAMES[b], 'min': b_min, 'max': b_max})

        # Appliquer la normalisation à tout le dataset
        denom = b_max - b_min if b_max > b_min else 1.0
        X_norm[:, :, b] = np.where(
            mask == 0,                          # pixel présent
            np.clip((X[:, :, b] - b_min) / denom, 0, 1),
            0.0                                 # missing → 0
        )
        print(f"  {BAND_NAMES[b]:4s} : min={b_min:7.0f}  max={b_max:7.0f}  "
              f"→ normalisé [0, 1]")

    return X_norm, band_stats


def print_normalized_stats(X_norm, mask):
    print("\n── Vérification post-normalisation ────────────────")
    present = X_norm[mask == 0]
    print(f"  Pixels présents — min={present.min():.4f}  "
          f"max={present.max():.4f}  mean={present.mean():.4f}")
    missing = X_norm[mask == 1]
    print(f"  Pixels missing  — valeur unique : {np.unique(missing)}")


def main():
    print("=" * 55)
    print("Étape 3 — Prétraitement et normalisation")
    print("=" * 55 + "\n")

    # 1. Charger
    X, y, mask = load_data()

    # 2. Split
    train_idx, val_idx, test_idx = make_split(y)

    # 3. Normaliser (basé sur le train set uniquement)
    X_norm, band_stats = normalize(X, train_idx, mask)
    print_normalized_stats(X_norm, mask)

    # 4. Construire Input 2 (masque binaire pour ALPE)
    # mask est déjà en uint8 avec 1=missing, 0=présent
    # Le modèle MCTNet attend : 1=missing dans ALPE pour zéro les positions
    input2 = mask.astype(np.float32)   # [N, 36]

    # 5. Extraire les splits
    X_train = X_norm[train_idx]      # [1200, 36, 10]
    X_val   = X_norm[val_idx]        # [300,  36, 10]
    X_test  = X_norm[test_idx]       # [8500, 36, 10]
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]
    m_train = input2[train_idx]      # [1200, 36]
    m_val   = input2[val_idx]        # [300,  36]
    m_test  = input2[test_idx]       # [8500, 36]

    print(f"\n── Shapes finales ──────────────────────────────────")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}     y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # 6. Stats min-max bandes pour affichage
    print(f"\n  Bandes band_stats (min/max train set) :")
    for s in band_stats:
        print(f"    {s['band']:4s} : [{s['min']:.0f}, {s['max']:.0f}]")

    # 7. Sauvegarder tout
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train, mask_train=m_train,
        X_val=X_val,     y_val=y_val,     mask_val=m_val,
        X_test=X_test,   y_test=y_test,   mask_test=m_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        # Garder aussi le dataset complet normalisé
        X_all=X_norm, y_all=y, mask_all=input2,
    )
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("\n  Contenu :")
    print("    X_train/val/test : [N, 36, 10]  float32  normalisé [0,1]")
    print("    y_train/val/test : [N]           int32")
    print("    mask_train/val/test : [N, 36]   float32  1=missing (Input 2 MCTNet)")
    print("\n  → Lancer maintenant : python step4_verify_split.py")


if __name__ == "__main__":
    main()