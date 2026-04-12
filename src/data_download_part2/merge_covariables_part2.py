"""
merge_covariables_part2.py
==========================
Part 2 — Intégration des covariables environnementales

Ce script fait DEUX choses :
  1. Fusionner les CSV GEE en 4 fichiers NPZ de covariables :
       ARK_cov_climat.npz   → X_climat [10000, 3]
       ARK_cov_soil.npz     → X_soil   [10000, 3]
       ARK_cov_topo.npz     → X_topo   [10000, 2]
       ARK_cov_all.npz      → X_all    [10000, 8]  (3+3+2)

  2. Fournir build_config() qui construit chaque configuration
     d'entraînement pour l'ablation study :
       Config 1 : Sentinel-2 seul           → X [10000, 36, 10]
       Config 2 : Sentinel-2 + climat       → X [10000, 36, 13]
       Config 3 : Sentinel-2 + sol          → X [10000, 36, 13]
       Config 4 : Sentinel-2 + topo         → X [10000, 36, 12]
       Config 5 : Sentinel-2 + tout         → X [10000, 36, 18]

STRATÉGIE DE FUSION (statique → temporel) :
  Les covariables sont statiques (1 vecteur par pixel).
  Pour les injecter dans MCTNet qui attend [N, T, C] :
  On répète le vecteur statique sur les 36 timesteps :
    X_cov_repeated = np.repeat(X_cov[:, np.newaxis, :], 36, axis=1)
    # shape : [10000, 36, n_cov]
  Puis on concatène sur l'axe des bandes (axis=2) :
    X_combined = np.concatenate([X_temporal, X_cov_repeated], axis=2)
  → MCTNet reçoit exactement le même format [N, T, C] qu'avant.
  → Les bandes statiques ont la même valeur sur tous les timesteps,
    le modèle apprend à les utiliser comme contexte global par pixel.

STRUCTURE DES FICHIERS INPUT :
  MCTNet_Covariables/
    ARK_COV_CLIMAT_Z0.csv   ARK_COV_CLIMAT_Z1.csv
    ARK_COV_SOL_Z0.csv      ARK_COV_SOL_Z1.csv
    ARK_COV_TOPO_Z0.csv     ARK_COV_TOPO_Z1.csv

  MCTNet_v5/npz/
    ARK_dataset.npz          ← baseline Sentinel existant

UTILISATION :
  python merge_covariables_part2.py
  → Produit les 4 NPZ dans MCTNet_v5/npz/
  → Importer build_config() dans ton script d'entraînement
"""

import os
import csv
import json
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adapter selon ta machine
# ══════════════════════════════════════════════════════════════════════════════
COV_DIR    = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_Covariables"
SENTINEL_NPZ = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset.npz"
OUT_DIR    = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz"

# Bandes de chaque groupe — DANS L'ORDRE du CSV GEE
CLIMAT_BANDS  = ['temp_mean', 'precip_total', 'dewpoint_mean']   # 3 features
SOIL_BANDS    = ['soil_ph', 'soil_oc', 'soil_texture']           # 3 features
TOPO_BANDS    = ['elevation', 'landforms']                        # 2 features

# Distribution Arkansas (identique au script GEE — pour vérification)
CLASS_POINTS  = {0: 760, 1: 380, 2: 1210, 3: 2340, 4: 310}
CLASS_NAMES   = {0: 'Corn', 1: 'Cotton', 2: 'Rice', 3: 'Soybean', 4: 'Others'}
N_ZONES       = 2
N_TOTAL       = 10000   # 5000 × 2 zones

# ══════════════════════════════════════════════════════════════════════════════


def parse_geo(geo_str):
    g = json.loads(geo_str)
    return float(g['coordinates'][0]), float(g['coordinates'][1])


def load_cdl_order():
    """
    Charge l'ordre de référence depuis ARK_dataset.npz.
    Le NPZ Sentinel contient lons + lats dans l'ordre exact des points.
    On reconstruit une clé (lon, lat) → indice ligne.

    Retourne key_to_row : dict (lon_str, lat_str) -> int
    """
    if not os.path.exists(SENTINEL_NPZ):
        raise FileNotFoundError(
            f"ARK_dataset.npz introuvable : {SENTINEL_NPZ}\n"
            "Lance d'abord le script merge du code de ton ami."
        )
    data = np.load(SENTINEL_NPZ)
    lons = data['lons']
    lats = data['lats']
    # Clé = chaîne arrondie à 6 décimales pour éviter les erreurs float
    key_to_row = {
        (f"{lons[i]:.6f}", f"{lats[i]:.6f}"): i
        for i in range(len(lons))
    }
    print(f"  Ordre Sentinel chargé : {len(key_to_row)} points")
    return key_to_row, len(lons)


def load_cov_csv(zone, prefix, bands, key_to_row, X, col_start):
    """
    Remplit X[:, col_start:col_start+len(bands)] depuis un CSV GEE.
    Match via (lon, lat) arrondi à 6 décimales.

    Retourne (n_matched, n_missing)
    """
    fname = f"{prefix}_Z{zone}.csv"
    fpath = os.path.join(COV_DIR, fname)

    if not os.path.exists(fpath):
        print(f"    ⚠  Manquant : {fname} → colonnes laissées à 0")
        return 0, 0

    n_matched, n_missing = 0, 0

    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lon, lat = parse_geo(row['.geo'])
            key = (f"{lon:.6f}", f"{lat:.6f}")

            if key not in key_to_row:
                n_missing += 1
                continue

            i = key_to_row[key]
            for j, band in enumerate(bands):
                val = row.get(band, '')
                X[i, col_start + j] = float(val) if val != '' else 0.0
            n_matched += 1

    return n_matched, n_missing


def build_cov_matrix(prefix, bands):
    """
    Construit X_cov [N, len(bands)] en lisant Z0 + Z1.
    Retourne X_cov float32.
    """
    key_to_row, N = load_cdl_order()
    n_bands = len(bands)
    X_cov = np.zeros((N, n_bands), dtype=np.float32)

    total_matched = 0
    for z in range(N_ZONES):
        matched, missing = load_cov_csv(z, prefix, bands, key_to_row, X_cov, col_start=0)
        total_matched += matched
        if missing > 0:
            print(f"    Z{z} : {matched} pts matchés, {missing} clés inconnues")
        else:
            print(f"    Z{z} : {matched} pts matchés ✓")

    coverage = 100 * total_matched / N
    print(f"    Couverture totale : {total_matched}/{N} ({coverage:.1f}%)")
    return X_cov


def print_stats(X, bands, name):
    print(f"\n  Stats {name} :")
    print(f"  {'Variable':<20s}  {'Moy':>9s}  {'Std':>9s}  {'Min':>9s}  {'Max':>9s}  Nulls%")
    print("  " + "-" * 65)
    for j, b in enumerate(bands):
        col   = X[:, j]
        nulls = 100 * (col == 0).mean()
        print(f"  {b:<20s}  {col.mean():9.3f}  {col.std():9.3f}  "
              f"{col.min():9.3f}  {col.max():9.3f}  {nulls:5.1f}%")


def save_npz(X_cov, bands, out_name):
    out_path = os.path.join(OUT_DIR, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(out_path, X_cov=X_cov,
                        band_names=np.array(bands))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  ✅ Sauvegardé : {out_name}  ({size_mb:.2f} MB)")
    print(f"     X_cov.shape = {X_cov.shape}  dtype={X_cov.dtype}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Production des 4 NPZ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 60
    print(sep)
    print("  merge_covariables_part2.py")
    print("  Part 2 — Covariables environnementales Arkansas")
    print(sep)

    if not os.path.isdir(COV_DIR):
        print(f"\n❌ Dossier covariables introuvable : {COV_DIR}")
        return

    # ── 1. CLIMAT ────────────────────────────────────────────────────────────
    print("\n── Climat (ERA5) ───────────────────────────────────────")
    X_climat = build_cov_matrix('ARK_COV_CLIMAT', CLIMAT_BANDS)
    print_stats(X_climat, CLIMAT_BANDS, 'Climat')
    save_npz(X_climat, CLIMAT_BANDS, 'ARK_cov_climat.npz')

    # ── 2. SOL ───────────────────────────────────────────────────────────────
    print("\n── Sol (OpenLandMap) ───────────────────────────────────")
    X_soil = build_cov_matrix('ARK_COV_SOL', SOIL_BANDS)
    print_stats(X_soil, SOIL_BANDS, 'Sol')
    save_npz(X_soil, SOIL_BANDS, 'ARK_cov_soil.npz')

    # ── 3. TOPO ──────────────────────────────────────────────────────────────
    print("\n── Topo (ETOPO1 + ALOS landforms) ─────────────────────")
    X_topo = build_cov_matrix('ARK_COV_TOPO', TOPO_BANDS)
    print_stats(X_topo, TOPO_BANDS, 'Topo')
    save_npz(X_topo, TOPO_BANDS, 'ARK_cov_topo.npz')

    # ── 4. TOUT COMBINÉ ──────────────────────────────────────────────────────
    print("\n── Combiné (3+3+2 = 8 features) ───────────────────────")
    X_all   = np.concatenate([X_climat, X_soil, X_topo], axis=1)
    all_bands = CLIMAT_BANDS + SOIL_BANDS + TOPO_BANDS
    print_stats(X_all, all_bands, 'All')
    save_npz(X_all, all_bands, 'ARK_cov_all.npz')

    # ── Résumé ───────────────────────────────────────────────────────────────
    print("\n" + sep)
    print("  RÉSUMÉ — 4 NPZ produits")
    print(sep)
    print("  ARK_cov_climat.npz  →  X_cov [10000, 3]  (temp, precip, dewpoint)")
    print("  ARK_cov_soil.npz    →  X_cov [10000, 3]  (ph, oc, texture)")
    print("  ARK_cov_topo.npz    →  X_cov [10000, 2]  (elevation, landforms)")
    print("  ARK_cov_all.npz     →  X_cov [10000, 8]  (3+3+2)")
    print()
    print("  → Utiliser build_config() dans ton script d'entraînement")
    print("  → Voir exemples ci-dessous")


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION À IMPORTER DANS TON SCRIPT D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_config(config_id, npz_dir=None):
    """
    Construit X_train et y pour une configuration d'ablation.

    Paramètres
    ----------
    config_id : int — 1 à 5
        1 = Sentinel-2 seul           (baseline)
        2 = Sentinel-2 + climat
        3 = Sentinel-2 + sol
        4 = Sentinel-2 + topo
        5 = Sentinel-2 + all covariates

    npz_dir : str — dossier contenant les NPZ
        Si None, utilise OUT_DIR défini en haut du script.

    Retourne
    --------
    X : np.ndarray  [N, 36, C]  float32
        C = 10 (config 1), 13 (config 2/3), 12 (config 4), 18 (config 5)
    y : np.ndarray  [N]         int32
    mask : np.ndarray [N, 36]   uint8   (1=missing, 0=données)
    config_name : str

    Exemple d'utilisation dans train.py
    ------------------------------------
    from merge_covariables_part2 import build_config

    for cfg_id in [1, 2, 3, 4, 5]:
        X, y, mask, name = build_config(cfg_id)
        print(f"{name}: X.shape={X.shape}")
        # → passer X, y, mask à ton DataLoader MCTNet
    """
    if npz_dir is None:
        npz_dir = OUT_DIR

    config_names = {
        1: 'Sentinel-2 only (baseline)',
        2: 'Sentinel-2 + climat',
        3: 'Sentinel-2 + sol',
        4: 'Sentinel-2 + topo',
        5: 'Sentinel-2 + all covariates',
    }
    cov_files = {
        2: 'ARK_cov_climat.npz',
        3: 'ARK_cov_soil.npz',
        4: 'ARK_cov_topo.npz',
        5: 'ARK_cov_all.npz',
    }

    # Charger le dataset Sentinel de base
    sent_path = os.path.join(npz_dir, 'ARK_dataset.npz')
    if not os.path.exists(sent_path):
        # Essayer le chemin absolu défini en configuration
        sent_path = SENTINEL_NPZ
    data = np.load(sent_path)
    X_temporal = data['X']     # [N, 36, 10]
    y          = data['y']     # [N]
    mask       = data['mask']  # [N, 36]
    N, T, C    = X_temporal.shape   # 10000, 36, 10

    if config_id == 1:
        # Config 1 : Sentinel seul — rien à ajouter
        return X_temporal, y, mask, config_names[1]

    # Configs 2-5 : charger les covariables et répéter sur T timesteps
    cov_path = os.path.join(npz_dir, cov_files[config_id])
    if not os.path.exists(cov_path):
        raise FileNotFoundError(
            f"NPZ covariables manquant : {cov_path}\n"
            "Lance d'abord merge_covariables_part2.main()"
        )

    cov_data = np.load(cov_path)
    X_cov = cov_data['X_cov']   # [N, n_cov]
    n_cov = X_cov.shape[1]

    # Répéter X_cov sur les 36 timesteps
    # [N, n_cov] → [N, 1, n_cov] → [N, 36, n_cov]
    X_cov_repeated = np.repeat(X_cov[:, np.newaxis, :], T, axis=1)

    # Concaténer sur l'axe des bandes
    # [N, 36, 10] + [N, 36, n_cov] → [N, 36, 10+n_cov]
    X_combined = np.concatenate([X_temporal, X_cov_repeated], axis=2)

    return X_combined, y, mask, config_names[config_id]


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION DE BUILD_CONFIG (exécuté seulement si lancé directement)
# ══════════════════════════════════════════════════════════════════════════════

def demo_configs():
    """Affiche les shapes des 5 configurations sans entraîner."""
    print("\n" + "=" * 60)
    print("  DÉMONSTRATION — 5 configurations d'ablation")
    print("=" * 60)

    expected_C = {1: 10, 2: 13, 3: 13, 4: 12, 5: 18}

    for cfg_id in range(1, 6):
        try:
            X, y, mask, name = build_config(cfg_id)
            C_actual = X.shape[2]
            ok = "✓" if C_actual == expected_C[cfg_id] else f"⚠ attendu {expected_C[cfg_id]}"
            print(f"\n  Config {cfg_id} : {name}")
            print(f"    X.shape   = {X.shape}   {ok}")
            print(f"    y.shape   = {y.shape}")
            print(f"    mask.shape= {mask.shape}")
            # Distribution des classes
            for lbl, cname in CLASS_NAMES.items():
                n = int((y == lbl).sum())
                print(f"    {cname:<10s}: {n:5d} pts ({100*n/len(y):.1f}%)")
        except FileNotFoundError as e:
            print(f"\n  Config {cfg_id} : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLE D'INTÉGRATION DANS STEP6_TRAIN.PY
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_EXAMPLE = """
# ── Dans ton step6_train.py ──────────────────────────────────────────────────
from merge_covariables_part2 import build_config
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

RESULTS = {}

for config_id in [1, 2, 3, 4, 5]:
    X, y, mask, config_name = build_config(config_id)
    print(f"\\n{'='*50}")
    print(f"Configuration {config_id}: {config_name}")
    print(f"X.shape = {X.shape}")   # [10000, 36, C]

    # Adapter l'entrée de MCTNet au bon nombre de canaux C
    # MCTNet attend [N, C, T] (channels-first) selon l'implémentation
    X_tensor    = torch.from_numpy(X).float().permute(0, 2, 1)  # [N, C, T]
    y_tensor    = torch.from_numpy(y).long()
    mask_tensor = torch.from_numpy(mask).float()

    dataset = TensorDataset(X_tensor, y_tensor, mask_tensor)
    n_train = int(0.7 * len(dataset))
    n_val   = int(0.15 * len(dataset))
    n_test  = len(dataset) - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Instancier MCTNet avec le bon n_channels
    n_channels = X.shape[2]   # 10, 13, 13, 12, ou 18
    model = MCTNet(in_channels=n_channels, n_classes=5, n_timesteps=36)

    # → Entraîner, évaluer, stocker résultats
    acc, f1, kappa = train_and_evaluate(model, train_loader, val_loader, test_loader)
    RESULTS[config_name] = {'acc': acc, 'f1': f1, 'kappa': kappa}

# Tableau ablation study
print("\\n=== ABLATION STUDY — Arkansas ===")
print(f"{'Configuration':<35s}  {'Acc':>6s}  {'F1':>6s}  {'Kappa':>6s}")
for name, res in RESULTS.items():
    print(f"{name:<35s}  {res['acc']:6.3f}  {res['f1']:6.3f}  {res['kappa']:6.3f}")
# ─────────────────────────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    main()
    demo_configs()
    print("\n" + "=" * 60)
    print("  Exemple d'intégration dans step6_train.py :")
    print("=" * 60)
    print(TRAIN_EXAMPLE)
