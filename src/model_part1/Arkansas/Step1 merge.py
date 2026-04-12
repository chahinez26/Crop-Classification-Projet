"""
Entrée  : dossier MCTNet_v5/ avec 74 CSV
Sortie  : ARK_dataset.npz → X[10000,36,10], y[10000], mask[10000,36]

Colonnes CDL  : system:index, cdl_raw, crop_label, .geo
Colonnes T**  : system:index, B02-B12, B8A, crop_label, .geo
=> system:index est identique entre CDL et T** pour le même pixel
    => on merge par system:index + zone, pas par coordonnées flottantes
"""

import os, json, csv
import numpy as np
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"       # dossier contenant tous les CSV GEE
OUTPUT_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset.npz"
N_TIMESTEPS = 36
ZONES       = [0, 1]
# Ordre des bandes dans la matrice X (identique au papier)
BAND_ORDER  = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
CLASS_NAMES = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'}
# ───────────────────────────────────────────────────────────────


def parse_geo(geo_str):
    g = json.loads(geo_str)
    lon, lat = g['coordinates']
    return float(lon), float(lat)


def load_cdl():
    """Charge les points de référence (index → label, lon, lat, zone)."""
    print("── Chargement CDL ──────────────────────────────────")
    ref = {}   # clé : (zone, system:index)  →  dict
    for z in ZONES:
        fpath = os.path.join(INPUT_DIR, f"ARK_CDL_Z{z}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Manquant : {fpath}")
        with open(fpath, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (z, row['system:index'])
                lon, lat = parse_geo(row['.geo'])
                ref[key] = {
                    'label': int(row['crop_label']),
                    'lon':   lon,
                    'lat':   lat,
                    'zone':  z,
                }
        n_zone = sum(1 for k in ref if k[0] == z)
        print(f"  Z{z} : {n_zone} points")
    print(f"  Total : {len(ref)} points\n")
    return ref


def build_index_order(ref):   
    """Trie les clés (zone, system:index) pour aligner les matrices X, y, mask."""
    z0 = sorted([(int(k[1]), k) for k in ref if k[0] == 0])
    z1 = sorted([(int(k[1]), k) for k in ref if k[0] == 1])
    ordered_keys = [k for _, k in z0] + [k for _, k in z1]
    key_to_row   = {k: i for i, k in enumerate(ordered_keys)}
    return ordered_keys, key_to_row


def load_spectral(t_idx, zone, key_to_row, X, mask):
    """Remplit X[i, t, :] et mask[i, t] depuis un fichier T**_Z*.csv."""
    t_str = f"{t_idx+1:02d}"
    fpath = os.path.join(INPUT_DIR, f"ARK_T{t_str}_Z{zone}.csv")
    matched, missing_count = 0, 0

    if not os.path.exists(fpath):
        print(f"  ⚠  ARK_T{t_str}_Z{zone}.csv absent → timestep entier = 0 (missing)")
        # Les pixels de cette zone restent à 0 et mask reste 1 (déjà initialisé)
        return

    with open(fpath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (zone, row['system:index'])
            if key not in key_to_row:
                continue
            i = key_to_row[key]
            bands = np.array([float(row.get(b, 0) or 0) for b in BAND_ORDER],
                             dtype=np.float32)
            X[i, t_idx, :] = bands
            if np.any(bands != 0):
                mask[i, t_idx] = 0     # données présentes
                matched += 1
            else:
                missing_count += 1

    return matched, missing_count


def main():
    print("=" * 55)
    print("Étape 1 — Merge CSV → ARK_dataset.npz")
    print("=" * 55 + "\n")

    if not os.path.isdir(INPUT_DIR):
        print(f"❌ Dossier '{INPUT_DIR}' introuvable.")
        print("   Place tous les CSV GEE dans ce dossier.")
        return

    # 1. Charger les labels CDL
    ref = load_cdl()
    ordered_keys, key_to_row = build_index_order(ref)
    N = len(ordered_keys)

    # 2. Initialiser les matrices
    X    = np.zeros((N, N_TIMESTEPS, 10), dtype=np.float32)
    y    = np.array([ref[k]['label'] for k in ordered_keys], dtype=np.int32)
    mask = np.ones((N, N_TIMESTEPS), dtype=np.uint8)   # 1 = missing
    lons = np.array([ref[k]['lon'] for k in ordered_keys], dtype=np.float64)
    lats = np.array([ref[k]['lat'] for k in ordered_keys], dtype=np.float64)

    # 3. Remplir timestep par timestep
    print("── Extraction spectrale ────────────────────────────")
    for t in range(N_TIMESTEPS):
        t_str = f"{t+1:02d}"
        total_matched, total_missing = 0, 0
        for z in ZONES:
            result = load_spectral(t, z, key_to_row, X, mask)
            if result:
                m, ms = result
                total_matched += m
                total_missing += ms
        miss_pct = 100 * mask[:, t].mean()
        print(f"  T{t_str} : {total_matched:5d} pts avec données | "
              f"{int(mask[:,t].sum()):4d} missing ({miss_pct:.1f}%)")

    # 4. Résumé
    print("\n── Résumé ──────────────────────────────────────────")
    print(f"  X    : {X.shape}  float32")
    print(f"  y    : {y.shape}  int32")
    print(f"  mask : {mask.shape}  uint8  (1=missing, 0=données)")
    print(f"\n  Distribution des classes :")
    paper = {0:1522, 1:762, 2:2423, 3:4677, 4:616}
    for lbl, name in CLASS_NAMES.items():
        n = int((y == lbl).sum())
        print(f"    {name:8s} : {n:5d}  (papier: {paper[lbl]})")
    print(f"    {'TOTAL':8s} : {N:5d}  (papier: 10000)")
    print(f"\n  Missing global : {100*mask.mean():.1f}%")

    # 5. Sauvegarder
    np.savez_compressed(OUTPUT_FILE,
                        X=X, y=y, mask=mask, lons=lons, lats=lats)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("\n  → Lancer maintenant : python step2_eda.py")


if __name__ == "__main__":
    main()