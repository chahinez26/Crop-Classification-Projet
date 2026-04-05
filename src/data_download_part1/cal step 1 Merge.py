"""
merge_california.py — Fusion CSV California → CAL_dataset.npz
=============================================================
Entrée  : dossier MCTNet_California_v2/ avec 74 CSV
Sortie  : CAL_dataset.npz → X[10020, 36, 10], y[10020], mask[10020, 36]

SPÉCIFICITÉS CALIFORNIA vs Arkansas :
  - 6 classes (Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5)
  - Zones ASYMÉTRIQUES : Z0=5720 pts, Z1=4300 pts → total=10020 pts
  - CLASS_POINTS différents par zone (stratégie géographique)
  - Moins de nuages qu'Arkansas (climat méditerranéen)

Papier Table 2 — California :
  Grapes=2054, Rice=2037, Alfalfa=974, Almonds=783, Pistachios=640, Others=3512
  TOTAL=10000  (notre total=10020, diff=+20 ≈ papier ✅)
"""

import os, json, csv
import numpy as np

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2"
OUTPUT_FILE = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset.npz"
N_TIMESTEPS = 36
ZONES       = [0, 1]
BAND_ORDER  = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
CLASS_NAMES = {0:'Grapes', 1:'Rice', 2:'Alfalfa',
               3:'Almonds', 4:'Pistachios', 5:'Others'}
PAPER_DIST  = {0:2054, 1:2037, 2:974, 3:783, 4:640, 5:3512}

# CLASS_POINTS asymétriques (identiques au script GEE v2)
CLASS_POINTS_Z0 = {0:1030, 1:2030, 2:490, 3:390, 4:20,  5:1760}  # Z0=5720
CLASS_POINTS_Z1 = {0:1030, 1:10,   2:490, 3:390, 4:620, 5:1760}  # Z1=4300
# ───────────────────────────────────────────────────────────────


def parse_geo(geo_str):
    g = json.loads(geo_str)
    return float(g['coordinates'][0]), float(g['coordinates'][1])


def load_cdl():
    """
    Charge les points CDL des 2 zones.
    Retourne ref : dict (zone, system:index) -> {label, lon, lat, zone}
    """
    print("── Chargement CDL California ───────────────────────")
    ref = {}
    expected = {0: sum(CLASS_POINTS_Z0.values()),
                1: sum(CLASS_POINTS_Z1.values())}

    for z in ZONES:
        fpath = os.path.join(INPUT_DIR, f"CAL_CDL_Z{z}.csv")
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
        status = "OK" if n_zone == expected[z] else f"attente {expected[z]}"
        print(f"  Z{z} : {n_zone} points ({status})")

    print(f"  Total : {len(ref)} points (Z0={expected[0]}, Z1={expected[1]})")
    print()
    return ref


def build_index_order(ref):
    """Ordre stable : Z0 trié par index, puis Z1 trié par index."""
    z0 = sorted([(int(k[1]), k) for k in ref if k[0] == 0])
    z1 = sorted([(int(k[1]), k) for k in ref if k[0] == 1])
    ordered_keys = [k for _, k in z0] + [k for _, k in z1]
    key_to_row   = {k: i for i, k in enumerate(ordered_keys)}
    return ordered_keys, key_to_row


def load_spectral(t_idx, zone, key_to_row, X, mask):
    """
    Remplit X[i, t, :] et mask[i, t] depuis CAL_T{t}_Z{z}.csv.
    Retourne (n_matched, n_missing).
    """
    t_str = f"{t_idx+1:02d}"
    fpath = os.path.join(INPUT_DIR, f"CAL_T{t_str}_Z{zone}.csv")

    if not os.path.exists(fpath):
        print(f"  Absent : CAL_T{t_str}_Z{zone}.csv -> timestep = 0 (missing)")
        return 0, 0

    matched, missing_count = 0, 0
    with open(fpath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (zone, row['system:index'])
            if key not in key_to_row:
                continue
            i     = key_to_row[key]
            bands = np.array([float(row.get(b, 0) or 0) for b in BAND_ORDER],
                             dtype=np.float32)
            X[i, t_idx, :] = bands
            if np.any(bands != 0):
                mask[i, t_idx] = 0
                matched += 1
            else:
                missing_count += 1

    return matched, missing_count


def print_class_distribution(y):
    N = len(y)
    print(f"\n  Distribution des classes (N={N}) :")
    print(f"  {'Classe':12s} {'Obtenu':>7s} {'%':>6s}  {'Papier':>7s} {'%':>6s}  OK?")
    for lbl, name in CLASS_NAMES.items():
        n     = int((y == lbl).sum())
        pct   = 100 * n / N
        pap   = PAPER_DIST[lbl]
        pap_p = 100 * pap / 10000
        diff  = abs(n - pap)
        ok    = "OK" if diff <= 50 else f"diff={diff:+d}"
        print(f"  {name:12s} {n:7d} {pct:6.1f}%  {pap:7d} {pap_p:6.1f}%  {ok}")
    print(f"  {'TOTAL':12s} {N:7d} {100.0:6.1f}%  {10000:7d} {100.0:6.1f}%")


def main():
    print("=" * 57)
    print("merge_california.py -> CAL_dataset.npz")
    print("=" * 57 + "\n")

    if not os.path.isdir(INPUT_DIR):
        print(f"ERREUR : Dossier '{INPUT_DIR}' introuvable.")
        print(f"   Place tous les CSV de MCTNet_California_v2/ dans ce dossier.")
        return

    # 1. Charger les labels CDL
    ref = load_cdl()
    ordered_keys, key_to_row = build_index_order(ref)
    N = len(ordered_keys)

    # 2. Initialiser les matrices
    X    = np.zeros((N, N_TIMESTEPS, 10), dtype=np.float32)
    y    = np.array([ref[k]['label'] for k in ordered_keys], dtype=np.int32)
    mask = np.ones((N, N_TIMESTEPS), dtype=np.uint8)   # 1=missing par defaut
    lons = np.array([ref[k]['lon'] for k in ordered_keys], dtype=np.float64)
    lats = np.array([ref[k]['lat'] for k in ordered_keys], dtype=np.float64)

    # 3. Remplir timestep par timestep
    print("── Extraction spectrale ────────────────────────────")
    for t in range(N_TIMESTEPS):
        t_str = f"{t+1:02d}"
        total_matched = 0
        for z in ZONES:
            result = load_spectral(t, z, key_to_row, X, mask)
            if result:
                total_matched += result[0]

        miss_n   = int(mask[:, t].sum())
        miss_pct = 100 * miss_n / N
        print(f"  T{t_str} : {total_matched:5d} pts avec donnees | "
              f"{miss_n:5d} missing ({miss_pct:.1f}%)")

    # 4. Resume
    print("\n── Resume final ────────────────────────────────────")
    print(f"  X    : {X.shape}  float32")
    print(f"  y    : {y.shape}   int32")
    print(f"  mask : {mask.shape}  uint8  (1=missing, 0=donnees)")
    print(f"  Missing global : {100*mask.mean():.2f}%")
    print(f"  (Arkansas = 21.1% -- California devrait etre < 10%)")

    print_class_distribution(y)

    print(f"\n  Repartition zones :")
    print(f"  Z0 (Sacramento Valley) : {sum(1 for k in ordered_keys if k[0]==0)} pts")
    print(f"  Z1 (San Joaquin Sud)   : {sum(1 for k in ordered_keys if k[0]==1)} pts")

    # 5. Sauvegarder
    np.savez_compressed(
        OUTPUT_FILE,
        X=X, y=y, mask=mask, lons=lons, lats=lats
    )
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n  Sauvegarde : {OUTPUT_FILE}  ({size_mb:.1f} MB)")

    print("\n  Charger dans Python :")
    print("    data = np.load('CAL_dataset.npz')")
    print("    X, y, mask = data['X'], data['y'], data['mask']")
    print(f"    # X.shape = {X.shape}")
    print("    # Classes : 0=Grapes, 1=Rice, 2=Alfalfa,")
    print("    #           3=Almonds, 4=Pistachios, 5=Others")
    print()
    print("  -> Lancer ensuite les steps 2 a 7 avec les adaptations California")


if __name__ == "__main__":
    main()