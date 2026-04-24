"""
PART 2 — ÉTAPE 1 : Merge covariables → CAL_covariates_v2.npz
=============================================================
Correction prof : climat = données TEMPORELLES (36 timesteps)
                  sol + topo = données STATIQUES (inchangé)

Entrées :
  CAL_dataset_preprocessed.npz
  CAL_CDL_Z0.csv, CAL_CDL_Z1.csv
  CAL_CLIM_T01_Z0.csv ... CAL_CLIM_T36_Z1.csv  (72 CSV)
  CAL_SOIL_Z0.csv, CAL_SOIL_Z1.csv
  CAL_TOPO_Z0.csv, CAL_TOPO_Z1.csv

Sortie : CAL_covariates_v2.npz
  X_s2_all   [10020, 36, 10]  S2 normalisé (Part 1)
  X_clim     [10020, 36,  3]  Climat temporel normalisé
  X_soil     [10020,      3]  Sol statique normalisé
  X_topo     [10020,      2]  Topo statique normalisé
  mask_s2    [10020, 36]      Masque S2
  mask_clim  [10020, 36]      Masque climat (0 partout — GRIDMET sans lacunes)
  y_all, train/val/test_idx
"""

import os, csv
import numpy as np

# ── CHEMINS ───────────────────────────────────────────────────────────────────
CSV_DIR      = r"data\raw\covariables\california"
CDL_DIR      = r"data\raw\cdl\california"
PREPROC_FILE = r"data\processed\CAL_dataset_preprocessed.npz"
OUTPUT_FILE  = r"data\processed\CAL_covariates_v2.npz"
ZONES        = [0, 1]

N_TIMESTEPS = 36
CLIM_COLS   = ['temp_mean', 'vpd_mean', 'solar_mean']   # ← vpd remplace precip
SOIL_COLS   = ['soil_ph', 'soil_oc', 'soil_texture']
TOPO_COLS   = ['elevation', 'landforms']
CLASS_NAMES = {
    0: 'Grapes',
    1: 'Rice',
    2: 'Alfalfa',
    3: 'Almonds',
    4: 'Pistachios',
    5: 'Others'
}
N_CLASSES = len(CLASS_NAMES)

# Préfixes fichiers California
CDL_PREFIX  = 'CAL_CDL'
CLIM_PREFIX = 'CAL_CLIM'
SOIL_PREFIX = 'CAL_SOIL'
TOPO_PREFIX = 'CAL_TOPO'
# ─────────────────────────────────────────────────────────────────────────────


# ─── Clés d'alignement ───────────────────────────────────────────────────────
def load_cdl_keys():
    print("── Clés CDL (alignement) ───────────────────────────────────")
    ordered_keys = []
    for z in ZONES:
        found = None
        for folder in [CDL_DIR, CSV_DIR]:
            p = os.path.join(folder, f"{CDL_PREFIX}_Z{z}.csv")
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"{CDL_PREFIX}_Z{z}.csv introuvable.\n"
                f"  Cherché dans : {CDL_DIR}\n                {CSV_DIR}")
        rows = []
        with open(found, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append((int(row['system:index']), z, row['system:index']))
        rows.sort(key=lambda x: x[0])
        ordered_keys += [(z, r[2]) for r in rows]
        print(f"  Z{z} : {len(rows)} points")
    print(f"  Total : {len(ordered_keys)} points\n")
    return ordered_keys


# ─── Merge climat TEMPOREL ───────────────────────────────────────────────────
def load_climate_timestep(t_idx, zone, key_to_row, X_clim, mask_clim):
    t_str = f"{t_idx + 1:02d}"
    fpath = os.path.join(CSV_DIR, f"{CLIM_PREFIX}_T{t_str}_Z{zone}.csv")
    if not os.path.exists(fpath):
        print(f"  ⚠  {CLIM_PREFIX}_T{t_str}_Z{zone}.csv absent → timestep manquant")
        return 0, 0
    matched, missing = 0, 0
    with open(fpath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (zone, row['system:index'])
            if key not in key_to_row:
                continue
            i = key_to_row[key]
            vals = []
            for col in CLIM_COLS:
                v = row.get(col, '') or ''
                try:
                    vals.append(float(v) if v not in ('', 'null', 'NaN') else 0.0)
                except:
                    vals.append(0.0)
            bands = np.array(vals, dtype=np.float32)
            X_clim[i, t_idx, :] = bands
            # temp=0 impossible en Californie → signal missing
            if bands[0] != 0.0:
                mask_clim[i, t_idx] = 0
                matched += 1
            else:
                missing += 1
    return matched, missing


# ─── Merge sol/topo STATIQUE ─────────────────────────────────────────────────
def load_static_csv(filename, feature_cols):
    fpath = os.path.join(CSV_DIR, filename)
    if not os.path.exists(fpath):
        print(f"  ⚠  {filename} absent")
        return {}
    records = {}
    with open(fpath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vals = []
            for col in feature_cols:
                v = row.get(col, '') or ''
                try:
                    vals.append(float(v) if v not in ('', 'null', 'NaN') else np.nan)
                except:
                    vals.append(np.nan)
            records[row['system:index']] = vals
    print(f"  {filename:28s} : {len(records)} points")
    return records


def build_static_matrix(recs_z0, recs_z1, key_to_row, N, n_feat):
    mat = np.full((N, n_feat), np.nan, dtype=np.float32)
    for z, recs in [(0, recs_z0), (1, recs_z1)]:
        for si, vals in recs.items():
            key = (z, si)
            if key in key_to_row:
                mat[key_to_row[key]] = vals
    return mat


def impute_nans(mat, y):
    mat[mat == -1] = np.nan
    for j in range(mat.shape[1]):
        nan_mask = np.isnan(mat[:, j])
        if nan_mask.sum() == 0:
            continue
        for lbl in range(N_CLASSES):
            sel = (y == lbl) & ~nan_mask
            if sel.sum() > 0:
                med = float(np.median(mat[sel, j]))
                mat[nan_mask & (y == lbl), j] = med
        still = np.isnan(mat[:, j])
        if still.sum() > 0:
            mat[still, j] = float(np.nanmedian(mat[:, j]))
    return mat


# ─── Normalisation ───────────────────────────────────────────────────────────
def normalize_temporal(X, mask, train_idx):
    X_norm = np.zeros_like(X, dtype=np.float32)
    X_tr   = X[train_idx]
    m_tr   = mask[train_idx]
    print(f"    {'Variable':15s} {'min_tr':>10s} {'max_tr':>10s}")
    for b, col in enumerate(CLIM_COLS):
        vals  = X_tr[:, :, b][m_tr == 0]
        b_min = float(vals.min()) if len(vals) > 0 else 0.0
        b_max = float(vals.max()) if len(vals) > 0 else 1.0
        denom = b_max - b_min if b_max > b_min else 1.0
        X_norm[:, :, b] = np.where(
            mask == 0,
            np.clip((X[:, :, b] - b_min) / denom, 0, 1),
            0.0
        )
        print(f"    {col:15s} {b_min:10.3f} {b_max:10.3f}")
    return X_norm


def normalize_static(mat, train_idx, cols):
    mat_norm = np.zeros_like(mat, dtype=np.float32)
    print(f"    {'Feature':15s} {'min_tr':>10s} {'max_tr':>10s}")
    for j, col in enumerate(cols):
        b_min = float(np.nanmin(mat[train_idx, j]))
        b_max = float(np.nanmax(mat[train_idx, j]))
        denom = b_max - b_min if b_max > b_min else 1.0
        mat_norm[:, j] = np.clip((mat[:, j] - b_min) / denom, 0, 1)
        print(f"    {col:15s} {b_min:10.3f} {b_max:10.3f}")
    return mat_norm


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("Part 2 — Étape 1 : Merge covariables California")
    print("=" * 62 + "\n")

    print("── Dataset Part 1 ──────────────────────────────────────────")
    if not os.path.exists(PREPROC_FILE):
        raise FileNotFoundError(f"Fichier manquant : {PREPROC_FILE}")
    pp        = np.load(PREPROC_FILE)
    X_s2_all  = pp['X_all']
    y_all     = pp['y_all']
    mask_s2   = pp['mask_all']
    train_idx = pp['train_idx']
    val_idx   = pp['val_idx']
    test_idx  = pp['test_idx']
    N = len(y_all)
    print(f"  N={N}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n")

    ordered_keys = load_cdl_keys()
    key_to_row   = {k: i for i, k in enumerate(ordered_keys)}
    assert len(ordered_keys) == N, \
        f"❌ {len(ordered_keys)} clés CDL ≠ {N} points dataset"

    print("── Merge Climat TEMPOREL (72 CSV) ──────────────────────────")
    X_clim    = np.zeros((N, N_TIMESTEPS, 3), dtype=np.float32)
    mask_clim = np.ones((N, N_TIMESTEPS), dtype=np.uint8)
    for t in range(N_TIMESTEPS):
        t_str = f"{t+1:02d}"
        tot_m = tot_ms = 0
        for z in ZONES:
            m, ms = load_climate_timestep(t, z, key_to_row, X_clim, mask_clim)
            tot_m += m; tot_ms += ms
        miss_pct = 100 * mask_clim[:, t].mean()
        print(f"  T{t_str} : {tot_m:5d} pts présents | "
              f"{tot_ms:4d} missing ({miss_pct:.1f}%)")
    print(f"\n  Missing global climat : {100*mask_clim.mean():.2f}%  (attendu ~0%)")

    print("\n── Merge Sol & Topo (statiques) ────────────────────────────")
    soil0 = load_static_csv(f'{SOIL_PREFIX}_Z0.csv', SOIL_COLS)
    soil1 = load_static_csv(f'{SOIL_PREFIX}_Z1.csv', SOIL_COLS)
    topo0 = load_static_csv(f'{TOPO_PREFIX}_Z0.csv', TOPO_COLS)
    topo1 = load_static_csv(f'{TOPO_PREFIX}_Z1.csv', TOPO_COLS)

    soil_raw = build_static_matrix(soil0, soil1, key_to_row, N, len(SOIL_COLS))
    topo_raw = build_static_matrix(topo0, topo1, key_to_row, N, len(TOPO_COLS))

    print("\n── Imputation NaN ──────────────────────────────────────────")
    soil_raw = impute_nans(soil_raw, y_all)
    topo_raw = impute_nans(topo_raw, y_all)
    print(f"  NaN résiduels soil : {int(np.isnan(soil_raw).sum())}")
    print(f"  NaN résiduels topo : {int(np.isnan(topo_raw).sum())}")

    print("\n── Conversions sol ─────────────────────────────────────────")
    soil_raw[:, SOIL_COLS.index('soil_ph')] /= 10.0
    soil_raw[:, SOIL_COLS.index('soil_oc')] /= 10.0
    print("  soil_ph ÷ 10 → pH réel  |  soil_oc ÷ 10 → g/kg")

    print("\n── Normalisation min-max (train set) ───────────────────────")
    print("  CLIMAT (temporel) :")
    X_clim_norm = normalize_temporal(X_clim, mask_clim, train_idx)
    print("  SOL :")
    soil_norm = normalize_static(soil_raw, train_idx, SOIL_COLS)
    print("  TOPO :")
    topo_norm = normalize_static(topo_raw, train_idx, TOPO_COLS)

    print("\n── Vérification ────────────────────────────────────────────")
    for name, mat in [('X_clim_norm', X_clim_norm),
                      ('soil_norm',   soil_norm),
                      ('topo_norm',   topo_norm)]:
        n_nan  = int(np.isnan(mat).sum())
        status = "✅" if n_nan == 0 else "❌"
        print(f"  {status} {name:18s} {str(mat.shape):18s} NaN={n_nan}")

    print(f"\n  Distribution classes :")
    for lbl, name in CLASS_NAMES.items():
        print(f"    {name:12s}: {int((y_all == lbl).sum()):5d}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        X_s2_all      = X_s2_all,
        mask_s2       = mask_s2,
        X_clim        = X_clim_norm,
        X_clim_raw    = X_clim,
        mask_clim     = mask_clim,
        X_soil        = soil_norm,
        X_topo        = topo_norm,
        X_soil_raw    = soil_raw,
        X_topo_raw    = topo_raw,
        y_all         = y_all,
        train_idx     = train_idx,
        val_idx       = val_idx,
        test_idx      = test_idx,
        clim_features = np.array(CLIM_COLS),
        soil_features = np.array(SOIL_COLS),
        topo_features = np.array(TOPO_COLS),
    )
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("\n   → Lancer : python Part2_Step2_eda_california.py")


if __name__ == "__main__":
    main()