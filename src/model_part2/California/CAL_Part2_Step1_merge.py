"""
Entrée  : CAL_dataset_preprocessed.npz + 6 CSV GEE
Sortie  : CAL_covariates.npz

Colonnes CSV réelles (vérifiées) :
  CLIMATE : temp_mean (°C), precip_total (mm), solar_mean (W/m²)
  SOIL    : soil_ph (×10 brut → ÷10), soil_oc (brut dg/kg → ÷10), soil_texture (1-12, -1=null)
  TOPO    : elevation (m), landforms (21-42)

Alignement : system:index identique entre tous les CSV (seed=42 garanti)

6 classes California :
  Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
"""

import os, json, csv
import numpy as np

# ── CHEMINS — adapter si nécessaire ────────────────────────────────────────
CSV_DIR      = r"data\raw\covariables\california"
PREPROC_FILE = r"data\processed\CAL_dataset_preprocessed.npz"
RAW_FILE     = r"data\merged_npz\california\CAL_dataset.npz"
OUTPUT_FILE  = r"data\merged_npz\california\CAL_covariates.npz"
ZONES        = [0, 1]

CLIMATE_COLS = ['temp_mean', 'precip_total', 'solar_mean']
SOIL_COLS    = ['soil_ph', 'soil_oc', 'soil_texture']
TOPO_COLS    = ['elevation', 'landforms']
ALL_COLS     = CLIMATE_COLS + SOIL_COLS + TOPO_COLS   # 8 features

CLASS_NAMES  = {0:'Grapes', 1:'Rice', 2:'Alfalfa',
                3:'Almonds', 4:'Pistachios', 5:'Others'}
N_CLASSES    = 6
# ───────────────────────────────────────────────────────────────────────────


def parse_geo(geo_str):
    try:
        g = json.loads(geo_str)
        return float(g['coordinates'][0]), float(g['coordinates'][1])
    except:
        return 0.0, 0.0


def load_csv(filename, feature_cols):
    """Charge un CSV GEE → dict {system:index → {col: val, _lon, _lat}}"""
    fpath = os.path.join(CSV_DIR, filename)
    if not os.path.exists(fpath):
        print(f"  ⚠  {filename} absent — vérifier CSV_DIR")
        return {}
    records = {}
    with open(fpath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lon, lat = parse_geo(row.get('.geo', '{}'))
            vals = {'_lon': lon, '_lat': lat,
                    '_label': int(row.get('crop_label', -1))}
            for col in feature_cols:
                try:
                    v = row.get(col, '')
                    vals[col] = float(v) if v not in ('', 'null', 'NaN', None) else np.nan
                except:
                    vals[col] = np.nan
            records[row['system:index']] = vals
    print(f"  {filename:25s} : {len(records)} points")
    return records


def build_matrix_by_coords(recs_z0, recs_z1, feature_cols, lons, lats):
    """Alignement par KD-tree sur coordonnées lon/lat."""
    from scipy.spatial import cKDTree
    all_recs = {**recs_z0, **recs_z1}
    if not all_recs:
        return np.zeros((len(lons), len(feature_cols)), dtype=np.float32)

    cov_lons = np.array([v['_lon'] for v in all_recs.values()])
    cov_lats = np.array([v['_lat'] for v in all_recs.values()])
    cov_vals = np.array([[v.get(c, np.nan) for c in feature_cols]
                          for v in all_recs.values()], dtype=np.float32)

    tree = cKDTree(np.column_stack([cov_lons, cov_lats]))
    dists, idxs = tree.query(np.column_stack([lons, lats]), k=1)
    result = cov_vals[idxs]
    print(f"    dist_max={dists.max():.5f}°  dist_med={np.median(dists):.6f}°"
          f"  n_nan={int(np.isnan(result).any(1).sum())}")
    return result


def impute_nans(mat, y, cols):
    """Imputation NaN + valeurs -1 (nulls GEE) par médiane de classe."""
    mat[mat == -1] = np.nan
    n_imp = 0
    for j in range(mat.shape[1]):
        nan_mask = np.isnan(mat[:, j])
        if nan_mask.sum() == 0:
            continue
        for lbl in range(N_CLASSES):
            sel = (y == lbl) & ~nan_mask
            if sel.sum() > 0:
                med = float(np.median(mat[sel, j]))
                target = nan_mask & (y == lbl)
                mat[target, j] = med
                n_imp += int(target.sum())
        still_nan = np.isnan(mat[:, j])
        if still_nan.sum() > 0:
            mat[still_nan, j] = float(np.nanmedian(mat[:, j]))
            n_imp += int(still_nan.sum())
    if n_imp > 0:
        print(f"    {n_imp} valeurs imputées par médiane de classe")
    return mat


def normalize(mat, train_idx, cols):
    """Min-max sur train set uniquement."""
    mat_norm = np.zeros_like(mat, dtype=np.float32)
    print(f"    {'Feature':15s} {'min_tr':>10s} {'max_tr':>10s}")
    for j, col in enumerate(cols):
        vmin = float(np.nanmin(mat[train_idx, j]))
        vmax = float(np.nanmax(mat[train_idx, j]))
        denom = vmax - vmin if vmax > vmin else 1.0
        mat_norm[:, j] = np.clip((mat[:, j] - vmin) / denom, 0, 1)
        print(f"    {col:15s} {vmin:10.3f} {vmax:10.3f}")
    return mat_norm


def main():
    print("=" * 60)
    print("Part 2 — Étape 1 : Merge covariables — California")
    print("=" * 60 + "\n")

    # 1. Charger dataset Part 1
    print("── Dataset Part 1 ─────────────────────────────────────────")
    pp = np.load(PREPROC_FILE)
    X_all, y_all, mask_all = pp['X_all'], pp['y_all'], pp['mask_all']
    train_idx, val_idx, test_idx = pp['train_idx'], pp['val_idx'], pp['test_idx']
    N = len(y_all)
    print(f"  N={N}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 2. Coordonnées depuis CAL_dataset.npz
    print("\n── Coordonnées ────────────────────────────────────────────")
    for path in [RAW_FILE,
                 RAW_FILE.replace('npz\\', ''),
                 os.path.join(os.path.dirname(PREPROC_FILE), 'CAL_dataset.npz')]:
        if os.path.exists(path):
            raw = np.load(path)
            lons, lats = raw['lons'], raw['lats']
            print(f"  Chargé depuis {path}")
            print(f"  lon [{lons.min():.2f}, {lons.max():.2f}]  "
                  f"lat [{lats.min():.2f}, {lats.max():.2f}]")
            break
    else:
        print("  ⚠ CAL_dataset.npz non trouvé — coordonnées zéro (KD-tree dégradé)")
        lons = np.zeros(N, dtype=np.float64)
        lats = np.zeros(N, dtype=np.float64)

    # 3. Charger les 6 CSV
    print("\n── Chargement CSV ─────────────────────────────────────────")
    clim0 = load_csv('CAL_CLIMATE_Z0.csv', CLIMATE_COLS)
    clim1 = load_csv('CAL_CLIMATE_Z1.csv', CLIMATE_COLS)
    soil0 = load_csv('CAL_SOIL_Z0.csv',    SOIL_COLS)
    soil1 = load_csv('CAL_SOIL_Z1.csv',    SOIL_COLS)
    topo0 = load_csv('CAL_TOPO_Z0.csv',    TOPO_COLS)
    topo1 = load_csv('CAL_TOPO_Z1.csv',    TOPO_COLS)

    # 4. Matrices brutes
    print("\n── Alignement KD-tree ─────────────────────────────────────")
    print("  CLIMAT :")
    clim_raw = build_matrix_by_coords(clim0, clim1, CLIMATE_COLS, lons, lats)
    print("  SOL :")
    soil_raw = build_matrix_by_coords(soil0, soil1, SOIL_COLS,    lons, lats)
    print("  TOPO :")
    topo_raw = build_matrix_by_coords(topo0, topo1, TOPO_COLS,    lons, lats)

    # 5. Imputation NaN / -1
    print("\n── Imputation ─────────────────────────────────────────────")
    clim_raw = impute_nans(clim_raw, y_all, CLIMATE_COLS)
    soil_raw = impute_nans(soil_raw, y_all, SOIL_COLS)
    topo_raw = impute_nans(topo_raw, y_all, TOPO_COLS)

    # 6. Conversions unités
    print("\n── Conversions unités ─────────────────────────────────────")
    soil_raw[:, SOIL_COLS.index('soil_ph')] /= 10.0
    soil_raw[:, SOIL_COLS.index('soil_oc')] /= 10.0
    print("  soil_ph ÷ 10 → pH réel  |  soil_oc ÷ 10 → g/kg")

    # 7. Statistiques brutes
    print("\n── Statistiques brutes ────────────────────────────────────")
    all_raw = np.concatenate([clim_raw, soil_raw, topo_raw], axis=1)
    print(f"  {'Feature':15s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>7s}")
    print(f"  {'-'*52}")
    for j, col in enumerate(ALL_COLS):
        v = all_raw[:, j]
        print(f"  {col:15s} {v.min():8.2f} {v.max():8.2f} "
              f"{v.mean():8.2f} {v.std():7.2f}")

    # 8. Normalisation min-max (train set)
    print("\n── Normalisation min-max (train set) ──────────────────────")
    print("  CLIMAT :")
    cov_clim = normalize(clim_raw, train_idx, CLIMATE_COLS)
    print("  SOL :")
    cov_soil = normalize(soil_raw, train_idx, SOIL_COLS)
    print("  TOPO :")
    cov_topo = normalize(topo_raw, train_idx, TOPO_COLS)
    cov_all  = np.concatenate([cov_clim, cov_soil, cov_topo], axis=1)

    # 9. Vérification
    print("\n── Vérification ───────────────────────────────────────────")
    for name, mat in [('cov_clim', cov_clim), ('cov_soil', cov_soil),
                       ('cov_topo', cov_topo), ('cov_all',  cov_all)]:
        n_nan = int(np.isnan(mat).sum())
        print(f"  {name} {str(mat.shape):12s} "
              f"[{mat.min():.3f}, {mat.max():.3f}]  NaN={n_nan}")
    assert not np.isnan(cov_all).any(), "❌ NaN résiduels!"
    print("  ✅ Aucun NaN résiduel")

    # 10. Distribution classes
    print("\n  Distribution :")
    for lbl, name in CLASS_NAMES.items():
        print(f"    {name:10s}: {int((y_all==lbl).sum()):5d}")

    # 11. Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        X_all=X_all, y_all=y_all, mask_all=mask_all,
        cov_clim=cov_clim, cov_soil=cov_soil,
        cov_topo=cov_topo, cov_all=cov_all,
        cov_clim_raw=clim_raw, cov_soil_raw=soil_raw, cov_topo_raw=topo_raw,
        clim_features=np.array(CLIMATE_COLS),
        soil_features=np.array(SOIL_COLS),
        topo_features=np.array(TOPO_COLS),
        all_features=np.array(ALL_COLS),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        lons=lons, lats=lats,
    )
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\n✅ Sauvegardé : {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("   → Lancer : python CAL_Part2_Step2_eda_covariates.py")


if __name__ == "__main__":
    main()
