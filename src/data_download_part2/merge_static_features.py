"""
merge_static_features.py — Fusion CSV Soil/Climate/Topo → *_dataset_static.npz
================================================================================
Ce script fusionne les 12 CSV GEE (6 par région) en deux fichiers NPZ :
  ARK_dataset_static.npz  →  X_static[10000, 17]  float32
  CAL_dataset_static.npz  →  X_static[10020, 17]  float32

STRUCTURE DES 17 VARIABLES (même ordre pour les deux régions) :
  Index  0 :  sol_texture_class   (USDA texture, 1-12)
  Index  1 :  sol_sand_pct        (% sable, 0-100)
  Index  2 :  sol_clay_pct        (% argile, 0-100)
  Index  3 :  sol_organic_carbon  (dg/kg)
  Index  4 :  sol_ph_h2o          (pH × 10)
  Index  5 :  sol_bulk_density    (cg/cm³)
  Index  6 :  clim_tmean          (°C × 10)
  Index  7 :  clim_prec           (mm/an)
  Index  8 :  clim_prec_seas      (%, saisonnalité)
  Index  9 :  clim_tmax           (°C × 10)
  Index 10 :  clim_tmin           (°C × 10)
  Index 11 :  clim_et_annual      (mm/an, MODIS MOD16)
  Index 12 :  topo_elevation      (m)
  Index 13 :  topo_slope          (degrés)
  Index 14 :  topo_aspect         (degrés)
  Index 15 :  topo_tpi            (m, Topographic Position Index)
  Index 16 :  topo_twi            (adimensionnel, Topographic Wetness Index)

COMPATIBILITÉ avec les datasets Sentinel existants :
  L'ordre des points est GARANTI identique car :
    - Même seed=42 dans stratifiedSample GEE
    - Même scale=30
    - Même region (GEOM)
    - Même classValues et classPoints
  → X_static[i] correspond à X[i, :, :] dans ARK_dataset.npz / CAL_dataset.npz

UTILISATION :
  python merge_static_features.py
  → Lit les CSV depuis INPUT_DIR_ARK et INPUT_DIR_CAL
  → Produit ARK_dataset_static.npz et CAL_dataset_static.npz

INTÉGRATION DANS MCTNET :
  data_ark = np.load('ARK_dataset.npz')
  stat_ark = np.load('ARK_dataset_static.npz')
  X_temporal = data_ark['X']          # [10000, 36, 10]
  X_static   = stat_ark['X_static']   # [10000, 17]
  y          = data_ark['y']           # [10000]
  # Concaténer ou utiliser comme branche séparée dans MCTNet
"""

import os
import csv
import json
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

# Arkansas
INPUT_DIR_ARK   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5"
OUTPUT_FILE_ARK = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_v5\npz\ARK_dataset_static.npz"

# California
INPUT_DIR_CAL   = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2"
OUTPUT_FILE_CAL = r"C:\Users\Stux\OneDrive\Bureau\projet_reseau\MCTNet_California_v2\npz\CAL_dataset_static.npz"

# ── Variables dans l'ordre fixe ──────────────────────────────────────────

SOIL_BANDS = [
    'sol_texture_class',
    'sol_sand_pct',
    'sol_clay_pct',
    'sol_organic_carbon',
    'sol_ph_h2o',
    'sol_bulk_density',
]
CLIM_BANDS = [
    'clim_tmean',
    'clim_prec',
    'clim_prec_seas',
    'clim_tmax',
    'clim_tmin',
    'clim_et_annual',
]
TOPO_BANDS = [
    'topo_elevation',
    'topo_slope',
    'topo_aspect',
    'topo_tpi',
    'topo_twi',
]
ALL_BANDS = SOIL_BANDS + CLIM_BANDS + TOPO_BANDS   # 17 variables
N_STATIC  = len(ALL_BANDS)                          # 17

# ── Configurations par région ─────────────────────────────────────────────

REGIONS = {
    'ARK': {
        'input_dir'     : INPUT_DIR_ARK,
        'output_file'   : OUTPUT_FILE_ARK,
        'prefix'        : 'ARK',
        'n_zones'       : 2,
        'class_values'  : [0, 1, 2, 3, 4],
        'class_points'  : {0: [760, 760], 1: [380, 380],
                           2: [1210, 1210], 3: [2340, 2340], 4: [310, 310]},
        # class_points[label][zone] = nb pts attendus dans cette zone
        'n_total_expected': 10000,
        'class_names'   : {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'},
        'cdl_prefix'    : 'ARK_CDL',
    },
    'CAL': {
        'input_dir'     : INPUT_DIR_CAL,
        'output_file'   : OUTPUT_FILE_CAL,
        'prefix'        : 'CAL',
        'n_zones'       : 2,
        'class_values'  : [0, 1, 2, 3, 4, 5],
        'class_points'  : {0: [1030, 1030], 1: [2030, 10],
                           2: [490, 490],   3: [390, 390],
                           4: [20, 620],    5: [1760, 1760]},
        'n_total_expected': 10020,
        'class_names'   : {0:'Grapes', 1:'Rice', 2:'Alfalfa',
                           3:'Almonds', 4:'Pistachios', 5:'Others'},
        'cdl_prefix'    : 'CAL_CDL',
    },
}


# ── Utilitaires ───────────────────────────────────────────────────────────

def parse_geo(geo_str):
    """Extrait (lon, lat) depuis la colonne .geo de GEE."""
    g = json.loads(geo_str)
    return float(g['coordinates'][0]), float(g['coordinates'][1])


def load_cdl_ref(cfg):
    """
    Charge les fichiers CDL des N zones.
    Retourne :
      ref          : dict (zone, system:index) -> {label, lon, lat, zone}
      ordered_keys : liste de clés dans l'ordre stable (Z0 trié, puis Z1 trié)
      key_to_row   : dict clé -> indice ligne dans X_static
    """
    prefix = cfg['cdl_prefix']
    ref    = {}

    for z in range(cfg['n_zones']):
        fpath = os.path.join(cfg['input_dir'], f"{prefix}_Z{z}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Fichier CDL manquant : {fpath}\n"
                f"Assurez-vous d'avoir lancé le script GEE CDL_SAMPLE pour Z{z}."
            )
        with open(fpath, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (z, row['system:index'])
                lon, lat = parse_geo(row['.geo'])
                ref[key] = {
                    'label': int(row['crop_label']),
                    'lon'  : lon,
                    'lat'  : lat,
                    'zone' : z,
                }
        n_z = sum(1 for k in ref if k[0] == z)
        print(f"  Z{z} : {n_z} points CDL chargés")

    # Ordre stable : Z0 trié par system:index, puis Z1 trié
    ordered_keys = []
    for z in range(cfg['n_zones']):
        zone_keys = sorted(
            [(int(k[1]), k) for k in ref if k[0] == z]
        )
        ordered_keys += [k for _, k in zone_keys]

    key_to_row = {k: i for i, k in enumerate(ordered_keys)}
    print(f"  Total : {len(ref)} points\n")
    return ref, ordered_keys, key_to_row


def load_static_csv(fpath, key_to_row, X_static, band_names, col_start):
    """
    Remplit X_static[:, col_start:col_start+len(band_names)]
    depuis un CSV GEE (SOIL, CLIMATE ou TOPO).

    Retourne (n_matched, n_missing_keys, n_null_vals).
    """
    if not os.path.exists(fpath):
        print(f"    ⚠  Absent : {os.path.basename(fpath)}")
        return 0, 0, 0

    n_matched, n_missing, n_null = 0, 0, 0

    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Identifier le point via (zone, system:index)
            # La zone est encodée dans le préfixe du nom de fichier (Z0/Z1)
            fname = os.path.basename(fpath)
            zone  = int(fname[fname.index('_Z') + 2])
            key   = (zone, row['system:index'])

            if key not in key_to_row:
                n_missing += 1
                continue

            i = key_to_row[key]
            for j, band in enumerate(band_names):
                val = row.get(band, None)
                if val is None or val == '':
                    n_null += 1
                    # Garder 0 (déjà initialisé)
                else:
                    X_static[i, col_start + j] = float(val)
            n_matched += 1

    return n_matched, n_missing, n_null


def print_static_stats(X_static, band_names):
    """Affiche moyenne ± std pour chaque variable statique."""
    print(f"\n  {'Variable':<25s} {'Moy':>10s} {'Std':>10s} "
          f"{'Min':>10s} {'Max':>10s}  Nulls")
    print("  " + "-" * 75)
    for j, name in enumerate(band_names):
        col    = X_static[:, j]
        nulls  = int((col == 0).sum())
        null_p = 100 * nulls / len(col)
        print(f"  {name:<25s} {col.mean():10.3f} {col.std():10.3f} "
              f"{col.min():10.3f} {col.max():10.3f}  "
              f"{nulls:5d} ({null_p:.1f}%)")


def process_region(region_name, cfg):
    """Traitement complet d'une région."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RÉGION : {region_name}")
    print(sep)
    print()

    input_dir   = cfg['input_dir']
    output_file = cfg['output_file']
    prefix      = cfg['prefix']
    n_zones     = cfg['n_zones']

    if not os.path.isdir(input_dir):
        print(f"  ❌ Dossier introuvable : {input_dir}")
        print(f"     Placez les CSV GEE dans ce dossier puis relancez.")
        return

    # 1. Charger les labels CDL (ordre de référence)
    print("── Chargement CDL (référence) ──────────────────────")
    ref, ordered_keys, key_to_row = load_cdl_ref(cfg)
    N = len(ordered_keys)

    # 2. Initialiser la matrice statique
    X_static = np.zeros((N, N_STATIC), dtype=np.float32)
    y        = np.array([ref[k]['label'] for k in ordered_keys], dtype=np.int32)
    lons     = np.array([ref[k]['lon']   for k in ordered_keys], dtype=np.float64)
    lats     = np.array([ref[k]['lat']   for k in ordered_keys], dtype=np.float64)

    # 3. Remplir SOIL (colonnes 0-5)
    print("── Chargement SOIL ─────────────────────────────────")
    total_soil = 0
    for z in range(n_zones):
        fpath = os.path.join(input_dir, f"{prefix}_SOIL_Z{z}.csv")
        matched, miss, null_v = load_static_csv(
            fpath, key_to_row, X_static, SOIL_BANDS, col_start=0
        )
        print(f"  Z{z} : {matched} pts matchés | {miss} clés inconnues | {null_v} valeurs nulles")
        total_soil += matched
    print(f"  Total sol : {total_soil} pts remplis\n")

    # 4. Remplir CLIMATE (colonnes 6-11)
    print("── Chargement CLIMATE ──────────────────────────────")
    total_clim = 0
    for z in range(n_zones):
        fpath = os.path.join(input_dir, f"{prefix}_CLIM_Z{z}.csv")
        matched, miss, null_v = load_static_csv(
            fpath, key_to_row, X_static, CLIM_BANDS, col_start=6
        )
        print(f"  Z{z} : {matched} pts matchés | {miss} clés inconnues | {null_v} valeurs nulles")
        total_clim += matched
    print(f"  Total climat : {total_clim} pts remplis\n")

    # 5. Remplir TOPO (colonnes 12-16)
    print("── Chargement TOPO ─────────────────────────────────")
    total_topo = 0
    for z in range(n_zones):
        fpath = os.path.join(input_dir, f"{prefix}_TOPO_Z{z}.csv")
        matched, miss, null_v = load_static_csv(
            fpath, key_to_row, X_static, TOPO_BANDS, col_start=12
        )
        print(f"  Z{z} : {matched} pts matchés | {miss} clés inconnues | {null_v} valeurs nulles")
        total_topo += matched
    print(f"  Total topo : {total_topo} pts remplis\n")

    # 6. Résumé statistique
    print("── Résumé des variables statiques ──────────────────")
    print_static_stats(X_static, ALL_BANDS)

    # 7. Distribution des classes
    print(f"\n── Distribution des classes ─────────────────────────")
    class_names = cfg['class_names']
    for lbl, name in class_names.items():
        n_cls = int((y == lbl).sum())
        pct   = 100 * n_cls / N
        print(f"  {name:<12s} : {n_cls:6d} pts ({pct:.1f}%)")
    print(f"  {'TOTAL':<12s} : {N:6d} pts")

    # 8. Sauvegarder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        X_static=X_static,
        y=y,
        lons=lons,
        lats=lats,
        band_names=np.array(ALL_BANDS),
    )
    size_mb = os.path.getsize(output_file) / 1e6
    print(f"\n  ✅ Sauvegardé : {output_file}  ({size_mb:.2f} MB)")

    # 9. Instructions d'utilisation
    print(f"\n── Utilisation dans MCTNet ──────────────────────────")
    print(f"  import numpy as np")
    print(f"  # Dataset Sentinel temporel (déjà créé)")
    print(f"  data   = np.load('{prefix}_dataset.npz')")
    print(f"  # Dataset statique (ce fichier)")
    print(f"  static = np.load('{prefix}_dataset_static.npz')")
    print(f"  X_temporal = data['X']           # {{{N}, 36, 10}}")
    print(f"  X_static   = static['X_static']  # {{{N}, 17}}")
    print(f"  y          = data['y']            # {{{N}}}")
    print(f"  mask       = data['mask']         # {{{N}, 36}}")
    print(f"  # Les indices i sont identiques entre les deux NPZ ✅")
    print(f"")
    print(f"  # Variables disponibles :")
    for j, name in enumerate(ALL_BANDS):
        group = "SOL  " if j < 6 else ("CLIM " if j < 12 else "TOPO ")
        print(f"  #   [{j:2d}] {group}  {name}")


def main():
    print("=" * 60)
    print("  merge_static_features.py")
    print("  Soil / Climate / Topography → NPZ")
    print("=" * 60)

    # Traiter les deux régions
    for region_name, cfg in REGIONS.items():
        process_region(region_name, cfg)

    print("\n" + "=" * 60)
    print("  TERMINÉ")
    print("  ARK_dataset_static.npz  →  X_static[10000, 17]")
    print("  CAL_dataset_static.npz  →  X_static[10020, 17]")
    print("=" * 60)


if __name__ == "__main__":
    main()
