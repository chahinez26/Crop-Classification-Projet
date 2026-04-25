"""
PART 3 — ÉTAPE 2 : Prétraitement — Early Fusion S2 + Tout
==========================================================
Entrée  : data/merged_npz/arkansas/ARK_covariates.npz
Sortie  : data/merged_npz/arkansas/Part3_ARK_dataset.npz

Stratégie : Early fusion
  • X_all    (N, 36, 10) : bandes S2, normalisées ÷10000 (déjà fait en Part 1)
  • cov_*    (N, k)      : covariables normalisées (déjà faites en Part 2)
  • Répétition covariables sur T=36 → concaténation → X_fused (N, 36, 18)

Vérifications effectuées :
  1. Pas de double normalisation S2
  2. Normalisation des covariables si besoin (z-score)
  3. Cohérence des splits train/val/test
  4. Rapport complet des statistiques
"""

import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_IN  = r"data\merged_npz\arkansas\ARK_covariates.npz"
NPZ_OUT = r"data\merged_npz\arkansas\Part3_ARK_dataset.npz"

CLASS_NAMES = ['Corn', 'Cotton', 'Rice', 'Soybean', 'Others']
N_CLASSES   = 5
N_FEATURES  = 18   # 10 S2 + 3 clim + 3 soil + 2 topo

COV_GROUPS = {
    'Climat': (0, 3,  ['temp_mean', 'precip_total', 'solar_mean']),
    'Sol'   : (3, 6,  ['soil_ph', 'soil_oc', 'soil_texture']),
    'Topo'  : (6, 8,  ['elevation', 'landforms']),
}


# ── Chargement ────────────────────────────────────────────────────────────────
def load_raw():
    if not os.path.exists(NPZ_IN):
        raise FileNotFoundError(f"Fichier introuvable : {NPZ_IN}\n"
                                f"→ S'assurer que ARK_covariates.npz est présent.")
    d = np.load(NPZ_IN, allow_pickle=True)

    X_s2      = d['X_all'].astype(np.float32)      # (N, 36, 10)
    mask      = d['mask_all'].astype(np.float32)    # (N, 36)
    y         = d['y_all'].astype(np.int64)         # (N,)
    cov_clim  = d['cov_clim'].astype(np.float32)   # (N, 3)
    cov_soil  = d['cov_soil'].astype(np.float32)   # (N, 3)
    cov_topo  = d['cov_topo'].astype(np.float32)   # (N, 2)
    train_idx = d['train_idx']
    val_idx   = d['val_idx']   if 'val_idx' in d else None
    test_idx  = d['test_idx']

    print(f"✅ Chargé : N={len(y)}")
    print(f"   X_s2     : {X_s2.shape}   mask: {mask.shape}")
    print(f"   cov_clim : {cov_clim.shape}  cov_soil: {cov_soil.shape}"
          f"  cov_topo: {cov_topo.shape}")
    print(f"   train={len(train_idx)}  "
          f"{'val='+str(len(val_idx)) if val_idx is not None else 'val=None'}  "
          f"test={len(test_idx)}")
    return X_s2, mask, y, cov_clim, cov_soil, cov_topo, train_idx, val_idx, test_idx


# ── Vérification normalisation S2 ─────────────────────────────────────────────
def check_s2_normalization(X_s2, mask):
    """
    S2 doit être dans [0, 1] (÷10000).
    Si les valeurs sont > 1 → probable oubli de division → on corrige.
    """
    X_clean = X_s2.copy()
    X_clean[mask == 1] = np.nan
    vmax = np.nanmax(X_clean)
    vmean = np.nanmean(X_clean)

    print(f"\n── Vérification S2 ─────────────────────────────────")
    print(f"   max={vmax:.4f}  mean={vmean:.4f}")

    if vmax > 2.0:
        print(f"   ⚠ Valeurs > 2.0 détectées → division par 10000")
        X_s2 = X_s2 / 10000.0
        X_clean = X_s2.copy()
        X_clean[mask == 1] = np.nan
        print(f"   Après correction : max={np.nanmax(X_clean):.4f}  "
              f"mean={np.nanmean(X_clean):.4f}")
    else:
        print(f"   ✅ S2 déjà normalisée (valeurs dans [0, ~1])")

    return X_s2


# ── Vérification / normalisation des covariables ─────────────────────────────
def check_and_normalize_covariates(cov_clim, cov_soil, cov_topo, train_idx):
    """
    Vérifie si les covariables sont normalisées (std ≈ 1).
    Si non → z-score sur le train set uniquement (pour éviter le data leakage).
    Retourne les covariables normalisées + les stats de normalisation.
    """
    print(f"\n── Vérification covariables ────────────────────────")
    cov_all = np.concatenate([cov_clim, cov_soil, cov_topo], axis=1)  # (N, 8)
    stds = cov_all[train_idx].std(axis=0)
    print(f"   Std par feature : {stds.round(3)}")

    if stds.max() > 5.0:
        print(f"   ⚠ Covariables non normalisées (std > 5) → z-score sur train")
        mu  = cov_all[train_idx].mean(axis=0)
        std = cov_all[train_idx].std(axis=0) + 1e-8
        cov_all_norm = (cov_all - mu) / std

        cov_clim = cov_all_norm[:, :3]
        cov_soil = cov_all_norm[:, 3:6]
        cov_topo = cov_all_norm[:, 6:8]
        norm_stats = {'mu': mu, 'std': std}
        print(f"   ✅ Normalisé | Std après : "
              f"{cov_all_norm[train_idx].std(axis=0).round(3)}")
    else:
        print(f"   ✅ Covariables déjà normalisées (std ≈ 1)")
        norm_stats = None

    return cov_clim, cov_soil, cov_topo, norm_stats


# ── Création splits val si absent ──────────────────────────────────────────────
def ensure_val_split(train_idx, val_idx, test_idx, y, val_ratio=0.15):
    """Si val_idx est absent, on le découpe depuis train."""
    if val_idx is not None and len(val_idx) > 0:
        return train_idx, val_idx, test_idx

    print(f"\n── Split val manquant → création depuis train ({val_ratio:.0%}) ──")
    rng = np.random.default_rng(42)
    n_val = int(len(train_idx) * val_ratio)
    val_pos = rng.choice(len(train_idx), n_val, replace=False)
    mask_val = np.zeros(len(train_idx), dtype=bool)
    mask_val[val_pos] = True

    new_val   = train_idx[mask_val]
    new_train = train_idx[~mask_val]
    print(f"   train={len(new_train)}  val={len(new_val)}  test={len(test_idx)}")
    return new_train, new_val, test_idx


# ── Early Fusion ──────────────────────────────────────────────────────────────
def build_early_fusion(X_s2, cov_clim, cov_soil, cov_topo):
    """
    Répète les covariables statiques sur T=36 et les concatène avec S2.

    X_s2      : (N, 36, 10)
    cov_*     : (N, k)  statiques
    → X_fused : (N, 36, 18)

    Ordre des features :
      [0:10]  → 10 bandes S2 (temporelles)
      [10:13] → 3 Climat    (répétées)
      [13:16] → 3 Sol       (répétées)
      [16:18] → 2 Topo      (répétées)
    """
    N, T, _ = X_s2.shape
    cov_all  = np.concatenate([cov_clim, cov_soil, cov_topo], axis=1)  # (N, 8)
    cov_rep  = np.repeat(cov_all[:, np.newaxis, :], T, axis=1)          # (N, 36, 8)
    X_fused  = np.concatenate([X_s2, cov_rep], axis=2).astype(np.float32)  # (N, 36, 18)
    return X_fused


# ── Statistiques par split ────────────────────────────────────────────────────
def print_split_stats(X_fused, y, train_idx, val_idx, test_idx):
    print(f"\n── Statistiques des données fusionnées ─────────────")
    print(f"   Shape X_fused : {X_fused.shape}  [N, T=36, F=18]")
    print(f"   dtype         : {X_fused.dtype}")
    print(f"   min / max     : {X_fused.min():.4f} / {X_fused.max():.4f}")
    print(f"   NaN count     : {np.isnan(X_fused).sum()}")

    print(f"\n   Splits & classes :")
    for sp, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        counts = [(y[idx] == i).sum() for i in range(N_CLASSES)]
        print(f"   {sp:5s} ({len(idx):5,}) : "
              + "  ".join(f"{CLASS_NAMES[i][:4]}={counts[i]}" for i in range(N_CLASSES)))

    print(f"\n   Stats par groupe de features :")
    feat_labels = ['S2(10)', 'Clim(3)', 'Sol(3)', 'Topo(2)']
    slices = [(0,10), (10,13), (13,16), (16,18)]
    for label, (s, e) in zip(feat_labels, slices):
        sub = X_fused[:, :, s:e]
        print(f"     {label:10s} : mean={sub.mean():.4f}  "
              f"std={sub.std():.4f}  min={sub.min():.4f}  max={sub.max():.4f}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_dataset(X_fused, y, mask, train_idx, val_idx, test_idx, norm_stats):
    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    save_dict = {
        'X'         : X_fused,      # (N, 36, 18) — features fusionnées
        'y'         : y,             # (N,) — labels entiers
        'mask'      : mask,          # (N, 36) — masque nuageux
        'train_idx' : train_idx,
        'val_idx'   : val_idx,
        'test_idx'  : test_idx,
        'n_features': np.array(18),
        'n_classes' : np.array(5),
        'feature_order': np.array(
            ['S2_B2','S2_B3','S2_B4','S2_B5','S2_B6',
             'S2_B7','S2_B8','S2_B8A','S2_B11','S2_B12',
             'Clim_T','Clim_P','Clim_R',
             'Sol_pH','Sol_OC','Sol_Tex',
             'Topo_E','Topo_L']
        )
    }
    if norm_stats is not None:
        save_dict['cov_mu']  = norm_stats['mu']
        save_dict['cov_std'] = norm_stats['std']

    np.savez_compressed(NPZ_OUT, **save_dict)
    size_mb = os.path.getsize(NPZ_OUT) / 1e6
    print(f"\n✅ Sauvegardé : {NPZ_OUT}  ({size_mb:.1f} MB)")


# ── Rapport de prétraitement ──────────────────────────────────────────────────
def print_preprocessing_report(X_fused, y, train_idx, val_idx, test_idx):
    n = len(y)
    print("\n" + "="*60)
    print("RAPPORT PRÉTRAITEMENT — Part 3 — Arkansas")
    print("="*60)
    print(f"""
  Source       : ARK_covariates.npz (Part 2)
  Stratégie    : Early Fusion (S2 + covariables répétées)

  Input model  : (N, T=36, F=18)
  ┌─────────────────────────────────────────┐
  │  Features [0:10]  → 10 bandes S2        │
  │  Features [10:13] → 3 Climat (répétées) │
  │  Features [13:16] → 3 Sol    (répétées) │
  │  Features [16:18] → 2 Topo   (répétées) │
  └─────────────────────────────────────────┘

  Train : {len(train_idx):,} points  ({len(train_idx)/n*100:.0f}%)
  Val   : {len(val_idx):,} points    ({len(val_idx)/n*100:.0f}%)
  Test  : {len(test_idx):,} points   ({len(test_idx)/n*100:.0f}%)

  Normalisation S2   : ÷10000 (vérifiée, Part 1)
  Normalisation Cov  : z-score sur train (si besoin)
  Pas de data leakage : stats calculées sur train uniquement
""")
    print("  → Lancer : python Part3_Step3_model.py")
    print("="*60)


def main():
    print("="*60)
    print("Part 3 — Étape 2 : Prétraitement Early Fusion")
    print("="*60 + "\n")

    # 1. Chargement
    X_s2, mask, y, cov_clim, cov_soil, cov_topo, \
        train_idx, val_idx, test_idx = load_raw()

    # 2. Vérification normalisation S2
    X_s2 = check_s2_normalization(X_s2, mask)

    # 3. Vérification / normalisation covariables
    cov_clim, cov_soil, cov_topo, norm_stats = \
        check_and_normalize_covariates(cov_clim, cov_soil, cov_topo, train_idx)

    # 4. Gestion val split
    train_idx, val_idx, test_idx = \
        ensure_val_split(train_idx, val_idx, test_idx, y)

    # 5. Early fusion
    print(f"\n── Early Fusion ────────────────────────────────────")
    X_fused = build_early_fusion(X_s2, cov_clim, cov_soil, cov_topo)
    print(f"   X_fused : {X_fused.shape}  [N, T=36, F=18] ✅")

    # 6. Stats complètes
    print_split_stats(X_fused, y, train_idx, val_idx, test_idx)

    # 7. Sauvegarde
    save_dataset(X_fused, y, mask, train_idx, val_idx, test_idx, norm_stats)

    # 8. Rapport final
    print_preprocessing_report(X_fused, y, train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()
