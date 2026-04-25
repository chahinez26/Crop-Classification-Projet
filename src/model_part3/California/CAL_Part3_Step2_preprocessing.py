"""
PART 3 — ÉTAPE 2 (CALIFORNIA) : Prétraitement — Early Fusion S2 + Sol
======================================================================
Entrée  : data/merged_npz/california/CAL_covariates.npz
Sortie  : data/merged_npz/california/Part3_CAL_dataset.npz

Stratégie : Early fusion S2 + Sol (meilleure config ablation)
  • X_all   (N, 36, 10) : bandes S2  (normalisées ÷10000 en Part 1)
  • cov_soil (N, 3)     : pH, OC, Texture  (normalisées en Part 2)
  • Répétition Sol sur T=36 → concaténation → X_fused (N, 36, 13)

6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
"""

import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_IN  = r"data\merged_npz\california\CAL_covariates.npz"
NPZ_OUT = r"data\merged_npz\california\Part3_CAL_dataset.npz"

CLASS_NAMES = ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others']
N_CLASSES   = 6
N_FEATURES  = 13   # 10 S2 + 3 sol


# ── Chargement ────────────────────────────────────────────────────────────────
def load_raw():
    if not os.path.exists(NPZ_IN):
        raise FileNotFoundError(
            f"Fichier introuvable : {NPZ_IN}\n"
            "→ S'assurer que CAL_covariates.npz est présent."
        )
    d = np.load(NPZ_IN, allow_pickle=True)

    X_s2      = d['X_all'].astype(np.float32)      # (N, 36, 10)
    mask      = d['mask_all'].astype(np.float32)    # (N, 36)
    y         = d['y_all'].astype(np.int64)         # (N,)
    cov_soil  = d['cov_soil'].astype(np.float32)   # (N, 3) — pH, OC, Texture
    train_idx = d['train_idx']
    val_idx   = d['val_idx']   if 'val_idx' in d else None
    test_idx  = d['test_idx']

    print(f"✅ Chargé : N={len(y)}")
    print(f"   X_s2     : {X_s2.shape}   mask: {mask.shape}")
    print(f"   cov_soil : {cov_soil.shape}  (pH, OC, Texture)")
    print(f"   train={len(train_idx)}  "
          f"{'val='+str(len(val_idx)) if val_idx is not None else 'val=None'}  "
          f"test={len(test_idx)}")
    print(f"   Classes : "
          f"{ {CLASS_NAMES[i]: int((y==i).sum()) for i in range(N_CLASSES)} }")
    return X_s2, mask, y, cov_soil, train_idx, val_idx, test_idx


# ── Vérification normalisation S2 ─────────────────────────────────────────────
def check_s2_normalization(X_s2, mask):
    X_clean = X_s2.copy()
    X_clean[mask == 1] = np.nan
    vmax  = np.nanmax(X_clean)
    vmean = np.nanmean(X_clean)

    print(f"\n── Vérification S2 ─────────────────────────────────")
    print(f"   max={vmax:.4f}  mean={vmean:.4f}")

    if vmax > 2.0:
        print(f"   ⚠ Valeurs > 2.0 → division par 10000")
        X_s2 = X_s2 / 10000.0
        X_clean = X_s2.copy(); X_clean[mask == 1] = np.nan
        print(f"   Après correction : max={np.nanmax(X_clean):.4f}")
    else:
        print(f"   ✅ S2 déjà normalisée")
    return X_s2


# ── Vérification / normalisation Sol ──────────────────────────────────────────
def check_and_normalize_soil(cov_soil, train_idx):
    print(f"\n── Vérification covariables Sol ────────────────────")
    stds = cov_soil[train_idx].std(axis=0)
    print(f"   Std [pH, OC, Texture] : {stds.round(3)}")

    norm_stats = None
    if stds.max() > 5.0:
        print(f"   ⚠ Non normalisées (std > 5) → z-score sur train")
        mu  = cov_soil[train_idx].mean(axis=0)
        std = cov_soil[train_idx].std(axis=0) + 1e-8
        cov_soil = (cov_soil - mu) / std
        norm_stats = {'mu': mu, 'std': std}
        print(f"   ✅ Std après : {cov_soil[train_idx].std(axis=0).round(3)}")
    else:
        print(f"   ✅ Covariables Sol déjà normalisées")
    return cov_soil, norm_stats


# ── Gestion split val ─────────────────────────────────────────────────────────
def ensure_val_split(train_idx, val_idx, test_idx, y, val_ratio=0.15):
    if val_idx is not None and len(val_idx) > 0:
        return train_idx, val_idx, test_idx

    print(f"\n── Split val manquant → création depuis train ({val_ratio:.0%}) ──")
    rng    = np.random.default_rng(42)
    n_val  = int(len(train_idx) * val_ratio)
    pos    = rng.choice(len(train_idx), n_val, replace=False)
    mv     = np.zeros(len(train_idx), dtype=bool); mv[pos] = True
    new_val   = train_idx[mv]
    new_train = train_idx[~mv]
    print(f"   train={len(new_train)}  val={len(new_val)}  test={len(test_idx)}")
    return new_train, new_val, test_idx


# ── Early Fusion S2 + Sol ─────────────────────────────────────────────────────
def build_early_fusion(X_s2, cov_soil):
    """
    X_s2     : (N, 36, 10)
    cov_soil : (N, 3)  — statique
    → X_fused : (N, 36, 13)

    Ordre des features :
      [0:10]  → 10 bandes S2 (temporelles)
      [10:13] → 3 Sol — pH, OC, Texture (répétées sur T=36)
    """
    N, T, _ = X_s2.shape
    cov_rep  = np.repeat(cov_soil[:, np.newaxis, :], T, axis=1)   # (N, 36, 3)
    X_fused  = np.concatenate([X_s2, cov_rep], axis=2).astype(np.float32)
    return X_fused   # (N, 36, 13)


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_split_stats(X_fused, y, train_idx, val_idx, test_idx):
    print(f"\n── Statistiques des données fusionnées ─────────────")
    print(f"   Shape X_fused : {X_fused.shape}  [N, T=36, F=13]")
    print(f"   dtype         : {X_fused.dtype}")
    print(f"   min / max     : {X_fused.min():.4f} / {X_fused.max():.4f}")
    print(f"   NaN count     : {np.isnan(X_fused).sum()}")

    print(f"\n   Splits & classes :")
    for sp, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        counts = [(y[idx] == i).sum() for i in range(N_CLASSES)]
        line   = "  ".join(f"{CLASS_NAMES[i][:4]}={counts[i]}" for i in range(N_CLASSES))
        print(f"   {sp:5s} ({len(idx):5,}) : {line}")

    print(f"\n   Stats par groupe :")
    for label, (s, e) in [('S2(10)', (0,10)), ('Sol(3)', (10,13))]:
        sub = X_fused[:, :, s:e]
        print(f"     {label:10s} : mean={sub.mean():.4f}  "
              f"std={sub.std():.4f}  min={sub.min():.4f}  max={sub.max():.4f}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_dataset(X_fused, y, mask, train_idx, val_idx, test_idx, norm_stats):
    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    save_dict = {
        'X'         : X_fused,
        'y'         : y,
        'mask'      : mask,
        'train_idx' : train_idx,
        'val_idx'   : val_idx,
        'test_idx'  : test_idx,
        'n_features': np.array(13),
        'n_classes' : np.array(6),
        'feature_order': np.array(
            ['S2_B2','S2_B3','S2_B4','S2_B5','S2_B6',
             'S2_B7','S2_B8','S2_B8A','S2_B11','S2_B12',
             'Sol_pH','Sol_OC','Sol_Tex']
        ),
        'class_names': np.array(CLASS_NAMES),
    }
    if norm_stats is not None:
        save_dict['soil_mu']  = norm_stats['mu']
        save_dict['soil_std'] = norm_stats['std']

    np.savez_compressed(NPZ_OUT, **save_dict)
    size_mb = os.path.getsize(NPZ_OUT) / 1e6
    print(f"\n✅ Sauvegardé : {NPZ_OUT}  ({size_mb:.1f} MB)")


# ── Rapport ───────────────────────────────────────────────────────────────────
def print_report(X_fused, y, train_idx, val_idx, test_idx):
    n = len(y)
    print("\n" + "="*60)
    print("RAPPORT PRÉTRAITEMENT — Part 3 — California")
    print("="*60)
    print(f"""
  Source       : CAL_covariates.npz (Part 2)
  Stratégie    : Early Fusion — S2 + Sol (meilleure config ablation)

  Justification S2+Sol :
    OA=0.848  Kappa=0.802  F1=0.831
    → F1 > papier (0.829)  ← seule config qui dépasse le papier
    → S2+Tout (0.824) moins bon : covariables redondantes

  Input model  : (N, T=36, F=13)
  ┌──────────────────────────────────────────┐
  │  Features [0:10]  → 10 bandes S2         │
  │  Features [10:13] → 3 Sol (répétées)     │
  │    [pH, OC, Texture]                     │
  └──────────────────────────────────────────┘

  Train : {len(train_idx):,} points  ({len(train_idx)/n*100:.0f}%)
  Val   : {len(val_idx):,} points  ({len(val_idx)/n*100:.0f}%)
  Test  : {len(test_idx):,} points  ({len(test_idx)/n*100:.0f}%)

  Classes (6) : {', '.join(CLASS_NAMES)}
""")
    print("  → Lancer : python CAL_Part3_Step3_model.py")
    print("="*60)


def main():
    print("="*60)
    print("Part 3 — Étape 2 : Prétraitement Early Fusion — California")
    print("="*60 + "\n")

    X_s2, mask, y, cov_soil, train_idx, val_idx, test_idx = load_raw()
    X_s2             = check_s2_normalization(X_s2, mask)
    cov_soil, norm_s = check_and_normalize_soil(cov_soil, train_idx)
    train_idx, val_idx, test_idx = ensure_val_split(
        train_idx, val_idx, test_idx, y)

    print(f"\n── Early Fusion S2 + Sol ───────────────────────────")
    X_fused = build_early_fusion(X_s2, cov_soil)
    print(f"   X_fused : {X_fused.shape}  [N, T=36, F=13] ✅")

    print_split_stats(X_fused, y, train_idx, val_idx, test_idx)
    save_dataset(X_fused, y, mask, train_idx, val_idx, test_idx, norm_s)
    print_report(X_fused, y, train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()
