
import os
import numpy as np


RAW_FILE = r"data\merged_npz\california\CAL_dataset.npz"
PART1_FILE = r"data\processed\CAL_dataset_preprocessed.npz"
OUTPUT_DIR = r"data\merged_npz\california"
OUTPUT_FILE = r"data\merged_npz\california\Part3_CAL_dataset.npz"

BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDVI", "EVI", "NDWI", "GNDVI", "NDRE"]
FEATURE_NAMES = BAND_NAMES + INDEX_NAMES


def compute_indices(x_raw, mask):
    refl = x_raw.astype(np.float32) / 10000.0

    blue = refl[:, :, 0]
    green = refl[:, :, 1]
    red = refl[:, :, 2]
    red_edge_1 = refl[:, :, 3]
    nir = refl[:, :, 6]
    nir_narrow = refl[:, :, 7]
    swir1 = refl[:, :, 8]

    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    ndwi = (nir - swir1) / (nir + swir1 + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ndre = (nir_narrow - red_edge_1) / (nir_narrow + red_edge_1 + eps)

    indices = np.stack([ndvi, evi, ndwi, gndvi, ndre], axis=-1)
    indices = np.where(mask[:, :, None] == 0, indices, 0.0)
    indices = np.clip(indices, -2.0, 2.0)
    return indices.astype(np.float32)


def normalize_features(x_full, train_idx, mask):
    x_norm = np.zeros_like(x_full, dtype=np.float32)
    feature_stats = []

    x_train = x_full[train_idx]
    mask_train = mask[train_idx]

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        vals_train = x_train[:, :, feat_idx][mask_train == 0]
        feat_min = float(vals_train.min())
        feat_max = float(vals_train.max())
        denom = feat_max - feat_min if feat_max > feat_min else 1.0

        x_norm[:, :, feat_idx] = np.where(
            mask == 0,
            np.clip((x_full[:, :, feat_idx] - feat_min) / denom, 0.0, 1.0),
            0.0,
        )
        feature_stats.append({"feature": feat_name, "min": feat_min, "max": feat_max})

    return x_norm, feature_stats


def main():
    print("=" * 60)
    print("Step 1 - Prepare Part 3 California dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = np.load(RAW_FILE)
    part1 = np.load(PART1_FILE)

    x_raw = raw["X"]
    y = raw["y"]
    mask = raw["mask"].astype(np.uint8)
    lons = raw["lons"] if "lons" in raw.files else np.zeros(len(y), dtype=np.float64)
    lats = raw["lats"] if "lats" in raw.files else np.zeros(len(y), dtype=np.float64)

    train_idx = part1["train_idx"]
    val_idx = part1["val_idx"]
    test_idx = part1["test_idx"]

    print(f"Loaded raw data     : X{x_raw.shape} y{y.shape} mask{mask.shape}")
    print(f"Loaded Part 1 split : train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x_indices = compute_indices(x_raw, mask)
    x_full = np.concatenate([x_raw.astype(np.float32), x_indices], axis=-1)
    x_norm, feature_stats = normalize_features(x_full, train_idx, mask)

    out = {
        "X_train": x_norm[train_idx],
        "y_train": y[train_idx],
        "mask_train": mask[train_idx].astype(np.float32),
        "X_val": x_norm[val_idx],
        "y_val": y[val_idx],
        "mask_val": mask[val_idx].astype(np.float32),
        "X_test": x_norm[test_idx],
        "y_test": y[test_idx],
        "mask_test": mask[test_idx].astype(np.float32),
        "X_all": x_norm,
        "y_all": y,
        "mask_all": mask.astype(np.float32),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "feature_names": np.array(FEATURE_NAMES),
        "lons": lons,
        "lats": lats,
    }

    np.savez_compressed(OUTPUT_FILE, **out)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\nSaved: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print(f"Final shapes: train={out['X_train'].shape} val={out['X_val'].shape} test={out['X_test'].shape}")
    print("\nFeature statistics based on train set:")
    for stats in feature_stats:
        print(f"  {stats['feature']:>5s} : min={stats['min']:.4f} max={stats['max']:.4f}")


if __name__ == "__main__":
    main()
