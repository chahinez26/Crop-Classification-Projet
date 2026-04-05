"""
Download USGS 3DEP topography data (Part 2 covariates).

Products: DEM, Slope, Aspect
Resolution: 10m (matches Sentinel-2)
Source: USGS 3DEP via py3dep (no auth needed)

Used in ablation study: S2 + topography variables
"""

import os
import numpy as np

STUDY_AREAS = {
    "california": {
        "bbox": (-120.5, 36.8, -120.0, 37.3),
        "description": "Central Valley, California",
    },
    "arkansas": {
        "bbox": (-91.0, 34.8, -90.5, 35.3),
        "description": "Eastern Arkansas Delta",
    },
}

OUTPUT_BASE = os.path.join("data", "raw", "topography")
RESOLUTION = 10  # meters


def compute_slope_aspect(dem_data, transform):
    """Compute slope and aspect from DEM using numpy gradients."""
    # Get pixel size in meters (approximate)
    dy = abs(transform[4]) if abs(transform[4]) > 0 else abs(transform[0])
    dx = abs(transform[0]) if abs(transform[0]) > 0 else abs(transform[4])

    # If CRS is geographic (degrees), convert to meters approximately
    if dx < 1:  # Likely in degrees
        lat_mid = 37.0  # approximate
        dx = dx * 111320 * np.cos(np.radians(lat_mid))
        dy = dy * 110540

    grad_y, grad_x = np.gradient(dem_data, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    aspect = np.degrees(np.arctan2(-grad_x, grad_y))
    aspect = (aspect + 360) % 360

    return slope.astype(np.float32), aspect.astype(np.float32)


def save_raster(data, reference_path, output_path, description=""):
    """Save array as GeoTIFF using reference file for metadata."""
    import rasterio

    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "float32",
            "compress": "deflate",
        })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data.astype(np.float32), 1)
        if description:
            dst.set_band_description(1, description)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"    Saved: {output_path} ({size_mb:.1f} MB)")


def download_topography(area_name, bbox):
    """Download DEM and compute slope/aspect."""
    import py3dep

    output_dir = os.path.join(OUTPUT_BASE, area_name)
    os.makedirs(output_dir, exist_ok=True)

    dem_path = os.path.join(output_dir, f"dem_{area_name}_{RESOLUTION}m.tif")
    slope_path = os.path.join(output_dir, f"slope_{area_name}_{RESOLUTION}m.tif")
    aspect_path = os.path.join(output_dir, f"aspect_{area_name}_{RESOLUTION}m.tif")

    if all(os.path.exists(p) for p in [dem_path, slope_path, aspect_path]):
        print(f"    All topography files exist for {area_name}")
        return

    # Download DEM
    print(f"    Downloading DEM for {area_name}...")
    try:
        dem = py3dep.get_map("DEM", bbox, resolution=RESOLUTION, crs="EPSG:4326")

        # Save DEM
        dem.rio.to_raster(dem_path, compress="deflate")
        print(f"    DEM saved: {dem_path}")

        # Compute slope and aspect
        import rasterio
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform

        slope, aspect = compute_slope_aspect(dem_data, transform)
        save_raster(slope, dem_path, slope_path, "Slope (degrees)")
        save_raster(aspect, dem_path, aspect_path, "Aspect (degrees 0-360)")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("  USGS 3DEP Topography Download (Part 2 Covariates)")
    print("=" * 70)
    print(f"  Products:   DEM, Slope, Aspect")
    print(f"  Resolution: {RESOLUTION}m")
    print(f"  Output:     {OUTPUT_BASE}/")
    print("=" * 70)

    for area_name, config in STUDY_AREAS.items():
        print(f"\n  REGION: {area_name.upper()} — {config['description']}")
        download_topography(area_name, config["bbox"])

    print("\n  Topography download complete!")


if __name__ == "__main__":
    main()