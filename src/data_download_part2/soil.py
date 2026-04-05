"""
Download ISRIC SoilGrids soil property data (Part 2 covariates).

Properties: clay, sand, silt, soc (organic carbon), bdod (bulk density), phh2o (pH)
Depths: 0-5cm, 5-15cm
Resolution: ~250m
Source: ISRIC SoilGrids WCS (no auth needed)

Used in ablation study: S2 + soil variables
"""

import os
import time
import requests
import numpy as np
from pyproj import Transformer

STUDY_AREAS = {
    "california": [-120.5, 36.8, -120.0, 37.3],
    "arkansas": [-91.0, 34.8, -90.5, 35.3],
}

PROPERTIES = ["clay", "sand", "silt", "soc", "bdod", "phh2o"]
DEPTHS = ["0-5cm", "5-15cm"]
STAT = "mean"

OUTPUT_BASE = os.path.join("data", "raw", "soil")

# SoilGrids uses Homolosine projection
HOMOLOSINE_PROJ4 = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs("EPSG:4326", HOMOLOSINE_PROJ4, always_xy=True)


def bbox_to_homolosine(bbox):
    """Convert WGS84 bbox to Homolosine coordinates."""
    min_xy = transformer.transform(bbox[0], bbox[1])
    max_xy = transformer.transform(bbox[2], bbox[3])
    return min_xy[0], min_xy[1], max_xy[0], max_xy[1]


def download_soilgrids(area_name, bbox, prop, depth):
    """Download one soil property via WCS."""
    output_dir = os.path.join(OUTPUT_BASE, area_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"soilgrids_{prop}_{depth}_{STAT}.tif")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"    Already exists: {output_path}")
        return output_path

    xmin, ymin, xmax, ymax = bbox_to_homolosine(bbox)
    depth_label = depth.replace("-", "_").replace("cm", "cm")

    url = (
        f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
        f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={prop}_{depth_label}_{STAT}"
        f"&FORMAT=image/tiff"
        f"&SUBSET=X({xmin},{xmax})"
        f"&SUBSET=Y({ymin},{ymax})"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/152160"
    )

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and len(resp.content) > 100:
                # Validate it's a TIFF
                if resp.content[:2] in [b"II", b"MM"]:
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    size_kb = os.path.getsize(output_path) / 1024
                    print(f"    {prop}_{depth}: {size_kb:.0f} KB")
                    return output_path
                else:
                    print(f"    {prop}_{depth}: Not a valid TIFF (attempt {attempt})")
            else:
                print(f"    {prop}_{depth}: HTTP {resp.status_code} (attempt {attempt})")
        except Exception as e:
            print(f"    {prop}_{depth}: Error — {e} (attempt {attempt})")

        if attempt < 3:
            time.sleep(10)

    print(f"    FAILED: {prop}_{depth}")
    return None


def main():
    print("=" * 70)
    print("  ISRIC SoilGrids Download (Part 2 Covariates)")
    print("=" * 70)
    print(f"  Properties: {PROPERTIES}")
    print(f"  Depths:     {DEPTHS}")
    print(f"  Output:     {OUTPUT_BASE}/")
    print("=" * 70)

    for area_name, bbox in STUDY_AREAS.items():
        print(f"\n  REGION: {area_name.upper()}")
        for prop in PROPERTIES:
            for depth in DEPTHS:
                download_soilgrids(area_name, bbox, prop, depth)

    print("\n  Soil download complete!")
    print("  NOTE: SoilGrids CRS metadata may be missing.")
    print(f"  Assign: {HOMOLOSINE_PROJ4}")


if __name__ == "__main__":
    main()