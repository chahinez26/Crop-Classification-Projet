"""

- 10 spectral bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
- Cloud masking via SCL (Scene Classification Layer)
- 10-day median composites -> 36 time steps per year
- Missing data filled with 0

Source: Copernicus Data Space (https://dataspace.copernicus.eu)
Auth: Credentials from .env file (username/password)

Pipeline per 10-day window:
1. Search all S2 L2A products in the window (cloud < 70%)
2. Download each product .zip
3. Extract 10 bands + SCL from .SAFE
4. For each pixel: mask clouds using SCL, compute median of valid values
5. Save as single GeoTIFF (10 bands)
"""

import os
import sys
import time
import json
import zipfile
import warnings
import numpy as np
from pathlib import Path
from datetime import date
from calendar import monthrange

warnings.filterwarnings("ignore")

try:
    import requests
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.merge import merge
    from rasterio.transform import from_bounds
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install requests rasterio python-dotenv")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
YEAR = 2021

# 10 bands from paper (Table 1)
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
ALL_BANDS = BANDS_10M + BANDS_20M
SCL_BAND = "SCL"  # Scene Classification Layer for cloud masking

# SCL classes to mask (0=nodata, 1=saturated, 2=dark, 3=shadow, 8=cloud_med, 9=cloud_high, 10=cirrus, 11=snow)
SCL_MASK_VALUES = {0, 1, 2, 3, 8, 9, 10, 11}

# Study areas [west, south, east, north]
# APRÈS (assez grand pour 10 000 points agricoles variés)
STUDY_AREAS = {
    "california": {
        "bbox": [-121.5, 36.5, -119.5, 38.0],
        "description": "Central Valley, California (Fresno-Sacramento area)",
    },
    "arkansas": {
        "bbox": [-92.0, 34.0, -90.0, 35.5],
        "description": "Eastern Arkansas Delta",
    },
}

OUTPUT_BASE = os.path.join("data", "raw", "sentinel2")
TEMP_DIR = os.path.join("data", "temp", "sentinel2")

# Copernicus Data Space API
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

TARGET_RESOLUTION = 10  # meters


# ============================================================
# AUTH
# ============================================================
def load_credentials():
    """Load Copernicus credentials from .env file."""
    username = None
    password = None

    # Read .env file directly (don't rely on os.environ — Windows has 'username' var)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key == "username":
                    username = val
                elif key == "password":
                    password = val

    # Fallback to env vars
    if not username:
        username = os.environ.get("COPERNICUS_USER")
    if not password:
        password = os.environ.get("COPERNICUS_PASS")

    if not username or not password:
        print("ERROR: No credentials found.")
        print("Create a .env file with:")
        print("  username=your_email@example.com")
        print("  password=your_password")
        print("Register at: https://dataspace.copernicus.eu")
        sys.exit(1)

    return username, password


def get_access_token(username, password):
    """Get OAuth2 access token from Copernicus Data Space."""
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    resp = requests.post(TOKEN_URL, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ============================================================
# SEARCH
# ============================================================
def search_products(bbox, start_date, end_date, max_cloud=70, max_results=10):
    """Search for S2 L2A products via OData API."""
    west, south, east, north = bbox
    footprint = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"

    filter_str = (
        f"Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T23:59:59.999Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})"
    )

    params = {
        "$filter": filter_str,
        "$orderby": "ContentDate/Start asc",
        "$top": max_results,
    }

    try:
        resp = requests.get(f"{ODATA_URL}/Products", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json().get("value", [])
    except requests.exceptions.RequestException as e:
        print(f"      Search error: {e}")
        return []


# ============================================================
# DOWNLOAD
# ============================================================
def download_product(product_id, product_name, output_dir, token, 
                     username=None, password=None):
    """Download avec refresh automatique du token si 401."""
    zip_path = os.path.join(output_dir, f"{product_name}.zip")

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.testzip()
            return zip_path, token  # ← retourne aussi le token
        except (zipfile.BadZipFile, Exception):
            os.remove(zip_path)

    url     = f"{DOWNLOAD_URL}({product_id})/$value"
    partial = zip_path + ".partial"

    for attempt in range(1, 6):
        try:
            headers    = {"Authorization": f"Bearer {token}"}
            downloaded = 0

            if os.path.exists(partial):
                downloaded = os.path.getsize(partial)
                headers["Range"] = f"bytes={downloaded}-"

            resp = requests.get(url, headers=headers, timeout=1800, stream=True)

            # ── refresh token si 401 ──────────────────────────
            if resp.status_code == 401:
                print(f"      Token expiré → refresh...")
                if username and password:
                    token = get_access_token(username, password)
                    headers["Authorization"] = f"Bearer {token}"
                    resp = requests.get(url, headers=headers, 
                                       timeout=1800, stream=True)
                else:
                    raise Exception("Token expiré et pas de credentials pour refresh")

            if resp.status_code == 200 and downloaded > 0:
                downloaded = 0
                mode = "wb"
            elif resp.status_code == 206:
                mode = "ab"
            else:
                resp.raise_for_status()
                mode = "wb"

            total = int(resp.headers.get("content-length", 0)) + downloaded

            with open(partial, mode) as f:
                for chunk in resp.iter_content(chunk_size=131072):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r      [{pct:5.1f}%] {downloaded/1e6:.0f}/{total/1e6:.0f} MB", 
                              end="", flush=True)
            print()

            os.replace(partial, zip_path)
            return zip_path, token   # ← retourne le token mis à jour

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"\n      Attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                time.sleep(attempt * 15)

    if os.path.exists(partial):
        os.remove(partial)
    return None, token

# ============================================================
# EXTRACT BANDS
# ============================================================
def extract_bands_from_safe(zip_path, output_dir):
    """
    Extract spectral bands + SCL from a .SAFE zip.
    Returns dict: {band_name: file_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted = {}
    product_name = Path(zip_path).stem

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            # Match band files
            for band in ALL_BANDS + [SCL_BAND]:
                if f"_{band}_" in entry and entry.endswith(".jp2"):
                    # For 10m bands, use R10m; for 20m, use R20m; for SCL, use R20m
                    if band in BANDS_10M and "R10m" in entry:
                        out_name = f"{product_name}_{band}_10m.jp2"
                    elif band in BANDS_20M and "R20m" in entry:
                        out_name = f"{product_name}_{band}_20m.jp2"
                    elif band == SCL_BAND and "R20m" in entry:
                        out_name = f"{product_name}_{band}_20m.jp2"
                    else:
                        continue

                    out_path = os.path.join(output_dir, out_name)
                    if band not in extracted:
                        if not os.path.exists(out_path):
                            data = zf.read(entry)
                            with open(out_path, "wb") as f:
                                f.write(data)
                        extracted[band] = out_path

    return extracted


def read_and_resample_band(band_path, target_shape, target_transform, target_crs):
    """Read a band file and resample to target grid (10m)."""
    with rasterio.open(band_path) as src:
        if src.height == target_shape[0] and src.width == target_shape[1] and src.crs == target_crs:
            return src.read(1).astype(np.float32)

        data = np.empty(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        return data


# ============================================================
# 10-DAY COMPOSITES
# ============================================================
def generate_10day_windows(year):
    """Generate 36 ten-day windows: day 1-10, 11-20, 21-end per month."""
    windows = []
    idx = 1
    for month in range(1, 13):
        _, last_day = monthrange(year, month)
        windows.append((date(year, month, 1), date(year, month, 10), idx))
        idx += 1
        windows.append((date(year, month, 11), date(year, month, 20), idx))
        idx += 1
        windows.append((date(year, month, 21), date(year, month, last_day), idx))
        idx += 1
    return windows


def create_composite(band_stacks, scl_stacks):
    """
    Create cloud-masked median composite from multiple observations.

    Args:
        band_stacks: dict {band_name: list of 2D arrays}
        scl_stacks: list of 2D SCL arrays

    Returns:
        dict {band_name: 2D median composite}
    """
    composites = {}

    for band_name in ALL_BANDS:
        if band_name not in band_stacks or not band_stacks[band_name]:
            composites[band_name] = None
            continue

        stack = np.stack(band_stacks[band_name], axis=0)  # (N_obs, H, W)
        n_obs = stack.shape[0]

        # Create cloud mask from SCL
        if scl_stacks:
            scl_stack = np.stack(scl_stacks, axis=0)  # (N_obs, H, W)
            # Resize SCL if needed (SCL is 20m, bands may be 10m)
            if scl_stack.shape[1:] != stack.shape[1:]:
                from scipy.ndimage import zoom
                factors = (1, stack.shape[1] / scl_stack.shape[1], stack.shape[2] / scl_stack.shape[2])
                scl_stack = zoom(scl_stack, factors, order=0)  # nearest neighbor

            # Mask: True = cloud/invalid
            cloud_mask = np.isin(scl_stack, list(SCL_MASK_VALUES))
            # Set cloudy pixels to NaN
            stack_masked = stack.astype(np.float32)
            stack_masked[cloud_mask[:n_obs]] = np.nan
        else:
            stack_masked = stack.astype(np.float32)

        # Median composite (ignoring NaN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            composite = np.nanmedian(stack_masked, axis=0)

        # Fill remaining NaN with 0 (paper: missing = 0)
        composite = np.nan_to_num(composite, nan=0.0)
        composites[band_name] = composite

    return composites


def save_composite_geotiff(composites, output_path, ref_path):
    """Save composite bands as a single multi-band GeoTIFF."""
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        crs = ref.crs
        height = ref.height
        width = ref.width

    # Stack bands in order
    data = np.zeros((len(ALL_BANDS), height, width), dtype=np.float32)
    for i, band in enumerate(ALL_BANDS):
        if composites.get(band) is not None:
            arr = composites[band]
            if arr.shape == (height, width):
                data[i] = arr
            else:
                # Resize to match reference
                from scipy.ndimage import zoom
                factors = (height / arr.shape[0], width / arr.shape[1])
                data[i] = zoom(arr, factors, order=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=len(ALL_BANDS),
        dtype="float32", crs=crs, transform=transform,
        compress="deflate", nodata=0,
    ) as dst:
        for i in range(len(ALL_BANDS)):
            dst.write(data[i], i + 1)
            dst.set_band_description(i + 1, ALL_BANDS[i])

    return output_path


# ============================================================
# MAIN PIPELINE
# ============================================================
def process_window(area_name, bbox, start_date, end_date, window_idx,
                   token, username, password):
    """
    Process one 10-day window:
    1. Search products
    2. Download best products (up to 3)
    3. Extract bands + SCL
    4. Create cloud-masked median composite
    5. Save as GeoTIFF
    """
    output_dir = os.path.join(OUTPUT_BASE, area_name)
    output_path = os.path.join(output_dir, f"S2_{area_name}_{YEAR}_T{window_idx:02d}.tif")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"    T{window_idx:02d} ({start_date} to {end_date}): exists, skipping")
        return True

    print(f"\n    T{window_idx:02d} ({start_date} to {end_date}):")

    # Search
    products = search_products(bbox, str(start_date), str(end_date), max_cloud=70, max_results=5)

    if not products:
        # Relax cloud filter
        products = search_products(bbox, str(start_date), str(end_date), max_cloud=95, max_results=3)

    if not products:
        print(f"      No products found — creating zero composite")
        _save_zero_composite(output_path, bbox)
        return True

    # Take 1 best product per window (saves bandwidth)
    products = products[:1]
    print(f"      Found {len(products)} products")

    temp_dir = os.path.join(TEMP_DIR, area_name, f"T{window_idx:02d}")
    os.makedirs(temp_dir, exist_ok=True)

    band_stacks = {b: [] for b in ALL_BANDS}
    scl_stacks = []
    ref_path = None

    for prod in products:
        prod_id = prod["Id"]
        prod_name = prod["Name"][:60]

        cloud = "?"
        for attr in prod.get("Attributes", []):
            if attr.get("Name") == "cloudCover":
                cloud = f"{attr.get('Value', 0):.0f}%"

        print(f"      Product: {prod_name}... (cloud: {cloud})")

        # Download
        try:
            zip_path, token = download_product(
                 prod["Id"], prod["Name"], temp_dir, token,
                 username=username, password=password )
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 401:
                print("      Token expired, refreshing...")
                token = get_access_token(username, password)
                zip_path = download_product(prod["Id"], prod["Name"], temp_dir, token)
            else:
                print(f"      Download failed: {e}")
                continue

        if zip_path is None:
            continue

        # Extract bands
        print(f"      Extracting bands...")
        bands = extract_bands_from_safe(zip_path, temp_dir)
        print(f"      Got {len(bands)} bands: {list(bands.keys())}")

        if not bands:
            continue

        # Read first 10m band as reference for grid
        first_10m = next((bands[b] for b in BANDS_10M if b in bands), None)
        if first_10m is None:
            continue

        with rasterio.open(first_10m) as src:
            target_shape = (src.height, src.width)
            target_transform = src.transform
            target_crs = src.crs

        if ref_path is None:
            ref_path = first_10m

        # Read all bands (resample 20m to 10m)
        for band in ALL_BANDS:
            if band in bands:
                try:
                    arr = read_and_resample_band(
                        bands[band], target_shape, target_transform, target_crs
                    )
                    band_stacks[band].append(arr)
                except Exception as e:
                    print(f"      Warning: {band} read failed — {e}")

        # Read SCL
        if SCL_BAND in bands:
            try:
                scl = read_and_resample_band(
                    bands[SCL_BAND], target_shape, target_transform, target_crs
                )
                scl_stacks.append(scl)
            except Exception:
                pass

    # Check if we got any data
    has_data = any(len(v) > 0 for v in band_stacks.values())
    if not has_data or ref_path is None:
        print(f"      No valid data — creating zero composite")
        _save_zero_composite(output_path, bbox)
        return True

    # Create composite
    print(f"      Creating cloud-masked median composite...")
    composites = create_composite(band_stacks, scl_stacks)
    save_composite_geotiff(composites, output_path, ref_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"      Saved: {output_path} ({size_mb:.1f} MB)")

    # Clean up ALL temp files for this window (save disk space)
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return True


def _save_zero_composite(output_path, bbox):
    """Create zero-filled GeoTIFF for missing windows."""
    from pyproj import Transformer

    # Determine UTM zone from bbox center
    lon_center = (bbox[0] + bbox[2]) / 2
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = "N" if (bbox[1] + bbox[3]) / 2 >= 0 else "S"
    epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])

    width = int((xmax - xmin) / TARGET_RESOLUTION)
    height = int((ymax - ymin) / TARGET_RESOLUTION)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=len(ALL_BANDS),
        dtype="float32", crs=f"EPSG:{epsg}", transform=transform,
        compress="deflate", nodata=0,
    ) as dst:
        for i in range(len(ALL_BANDS)):
            dst.write(np.zeros((height, width), dtype=np.float32), i + 1)
            dst.set_band_description(i + 1, ALL_BANDS[i])


def main():
    print("=" * 70)
    print("  Sentinel-2 Download — Copernicus Data Space")
    print("  Following MCTNet Paper (Wang et al. 2024)")
    print("=" * 70)
    print(f"  Year:       {YEAR}")
    print(f"  Bands:      {len(ALL_BANDS)} ({', '.join(ALL_BANDS)})")
    print(f"  Composites: 10-day median -> 36 time steps")
    print(f"  Cloud mask: SCL-based")
    print(f"  Missing:    Filled with 0")
    print(f"  Output:     {OUTPUT_BASE}/")
    print("=" * 70)

    # Load credentials
    username, password = load_credentials()
    print(f"\n  Authenticating as {username}...")
    try:
        token = get_access_token(username, password)
        print(f"  Authenticated OK!")
    except Exception as e:
        print(f"  Auth failed: {e}")
        print(f"  Check your .env credentials")
        sys.exit(1)

    windows = generate_10day_windows(YEAR)
    print(f"  Time windows: {len(windows)}")

    for area_name, area_config in STUDY_AREAS.items():
        bbox = area_config["bbox"]
        print(f"\n{'='*70}")
        print(f"  REGION: {area_name.upper()} — {area_config['description']}")
        print(f"  Bbox: {bbox}")
        print(f"{'='*70}")

        success = 0
        for start_date, end_date, window_idx in windows:
            try:
                ok = process_window(
                    area_name, bbox, start_date, end_date, window_idx,
                    token, username, password,
                )
                if ok:
                    success += 1
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 401:
                    print("      Token expired, refreshing...")
                    token = get_access_token(username, password)
                    ok = process_window(
                        area_name, bbox, start_date, end_date, window_idx,
                        token, username, password,
                    )
                    if ok:
                        success += 1
                else:
                    print(f"      ERROR: {e}")
            except Exception as e:
                print(f"      ERROR: {e}")

        print(f"\n  {area_name}: {success}/{len(windows)} windows completed")

    print(f"\n{'='*70}")
    print("  Sentinel-2 download complete!")
    print(f"  Output: {OUTPUT_BASE}/{{region}}/S2_{{region}}_{YEAR}_T{{01-36}}.tif")
    print("=" * 70)


if __name__ == "__main__":
    username, password = load_credentials()
    #print(f"[DEBUG] username='{username}'")
    #print(f"[DEBUG] password='{password}'")
    main()