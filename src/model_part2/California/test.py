import json, csv, numpy as np

# Adapter ce chemin
CSV_DIR = r"data\raw\covariables\california"
fname   = "CAL_CLIMATE_Z0.csv"

import os
fpath = os.path.join(CSV_DIR, fname)

lons, lats = [], []
with open(fpath, newline='', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f)):
        geo = row.get('.geo', '{}')
        try:
            g = json.loads(geo)
            lon = float(g['coordinates'][0])
            lat = float(g['coordinates'][1])
        except:
            lon, lat = 0.0, 0.0
        lons.append(lon)
        lats.append(lat)
        if i < 3:
            print(f"Row {i}: system:index={row.get('system:index','?')}")
            print(f"         .geo = {geo[:80]}")
            print(f"         lon={lon:.4f}  lat={lat:.4f}")
            print(f"         temp_mean={row.get('temp_mean','?')}")

print(f"\nTotal lignes : {len(lons)}")
print(f"Lons uniques : {len(set(lons))}")
print(f"Lats uniques : {len(set(lats))}")
print(f"lon range : [{min(lons):.4f}, {max(lons):.4f}]")
print(f"lat range : [{min(lats):.4f}, {max(lats):.4f}]")