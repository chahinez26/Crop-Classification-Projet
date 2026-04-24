"""
MCTNet — Climate Covariates v3 — TIMESTEP — Arkansas
Lance les 72 exports GEE en boucle (36 timesteps × 2 zones)
"""

import ee
import time

ee.Initialize(project='sentinel-2-491600')

# =============================================================================
# PARAMÈTRES FIXES
# =============================================================================
YEAR         = 2021
CDL_CONF     = 95
FOLDER       = 'MCTNet_v5_PART2'
CLASS_VALUES = [0, 1, 2, 3, 4]
CLASS_POINTS = [760, 380, 1210, 2340, 310]

ZONES = [
    ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),
    ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])
]

CDL_CORN    = 1
CDL_COTTON  = 2
CDL_RICE    = 3
CDL_SOYBEAN = 5

# =============================================================================
# FONCTIONS
# =============================================================================
def window_dates(t, year):
    m   = t // 3
    w   = t % 3
    def pad(n): return f"{n:02d}"
    nxt_m = 1      if m == 11 else m + 2
    nxt_y = year+1 if m == 11 else year
    nxt   = f"{nxt_y}-{pad(nxt_m)}-01"
    mstr  = pad(m + 1)
    if w == 0: return f"{year}-{mstr}-01", f"{year}-{mstr}-11"
    if w == 1: return f"{year}-{mstr}-11", f"{year}-{mstr}-21"
    return f"{year}-{mstr}-21", nxt


def get_label_image(geom):
    cdl = (ee.ImageCollection('USDA/NASS/CDL')
           .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
           .first())
    conf_mask = cdl.select('confidence').gte(CDL_CONF)
    wc_mask   = (ee.ImageCollection('ESA/WorldCover/v200')
                 .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
                 .first().select('Map').eq(40))
    cdl_masked = (cdl.select('cropland')
                  .updateMask(conf_mask)
                  .updateMask(wc_mask)
                  .clip(geom))
    return (ee.Image(0).rename('crop_label')
            .where(cdl_masked.eq(CDL_CORN),    0)
            .where(cdl_masked.eq(CDL_COTTON),  1)
            .where(cdl_masked.eq(CDL_RICE),    2)
            .where(cdl_masked.eq(CDL_SOYBEAN), 3)
            .where(
                cdl_masked.gt(0)
                .And(cdl_masked.neq(CDL_CORN))
                .And(cdl_masked.neq(CDL_COTTON))
                .And(cdl_masked.neq(CDL_RICE))
                .And(cdl_masked.neq(CDL_SOYBEAN)),
                4)
            .updateMask(cdl_masked.gt(0))
            .toInt()
            .clip(geom))


def get_climate_composite(geom, start, end):
    gridmet = (ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
               .filterDate(start, end)
               .filterBounds(geom))

    temp = (gridmet.select(['tmmx', 'tmmn'])
            .map(lambda img:
                 img.select('tmmx').add(img.select('tmmn'))
                 .divide(2).subtract(273.15)
                 .rename('temp_mean'))
            .mean().unmask(0).clip(geom))

    vpd = (gridmet.select('vpd')
           .mean()
           .rename('vpd_mean')
           .unmask(0).clip(geom))

    solar = (gridmet.select('srad')
             .mean()
             .rename('solar_mean')
             .unmask(0).clip(geom))

    return temp.addBands(vpd).addBands(solar).toFloat()


# =============================================================================
# BOUCLE PRINCIPALE — 72 exports
# =============================================================================
tasks = []

for t in range(36):
    z = 1  # Zone 1 seulement
    tstr = f"{t+1:02d}"
    zstr = "1"
    name = f"ARK_CLIM_T{tstr}_Z{zstr}"

    start, end = window_dates(t, YEAR)
    geom       = ZONES[1]
    labels     = get_label_image(geom)
    clim_comp  = get_climate_composite(geom, start, end)

    img_for_sample = clim_comp.addBands(labels)

    samples = img_for_sample.stratifiedSample(
        numPoints   = 0,
        classBand   = 'crop_label',
        region      = geom,
        scale       = 30,
        classValues = CLASS_VALUES,
        classPoints = CLASS_POINTS,
        seed        = 42,
        dropNulls   = False,
        geometries  = True,
        tileScale   = 16
    )

    task = ee.batch.Export.table.toDrive(
        collection     = samples,
        description    = name,
        folder         = FOLDER,
        fileNamePrefix = name,
        fileFormat     = 'CSV'
    )
    task.start()
    tasks.append((name, task))
    print(f"✅ Lancé : {name}  ({start} → {end})")
    time.sleep(0.5)

print(f"\n🚀 36 exports Z1 lancés")

print(f"\n🚀 {len(tasks)} exports lancés — vérifier sur code.earthengine.google.com/tasks")