"""
MCTNet — Climate Covariates v3 — TIMESTEP — California
Lance les 72 exports GEE en boucle (36 timesteps × 2 zones)
"""

import ee
import time

ee.Initialize(project='sentinel-2-491600')

YEAR   = 2021
FOLDER = 'MCTNet_California_v2'

CLASS_VALUES    = [0, 1, 2, 3, 4, 5]
CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20,  1760]
CLASS_POINTS_Z1 = [1030, 10,   490, 390, 620, 1760]

ZONES = [
    ee.Geometry.Rectangle([-122.50, 37.50, -119.80, 40.50]),
    ee.Geometry.Rectangle([-121.00, 34.50, -118.50, 37.50])
]

CDL_CONF       = 95
CDL_GRAPES     = 69
CDL_RICE       = 3
CDL_ALFALFA    = 36
CDL_ALMONDS    = 75
CDL_PISTACHIOS = 77


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
            .where(cdl_masked.eq(CDL_GRAPES),     0)
            .where(cdl_masked.eq(CDL_RICE),       1)
            .where(cdl_masked.eq(CDL_ALFALFA),    2)
            .where(cdl_masked.eq(CDL_ALMONDS),    3)
            .where(cdl_masked.eq(CDL_PISTACHIOS), 4)
            .where(
                cdl_masked.gt(0)
                .And(cdl_masked.neq(CDL_GRAPES))
                .And(cdl_masked.neq(CDL_RICE))
                .And(cdl_masked.neq(CDL_ALFALFA))
                .And(cdl_masked.neq(CDL_ALMONDS))
                .And(cdl_masked.neq(CDL_PISTACHIOS)),
                5)
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
# BOUCLE PRINCIPALE — 72 exports California
# =============================================================================
tasks = []

for t in range(36):
    for z in range(2):
        tstr         = f"{t+1:02d}"
        zstr         = str(z)
        name         = f"CAL_CLIM_T{tstr}_Z{zstr}"
        class_points = CLASS_POINTS_Z0 if z == 0 else CLASS_POINTS_Z1

        start, end = window_dates(t, YEAR)
        geom       = ZONES[z]
        labels     = get_label_image(geom)
        clim_comp  = get_climate_composite(geom, start, end)

        img_for_sample = clim_comp.addBands(labels)

        samples = img_for_sample.stratifiedSample(
            numPoints   = 0,
            classBand   = 'crop_label',
            region      = geom,
            scale       = 30,
            classValues = CLASS_VALUES,
            classPoints = class_points,
            seed        = 42,
            dropNulls   = False,
            geometries  = True,
            tileScale   = 16
        )

        task = ee.batch.Export.table.toDrive(
        collection     = samples,
        description    = name.replace('_', ''),  # sans underscore
         folder         = 'MCTNet_California_v2',
        fileNamePrefix = name,                   # nom fichier avec underscore
        fileFormat     = 'CSV',
         selectors      = ['system:index', 'crop_label',
                      'temp_mean', 'vpd_mean', 'solar_mean']
)
        task.start()
        tasks.append((name, task))
        print(f"✅ Lancé : {name}  ({start} → {end})")
        time.sleep(0.5)

print(f"\n🚀 {len(tasks)} exports lancés")