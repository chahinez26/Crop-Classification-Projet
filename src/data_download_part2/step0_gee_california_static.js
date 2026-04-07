// =============================================================================
// MCTNet GEE — STATIC FEATURES — CALIFORNIA
// Soil / Climate / Topography
// Wang et al. 2024 — Computers and Electronics in Agriculture
//
// SPÉCIFICITÉS CALIFORNIA (vs Arkansas) :
//   - 6 classes : Grapes=0, Rice=1, Alfalfa=2, Almonds=3, Pistachios=4, Others=5
//   - Zones ASYMÉTRIQUES : Z0=5720 pts (Sacramento Valley), Z1=4300 pts (San Joaquin Sud)
//   - Climat méditerranéen → valeurs WorldClim très différentes d'Arkansas
//   - Relief important (Sierra Nevada) → variables topo plus discriminantes
//
// PROCÉDURE (6 runs) :
//   Run 1 : MODE='SOIL',    ZONE_INDEX=0  → CAL_SOIL_Z0.csv
//   Run 2 : MODE='SOIL',    ZONE_INDEX=1  → CAL_SOIL_Z1.csv
//   Run 3 : MODE='CLIMATE', ZONE_INDEX=0  → CAL_CLIM_Z0.csv
//   Run 4 : MODE='CLIMATE', ZONE_INDEX=1  → CAL_CLIM_Z1.csv
//   Run 5 : MODE='TOPO',    ZONE_INDEX=0  → CAL_TOPO_Z0.csv
//   Run 6 : MODE='TOPO',    ZONE_INDEX=1  → CAL_TOPO_Z1.csv
//
// MERGE PYTHON :
//   cal_step1_merge_static.py → CAL_dataset_static.npz
//   X_static[10020, 17]  float32
//   Compatible avec CAL_dataset.npz existant (même ordre de points)
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CES VARIABLES À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var MODE       = 'SOIL';   // 'SOIL' | 'CLIMATE' | 'TOPO'
var ZONE_INDEX = 0;         // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — IDENTIQUES au script Sentinel California v2
// =============================================================================
var FOLDER     = 'MCTNet_California_v2';
var YEAR       = 2021;
var CDL_CONF   = 95;

// CLASS_POINTS asymétriques — IDENTIQUES au script Sentinel California v2
// Z0 : Sacramento Valley (nord) — 5720 pts
// Z1 : San Joaquin Sud          — 4300 pts
var CLASS_VALUES     = [0, 1, 2, 3, 4, 5];
var CLASS_POINTS_Z0  = [1030, 2030, 490, 390, 20,  1760];  // Total = 5720
var CLASS_POINTS_Z1  = [1030, 10,   490, 390, 620, 1760];  // Total = 4300

// Choisir la bonne distribution selon la zone
var CLASS_POINTS = (ZONE_INDEX === 0) ? CLASS_POINTS_Z0 : CLASS_POINTS_Z1;

// Codes CDL California
var CDL_GRAPES     = 27;
var CDL_RICE       = 3;
var CDL_ALFALFA    = 36;
var CDL_ALMONDS    = 75;
var CDL_PISTACHIOS = 204;

// =============================================================================
// ZONES CALIFORNIA — Identiques au script Sentinel California v2
// Zone 0 : Sacramento Valley (nord, riziculture dominante)
// Zone 1 : San Joaquin Sud (fruits à coque + raisins)
// =============================================================================
// ⚠️  REMPLACER par tes coordonnées exactes du script v2 California
var ZONES = [
  // Zone 0 : Sacramento Valley
  ee.Geometry.Rectangle([-122.50, 38.50, -121.00, 40.50]),

  // Zone 1 : San Joaquin Sud
  ee.Geometry.Rectangle([-120.50, 35.50, -119.00, 37.50])
];
var GEOM = ZONES[ZONE_INDEX];
var zStr = '' + ZONE_INDEX;

// =============================================================================
// FONCTION : Image de labels CDL California
// Identique au script Sentinel v2 California (même masques, même seed)
// =============================================================================
function getLabelImage(geom) {
  var cdl = ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first();
  var confMask = cdl.select('confidence').gte(CDL_CONF);
  var wcMask   = ee.ImageCollection('ESA/WorldCover/v200')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first().select('Map').eq(40);
  var cdlMasked = cdl.select('cropland')
    .updateMask(confMask).updateMask(wcMask).clip(geom);

  var grapes    = cdlMasked.eq(CDL_GRAPES);
  var rice      = cdlMasked.eq(CDL_RICE);
  var alfalfa   = cdlMasked.eq(CDL_ALFALFA);
  var almonds   = cdlMasked.eq(CDL_ALMONDS);
  var pistachios= cdlMasked.eq(CDL_PISTACHIOS);
  var others    = cdlMasked.gt(0)
                    .and(cdlMasked.neq(CDL_GRAPES))
                    .and(cdlMasked.neq(CDL_RICE))
                    .and(cdlMasked.neq(CDL_ALFALFA))
                    .and(cdlMasked.neq(CDL_ALMONDS))
                    .and(cdlMasked.neq(CDL_PISTACHIOS));

  return ee.Image(0).rename('crop_label')
    .where(grapes,     0)
    .where(rice,       1)
    .where(alfalfa,    2)
    .where(almonds,    3)
    .where(pistachios, 4)
    .where(others,     5)
    .updateMask(cdlMasked.gt(0)).toInt().clip(geom);
}

// =============================================================================
// FONCTIONS DE FEATURES (identiques au script Arkansas)
// =============================================================================

function getSoilImage(geom) {
  var tex  = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
               .select('b0').rename('sol_texture_class').clip(geom);
  var sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02')
               .select('b0').rename('sol_sand_pct').clip(geom);
  var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02')
               .select('b0').rename('sol_clay_pct').clip(geom);
  var oc   = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
               .select('b0').rename('sol_organic_carbon').clip(geom);
  var ph   = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
               .select('b0').rename('sol_ph_h2o').clip(geom);
  var bd   = ee.Image('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02')
               .select('b0').rename('sol_bulk_density').clip(geom);
  return tex.addBands(sand).addBands(clay)
            .addBands(oc).addBands(ph).addBands(bd).toFloat();
}

function getClimateImage(geom) {
  var bio   = ee.Image('WORLDCLIM/V1/BIO').clip(geom);
  var tmean = bio.select('bio01').rename('clim_tmean');
  var prec  = bio.select('bio12').rename('clim_prec');
  var pseas = bio.select('bio15').rename('clim_prec_seas');
  var tmax  = bio.select('bio05').rename('clim_tmax');
  var tmin  = bio.select('bio06').rename('clim_tmin');

  // ETP annuelle MODIS MOD16 — 2021
  // Spécificité California : ET très élevée en été (irrigation intensive)
  var et = ee.ImageCollection('MODIS/006/MOD16A2')
    .filterBounds(geom)
    .filterDate('2021-01-01', '2022-01-01')
    .select('ET')
    .sum()
    .multiply(0.1)
    .rename('clim_et_annual')
    .clip(geom);

  return tmean.addBands(prec).addBands(pseas)
              .addBands(tmax).addBands(tmin)
              .addBands(et).toFloat();
}

function getTopoImage(geom) {
  var dem    = ee.Image('USGS/SRTMGL1_003').clip(geom);
  var elev   = dem.rename('topo_elevation');
  var slope  = ee.Terrain.slope(dem).rename('topo_slope');
  var aspect = ee.Terrain.aspect(dem).rename('topo_aspect');

  // TPI — fenêtre 300m
  var focal_mean = dem.focal_mean({radius: 300, units: 'meters'});
  var tpi = dem.subtract(focal_mean).rename('topo_tpi');

  // TWI approximé
  var slope_rad = slope.multiply(Math.PI / 180);
  var twi = slope_rad.tan().max(0.001).pow(-1).log().rename('topo_twi');

  return elev.addBands(slope).addBands(aspect)
             .addBands(tpi).addBands(twi).toFloat();
}

// =============================================================================
// MAIN
// =============================================================================
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE — Static Features California ===');
print('Mode : ' + MODE + '  |  Zone : Z' + zStr);
var nPts = (ZONE_INDEX === 0) ? 5720 : 4300;
print('Cible : ' + nPts + ' points');
print('');

var featImage, outName, featBands;

if (MODE === 'SOIL') {
  featImage = getSoilImage(GEOM);
  outName   = 'CAL_SOIL_Z' + zStr;
  featBands = ['sol_texture_class','sol_sand_pct','sol_clay_pct',
               'sol_organic_carbon','sol_ph_h2o','sol_bulk_density'];
  print('Variables sol (6) : texture, sand, clay, OC, pH, bulk_density');

} else if (MODE === 'CLIMATE') {
  featImage = getClimateImage(GEOM);
  outName   = 'CAL_CLIM_Z' + zStr;
  featBands = ['clim_tmean','clim_prec','clim_prec_seas',
               'clim_tmax','clim_tmin','clim_et_annual'];
  print('Variables climat (6) : tmean, prec, prec_seas, tmax, tmin, ET');

} else if (MODE === 'TOPO') {
  featImage = getTopoImage(GEOM);
  outName   = 'CAL_TOPO_Z' + zStr;
  featBands = ['topo_elevation','topo_slope','topo_aspect',
               'topo_tpi','topo_twi'];
  print('Variables topo (5) : elevation, slope, aspect, TPI, TWI');

} else {
  print('❌ MODE inconnu. Utiliser SOIL | CLIMATE | TOPO');
  throw new Error('MODE invalide');
}

// Extraction aux mêmes pixels — seed=42 identique au script Sentinel v2
var imgForSample = featImage.addBands(labels);

var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,           // ← CRITIQUE : même seed = mêmes pixels
  dropNulls   : true,
  geometries  : true,
  tileScale   : 16
});

var nTotal = samples.size();
print('Points échantillonnés :', nTotal);
print('(cible : ' + nPts + ')');
print('');

// Vérification par classe
var cNames = {0:'Grapes    ',1:'Rice      ',2:'Alfalfa   ',
              3:'Almonds   ',4:'Pistachios',5:'Others    '};
for (var i = 0; i < 6; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(cNames[i] + ' (label=' + i + ') :', n);
}
print('');

Export.table.toDrive({
  collection     : samples,
  description    : outName,
  folder         : FOLDER,
  fileNamePrefix : outName,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 9);
if (MODE === 'SOIL') {
  Map.addLayer(featImage.select('sol_clay_pct'),
    {min:10, max:50, palette:['yellow','brown']}, 'Clay % Z' + zStr);
} else if (MODE === 'CLIMATE') {
  Map.addLayer(featImage.select('clim_prec'),
    {min:100, max:1200, palette:['red','white','blue']}, 'Precip Z' + zStr);
} else if (MODE === 'TOPO') {
  Map.addLayer(featImage.select('topo_elevation'),
    {min:0, max:1500, palette:['green','yellow','white']}, 'Elevation Z' + zStr);
}
Map.addLayer(labels,
  {min:0, max:5, palette:['9C27B0','2196F3','8BC34A','FF9800','4CAF50','9E9E9E']},
  'CDL Classes Z' + zStr, false);
Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points ' + outName);

print('✅ Export lancé → ' + outName);
print('');
print('RÉCAPITULATIF DES 6 RUNS CALIFORNIA :');
print('  Run 1 : SOIL,    Z0 → CAL_SOIL_Z0.csv  (6 vars sol)');
print('  Run 2 : SOIL,    Z1 → CAL_SOIL_Z1.csv');
print('  Run 3 : CLIMATE, Z0 → CAL_CLIM_Z0.csv  (6 vars climat)');
print('  Run 4 : CLIMATE, Z1 → CAL_CLIM_Z1.csv');
print('  Run 5 : TOPO,    Z0 → CAL_TOPO_Z0.csv  (5 vars topo)');
print('  Run 6 : TOPO,    Z1 → CAL_TOPO_Z1.csv');
