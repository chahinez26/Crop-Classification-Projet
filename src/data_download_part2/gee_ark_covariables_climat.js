// =============================================================================
// MCTNet GEE — COVARIABLES CLIMAT — Arkansas  (v2 — consignes prof)
// Source imposée : ECMWF/ERA5_LAND/DAILY_AGGR (résolution ~9 km)
//
// 3 attributs retenus parmi les équivalents ERA5-Land :
//
//   1. temperature_2m            → temp_mean
//   2. total_precipitation_sum   → precip_total
//   3. dewpoint_temperature_2m   → dewpoint_mean
//
// STRATÉGIE (identique au script de ton ami) :
//   - Même stratifiedSample, même seed=42, même scale=30
//   → MÊMES coordonnées géographiques que ARK_CDL_Z0/Z1
//   - Agrégation annuelle 2021 des données journalières ERA5-Land
//   - Merge en Python par system:index + zone
//
// PROCÉDURE (2 runs) :
//   Run 1 : ZONE_INDEX=0 → ARK_COV_CLIMAT_Z0  (dossier MCTNet_Covariables)
//   Run 2 : ZONE_INDEX=1 → ARK_COV_CLIMAT_Z1
//
// OUTPUT CSV columns :
//   system:index, crop_label, temp_mean, precip_total, dewpoint_mean, .geo
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CETTE VARIABLE À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES (identiques à gee_arkansas_v5.js de ton ami)
// =============================================================================
var YEAR         = 2021;
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_Covariables';
var CLASS_VALUES = [0, 1, 2, 3, 4];
var CLASS_POINTS = [760, 380, 1210, 2340, 310];

var CDL_CORN    = 1;
var CDL_COTTON  = 2;
var CDL_RICE    = 3;
var CDL_SOYBEAN = 5;

var ZONES = [
  ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),  // Z0 Nord Delta
  ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])   // Z1 Sud Delta
];
var GEOM = ZONES[ZONE_INDEX];
var zStr = '' + ZONE_INDEX;

// =============================================================================
// LABELS CDL — identique à ton ami
// =============================================================================
function getLabelImage(geom) {
  var cdl = ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01')).first();
  var confMask = cdl.select('confidence').gte(CDL_CONF);
  var wcMask = ee.ImageCollection('ESA/WorldCover/v200')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first().select('Map').eq(40);
  var cdlMasked = cdl.select('cropland')
    .updateMask(confMask).updateMask(wcMask).clip(geom);

  var corn    = cdlMasked.eq(CDL_CORN);
  var cotton  = cdlMasked.eq(CDL_COTTON);
  var rice    = cdlMasked.eq(CDL_RICE);
  var soybean = cdlMasked.eq(CDL_SOYBEAN);
  var others  = cdlMasked.gt(0)
    .and(cdlMasked.neq(CDL_CORN)).and(cdlMasked.neq(CDL_COTTON))
    .and(cdlMasked.neq(CDL_RICE)).and(cdlMasked.neq(CDL_SOYBEAN));

  return ee.Image(0).rename('crop_label')
    .where(corn, 0).where(cotton, 1).where(rice, 2)
    .where(soybean, 3).where(others, 4)
    .updateMask(cdlMasked.gt(0)).toInt().clip(geom);
}

// =============================================================================
// VARIABLES CLIMATIQUES — ERA5-Land DAILY 2021 (FONCTIONNE)
// =============================================================================
var era5Land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(GEOM);

// 1. TEMPÉRATURE MOYENNE ANNUELLE
var temp_mean = era5Land
  .select('temperature_2m')
  .mean()
  .subtract(273.15)
  .rename('temp_mean')
  .clip(GEOM);

// 2. PRÉCIPITATIONS TOTALES ANNUELLES (mm)
var precip_total = era5Land
  .select('total_precipitation_sum')
  .sum()
  .multiply(1000)
  .rename('precip_total')
  .clip(GEOM);

// 3. POINT DE ROSÉE MOYEN ANNUEL
var dewpoint_mean = era5Land
  .select('dewpoint_temperature_2m')
  .mean()
  .subtract(273.15)
  .rename('dewpoint_mean')
  .clip(GEOM);

// =============================================================================
// COMBINER TOUTES LES VARIABLES + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var covClimate = temp_mean
  .addBands(precip_total)
  .addBands(dewpoint_mean)
  .addBands(labels);

// =============================================================================
// STATS DE VÉRIFICATION
// =============================================================================
print('=== Covariables Climat ERA5-Land — Arkansas Z' + zStr + ' ===');
print('');
print('Source : ECMWF/ERA5_LAND/DAILY_AGGR (résolution ~9 km)');
print('Période : 2021-01-01 → 2022-01-01');
print('');
print('Variables retenues :');
print('  temp_mean    : Température moy. annuelle (°C)  — GDD crop discrimination');
print('  precip_total : Précipitations totales (mm)     — Rice vs rainfed crops');
print('  dewpoint_mean: Point de rosée moy. (°C)        — Humidity / ET proxy');
print('');

var n_images = era5Land.size();
print('Nombre d images ERA5-Land disponibles :', n_images);
print('(attendu ~365)');
print('');

print('temp_mean stats (°C) :', temp_mean.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 9000,   // résolution native ~9 km, on peut mettre 9000 pour cohérence
  maxPixels: 1e8
}));

print('precip_total stats (mm) :', precip_total.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 9000,
  maxPixels: 1e8
}));

print('dewpoint_mean stats (°C) :', dewpoint_mean.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 9000,
  maxPixels: 1e8
}));

// =============================================================================
// EXTRACTION DES POINTS D'ÉCHANTILLONNAGE
// =============================================================================
var samples = covClimate.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,
  dropNulls   : true,
  geometries  : true,
  tileScale   : 16
});

samples = samples.map(function(f) {
  return f.set('lon', f.geometry().coordinates().get(0))
          .set('lat', f.geometry().coordinates().get(1));
});

print('');
print('Points extraits :', samples.size());
print('(cible : 5000)');
print('');

// Vérification par classe
var names = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (cible ' + targets[i] + ') :', n);
}

// Vérification absence de nulls
var n_null_temp = samples.filter(ee.Filter.notNull(['temp_mean']).not()).size();
print('');
print('Points avec temp_mean=null (doit être 0) :', n_null_temp);

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_COV_CLIMAT_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_COV_CLIMAT_Z' + zStr,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 10);
Map.addLayer(temp_mean,
  {min: 10, max: 22, palette: ['blue', 'white', 'red']},
  'Temp moy. annuelle ERA5-Land (°C) Z' + zStr);
Map.addLayer(precip_total,
  {min: 800, max: 1800, palette: ['white', 'cyan', 'darkblue']},
  'Précipitations totales ERA5-Land (mm) Z' + zStr, false);
Map.addLayer(dewpoint_mean,
  {min: 5, max: 18, palette: ['white', 'lightblue', 'blue']},
  'Point de rosée moy. ERA5-Land (°C) Z' + zStr, false);
Map.addLayer(labels,
  {min: 0, max: 4, palette: ['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

print('');
print('✅ Export lancé → ARK_COV_CLIMAT_Z' + zStr);
print('   Dossier Drive : ' + FOLDER);
print('');
if (ZONE_INDEX === 0) {
  print('→ Prochain run : ZONE_INDEX = 1');
} else {
  print('→ Les 2 zones Climat sont prêtes !');
  print('   Passer au script SOL (gee_ark_covariables_sol_v2.js)');
}

// =============================================================================
// JUSTIFICATION DES CHOIX (à inclure dans le rapport)
// =============================================================================
// Variables ERA5-Land utilisées (équivalentes à celles demandées) :
//   temperature_2m            ← RETENU  (température de croissance)
//   total_precipitation_sum   ← RETENU  (besoin hydrique annuel)
//   dewpoint_temperature_2m   ← RETENU  (humidité, ET proxy)
// Autres variables non retenues car redondantes ou peu discriminantes.
// =============================================================================