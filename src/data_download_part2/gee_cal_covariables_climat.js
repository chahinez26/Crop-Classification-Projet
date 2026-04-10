
var ZONE_INDEX = 0;   // 0 ou 1

// =============================================================================
// PARAMÈTRES FIXES 
// =============================================================================
var YEAR         = 2021;
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_Covariables';
var CLASS_VALUES = [0, 1, 2, 3, 4, 5];
// CLASS_POINTS ASYMÉTRIQUES par zone
// Z0 : prend tout le Rice, très peu de Pistachios
var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];

// Z1 : prend tout le Pistachios, très peu de Rice
var CLASS_POINTS_Z1 = [1030, 10, 490, 390, 620, 1760];

// Codes CDL California
var CDL_GRAPES     = 69;
var CDL_RICE       = 3;
var CDL_ALFALFA    = 36;
var CDL_ALMONDS    = 75;
var CDL_PISTACHIOS = 77;

var ZONES = [
  ee.Geometry.Rectangle([-122.50, 37.50, -119.80, 40.50]),
  ee.Geometry.Rectangle([-121.00, 34.50, -118.50, 37.50])
];

var CLASS_POINTS = (ZONE_INDEX === 0) ? CLASS_POINTS_Z0 : CLASS_POINTS_Z1;
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

  var grapes     = cdlMasked.eq(CDL_GRAPES);
  var rice       = cdlMasked.eq(CDL_RICE);
  var alfalfa    = cdlMasked.eq(CDL_ALFALFA);
  var almonds    = cdlMasked.eq(CDL_ALMONDS);
  var pistachios = cdlMasked.eq(CDL_PISTACHIOS);
  var others     = cdlMasked.gt(0)
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
    .updateMask(cdlMasked.gt(0))
    .toInt()
    .clip(geom);
}

// =============================================================================
// VARIABLES CLIMATIQUES — ERA5 DAILY 2021

// =============================================================================
// =============================================================================
// VARIABLES CLIMATIQUES — ERA5-Land DAILY 2021 (FONCTIONNE)
// =============================================================================
var era5Land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(GEOM);

// Vérification : afficher le nombre d'images (doit être 365)
print('ERA5-Land count:', era5Land.size());

// 1. Température moyenne annuelle (°C)
var temp_mean = era5Land
  .select('temperature_2m')
  .mean()
  .subtract(273.15)
  .rename('temp_mean')
  .clip(GEOM);

// 2. Précipitations totales annuelles (mm)
var precip_total = era5Land
  .select('total_precipitation_sum')
  .sum()
  .multiply(1000)    // m -> mm
  .rename('precip_total')
  .clip(GEOM);

// 3. Point de rosée moyen annuel (°C)
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
print('=== Covariables Climat ERA5 — California Z' + zStr + ' ===');
print('');
print('Source : ECMWF/ERA5/DAILY  (résolution ~27.8 km)');
print('Période : 2021-01-01 → 2022-01-01');
print('');
print('Variables retenues :');
print('  temp_mean    : Température moy. annuelle (°C)  — GDD crop discrimination');
print('  precip_total : Précipitations totales (mm)     — Rice vs rainfed crops');
print('  dewpoint_mean: Point de rosée moy. (°C)        — Humidity / ET proxy');
print('');

var n_images = era5Land.size();
print('Nombre d images ERA5 disponibles :', n_images);
print('(attendu ~365)');
print('');

print('temp_mean stats (°C) :', temp_mean.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 27830,
  maxPixels: 1e8
}));

print('precip_total stats (mm) :', precip_total.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 27830,
  maxPixels: 1e8
}));

print('dewpoint_mean stats (°C) :', dewpoint_mean.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 27830,
  maxPixels: 1e8
}));

// =============================================================================
// EXTRACTION DES POINTS D'ÉCHANTILLONNAGE
// =============================================================================
var samples = covClimate.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,          // même scale que CDL → même grille de pixels
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,          // MÊME seed → mêmes coordonnées géographiques
  dropNulls   : false,       // ERA5 est continu → pas de null attendu
  geometries  : true,
  tileScale   : 16
});

print('');
print('Points extraits :', samples.size());
print('');

// Vérification par classe
var classNames = ['Grapes    ','Rice      ','Alfalfa   ',
                    'Almonds   ','Pistachios','Others    '];
  for (var i = 0; i < 6; i++) {
    var cible = CLASS_POINTS[i];
    var n     = samples.filter(ee.Filter.eq('crop_label', i)).size();
    print(classNames[i] + ' (label=' + i + ') — cible ' + cible + ' :', n);
  }


// Vérification absence de nulls (ERA5 est un raster global sans trous)
var n_null_temp = samples.filter(ee.Filter.eq('temp_mean', 0)).size();
print('');
print('Points avec temp_mean=0 (doit être 0) :', n_null_temp);

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_COV_CLIMAT_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_COV_CLIMAT_Z' + zStr,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 10);
Map.addLayer(temp_mean,
  {min: 10, max: 22, palette: ['blue', 'white', 'red']},
  'Temp moy. annuelle ERA5 (°C) Z' + zStr);
Map.addLayer(precip_total,
  {min: 800, max: 1800, palette: ['white', 'cyan', 'darkblue']},
  'Précipitations totales ERA5 (mm) Z' + zStr, false);
Map.addLayer(dewpoint_mean,
  {min: 5, max: 18, palette: ['white', 'lightblue', 'blue']},
  'Point de rosée moy. ERA5 (°C) Z' + zStr, false);
Map.addLayer(labels,
  {min: 0, max: 4, palette: ['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

print('');
print('✅ Export lancé → CAL_COV_CLIMAT_Z' + zStr);
print('   Dossier Drive : ' + FOLDER);
print('');
if (ZONE_INDEX === 0) {
  print('→ Prochain run : ZONE_INDEX = 1');
} else {
  print('→ Les 2 zones Climat sont prêtes !');
  print('   Passer au script SOL (gee_ark_covariables_sol_v2.js)');
}

