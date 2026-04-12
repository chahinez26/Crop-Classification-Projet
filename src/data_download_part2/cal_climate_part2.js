// =============================================================================
// MCTNet GEE — CALIFORNIA CLIMATE COVARIATES — Part 2
// Basé sur gee_california_v2.js (Part 1)
//
// Différences California vs Arkansas :
//   - 6 classes au lieu de 5
//   - Zones ASYMÉTRIQUES : Z0=5720 pts, Z1=4300 pts
//   - CLASS_POINTS change selon ZONE_INDEX
//   - Même dataset GRIDMET (USA continental → California ✅)
//
// Variables climatiques (3) :
//   1. temp_mean    : moyenne annuelle (tmax+tmin)/2 (°C)
//      → stades phénologiques : riz/alfalfa (chaud), raisins (cycle long)
//   2. precip_total : somme annuelle pr (mm)
//      → Californie très variable : Sacramento humide vs San Joaquin sec
//      → riz (pluvieux nord), pistaches/amandes (irrigué sec sud)
//   3. solar_mean   : moyenne annuelle srad (W/m²)
//      → Fort ensoleillement Valley → photosynthèse raisins/amandes
//
// PROCÉDURE (2 runs) :
//   ZONE_INDEX = 0 → CAL_CLIMATE_Z0.csv  (5720 points)
//   ZONE_INDEX = 1 → CAL_CLIMATE_Z1.csv  (4300 points)
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CETTE VARIABLE À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — identiques à gee_california_v2.js
// =============================================================================
var CDL_CONF       = 95;
var FOLDER         = 'MCTNet_California_v2';
var CLASS_VALUES   = [0, 1, 2, 3, 4, 5];

// ⚠️  ASYMÉTRIQUE : CLASS_POINTS différents selon la zone
var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];   // total = 5720
var CLASS_POINTS_Z1 = [1030, 10,   490, 390, 620, 1760];   // total = 4300

var CDL_GRAPES     = 69;
var CDL_RICE       = 3;
var CDL_ALFALFA    = 36;
var CDL_ALMONDS    = 75;
var CDL_PISTACHIOS = 77;

var ZONES = [
  ee.Geometry.Rectangle([-122.50, 37.50, -119.80, 40.50]),  // Z0 Sacramento
  ee.Geometry.Rectangle([-121.00, 34.50, -118.50, 37.50])   // Z1 San Joaquin
];

// Sélection automatique selon ZONE_INDEX
var CLASS_POINTS = (ZONE_INDEX === 0) ? CLASS_POINTS_Z0 : CLASS_POINTS_Z1;
var GEOM         = ZONES[ZONE_INDEX];
var zStr         = '' + ZONE_INDEX;
var N_EXPECTED   = (ZONE_INDEX === 0) ? 5720 : 4300;

// =============================================================================
// IMAGE DE LABELS CDL — identique à gee_california_v2.js
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

  return ee.Image(0).rename('crop_label')
    .where(cdlMasked.eq(CDL_GRAPES),     0)
    .where(cdlMasked.eq(CDL_RICE),       1)
    .where(cdlMasked.eq(CDL_ALFALFA),    2)
    .where(cdlMasked.eq(CDL_ALMONDS),    3)
    .where(cdlMasked.eq(CDL_PISTACHIOS), 4)
    .where(
      cdlMasked.gt(0)
        .and(cdlMasked.neq(CDL_GRAPES))
        .and(cdlMasked.neq(CDL_RICE))
        .and(cdlMasked.neq(CDL_ALFALFA))
        .and(cdlMasked.neq(CDL_ALMONDS))
        .and(cdlMasked.neq(CDL_PISTACHIOS)),
      5)
    .updateMask(cdlMasked.gt(0))
    .toInt()
    .clip(geom);
}

// =============================================================================
// GRIDMET — Variables climatiques 2021
// ✅ Couvre la Californie (USA continental)
// ✅ Données 2021 complètes (365 images)
// =============================================================================
var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(GEOM);

print('=== MCTNet GEE — California Climate Covariates (GRIDMET) ===');
print('Zone             : Z' + zStr);
print('Points attendus  : ' + N_EXPECTED);
print('Images GRIDMET   :', gridmet.size());   // attendu : 365
print('');

// 1. Température moyenne annuelle (°C)
var temp_mean = gridmet
  .select(['tmmx', 'tmmn'])
  .map(function(img) {
    return img.select('tmmx').add(img.select('tmmn')).divide(2)
      .subtract(273.15)
      .rename('temp_mean')
      .copyProperties(img, ['system:time_start']);
  })
  .mean()
  .unmask(-9999)   // sécurité null
  .clip(GEOM);

// 2. Précipitation totale annuelle (mm)
var precip_total = gridmet
  .select('pr')
  .sum()
  .rename('precip_total')
  .unmask(-9999)
  .clip(GEOM);

// 3. Rayonnement solaire moyen annuel (W/m²)
var solar_mean = gridmet
  .select('srad')
  .mean()
  .rename('solar_mean')
  .unmask(-9999)
  .clip(GEOM);

// =============================================================================
// ASSEMBLAGE + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(temp_mean)
  .addBands(precip_total)
  .addBands(solar_mean)
  .toFloat()
  .addBands(labels, null, true);   // crop_label en int pour stratifiedSample

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que gee_california_v2.js
// seed=42, scale=30, CLASS_POINTS asymétriques → MÊMES pixels que Part 1
// =============================================================================
var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,    // ← asymétrique selon ZONE_INDEX
  seed        : 42,              // ← MÊME seed que Part 1
  dropNulls   : true,
  geometries  : true,
  tileScale   : 16
});

// =============================================================================
// VÉRIFICATIONS
// =============================================================================
print('Points extraits :', samples.size());
print('Attendu         : ' + N_EXPECTED);
print('');

var classNames = ['Grapes    ', 'Rice      ', 'Alfalfa   ',
                  'Almonds   ', 'Pistachios', 'Others    '];
for (var i = 0; i < 6; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(classNames[i] + ' (label=' + i + ') — cible ' + CLASS_POINTS[i] + ' :', n);
}

print('');
print('Exemple valeurs (5 points) :');
print(samples.limit(5));

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_CLIMATE_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_CLIMATE_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 8);

Map.addLayer(temp_mean,
  {min: 10, max: 30, palette: ['blue','yellow','red']},
  'Température moyenne 2021 (°C) Z' + zStr);

Map.addLayer(precip_total,
  {min: 100, max: 1200, palette: ['white','cyan','blue','darkblue']},
  'Précipitation totale 2021 (mm) Z' + zStr, false);

Map.addLayer(solar_mean,
  {min: 150, max: 320, palette: ['black','orange','yellow','white']},
  'Rayonnement solaire (W/m²) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:5, palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points climatiques Z' + zStr);

print('');
print('✅ Export lancé → CAL_CLIMATE_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : ZONE_INDEX = 1  (4300 pts attendus)');
} else {
  print('🏁 Les deux zones extraites !');
  print('   CAL_CLIMATE_Z0 : 5720 pts  |  CAL_CLIMATE_Z1 : 4300 pts');
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à CAL_CDL_Z*.csv de Part 1 ✅
// crop_label    ← 0=Grapes, 1=Rice, 2=Alfalfa, 3=Almonds, 4=Pistachios, 5=Others
// temp_mean     ← température moyenne annuelle 2021 (°C)
// precip_total  ← précipitation totale annuelle 2021 (mm)
// solar_mean    ← rayonnement solaire moyen 2021 (W/m²)
// .geo          ← lon/lat identiques à CAL_CDL_Z*.csv ✅
// =============================================================================
