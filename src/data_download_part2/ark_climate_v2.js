// =============================================================================
// MCTNet GEE — CLIMATE COVARIATES v2 — Part 2
// CORRECTION : ERA5 Daily → GRIDMET (USA, 2021 garanti, ~4km)
//
// POURQUOI ERA5 échouait :
//   'ECMWF/ERA5/DAILY' → 0 images pour 2021 dans GEE (accès restreint)
//   → Image vide → erreur "Got 0 and 1 bands"

// Variables climatiques choisies (3) :
//   1. temp_mean   : moyenne de (tmax + tmin)/2 annuelle (°C)
//                   → contrôle stades phénologiques, degrés-jours
//                   (riz >20°C, coton sensible froid, soja/maïs seuils distincts)
//   2. precip_total: somme annuelle de pr (mm)
//                   → variable la plus discriminante Delta Arkansas
//                   (riz irrigué insensible, coton drainage, soja pluvial)
//   3. solar_mean  : moyenne annuelle de srad (W/m²)
//                   → conditionne photosynthèse et biomasse
//                   (pic juillet = cultures d'été vs autres saisons)
//
// PROCÉDURE (2 runs) :
//   ZONE_INDEX = 0 → ARK_CLIMATE_Z0.csv
//   ZONE_INDEX = 1 → ARK_CLIMATE_Z1.csv
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CETTE VARIABLE À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — identiques à gee_arkansas_v5.js
// =============================================================================
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_v5_PART2';
var CLASS_VALUES = [0, 1, 2, 3, 4];
var CLASS_POINTS = [760, 380, 1210, 2340, 310];

var CDL_CORN    = 1;
var CDL_COTTON  = 2;
var CDL_RICE    = 3;
var CDL_SOYBEAN = 5;

var ZONES = [
  ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),
  ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])
];

var GEOM = ZONES[ZONE_INDEX];
var zStr = '' + ZONE_INDEX;

// =============================================================================
// IMAGE DE LABELS CDL — identique à gee_arkansas_v5.js
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
    .where(cdlMasked.eq(CDL_CORN),    0)
    .where(cdlMasked.eq(CDL_COTTON),  1)
    .where(cdlMasked.eq(CDL_RICE),    2)
    .where(cdlMasked.eq(CDL_SOYBEAN), 3)
    .where(
      cdlMasked.gt(0)
        .and(cdlMasked.neq(CDL_CORN)).and(cdlMasked.neq(CDL_COTTON))
        .and(cdlMasked.neq(CDL_RICE)).and(cdlMasked.neq(CDL_SOYBEAN)),
      4)
    .updateMask(cdlMasked.gt(0))
    .toInt()
    .clip(geom);
}

// =============================================================================
// GRIDMET — Variables climatiques 2021
// Dataset : IDAHO_EPSCOR/GRIDMET  (~4km, journalier, USA)
// =============================================================================
var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(GEOM);

// Vérification disponibilité
print('=== MCTNet GEE — Climate Covariates v2 (GRIDMET) ===');
print('Zone    : Z' + zStr);
print('Images GRIDMET 2021 disponibles :', gridmet.size());
// Doit afficher 365 (une image par jour)

// 1. TEMPÉRATURE MOYENNE ANNUELLE (°C)
//    GRIDMET tmmx = T max (K), tmmn = T min (K)
//    temp_mean = moyenne annuelle de (tmax+tmin)/2 convertie en °C
var temp_mean = gridmet
  .select(['tmmx', 'tmmn'])
  .map(function(img) {
    var tmax = img.select('tmmx');
    var tmin = img.select('tmmn');
    return tmax.add(tmin).divide(2)
      .subtract(273.15)          // K → °C
      .rename('temp_mean')
      .copyProperties(img, ['system:time_start']);
  })
  .mean()                        // moyenne annuelle
  .clip(GEOM);

// 2. PRÉCIPITATION TOTALE ANNUELLE (mm)
//    GRIDMET pr = précipitation journalière (mm)
var precip_total = gridmet
  .select('pr')
  .sum()                         // somme 2021 en mm
  .rename('precip_total')
  .clip(GEOM);

// 3. RAYONNEMENT SOLAIRE MOYEN ANNUEL (W/m²)
//    GRIDMET srad = downward surface shortwave radiation (W/m²)
var solar_mean = gridmet
  .select('srad')
  .mean()                        // moyenne annuelle
  .rename('solar_mean')
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
  // remettre crop_label en int pour stratifiedSample
  .addBands(labels, null, true);

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que Part 1
// seed=42, scale=30 → MÊMES pixels que ARK_CDL_Z*.csv ✅
// =============================================================================
print('');
print('Extraction en cours...');
print('Cible : 5000 pts (760/380/1210/2340/310)');
print('');

var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,             // ← MÊME seed que Part 1
  dropNulls   : true,
  geometries  : true,
  tileScale   : 16
});

// =============================================================================
// VÉRIFICATIONS
// =============================================================================
print('Points extraits :', samples.size());
print('');

var names   = {0:'Corn   ', 1:'Cotton ', 2:'Rice   ', 3:'Soybean', 4:'Others '};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (label=' + i + ') — cible ' + targets[i] + ' :', n);
}

// Vérifier les valeurs climatiques (doit être non-null, non-zero)
print('');
print('Vérification valeurs (5 premiers points) :');
print(samples.limit(5));

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_CLIMATE_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_CLIMATE_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 9);

Map.addLayer(temp_mean,
  {min: 10, max: 25, palette: ['blue', 'yellow', 'red']},
  'Température moyenne 2021 (°C) Z' + zStr);

Map.addLayer(precip_total,
  {min: 800, max: 1600, palette: ['white', 'cyan', 'blue', 'darkblue']},
  'Précipitation totale 2021 (mm) Z' + zStr, false);

Map.addLayer(solar_mean,
  {min: 150, max: 280, palette: ['black', 'orange', 'yellow', 'white']},
  'Rayonnement solaire moyen (W/m²) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color: 'FFFF00', pointRadius: 2}),
  {}, 'Points climatiques Z' + zStr);

print('');
print('✅ Export lancé → ARK_CLIMATE_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : ZONE_INDEX = 1');
} else {
  print('🏁 Les deux zones extraites !');
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à ARK_CDL_Z*.csv de Part 1 ✅
// crop_label    ← 0-4 (vérification)
// temp_mean     ← température moyenne annuelle 2021 (°C)
// precip_total  ← précipitation totale annuelle 2021 (mm)
// solar_mean    ← rayonnement solaire moyen 2021 (W/m²)
// .geo          ← lon/lat identiques à ARK_CDL_Z*.csv ✅
// =============================================================================
