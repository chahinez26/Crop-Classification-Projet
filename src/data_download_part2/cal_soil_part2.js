// =============================================================================
// MCTNet GEE — CALIFORNIA SOIL COVARIATES — Part 2
// Basé sur gee_california_v2.js (Part 1) + corrections ark_soil_v2.js
//
// Variables sol (3) — OpenLandMap, profondeur 0-5 cm :
//   1. soil_ph      → pH brut ×10 (ex: 70 = pH 7.0)
//      → Raisins préfèrent pH 5.5-6.5, riz pH 5.5-6.5
//      → Pistaches/amandes tolèrent pH 7-8 (sols calcaires San Joaquin)
//      → Fort gradient N→S en Californie (acide nord, alcalin sud)
//
//   2. soil_oc      → Carbone organique brut dg/kg (ex: 80 = 8.0 g/kg)
//      → Sols alluviaux Sacramento (haut OC) vs sables Kern (faible OC)
//      → Discriminant entre cultures intensives et extensives
//
//   3. soil_texture → Classe USDA 1-12
//      → Sacramento : limons argileux (riz) vs San Joaquin : sablo-limoneux
//      → Pistaches préfèrent sols profonds bien drainés (texture 7-9)
//
// ⚠️  Zones ASYMÉTRIQUES California :
//   Z0 : 5720 points  (Sacramento + Nord San Joaquin)
//   Z1 : 4300 points  (San Joaquin Central + Tulare)
//
// PROCÉDURE (2 runs) :
//   ZONE_INDEX = 0 → CAL_SOIL_Z0.csv  (5720 pts)
//   ZONE_INDEX = 1 → CAL_SOIL_Z1.csv  (4300 pts)
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

var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];
var CLASS_POINTS_Z1 = [1030, 10,   490, 390, 620, 1760];

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
// DONNÉES SOL — OpenLandMap (profondeur 0-5 cm, bande 'b0')
// unmask(-1) : sécurité perte de points (même correction qu'Arkansas)
// =============================================================================

// 1. pH brut ×10 — conversion ÷10 dans Step9_merge_covariates.py
var soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0').rename('soil_ph')
  .unmask(-1).clip(GEOM);

// 2. OC brut dg/kg — conversion ÷10 dans Step9
var soil_oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0').rename('soil_oc')
  .unmask(-1).clip(GEOM);

// 3. Texture USDA 1-12 — pas de conversion
var soil_texture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0').rename('soil_texture')
  .unmask(-1).clip(GEOM);

// =============================================================================
// ASSEMBLAGE + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(soil_ph)
  .addBands(soil_oc)
  .addBands(soil_texture)
  .toFloat()
  .addBands(labels, null, true);

// =============================================================================
// VÉRIFICATIONS PRÉALABLES
// =============================================================================
print('=== MCTNet GEE — California Soil Covariates (OpenLandMap) ===');
print('Zone       : Z' + zStr);
print('Attendus   : ' + N_EXPECTED + ' points');
print('');

var phStats = soil_ph.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry: GEOM, scale: 250, maxPixels: 1e9
});
print('Statistiques soil_ph Z' + zStr + ' (brut ×10) :');
print(phStats);
// Z0 Sacramento : mean ~60-65 (pH 6.0-6.5, légèrement acide)
// Z1 San Joaquin: mean ~70-80 (pH 7.0-8.0, alcalin = calcaire)

print('');

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que gee_california_v2.js
// =============================================================================
var samples = imgForSample.stratifiedSample({
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
var n_ph_null = samples.filter(ee.Filter.eq('soil_ph', -1)).size();
var n_oc_null = samples.filter(ee.Filter.eq('soil_oc', -1)).size();
var n_tx_null = samples.filter(ee.Filter.eq('soil_texture', -1)).size();
print('Nulls soil_ph :', n_ph_null);
print('Nulls soil_oc :', n_oc_null);
print('Nulls soil_tx :', n_tx_null);
// Attendu : 0 pour tous

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_SOIL_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_SOIL_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 8);

Map.addLayer(soil_ph,
  {min: 50, max: 85, palette: ['red','orange','yellow','green','blue']},
  'pH sol 0-5cm (×10) Z' + zStr);

Map.addLayer(soil_oc,
  {min: 0, max: 120, palette: ['white','yellow','brown','black']},
  'OC sol 0-5cm (dg/kg) Z' + zStr, false);

Map.addLayer(soil_texture,
  {min: 1, max: 12, palette: [
    '8B0000','FF0000','FF6347','FFA500','FFD700','ADFF2F',
    '00FF00','00CED1','0000FF','4B0082','8B008B','FF69B4'
  ]},
  'Texture USDA Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:5, palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points sol Z' + zStr);

print('');
print('✅ Export lancé → CAL_SOIL_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : ZONE_INDEX = 1  (4300 pts attendus)');
} else {
  print('🏁 Les deux zones extraites !');
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à CAL_CDL_Z*.csv de Part 1 ✅
// crop_label    ← 0-5 (vérification)
// soil_ph       ← pH brut ×10 | -1 si null
// soil_oc       ← OC brut dg/kg | -1 si null
// soil_texture  ← classe USDA 1-12 | -1 si null
// .geo          ← lon/lat identiques à CAL_CDL_Z*.csv ✅
// =============================================================================
