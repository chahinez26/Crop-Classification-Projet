// =============================================================================
// MCTNet GEE — CALIFORNIA TOPOGRAPHY COVARIATES — Part 2
// Basé sur gee_california_v2.js (Part 1) + corrections ark_topo_v2.js
//
// Variables topo (2) — imposées par la prof :
//   1. elevation  → USGS SRTM 30m (mètres)
//      → Californie très contrastée : Delta (0-5m), Valley (50-150m),
//        collines côtières (200-800m) → fort pouvoir discriminant
//      → Riz = zones basses inondables (<20m), pistaches/amandes = coteaux
//      → Raisins : vignobles en pente (50-300m altitude optimale)
//
//   2. landforms  → CSP/ERGo ALOS_landforms (classes Weiss 1-16)
//      → Plaines = riz/alfalfa, coteaux = raisins/amandes, fonds de vallée = pistaches
//      → Complément essentiel à l'altitude seule en terrain vallonné
//
// ⚠️  Zones ASYMÉTRIQUES California :
//   Z0 : 5720 points  (Sacramento + Nord San Joaquin)
//   Z1 : 4300 points  (San Joaquin Central + Tulare)
//
// PROCÉDURE (2 runs) :
//   ZONE_INDEX = 0 → CAL_TOPO_Z0.csv  (5720 pts)
//   ZONE_INDEX = 1 → CAL_TOPO_Z1.csv  (4300 pts)
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
// DONNÉES TOPOGRAPHIQUES
// USGS SRTM 30m (identique à ark_topo_v2.js)
// unmask(0) : sécurité perte de points par dropNulls
// =============================================================================

// 1. ELEVATION — USGS SRTM 30m
//    Californie : relief plus varié qu'Arkansas → meilleur pouvoir discriminant
var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation')
  .unmask(0)
  .clip(GEOM);

// 2. LANDFORMS — CSP/ERGo ALOS
var landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms')
  .select('constant')
  .rename('landforms')
  .unmask(0)
  .clip(GEOM);

// =============================================================================
// ASSEMBLAGE + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(elevation)
  .addBands(landforms)
  .toFloat()
  .addBands(labels, null, true);

// =============================================================================
// VÉRIFICATIONS PRÉALABLES
// =============================================================================
print('=== MCTNet GEE — California Topography Covariates (SRTM) ===');
print('Zone      : Z' + zStr);
print('Attendus  : ' + N_EXPECTED + ' points');
print('');

var elevStats = elevation.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry: GEOM, scale: 1000, maxPixels: 1e9
});
print('Statistiques elevation SRTM Z' + zStr + ' :');
print(elevStats);
// Z0 Sacramento : mean ~50-100m (basse plaine + quelques collines)
// Z1 San Joaquin: mean ~100-200m (plaine + coteaux Tulare)
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
print('Exemple valeurs (5 points) :');
print(samples.limit(5));

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 8);

Map.addLayer(elevation,
  {min: 0, max: 600, palette: ['#E8F5E9','#66BB6A','#2E7D32','#795548','#BDBDBD']},
  'Elevation SRTM 30m (m) Z' + zStr);

Map.addLayer(landforms,
  {min: 11, max: 42, palette: [
    '141414','383838','808080','EBEB8F','F7D311',
    'AA0000','D89382','DDC9C9','1C6330','68AA63','B5C98E'
  ]},
  'Landforms ALOS (Weiss) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:5, palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points topo Z' + zStr);

print('');
print('✅ Export lancé → CAL_TOPO_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : ZONE_INDEX = 1  (4300 pts attendus)');
} else {
  print('🏁 Les deux zones extraites !');
  print('   CAL_TOPO_Z0 : 5720 pts  |  CAL_TOPO_Z1 : 4300 pts');
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à CAL_CDL_Z*.csv de Part 1 ✅
// crop_label    ← 0=Grapes,1=Rice,2=Alfalfa,3=Almonds,4=Pistachios,5=Others
// elevation     ← altitude SRTM (m)
// landforms     ← classe Weiss ALOS (11-42)
// .geo          ← lon/lat identiques à CAL_CDL_Z*.csv ✅
// =============================================================================
