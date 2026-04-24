// Variables topographiques choisies (2) :
//   1. elevation  → USGS SRTM 30m (mètres)
//      Justification : le Delta Arkansas est très plat (30-100m).
//      Même 1-5m de variation distingue zones drainées (maïs/soja)
//      vs zones basses inondables (riz). Variable topographique principale.
//
//   2. landforms  → CSP/ERGo ALOS_landforms (classes Weiss 1-16)
//      Justification : complète l'élévation en capturant la position
//      dans le paysage. Riz = vallées/bas-fonds (classes 41-42),
//      coton = terrasses surélevées (classes 21-24).
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
var FOLDER       = 'MCTNet_v5';
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
// DONNÉES TOPOGRAPHIQUES
// =============================================================================

// 1. ELEVATION — USGS SRTM (30m, terres émergées uniquement)
//    CORRECTION : remplace ETOPO1 (1.8km, bandes marines) par SRTM (30m, terrestre)
//    unmask(0) : sécurité si pixel isolé sans valeur → sera 0m (impossible dans ARK Delta)
var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation')
  .unmask(0)             // ← sécurité : aucun null ne causera dropNulls
  .clip(GEOM);

// 2. LANDFORMS — CSP/ERGo ALOS (inchangé, dataset correct)
//    unmask(0) : sécurité pour dropNulls
var landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms')
  .select('constant')
  .rename('landforms')
  .unmask(0)             // ← sécurité
  .clip(GEOM);

// =============================================================================
// ASSEMBLAGE — labels EN PREMIER (classBand), puis covariables
// =============================================================================
var labels = getLabelImage(GEOM);

// ORDRE IMPORTANT :
//   labels en premier → classBand='crop_label' toujours trouvée
//   covariables ensuite → extraites aux mêmes pixels
var imgForSample = labels
  .addBands(elevation)
  .addBands(landforms)
  .toFloat()
  .addBands(labels, null, true);   // remettre crop_label en int pour stratifiedSample

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que Part 1
// =============================================================================
print('=== MCTNet GEE — Topography Covariates v2 (SRTM) ===');
print('Zone      : Z' + zStr);
print('elevation : USGS/SRTMGL1_003 (30m) ← CORRECTION vs ETOPO1 (1.8km)');
print('landforms : CSP/ERGo/1_0/Global/ALOS_landforms');
print('');

// Vérification : SRTM a bien des valeurs dans la zone
var elevStats = elevation.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry : GEOM,
  scale    : 1000,
  maxPixels: 1e9
});
print('Statistiques elevation SRTM Z' + zStr + ' :');
print(elevStats);
// Attendu : mean ~50-80m, min ~30m, max ~120m pour le Delta Arkansas

print('');

var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,            // identique Part 1
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,            // MÊME seed → MÊMES pixels
  dropNulls   : true,          // safe car unmask(0) appliqué
  geometries  : true,
  tileScale   : 16
});

// =============================================================================
// VÉRIFICATIONS
// =============================================================================
print('Points extraits :', samples.size());   // doit être 5000
print('');

var names   = {0:'Corn   ', 1:'Cotton ', 2:'Rice   ', 3:'Soybean', 4:'Others '};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (label=' + i + ') — cible ' + targets[i] + ' :', n);
}

print('');
print('Exemple valeurs (5 premiers points) :');
print(samples.limit(5));

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 9);

Map.addLayer(elevation,
  {min: 30, max: 120, palette: ['#E8F5E9', '#66BB6A', '#2E7D32', '#1B5E20']},
  'Elevation SRTM 30m (m) Z' + zStr);

Map.addLayer(landforms,
  {min: 11, max: 42, palette: [
    '141414','383838','808080','EBEB8F','F7D311',
    'AA0000','D89382','DDC9C9','1C6330','68AA63','B5C98E'
  ]},
  'Landforms ALOS (Weiss) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points topo Z' + zStr);

print('');
print('✅ Export lancé → ARK_TOPO_Z' + zStr + '.csv');
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
// elevation     ← altitude SRTM en mètres (30m résolution)
// landforms     ← classe Weiss ALOS (11-42 ou 1-16 selon version)
// .geo          ← lon/lat identiques à ARK_CDL_Z*.csv ✅
// =============================================================================
