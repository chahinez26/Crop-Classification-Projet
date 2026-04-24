// Variables :
//   1. soil_ph       → pH × 10 dans dataset → on garde brut, Step9 divise par 10
//      Justification : riz pH 5.5-6.5, soja pH 6-7, coton tolère pH 5.8-8
//
//   2. soil_oc       → OC en dg/kg dans dataset → on garde brut, Step9 divise par 10
//      Justification : fertilité et rétention d'eau ; delta alluvial riche en OC
//
//   3. soil_texture  → classe USDA 1-12 (catégorielle, pas de conversion)
//      Justification : argile (1-3) = riz/coton ; limon-sable (7-12) = soja/maïs
//
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
// DONNÉES SOL — OpenLandMap (profondeur 0-5 cm, bande 'b0')
// CORRECTION : valeurs brutes conservées + unmask(-1) pour sécurité
// =============================================================================

// 1. pH du sol (valeur brute × 10 dans le dataset)
//    Ex: valeur stockée 65 = pH 6.5
//    La conversion ÷ 10 est faite dans Step9_merge_covariates.py
//    unmask(-1) : signal explicite "donnée manquante" → imputé par Step9
var soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0')
  .rename('soil_ph')
  .unmask(-1)     // ← CORRECTION : évite perte de points par dropNulls
  .clip(GEOM);

// 2. Carbone organique (valeur brute en dg/kg dans le dataset)
//    Ex: valeur 120 = 12.0 g/kg
//    La conversion ÷ 10 est faite dans Step9_merge_covariates.py
var soil_oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0')
  .rename('soil_oc')
  .unmask(-1)     // ← CORRECTION
  .clip(GEOM);

// 3. Classe texturale USDA (catégorielle, 1-12, pas de conversion)
//    1=Clay, 2=Silty Clay, 3=Sandy Clay, 4=Clay Loam, 5=Silty Clay Loam,
//    6=Sandy Clay Loam, 7=Loam, 8=Silty Loam, 9=Sandy Loam,
//    10=Silt, 11=Loamy Sand, 12=Sand
var soil_texture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0')
  .rename('soil_texture')
  .unmask(-1)     // ← CORRECTION
  .clip(GEOM);

// =============================================================================
// ASSEMBLAGE — labels EN PREMIER (classBand), puis sol
// =============================================================================
var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(soil_ph)
  .addBands(soil_oc)
  .addBands(soil_texture)
  .toFloat()
  .addBands(labels, null, true);   // remettre crop_label en int

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que Part 1
// =============================================================================
print('=== MCTNet GEE — Soil Covariates v2 (OpenLandMap) ===');
print('Zone       : Z' + zStr);
print('Profondeur : 0-5 cm (bande b0)');
print('Variables  : soil_ph (brut ×10), soil_oc (brut dg/kg), soil_texture (1-12)');
print('unmask(-1) : pixels null → -1 (imputés par Step9_merge_covariates.py)');
print('');

// Statistiques de vérification
var phStats = soil_ph.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry : GEOM,
  scale    : 250,
  maxPixels: 1e9
});
print('Statistiques soil_ph (brut ×10) Z' + zStr + ' :');
print(phStats);
// Attendu : mean ~60-70 (= pH 6.0-7.0), min ~50, max ~80

var ocStats = soil_oc.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry : GEOM,
  scale    : 250,
  maxPixels: 1e9
});
print('Statistiques soil_oc (brut dg/kg) Z' + zStr + ' :');
print(ocStats);
print('');

var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,            // identique Part 1
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,            // MÊME seed → MÊMES pixels
  dropNulls   : true,          // safe car unmask(-1) appliqué
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

// Vérifier les valeurs nulles résiduelles (-1)
print('');
var n_ph_null = samples.filter(ee.Filter.eq('soil_ph', -1)).size();
var n_oc_null = samples.filter(ee.Filter.eq('soil_oc', -1)).size();
var n_tx_null = samples.filter(ee.Filter.eq('soil_texture', -1)).size();
print('Points soil_ph = -1 (null) :', n_ph_null);
print('Points soil_oc = -1 (null) :', n_oc_null);
print('Points soil_texture = -1 (null) :', n_tx_null);
// Attendu : 0 pour tous (Arkansas bien couvert par OpenLandMap)

print('');
print('Exemple valeurs (5 premiers points) :');
print(samples.limit(5));

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_SOIL_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_SOIL_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 9);

Map.addLayer(soil_ph,
  {min: 50, max: 80, palette: ['red','orange','yellow','green','blue']},
  'pH sol 0-5cm (brut ×10) Z' + zStr);

Map.addLayer(soil_oc,
  {min: 0, max: 150, palette: ['white','lightyellow','yellow','brown','black']},
  'OC sol 0-5cm (brut dg/kg) Z' + zStr, false);

Map.addLayer(soil_texture,
  {min: 1, max: 12, palette: [
    '8B0000','FF0000','FF6347','FFA500','FFD700','ADFF2F',
    '00FF00','00CED1','0000FF','4B0082','8B008B','FF69B4'
  ]},
  'Texture USDA 0-5cm Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points sol Z' + zStr);

print('');
print('✅ Export lancé → ARK_SOIL_Z' + zStr + '.csv');
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
// soil_ph       ← pH brut ×10 (ex: 65 = pH 6.5) | -1 si null (< 0.1%)
// soil_oc       ← OC brut dg/kg (ex: 120 = 12g/kg) | -1 si null
// soil_texture  ← classe USDA 1-12 | -1 si null
// .geo          ← lon/lat identiques à ARK_CDL_Z*.csv ✅
//
// CONVERSIONS FAITES DANS Step9_merge_covariates.py :
//   soil_ph_reel  = soil_ph / 10     (si soil_ph != -1)
//   soil_oc_reel  = soil_oc / 10     (si soil_oc != -1)
//   soil_texture  → inchangée (catégorielle)
//   valeurs -1    → imputées par médiane de classe
// =============================================================================
