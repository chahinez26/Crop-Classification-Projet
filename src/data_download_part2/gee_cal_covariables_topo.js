// =============================================================================
// MCTNet GEE — COVARIABLES TOPOGRAPHIE — CALIFORNIA  (v2 — consignes prof)
//
// 2 attributs imposés par le prof : elevation + landforms
//
//   1. elevation  : Altitude (m)
//      Source : NOAA/NGDC/ETOPO1  — résolution ~1.8 km (1 arc-minute)
//      Bande  : 'bedrock'
//      Justification : L'altitude segmente la vallée centrale (0-200m)
//      des piémonts de la Sierra Nevada (>500m). Les cultures de riz
//      se concentrent dans la plaine inondable proche de Sacramento.
//
//   2. landforms  : Classe de forme de relief (1-11)
//      Source : CSP/ERGo/1_0/Global/ALOS_landforms  — résolution 90m
//      Bande  : 'constant'
//      Classes pertinentes pour la Californie :
//        5 = Flat/Plain  → vallée centrale (riz, luzerne, tomate)
//       11 = Bottom flat → zones inondables (riz)
//        6/7 = Valley    → couloirs hydrographiques
//      Justification : Distingue les plaines cultivables des reliefs.
//
// NOTE : Le prof impose ETOPO1 pour l'élévation (pas SRTM).
//        ETOPO1 a une résolution de ~1.8 km.
//
// STRATÉGIE : même seed=42, scale=30 → mêmes points que CDL
// PROCÉDURE (2 runs) :
//   Run 1 : ZONE_INDEX=0 → CAL_COV_TOPO_Z0
//   Run 2 : ZONE_INDEX=1 → CAL_COV_TOPO_Z1
//
// OUTPUT CSV columns :
//   system:index, crop_label, elevation, landforms, .geo
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — identiques à ton ami
// =============================================================================
var YEAR         = 2021;
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_Covariables';
var CLASS_VALUES = [0, 1, 2, 3, 4, 5];
// CLASS_POINTS ASYMÉTRIQUES par zone
// Z0 : beaucoup de Rice, très peu de Pistachios
var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];
// Z1 : beaucoup de Pistachios, très peu de Rice
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
// VARIABLE 1 — ÉLÉVATION : ETOPO1
// =============================================================================
var etopo1 = ee.Image('NOAA/NGDC/ETOPO1');
var elevation = etopo1
  .select('bedrock')
  .rename('elevation')
  .clip(GEOM);

// =============================================================================
// VARIABLE 2 — LANDFORMS : CSP ERGo ALOS
// =============================================================================
var alos_landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms');
var landforms = alos_landforms
  .select('constant')
  .rename('landforms')
  .clip(GEOM);

// =============================================================================
// COMBINER + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

// Conversion en float pour elevation seulement (landforms reste entier)
var covTopo = elevation.float()
  .addBands(landforms)
  .addBands(labels);

// =============================================================================
// STATS DE VÉRIFICATION
// =============================================================================
print('=== Covariables Topographie — California Z' + zStr + ' ===');
print('');
print('Variable 1 : elevation');
print('  Source  : NOAA/NGDC/ETOPO1  (résolution ~1.85 km)');
print('  Bande   : bedrock  (m)');
print('');
print('Variable 2 : landforms');
print('  Source  : CSP/ERGo/1_0/Global/ALOS_landforms  (résolution ~90m)');
print('  Bande   : constant  (classe 1-11)');
print('');

print('elevation stats (m) :', elevation.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 1852,
  maxPixels: 1e8
}));

print('landforms (fréquences) :', landforms.reduceRegion({
  reducer  : ee.Reducer.frequencyHistogram(),
  geometry : GEOM,
  scale    : 90,
  maxPixels: 1e8
}));

// =============================================================================
// EXTRACTION — même seed=42, scale=30, classPoints que CDL
// =============================================================================
var samples = covTopo.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,
  dropNulls   : false,
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

// Vérification des valeurs manquantes (remplace l'ancien test sur temp_mean)
var n_missing = samples.filter(ee.Filter.notNull(['elevation', 'landforms']).not()).size();
print('');
print('Points avec valeurs manquantes (doit être 0) :', n_missing);

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_COV_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_COV_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 10);
Map.addLayer(elevation,
  {min: 0, max: 300, palette: ['#e8f4f8','#a8d5e2','#5b9db5','#2c6e8a','#1a4a5c']},
  'Elevation ETOPO1 (m) Z' + zStr);
Map.addLayer(landforms,
  {min: 1, max: 11, palette: ['#8B0000','#DC143C','#FF8C00','#FFD700',
                               '#9ACD32','#32CD32','#00CED1','#1E90FF',
                               '#4B0082','#FF69B4','#00FA9A']},
  'Landforms ALOS Z' + zStr, false);
// Palette corrigée : 6 classes (0 à 5)
Map.addLayer(labels,
  {min: 0, max: 5, palette: ['#4CAF50','#F44336','#2196F3','#FF9800','#9E9E9E','#888888']},
  'CDL Classes Z' + zStr, false);

// =============================================================================
// MESSAGES FINAUX
// =============================================================================
print('');
print('✅ Export lancé → CAL_COV_TOPO_Z' + zStr);
print('   Dossier Drive : ' + FOLDER);
print('');
if (ZONE_INDEX === 0) {
  print('→ Prochain run : changer ZONE_INDEX = 1');
} else {
  print('→ Les 2 zones Topo Californie sont prêtes !');
  print('   Fichiers : CAL_COV_TOPO_Z0.csv  et  CAL_COV_TOPO_Z1.csv');
  print('   Passer au merge Python.');
}

// =============================================================================
// LÉGENDE LANDFORMS CSP ERGo ALOS (classe → description)
// =============================================================================
// 1  Peak / Ridge top
// 2  Upper slope
// 3  Middle slope
// 4  Lower slope
// 5  Flat / Plain              ← DOMINANT en vallée centrale
// 6  Valley
// 7  Valley / Hollow
// 8  Local ridge in valley
// 9  Mid-slope ridge
// 10 Upper slope ridge
// 11 Bottom flat               ← plaine inondable (riz)
// =============================================================================