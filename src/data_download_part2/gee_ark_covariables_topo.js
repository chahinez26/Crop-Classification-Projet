// =============================================================================
// MCTNet GEE — COVARIABLES TOPOGRAPHIE — Arkansas  (v2 — consignes prof)
//
// 2 attributs imposés par le prof : elevation + landforms
//
//   1. elevation  : Altitude (m)
//      Source : NOAA/NGDC/ETOPO1  — résolution ~1.8 km (1 arc-minute)
//      Bande  : 'bedrock'
//      Justification : L'altitude segmente le Delta du Mississippi (plaine
//      alluviale 40-100m) des plateaux de l'Ozark/Ouachita (>300m).
//      Les cultures irriguées (riz) se concentrent en basse plaine.
//
//   2. landforms  : Classe de forme de relief (1-11)
//      Source : CSP/ERGo/1_0/Global/ALOS_landforms  — résolution 90m
//      Bande  : 'constant'
//      Classes pertinentes pour l'Arkansas Delta :
//        1 = Peak/ridge → absent du Delta
//        2 = Upper slope → faibles reliefs
//        5 = Flat/plain  → zones rizicoles et soja
//        7 = Valley      → couloirs hydrographiques
//       11 = Bottom flat → plaine alluviale inondable (riz)
//      Justification : Les landforms distinguent directement les
//      fonds de plaine inondables (riz) des coteaux drainés (coton, maïs).
//      C'est une variable catégorielle complémentaire à l'élévation continue.
//
// NOTE : Le prof impose ETOPO1 pour l'élévation (pas SRTM).
//        ETOPO1 a une résolution de ~1.8 km, moins précise que SRTM 30m,
//        mais c'est la source indiquée dans la consigne.
//
// STRATÉGIE : même seed=42, scale=30 → mêmes points que CDL
// PROCÉDURE (2 runs) :
//   Run 1 : ZONE_INDEX=0 → ARK_COV_TOPO_Z0
//   Run 2 : ZONE_INDEX=1 → ARK_COV_TOPO_Z1
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
var FOLDER       = 'MCTNet_Covariables';
var CLASS_VALUES = [0, 1, 2, 3, 4];
var CLASS_POINTS = [760, 380, 1210, 2340, 310];
var CDL_CONF     = 95;

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
// VARIABLE 1 — ÉLÉVATION : ETOPO1
// Collection GEE : 'NOAA/NGDC/ETOPO1'
// Type          : ee.Image (image unique, pas une collection)
// Bandes        : 'bedrock' (topographie du socle rocheux, en mètres)
//                 'ice_surface' (surface de la glace, identique sauf Antarctique)
// Résolution    : ~1 852 m (1 arc-minute)
// Unité         : mètres (entier signé, négatif pour les fonds marins)
// =============================================================================
var etopo1 = ee.Image('NOAA/NGDC/ETOPO1');

var elevation = etopo1
  .select('bedrock')      // bande d'élévation topographique
  .rename('elevation')
  .clip(GEOM);

// =============================================================================
// VARIABLE 2 — LANDFORMS : CSP ERGo ALOS
// Collection GEE : 'CSP/ERGo/1_0/Global/ALOS_landforms'
// Type          : ee.Image (image unique)
// Bande         : 'constant'
// Résolution    : ~90 m (basé sur ALOS DEM 30m agrégé à 90m)
// Unité         : classe entière 1-11
//
// Légende des 11 classes :
//   1  = Peak / Ridge top
//   2  = Upper slope (shoulder)
//   3  = Middle slope
//   4  = Lower slope (footslope)
//   5  = Flat / Plain
//   6  = Valley
//   7  = Valley / Hollow
//   8  = Local ridge / hill in valley
//   9  = Mid-slope ridge
//  10  = Upper slope ridge
//  11  = Bottom flat (plaine de fond, inondable)
//
// Arkansas Delta : classes 5 (plaine) et 11 (fond plat inondable)
// dominent → discriminant pour les rizières
// =============================================================================
var alos_landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms');

var landforms = alos_landforms
  .select('constant')     // seule bande disponible dans ce dataset
  .rename('landforms')
  .clip(GEOM);

// =============================================================================
// COMBINER + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var covTopo = elevation.float().addBands(landforms).addBands(labels);

// =============================================================================
// STATS DE VÉRIFICATION
// =============================================================================
print('=== Covariables Topographie — Arkansas Z' + zStr + ' ===');
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
// EXTRACTION — même seed=42, scale=30, classPoints que ton ami
// =============================================================================
var samples = covTopo.stratifiedSample({
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

print('');
print('Points extraits :', samples.size());
print('(cible : 5000)');
print('');

var names   = {0:'Corn', 1:'Cotton', 2:'Rice', 3:'Soybean', 4:'Others'};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (cible ' + targets[i] + ') :', n);
}

// Vérification : pas de valeurs nulles (ETOPO1 et ALOS sont globaux)
var n_missing = samples.filter(ee.Filter.notNull(['elevation', 'landforms']).not()).size();
print('Points avec valeur manquante (doit être 0) :', n_missing);

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_COV_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_COV_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 10);
Map.addLayer(elevation,
  {min: 40, max: 200, palette: ['#e8f4f8','#a8d5e2','#5b9db5','#2c6e8a','#1a4a5c']},
  'Elevation ETOPO1 (m) Z' + zStr);
Map.addLayer(landforms,
  {min: 1, max: 11, palette: ['#8B0000','#DC143C','#FF8C00','#FFD700',
                               '#9ACD32','#32CD32','#00CED1','#1E90FF',
                               '#4B0082','#FF69B4','#00FA9A']},
  'Landforms ALOS Z' + zStr, false);
Map.addLayer(labels,
  {min: 0, max: 4, palette: ['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

print('');
print('✅ Export lancé → ARK_COV_TOPO_Z' + zStr);
print('   Dossier Drive : ' + FOLDER);
print('');
if (ZONE_INDEX === 0) {
  print('→ Prochain run : ZONE_INDEX = 1');
} else {
  print('→ Les 2 zones Topo sont prêtes !');
  print('');
  print('=== RÉCAPITULATIF FINAL — 6 CSV à télécharger ===');
  print('  Drive/MCTNet_Covariables/');
  print('  ARK_COV_CLIMAT_Z0.csv   ARK_COV_CLIMAT_Z1.csv');
  print('  ARK_COV_SOL_Z0.csv      ARK_COV_SOL_Z1.csv');
  print('  ARK_COV_TOPO_Z0.csv     ARK_COV_TOPO_Z1.csv');
  print('');
  print('→ Lancer : python merge_covariables_v2.py');
  print('→ Sortie : ARK_covariables.npz  [10000, 8] float32');
  print('   (8 features = 3 climat + 3 sol + 2 topo)');
}

// =============================================================================
// LÉGENDE LANDFORMS CSP ERGo ALOS (classe → description)
// =============================================================================
// 1  Peak / Ridge top          → absent du Delta plat
// 2  Upper slope               → faibles reliefs NW Arkansas
// 3  Middle slope              → transition plaine-plateau
// 4  Lower slope               → pied de versant
// 5  Flat / Plain              ← DOMINANT en Arkansas Delta (soja, maïs, riz)
// 6  Valley                    → couloirs des bayous
// 7  Valley / Hollow           → méandres Mississippi
// 8  Local ridge in valley     → levées naturelles
// 9  Mid-slope ridge           → terrasses alluviales
// 10 Upper slope ridge
// 11 Bottom flat               ← plaine inondable = rizières
// =============================================================================
