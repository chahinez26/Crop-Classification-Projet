// =============================================================================
// MCTNet GEE — CLIMATE COVARIATES v3 — TIMESTEP — Part 2 — Arkansas
// =============================================================================
//
// CORRECTION PROF : données climatiques par TIMESTEP (pas annuelles)
//
// PRINCIPE :
//   Exactement comme gee_arkansas_v5.js pour Sentinel-2 :
//   - 36 fenêtres de ~10 jours (T_INDEX = 0 à 35)
//   - Mêmes pixels que Part 1 (seed=42, scale=30, même ZONE_INDEX)
//   - Chaque run produit 1 CSV avec les valeurs climatiques de la fenêtre
//
// VARIABLES CLIMATIQUES (3) — GRIDMET (~4km, USA, 2021 complet) :
//   1. temp_mean   : moyenne de (tmax + tmin)/2 sur la fenêtre de 10j (°C)
//      → capte les variations saisonnières de température par culture
//      → riz : pic T° juillet/août, coton : sensible aux nuits froides avril
//
//   2. precip_total: somme des précipitations sur la fenêtre de 10j (mm)
//      → événements pluvieux corrélés aux stades phénologiques
//      → différencie épisodes secs (stress) vs humides (croissance)
//
//   3. solar_mean  : moyenne du rayonnement solaire sur la fenêtre (W/m²)
//      → énergie disponible pour la photosynthèse par période
//      → discriminant entre cultures d'hiver et d'été
//
// RÉSULTAT FINAL (après merge Python) :
//   X_clim[10000, 36, 3]  → même structure que X_s2[10000, 36, 10]
//   Permet une fusion temporelle cohérente dans MCTNetCov
//
// PROCÉDURE (72 runs) :
//   T_INDEX = 0..35, ZONE_INDEX = 0 et 1
//   → ARK_CLIM_T01_Z0.csv ... ARK_CLIM_T36_Z1.csv
//
// ORDRE RECOMMANDÉ (comme Part 1) :
//   T=0, Z=0 → ARK_CLIM_T01_Z0
//   T=0, Z=1 → ARK_CLIM_T01_Z1
//   T=1, Z=0 → ARK_CLIM_T02_Z0
//   ...
//   T=35,Z=1 → ARK_CLIM_T36_Z1
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CES VARIABLES À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
var T_INDEX    = 0;   // 0 à 35
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — identiques à gee_arkansas_v5.js
// =============================================================================
var YEAR         = 2021;
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_v5_PART2';
var CLASS_VALUES = [0, 1, 2, 3, 4];
var CLASS_POINTS = [760, 380, 1210, 2340, 310];   // 5000 pts par zone

var CDL_CORN    = 1;
var CDL_COTTON  = 2;
var CDL_RICE    = 3;
var CDL_SOYBEAN = 5;

var ZONES = [
  ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),  // Zone 0 : Nord Delta
  ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])   // Zone 1 : Sud Delta
];

var GEOM = ZONES[ZONE_INDEX];
var zStr = '' + ZONE_INDEX;
var tStr = (T_INDEX + 1) < 10 ? '0' + (T_INDEX + 1) : '' + (T_INDEX + 1);

// =============================================================================
// FONCTION : Dates de début/fin pour chaque timestep
// Identique à gee_arkansas_v5.js — 36 timesteps = 12 mois × 3 fenêtres de 10j
// =============================================================================
function windowDates(t, year) {
  var m    = Math.floor(t / 3);
  var w    = t % 3;
  var mStr = (m + 1) < 10 ? '0' + (m + 1) : '' + (m + 1);
  var nxtM = (m === 11) ? 1 : m + 2;
  var nxtY = (m === 11) ? year + 1 : year;
  var nxtMs = nxtM < 10 ? '0' + nxtM : '' + nxtM;
  var nxt  = nxtY + '-' + nxtMs + '-01';
  if (w === 0) return [year + '-' + mStr + '-01', year + '-' + mStr + '-11'];
  if (w === 1) return [year + '-' + mStr + '-11', year + '-' + mStr + '-21'];
  return [year + '-' + mStr + '-21', nxt];
}

// =============================================================================
// FONCTION : Image de labels CDL — identique à gee_arkansas_v5.js
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
// FONCTION : Composite climatique GRIDMET sur une fenêtre de 10 jours
//
// Analogie avec getComposite() Sentinel-2 de gee_arkansas_v5.js :
//   S2  : médiane des réflectances sur la fenêtre → 10 bandes
//   CLIM: agrégation météorologique sur la fenêtre → 3 variables
//
// Gestion pixels manquants :
//   GRIDMET couvre tout le territoire USA sans lacunes (pas de nuages)
//   → unmask(0) utilisé uniquement par sécurité, jamais déclenché en Arkansas
// =============================================================================
function getClimateComposite(geom, start, end) {
  var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterDate(start, end)
    .filterBounds(geom);

  // 1. Température moyenne (garder, mais vérifier la normalisation)
 var temp = gridmet.select(['tmmx','tmmn'])
  .map(function(img) {
    return img.select('tmmx').add(img.select('tmmn'))
      .divide(2).subtract(273.15).rename('temp_mean');
  }).mean().unmask(0).clip(geom);
 
 // 2. REMPLACER precip_total par VPD moyen (stress hydrique)
 //    vpd dans GRIDMET = mean daily vapor pressure deficit (kPa)
 var vpd = gridmet
  .select('vpd')
  .mean()                  // moyenne sur la fenêtre de 10j
  .rename('vpd_mean')
  .unmask(0)
  .clip(geom);

 // 3. Garder solar_mean (rayonnement = bonne variable phénologique)
 var solar = gridmet
  .select('srad')
  .mean()
  .rename('solar_mean')
  .unmask(0)
  .clip(geom);

return temp.addBands(vpd).addBands(solar).toFloat();
}

// =============================================================================
// VARIABLES DU RUN
// =============================================================================
var dates  = windowDates(T_INDEX, YEAR);
var start  = dates[0];
var end    = dates[1];
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE — Climate Covariates v3 (TIMESTEP) ===');
print('Zone      : Z' + zStr);
print('Timestep  : T' + tStr + '  (' + start + ' → ' + end + ')');

// Vérification nombre d'images GRIDMET disponibles pour cette fenêtre
var nImgs = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate(start, end)
  .filterBounds(GEOM)
  .size();
print('Images GRIDMET disponibles :', nImgs);   // attendu : ~10 (1/jour)
print('');

// =============================================================================
// COMPOSITE CLIMATIQUE DE LA FENÊTRE
// =============================================================================
var climComp = getClimateComposite(GEOM, start, end);

// =============================================================================
// ASSEMBLAGE — labels + covariables climatiques
// Même logique que SPECTRAL dans gee_arkansas_v5.js :
//   comp.addBands(labels) → stratifiedSample
// =============================================================================
var imgForSample = climComp.addBands(labels);

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que Part 1
// seed=42, scale=30, classValues, classPoints → MÊMES pixels que S2
// dropNulls=false : garder tous les pixels même si valeur = 0
//   (GRIDMET : 0 précipitation est une valeur valide en été)
// =============================================================================
var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,            // identique Part 1
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,            // MÊME seed → MÊMES pixels que S2
  dropNulls   : false,         // garder même si valeur = 0
  geometries  : true,
  tileScale   : 16
});

// =============================================================================
// VÉRIFICATIONS
// =============================================================================
var nTotal = samples.size();
print('Points extraits :', nTotal);   // doit être 5000
print('');

var names   = {0:'Corn   ', 1:'Cotton ', 2:'Rice   ', 3:'Soybean', 4:'Others '};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (label=' + i + ') — cible ' + targets[i] + ' :', n);
}

// Vérification valeurs (temperature ne peut pas être 0 en Arkansas)
var n_temp_zero = samples.filter(ee.Filter.eq('temp_mean', 0)).size();
print('');
print('Points temp_mean = 0 (anomalie) :', n_temp_zero);
// Attendu : 0 (GRIDMET couvre tout Arkansas sans lacunes)

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_CLIM_T' + tStr + '_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_CLIM_T' + tStr + '_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 9);

Map.addLayer(climComp.select('temp_mean'),
  {min: -5, max: 35, palette: ['blue', 'cyan', 'yellow', 'orange', 'red']},
  'Temp moyenne T' + tStr + ' (°C) Z' + zStr);

Map.addLayer(climComp.select('vpd_mean'),
  {min: 0, max: 4, palette: ['green', 'yellow', 'orange', 'red']},
  'VPD moyen T' + tStr + ' (kPa) Z' + zStr, false);

Map.addLayer(climComp.select('solar_mean'),
  {min: 50, max: 350, palette: ['black', 'orange', 'yellow', 'white']},
  'Rayonnement T' + tStr + ' (W/m²) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points climat T' + tStr + ' Z' + zStr);

print('');
print('✅ Export lancé → ARK_CLIM_T' + tStr + '_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : T_INDEX=' + T_INDEX + ', ZONE_INDEX=1');
} else {
  var nextT = T_INDEX + 1;
  if (nextT <= 35) {
    var nStr = nextT < 10 ? '0' + nextT : '' + nextT;
    print('⏭  Prochain : T_INDEX=' + nextT + ' (T' + nStr + '), ZONE_INDEX=0');
  } else {
    print('🏁 TERMINÉ — 72 fichiers climatiques exportés !');
    print('   → python Part2_Step1_merge_climate_timestep.py');
  }
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à ARK_T**_Z*.csv de Part 1 ✅
// crop_label    ← 0-4 (vérification)
// temp_mean     ← température moyenne fenêtre T (°C)
// vpd_mean      ← VPD moyen fenêtre T (kPa)
// solar_mean    ← rayonnement solaire moyen fenêtre T (W/m²)
// .geo          ← lon/lat identiques à ARK_CDL_Z*.csv ✅
//
// MERGE PYTHON :
//   72 CSV → X_clim[10000, 36, 3]
//   Même logique que Step1_merge.py de Part 1 :
//   joindre par system:index + zone → matrice temporelle
// =============================================================================

// =============================================================================
// RÉCAPITULATIF PROCÉDURE (72 runs)
// =============================================================================
// T=0,  Z=0 → ARK_CLIM_T01_Z0    T=0,  Z=1 → ARK_CLIM_T01_Z1
// T=1,  Z=0 → ARK_CLIM_T02_Z0    T=1,  Z=1 → ARK_CLIM_T02_Z1
// ...
// T=35, Z=0 → ARK_CLIM_T36_Z0    T=35, Z=1 → ARK_CLIM_T36_Z1
//
// FICHIERS DRIVE (MCTNet_v5_PART2/) :
//   ARK_CLIM_T01_Z0.csv ... ARK_CLIM_T36_Z1.csv
//
// DIMENSIONS FINALES APRÈS MERGE :
//   X_clim : [10000, 36, 3]   float32
//   y      : [10000]           int32    (même que Part 1)
//   mask   : [10000, 36]       uint8    (1=missing, toujours 0 pour GRIDMET)
// =============================================================================
