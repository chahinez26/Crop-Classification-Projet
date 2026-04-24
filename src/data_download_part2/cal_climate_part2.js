
// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CES VARIABLES À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
var T_INDEX    = 0;   // 0 à 35
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES — identiques à gee_california_v2.js
// =============================================================================
var YEAR       = 2021;
var CDL_CONF   = 95;
var FOLDER     = 'MCTNet_California_v2';
var CLASS_VALUES = [0, 1, 2, 3, 4, 5];

var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20,  1760];  // 5720 pts
var CLASS_POINTS_Z1 = [1030, 10,   490, 390, 620, 1760];  // 4300 pts

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
var tStr         = (T_INDEX + 1) < 10 ? '0' + (T_INDEX + 1) : '' + (T_INDEX + 1);
var N_EXPECTED   = (ZONE_INDEX === 0) ? 5720 : 4300;

// =============================================================================
// FONCTION : Dates début/fin pour chaque timestep
// Identique à ark_climate_v3_timestep.js
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
// FONCTION : Labels CDL — identique à gee_california_v2.js
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
// FONCTION : Composite climatique GRIDMET sur fenêtre 10 jours
// Différence vs Arkansas : VPD remplace precip_total
// =============================================================================
function getClimateComposite(geom, start, end) {
  var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterDate(start, end)
    .filterBounds(geom);

  // 1. Température moyenne de la fenêtre (°C)
  var temp = gridmet
    .select(['tmmx', 'tmmn'])
    .map(function(img) {
      return img.select('tmmx').add(img.select('tmmn'))
        .divide(2)
        .subtract(273.15)
        .rename('temp_mean')
        .copyProperties(img, ['system:time_start']);
    })
    .mean()
    .unmask(0)
    .clip(geom);

  // 2. VPD moyen de la fenêtre (kPa)
  //    vpd = mean daily vapor pressure deficit
  //    Faible VPD → air humide → riz irrigué, zones Delta humides
  //    Fort VPD  → air sec/chaud → amandes, pistaches, zones sud CA
  //    Bien plus discriminant que precip en CA (irrigation omniprésente)
  var vpd = gridmet
    .select('vpd')
    .mean()
    .rename('vpd_mean')
    .unmask(0)
    .clip(geom);

  // 3. Rayonnement solaire moyen de la fenêtre (W/m²)
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
var dates    = windowDates(T_INDEX, YEAR);
var start    = dates[0];
var end      = dates[1];
var labels   = getLabelImage(GEOM);
var climComp = getClimateComposite(GEOM, start, end);

print('=== MCTNet GEE — California Climate v3 (TIMESTEP + VPD) ===');
print('Zone      : Z' + zStr + '  (' + N_EXPECTED + ' pts attendus)');
print('Timestep  : T' + tStr + '  (' + start + ' → ' + end + ')');
print('Variables : temp_mean | vpd_mean | solar_mean');

var nImgs = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate(start, end).filterBounds(GEOM).size();
print('Images GRIDMET disponibles :', nImgs);  // attendu ~10
print('');

// Vérification VPD — ne peut pas être 0 en Californie (T° toujours > 0)
var vpdStats = climComp.select('vpd_mean').reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry : GEOM,
  scale    : 1000,
  maxPixels: 1e9
});
print('Stats VPD T' + tStr + ' Z' + zStr + ' (kPa) :');
print(vpdStats);
// Attendu : mean 0.3-2.5 kPa selon saison ; 0 en hiver possible

// =============================================================================
// ASSEMBLAGE
// =============================================================================
var imgForSample = climComp.addBands(labels);

// =============================================================================
// STRATIFIED SAMPLE — MÊMES paramètres que gee_california_v2.js
// seed=42, scale=30 → MÊMES pixels que S2 Part 1
// dropNulls=false : VPD=0 est valide en hiver
// =============================================================================
var samples = imgForSample.stratifiedSample({
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

// =============================================================================
// VÉRIFICATIONS
// =============================================================================
print('Points extraits :', samples.size());
print('Attendu         : ' + N_EXPECTED);
print('');

var classNames = ['Grapes    ','Rice      ','Alfalfa   ',
                  'Almonds   ','Pistachios','Others    '];
for (var i = 0; i < 6; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(classNames[i] + ' (label=' + i + ') — cible ' + CLASS_POINTS[i] + ' :', n);
}

var n_temp_zero = samples.filter(ee.Filter.eq('temp_mean', 0)).size();
print('');
print('Points temp_mean = 0 (anomalie) :', n_temp_zero);
// Attendu : 0 (T° ne peut pas être 0°C en CA en plein été)

// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_CLIM_T' + tStr + '_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_CLIM_T' + tStr + '_Z' + zStr,
  fileFormat     : 'CSV'
});

// =============================================================================
// VISUALISATION
// =============================================================================
Map.centerObject(GEOM, 8);

Map.addLayer(climComp.select('temp_mean'),
  {min: -5, max: 40, palette: ['blue','cyan','yellow','orange','red']},
  'Temp moyenne T' + tStr + ' (°C) Z' + zStr);

Map.addLayer(climComp.select('vpd_mean'),
  {min: 0, max: 3, palette: ['white','lightyellow','orange','red','darkred']},
  'VPD moyen T' + tStr + ' (kPa) Z' + zStr, false);

Map.addLayer(climComp.select('solar_mean'),
  {min: 50, max: 350, palette: ['black','orange','yellow','white']},
  'Rayonnement T' + tStr + ' (W/m²) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:5, palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points climat T' + tStr + ' Z' + zStr);

print('');
print('✅ Export lancé → CAL_CLIM_T' + tStr + '_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : T_INDEX=' + T_INDEX + ', ZONE_INDEX=1');
} else {
  var nextT = T_INDEX + 1;
  if (nextT <= 35) {
    var nStr = nextT < 10 ? '0' + nextT : '' + nextT;
    print('⏭  Prochain : T_INDEX=' + nextT + ' (T' + nStr + '), ZONE_INDEX=0');
  } else {
    print('🏁 TERMINÉ — 72 fichiers climatiques California exportés !');
    print('   → python Part2_Step1_merge_climate_timestep_california.py');
  }
}

// =============================================================================
// COLONNES DU CSV
// =============================================================================
// system:index  ← identique à CAL_T**_Z*.csv de Part 1 ✅
// crop_label    ← 0=Grapes,1=Rice,2=Alfalfa,3=Almonds,4=Pistachios,5=Others
// temp_mean     ← température moyenne fenêtre T (°C)
// vpd_mean      ← VPD moyen fenêtre T (kPa)      ← nouveau vs Arkansas
// solar_mean    ← rayonnement solaire moyen T (W/m²)
// .geo          ← lon/lat identiques à CAL_CDL_Z*.csv ✅
//
// MERGE PYTHON :
//   72 CSV → X_clim[10020, 36, 3]
//   Même logique que Arkansas mais N=10020 (5720+4300)
// =============================================================================