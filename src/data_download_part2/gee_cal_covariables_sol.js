// =============================================================================
// MCTNet GEE — COVARIABLES SOL — CALansas  (v2 — consignes prof)
//
// 3 attributs imposés par le prof : PH, OC, Texture
//
//   1. soil_ph      : pH du sol 0-5cm × 10
//      Source : OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02
//      Justification : Le pH conditionne la disponibilité des nutriments.
//      Le riz préfère pH 5.5-6.5 (sol acide), le coton pH 6-7.5 (neutre)
//      → variable discriminante entre rizières et cultures à fibres.
//
//   2. soil_oc      : Carbone organique 0-5cm (dg/kg)
//      Source : OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02
//      Justification : L'OC reflète la fertilité et la rétention d'eau du sol.
//      Les sols alluviaux du Delta (rizières) ont un OC plus élevé que les
//      uplands à coton → signal complémentaire au pH.
//
//   3. soil_texture : Classe texturale USDA (1-12)
//      Source : OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02
//      Justification : La texture contrôle la perméabilité du sol.
//      Les rizières du Delta d'CALansas sont sur sols argileux imperméables
//      (classe 12 = clay) favorisant la rétention d'eau pour l'inondation.
//      Le maïs et le soja préfèrent les limons sableux (classes 4-6).
//
// STRATÉGIE : même seed=42, scale=30, classPoints → mêmes points que CDL
// PROCÉDURE (2 runs) :
//   Run 1 : ZONE_INDEX=0 → CAL_COV_SOL_Z0
//   Run 2 : ZONE_INDEX=1 → CAL_COV_SOL_Z1
//
// OUTPUT CSV columns :
//   system:index, crop_label, soil_ph, soil_oc, soil_texture, .geo
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CETTE VARIABLE À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var ZONE_INDEX = 0;   // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES 
// =============================================================================
var YEAR         = 2021;
var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_Covariables';
var CLASS_VALUES = [0, 1, 2, 3, 4, 5];
// CLASS_POINTS ASYMÉTRIQUES par zone
// Z0 : prend tout le Rice, très peu de Pistachios
var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];

// Z1 : prend tout le Pistachios, très peu de Rice
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
// VARIABLES PÉDOLOGIQUES — OpenLandMap SoilGrids 250m
// Profondeur : bande 'b0' = surface (0-5 cm) pour toutes les variables
//
// Unités :
//   soil_ph      → pH × 10  (ex: 65 = pH 6.5)
//   soil_oc      → dg/kg    (diviser par 10 pour g/kg en Python)
//   soil_texture → classe USDA TT (entier 1-12)
//                  1=Cl, 2=SiCl, 3=SaCl, 4=ClLo, 5=SiClLo, 6=SaClLo,
//                  7=Lo, 8=SiLo, 9=SaLo, 10=Si, 11=LoSa, 12=Sa
// =============================================================================

// 1. pH du sol 0-5 cm
var soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0')
  .rename('soil_ph')
  .clip(GEOM);

// 2. Carbone organique 0-5 cm (dg/kg)
var soil_oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0')
  .rename('soil_oc')
  .clip(GEOM);

// 3. Classe texturale USDA 0-5 cm (1-12)
var soil_texture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0')
  .rename('soil_texture')
  .clip(GEOM);

// =============================================================================
// COMBINER + LABELS
// =============================================================================
var labels = getLabelImage(GEOM);

var covSoil = soil_ph
  .addBands(soil_oc)
  .addBands(soil_texture)
  .addBands(labels);

// =============================================================================
// STATS DE VÉRIFICATION
// =============================================================================
print('=== Covariables Sol — CALIFORNIA Z' + zStr + ' ===');
print('');
print('Source : OpenLandMap / SoilGrids 250m  (résolution 250m)');
print('Profondeur : bande b0 = 0-5 cm');
print('');
print('Variables imposées par le prof :');
print('  soil_ph      : pH × 10  (ex: 65 = pH 6.5)');
print('  soil_oc      : Carbone organique (dg/kg)');
print('  soil_texture : Classe texturale USDA (1-12)');
print('');

print('soil_ph stats :', soil_ph.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 250,
  maxPixels: 1e8
}));

print('soil_oc stats :', soil_oc.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
             .combine(ee.Reducer.minMax(), null, true),
  geometry : GEOM,
  scale    : 250,
  maxPixels: 1e8
}));

print('soil_texture (fréquences) :', soil_texture.reduceRegion({
  reducer  : ee.Reducer.frequencyHistogram(),
  geometry : GEOM,
  scale    : 250,
  maxPixels: 1e8
}));

// =============================================================================
// EXTRACTION — même seed=42, scale=30, classPoints que ton ami
// =============================================================================
var samples = covSoil.stratifiedSample({
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


// Vérification absence de nulls (ERA5 est un raster global sans trous)
var n_null_temp = samples.filter(ee.Filter.eq('temp_mean', 0)).size();
print('');
print('Points avec temp_mean=0 (doit être 0) :', n_null_temp);


// =============================================================================
// EXPORT
// =============================================================================
Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_COV_SOL_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_COV_SOL_Z' + zStr,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 10);
Map.addLayer(soil_ph,
  {min: 50, max: 80, palette: ['red', 'yellow', 'blue']},
  'pH sol × 10  Z' + zStr);
Map.addLayer(soil_oc,
  {min: 0, max: 40, palette: ['white', '#90EE90', 'dCALgreen']},
  'Carbone organique (dg/kg) Z' + zStr, false);
Map.addLayer(soil_texture,
  {min: 1, max: 12, palette: ['#f5deb3','#deb887','#d2691e','#8b4513',
                               '#a0522d','#6b4226','#8b7355','#c4a882',
                               '#bc9b6a','#e8d5b7','#f4e4c1','#fffacd']},
  'Texture USDA Z' + zStr, false);
Map.addLayer(labels,
  {min: 0, max: 4, palette: ['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

print('');
print('✅ Export lancé → CAL_COV_SOL_Z' + zStr);
print('   Dossier Drive : ' + FOLDER);
print('');
if (ZONE_INDEX === 0) {
  print('→ Prochain run : ZONE_INDEX = 1');
} else {
  print('→ Les 2 zones Sol sont prêtes !');
  print('   Passer au script TOPO (gee_CAL_covariables_topo_v2.js)');
}

// =============================================================================
// CORRESPONDANCE CLASSES TEXTURE USDA
// =============================================================================
// 1  = Clay (Cl)              → rizières Delta
// 2  = Silty Clay (SiCl)
// 3  = Sandy Clay (SaCl)
// 4  = Clay Loam (ClLo)
// 5  = Silty Clay Loam (SiClLo)
// 6  = Sandy Clay Loam (SaClLo)
// 7  = Loam (Lo)              → soja, maïs
// 8  = Silt Loam (SiLo)
// 9  = Sandy Loam (SaLo)
// 10 = Silt (Si)
// 11 = Loamy Sand (LoSa)
// 12 = Sand (Sa)              → zones drainées, coton
// =============================================================================
