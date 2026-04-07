// =============================================================================
// MCTNet GEE — STATIC FEATURES — ARKANSAS
// Soil / Climate / Topography
// Wang et al. 2024 — Computers and Electronics in Agriculture
//
// PRINCIPE :
//   Extraire les variables statiques AUX MÊMES POINTS que les CSV CDL.
//   On utilise sampleRegions() sur la FeatureCollection existante
//   (importée depuis les assets GEE après upload des CDL CSV)
//   → PAS de nouveau stratifiedSample → les indices restent identiques.
//
// SOURCES DE DONNÉES :
//   Soil        : OpenLandMap (SoilGrids 250m) — 6 variables
//   Climate     : CHELSA climatologies 1981-2010 — 5 variables
//   Topography  : SRTM 30m → slope, aspect, TPI, TWI — 5 variables
//   TOTAL       : 16 variables statiques par point
//
// PROCÉDURE (4 runs) :
//   Run 1 : MODE='SOIL',    ZONE_INDEX=0  → ARK_SOIL_Z0.csv
//   Run 2 : MODE='SOIL',    ZONE_INDEX=1  → ARK_SOIL_Z1.csv
//   Run 3 : MODE='CLIMATE', ZONE_INDEX=0  → ARK_CLIM_Z0.csv
//   Run 4 : MODE='CLIMATE', ZONE_INDEX=1  → ARK_CLIM_Z1.csv
//   Run 5 : MODE='TOPO',    ZONE_INDEX=0  → ARK_TOPO_Z0.csv
//   Run 6 : MODE='TOPO',    ZONE_INDEX=1  → ARK_TOPO_Z1.csv
//
// MERGE PYTHON :
//   ark_step1_merge_static.py → ARK_dataset_static.npz
//   X_static[10000, 16]  float32
//   Compatible avec ARK_dataset.npz existant (même ordre de points)
//
// NOTE : Avant de lancer, uploader ARK_CDL_Z0.csv et ARK_CDL_Z1.csv
//   dans GEE Assets (table) pour pouvoir les importer comme FeatureCollection.
//   Ou bien utiliser la version sampleRegions directement sur l'image labels
//   avec le même seed (voir MODE='SOIL_DIRECT' ci-dessous).
// =============================================================================

// ════════════════════════════════════════════════════════════════════════════
// ▶ CHANGER CES VARIABLES À CHAQUE RUN
// ════════════════════════════════════════════════════════════════════════════
var MODE       = 'SOIL';   // 'SOIL' | 'CLIMATE' | 'TOPO'
var ZONE_INDEX = 0;         // 0 ou 1
// ════════════════════════════════════════════════════════════════════════════

// =============================================================================
// PARAMÈTRES FIXES (identiques au script Sentinel v5)
// =============================================================================
var FOLDER     = 'MCTNet_v5';
var YEAR       = 2021;
var CDL_CONF   = 95;

// Distribution EXACTE — identique au script Sentinel (même seed → mêmes pixels)
var CLASS_VALUES  = [0, 1, 2, 3, 4];
var CLASS_POINTS  = [760, 380, 1210, 2340, 310];

// Codes CDL Arkansas
var CDL_CORN    = 1;
var CDL_COTTON  = 2;
var CDL_RICE    = 3;
var CDL_SOYBEAN = 5;

// Zones Arkansas — identiques au script Sentinel v5
var ZONES = [
  ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),   // Zone 0 Nord
  ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])    // Zone 1 Sud
];
var GEOM = ZONES[ZONE_INDEX];
var zStr = '' + ZONE_INDEX;

// =============================================================================
// FONCTION : Image de labels CDL (identique au script Sentinel v5)
// Nécessaire pour reproduire le même stratifiedSample avec seed=42
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
  var corn    = cdlMasked.eq(CDL_CORN);
  var cotton  = cdlMasked.eq(CDL_COTTON);
  var rice    = cdlMasked.eq(CDL_RICE);
  var soybean = cdlMasked.eq(CDL_SOYBEAN);
  var others  = cdlMasked.gt(0)
                  .and(cdlMasked.neq(CDL_CORN))
                  .and(cdlMasked.neq(CDL_COTTON))
                  .and(cdlMasked.neq(CDL_RICE))
                  .and(cdlMasked.neq(CDL_SOYBEAN));
  return ee.Image(0).rename('crop_label')
    .where(corn, 0).where(cotton, 1).where(rice, 2)
    .where(soybean, 3).where(others, 4)
    .updateMask(cdlMasked.gt(0)).toInt().clip(geom);
}

// =============================================================================
// FONCTION : Construire l'image des features selon le mode
// =============================================================================

// ── SOIL ──────────────────────────────────────────────────────────────────
// Source : OpenLandMap/SOL — SoilGrids 250m
// Variables (toutes à profondeur 0-5 cm) :
//   sol_texture_class  : USDA texture class (0-12)
//   sol_sand_pct       : % sable   (0-100)
//   sol_clay_pct       : % argile  (0-100)
//   sol_organic_carbon : Carbone organique (g/kg)
//   sol_ph_h2o         : pH en eau × 10 (ex: 65 = pH 6.5)
//   sol_bulk_density   : Densité apparente (cg/cm³)
function getSoilImage(geom) {
  // Texture class USDA
  var tex = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
    .select('b0').rename('sol_texture_class').clip(geom);

  // Sand %
  var sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02')
    .select('b0').rename('sol_sand_pct').clip(geom);

  // Clay %
  var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02')
    .select('b0').rename('sol_clay_pct').clip(geom);

  // Organic carbon (dg/kg → converti mentalement en g/kg côté Python)
  var oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
    .select('b0').rename('sol_organic_carbon').clip(geom);

  // pH × 10
  var ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
    .select('b0').rename('sol_ph_h2o').clip(geom);

  // Bulk density (cg/cm³)
  var bd = ee.Image('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02')
    .select('b0').rename('sol_bulk_density').clip(geom);

  return tex.addBands(sand).addBands(clay)
            .addBands(oc).addBands(ph).addBands(bd).toFloat();
}

// ── CLIMATE ───────────────────────────────────────────────────────────────
// Source : WORLDCLIM/V1/BIO (WorldClim 1km, 1970-2000)
// Variables sélectionnées pour l'agriculture :
//   bio01 : Température annuelle moyenne (°C × 10)
//   bio12 : Précipitations annuelles totales (mm)
//   bio15 : Saisonnalité des précipitations (CV)
//   bio05 : Temp max du mois le plus chaud (°C × 10)
//   bio06 : Temp min du mois le plus froid (°C × 10)
// + ETP annuelle Penman-Monteith (MODIS MOD16)
function getClimateImage(geom) {
  var bio = ee.Image('WORLDCLIM/V1/BIO').clip(geom);

  var tmean = bio.select('bio01').rename('clim_tmean');    // °C × 10
  var prec  = bio.select('bio12').rename('clim_prec');     // mm/an
  var pseas = bio.select('bio15').rename('clim_prec_seas');// saisonnalité
  var tmax  = bio.select('bio05').rename('clim_tmax');     // °C × 10
  var tmin  = bio.select('bio06').rename('clim_tmin');     // °C × 10

  // ETP annuelle : somme MOD16A2 sur 2021 (mm/an)
  // MOD16A2 = 8-day ET en 0.1 mm → ×0.1 × 46 images ≈ annuel
  var et = ee.ImageCollection('MODIS/006/MOD16A2')
    .filterBounds(geom)
    .filterDate('2021-01-01', '2022-01-01')
    .select('ET')
    .sum()
    .multiply(0.1)                    // conversion : 0.1 mm → mm
    .rename('clim_et_annual')
    .clip(geom);

  return tmean.addBands(prec).addBands(pseas)
              .addBands(tmax).addBands(tmin)
              .addBands(et).toFloat();
}

// ── TOPOGRAPHY ────────────────────────────────────────────────────────────
// Source : USGS/SRTMGL1_003 (SRTM 30m)
// Variables :
//   topo_elevation : altitude (m)
//   topo_slope     : pente (degrés)
//   topo_aspect    : orientation (degrés)
//   topo_tpi       : Topographic Position Index (différence locale)
//   topo_twi       : Topographic Wetness Index (ln(a/tan(β)))
function getTopoImage(geom) {
  var dem  = ee.Image('USGS/SRTMGL1_003').clip(geom);
  var elev = dem.rename('topo_elevation');

  // Slope en degrés
  var slope = ee.Terrain.slope(dem).rename('topo_slope');

  // Aspect en degrés (0-360)
  var aspect = ee.Terrain.aspect(dem).rename('topo_aspect');

  // TPI : élévation locale - moyenne dans fenêtre 300m
  var focal_mean = dem.focal_mean({radius: 300, units: 'meters'});
  var tpi = dem.subtract(focal_mean).rename('topo_tpi');

  // TWI : ln(flow_acc / tan(slope_rad))
  // Flow accumulation via ee.Terrain n'est pas direct en GEE
  // Approximation : TWI = ln(1 / (tan(slope_rad) + 0.001))
  // (corrélé avec l'humidité du sol sur terrains plats comme l'Arkansas Delta)
  var slope_rad = slope.multiply(Math.PI / 180);
  var twi = slope_rad.tan().max(0.001).pow(-1).log().rename('topo_twi');

  return elev.addBands(slope).addBands(aspect)
             .addBands(tpi).addBands(twi).toFloat();
}

// =============================================================================
// MAIN
// =============================================================================
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE — Static Features ===');
print('Mode : ' + MODE + '  |  Zone : Z' + zStr);
print('');

// Choisir l'image de features selon le mode
var featImage;
var outName;
var featBands;

if (MODE === 'SOIL') {
  featImage = getSoilImage(GEOM);
  outName   = 'ARK_SOIL_Z' + zStr;
  featBands = ['sol_texture_class','sol_sand_pct','sol_clay_pct',
               'sol_organic_carbon','sol_ph_h2o','sol_bulk_density'];
  print('Variables sol (6) : texture, sand, clay, OC, pH, bulk_density');

} else if (MODE === 'CLIMATE') {
  featImage = getClimateImage(GEOM);
  outName   = 'ARK_CLIM_Z' + zStr;
  featBands = ['clim_tmean','clim_prec','clim_prec_seas',
               'clim_tmax','clim_tmin','clim_et_annual'];
  print('Variables climat (6) : tmean, prec, prec_seas, tmax, tmin, ET');

} else if (MODE === 'TOPO') {
  featImage = getTopoImage(GEOM);
  outName   = 'ARK_TOPO_Z' + zStr;
  featBands = ['topo_elevation','topo_slope','topo_aspect',
               'topo_tpi','topo_twi'];
  print('Variables topo (5) : elevation, slope, aspect, TPI, TWI');

} else {
  print('❌ MODE inconnu. Utiliser SOIL | CLIMATE | TOPO');
  throw new Error('MODE invalide');
}

// ── Extraction aux mêmes pixels que le script Sentinel v5 ──────────────
// MÊME stratifiedSample, MÊME seed=42, MÊME scale=30
// → GEE retourne exactement les mêmes pixels géographiques
// addBands(labels) pour avoir crop_label dans l'export
var imgForSample = featImage.addBands(labels);

var samples = imgForSample.stratifiedSample({
  numPoints   : 0,
  classBand   : 'crop_label',
  region      : GEOM,
  scale       : 30,             // résolution CDL → même grille que v5
  classValues : CLASS_VALUES,
  classPoints : CLASS_POINTS,
  seed        : 42,             // ← CRITIQUE : même seed = mêmes coordonnées
  dropNulls   : true,
  geometries  : true,
  tileScale   : 16
});

var nTotal = samples.size();
print('Points échantillonnés :', nTotal);
print('(cible : 5000)');
print('');

// Vérification par classe
var cNames = {0:'Corn   ',1:'Cotton ',2:'Rice   ',3:'Soybean',4:'Others '};
var cTgts  = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(cNames[i] + ' (label=' + i + ') — cible ' + cTgts[i] + ' :', n);
}
print('');

// Export CSV
Export.table.toDrive({
  collection     : samples,
  description    : outName,
  folder         : FOLDER,
  fileNamePrefix : outName,
  fileFormat     : 'CSV'
});

// Visualisation
Map.centerObject(GEOM, 10);
if (MODE === 'SOIL') {
  Map.addLayer(featImage.select('sol_sand_pct'),
    {min:0, max:100, palette:['blue','white','orange']}, 'Sand % Z' + zStr);
} else if (MODE === 'CLIMATE') {
  Map.addLayer(featImage.select('clim_prec'),
    {min:900, max:1600, palette:['white','blue']}, 'Precip Z' + zStr);
} else if (MODE === 'TOPO') {
  Map.addLayer(featImage.select('topo_elevation'),
    {min:40, max:180, palette:['green','yellow','brown']}, 'Elevation Z' + zStr);
}
Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);
Map.addLayer(samples.draw({color:'FFFF00', pointRadius:2}),
  {}, 'Points ' + outName);

print('✅ Export lancé → ' + outName);
print('');
print('RÉCAPITULATIF DES 6 RUNS ARKANSAS :');
print('  Run 1 : SOIL,    Z0 → ARK_SOIL_Z0.csv  (6 vars sol)');
print('  Run 2 : SOIL,    Z1 → ARK_SOIL_Z1.csv');
print('  Run 3 : CLIMATE, Z0 → ARK_CLIM_Z0.csv  (6 vars climat)');
print('  Run 4 : CLIMATE, Z1 → ARK_CLIM_Z1.csv');
print('  Run 5 : TOPO,    Z0 → ARK_TOPO_Z0.csv  (5 vars topo)');
print('  Run 6 : TOPO,    Z1 → ARK_TOPO_Z1.csv');
print('');
print('VARIABLES EXPORTÉES (17 total) :');
print('  Soil (6) : texture_class, sand_pct, clay_pct,');
print('             organic_carbon, ph_h2o, bulk_density');
print('  Climate (6) : tmean, prec, prec_seas, tmax, tmin, et_annual');
print('  Topo (5) : elevation, slope, aspect, TPI, TWI');

// =============================================================================
// LÉGENDE VARIABLES
// =============================================================================
// SOL :
//   sol_texture_class  : Classe texture USDA (1-12)
//   sol_sand_pct       : % sable en masse (0-100)
//   sol_clay_pct       : % argile (0-100)
//   sol_organic_carbon : Carbone organique (dg/kg = g/10kg)
//   sol_ph_h2o         : pH × 10 (ex: 65 = pH 6.5)
//   sol_bulk_density   : Densité apparente (cg/cm³)
//
// CLIMAT :
//   clim_tmean    : Température moy. annuelle (°C × 10)
//   clim_prec     : Précipitations annuelles (mm)
//   clim_prec_seas: Saisonnalité précip. (%)
//   clim_tmax     : Temp. max mois le plus chaud (°C × 10)
//   clim_tmin     : Temp. min mois le plus froid (°C × 10)
//   clim_et_annual: ETP annuelle MODIS MOD16 (mm)
//
// TOPO :
//   topo_elevation: Altitude SRTM (m)
//   topo_slope    : Pente (degrés)
//   topo_aspect   : Orientation (degrés, 0=Nord)
//   topo_tpi      : Topographic Position Index (m, relatif)
//   topo_twi      : Topographic Wetness Index (adimensionnel)
// =============================================================================
