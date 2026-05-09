

var ZONE_INDEX = 0;   


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


var soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0').rename('soil_ph')
  .unmask(-1).clip(GEOM);


var soil_oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0').rename('soil_oc')
  .unmask(-1).clip(GEOM);


var soil_texture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0').rename('soil_texture')
  .unmask(-1).clip(GEOM);

var slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003'))
  .rename('slope')
  .unmask(0)
  .clip(GEOM);


var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(soil_ph)
  .addBands(soil_oc)
  .addBands(soil_texture)
  .toFloat()
  .addBands(labels, null, true);


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


print('');


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


Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_SOIL_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_SOIL_Z' + zStr,
  fileFormat     : 'CSV'
});


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


