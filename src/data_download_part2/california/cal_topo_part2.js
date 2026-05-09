


























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









var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation')
  .unmask(0)
  .clip(GEOM);






var slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003'))
  .rename('slope')
  .unmask(0)
  .clip(GEOM);




var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(elevation)
  .addBands(landforms)
  .toFloat()
  .addBands(labels, null, true);




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
print('Exemple valeurs (5 points) :');
print(samples.limit(5));




Export.table.toDrive({
  collection     : samples,
  description    : 'CAL_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'CAL_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});




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










