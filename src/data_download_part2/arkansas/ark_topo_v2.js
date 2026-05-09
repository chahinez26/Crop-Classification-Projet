














var ZONE_INDEX = 0;   





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








var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation')
  .unmask(0)             
  .clip(GEOM);



var landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms')
  .select('constant')
  .rename('landforms')
  .unmask(0)             
  .clip(GEOM);




var labels = getLabelImage(GEOM);




var imgForSample = labels
  .addBands(elevation)
  .addBands(landforms)
  .toFloat()
  .addBands(labels, null, true);   




print('=== MCTNet GEE — Topography Covariates v2 (SRTM) ===');
print('Zone      : Z' + zStr);
print('elevation : USGS/SRTMGL1_003 (30m) ← CORRECTION vs ETOPO1 (1.8km)');
print('landforms : CSP/ERGo/1_0/Global/ALOS_landforms');
print('');


var elevStats = elevation.reduceRegion({
  reducer  : ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry : GEOM,
  scale    : 1000,
  maxPixels: 1e9
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




Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_TOPO_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_TOPO_Z' + zStr,
  fileFormat     : 'CSV'
});




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










