














var ZONE_INDEX = 0;   





var CDL_CONF     = 95;
var FOLDER       = 'MCTNet_v5_PART2';
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





var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(GEOM);


print('=== MCTNet GEE — Climate Covariates v2 (GRIDMET) ===');
print('Zone    : Z' + zStr);
print('Images GRIDMET 2021 disponibles :', gridmet.size());





var temp_mean = gridmet
  .select(['tmmx', 'tmmn'])
  .map(function(img) {
    var tmax = img.select('tmmx');
    var tmin = img.select('tmmn');
    return tmax.add(tmin).divide(2)
      .subtract(273.15)          
      .rename('temp_mean')
      .copyProperties(img, ['system:time_start']);
  })
  .mean()                        
  .clip(GEOM);



var precip_total = gridmet
  .select('pr')
  .sum()                         
  .rename('precip_total')
  .clip(GEOM);



var solar_mean = gridmet
  .select('srad')
  .mean()                        
  .rename('solar_mean')
  .clip(GEOM);




var labels = getLabelImage(GEOM);

var imgForSample = labels
  .addBands(temp_mean)
  .addBands(precip_total)
  .addBands(solar_mean)
  .toFloat()
  
  .addBands(labels, null, true);





print('');
print('Extraction en cours...');
print('Cible : 5000 pts (760/380/1210/2340/310)');
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
print('Vérification valeurs (5 premiers points) :');
print(samples.limit(5));




Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_CLIMATE_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_CLIMATE_Z' + zStr,
  fileFormat     : 'CSV'
});




Map.centerObject(GEOM, 9);

Map.addLayer(temp_mean,
  {min: 10, max: 25, palette: ['blue', 'yellow', 'red']},
  'Température moyenne 2021 (°C) Z' + zStr);

Map.addLayer(precip_total,
  {min: 800, max: 1600, palette: ['white', 'cyan', 'blue', 'darkblue']},
  'Précipitation totale 2021 (mm) Z' + zStr, false);

Map.addLayer(solar_mean,
  {min: 150, max: 280, palette: ['black', 'orange', 'yellow', 'white']},
  'Rayonnement solaire moyen (W/m²) Z' + zStr, false);

Map.addLayer(labels,
  {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
  'CDL Classes Z' + zStr, false);

Map.addLayer(samples.draw({color: 'FFFF00', pointRadius: 2}),
  {}, 'Points climatiques Z' + zStr);

print('');
print('✅ Export lancé → ARK_CLIMATE_Z' + zStr + '.csv');
if (ZONE_INDEX === 0) {
  print('⏭  Prochain : ZONE_INDEX = 1');
} else {
  print('🏁 Les deux zones extraites !');
}











