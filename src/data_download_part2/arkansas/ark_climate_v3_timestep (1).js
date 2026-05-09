











































var ZONE_INDEX = 0;   
var T_INDEX    = 0;   





var YEAR         = 2021;
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
var tStr = (T_INDEX + 1) < 10 ? '0' + (T_INDEX + 1) : '' + (T_INDEX + 1);





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












function getClimateComposite(geom, start, end) {
  var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterDate(start, end)
    .filterBounds(geom);

  
 var temp = gridmet.select(['tmmx','tmmn'])
  .map(function(img) {
    return img.select('tmmx').add(img.select('tmmn'))
      .divide(2).subtract(273.15).rename('temp_mean');
  }).mean().unmask(0).clip(geom);
 
 
 
 var vpd = gridmet
  .select('vpd')
  .mean()                  
  .rename('vpd_mean')
  .unmask(0)
  .clip(geom);

 
 var solar = gridmet
  .select('srad')
  .mean()
  .rename('solar_mean')
  .unmask(0)
  .clip(geom);

return temp.addBands(vpd).addBands(solar).toFloat();
}




var dates  = windowDates(T_INDEX, YEAR);
var start  = dates[0];
var end    = dates[1];
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE — Climate Covariates v3 (TIMESTEP) ===');
print('Zone      : Z' + zStr);
print('Timestep  : T' + tStr + '  (' + start + ' → ' + end + ')');


var nImgs = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate(start, end)
  .filterBounds(GEOM)
  .size();
print('Images GRIDMET disponibles :', nImgs);   
print('');




var climComp = getClimateComposite(GEOM, start, end);






var imgForSample = climComp.addBands(labels);







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




var nTotal = samples.size();
print('Points extraits :', nTotal);   
print('');

var names   = {0:'Corn   ', 1:'Cotton ', 2:'Rice   ', 3:'Soybean', 4:'Others '};
var targets = {0:760, 1:380, 2:1210, 3:2340, 4:310};
for (var i = 0; i < 5; i++) {
  var n = samples.filter(ee.Filter.eq('crop_label', i)).size();
  print(names[i] + ' (label=' + i + ') — cible ' + targets[i] + ' :', n);
}


var n_temp_zero = samples.filter(ee.Filter.eq('temp_mean', 0)).size();
print('');
print('Points temp_mean = 0 (anomalie) :', n_temp_zero);





Export.table.toDrive({
  collection     : samples,
  description    : 'ARK_CLIM_T' + tStr + '_Z' + zStr,
  folder         : FOLDER,
  fileNamePrefix : 'ARK_CLIM_T' + tStr + '_Z' + zStr,
  fileFormat     : 'CSV'
});




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

































