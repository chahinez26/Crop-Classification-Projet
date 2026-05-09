
var MODE       = 'CDL_SAMPLE';  
var ZONE_INDEX = 0;              
var T_INDEX    = 0;              
var YEAR       = 2021;
var CDL_CONF   = 95;        
var FOLDER     = 'MCTNet_v5';
var S2_BANDS   = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var BAND_NAMES = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12'];


var CLASS_VALUES  = [0, 1, 2, 3, 4];
var CLASS_POINTS  = [760, 380, 1210, 2340, 310];  


var CDL_CORN    = 1;
var CDL_COTTON  = 2;
var CDL_RICE    = 3;
var CDL_SOYBEAN = 5;


var ZONES = [

  ee.Geometry.Rectangle([-91.50, 34.75, -90.05, 35.85]),


  ee.Geometry.Rectangle([-91.80, 33.15, -90.25, 34.75])
];

var GEOM = ZONES[ZONE_INDEX];


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


function getComposite(geom, start, end) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(start, end)
    .map(function(img) {
      var scl  = img.select('SCL');
      var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6))
                    .or(scl.eq(7)).or(scl.eq(11));
      return img.select(S2_BANDS).rename(BAND_NAMES).updateMask(mask);
    });
  var zero = ee.Image.constant([0,0,0,0,0,0,0,0,0,0]).rename(BAND_NAMES).toFloat().clip(geom);
  return ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.median().unmask(0).toFloat(),
    zero
  ));
}


function getLabelImage(geom) {
  var cdl = ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first();

  var confMask = cdl.select('confidence').gte(CDL_CONF);

  var wcMask = ee.ImageCollection('ESA/WorldCover/v200')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first().select('Map').eq(40);

  var cdlMasked = cdl.select('cropland')
    .updateMask(confMask)
    .updateMask(wcMask)
    .clip(geom);

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
    .where(corn,    0)
    .where(cotton,  1)
    .where(rice,    2)
    .where(soybean, 3)
    .where(others,  4)
    .updateMask(cdlMasked.gt(0))
    .toInt()
    .clip(geom);
}


var tStr   = (T_INDEX + 1) < 10 ? '0' + (T_INDEX + 1) : '' + (T_INDEX + 1);
var zStr   = '' + ZONE_INDEX;
var dates  = windowDates(T_INDEX, YEAR);
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE v5 ===');
print('Mode      : ' + MODE);
print('Zone      : Z' + zStr);
if (MODE === 'SPECTRAL') {
  print('Timestep  : T' + tStr + '  (' + dates[0] + ' → ' + dates[1] + ')');
}
print('');


if (MODE === 'CDL_SAMPLE') {

  print('→ stratifiedSample depuis CDL 2021');
  print('  Cible : Corn=760, Cotton=380, Rice=1210, Soybean=2340, Others=310');
  print('  Total cible : 5000 points par zone');
  print('');


  var points = labels.stratifiedSample({
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


  var cdlRaw = ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
    .first().select('cropland').rename('cdl_raw').clip(GEOM);


  var labelsWithRaw = labels.addBands(cdlRaw);

  var pointsFinal = labelsWithRaw.stratifiedSample({
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

  var total = pointsFinal.size();
  print('Points obtenus (total) :', total);
  print('');

  var names = {0: 'Corn   ', 1: 'Cotton ', 2: 'Rice   ', 3: 'Soybean', 4: 'Others '};
  var targets = {0: 760, 1: 380, 2: 1210, 3: 2340, 4: 310};
  for (var i = 0; i < 5; i++) {
    var n = pointsFinal.filter(ee.Filter.eq('crop_label', i)).size();
    print(names[i] + ' (label=' + i + ') — cible ' + targets[i] + ' :', n);
  }

  Export.table.toDrive({
    collection     : pointsFinal,
    description    : 'ARK_CDL_Z' + zStr,
    folder         : FOLDER,
    fileNamePrefix : 'ARK_CDL_Z' + zStr,
    fileFormat     : 'CSV'
  });


  Map.centerObject(GEOM, 10);
  Map.addLayer(labels,
    {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
    'CDL Classes Z' + zStr);
  Map.addLayer(pointsFinal.draw({color:'FFFF00', pointRadius:2}),
    {}, 'Points Z' + zStr);

  print('');
  print('✅ Export lancé → ARK_CDL_Z' + zStr);
  print('');
  print('PROCÉDURE :');
  print('  1. Lancer CDL_SAMPLE Z0 → ARK_CDL_Z0');
  print('  2. Lancer CDL_SAMPLE Z1 → ARK_CDL_Z1');
  print('  3. Vérifier ~5000 pts/zone avec bonne distribution');
  print('  4. Passer à MODE="SPECTRAL", T_INDEX=0..35');


} else if (MODE === 'SPECTRAL') {

  print('→ Composite Sentinel-2 T' + tStr);
  print('  Fenêtre : ' + dates[0] + ' → ' + dates[1]);

  var nImgs = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(GEOM).filterDate(dates[0], dates[1]).size();
  print('  Images S2 disponibles :', nImgs);
  print('');

  var comp = getComposite(GEOM, dates[0], dates[1]);


  var imgForSample = comp.addBands(labels);

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

  var nTotal   = samples.size();
  var nMissing = samples.filter(ee.Filter.eq('B02', 0)).size();
  print('Points extraits :', nTotal);
  print('Points missing (B02=0) :', nMissing);
  print('');

  Export.table.toDrive({
    collection     : samples,
    description    : 'ARK_T' + tStr + '_Z' + zStr,
    folder         : FOLDER,
    fileNamePrefix : 'ARK_T' + tStr + '_Z' + zStr,
    fileFormat     : 'CSV'
  });


  Map.centerObject(GEOM, 10);
  Map.addLayer(comp.select(['B04','B03','B02']),
    {min:0, max:3000, gamma:1.4}, 'RGB T' + tStr + ' Z' + zStr);
  Map.addLayer(labels,
    {min:0, max:4, palette:['4CAF50','F44336','2196F3','FF9800','9E9E9E']},
    'CDL Classes Z' + zStr, false);

  print('✅ Export lancé → ARK_T' + tStr + '_Z' + zStr);
  if (ZONE_INDEX === 0) {
    print('⏭  Prochain : T_INDEX=' + T_INDEX + ', ZONE_INDEX=1');
  } else {
    var nextT = T_INDEX + 1;
    if (nextT <= 35) {
      var nStr = nextT < 10 ? '0' + nextT : '' + nextT;
      print('⏭  Prochain : T_INDEX=' + nextT + ' (T' + nStr + '), ZONE_INDEX=0');
    } else {
      print('🏁 TERMINÉ — tous les timesteps exportés !');
    }
  }

} else {
  print('❌ MODE inconnu : "' + MODE + '"');
  print('   Utiliser "CDL_SAMPLE" ou "SPECTRAL"');
}


