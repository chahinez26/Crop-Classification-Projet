

var MODE       = 'CDL_SAMPLE';  
var ZONE_INDEX = 0;              
var T_INDEX    = 0;              


var YEAR       = 2021;
var CDL_CONF   = 95;
var FOLDER     = 'MCTNet_California_v2';
var S2_BANDS   = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var BAND_NAMES = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12'];


var CLASS_VALUES = [0, 1, 2, 3, 4, 5];


var CLASS_POINTS_Z0 = [1030, 2030, 490, 390, 20, 1760];


var CLASS_POINTS_Z1 = [1030, 10, 490, 390, 620, 1760];


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
  var zero = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
               .rename(BAND_NAMES).toFloat().clip(geom);
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
  var wcMask   = ee.ImageCollection('ESA/WorldCover/v200')
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


var tStr   = (T_INDEX + 1) < 10 ? '0' + (T_INDEX + 1) : '' + (T_INDEX + 1);
var zStr   = '' + ZONE_INDEX;
var dates  = windowDates(T_INDEX, YEAR);
var labels = getLabelImage(GEOM);

print('=== MCTNet GEE — California v2 (zones asymétriques) ===');
print('Mode      : ' + MODE);
print('Zone      : Z' + zStr);
if (ZONE_INDEX === 0) {
  print('CLASS_POINTS Z0 : Grapes=1030, Rice=2030, Alfalfa=490,');
  print('                  Almonds=390, Pistachios=20, Others=1760');
} else {
  print('CLASS_POINTS Z1 : Grapes=1030, Rice=10, Alfalfa=490,');
  print('                  Almonds=390, Pistachios=620, Others=1760');
}
if (MODE === 'SPECTRAL') {
  print('Timestep  : T' + tStr + '  (' + dates[0] + ' → ' + dates[1] + ')');
}
print('');


if (MODE === 'CDL_SAMPLE') {

  print('→ stratifiedSample depuis CDL 2021 — California v2');
  print('');

  var labelsWithRaw = labels.addBands(
    ee.ImageCollection('USDA/NASS/CDL')
      .filter(ee.Filter.date('2021-01-01', '2022-01-01'))
      .first().select('cropland').rename('cdl_raw').clip(GEOM)
  );

  var points = labelsWithRaw.stratifiedSample({
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

  print('Points obtenus (total) :', points.size());
  print('');

  var classNames = ['Grapes    ','Rice      ','Alfalfa   ',
                    'Almonds   ','Pistachios','Others    '];
  for (var i = 0; i < 6; i++) {
    var cible = CLASS_POINTS[i];
    var n     = points.filter(ee.Filter.eq('crop_label', i)).size();
    print(classNames[i] + ' (label=' + i + ') — cible ' + cible + ' :', n);
  }

  Export.table.toDrive({
    collection     : points,
    description    : 'CAL_CDL_Z' + zStr,
    folder         : FOLDER,
    fileNamePrefix : 'CAL_CDL_Z' + zStr,
    fileFormat     : 'CSV'
  });

  Map.centerObject(GEOM, 9);
  Map.addLayer(labels, {
    min:0, max:5,
    palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']
  }, 'CDL Classes Z' + zStr);
  Map.addLayer(points.draw({color:'FFFF00', pointRadius:2}), {}, 'Points Z' + zStr);

  print('');
  print('✅ Export lancé → CAL_CDL_Z' + zStr + ' (folder: ' + FOLDER + ')');
  print('');
  if (ZONE_INDEX === 0) {
    print('→ Prochain run : ZONE_INDEX=1');
    print('  Vérifier que Pistachios Z1 = 620 ✅');
    print('  et que Rice Z0 = 2030 ✅');
  } else {
    print('→ Les 2 zones CDL sont prêtes !');
    print('  Vérifier les totaux Z0+Z1 :');
    print('  Grapes=2060, Rice=2040, Alfalfa=980,');
    print('  Almonds=780, Pistachios=640, Others=3520');
    print('  → Passer à MODE="SPECTRAL", T_INDEX=0..35');
  }


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

  print('Points extraits :', samples.size());
  print('Points missing (B02=0) :', samples.filter(ee.Filter.eq('B02', 0)).size());

  Export.table.toDrive({
    collection     : samples,
    description    : 'CAL_T' + tStr + '_Z' + zStr,
    folder         : FOLDER,
    fileNamePrefix : 'CAL_T' + tStr + '_Z' + zStr,
    fileFormat     : 'CSV'
  });

  Map.centerObject(GEOM, 9);
  Map.addLayer(comp.select(['B04','B03','B02']),
    {min:0, max:3000, gamma:1.4}, 'RGB T' + tStr + ' Z' + zStr);
  Map.addLayer(labels, {
    min:0, max:5,
    palette:['9400D3','2196F3','FF9800','8B4513','90EE90','9E9E9E']
  }, 'CDL Classes Z' + zStr, false);

  print('');
  print('✅ Export lancé → CAL_T' + tStr + '_Z' + zStr);
  if (ZONE_INDEX === 0) {
    print('⏭  Prochain : T_INDEX=' + T_INDEX + ', ZONE_INDEX=1');
  } else {
    var nextT = T_INDEX + 1;
    if (nextT <= 35) {
      var nStr = nextT < 10 ? '0' + nextT : '' + nextT;
      print('⏭  Prochain : T_INDEX=' + nextT + ' (T' + nStr + '), ZONE_INDEX=0');
    } else {
      print('🏁 TERMINÉ — tous les timesteps California exportés !');
    }
  }

} else {
  print('❌ MODE inconnu : "' + MODE + '"');
}


