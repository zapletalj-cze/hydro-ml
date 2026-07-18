"""
Levee Detection — Inference only
=================================
Loads a trained XGBoost model and the pre-built feature stack, runs
chunk-wise inference over the full AOI, and writes:
  levee_probability.tif  — float32 sigmoid probabilities [0, 1]
  levee_prediction.tif   — uint8 binary map (0=background, 1=levee, 255=nodata)
"""

from pathlib import Path

import numpy as np
import xgboost as xgb
from osgeo import gdal

gdal.UseExceptions()

# ============================================================
# CONFIGURATION
# ============================================================

OUT_DIR            = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\02_modeldevelopment\v04')
FEATURE_STACK_PATH = OUT_DIR / 'feature_stack.tif'
MODEL_PATH         = OUT_DIR / 'xgb_levee_model.json'

PIXEL_SIZE           = 10.0
BEST_THR             = 0.349
INFERENCE_CHUNK_ROWS = 256   # rows per batch; reduce if RAM is tight

# ============================================================
# HELPERS
# ============================================================

def write_multiband_tiff(path: Path, arrays: list, band_names: list,
                         geo_info: dict, nodata: float = -9999.0):
    nrows, ncols = arrays[0].shape
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        str(path), ncols, nrows, len(arrays), gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TILED=YES'],
    )
    ds.SetGeoTransform(geo_info['geotransform'])
    ds.SetProjection(geo_info['projection'])
    for i, (arr, name) in enumerate(zip(arrays, band_names), start=1):
        out  = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
        band = ds.GetRasterBand(i)
        band.WriteArray(out)
        band.SetNoDataValue(nodata)
        band.SetDescription(name)
    ds.FlushCache()
    ds = None
    print(f'  Saved: {path.name}  ({len(arrays)} bands, {nrows}x{ncols})')

# ============================================================
# LOAD MODEL
# ============================================================

print(f'Loading model: {MODEL_PATH}')
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))
print(f'  Model loaded.  Threshold: {BEST_THR}')
best_thr = BEST_THR

print('Reading feature stack metadata...')
ds_feat = gdal.Open(str(FEATURE_STACK_PATH), gdal.GA_ReadOnly)
if ds_feat is None:
    raise FileNotFoundError(f'Feature stack not found: {FEATURE_STACK_PATH}')

nrows_inf  = ds_feat.RasterYSize
ncols_inf  = ds_feat.RasterXSize
nbands_inf = ds_feat.RasterCount
gt_inf     = ds_feat.GetGeoTransform()
proj_inf   = ds_feat.GetProjection()
print(f'  Stack size  : {nrows_inf} x {ncols_inf} px  ({nbands_inf} bands)')
print(f'  Total pixels: {nrows_inf * ncols_inf:,}')

proba_map = np.full((nrows_inf, ncols_inf), np.nan, dtype=np.float32)

print(f'Running inference in strips of {INFERENCE_CHUNK_ROWS} rows...')
for row_start in range(0, nrows_inf, INFERENCE_CHUNK_ROWS):
    row_end      = min(row_start + INFERENCE_CHUNK_ROWS, nrows_inf)
    n_chunk_rows = row_end - row_start

    chunk = np.stack([
        ds_feat.GetRasterBand(b + 1).ReadAsArray(
            0, row_start, ncols_inf, n_chunk_rows
        ).astype(np.float32)
        for b in range(nbands_inf)
    ], axis=-1)

    flat  = chunk.reshape(-1, nbands_inf)
    valid = np.isfinite(flat).all(axis=1)

    if valid.any():
        proba_flat = np.full(len(flat), np.nan, dtype=np.float32)
        proba_flat[valid] = model.predict_proba(flat[valid])[:, 1]
        proba_map[row_start:row_end] = proba_flat.reshape(n_chunk_rows, ncols_inf)

    print(f'  Rows {row_end}/{nrows_inf}  '
          f'({row_end / nrows_inf * 100:.1f}%)  '
          f'valid in strip: {valid.sum():,}')

ds_feat = None
print()
print('Inference complete.')
print(f'  Valid pixels scored : {np.isfinite(proba_map).sum():,}')
geo_info_inf = {
    'geotransform': gt_inf,
    'projection':   proj_inf,
    'nrows':        nrows_inf,
    'ncols':        ncols_inf,
}

# Probability map (float32)
proba_path = OUT_DIR / 'levee_probability.tif'
write_multiband_tiff(
    path=proba_path,
    arrays=[proba_map],
    band_names=['levee_probability'],
    geo_info=geo_info_inf,
    nodata=-9999.0,
)

# Binary prediction map (uint8): 0=background, 1=levee, 255=nodata
pred_map = np.where(
    np.isnan(proba_map),
    np.uint8(255),
    (proba_map >= best_thr).astype(np.uint8),
)

pred_path = OUT_DIR / 'levee_prediction.tif'
ds_pred = gdal.GetDriverByName('GTiff').Create(
    str(pred_path), ncols_inf, nrows_inf, 1, gdal.GDT_Byte,
    options=['COMPRESS=LZW', 'PREDICTOR=2', 'NUM_THREADS=ALL_CPUS',
             'BIGTIFF=IF_SAFER', 'TILED=YES'],
)
ds_pred.SetGeoTransform(gt_inf)
ds_pred.SetProjection(proj_inf)
b_pred = ds_pred.GetRasterBand(1)
b_pred.WriteArray(pred_map)
b_pred.SetNoDataValue(255)
b_pred.SetDescription('levee_prediction')
ds_pred.FlushCache()
ds_pred = None

n_levee = int((pred_map == 1).sum())
print(f'Binary prediction map -> {pred_path}')
print(f'  Threshold   : {best_thr:.3f}')
print(f'  Levee pixels: {n_levee:,}')
print(f'  Levee area  : {n_levee * PIXEL_SIZE**2 / 1e6:.2f} km²')
print(f'  Probability map -> {proba_path}')
print()
print('Load both GeoTIFFs in QGIS:')
print('  levee_probability.tif — Singleband pseudocolor (0–1)')
print('  levee_prediction.tif  — Paletted (0=background, 1=levee)')
