"""
Levee Detection from Sentinel-1 SAR
Feature Engineering → Training Data → XGBoost

Stack:
  - Raster I/O: GDAL directly (osgeo.gdal)
  - Vector I/O: GeoPandas + pyogrio
  - Texture: PyTorch GLCM (GPU accelerated)
  - Model: XGBoost

Expected input:
  - Preprocessed Sentinel-1 GeoTIFFs: sigma0 VV and VH, linear units, EPSG:2180, 10 m
  - pyroSAR output naming: S1A__IW___A_YYYYMMDDTHHMMSS_VV_grd_elp.tif
  - BDOT10k GeoPackage with layer OT_BUZM_L
"""

import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn.functional as F
import xgboost as xgb
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from tqdm import tqdm

gdal.UseExceptions()

# ── Environment info ──────────────────────────────────────────────────────────
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'NumPy       : {np.__version__}')

cuda_ok = torch.cuda.is_available()
print(f'CUDA available  : {cuda_ok}')
if cuda_ok:
    print(f'CUDA version    : {torch.version.cuda}')
    print(f'cuDNN version   : {torch.backends.cudnn.version()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB')


# ============================================================
# SECTION 0 — CONFIGURATION
# ============================================================

# --- Input directories -------------------------------------------------------
ASC_DIR      = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\sentinel1_data\processed\ascending')
DESC_DIR     = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\sentinel1_data\processed\descending')
BDOT10K_PATH = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg')
AOI_PATH     = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\AOI_Poland.gpkg')

# --- Output directory --------------------------------------------------------
OUT_DIR = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\02_modeldevelopment\v04')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Coordinate system -------------------------------------------------------
EPSG       = 2180
PIXEL_SIZE = 10.0

# --- GLCM parameters ---------------------------------------------------------
GLCM_WINDOW = 7
GLCM_LEVELS = 64
GLCM_DIST   = 1

# --- Ground truth filter thresholds ------------------------------------------
NASYP_MIN_LENGTH = 1000.0
NASYP_MIN_HEIGHT = 2.0
NASYP_MIN_WIDTH  = 12.0

# --- Levee rasterization buffer ----------------------------------------------
LEVEE_BUFFER = 10.0   # 1 pixel at 10 m

# --- Sampling ----------------------------------------------------------------
N_POSITIVE   = 50_000
N_NEGATIVE   = 50_000
RANDOM_STATE = 42

# --- Spatial block split -----------------------------------------------------
BLOCK_SIZE = 256    # 256 px × 10 m = 2.56 km

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Configuration loaded.')


# ============================================================
# SECTION 1 — GDAL UTILITY FUNCTIONS
# ============================================================

def read_band(path: Path) -> tuple:
    """Reads a single-band GeoTIFF. Returns (float32 array, geo_info dict)."""
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo_info = {
        'geotransform': ds.GetGeoTransform(),
        'projection':   ds.GetProjection(),
        'nrows':        ds.RasterYSize,
        'ncols':        ds.RasterXSize,
    }
    ds = None
    return arr, geo_info


def write_multiband_tiff(path: Path, arrays: list, band_names: list,
                         geo_info: dict, nodata: float = -9999.0):
    """Writes a list of 2-D arrays as a LZW-compressed multiband GeoTIFF."""
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
# SECTION 1.5 — AOI EXTENT & WARP TARGET
# ============================================================

def load_aoi_extent(aoi_path: Path, target_epsg: int = EPSG) -> dict:
    aoi = gpd.read_file(aoi_path, engine='pyogrio')
    if aoi.crs and aoi.crs.to_epsg() != target_epsg:
        aoi = aoi.to_crs(epsg=target_epsg)
    bounds = aoi.total_bounds
    snap = 10.0
    extent = {
        'xmin': float(np.floor(bounds[0] / snap) * snap),
        'ymin': float(np.floor(bounds[1] / snap) * snap),
        'xmax': float(np.ceil(bounds[2]  / snap) * snap),
        'ymax': float(np.ceil(bounds[3]  / snap) * snap),
    }
    print(f'AOI extent (EPSG:{target_epsg}):')
    print(f'  xmin={extent["xmin"]:.1f}  ymin={extent["ymin"]:.1f}')
    print(f'  xmax={extent["xmax"]:.1f}  ymax={extent["ymax"]:.1f}')
    return extent


aoi_extent = load_aoi_extent(AOI_PATH)

# --- User hook: apply custom extent transformation here ----------------------
# Example:
#   from my_library import adjust_extent
#   aoi_extent = adjust_extent(aoi_extent, ...)

print(f'Final AOI extent:')
print(f'  xmin={aoi_extent["xmin"]:.1f}  ymin={aoi_extent["ymin"]:.1f}')
print(f'  xmax={aoi_extent["xmax"]:.1f}  ymax={aoi_extent["ymax"]:.1f}')


def warp_to_aoi(src_path: Path, aoi_path: Path, extent: dict,
                pixel_size: float = PIXEL_SIZE, target_epsg: int = EPSG) -> np.ndarray:
    """Warps a single-band GeoTIFF to the AOI extent. Returns float32 array."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(target_epsg)
    warp_opts = gdal.WarpOptions(
        format='MEM',
        outputBounds=(extent['xmin'], extent['ymin'],
                      extent['xmax'], extent['ymax']),
        xRes=pixel_size,
        yRes=pixel_size,
        dstSRS=srs.ExportToWkt(),
        resampleAlg='near',
        cutlineDSName=str(aoi_path),
        cropToCutline=False,
        dstNodata=np.nan,
        outputType=gdal.GDT_Float32,
    )
    ds = gdal.Warp('', str(src_path), options=warp_opts)
    if ds is None:
        raise RuntimeError(f'gdal.Warp failed for {src_path}')
    arr    = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    ds = None
    return arr


def build_aoi_geo_ref(extent: dict, pixel_size: float = PIXEL_SIZE,
                      epsg: int = EPSG) -> dict:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ncols = int(round((extent['xmax'] - extent['xmin']) / pixel_size))
    nrows = int(round((extent['ymax'] - extent['ymin']) / pixel_size))
    return {
        'geotransform': (extent['xmin'], pixel_size, 0.0,
                         extent['ymax'], 0.0, -pixel_size),
        'projection': srs.ExportToWkt(),
        'nrows': nrows,
        'ncols': ncols,
    }


aoi_geo_ref = build_aoi_geo_ref(aoi_extent)
print(f'AOI grid: {aoi_geo_ref["nrows"]} x {aoi_geo_ref["ncols"]} px')


# ============================================================
# SECTION 2 — MULTI-TEMPORAL AVERAGING (WELFORD)
# ============================================================

CACHE_DIR = OUT_DIR / 'stack_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STACK_KEYS = [
    'asc_vv_mean', 'asc_vv_std',
    'asc_vh_mean', 'asc_vh_std',
    'desc_vv_mean', 'desc_vv_std',
    'desc_vh_mean', 'desc_vh_std',
]


def save_cache(arrays: dict, geo_ref: dict):
    np.savez_compressed(CACHE_DIR / 'stacks.npz', **arrays)
    with open(CACHE_DIR / 'geo_ref.json', 'w') as f:
        json.dump({
            'geotransform': list(geo_ref['geotransform']),
            'projection':   geo_ref['projection'],
            'nrows':        geo_ref['nrows'],
            'ncols':        geo_ref['ncols'],
        }, f, indent=2)
    print(f'Cache saved -> {CACHE_DIR}')


def load_cache() -> tuple:
    npz_path  = CACHE_DIR / 'stacks.npz'
    json_path = CACHE_DIR / 'geo_ref.json'
    if not npz_path.exists() or not json_path.exists():
        return None, None
    arrays = dict(np.load(npz_path))
    with open(json_path) as f:
        raw = json.load(f)
    geo_ref = {
        'geotransform': tuple(raw['geotransform']),
        'projection':   raw['projection'],
        'nrows':        raw['nrows'],
        'ncols':        raw['ncols'],
    }
    print(f'Cache loaded from {CACHE_DIR}')
    return arrays, geo_ref


def cache_is_valid() -> bool:
    npz_path  = CACHE_DIR / 'stacks.npz'
    json_path = CACHE_DIR / 'geo_ref.json'
    if not npz_path.exists() or not json_path.exists():
        return False
    with np.load(npz_path) as f:
        return all(k in f for k in STACK_KEYS)


def collect_scenes(directory: Path, polarization: str) -> list:
    for pattern in [
        f'*_{polarization}_sigma0-elp.tif',
        f'*_{polarization}_gamma0-elp.tif',
        f'*_{polarization}_grd_elp.tif',
        f'*{polarization}*.tif',
    ]:
        scenes = sorted(directory.glob(pattern))
        if scenes:
            return scenes
    return []


def temporal_stack(directory: Path, polarization: str) -> tuple:
    """Welford online mean/std, one scene at a time, AOI-clipped."""
    scenes = collect_scenes(directory, polarization)
    if not scenes:
        raise FileNotFoundError(f'No scenes for {polarization} in {directory}')
    print(f'  {polarization}: {len(scenes)} scenes in {directory.name}')

    nrows, ncols = aoi_geo_ref['nrows'], aoi_geo_ref['ncols']
    count = np.zeros((nrows, ncols), dtype=np.int32)
    mean  = np.zeros((nrows, ncols), dtype=np.float64)
    M2    = np.zeros((nrows, ncols), dtype=np.float64)
    skipped = 0

    for scene in tqdm(scenes, desc=f'  Welford {polarization}'):
        try:
            x = warp_to_aoi(scene, AOI_PATH, aoi_extent)
            valid = ~np.isnan(x)
            count[valid] += 1
            delta  = np.where(valid, x - mean, 0.0)
            mean  += np.where(valid, delta / np.maximum(count, 1), 0.0)
            delta2 = np.where(valid, x - mean, 0.0)
            M2    += delta * delta2
        except Exception as e:
            print(f'  Warning: skipping {scene.name} — {e}')
            skipped += 1

    if skipped:
        print(f'  Skipped {skipped} scene(s).')

    valid_mask = count > 0
    out_mean   = np.where(valid_mask, mean, np.nan).astype(np.float32)
    out_std    = np.where(valid_mask, np.sqrt(M2 / np.maximum(count, 1)),
                          np.nan).astype(np.float32)
    print(f'  Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}')
    return out_mean, out_std


geo_ref = aoi_geo_ref

if cache_is_valid():
    print('Cache found — loading stack arrays...')
    _arrays, geo_ref = load_cache()
    asc_vv_mean  = _arrays['asc_vv_mean']
    asc_vv_std   = _arrays['asc_vv_std']
    asc_vh_mean  = _arrays['asc_vh_mean']
    asc_vh_std   = _arrays['asc_vh_std']
    desc_vv_mean = _arrays['desc_vv_mean']
    desc_vv_std  = _arrays['desc_vv_std']
    desc_vh_mean = _arrays['desc_vh_mean']
    desc_vh_std  = _arrays['desc_vh_std']
else:
    print('No cache — running Welford stacking (AOI-clipped)...')
    print('\nAscending stack...')
    asc_vv_mean,  asc_vv_std  = temporal_stack(ASC_DIR,  'VV')
    asc_vh_mean,  asc_vh_std  = temporal_stack(ASC_DIR,  'VH')
    print('\nDescending stack...')
    desc_vv_mean, desc_vv_std = temporal_stack(DESC_DIR, 'VV')
    desc_vh_mean, desc_vh_std = temporal_stack(DESC_DIR, 'VH')
    save_cache({
        'asc_vv_mean':  asc_vv_mean,  'asc_vv_std':  asc_vv_std,
        'asc_vh_mean':  asc_vh_mean,  'asc_vh_std':  asc_vh_std,
        'desc_vv_mean': desc_vv_mean, 'desc_vv_std': desc_vv_std,
        'desc_vh_mean': desc_vh_mean, 'desc_vh_std': desc_vh_std,
    }, geo_ref)

print(f'\nSection 2 complete. Array shape: {asc_vv_mean.shape}')


# ============================================================
# SECTION 3 — FEATURE ENGINEERING
# ============================================================

# --- 3.1 VH/VV ratio ---------------------------------------------------------

def compute_ratio(vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(vv > 0, vh / vv, np.nan)
    return ratio.astype(np.float32)


asc_ratio  = compute_ratio(asc_vh_mean,  asc_vv_mean)
desc_ratio = compute_ratio(desc_vh_mean, desc_vv_mean)
print(f'VH/VV ratio — ASC mean: {np.nanmean(asc_ratio):.4f}  '
      f'DESC mean: {np.nanmean(desc_ratio):.4f}')


"""
Corrected GLCM block for the Sentinel-1 / XGBoost script
========================================================

Drop-in replacement for `quantise` and `compute_glcm_pytorch` (section 3.2).
Two fixes; everything else (angles, symmetrization, features, batching,
GPU path, prints) is identical to the original.

FIX 1 - tile seams (the real bug):
    The original pads EVERY tile with reflect padding on all four sides.
    Interior tiles must instead continue into the REAL neighbouring rows;
    reflecting the tile into itself corrupts pad = window//2 rows on each
    side of every internal tile boundary. Verified on synthetic data:
    original tiled vs untiled -> contrast errors up to ~16 in corrupted
    rows; halo-aware tiled vs untiled -> bitwise identical.
    The fix reads a halo of `pad` real rows around each tile and reflects
    only what falls outside the raster.

FIX 2 - quantisation percentiles:
    The original replaces NaN with 0.0 BEFORE computing the 2-98 percentiles,
    so a large NaN fraction (outside the AOI cutline) drags p2 to 0 and
    wastes grey levels. Percentiles are now computed from finite values only;
    NaN pixels are then mapped to the lowest level as before.

Note (unchanged behaviour, worth one sentence in the thesis): windows whose
centre lies within `pad` pixels of the AOI boundary still include NaN pixels
encoded as the lowest level, exactly as in the original.

Audit note: the GLCM mathematics itself was verified against a hand-computed
reference (symmetric normalized GLCM, offsets 0/45/90/135 deg, d=1) and is
correct to machine precision.
"""

_INT32_MAX = 2**31 - 1


def quantise(arr: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """FIX 2: percentiles from finite values only, then fill NaN."""
    finite = np.isfinite(arr)
    if finite.any():
        p2, p98 = np.percentile(arr[finite], [2, 98])
    else:
        p2, p98 = 0.0, 1.0
    arr = np.where(finite, arr, p2)
    arr = np.clip(arr, p2, p98)
    arr = (arr - p2) / (p98 - p2 + 1e-10) * (levels - 1)
    return arr.astype(np.uint8)


def _safe_tile_rows(ncols: int, window: int) -> int:
    elems_per_row = ncols * window * window
    return max(1, int((_INT32_MAX // elems_per_row) * 0.8))


def compute_glcm_pytorch(arr: np.ndarray, window: int = GLCM_WINDOW,
                         levels: int = GLCM_LEVELS, distance: int = GLCM_DIST,
                         batch_size: int = 4096, device: str = DEVICE) -> dict:
    """GLCM texture features via PyTorch - GPU accelerated, halo-aware tiling."""
    nrows, ncols = arr.shape
    pad = window // 2
    d   = distance
    q   = quantise(arr, levels)

    angle_pairs = [
        (slice(None),    slice(None),    slice(None, -d), slice(d, None)),
        (slice(d, None), slice(None,-d), slice(None, -d), slice(d, None)),
        (slice(None,-d), slice(d, None), slice(None),     slice(None)),
        (slice(None,-d), slice(d, None), slice(None, -d), slice(d, None)),
    ]

    i_idx = torch.arange(levels, dtype=torch.float32, device=device)
    j_idx = torch.arange(levels, dtype=torch.float32, device=device)
    I, J  = torch.meshgrid(i_idx, j_idx, indexing='ij')
    diff  = I - J
    contrast_w    = diff ** 2
    homogeneity_w = 1.0 / (1.0 + diff.abs())

    out_contrast    = np.zeros((nrows, ncols), dtype=np.float32)
    out_homogeneity = np.zeros((nrows, ncols), dtype=np.float32)
    out_energy      = np.zeros((nrows, ncols), dtype=np.float32)
    out_correlation = np.zeros((nrows, ncols), dtype=np.float32)

    max_tile_rows = _safe_tile_rows(ncols, window)
    n_tiles = (nrows + max_tile_rows - 1) // max_tile_rows
    if n_tiles > 1:
        print(f'  Tiling: {n_tiles} strips of ~{max_tile_rows} rows (halo-aware)')

    total_pixels = nrows * ncols
    processed    = 0

    for tile_idx in range(n_tiles):
        r0 = tile_idx * max_tile_rows
        r1 = min(r0 + max_tile_rows, nrows)
        tile_nrows = r1 - r0
        tile_N     = tile_nrows * ncols

        # --- FIX 1: read a halo of REAL rows around the tile; reflect only
        # what falls outside the raster. Output rows align exactly to r0..r1.
        h0 = max(0, r0 - pad)
        h1 = min(nrows, r1 + pad)
        top_missing = pad - (r0 - h0)
        bot_missing = pad - (h1 - r1)

        t = torch.from_numpy(q[h0:h1, :].astype(np.int64)).to(device)
        t_pad = F.pad(
            t.float().unsqueeze(0).unsqueeze(0),
            (pad, pad, top_missing, bot_missing), mode='reflect',
        ).squeeze().long()

        patches = t_pad.unfold(0, window, 1).unfold(1, window, 1)
        patches = patches.reshape(tile_N, window, window)

        n_batches = (tile_N + batch_size - 1) // batch_size
        tile_contrast    = torch.zeros(tile_N, dtype=torch.float32, device=device)
        tile_homogeneity = torch.zeros(tile_N, dtype=torch.float32, device=device)
        tile_energy      = torch.zeros(tile_N, dtype=torch.float32, device=device)
        tile_correlation = torch.zeros(tile_N, dtype=torch.float32, device=device)

        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, tile_N)
            B     = end - start
            pb    = patches[start:end]

            glcm = torch.zeros(B, levels, levels, dtype=torch.float32, device=device)
            gf   = glcm.view(B, -1)

            for dy_r, dy_n, dx_r, dx_n in angle_pairs:
                ref   = pb[:, dy_r, dx_r].reshape(B, -1)
                neigh = pb[:, dy_n, dx_n].reshape(B, -1)
                M     = ref.shape[1]
                ones_M = torch.ones(B, M, dtype=torch.float32, device=device)
                gf.scatter_add_(1, (ref   * levels + neigh).long(), ones_M)
                gf.scatter_add_(1, (neigh * levels + ref  ).long(), ones_M)

            g = glcm / glcm.sum(dim=(1, 2), keepdim=True).clamp(min=1e-10)
            tile_contrast[start:end]    = (g * contrast_w).sum(dim=(1, 2))
            tile_homogeneity[start:end] = (g * homogeneity_w).sum(dim=(1, 2))
            tile_energy[start:end]      = (g ** 2).sum(dim=(1, 2))
            mu_i  = (g * I).sum(dim=(1, 2))
            mu_j  = (g * J).sum(dim=(1, 2))
            var_i = (g * (I - mu_i.view(-1, 1, 1)) ** 2).sum(dim=(1, 2))
            var_j = (g * (J - mu_j.view(-1, 1, 1)) ** 2).sum(dim=(1, 2))
            std   = (var_i * var_j).sqrt().clamp(min=1e-10)
            num   = (g * (I - mu_i.view(-1, 1, 1))
                       * (J - mu_j.view(-1, 1, 1))).sum(dim=(1, 2))
            tile_correlation[start:end] = num / std

        out_contrast[r0:r1, :]    = tile_contrast.cpu().numpy().reshape(tile_nrows, ncols)
        out_homogeneity[r0:r1, :] = tile_homogeneity.cpu().numpy().reshape(tile_nrows, ncols)
        out_energy[r0:r1, :]      = tile_energy.cpu().numpy().reshape(tile_nrows, ncols)
        out_correlation[r0:r1, :] = tile_correlation.cpu().numpy().reshape(tile_nrows, ncols)

        del patches, t, t_pad, tile_contrast, tile_homogeneity, tile_energy, tile_correlation
        if device == 'cuda':
            torch.cuda.empty_cache()

        processed += tile_N
        print(f'\r  GLCM [{device}]: {processed:,}/{total_pixels:,} px '
              f'({100*processed/total_pixels:.0f}%)', end='', flush=True)

    print()
    return {
        'contrast':    out_contrast,
        'homogeneity': out_homogeneity,
        'energy':      out_energy,
        'correlation': out_correlation,
    }


# --- 3.2 Compute GLCM features ----------------------------------------------
_asc_glcm_cache  = CACHE_DIR / 'asc_glcm.npz'
_desc_glcm_cache = CACHE_DIR / 'desc_glcm.npz'

if _asc_glcm_cache.exists() and _desc_glcm_cache.exists():
    print('GLCM cache found — loading...')
    asc_glcm  = dict(np.load(_asc_glcm_cache))
    desc_glcm = dict(np.load(_desc_glcm_cache))
    print('  GLCM loaded from cache.')
else:
    print('\nComputing GLCM — ascending VV...')
    asc_glcm  = compute_glcm_pytorch(asc_vv_mean)
    np.savez_compressed(_asc_glcm_cache, **asc_glcm)
    print('Computing GLCM — descending VV...')
    desc_glcm = compute_glcm_pytorch(desc_vv_mean)
    np.savez_compressed(_desc_glcm_cache, **desc_glcm)
    print('  GLCM saved to cache.')

# --- 3.3 Feature stack assembly ----------------------------------------------

BAND_NAMES = [
    'asc_vv_mean', 'asc_vh_mean', 'asc_vv_std', 'asc_vh_std', 'asc_ratio',
    'asc_glcm_contrast', 'asc_glcm_homogeneity', 'asc_glcm_energy', 'asc_glcm_correlation',
    'desc_vv_mean', 'desc_vh_mean', 'desc_vv_std', 'desc_vh_std', 'desc_ratio',
    'desc_glcm_contrast', 'desc_glcm_homogeneity', 'desc_glcm_energy', 'desc_glcm_correlation',
]

FEATURE_ARRAYS = [
    asc_vv_mean, asc_vh_mean, asc_vv_std, asc_vh_std, asc_ratio,
    asc_glcm['contrast'], asc_glcm['homogeneity'], asc_glcm['energy'], asc_glcm['correlation'],
    desc_vv_mean, desc_vh_mean, desc_vv_std, desc_vh_std, desc_ratio,
    desc_glcm['contrast'], desc_glcm['homogeneity'], desc_glcm['energy'], desc_glcm['correlation'],
]

assert len(BAND_NAMES) == len(FEATURE_ARRAYS) == 18
assert all(a.shape == FEATURE_ARRAYS[0].shape for a in FEATURE_ARRAYS), \
    f'Feature array shape mismatch: {set(a.shape for a in FEATURE_ARRAYS)}'

FEATURE_STACK_PATH = OUT_DIR / 'feature_stack.tif'
if not FEATURE_STACK_PATH.exists():
    write_multiband_tiff(
        path=FEATURE_STACK_PATH,
        arrays=FEATURE_ARRAYS,
        band_names=BAND_NAMES,
        geo_info=geo_ref,
    )
else:
    print(f'  Feature stack already exists — skipping write.')
print(f'Feature stack: {FEATURE_STACK_PATH}')


# ============================================================
# SECTION 4 — GROUND TRUTH FROM BDOT10K
# ============================================================

def load_ground_truth(bdot10k_path: Path, target_epsg: int = EPSG) -> gpd.GeoDataFrame:
    print(f'Loading BDOT10k: {bdot10k_path}')
    gdf = gpd.read_file(bdot10k_path, engine='pyogrio')
    
    if gdf.crs and gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)
    print(f'  Combined: {len(gdf)} features')
    return gdf


ground_truth = load_ground_truth(BDOT10K_PATH)


# --- 4.1 Rasterize ground truth ----------------------------------------------

def rasterize_ground_truth(gdf: gpd.GeoDataFrame, geo_info: dict) -> np.ndarray:
    gt    = geo_info['geotransform']
    proj  = geo_info['projection']
    nrows = geo_info['nrows']
    ncols = geo_info['ncols']

    gdf_buf = gdf.copy()
    gdf_buf['geometry'] = gdf_buf.geometry.buffer(LEVEE_BUFFER)

    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    gdf_buf.to_file(tmp_path, driver='GPKG', engine='pyogrio')

    vec_ds  = ogr.Open(str(tmp_path))
    vec_lyr = vec_ds.GetLayer(0)

    out_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.Fill(0)
    out_band.SetNoDataValue(255)
    gdal.RasterizeLayer(out_ds, [1], vec_lyr, burn_values=[1])
    out_ds.FlushCache()
    mask = out_band.ReadAsArray().astype(np.uint8)
    out_ds = None
    vec_ds = None
    tmp_path.unlink(missing_ok=True)

    print(f'  Positive pixels (levee): {int(mask.sum()):,}')
    print(f'  Negative pixels (other): {int((mask == 0).sum()):,}')
    return mask


print('Rasterizing ground truth...')
label_mask = rasterize_ground_truth(ground_truth, geo_ref)

label_path = OUT_DIR / 'label_mask.tif'
write_multiband_tiff(
    path=label_path,
    arrays=[label_mask.astype(np.float32)],
    band_names=['label'],
    geo_info=geo_ref,
    nodata=255,
)
print(f'Label mask saved -> {label_path}')


# ============================================================
# SECTION 5 — SPATIALLY BLOCKED PIXEL SAMPLING
# ============================================================

def assign_spatial_blocks(nrows: int, ncols: int, block_size: int,
                          test_fraction: float = 0.3,
                          random_state: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_br = int(np.ceil(nrows / block_size))
    n_bc = int(np.ceil(ncols / block_size))
    n_blocks = n_br * n_bc
    n_test   = max(1, int(round(n_blocks * test_fraction)))
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)
    test_set = set(block_ids[:n_test].tolist())

    split_mask = np.zeros((nrows, ncols), dtype=np.uint8)
    for bi in range(n_br):
        for bj in range(n_bc):
            if bi * n_bc + bj in test_set:
                r0, r1 = bi * block_size, min((bi + 1) * block_size, nrows)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, ncols)
                split_mask[r0:r1, c0:c1] = 1

    print(f'  Spatial blocks: {n_br} x {n_bc} = {n_blocks}  '
          f'(train: {n_blocks - n_test}  test: {n_test})')
    return split_mask


def build_feature_matrix(feature_arrays: list, label_mask: np.ndarray,
                         split_mask: np.ndarray) -> tuple:
    rng = np.random.default_rng(RANDOM_STATE)
    X_flat = np.stack(feature_arrays, axis=-1).reshape(-1, len(feature_arrays))
    y_flat = label_mask.ravel()
    s_flat = split_mask.ravel()
    valid  = ~np.any(np.isnan(X_flat), axis=1)
    print(f'  Valid pixels: {valid.sum():,} / {len(valid):,}')

    def sample_set(set_id: int, label: str, frac: float) -> tuple:
        mask    = (s_flat == set_id) & valid
        pos_idx = np.where(mask & (y_flat == 1))[0]
        neg_idx = np.where(mask & (y_flat == 0))[0]
        print(f'  {label}: {len(pos_idx):,} pos / {len(neg_idx):,} neg available')
        n_pos = min(int(N_POSITIVE * frac), len(pos_idx))
        n_neg = min(int(N_NEGATIVE * frac), len(neg_idx))
        idx   = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])
        rng.shuffle(idx)
        return X_flat[idx].astype(np.float32), y_flat[idx].astype(np.int8)

    X_train, y_train = sample_set(0, 'Train blocks', 0.7)
    X_test,  y_test  = sample_set(1, 'Test blocks',  0.3)
    print(f'  Sampled — train: {len(X_train):,}  test: {len(X_test):,}')
    return X_train, X_test, y_train, y_test


print('Assigning spatial blocks...')
nrows, ncols = FEATURE_ARRAYS[0].shape
split_mask   = assign_spatial_blocks(nrows, ncols, BLOCK_SIZE)

print('\nBuilding feature matrix (spatially blocked)...')
X_train, X_test, y_train, y_test = build_feature_matrix(
    FEATURE_ARRAYS, label_mask, split_mask
)
print(f'\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}')
print(f'Positive class ratio — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}')


# ============================================================
# SECTION 6 — XGBOOST TRAINING
# ============================================================

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight: {scale_pos_weight:.2f}')

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
)

print('Training XGBoost...')
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
print(f'Best iteration: {model.best_iteration}')


# --- 6.1 Threshold optimisation via PR curve ---------------------------------

y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_idx  = np.argmax(f1_scores[:-1])
best_thr  = float(thresholds[best_idx])
best_f1   = float(f1_scores[best_idx])

print(f'Optimal threshold : {best_thr:.3f}')
print(f'Best F1           : {best_f1:.3f}')
print(f'AP score          : {average_precision_score(y_test, y_proba):.3f}')
print(f'ROC-AUC           : {roc_auc_score(y_test, y_proba):.3f}')

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, lw=2, label='PR curve')
ax.axvline(recall[best_idx], color='red', linestyle='--', alpha=0.7,
           label=f'Optimal thr={best_thr:.2f} (F1={best_f1:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — Levee Detection')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'pr_curve.png', dpi=150)
plt.close(fig)


# --- 6.2 Classification report -----------------------------------------------

y_pred = (y_proba >= best_thr).astype(int)
print(classification_report(y_test, y_pred, target_names=['non-levee', 'levee'], digits=3))


# --- 6.3 Feature importance --------------------------------------------------

importance  = model.feature_importances_
sorted_idx  = np.argsort(importance)[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([BAND_NAMES[i] for i in sorted_idx[::-1]], importance[sorted_idx[::-1]])
ax.set_xlabel('Feature importance (gain)')
ax.set_title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig(OUT_DIR / 'feature_importance.png', dpi=150)
plt.close(fig)

print('Feature importance ranking:')
for rank, i in enumerate(sorted_idx, 1):
    print(f'  {rank:2d}. {BAND_NAMES[i]:<35s} {importance[i]:.4f}')


# ============================================================
# SECTION 7 — SAVE MODEL AND METADATA
# ============================================================

model_path = OUT_DIR / 'xgb_levee_model.json'
model.save_model(str(model_path))
print(f'Model saved -> {model_path}')

metadata = {
    'created':            datetime.now().isoformat(),
    'model':              'XGBoostClassifier',
    'best_iteration':     int(model.best_iteration),
    'optimal_threshold':  best_thr,
    'best_f1':            best_f1,
    'ap_score':           float(average_precision_score(y_test, y_proba)),
    'roc_auc':            float(roc_auc_score(y_test, y_proba)),
    'band_names':         BAND_NAMES,
    'n_train':            int(len(X_train)),
    'n_test':             int(len(X_test)),
    'epsg':               EPSG,
    'pixel_size_m':       PIXEL_SIZE,
    'glcm_window':        GLCM_WINDOW,
    'glcm_levels':        GLCM_LEVELS,
    'levee_buffer_m':     LEVEE_BUFFER,
    'nasyp_min_length_m': NASYP_MIN_LENGTH,
    'nasyp_min_height_m': NASYP_MIN_HEIGHT,
    'nasyp_min_width_m':  NASYP_MIN_WIDTH,
}

meta_path = OUT_DIR / 'model_metadata.json'
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Metadata saved -> {meta_path}')

print('\n=== Training complete ===')
print(f"  Threshold : {best_thr:.3f}")
print(f"  AP score  : {metadata['ap_score']:.3f}")
print(f"  ROC-AUC   : {metadata['roc_auc']:.3f}")
