"""
End-to-end inference pipeline for levee detection over a user-defined Area of Interest (AOI).

This script applies a trained SegFormer deep learning model to detect levee centerlines
in a new geographic region. The workflow:

1. **Load data**: Read AOI polygon (GeoPackage), load raster datasets (DSM, canopy)
2. **Define corridor**: Restrict processing to river corridors from MERIT Hydro reaches
3. **Generate patches**: Create a sliding-window grid of 2560m × 2560m patches (256×256 px @ 10m res)
   with 50% overlap to cover the entire corridor
4. **Extract & prepare**: For each patch, extract 6-channel input (DSM, 3×TPI at different scales,
   canopy height, canopy height SD) and normalize using training statistics
5. **Inference**: Run batched forward passes through the model to produce probability maps [0,1]
   indicating levee likelihood at each pixel
6. **Stitch**: Blend overlapping patches using weighted averaging to create a seamless probability raster
7. **Post-process**: Threshold predictions, apply morphological cleanup (closing, small object removal),
   skeletonize to extract centerlines, and convert to vector geometries
8. **Export**: Save detected levee centerlines as GeoPackage (linestrings) with length attributes,
   and probability raster as GeoTIFF (32-bit float)

Target CRS: EPSG:2180 (PL-1992 Polish coordinate system)
Model: SegFormer with mit_b2 backbone, 6-channel input, trained for binary levee segmentation

Author:   Jakub Zapletal
Date:     2026-05-08
Version:  0.2
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from osgeo import gdal, gdalconst
from scipy.ndimage import uniform_filter
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import linemerge, unary_union
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize
from tqdm import tqdm

warnings.filterwarnings("ignore")
gdal.UseExceptions()

# ============================================================
# CONFIG
# ============================================================

# User-provided paths
AOI_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\AOI_Poland.gpkg"
)
DSM_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif"
)
CANOPY_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_2180.tif"
)
CANOPY_SD_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_SD_2180.tif"
)
MERIT_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\riv_pfaf_2x_MERIT_Hydro_v07_Basin_flip_2180.gpkg"
)
MERIT_LAYER = "reaches"
MERIT_UPAREA_COL = "uparea"

CHECKPOINT_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v02_segformer\best_model.pt"
)
NORM_STATS_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v02_segformer\norm_stats.json"
)

OUTPUT_GPKG = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\interference_output_smaller_buffered\detected_levees_AOI_PL_smaller.gpkg"
)
OUTPUT_PROB_TIF = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_prob_AOI_PL_small.tif")

# Geographic & raster constants (must match training)
CRS_TARGET = 2180  # EPSG:2180 (PL-1992)
PATCH_SIZE_PX = 256
PATCH_RES_M = 10
PATCH_EXTENT_M = PATCH_SIZE_PX * PATCH_RES_M  # 2560 m
STRIDE_PX = 128
STRIDE_M = STRIDE_PX * PATCH_RES_M

TPI_RADII_PX = [5, 10, 15]
RIVER_BUFFER_M = 1000
MIN_UPAREA_KM2 = 10

# Model / inference constants
SEGFORMER_BACKBONE = "mit_b2"
N_INPUT_CHANNELS = 6
INFERENCE_BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Postprocessing constants
PROB_THRESHOLD = 0.5
MIN_COMPONENT_PX = 50
CLOSING_RADIUS_PX = 1
MIN_LINE_LENGTH_M = 50


# ============================================================
# MODEL LOADING
# ============================================================

def adapt_first_conv_segformer(model, n_input_channels):
    """
    Adapt SegFormer model's first convolution layer to accept N input channels.
    
    The SegFormer encoder is pretrained on 3-channel RGB images. This function
    expands the first convolutional layer to accept 6 channels (DSM + 3×TPI + canopy + canopy_SD)
    by replicating and normalizing pretrained weights, enabling transfer learning with custom inputs.
    
    Args:
        model: SegFormer model instance
        n_input_channels: Number of input channels (e.g., 6 for this inference task)
    
    Returns:
        Modified model with adapted first convolution layer
    """
    encoder = model.encoder
    first_conv = encoder.patch_embed1.proj
    old_weight = first_conv.weight.data
    out_ch, _, kh, kw = old_weight.shape

    new_weight = old_weight.repeat(1, (n_input_channels // 3) + 1, 1, 1)
    new_weight = new_weight[:, :n_input_channels, :, :]
    new_weight = new_weight / (n_input_channels / 3)

    new_conv = nn.Conv2d(
        n_input_channels,
        out_ch,
        kernel_size=(kh, kw),
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    encoder.patch_embed1.proj = new_conv
    return model


def build_and_load_model():
    """
    Initialize SegFormer model and load trained weights from checkpoint.
    
    Creates a SegFormer encoder-decoder with mit_b2 backbone, adapts first layer
    for 6-channel input, and loads the best model checkpoint. Model is set to
    evaluation mode and moved to the appropriate device (CUDA if available, else CPU).
    
    Returns:
        tuple: (model, checkpoint_dict)
            - model: Loaded model ready for inference
            - checkpoint_dict: Dictionary with 'model_state', 'epoch', 'val_score', etc.
    """
    model = smp.Segformer(
        encoder_name=SEGFORMER_BACKBONE,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = adapt_first_conv_segformer(model, N_INPUT_CHANNELS)
    model = model.to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


# ============================================================
# AOI + RIVER CORRIDOR
# ============================================================

def load_aoi_polygon():
    """
    Load Area of Interest (AOI) polygon from GeoPackage file.
    
    Reads the GeoPackage specified by AOI_PATH, reprojects to target CRS (EPSG:2180),
    and merges all features into a single unified geometry to use as the processing boundary.
    
    Returns:
        shapely.geometry.Polygon: Merged AOI geometry in EPSG:2180
    """
    aoi = gpd.read_file(AOI_PATH)
    if aoi.crs.to_epsg() != CRS_TARGET:
        aoi = aoi.to_crs(epsg=CRS_TARGET)
    return unary_union(aoi.geometry.tolist())


def build_river_corridor(aoi_geom):
    """
    Define the river corridor as buffered MERIT Hydro reaches within the AOI.
    
    Loads MERIT Hydro stream reaches, filters to those with upstream area ≥ MIN_UPAREA_KM2
    (ensures major rivers/streams, excludes tiny tributary sources), intersects with AOI,
    buffers by RIVER_BUFFER_M to create a processing corridor, and clips to AOI boundary.
    Levee detection is restricted to this corridor to avoid false detections in non-fluvial areas.
    
    Args:
        aoi_geom: Shapely geometry of the AOI boundary
    
    Returns:
        tuple: (corridor_geom, merit_gdf)
            - corridor_geom: Buffered river corridor geometry in EPSG:2180
            - merit_gdf: GeoDataFrame of filtered MERIT reaches used to define corridor
    
    Raises:
        RuntimeError: If no MERIT reaches pass the uparea filter in the AOI
    """
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs=f"EPSG:{CRS_TARGET}")
    aoi_bbox = aoi_geom.bounds

    merit = gpd.read_file(MERIT_PATH, bbox=aoi_bbox)
    if merit.crs.to_epsg() != CRS_TARGET:
        merit = merit.to_crs(epsg=CRS_TARGET)

    merit = merit[merit[MERIT_UPAREA_COL] >= MIN_UPAREA_KM2].copy()
    merit = gpd.overlay(merit, aoi_gdf, how="intersection")

    if len(merit) == 0:
        raise RuntimeError("No MERIT reaches in AOI passed uparea filter")

    corridor = unary_union(merit.geometry.buffer(RIVER_BUFFER_M).tolist())
    corridor = corridor.intersection(aoi_geom)
    return corridor, merit


# ============================================================
# PATCH GRID GENERATION
# ============================================================

def generate_patch_centers(corridor_geom):
    """
    Generate sliding-window patch center coordinates covering the river corridor.
    
    Creates a regular grid of patch centers spaced by STRIDE_M (1280m = 50% overlap on 2560m patches).
    Only includes patches whose 2560m × 2560m extent intersects the corridor geometry,
    ensuring efficient coverage of the processing area.
    
    Args:
        corridor_geom: Shapely geometry of the river corridor
    
    Returns:
        list: [(center_x, center_y), ...] tuples in EPSG:2180 coordinates
    """
    minx, miny, maxx, maxy = corridor_geom.bounds

    x_start = (minx // STRIDE_M) * STRIDE_M + STRIDE_M / 2
    y_start = (miny // STRIDE_M) * STRIDE_M + STRIDE_M / 2

    centers = []
    y = y_start
    while y <= maxy + STRIDE_M:
        x = x_start
        while x <= maxx + STRIDE_M:
            patch_box = box(
                x - PATCH_EXTENT_M / 2,
                y - PATCH_EXTENT_M / 2,
                x + PATCH_EXTENT_M / 2,
                y + PATCH_EXTENT_M / 2,
            )
            if patch_box.intersects(corridor_geom):
                centers.append((x, y))
            x += STRIDE_M
        y += STRIDE_M

    return centers


# ============================================================
# RASTER WINDOW READING
# ============================================================

def read_window(ds, bbox, target_pixels, resample=gdalconst.GRA_Bilinear):
    """
    Read and resample a rectangular window from a GDAL raster dataset.
    
    Handles edge cases: reads data from any part of the raster, zero-pads areas outside
    the raster extent, and resamples to exact output dimensions using specified algorithm.
    
    Args:
        ds: Open GDAL dataset handle
        bbox: (xmin, ymin, xmax, ymax) bounding box in dataset CRS
        target_pixels: Output dimensions (target_pixels × target_pixels)
        resample: GDAL resampling algorithm (default: bilinear)
    
    Returns:
        numpy.ndarray: Float32 array of shape (target_pixels, target_pixels)
                       Zero-padded if bbox extends beyond raster bounds
    """
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    px_ulx, px_uly = gdal.ApplyGeoTransform(inv_gt, bbox[0], bbox[3])
    px_lrx, px_lry = gdal.ApplyGeoTransform(inv_gt, bbox[2], bbox[1])

    col_off = int(np.floor(px_ulx))
    row_off = int(np.floor(px_uly))
    col_size = max(1, int(np.ceil(px_lrx - px_ulx)))
    row_size = max(1, int(np.ceil(px_lry - px_uly)))

    raster_xsize = ds.RasterXSize
    raster_ysize = ds.RasterYSize

    if (
        col_off >= raster_xsize
        or row_off >= raster_ysize
        or col_off + col_size <= 0
        or row_off + row_size <= 0
    ):
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    read_col = max(0, col_off)
    read_row = max(0, row_off)
    read_col_size = min(col_size - (read_col - col_off), raster_xsize - read_col)
    read_row_size = min(row_size - (read_row - row_off), raster_ysize - read_row)

    if read_col_size <= 0 or read_row_size <= 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    if (
        read_col == col_off
        and read_row == row_off
        and read_col_size == col_size
        and read_row_size == row_size
    ):
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray(
            read_col,
            read_row,
            read_col_size,
            read_row_size,
            buf_xsize=target_pixels,
            buf_ysize=target_pixels,
            resample_alg=resample,
        ).astype(np.float32)
        return arr

    sub_target_w = int(round(target_pixels * read_col_size / col_size))
    sub_target_h = int(round(target_pixels * read_row_size / row_size))
    if sub_target_w == 0 or sub_target_h == 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    band = ds.GetRasterBand(1)
    sub = band.ReadAsArray(
        read_col,
        read_row,
        read_col_size,
        read_row_size,
        buf_xsize=sub_target_w,
        buf_ysize=sub_target_h,
        resample_alg=resample,
    ).astype(np.float32)

    out = np.zeros((target_pixels, target_pixels), dtype=np.float32)
    out_col_off = int(round(target_pixels * (read_col - col_off) / col_size))
    out_row_off = int(round(target_pixels * (read_row - row_off) / row_size))
    out[
        out_row_off : out_row_off + sub_target_h,
        out_col_off : out_col_off + sub_target_w,
    ] = sub
    return out


# ============================================================
# PATCH EXTRACTION + NORMALIZATION
# ============================================================

def compute_tpi(z, radius_px):
    """
    Compute Topographic Position Index (TPI) at specified scale.
    
    TPI quantifies local landform: positive values indicate ridges/peaks, negative values
    indicate valleys. Computed as elevation minus the mean of a circular neighborhood.
    This matches the training pipeline preprocessing.
    
    Args:
        z: 2D elevation array (e.g., DSM)
        radius_px: Radius in pixels for the neighborhood window
    
    Returns:
        numpy.ndarray: TPI array of same shape as z
    """
    size = 2 * radius_px + 1
    return z - uniform_filter(z, size=size, mode="nearest")


def extract_patch(center_x, center_y, dsm_ds, canopy_ds, canopy_sd_ds):
    """
    Extract a 6-channel input patch for the neural network at given location.
    
    Extracts 2560m × 2560m patches at 10m resolution (256×256 pixels) from:
    1. Digital Surface Model (DSM) - terrain elevation
    2-4. Topographic Position Index at 3 scales (50m, 100m, 150m radii)
    5. Canopy height (ETH Global Canopy Height Model)
    6. Canopy height standard deviation (uncertainty estimate)
    
    Args:
        center_x, center_y: Patch center coordinates in EPSG:2180
        dsm_ds, canopy_ds, canopy_sd_ds: Open GDAL dataset handles for each input layer
    
    Returns:
        numpy.ndarray: Shape (6, 256, 256) with channels as above, dtype float32
    """
    bbox = (
        center_x - PATCH_EXTENT_M / 2,
        center_y - PATCH_EXTENT_M / 2,
        center_x + PATCH_EXTENT_M / 2,
        center_y + PATCH_EXTENT_M / 2,
    )

    dsm = read_window(dsm_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    canopy = read_window(canopy_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    canopy_sd = read_window(canopy_sd_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)

    tpi_channels = [compute_tpi(dsm, r) for r in TPI_RADII_PX]
    patch = np.stack([dsm, *tpi_channels, canopy, canopy_sd], axis=0)
    return patch


def normalize_patch(patch, norm_stats):
    """
    Apply normalization to a 6-channel patch using training statistics.
    
    Normalization strategy:
    - Channel 0 (DSM): Per-patch median subtraction → centers elevation relative to local terrain
      (improves generalization across regions with different absolute elevations)
    - Channels 1-5 (TPI scales, canopy, canopy_sd): Per-channel z-score using training mean/std
      (matches training preprocessing for stable model predictions)
    
    Args:
        patch: numpy.ndarray of shape (6, 256, 256)
        norm_stats: Dictionary with keys ['tpi_r5', 'tpi_r10', 'tpi_r15', 'canopy_height', 'canopy_height_sd']
                    Each containing {'mean': float, 'std': float}
    
    Returns:
        numpy.ndarray: Normalized patch, same shape as input
    """
    out = patch.copy()

    out[0] = out[0] - np.median(out[0])

    channel_names = ["tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd"]
    for i, name in enumerate(channel_names, start=1):
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        out[i] = (out[i] - mean) / (std + 1e-8)

    return out


# ============================================================
# BATCHED INFERENCE
# ============================================================

def run_inference(model, centers, norm_stats):
    """
    Execute batched inference over all patch centers using the trained model.
    
    Opens raster sources, extracts patches in batches (default 16), applies normalization,
    runs forward pass with torch.no_grad(), and applies sigmoid activation to convert
    logits to probability predictions [0, 1]. Progress displayed via tqdm.
    
    Args:
        model: Trained SegFormer model in eval mode
        centers: List of (center_x, center_y) tuples for all patches
        norm_stats: Channel normalization statistics from training
    
    Returns:
        list: [(center_x, center_y, prob_map), ...] where prob_map is 256×256 probability array
    """
    dsm_ds = gdal.Open(str(DSM_PATH))
    canopy_ds = gdal.Open(str(CANOPY_PATH))
    canopy_sd_ds = gdal.Open(str(CANOPY_SD_PATH))

    if dsm_ds is None or canopy_ds is None or canopy_sd_ds is None:
        raise RuntimeError("Failed to open one or more raster sources")

    predictions = []
    n_centers = len(centers)

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, n_centers, INFERENCE_BATCH),
            desc="Inference",
            total=(n_centers + INFERENCE_BATCH - 1) // INFERENCE_BATCH,
        ):
            batch_centers = centers[batch_start : batch_start + INFERENCE_BATCH]

            patches = []
            for cx, cy in batch_centers:
                p = extract_patch(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds)
                p = normalize_patch(p, norm_stats)
                patches.append(p)

            batch = torch.from_numpy(np.stack(patches, axis=0)).float().to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for (cx, cy), prob in zip(batch_centers, probs):
                predictions.append((cx, cy, prob))

    dsm_ds = None
    canopy_ds = None
    canopy_sd_ds = None

    return predictions


# ============================================================
# STITCHING
# ============================================================

def stitch_predictions(predictions, corridor_geom):
    """
    Merge overlapping patch predictions using maximum operator.
    
    Creates a raster covering the corridor bounding box and combines predictions
    using maximum: where patches overlap, the highest probability value is retained.
    This preserves strong detections without smoothing/averaging.
    Non-overlapped areas remain zero.
    
    Args:
        predictions: List of (center_x, center_y, prob_map) tuples from run_inference
        corridor_geom: Corridor boundary (used to determine raster extent)
    
    Returns:
        tuple: (prob_raster, geotransform)
            - prob_raster: 2D float32 array of stitched probabilities [0, 1] (maximum merged)
            - geotransform: GDAL geotransform tuple for georeferencing
    """
    minx, miny, maxx, maxy = corridor_geom.bounds

    origin_x = (minx // STRIDE_M) * STRIDE_M - PATCH_EXTENT_M / 2
    origin_y = (maxy // STRIDE_M + 1) * STRIDE_M + PATCH_EXTENT_M / 2

    width_m = (maxx - origin_x) + PATCH_EXTENT_M
    height_m = (origin_y - miny) + PATCH_EXTENT_M

    width_px = int(np.ceil(width_m / PATCH_RES_M))
    height_px = int(np.ceil(height_m / PATCH_RES_M))

    prob = np.zeros((height_px, width_px), dtype=np.float32)

    for cx, cy, pred in predictions:
        col_off = int(round((cx - PATCH_EXTENT_M / 2 - origin_x) / PATCH_RES_M))
        row_off = int(round((origin_y - cy - PATCH_EXTENT_M / 2) / PATCH_RES_M))

        r0, r1 = max(0, row_off), min(height_px, row_off + PATCH_SIZE_PX)
        c0, c1 = max(0, col_off), min(width_px, col_off + PATCH_SIZE_PX)
        pr0 = r0 - row_off
        pr1 = pr0 + (r1 - r0)
        pc0 = c0 - col_off
        pc1 = pc0 + (c1 - c0)

        prob[r0:r1, c0:c1] = np.maximum(prob[r0:r1, c0:c1], pred[pr0:pr1, pc0:pc1])

    geotransform = (origin_x, PATCH_RES_M, 0.0, origin_y, 0.0, -PATCH_RES_M)
    return prob, geotransform


def export_patches_as_geodataframe(centers):
    """
    Export patch grid geometries as a GeoDataFrame for visualization/inspection.
    
    Creates polygon geometries for each patch, with extent PATCH_EXTENT_M x PATCH_EXTENT_M
    centered at each provided center coordinate.
    
    Args:
        centers: List of (center_x, center_y) patch center tuples
    
    Returns:
        geopandas.GeoDataFrame: Columns = ['patch_id', 'geometry'], CRS = EPSG:2180
    """
    geometries = []
    for i, (cx, cy) in enumerate(centers):
        half_extent = PATCH_EXTENT_M / 2
        patch_geom = box(cx - half_extent, cy - half_extent, cx + half_extent, cy + half_extent)
        geometries.append(patch_geom)

    gdf = gpd.GeoDataFrame(
        {"patch_id": list(range(len(centers)))},
        geometry=geometries,
        crs=f"EPSG:{CRS_TARGET}",
    )
    return gdf


def rasterize_corridor_mask(corridor_geom, geotransform, shape):
    """
    Convert corridor polygon to a binary raster mask aligned with probability raster.
    
    Uses GDAL rasterization to burn corridor geometry into a raster grid matching
    the stitched probability raster's extent and resolution. Used in postprocessing
    to mask out-of-corridor detections.
    
    Args:
        corridor_geom: Shapely polygon of river corridor
        geotransform: GDAL geotransform tuple defining raster georeferencing
        shape: (height_px, width_px) output dimensions
    
    Returns:
        numpy.ndarray: Boolean array (True = inside corridor) matching shape
    """
    height_px, width_px = shape
    mem_drv = gdal.GetDriverByName("MEM")
    target_ds = mem_drv.Create("", width_px, height_px, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)

    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(CRS_TARGET)
    target_ds.SetProjection(srs.ExportToWkt())

    corridor_gdf = gpd.GeoDataFrame(geometry=[corridor_geom], crs=f"EPSG:{CRS_TARGET}")
    tmp_path = f"/vsimem/corridor_{id(corridor_geom)}.gpkg"
    corridor_gdf.to_file(tmp_path, driver="GPKG")

    src_ds = gdal.OpenEx(tmp_path, gdal.OF_VECTOR)
    gdal.RasterizeLayer(target_ds, [1], src_ds.GetLayer(0), burn_values=[1])

    mask = target_ds.GetRasterBand(1).ReadAsArray().astype(bool)
    target_ds = None
    src_ds = None
    gdal.Unlink(tmp_path)

    return mask


def save_prob_raster_geotiff(prob, geotransform, output_path):
    """
    Write probability raster to GeoTIFF file with georeferencing and compression.
    
    Creates a 32-bit float GeoTIFF with LZW compression, tiled structure, and BigTIFF
    format (supports rasters >4GB). Sets NoData value to -1. Output is in EPSG:2180.
    
    Args:
        prob: 2D float32 probability array
        geotransform: GDAL geotransform tuple
        output_path: Output file path (Path or str)
    
    Returns:
        None (writes to disk)
    """
    height_px, width_px = prob.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(output_path),
        width_px,
        height_px,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"],
    )
    ds.SetGeoTransform(geotransform)
    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(CRS_TARGET)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(prob)
    ds.GetRasterBand(1).SetNoDataValue(-1.0)
    ds.FlushCache()
    ds = None


# ============================================================
# POSTPROCESSING: skeleton -> vector
# ============================================================

def skeleton_to_linestrings(skeleton_mask, geotransform):
    """
    Convert binary skeleton raster to vector LineStrings in world coordinates.
    
    Identifies connected skeleton pixels, traces pixel-to-pixel connections as line segments,
    converts pixel coordinates to world coordinates using geotransform, and merges
    connected segments into continuous polylines.
    
    Args:
        skeleton_mask: 2D boolean array where True = skeleton pixel
        geotransform: GDAL geotransform tuple for pixel-to-world conversion
    
    Returns:
        list: [LineString, ...] geometries in world coordinates
    """
    rows, cols = np.where(skeleton_mask)
    if len(rows) == 0:
        return []

    coord_set = set(zip(rows.tolist(), cols.tolist()))

    origin_x, pixel_w, _, origin_y, _, pixel_h_neg = geotransform
    pixel_h = -pixel_h_neg

    def pixel_to_world(r, c):
        x = origin_x + (c + 0.5) * pixel_w
        y = origin_y - (r + 0.5) * pixel_h
        return (x, y)

    segments = []
    for r, c in coord_set:
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coord_set:
                segments.append(LineString([pixel_to_world(r, c), pixel_to_world(nr, nc)]))

    if not segments:
        return []

    merged = linemerge(MultiLineString(segments))
    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return list(merged.geoms)
    return []


def postprocess_to_vector(prob_raster, corridor_mask, geotransform):
    """
    Convert probability raster predictions into clean vector centerlines.
    
    Processing pipeline:
    1. Threshold: probability > 0.5 → binary map
    2. Mask: apply corridor mask to exclude out-of-area pixels
    3. Cleanup: morphological closing (size 1px) + remove small objects (min 50px)
    4. Skeletonize: reduce levee bodies to 1-pixel centerlines
    5. Vectorize: convert skeleton to LineString geometries with world coordinates
    6. Filter: keep only lines ≥ MIN_LINE_LENGTH_M (50m, removes noise)
    
    Args:
        prob_raster: 2D float32 array of probabilities [0, 1]
        corridor_mask: 2D boolean array (True = inside corridor)
        geotransform: GDAL geotransform for raster georeferencing
    
    Returns:
        geopandas.GeoDataFrame: Columns = ['length_m', 'geometry'], CRS = EPSG:2180
                                (empty DataFrame if no detections pass filters)
    """
    binary = (prob_raster > PROB_THRESHOLD) & corridor_mask

    if CLOSING_RADIUS_PX > 0:
        binary = binary_closing(binary, disk(CLOSING_RADIUS_PX))

    binary = remove_small_objects(binary, min_size=MIN_COMPONENT_PX, connectivity=2)

    if binary.sum() == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    skel = skeletonize(binary)

    lines = skeleton_to_linestrings(skel, geotransform)
    if not lines:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    lines = [ln for ln in lines if ln.length >= MIN_LINE_LENGTH_M]

    gdf = gpd.GeoDataFrame(
        {"length_m": [ln.length for ln in lines]},
        geometry=lines,
        crs=f"EPSG:{CRS_TARGET}",
    )
    return gdf


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Execute the complete end-to-end levee detection inference pipeline.
    
    Orchestrates all steps: loads model and data, generates patches, runs inference,
    stitches predictions, post-processes to vector, and saves outputs (GPKG + probability GeoTIFF).
    Prints progress and summary statistics to console.
    """
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"AOI: {AOI_PATH}")
    print(f"Output: {OUTPUT_GPKG}")
    print()

    print("Loading model...")
    model, ckpt = build_and_load_model()
    print(f"  Best epoch from checkpoint: {ckpt.get('epoch', '?')}")
    print(f"  val_score: {ckpt.get('val_score', float('nan')):.4f}")

    print("Loading norm stats...")
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    print("Loading AOI + MERIT corridor...")
    aoi_geom = load_aoi_polygon()
    corridor_geom, merit_in_aoi = build_river_corridor(aoi_geom)
    print(f"  AOI area:       {aoi_geom.area / 1e6:.1f} km2")
    print(f"  Corridor area:  {corridor_geom.area / 1e6:.1f} km2 ({corridor_geom.area / aoi_geom.area * 100:.1f}%)")
    print(f"  MERIT reaches:  {len(merit_in_aoi)}")

    print("Generating patch centers...")
    centers = generate_patch_centers(corridor_geom)
    print(f"  Total patches: {len(centers)}")
    if len(centers) == 0:
        raise RuntimeError("No patches generated - AOI / corridor empty?")

    print("Running inference...")
    predictions = run_inference(model, centers, norm_stats)

    print("Stitching predictions...")
    prob_raster, geotransform = stitch_predictions(predictions, corridor_geom)
    print(f"  Probability raster: {prob_raster.shape} ({prob_raster.nbytes / 1e6:.1f} MB)")

    print(f"Saving probability raster to {OUTPUT_PROB_TIF}...")
    OUTPUT_PROB_TIF.parent.mkdir(parents=True, exist_ok=True)
    save_prob_raster_geotiff(prob_raster, geotransform, OUTPUT_PROB_TIF)

    print("Rasterizing corridor mask...")
    corridor_mask = rasterize_corridor_mask(corridor_geom, geotransform, prob_raster.shape)

    print("Postprocessing (threshold -> skeleton -> vectorize)...")
    detected = postprocess_to_vector(prob_raster, corridor_mask, geotransform)
    print(f"  Detected lines: {len(detected)}")
    if len(detected) > 0:
        print(f"  Total length:   {detected['length_m'].sum() / 1000:.1f} km")
        print(f"  Mean length:    {detected['length_m'].mean():.0f} m")
        print(f"  Max length:     {detected['length_m'].max():.0f} m")

    print(f"Saving GPKG to {OUTPUT_GPKG}...")
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    detected.to_file(OUTPUT_GPKG, driver="GPKG")

    print(f"Saving patches grid to {OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + '_patches.gpkg')}...")
    patches_gdf = export_patches_as_geodataframe(centers)
    patches_output = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_patches.gpkg")
    patches_gdf.to_file(patches_output, driver="GPKG")

    print("Done.")


if __name__ == "__main__":
    main()
