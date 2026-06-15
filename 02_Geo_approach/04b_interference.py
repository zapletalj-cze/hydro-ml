"""
Inference AOI: Apply trained levee detection model to a new geographic area.

Loads a trained SegFormer (or compatible) checkpoint, runs sliding-window
inference over a user-defined AOI restricted to river corridors (MERIT Hydro
buffer), stitches predictions with overlap averaging, and exports detected
levee centerlines as a GPKG.

Pipeline:
    1. Load AOI polygon and MERIT river corridor (intersect)
    2. Generate sliding-window patch grid over the corridor
    3. Per patch: extract DSM + TPI + canopy + canopy SD, normalize, forward pass
    4. Stitch patch predictions into a single probability raster
    5. Threshold + morphological cleanup -> connected components
    6. Per component: skeleton graph -> iterative longest-path extraction
    7. Simplify + filter, save as GPKG (EPSG:2180)

Vectorization paradigm (differs from 04_inference.py):
    Each connected high-probability region is reduced to its principal
    centerline(s) by repeatedly extracting the longest internal path
    (graph diameter). Short branches below MIN_LINE_LENGTH_M are discarded.
    This avoids skeleton-walking fragmentation: whole paths are emitted,
    not edges between junctions, so a continuous levee stays continuous.

Author:   Jakub Zapletal
Date:     2026-06-15
Version:  0.3
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.ndimage import uniform_filter
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import unary_union, linemerge

from osgeo import gdal, gdalconst
gdal.UseExceptions()

from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
from skimage.measure import label as label_components

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

# --- User-provided paths ---
AOI_PATH         = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\AOI_Poland.gpkg")
DSM_PATH         = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif")
CANOPY_PATH      = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_2180.tif")
CANOPY_SD_PATH   = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_SD_2180.tif")
MERIT_PATH       = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\riv_pfaf_2x_MERIT_Hydro_v07_Basin_flip_2180.gpkg")
MERIT_UPAREA_COL = "uparea"

CHECKPOINT_PATH  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight\best_model.pt")
NORM_STATS_PATH  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight\norm_stats.json")

OUTPUT_GPKG      = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight\interference_outputs\detected_levees.gpkg")

# Probability raster output (intermediate result, used by ensemble script).
# Derived from OUTPUT_GPKG path with _prob.tif suffix.
OUTPUT_PROB_TIF  = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_prob.tif")

# --- Geographic & raster constants (must match training) ---
CRS_TARGET      = 2180          # EPSG:2180 (PL-1992)
PATCH_SIZE_PX   = 256
PATCH_RES_M     = 10
PATCH_EXTENT_M  = PATCH_SIZE_PX * PATCH_RES_M    # 2560 m
STRIDE_PX       = 128            # 50% overlap
STRIDE_M        = STRIDE_PX * PATCH_RES_M        # 1280 m

TPI_RADII_PX    = [5, 10, 15]    # 50, 100, 150 m on 10 m grid
RIVER_BUFFER_M  = 500
MIN_UPAREA_KM2  = 10

# --- Model / inference constants ---
SEGFORMER_BACKBONE = "mit_b2"
N_INPUT_CHANNELS = 6
INFERENCE_BATCH  = 16
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# --- Postprocessing constants ---
PROB_THRESHOLD       = 0.5
MIN_COMPONENT_PX     = 50    # drop high-prob blobs smaller than this (noise)
CLOSING_RADIUS_PX    = 3     # morphological closing to bridge small along-line gaps
MIN_LINE_LENGTH_M    = 50    # discard extracted paths shorter than this

# Corridor masking. Patches already run only where their CENTER is in the
# corridor, so each patch extends ~1.28 km beyond it. Re-masking the stitched
# raster by the narrow corridor clips valid levees on the floodplain edge.
# Default off; set True (and tune POSTPROCESS_BUFFER_M) to restrict to channels.
APPLY_CORRIDOR_MASK  = False
POSTPROCESS_BUFFER_M = 500

# Geometry simplification (Douglas-Peucker tolerance, m). Removes pixel-staircase
# vertices for cleaner, lighter lines. 0 disables.
SIMPLIFY_TOLERANCE_M = 10


# ============================================================
# MODEL LOADING
# ============================================================

def adapt_first_conv_segformer(model, n_input_channels):
    """Replicate pretrained 3-channel patch_embed1.proj weights for N channels."""
    encoder = model.encoder
    first_conv = encoder.patch_embed1.proj
    old_weight = first_conv.weight.data
    out_ch, _, kh, kw = old_weight.shape

    new_weight = old_weight.repeat(1, (n_input_channels // 3) + 1, 1, 1)
    new_weight = new_weight[:, :n_input_channels, :, :]
    new_weight = new_weight / (n_input_channels / 3)

    new_conv = nn.Conv2d(
        n_input_channels, out_ch,
        kernel_size=(kh, kw), stride=first_conv.stride, padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    encoder.patch_embed1.proj = new_conv
    return model


def build_and_load_model():
    """Build SegFormer with 6-channel input, load checkpoint."""
    model = smp.Segformer(
        encoder_name=SEGFORMER_BACKBONE,
        encoder_weights=None,        # weights come from checkpoint
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
    """Load AOI polygon, reproject to target CRS, return single MultiPolygon."""
    aoi = gpd.read_file(AOI_PATH)
    if aoi.crs.to_epsg() != CRS_TARGET:
        aoi = aoi.to_crs(epsg=CRS_TARGET)
    return unary_union(aoi.geometry.tolist())


def build_river_corridor(aoi_geom):
    """Load MERIT reaches in AOI, filter by upstream area, buffer by RIVER_BUFFER_M."""
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs=f"EPSG:{CRS_TARGET}")
    aoi_bbox = aoi_geom.bounds  # (minx, miny, maxx, maxy)

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
    Generate sliding-window patch center coordinates covering the corridor.
    Returns a list of (center_x, center_y) tuples in EPSG:2180.
    """
    minx, miny, maxx, maxy = corridor_geom.bounds

    # Align grid origin to STRIDE_M for reproducibility
    x_start = (minx // STRIDE_M) * STRIDE_M + STRIDE_M / 2
    y_start = (miny // STRIDE_M) * STRIDE_M + STRIDE_M / 2

    centers = []
    y = y_start
    while y <= maxy + STRIDE_M:
        x = x_start
        while x <= maxx + STRIDE_M:
            # Keep patch if its bounding box intersects the corridor
            patch_box = box(x - PATCH_EXTENT_M / 2, y - PATCH_EXTENT_M / 2,
                            x + PATCH_EXTENT_M / 2, y + PATCH_EXTENT_M / 2)
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
    Read a window from an open GDAL dataset, resampled to target_pixels x target_pixels.
    bbox: (xmin, ymin, xmax, ymax) in dataset CRS.
    Returns float32 numpy array (target_pixels, target_pixels).
    """
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    # Compute source pixel offsets of the bbox corners
    px_ulx, px_uly = gdal.ApplyGeoTransform(inv_gt, bbox[0], bbox[3])
    px_lrx, px_lry = gdal.ApplyGeoTransform(inv_gt, bbox[2], bbox[1])

    col_off = int(np.floor(px_ulx))
    row_off = int(np.floor(px_uly))
    col_size = max(1, int(np.ceil(px_lrx - px_ulx)))
    row_size = max(1, int(np.ceil(px_lry - px_uly)))

    # Clip to dataset bounds
    raster_xsize = ds.RasterXSize
    raster_ysize = ds.RasterYSize

    # If window is fully outside, return zeros
    if col_off >= raster_xsize or row_off >= raster_ysize or col_off + col_size <= 0 or row_off + row_size <= 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    # Handle partial out-of-bounds by clipping and zero-padding
    read_col = max(0, col_off)
    read_row = max(0, row_off)
    read_col_size = min(col_size - (read_col - col_off), raster_xsize - read_col)
    read_row_size = min(row_size - (read_row - row_off), raster_ysize - read_row)

    if read_col_size <= 0 or read_row_size <= 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    # If we had to clip, scale target pixels proportionally; otherwise read direct
    if (read_col == col_off and read_row == row_off and
            read_col_size == col_size and read_row_size == row_size):
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray(
            read_col, read_row, read_col_size, read_row_size,
            buf_xsize=target_pixels, buf_ysize=target_pixels,
            resample_alg=resample,
        ).astype(np.float32)
        return arr

    # Partial OOB: read what we can, place into zero-padded output
    sub_target_w = int(round(target_pixels * read_col_size / col_size))
    sub_target_h = int(round(target_pixels * read_row_size / row_size))
    if sub_target_w == 0 or sub_target_h == 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    band = ds.GetRasterBand(1)
    sub = band.ReadAsArray(
        read_col, read_row, read_col_size, read_row_size,
        buf_xsize=sub_target_w, buf_ysize=sub_target_h,
        resample_alg=resample,
    ).astype(np.float32)

    out = np.zeros((target_pixels, target_pixels), dtype=np.float32)
    out_col_off = int(round(target_pixels * (read_col - col_off) / col_size))
    out_row_off = int(round(target_pixels * (read_row - row_off) / row_size))
    out[out_row_off:out_row_off + sub_target_h, out_col_off:out_col_off + sub_target_w] = sub
    return out


# ============================================================
# PATCH EXTRACTION + NORMALIZATION
# ============================================================

def compute_tpi(z, radius_px):
    """TPI = z minus mean of NxN neighborhood. Matches training pipeline."""
    size = 2 * radius_px + 1
    return z - uniform_filter(z, size=size, mode="nearest")  # match patch generator


def extract_patch(center_x, center_y, dsm_ds, canopy_ds, canopy_sd_ds):
    """
    Extract a 6-channel patch (256x256) at given center coordinates.
    Returns float32 array (6, 256, 256): DSM, TPI x3, Canopy, Canopy SD.
    """
    bbox = (
        center_x - PATCH_EXTENT_M / 2,
        center_y - PATCH_EXTENT_M / 2,
        center_x + PATCH_EXTENT_M / 2,
        center_y + PATCH_EXTENT_M / 2,
    )

    dsm       = read_window(dsm_ds,       bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    canopy    = read_window(canopy_ds,    bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    canopy_sd = read_window(canopy_sd_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)

    tpi_channels = [compute_tpi(dsm, r) for r in TPI_RADII_PX]

    patch = np.stack([dsm, *tpi_channels, canopy, canopy_sd], axis=0)
    return patch


def normalize_patch(patch, norm_stats):
    """
    Normalize 6-channel patch.
        DSM: per-patch median subtraction (generalization across regions)
        TPI / canopy / canopy_sd: per-channel z-score from training stats
    """
    out = patch.copy()

    # Channel 0: DSM — per-patch median
    out[0] = out[0] - np.median(out[0])

    # Channels 1..5: per-channel z-score
    channel_names = ["tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd"]
    for i, name in enumerate(channel_names, start=1):
        mean = norm_stats[name]["mean"]
        std  = norm_stats[name]["std"]
        out[i] = (out[i] - mean) / (std + 1e-8)

    return out


# ============================================================
# BATCHED INFERENCE
# ============================================================

def run_inference(model, centers, norm_stats):
    """
    Run batched inference. For each center, return (center, prediction_256x256).
    """
    dsm_ds       = gdal.Open(str(DSM_PATH))
    canopy_ds    = gdal.Open(str(CANOPY_PATH))
    canopy_sd_ds = gdal.Open(str(CANOPY_SD_PATH))

    if dsm_ds is None or canopy_ds is None or canopy_sd_ds is None:
        raise RuntimeError("Failed to open one or more raster sources")

    predictions = []
    n_centers = len(centers)

    with torch.no_grad():
        for batch_start in tqdm(range(0, n_centers, INFERENCE_BATCH),
                                desc="Inference", total=(n_centers + INFERENCE_BATCH - 1) // INFERENCE_BATCH):
            batch_centers = centers[batch_start:batch_start + INFERENCE_BATCH]

            patches = []
            for cx, cy in batch_centers:
                p = extract_patch(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds)
                p = normalize_patch(p, norm_stats)
                patches.append(p)

            batch = torch.from_numpy(np.stack(patches, axis=0)).float().to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, 256, 256)

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
    Stitch patch predictions into a single probability raster covering the corridor bbox.
    Returns (prob_raster, geotransform) where geotransform is
    (origin_x, pixel_w, 0, origin_y, 0, -pixel_h).
    """
    minx, miny, maxx, maxy = corridor_geom.bounds

    # Align to STRIDE grid so patch positions land cleanly
    origin_x = (minx // STRIDE_M) * STRIDE_M - PATCH_EXTENT_M / 2
    origin_y = (maxy // STRIDE_M + 1) * STRIDE_M + PATCH_EXTENT_M / 2

    width_m  = (maxx - origin_x) + PATCH_EXTENT_M
    height_m = (origin_y - miny) + PATCH_EXTENT_M

    width_px  = int(np.ceil(width_m / PATCH_RES_M))
    height_px = int(np.ceil(height_m / PATCH_RES_M))

    sum_pred = np.zeros((height_px, width_px), dtype=np.float32)
    sum_wts  = np.zeros((height_px, width_px), dtype=np.float32)

    for cx, cy, pred in predictions:
        col_off = int(round((cx - PATCH_EXTENT_M / 2 - origin_x) / PATCH_RES_M))
        row_off = int(round((origin_y - cy - PATCH_EXTENT_M / 2) / PATCH_RES_M))

        # Clip in case patch extends outside the allocated raster
        r0, r1 = max(0, row_off), min(height_px, row_off + PATCH_SIZE_PX)
        c0, c1 = max(0, col_off), min(width_px, col_off + PATCH_SIZE_PX)
        pr0 = r0 - row_off
        pr1 = pr0 + (r1 - r0)
        pc0 = c0 - col_off
        pc1 = pc0 + (c1 - c0)

        sum_pred[r0:r1, c0:c1] += pred[pr0:pr1, pc0:pc1]
        sum_wts [r0:r1, c0:c1] += 1.0

    prob = np.zeros_like(sum_pred)
    mask = sum_wts > 0
    prob[mask] = sum_pred[mask] / sum_wts[mask]

    geotransform = (origin_x, PATCH_RES_M, 0.0, origin_y, 0.0, -PATCH_RES_M)
    return prob, geotransform


def rasterize_corridor_mask(corridor_geom, geotransform, shape):
    """Rasterize the corridor polygon to a boolean mask matching the prob raster."""
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
    """Save stitched probability raster as a GeoTIFF (float32, LZW compressed, EPSG:2180)."""
    height_px, width_px = prob.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(output_path), width_px, height_px, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"],
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
# VECTORIZATION: probability -> centerlines (graph longest-path)
# ============================================================
#
# Paradigm: each connected high-probability region is reduced to its principal
# centerline(s) by repeatedly extracting the longest internal path through the
# region's skeleton graph (the graph diameter). Short branches are discarded by
# the length filter. This is robust against the fragmentation and spurs that
# plague naive skeleton-walking, and is simple to state and defend.

def _skeleton_graph(skel_mask):
    """
    Build an 8-connectivity graph from a skeleton mask.
    Nodes are (row, col) pixels; edge weights are Euclidean pixel distances
    (1 for orthogonal, sqrt(2) for diagonal neighbours).
    """
    G = nx.Graph()
    pts = set(map(tuple, np.argwhere(skel_mask).tolist()))
    G.add_nodes_from(pts)
    for (r, c) in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pts and not G.has_edge((r, c), nb):
                    w = 1.0 if (dr == 0 or dc == 0) else 1.4142135623730951
                    G.add_edge((r, c), nb, weight=w)
    return G


def _longest_path_nodes(G):
    """
    Approximate the longest shortest-path (graph diameter) via double Dijkstra:
    from an arbitrary node find the farthest node A, then from A find the
    farthest node B; the A-B shortest path is (near) the diameter. Exact for
    trees; skeletons are nearly trees, so any small loop error is negligible.
    Returns the ordered list of nodes on the path.
    """
    start = next(iter(G.nodes))
    lengths_1 = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    node_a = max(lengths_1, key=lengths_1.get)

    paths_a = nx.single_source_dijkstra_path(G, node_a, weight="weight")
    lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
    node_b = max(lengths_a, key=lengths_a.get)

    return paths_a[node_b], lengths_a[node_b]   # (node list, path length in px)


def _path_to_linestring(path_nodes, geotransform):
    """Convert an ordered list of (row, col) pixels to a world-coordinate LineString."""
    origin_x, pixel_w, _, origin_y, _, pixel_h_neg = geotransform
    pixel_h = -pixel_h_neg
    coords = [
        (origin_x + (c + 0.5) * pixel_w, origin_y - (r + 0.5) * pixel_h)
        for (r, c) in path_nodes
    ]
    return LineString(coords)


def extract_centerlines(component_skeleton, geotransform, min_length_m):
    """
    Iterative longest-path extraction for one connected component's skeleton.

    Repeatedly:
        1. take the longest path (diameter) of the current graph,
        2. emit it if it is at least min_length_m,
        3. remove its nodes; the remainder may split into sub-graphs (branches),
        4. process those sub-graphs the same way.

    Short spurs are naturally excluded: once the main path is removed, leftover
    spur fragments are shorter than min_length_m and dropped.
    """
    min_length_px = min_length_m / abs(geotransform[1])
    lines = []

    G_full = _skeleton_graph(component_skeleton)
    # Work queue of connected sub-graphs
    queue = [G_full.subgraph(c).copy() for c in nx.connected_components(G_full)]

    while queue:
        G = queue.pop()
        if G.number_of_nodes() < 2:
            continue

        path_nodes, path_len_px = _longest_path_nodes(G)

        if path_len_px < min_length_px:
            continue   # whole remaining sub-graph is shorter than a levee; drop

        lines.append(_path_to_linestring(path_nodes, geotransform))

        # Remove the extracted path; re-queue any remaining branches
        G.remove_nodes_from(path_nodes)
        for c in nx.connected_components(G):
            sub = G.subgraph(c).copy()
            if sub.number_of_nodes() >= 2:
                queue.append(sub)

    return lines


def probability_to_centerlines(prob_raster, corridor_mask, geotransform):
    """
    Convert a probability raster to levee centerlines.

    Chain:
        threshold -> (optional corridor mask) -> closing -> remove small blobs
        -> per connected component: skeletonize -> longest-path extraction
        -> simplify -> length filter

    Prints diagnostics so it is clear where geometry is gained or lost.
    Returns a GeoDataFrame of LineStrings in EPSG:2180.
    """
    above = prob_raster > PROB_THRESHOLD
    print(f"    pixels > {PROB_THRESHOLD}: {above.sum():,}")

    if APPLY_CORRIDOR_MASK:
        binary = above & corridor_mask
        print(f"    after corridor mask:  {binary.sum():,} "
              f"({100 * binary.sum() / max(above.sum(), 1):.0f}% kept)")
    else:
        binary = above

    if CLOSING_RADIUS_PX > 0:
        binary = binary_closing(binary, disk(CLOSING_RADIUS_PX))
    binary = remove_small_objects(binary, min_size=MIN_COMPONENT_PX, connectivity=2)

    if binary.sum() == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    # Process each connected component independently
    labels, n_comp = label_components(binary, connectivity=2, return_num=True)
    print(f"    connected components: {n_comp}")

    all_lines = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = labels == comp_id
        comp_skeleton = skeletonize(comp_mask)
        comp_lines = extract_centerlines(comp_skeleton, geotransform, MIN_LINE_LENGTH_M)
        all_lines.extend(comp_lines)

    print(f"    extracted paths:      {len(all_lines)}")

    if not all_lines:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    # Simplify (Douglas-Peucker) to remove pixel-staircase vertices
    if SIMPLIFY_TOLERANCE_M > 0:
        all_lines = [ln.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=False)
                     for ln in all_lines]

    # Length filter (paths shorter than a levee, after simplification)
    n_before = len(all_lines)
    all_lines = [ln for ln in all_lines if ln.length >= MIN_LINE_LENGTH_M]
    print(f"    after length filter:  {len(all_lines)} (removed {n_before - len(all_lines)})")

    gdf = gpd.GeoDataFrame(
        {"length_m": [ln.length for ln in all_lines]},
        geometry=all_lines, crs=f"EPSG:{CRS_TARGET}",
    )
    return gdf

def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"AOI: {AOI_PATH}")
    print(f"Output: {OUTPUT_GPKG}")
    print()

    # 1. Load model
    print("Loading model...")
    model, ckpt = build_and_load_model()
    print(f"  Best epoch from checkpoint: {ckpt.get('epoch', '?')}")
    print(f"  val_score: {ckpt.get('val_score', float('nan')):.4f}")

    # 2. Load normalization stats
    print("Loading norm stats...")
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    # 3. AOI + corridor
    print("Loading AOI + MERIT corridor...")
    aoi_geom = load_aoi_polygon()
    corridor_geom, merit_in_aoi = build_river_corridor(aoi_geom)
    print(f"  AOI area:       {aoi_geom.area / 1e6:.1f} km²")
    print(f"  Corridor area:  {corridor_geom.area / 1e6:.1f} km² ({corridor_geom.area / aoi_geom.area * 100:.1f}%)")
    print(f"  MERIT reaches:  {len(merit_in_aoi)}")

    # 4. Generate patch grid
    print("Generating patch centers...")
    centers = generate_patch_centers(corridor_geom)
    print(f"  Total patches: {len(centers)}")
    if len(centers) == 0:
        raise RuntimeError("No patches generated — AOI / corridor empty?")

    # 5. Run inference
    print("Running inference...")
    predictions = run_inference(model, centers, norm_stats)

    # 6. Stitch probability raster
    print("Stitching predictions...")
    prob_raster, geotransform = stitch_predictions(predictions, corridor_geom)
    print(f"  Probability raster: {prob_raster.shape} ({prob_raster.nbytes / 1e6:.1f} MB)")

    # 6b. Save prob raster as GeoTIFF (intermediate result for ensembling)
    print(f"Saving probability raster to {OUTPUT_PROB_TIF}...")
    OUTPUT_PROB_TIF.parent.mkdir(parents=True, exist_ok=True)
    save_prob_raster_geotiff(prob_raster, geotransform, OUTPUT_PROB_TIF)

    # 7. Rasterize corridor mask
    print("Rasterizing corridor mask...")
    corridor_mask = rasterize_corridor_mask(corridor_geom, geotransform, prob_raster.shape)

    # 8. Postprocess to vector
    print("Vectorizing (threshold -> components -> longest-path)...")
    detected = probability_to_centerlines(prob_raster, corridor_mask, geotransform)
    print(f"  Detected lines: {len(detected)}")
    if len(detected) > 0:
        print(f"  Total length:   {detected['length_m'].sum() / 1000:.1f} km")
        print(f"  Mean length:    {detected['length_m'].mean():.0f} m")
        print(f"  Max length:     {detected['length_m'].max():.0f} m")

    # 9. Save
    print(f"Saving GPKG to {OUTPUT_GPKG}...")
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    detected.to_file(OUTPUT_GPKG, driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()