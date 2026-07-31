"""
Sliding-window levee inference over an AOI, vectorized to centerlines (GPKG).

Author: Jakub Zapletal
Date:   2026-04-21
"""

import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
from skimage.measure import label as label_components

from tqdm import tqdm

from gis import Vector, Raster

# ============================================================
# CONFIG
# ============================================================

# --- Input paths ---
AOI_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\AOI_CZE.gpkg"
)
DSM_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\DSM_COP_30_Czechia.tif"
)
CANOPY_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\ETH_CanopyHeight_10m_Czechia.tif"
)
CANOPY_SD_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\ETH_CanopyHeight_10m_Czechia_SD.tif"
)
WATER_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\Watebodies_raster_CZE_10m.tif"
)

MERIT_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Czechia\MERIT_5514_EU.gpkg"
)
MERIT_UPAREA_COL = "uparea"

CHECKPOINT_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\best_model.pt"
)
NORM_STATS_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\norm_stats.json"
)

OUTPUT_GPKG = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\predictioons_cze\levees_predicted_CZE_Morava.gpkg"
)

# Probability raster (intermediate result, used by the ensemble script)
OUTPUT_PROB_TIF = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_prob.tif")
OUTPUT_PATCH_GRID = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_patch_grid.gpkg")

# --- Geographic & raster constants (must match training) ---
CRS_TARGET = 5514  # S-JTSK / Krovak East North
PATCH_SIZE_PX = 256
PATCH_RES_M = 10
PATCH_EXTENT_M = PATCH_SIZE_PX * PATCH_RES_M  # 2560 m
STRIDE_PX = 128  # 50% overlap
STRIDE_M = STRIDE_PX * PATCH_RES_M  # 1280 m

# Stitching of overlapping patches:
#   "feather" - raised-cosine weighted blend, no patch-boundary seams
#   "max"     - maximum across overlaps (propagates confident noise too)
#   "average" - plain mean (visible seams at patch boundaries)
STITCH_METHOD = "feather"

TPI_RADII_PX = [5, 10, 15]  # 50, 100, 150 m on a 10 m grid
RIVER_BUFFER_M = 500
MIN_UPAREA_KM2 = 2000  # match the training corridor (large rivers only)

# --- Model / inference constants ---
SEGFORMER_BACKBONE = "mit_b2"
N_INPUT_CHANNELS = 7
INFERENCE_BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Postprocessing constants ---
PROB_THRESHOLD = 0.5
MIN_COMPONENT_PX = 50  # drop high-prob blobs smaller than this
CLOSING_RADIUS_PX = 3  # morphological closing to bridge small along-line gaps
MIN_LINE_LENGTH_M = 50  # discard extracted paths shorter than this

# Corridor masking: patches run wherever their bbox touches the corridor, so
# re-masking the stitched raster clips valid levees on the floodplain edge
APPLY_CORRIDOR_MASK = False
POSTPROCESS_BUFFER_M = 500

# Douglas-Peucker simplification tolerance [m], 0 disables
SIMPLIFY_TOLERANCE_M = 10

# --- Test-time augmentation ---
#   "d4"   - 4 rotations x 2 mirrors = 8 passes
#   "flip" - identity + horizontal + vertical + both = 4 passes
USE_TTA = True
TTA_MODE = "d4"

# --- Diagnostics ---
# Export the used patch squares to check gaps against the patch grid
EXPORT_PATCH_GRID = False


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
    """Build SegFormer with 7-channel input, load checkpoint."""
    model = smp.Segformer(
        encoder_name=SEGFORMER_BACKBONE,
        encoder_weights=None,  # weights come from the checkpoint
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
    """Load AOI polygon, reproject to target CRS, return single geometry."""
    aoi = Vector.load_vector(AOI_PATH, target_epsg=CRS_TARGET)
    return unary_union(aoi.geometry.tolist())


def build_river_corridor(aoi_geom):
    """MERIT reaches in the AOI, filtered by upstream area, buffered."""
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs=f"EPSG:{CRS_TARGET}")
    aoi_bbox = aoi_geom.bounds

    merit = Vector.load_vector(MERIT_PATH, bbox=aoi_bbox, target_epsg=CRS_TARGET)
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
    """Sliding-window patch centers covering the corridor, as (x, y) tuples."""
    minx, miny, maxx, maxy = corridor_geom.bounds

    # Align grid origin to STRIDE_M for reproducibility
    x_start = (minx // STRIDE_M) * STRIDE_M + STRIDE_M / 2
    y_start = (miny // STRIDE_M) * STRIDE_M + STRIDE_M / 2

    centers = []
    y = y_start
    while y <= maxy + STRIDE_M:
        x = x_start
        while x <= maxx + STRIDE_M:
            # Keep the patch if its bounding box intersects the corridor
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
# PATCH EXTRACTION + NORMALIZATION
# ============================================================


def extract_patch(center_x, center_y, dsm_ds, canopy_ds, canopy_sd_ds, water_ds):
    """
    Extract a 7-channel patch (256x256) at the given center:
    DSM, TPI x3, canopy, canopy SD, binary water mask.
    """
    bbox = (
        center_x - PATCH_EXTENT_M / 2,
        center_y - PATCH_EXTENT_M / 2,
        center_x + PATCH_EXTENT_M / 2,
        center_y + PATCH_EXTENT_M / 2,
    )

    dsm = Raster.read_window(dsm_ds, bbox, PATCH_SIZE_PX, "bilinear")
    canopy = Raster.read_window(canopy_ds, bbox, PATCH_SIZE_PX, "bilinear")
    canopy_sd = Raster.read_window(canopy_sd_ds, bbox, PATCH_SIZE_PX, "bilinear")
    # Water mask is categorical: nearest neighbour, then strict 0/1
    water = Raster.read_window(water_ds, bbox, PATCH_SIZE_PX, "nearest")
    water = (water > 0.5).astype(np.float32)

    tpi_channels = [Raster.compute_tpi(dsm, r) for r in TPI_RADII_PX]

    patch = np.stack([dsm, *tpi_channels, canopy, canopy_sd, water], axis=0)
    return patch


def normalize_patch(patch, norm_stats):
    """
    Normalize a 7-channel patch, matching training:
    DSM per-patch median subtraction, TPI/canopy z-score, water kept 0/1.
    """
    out = patch.copy()

    # Channel 0: DSM, per-patch median
    out[0] = out[0] - np.median(out[0])

    # Channels 1..5: per-channel z-score
    channel_names = [
        "tpi_r5",
        "tpi_r10",
        "tpi_r15",
        "canopy_height",
        "canopy_height_sd",
    ]
    for i, name in enumerate(channel_names, start=1):
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        out[i] = (out[i] - mean) / (std + 1e-8)

    # Channel 6: water, kept binary
    return out


# ============================================================
# BATCHED INFERENCE
# ============================================================


def _tta_transforms(mode):
    """
    (forward, inverse) tensor transform pairs for test-time augmentation,
    applied to (B, C, H, W); the output is mapped back before averaging.
    """
    def rot(x, k):
        return torch.rot90(x, k, dims=(2, 3))

    def hflip(x):
        return torch.flip(x, dims=(3,))

    if mode == "flip":
        return [
            (lambda x: x, lambda y: y),
            (hflip, hflip),
            (lambda x: torch.flip(x, dims=(2,)), lambda y: torch.flip(y, dims=(2,))),
            (lambda x: torch.flip(x, dims=(2, 3)), lambda y: torch.flip(y, dims=(2, 3))),
        ]

    transforms = []
    for f in (False, True):
        for k in (0, 1, 2, 3):
            def fwd(x, f=f, k=k):
                return rot(hflip(x) if f else x, k)

            def inv(y, f=f, k=k):
                yk = rot(y, -k)
                return hflip(yk) if f else yk

            transforms.append((fwd, inv))
    return transforms


def run_inference(model, centers, norm_stats):
    """Batched inference; returns a list of (cx, cy, prediction_256x256)."""
    dsm_ds = Raster.open_raster(DSM_PATH)
    canopy_ds = Raster.open_raster(CANOPY_PATH)
    canopy_sd_ds = Raster.open_raster(CANOPY_SD_PATH)
    water_ds = Raster.open_raster(WATER_PATH)

    predictions = []
    n_centers = len(centers)
    tta = _tta_transforms(TTA_MODE) if USE_TTA else [(lambda x: x, lambda y: y)]

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, n_centers, INFERENCE_BATCH),
            desc="Inference",
            total=(n_centers + INFERENCE_BATCH - 1) // INFERENCE_BATCH,
        ):
            batch_centers = centers[batch_start : batch_start + INFERENCE_BATCH]

            patches = []
            for cx, cy in batch_centers:
                p = extract_patch(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds, water_ds)
                p = normalize_patch(p, norm_stats)
                patches.append(p)

            batch = torch.from_numpy(np.stack(patches, axis=0)).float().to(DEVICE)

            prob_sum = None
            for fwd, inv in tta:
                logits = model(fwd(batch))
                prob = inv(torch.sigmoid(logits))
                prob_sum = prob if prob_sum is None else prob_sum + prob
            probs = (prob_sum / len(tta)).cpu().numpy()[:, 0]  # (B, 256, 256)

            for (cx, cy), prob in zip(batch_centers, probs):
                predictions.append((cx, cy, prob))

    dsm_ds = None
    canopy_ds = None
    canopy_sd_ds = None
    water_ds = None

    return predictions


# ============================================================
# STITCHING
# ============================================================


def _blend_window(size_px, floor=0.02):
    """2D raised-cosine (Hann) window with a small floor for feathered stitching."""
    n = np.arange(size_px)
    w1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size_px - 1)))
    w1d = np.clip(w1d, floor, None)
    return np.outer(w1d, w1d).astype(np.float32)


def stitch_predictions(predictions, corridor_geom):
    """
    Stitch patch predictions into one probability raster over the corridor bbox.
    Returns (prob_raster, geotransform).
    """
    minx, miny, maxx, maxy = corridor_geom.bounds

    # Align to the STRIDE grid so patch positions land cleanly
    origin_x = (minx // STRIDE_M) * STRIDE_M - PATCH_EXTENT_M / 2
    origin_y = (maxy // STRIDE_M + 1) * STRIDE_M + PATCH_EXTENT_M / 2

    width_m = (maxx - origin_x) + PATCH_EXTENT_M
    height_m = (origin_y - miny) + PATCH_EXTENT_M

    width_px = int(np.ceil(width_m / PATCH_RES_M))
    height_px = int(np.ceil(height_m / PATCH_RES_M))

    def _offsets(cx, cy):
        col_off = int(round((cx - PATCH_EXTENT_M / 2 - origin_x) / PATCH_RES_M))
        row_off = int(round((origin_y - cy - PATCH_EXTENT_M / 2) / PATCH_RES_M))
        r0, r1 = max(0, row_off), min(height_px, row_off + PATCH_SIZE_PX)
        c0, c1 = max(0, col_off), min(width_px, col_off + PATCH_SIZE_PX)
        pr0 = r0 - row_off
        pr1 = pr0 + (r1 - r0)
        pc0 = c0 - col_off
        pc1 = pc0 + (c1 - c0)
        return (r0, r1, c0, c1, pr0, pr1, pc0, pc1)

    if STITCH_METHOD == "max":
        prob = np.zeros((height_px, width_px), dtype=np.float32)
        for cx, cy, pred in predictions:
            r0, r1, c0, c1, pr0, pr1, pc0, pc1 = _offsets(cx, cy)
            prob[r0:r1, c0:c1] = np.maximum(prob[r0:r1, c0:c1], pred[pr0:pr1, pc0:pc1])
    else:
        # Weighted blend; uniform window for "average", raised-cosine for "feather"
        if STITCH_METHOD == "feather":
            window = _blend_window(PATCH_SIZE_PX)
        else:
            window = np.ones((PATCH_SIZE_PX, PATCH_SIZE_PX), dtype=np.float32)

        sum_pred = np.zeros((height_px, width_px), dtype=np.float32)
        sum_wts = np.zeros((height_px, width_px), dtype=np.float32)
        for cx, cy, pred in predictions:
            r0, r1, c0, c1, pr0, pr1, pc0, pc1 = _offsets(cx, cy)
            w = window[pr0:pr1, pc0:pc1]
            sum_pred[r0:r1, c0:c1] += pred[pr0:pr1, pc0:pc1] * w
            sum_wts[r0:r1, c0:c1] += w

        prob = np.zeros_like(sum_pred)
        mask = sum_wts > 0
        prob[mask] = sum_pred[mask] / sum_wts[mask]

    geotransform = (origin_x, PATCH_RES_M, 0.0, origin_y, 0.0, -PATCH_RES_M)
    return prob, geotransform


def rasterize_corridor_mask(corridor_geom, geotransform, shape):
    """Rasterize the corridor polygon to a boolean mask matching the prob raster."""
    mask = Raster.rasterize_geometries(
        [corridor_geom], geotransform, shape, CRS_TARGET, burn_value=1
    )
    return mask.astype(bool)


# ============================================================
# VECTORIZATION: probability -> centerlines
# ============================================================
#
# Each connected high-probability region is reduced to its principal
# centerline(s) by repeatedly extracting the longest internal path through the
# region's skeleton graph; short branches fall below the length filter. Whole
# paths are emitted, so a continuous levee stays continuous.


def _neighbors(p, pts):
    """8-connectivity neighbours of pixel p present in the set pts."""
    r, c = p
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            q = (r + dr, c + dc)
            if q in pts:
                out.append(q)
    return out


def _degree_map(skel_mask):
    """Vectorized 8-connectivity degree of each skeleton pixel (0 elsewhere)."""
    s = skel_mask.astype(np.uint8)
    p = np.pad(s, 1)
    nb = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )
    return nb * s


def _chain_length_px(path):
    """Euclidean length (in pixels) of an ordered pixel path."""
    return sum(
        np.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
        for k in range(len(path) - 1)
    )


def _build_reduced_graph(pts, deg, nodes):
    """
    Reduced skeleton graph: endpoints (deg 1) and junctions (deg >= 3) are
    nodes, degree-2 pixel chains between them are edges (with pixel path and
    Euclidean length). One node per junction instead of per pixel keeps
    Dijkstra fast on large AOIs. Returns (graph, loop_paths), loop_paths
    being self-closing chains handled separately.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    loop_paths = []
    visited_starts = set()

    for node in nodes:
        for nb in _neighbors(node, pts):
            if (node, nb) in visited_starts:
                continue
            # Walk the degree-2 chain from node until the next node
            path = [node, nb]
            prev, cur = node, nb
            while deg.get(cur, 0) == 2:
                nxts = [n for n in _neighbors(cur, pts) if n != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
            end = path[-1]
            visited_starts.add((node, nb))
            if len(path) >= 2:
                visited_starts.add((end, path[-2]))

            length = _chain_length_px(path)
            if node == end:
                loop_paths.append(path)  # self-loop chain, emit separately
                continue
            if G.has_edge(node, end) and G[node][end]["weight"] >= length:
                continue  # keep the longer of parallel chains
            G.add_edge(node, end, weight=length, path=path)

    return G, loop_paths


def _walk_loop(pts):
    """Walk a pure loop component (all pixels degree 2) into an ordered path."""
    start = next(iter(pts))
    path = [start]
    visited = {start}
    cur = start
    while True:
        nxt = [n for n in _neighbors(cur, pts) if n not in visited]
        if not nxt:
            break
        cur = nxt[0]
        visited.add(cur)
        path.append(cur)
    return path


def _longest_path_nodes(G):
    """
    Longest weighted path (graph diameter) via double Dijkstra: farthest node A
    from an arbitrary start, then farthest node B from A. Exact for trees;
    reduced skeleton graphs are nearly trees. Returns (node list, length).
    """
    start = next(iter(G.nodes))
    l1 = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    node_a = max(l1, key=l1.get)
    paths_a = nx.single_source_dijkstra_path(G, node_a, weight="weight")
    lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
    node_b = max(lengths_a, key=lengths_a.get)
    return paths_a[node_b], lengths_a[node_b]


def _reconstruct(G, node_path):
    """Concatenate edge pixel-paths along a node sequence into one pixel path."""
    pixels = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        seg = list(G[a][b]["path"])
        if seg[0] != a:
            seg = seg[::-1]
        if pixels and pixels[-1] == seg[0]:
            pixels.extend(seg[1:])
        else:
            pixels.extend(seg)
    return pixels


def _path_to_linestring(path_nodes, geotransform):
    """Convert an ordered list of (row, col) pixels to a world LineString."""
    origin_x, pixel_w, _, origin_y, _, pixel_h_neg = geotransform
    pixel_h = -pixel_h_neg
    coords = [
        (origin_x + (c + 0.5) * pixel_w, origin_y - (r + 0.5) * pixel_h)
        for (r, c) in path_nodes
    ]
    return LineString(coords)


def extract_centerlines(coords, deg_map, geotransform, min_length_m):
    """
    Iterative longest-path extraction for one connected component, given its
    skeleton pixel coordinates and the global degree map (a skeleton pixel's
    8-neighbours are all in the same component, so the global degree equals
    the component degree and the raster is skeletonized only once).

    Repeatedly: take the diameter path of the current graph, emit it if long
    enough, remove its edges and process the remaining sub-graphs the same
    way. Leftover branches shorter than min_length_m are dropped.
    """
    min_length_px = min_length_m / abs(geotransform[1])

    pts = set(map(tuple, coords.tolist()))
    if len(pts) < 2:
        return []

    deg = {(r, c): int(deg_map[r, c]) for (r, c) in pts}
    nodes = {p for p in pts if deg[p] != 2}

    lines = []

    # Pure loop component (no endpoints or junctions)
    if not nodes:
        path = _walk_loop(pts)
        if _chain_length_px(path) >= min_length_px:
            lines.append(_path_to_linestring(path, geotransform))
        return lines

    G, loop_paths = _build_reduced_graph(pts, deg, nodes)

    for lp in loop_paths:
        if _chain_length_px(lp) >= min_length_px:
            lines.append(_path_to_linestring(lp, geotransform))

    queue = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    while queue:
        sub = queue.pop()
        if sub.number_of_edges() == 0:
            continue

        node_path, total_len = _longest_path_nodes(sub)
        if total_len < min_length_px:
            continue

        lines.append(_path_to_linestring(_reconstruct(sub, node_path), geotransform))

        for a, b in zip(node_path[:-1], node_path[1:]):
            if sub.has_edge(a, b):
                sub.remove_edge(a, b)
        for c in nx.connected_components(sub):
            s2 = sub.subgraph(c).copy()
            if s2.number_of_edges() >= 1:
                queue.append(s2)

    return lines


def probability_to_centerlines(prob_raster, corridor_mask, geotransform):
    """
    Probability raster -> levee centerlines:
    threshold -> (optional corridor mask) -> closing -> remove small blobs
    -> skeletonize once -> per component longest-path -> simplify -> filter.
    Returns a GeoDataFrame of LineStrings.
    """
    above = prob_raster > PROB_THRESHOLD
    print(f"    pixels > {PROB_THRESHOLD}: {above.sum():,}")

    if APPLY_CORRIDOR_MASK:
        binary = above & corridor_mask
        print(
            f"    after corridor mask:  {binary.sum():,} "
            f"({100 * binary.sum() / max(above.sum(), 1):.0f}% kept)"
        )
    else:
        binary = above

    if CLOSING_RADIUS_PX > 0:
        binary = binary_closing(binary, disk(CLOSING_RADIUS_PX))
    binary = remove_small_objects(binary, min_size=MIN_COMPONENT_PX, connectivity=2)

    if binary.sum() == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    # Skeletonize the whole mask once, then split into components on the
    # skeleton; per-component skeletonization is far too slow on large rasters
    skel = skeletonize(binary)
    deg_map = _degree_map(skel)
    labels, n_comp = label_components(skel, connectivity=2, return_num=True)
    print(f"    connected components: {n_comp}")

    coords_all = np.argwhere(skel)
    if coords_all.size == 0:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")
    comp_ids = labels[coords_all[:, 0], coords_all[:, 1]]
    order = np.argsort(comp_ids, kind="stable")
    coords_all = coords_all[order]
    comp_ids = comp_ids[order]
    cut = np.flatnonzero(np.diff(comp_ids)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(comp_ids)]))

    all_lines = []
    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Vectorizing components"):
        comp_lines = extract_centerlines(coords_all[s:e], deg_map, geotransform, MIN_LINE_LENGTH_M)
        all_lines.extend(comp_lines)

    print(f"    extracted paths:      {len(all_lines)}")

    if not all_lines:
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs=f"EPSG:{CRS_TARGET}")

    # Simplify to remove pixel-staircase vertices
    if SIMPLIFY_TOLERANCE_M > 0:
        all_lines = [
            ln.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=False)
            for ln in all_lines
        ]

    # Length filter after simplification
    n_before = len(all_lines)
    all_lines = [ln for ln in all_lines if ln.length >= MIN_LINE_LENGTH_M]
    print(
        f"    after length filter:  {len(all_lines)} (removed {n_before - len(all_lines)})"
    )

    gdf = gpd.GeoDataFrame(
        {"length_m": [ln.length for ln in all_lines]},
        geometry=all_lines,
        crs=f"EPSG:{CRS_TARGET}",
    )
    return gdf


def export_patch_grid(centers, output_path):
    """Save the used patch squares as a GPKG."""
    squares = [
        box(
            cx - PATCH_EXTENT_M / 2,
            cy - PATCH_EXTENT_M / 2,
            cx + PATCH_EXTENT_M / 2,
            cy + PATCH_EXTENT_M / 2,
        )
        for cx, cy in centers
    ]
    gdf = gpd.GeoDataFrame(
        {"center_x": [c[0] for c in centers], "center_y": [c[1] for c in centers]},
        geometry=squares,
        crs=f"EPSG:{CRS_TARGET}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Vector.save_vector(gdf, output_path)


# ============================================================
# MAIN
# ============================================================


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
    print(
        f"  Corridor area:  {corridor_geom.area / 1e6:.1f} km² ({corridor_geom.area / aoi_geom.area * 100:.1f}%)"
    )
    print(f"  MERIT reaches:  {len(merit_in_aoi)}")

    # 4. Generate patch grid
    print("Generating patch centers...")
    centers = generate_patch_centers(corridor_geom)
    print(f"  Total patches: {len(centers)}")
    if len(centers) == 0:
        raise RuntimeError("No patches generated - AOI / corridor empty?")

    if EXPORT_PATCH_GRID:
        print(f"Exporting patch grid to {OUTPUT_PATCH_GRID}...")
        export_patch_grid(centers, OUTPUT_PATCH_GRID)

    # 5. Run inference
    print("Running inference...")
    predictions = run_inference(model, centers, norm_stats)

    # 6. Stitch probability raster
    print("Stitching predictions...")
    prob_raster, geotransform = stitch_predictions(predictions, corridor_geom)
    print(
        f"  Probability raster: {prob_raster.shape} ({prob_raster.nbytes / 1e6:.1f} MB)"
    )

    # 6b. Save prob raster (intermediate result for ensembling)
    print(f"Saving probability raster to {OUTPUT_PROB_TIF}...")
    OUTPUT_PROB_TIF.parent.mkdir(parents=True, exist_ok=True)
    Raster.save_array(OUTPUT_PROB_TIF, prob_raster, geotransform, CRS_TARGET, nodata=-1.0)

    # 7. Rasterize corridor mask
    print("Rasterizing corridor mask...")
    corridor_mask = rasterize_corridor_mask(
        corridor_geom, geotransform, prob_raster.shape
    )

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
    Vector.save_vector(detected, OUTPUT_GPKG)
    print("Done.")


if __name__ == "__main__":
    main()
