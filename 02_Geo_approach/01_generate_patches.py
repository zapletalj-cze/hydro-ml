"""
Levee Detection - Patch Generation Pipeline
============================================

Generates training patches for levee detection from Copernicus DSM,
stratified by upstream catchment area using MERIT Basins.

Author:   Jakub Zapletal
Date:     2026-04-12
Version:  0.1
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter, zoom
from shapely.geometry import Point, box
from shapely.ops import substring
from tqdm import tqdm


# ------------------------------------------------------------
# Input paths
# ------------------------------------------------------------
BDOT_GPKG        = r"C:\data\bdot10k_waly.gpkg"
MERIT_BASINS_SHP = r"C:\data\riv_pfaf_02_MERIT_Hydro_v07_Basins_v01.shp"
COPDEM_TIFF      = r"C:\data\copdem_glo30_poland.tif"

OUTPUT_DIR = Path(r"C:\data\patches_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Spatial reference
# ------------------------------------------------------------
TARGET_CRS = "EPSG:2180"        # PUWG 1992, Polish national grid
POLAND_BBOX_WGS84 = (14.0, 49.0, 24.2, 55.0)


# ------------------------------------------------------------
# Catchment size categories (upstream area in km²)
# ------------------------------------------------------------
UPAREA_BINS = {
    "S": (10, 1000),         # small tributaries
    "M": (1000, 10000),      # medium rivers
    "L": (10000, 1e9),       # large rivers (Vistula, Oder, lower Warta)
}


# ------------------------------------------------------------
# Levee processing
# ------------------------------------------------------------
SEGMENT_LENGTH_M    = 500     # cut levees into 500 m chunks
MIN_LEVEE_LENGTH_M  = 100     # ignore anything shorter
MAX_DIST_TO_REACH_M = 1000    # drop segments farther than this from any MERIT reach


# ------------------------------------------------------------
# Patch geometry
# ------------------------------------------------------------
PATCH_SIZE_PX = 256                          # output patch dimension
PATCH_RES_M   = 10                           # target resolution
PATCH_SIZE_M  = PATCH_SIZE_PX * PATCH_RES_M  # 2560 m on a side


# ------------------------------------------------------------
# Label rasterization
# ------------------------------------------------------------
LEVEE_BUFFER_M = 15           # buffer around BDOT lines for labels


# ------------------------------------------------------------
# Negative sampling
# ------------------------------------------------------------
NEG_EXCLUSION_BUFFER_M = 100  # keep negative patches at least this far from any levee
NEG_MAX_DISTANCE_M     = 5000 # ...but within this distance, to stay in similar terrain


# ------------------------------------------------------------
# DSM derivatives
# ------------------------------------------------------------
TPI_RADII_PX = [5, 10, 15]    # 50 m, 100 m, 150 m at 10 m resolution


# ============================================================
# SECTION 2: Load BDOT levees and MERIT Basins
# ============================================================

def load_bdot_levees(path, target_crs):
    """
    Load BDOT flood-protection levees (wał przeciwpowodziowy).
    Assumes the GPKG has a single pre-filtered layer.
    """
    gdf_levees = gpd.read_file(path)

    if str(gdf_levees.crs) != target_crs:
        gdf_levees = gdf_levees.to_crs(target_crs)

    gdf_levees = gdf_levees[gdf_levees.geometry.notna() & ~gdf_levees.geometry.is_empty].copy()

    return gdf_levees


def load_merit_basins(path, bbox_wgs84, target_crs):
    """
    Load MERIT Basins river reaches, clip to Poland bbox.
    The full pfaf_02 shapefile covers all of Europe, so we filter at
    read time to keep memory usage reasonable.
    """
    bbox_geom = box(*bbox_wgs84)
    gdf_basins = gpd.read_file(path, bbox=bbox_geom)

    if str(gdf_basins.crs) != target_crs:
        gdf_basins = gdf_basins.to_crs(target_crs)

    return gdf_basins


# ============================================================
# SECTION 3: Cut levees into fixed-length segments
# ============================================================

def segment_line(line, segment_length, min_length):
    """
    Cut a single LineString into chunks of approximately segment_length.
    Returns an empty list if the line is shorter than min_length.

    The actual segment length is adjusted so all chunks are equal —
    this avoids leaving a tiny remainder at the end.
    """
    if line.length < min_length:
        return []

    n_segments = max(1, round(line.length / segment_length))
    actual_length = line.length / n_segments

    segments = []
    for i in range(n_segments):
        start = i * actual_length
        end = (i + 1) * actual_length
        seg = substring(line, start, end)
        if seg.length > 0:
            segments.append(seg)

    return segments


def segment_levees(gdf_levees, segment_length, min_length):
    """
    Apply segmentation to all levee features.
    Handles both LineString and MultiLineString inputs.
    Returns a new GeoDataFrame where each row is one segment.
    """
    segment_records = []

    for idx, row in gdf_levees.iterrows():
        geom = row.geometry

        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            continue

        for part in parts:
            for seg in segment_line(part, segment_length, min_length):
                segment_records.append({
                    "source_idx": idx,
                    "geometry": seg,
                    "length_m": seg.length,
                })

    gdf_segments = gpd.GeoDataFrame(
        segment_records,
        geometry="geometry",
        crs=gdf_levees.crs,
    )
    gdf_segments["segment_id"] = range(len(gdf_segments))

    return gdf_segments


# ============================================================
# SECTION 4: Assign upstream area to each levee segment
# ============================================================

def assign_uparea_to_segments(gdf_segments, gdf_reaches, max_dist):
    """
    For each levee segment, find the nearest MERIT river reach
    and copy its upstream area (uparea, km²) and stream order.

    Segments farther than max_dist from any reach are dropped — these
    typically represent small streams that MERIT doesn't resolve, or
    levees protecting features unrelated to a mapped river.
    """
    gdf_joined = gpd.sjoin_nearest(
        gdf_segments,
        gdf_reaches[["COMID", "uparea", "order", "geometry"]],
        how="left",
        distance_col="dist_to_reach_m",
    )

    gdf_joined = gdf_joined.drop_duplicates(subset="segment_id", keep="first")
    gdf_joined = gdf_joined.drop(columns=["index_right"], errors="ignore")

    gdf_joined = gdf_joined[gdf_joined["dist_to_reach_m"] <= max_dist].copy()

    return gdf_joined


# ============================================================
# SECTION 5: Categorize segments by catchment size
# ============================================================

def categorize_uparea(uparea_km2, bins):
    """
    Map an upstream area value to a category label (S/M/L).
    Returns None if the value falls outside all defined bins.
    """
    if uparea_km2 is None or np.isnan(uparea_km2):
        return None

    for label, (lo, hi) in bins.items():
        if lo <= uparea_km2 < hi:
            return label
    return None


def add_category_column(gdf_segments, bins):
    """
    Add a 'category' column based on uparea.
    Drops segments that don't fall into any category.
    """
    gdf_segments = gdf_segments.copy()
    gdf_segments["category"] = gdf_segments["uparea"].apply(
        lambda x: categorize_uparea(x, bins)
    )
    gdf_segments = gdf_segments[gdf_segments["category"].notna()].copy()

    return gdf_segments


# ============================================================
# SECTION 6: Generate positive patch centers
# ============================================================

def generate_positive_centers(gdf_segments):
    """
    Each segment becomes one positive patch.
    The patch center is the midpoint of the segment.
    """
    gdf_centers = gdf_segments.copy()
    gdf_centers["geometry"] = gdf_centers.geometry.interpolate(0.5, normalized=True)
    gdf_centers["patch_type"] = "positive"
    gdf_centers["patch_id"] = [f"pos_{i:07d}" for i in range(len(gdf_centers))]
    return gdf_centers


# ============================================================
# SECTION 7: Generate negative patch centers
# ============================================================

def generate_negative_centers(gdf_positive_centers, gdf_segments,
                              exclusion_buffer, max_distance, seed=42):
    """
    For each positive patch, generate one negative patch in the same
    catchment category. The negative center must be:
      - within max_distance from the positive center (similar terrain)
      - at least exclusion_buffer away from any levee segment
    """
    rng = np.random.default_rng(seed)

    # Pre-build exclusion zones per category — union of buffered segments
    exclusion_by_cat = {}
    for cat in gdf_positive_centers["category"].unique():
        gdf_cat_segments = gdf_segments[gdf_segments["category"] == cat]
        exclusion_by_cat[cat] = gdf_cat_segments.geometry.buffer(exclusion_buffer).union_all()

    negative_records = []

    for _, row in tqdm(gdf_positive_centers.iterrows(),
                       total=len(gdf_positive_centers),
                       desc="Generating negatives"):
        cat = row["category"]
        cx, cy = row.geometry.x, row.geometry.y
        exclusion = exclusion_by_cat[cat]

        found = None
        for _ in range(50):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(max_distance * 0.3, max_distance)
            nx = cx + radius * np.cos(angle)
            ny = cy + radius * np.sin(angle)
            candidate = Point(nx, ny)

            if not exclusion.contains(candidate):
                found = candidate
                break

        if found is None:
            continue

        negative_records.append({
            "geometry": found,
            "category": cat,
            "patch_type": "negative",
            "source_positive_id": row["patch_id"],
            "uparea": row["uparea"],
            "COMID": row["COMID"],
        })

    gdf_negatives = gpd.GeoDataFrame(
        negative_records,
        geometry="geometry",
        crs=gdf_positive_centers.crs,
    )
    gdf_negatives["patch_id"] = [f"neg_{i:07d}" for i in range(len(gdf_negatives))]

    return gdf_negatives


# ============================================================
# SECTION 8: Extract DSM windows for all patches
# ============================================================

def extract_dsm_window(dsm_src, center_x, center_y,
                      patch_size_m, target_size_px):
    """
    Extract a square DSM window centered on (center_x, center_y),
    resampled to target_size_px x target_size_px.

    Returns (data, transform) or (None, None) if the window is outside
    the raster or contains only nodata.
    """
    half = patch_size_m / 2
    bounds = (center_x - half, center_y - half,
              center_x + half, center_y + half)

    window = from_bounds(*bounds, dsm_src.transform)
    window = window.round_offsets().round_lengths()

    data = dsm_src.read(1, window=window, boundless=True, fill_value=np.nan)

    if data.size == 0 or np.all(np.isnan(data)):
        return None, None

    if data.shape != (target_size_px, target_size_px):
        zoom_factors = (
            target_size_px / data.shape[0],
            target_size_px / data.shape[1],
        )
        data = zoom(data, zoom_factors, order=1)

    new_transform = rasterio.transform.from_origin(
        bounds[0], bounds[3],
        patch_size_m / target_size_px,
        patch_size_m / target_size_px,
    )

    return data.astype(np.float32), new_transform


def extract_all_dsm_patches(gdf_centers, dsm_path,
                            patch_size_m, patch_size_px):
    """
    Iterate over all patch centers and extract DSM windows.
    Returns a dict {patch_id: (dsm_array, transform)} for valid patches.
    """
    patches = {}

    with rasterio.open(dsm_path) as dsm_src:
        for _, row in tqdm(gdf_centers.iterrows(),
                           total=len(gdf_centers),
                           desc="Extracting DSM patches"):
            cx, cy = row.geometry.x, row.geometry.y
            data, transform = extract_dsm_window(
                dsm_src, cx, cy, patch_size_m, patch_size_px
            )
            if data is not None:
                patches[row["patch_id"]] = (data, transform)

    return patches


# ============================================================
# SECTION 9: Compute DSM derivatives (TPI)
# ============================================================

def compute_patch_derivatives(dsm_patch, tpi_radii):
    """
    Compute derivatives for a single DSM patch.
    Returns a dict of named arrays, one entry per derivative.
    """
    derivatives = {}

    for radius in tpi_radii:
        kernel_size = 2 * radius + 1
        local_mean = uniform_filter(dsm_patch, size=kernel_size, mode="nearest")
        derivatives[f"tpi_r{radius}"] = (dsm_patch - local_mean).astype(np.float32)

    return derivatives


def compute_all_derivatives(dsm_patches, tpi_radii):
    """
    Apply derivative computation to all DSM patches.
    Returns a dict {patch_id: {channel_name: array}} including the
    original DSM as 'dsm'.
    """
    result = {}
    for patch_id, (dsm_data, transform) in tqdm(dsm_patches.items(),
                                                desc="Computing derivatives"):
        channels = {"dsm": dsm_data}
        channels.update(compute_patch_derivatives(dsm_data, tpi_radii))
        result[patch_id] = {
            "channels": channels,
            "transform": transform,
        }
    return result


# ============================================================
# SECTION 10: Rasterize levee labels for each patch
# ============================================================

def rasterize_levees_for_patch(gdf_levees, transform, shape, buffer_m):
    """
    Rasterize buffered BDOT levees into a binary mask aligned with
    the patch grid.

    Returns a uint8 array with 1 where any buffered levee covers a pixel,
    0 elsewhere.
    """
    if len(gdf_levees) == 0:
        return np.zeros(shape, dtype=np.uint8)

    buffered = gdf_levees.geometry.buffer(buffer_m)
    shapes = [(geom, 1) for geom in buffered if geom is not None and not geom.is_empty]

    if not shapes:
        return np.zeros(shape, dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask


def add_labels_to_patches(patches_dict, gdf_levees, buffer_m, patch_size_px):
    """
    For each patch, find levees that intersect its bounding box and
    rasterize them as the label channel.
    """
    levees_sindex = gdf_levees.sindex

    for patch_id, patch_data in tqdm(patches_dict.items(),
                                     desc="Rasterizing labels"):
        transform = patch_data["transform"]

        minx = transform.c
        maxy = transform.f
        maxx = minx + transform.a * patch_size_px
        miny = maxy + transform.e * patch_size_px
        patch_bounds = (minx, miny, maxx, maxy)

        candidate_idx = list(levees_sindex.intersection(patch_bounds))
        gdf_candidates = gdf_levees.iloc[candidate_idx]

        patch_box = box(*patch_bounds)
        gdf_relevant = gdf_candidates[gdf_candidates.geometry.intersects(patch_box.buffer(buffer_m))]

        label = rasterize_levees_for_patch(
            gdf_relevant, transform, (patch_size_px, patch_size_px), buffer_m
        )

        patch_data["channels"]["label"] = label

    return patches_dict


# ============================================================
# SECTION 11: Save patches and metadata
# ============================================================

def save_patches(patches_dict, gdf_centers, output_dir):
    """
    Save each patch as a compressed .npz file containing all channels.
    Build a metadata CSV with one row per patch.
    """
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    gdf_centers_lookup = gdf_centers.set_index("patch_id")

    metadata_rows = []

    for patch_id, patch_data in tqdm(patches_dict.items(), desc="Saving patches"):
        channels = patch_data["channels"]
        transform = patch_data["transform"]

        npz_path = patches_dir / f"{patch_id}.npz"
        np.savez_compressed(npz_path, **channels)

        center_row = gdf_centers_lookup.loc[patch_id]

        metadata_rows.append({
            "patch_id":     patch_id,
            "patch_type":   center_row["patch_type"],
            "category":     center_row["category"],
            "center_x":     center_row.geometry.x,
            "center_y":     center_row.geometry.y,
            "uparea":       center_row.get("uparea", np.nan),
            "comid":        center_row.get("COMID", None),
            "n_label_px":   int(channels["label"].sum()),
            "transform_a":  transform.a,
            "transform_b":  transform.b,
            "transform_c":  transform.c,
            "transform_d":  transform.d,
            "transform_e":  transform.e,
            "transform_f":  transform.f,
            "npz_path":     str(npz_path.relative_to(output_dir)),
        })

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / "patches_metadata.csv", index=False)

    return metadata_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Section 2: load inputs
    gdf_levees = load_bdot_levees(BDOT_GPKG, TARGET_CRS)
    gdf_reaches = load_merit_basins(MERIT_BASINS_SHP, POLAND_BBOX_WGS84, TARGET_CRS)

    # Section 3: segment levees
    gdf_segments = segment_levees(gdf_levees, SEGMENT_LENGTH_M, MIN_LEVEE_LENGTH_M)

    # Section 4: assign upstream area
    gdf_segments = assign_uparea_to_segments(gdf_segments, gdf_reaches, MAX_DIST_TO_REACH_M)

    # Section 5: categorize
    gdf_segments = add_category_column(gdf_segments, UPAREA_BINS)
    gdf_segments.to_file(OUTPUT_DIR / "segments_categorized.gpkg", driver="GPKG")

    # Section 6: positive patch centers
    gdf_positive_centers = generate_positive_centers(gdf_segments)
    gdf_positive_centers.to_file(OUTPUT_DIR / "patch_centers_positive.gpkg", driver="GPKG")

    # Section 7: negative patch centers
    gdf_negative_centers = generate_negative_centers(
        gdf_positive_centers, gdf_segments,
        NEG_EXCLUSION_BUFFER_M, NEG_MAX_DISTANCE_M,
    )
    gdf_negative_centers.to_file(OUTPUT_DIR / "patch_centers_negative.gpkg", driver="GPKG")

    # Combine all centers
    gdf_all_centers = pd.concat([gdf_positive_centers, gdf_negative_centers], ignore_index=True)
    gdf_all_centers = gpd.GeoDataFrame(
        gdf_all_centers, geometry="geometry", crs=gdf_positive_centers.crs,
    )

    # Section 8: extract DSM windows
    dsm_patches = extract_all_dsm_patches(
        gdf_all_centers, COPDEM_TIFF, PATCH_SIZE_M, PATCH_SIZE_PX,
    )

    # Section 9: compute derivatives
    patches_with_features = compute_all_derivatives(dsm_patches, TPI_RADII_PX)

    # Section 10: rasterize labels
    patches_with_features = add_labels_to_patches(
        patches_with_features, gdf_levees, LEVEE_BUFFER_M, PATCH_SIZE_PX,
    )

    # Section 11: save everything
    metadata = save_patches(patches_with_features, gdf_all_centers, OUTPUT_DIR)
