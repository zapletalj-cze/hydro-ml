"""
Sliding-window levee inference over an AOI, vectorized to centerlines (GPKG).

Author: Jakub Zapletal
Date:   2026-04-21
"""

import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import geopandas as gpd
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from toolset.gis import Vector, Raster
from toolset import models
from toolset import inference

# ============================================================
# CONFIG
# ============================================================

# --- Input paths ---
AOI_PATH = Path("data/aoi.gpkg")
DSM_PATH = Path("data/dsm.tif")
CANOPY_PATH = Path("data/canopy_height.tif")
CANOPY_SD_PATH = Path("data/canopy_height_sd.tif")
WATER_PATH = Path("data/water_mask.tif")

MERIT_PATH = Path("data/merit_rivers.gpkg")
MERIT_UPAREA_COL = "uparea"

CHECKPOINT_PATH = Path("models/best_model.pt")
NORM_STATS_PATH = Path("models/norm_stats.json")

OUTPUT_GPKG = Path("output/levees_predicted.gpkg")

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

# z-scored channels, in stacking order after the DSM (must match training)
ZSCORE_CHANNELS = [
    "tpi_r5",
    "tpi_r10",
    "tpi_r15",
    "canopy_height",
    "canopy_height_sd",
]

RASTER_PATHS = {
    "dsm": DSM_PATH,
    "canopy_height": CANOPY_PATH,
    "canopy_height_sd": CANOPY_SD_PATH,
    "water": WATER_PATH,
}

# --- Postprocessing constants ---
PROB_THRESHOLD = 0.5
MIN_COMPONENT_PX = 50  # drop high-prob blobs smaller than this
CLOSING_RADIUS_PX = 3  # morphological closing to bridge small along-line gaps
MIN_LINE_LENGTH_M = 50  # discard extracted paths shorter than this

# Corridor masking: patches run wherever their bbox touches the corridor, so
# re-masking the stitched raster clips valid levees on the floodplain edge
APPLY_CORRIDOR_MASK = False

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
    model, ckpt = models.load_segformer_checkpoint(
        CHECKPOINT_PATH, SEGFORMER_BACKBONE, N_INPUT_CHANNELS, DEVICE
    )
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
    centers = inference.generate_patch_centers(corridor_geom, STRIDE_M, PATCH_EXTENT_M)
    print(f"  Total patches: {len(centers)}")
    if len(centers) == 0:
        raise RuntimeError("No patches generated - AOI / corridor empty?")

    if EXPORT_PATCH_GRID:
        print(f"Exporting patch grid to {OUTPUT_PATCH_GRID}...")
        export_patch_grid(centers, OUTPUT_PATCH_GRID)

    # 5. Run inference
    print("Running inference...")
    predictions = inference.run_inference(
        model,
        centers,
        norm_stats,
        RASTER_PATHS,
        PATCH_SIZE_PX,
        PATCH_EXTENT_M,
        TPI_RADII_PX,
        ZSCORE_CHANNELS,
        INFERENCE_BATCH,
        DEVICE,
        USE_TTA,
        TTA_MODE,
    )

    # 6. Stitch probability raster
    print("Stitching predictions...")
    prob_raster, geotransform = inference.stitch_predictions(
        predictions,
        corridor_geom.bounds,
        STRIDE_M,
        PATCH_EXTENT_M,
        PATCH_RES_M,
        PATCH_SIZE_PX,
        STITCH_METHOD,
    )
    print(
        f"  Probability raster: {prob_raster.shape} ({prob_raster.nbytes / 1e6:.1f} MB)"
    )

    # 6b. Save prob raster (intermediate result for ensembling)
    print(f"Saving probability raster to {OUTPUT_PROB_TIF}...")
    OUTPUT_PROB_TIF.parent.mkdir(parents=True, exist_ok=True)
    Raster.save_array(OUTPUT_PROB_TIF, prob_raster, geotransform, CRS_TARGET, nodata=-1.0)

    # 7. Optional corridor mask
    corridor_mask = None
    if APPLY_CORRIDOR_MASK:
        print("Rasterizing corridor mask...")
        corridor_mask = Raster.rasterize_geometries(
            [corridor_geom], geotransform, prob_raster.shape, CRS_TARGET
        ).astype(bool)

    # 8. Postprocess to vector
    print("Vectorizing (threshold -> components -> longest-path)...")
    detected = inference.probability_to_centerlines(
        prob_raster,
        geotransform,
        CRS_TARGET,
        PROB_THRESHOLD,
        MIN_COMPONENT_PX,
        CLOSING_RADIUS_PX,
        MIN_LINE_LENGTH_M,
        SIMPLIFY_TOLERANCE_M,
        corridor_mask,
    )
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
