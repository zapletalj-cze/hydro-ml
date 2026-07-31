"""
Training patch generation for levee detection (Poland, large rivers only).

Author: Jakub Zapletal
Date:   2026-04-06
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from toolset.gis import Vector
from toolset import patches

# ============================================================
# CONFIG
# ============================================================

# ------- Input paths ----------------------------------------
BDOT_GPKG = Path("data/levees.gpkg")
MERIT_GPKG = Path("data/merit_rivers.gpkg")
COPDEM_TIFF = Path("data/dsm.tif")
CANOPY_HEIGHT_TIFF = Path("data/canopy_height.tif")
CANOPY_HEIGHT_SD_TIFF = Path("data/canopy_height_sd.tif")
# Binary water mask on the same grid as the DSM
WATER_TIFF = Path("data/water_mask.tif")

OUTPUT_DIR = Path("output/patches_PL")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------- Spatial reference ----------------------------------
TARGET_EPSG = 2180  # PUWG 1992
POLAND_BBOX_WGS84 = (14.0, 49.0, 24.2, 55.0)  # MERIT file is in WGS84


# ------- MERIT reach attribute columns ----------------------
COMID_COL = "COMID"
NEXTDOWN_COL = "NextDownID"
UPAREA_COL = "uparea"


# ------- River / segment selection --------------------------
MIN_UPAREA_KM2 = 2000  # keep only segments on rivers at least this large
SEGMENT_LENGTH_M = 500
MIN_LEVEE_LENGTH_M = 100
MAX_DIST_TO_REACH_M = 500


# ------- Geographic hold-out (by drainage basin) ------------
# Run with REPORT_BASINS_ONLY = True first to print the basin table, then set
# TRAIN_BASINS to the outlet COMIDs kept for training (rest = test).
REPORT_BASINS_ONLY = True
TRAIN_BASINS = None  # list of outlet COMIDs; None keeps all basins


# ------- Patch geometry -------------------------------------
PATCH_SIZE_PX = 256
PATCH_RES_M = 10


# ------- Label rasterization --------------------------------
LEVEE_BUFFER_M = 15


# ------- Negative sampling ----------------------------------
RIVER_BUFFER_M = 500  # corridor half-width around large reaches
NEG_EXCLUSION_BUFFER_M = 100  # min distance of a negative from any levee
NEG_POS_RATIO = 3  # negatives per positive
NEG_TRIES_FACTOR = 40  # rejection-sampling budget = ratio * n_pos * this


# ------- DSM derivatives ------------------------------------
TPI_RADII_PX = [5, 10, 15]


# ------- Reproducibility ------------------------------------
SEED = 42


# Channel order written into each .npz (label added separately)
CHANNEL_KEYS = [
    "dsm",
    "tpi_r5",
    "tpi_r10",
    "tpi_r15",
    "canopy_height",
    "canopy_height_sd",
    "water",
]

RASTER_PATHS = {
    "dsm": COPDEM_TIFF,
    "canopy_height": CANOPY_HEIGHT_TIFF,
    "canopy_height_sd": CANOPY_HEIGHT_SD_TIFF,
    "water": WATER_TIFF,
}


# ============================================================
# MAIN
# ============================================================


def main():
    rng = np.random.default_rng(SEED)

    # Vector inputs
    gdf_levees = Vector.drop_empty_geometries(
        Vector.load_vector(BDOT_GPKG, target_epsg=TARGET_EPSG)
    )
    gdf_reaches = Vector.load_vector(
        MERIT_GPKG, bbox=POLAND_BBOX_WGS84, target_epsg=TARGET_EPSG
    )

    # Segment levees + attach the nearest reach
    gdf_segments = patches.segment_levees(
        gdf_levees, SEGMENT_LENGTH_M, MIN_LEVEE_LENGTH_M
    )
    gdf_segments = patches.assign_reach_to_segments(
        gdf_segments, gdf_reaches, MAX_DIST_TO_REACH_M,
        COMID_COL, UPAREA_COL, NEXTDOWN_COL,
    )

    # Basins + uparea filter + basin tag
    basin_of = patches.trace_basins(gdf_reaches, COMID_COL, NEXTDOWN_COL)
    gdf_segments = patches.filter_and_tag(
        gdf_segments, basin_of, MIN_UPAREA_KM2, COMID_COL, UPAREA_COL
    )
    gdf_segments["category"] = gdf_segments[UPAREA_COL].apply(patches.categorize_uparea)

    if len(gdf_segments) == 0:
        raise RuntimeError("No segments left after uparea filter; lower MIN_UPAREA_KM2?")

    patches.print_basin_report(
        gdf_segments, gdf_reaches, MIN_UPAREA_KM2, OUTPUT_DIR, basin_of,
        COMID_COL, UPAREA_COL,
    )
    Vector.save_vector(gdf_segments, OUTPUT_DIR / "segments_filtered.gpkg")

    if REPORT_BASINS_ONLY:
        print(
            "REPORT_BASINS_ONLY is True -> stopping. "
            "Set TRAIN_BASINS and REPORT_BASINS_ONLY=False to generate patches."
        )
        return

    # Geographic hold-out: keep only training basins
    if TRAIN_BASINS is not None:
        before = len(gdf_segments)
        gdf_segments = gdf_segments[
            gdf_segments["basin_id"].isin(set(TRAIN_BASINS))
        ].copy()
        print(
            f"Geographic hold-out: kept {len(gdf_segments)}/{before} positive segments "
            f"in basins {TRAIN_BASINS}"
        )
        if len(gdf_segments) == 0:
            raise RuntimeError("TRAIN_BASINS selected no segments; check basin ids.")

    # Patch centers
    gdf_pos = patches.generate_positive_centers(gdf_segments)

    corridor = patches.build_corridor(
        gdf_reaches, MIN_UPAREA_KM2, RIVER_BUFFER_M,
        COMID_COL, UPAREA_COL, basin_of, TRAIN_BASINS,
    )
    gdf_neg = patches.generate_negative_centers(
        len(gdf_pos),
        corridor,
        gdf_levees,
        gdf_reaches,
        basin_of,
        TARGET_EPSG,
        MIN_UPAREA_KM2,
        COMID_COL,
        UPAREA_COL,
        NEG_EXCLUSION_BUFFER_M,
        NEG_POS_RATIO,
        rng,
        NEG_TRIES_FACTOR,
    )
    print(
        f"Positives: {len(gdf_pos)} | negatives: {len(gdf_neg)} "
        f"(target ratio {NEG_POS_RATIO}:1)"
    )

    Vector.save_vector(gdf_pos, OUTPUT_DIR / "patch_centers_positive.gpkg")
    Vector.save_vector(gdf_neg, OUTPUT_DIR / "patch_centers_negative.gpkg")

    # Combine centers
    keep_cols = [
        "geometry",
        "patch_type",
        "patch_id",
        "category",
        "source_idx",
        UPAREA_COL,
        COMID_COL,
        "comid",
        "basin_id",
    ]
    pos = gdf_pos.reindex(columns=[c for c in keep_cols if c in gdf_pos.columns])
    neg = gdf_neg.reindex(columns=[c for c in keep_cols if c in gdf_neg.columns])
    gdf_all = gpd.GeoDataFrame(
        pd.concat([pos, neg], ignore_index=True),
        geometry="geometry",
        crs=f"EPSG:{TARGET_EPSG}",
    )

    # Coalesce positives' COMID and negatives' comid into one column so every
    # patch carries a reach id for the train/val/test grouping downstream
    if "comid" not in gdf_all.columns:
        gdf_all["comid"] = np.nan
    if COMID_COL in gdf_all.columns:
        gdf_all["comid"] = gdf_all["comid"].fillna(gdf_all[COMID_COL])

    # Extract channels, rasterize labels, save
    patches.build_and_save(
        gdf_all,
        gdf_levees,
        OUTPUT_DIR,
        RASTER_PATHS,
        CHANNEL_KEYS,
        PATCH_SIZE_PX,
        PATCH_RES_M,
        LEVEE_BUFFER_M,
        TPI_RADII_PX,
        TARGET_EPSG,
        UPAREA_COL,
    )


if __name__ == "__main__":
    main()
