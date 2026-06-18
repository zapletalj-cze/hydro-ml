"""
Export Patch Footprints to GPKG
================================

Builds square patch-footprint polygons (2560 x 2560 m) from patch metadata and
writes them to a GeoPackage for inspection in QGIS. Each polygon carries the
train/val/test split as an attribute, so the spatial distribution of the
training data can be mapped and styled by split.

Primary input is metadata_with_split.csv (written by train_segformer.py), which
already contains the split assignment. If only the raw patches_metadata.csv is
available (no split column), the script still exports footprints but marks the
split as "unknown".

Edit the CONFIG block, then run:  python patches_to_gpkg.py

Author:   Jakub Zapletal
Date:     2026-06-16
Version:  0.1
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ============================================================
# CONFIG
# ============================================================

# Preferred: the split file written by training (has the 'split' column).
# Fall back to a region patches_metadata.csv if the split file is absent.
METADATA_CSV = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight\metadata_with_split.csv"
)

OUTPUT_GPKG = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patch_footprints.gpkg"
)

CRS_TARGET = 2180  # EPSG:2180 (PL-1992)
PATCH_SIZE_M = 256 * 10  # 2560 m on a side (256 px at 10 m/px)

# Which splits to export. Use ["train"] for training patches only, or
# ["train", "val", "test"] to export all and distinguish them by attribute.
SPLITS_TO_EXPORT = ["train", "val", "test"]

# Also write a separate GPKG containing only the training footprints.
WRITE_TRAIN_ONLY = True


# ============================================================
# HELPERS
# ============================================================


def build_footprints(df, patch_size_m, crs_target):
    """
    Build square footprint polygons from center_x / center_y.

    A patch is centered on (center_x, center_y) and spans patch_size_m on a
    side, so its footprint is the square [cx - h, cy - h, cx + h, cy + h] with
    h = patch_size_m / 2.
    """
    if "center_x" not in df.columns or "center_y" not in df.columns:
        raise RuntimeError(
            "Metadata has no center_x / center_y columns; cannot build footprints."
        )

    half = patch_size_m / 2.0
    geometries = [
        box(cx - half, cy - half, cx + half, cy + half)
        for cx, cy in zip(df["center_x"].to_numpy(), df["center_y"].to_numpy())
    ]

    # Keep a tidy set of attributes if present
    keep_cols = [
        c
        for c in [
            "patch_id",
            "split",
            "region",
            "patch_type",
            "category",
            "n_label_px",
            "uparea",
            "comid",
            "source_idx_global",
        ]
        if c in df.columns
    ]

    gdf = gpd.GeoDataFrame(
        df[keep_cols].copy(), geometry=geometries, crs=f"EPSG:{crs_target}"
    )
    return gdf


# ============================================================
# MAIN
# ============================================================


def main():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} patch rows from {METADATA_CSV.name}")

    # Handle missing split column gracefully
    if "split" not in df.columns:
        print("  No 'split' column found; marking all patches as 'unknown'.")
        df["split"] = "unknown"

    # Report split / region composition
    print("\nPatch composition:")
    if "region" in df.columns:
        comp = df.groupby(["split", "region"]).size().rename("n").reset_index()
        for _, r in comp.iterrows():
            print(f"  {r['split']:>8} | {r['region']:>4} : {r['n']}")
    else:
        for split, n in df["split"].value_counts().items():
            print(f"  {split:>8} : {n}")

    # Filter to requested splits
    available = set(df["split"].unique())
    requested = [s for s in SPLITS_TO_EXPORT if s in available]
    if not requested:
        print(
            f"\nNone of SPLITS_TO_EXPORT={SPLITS_TO_EXPORT} present "
            f"(available: {sorted(available)}). Exporting all."
        )
        df_sel = df
    else:
        df_sel = df[df["split"].isin(requested)].copy()

    # Build footprints and export
    gdf = build_footprints(df_sel, PATCH_SIZE_M, CRS_TARGET)
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"\nWrote {len(gdf)} footprints to {OUTPUT_GPKG}")
    print(
        f"  total mapped area: {gdf.geometry.area.sum() / 1e6:.0f} km^2 "
        f"(note: overlapping patches counted multiple times)"
    )

    # Optional training-only export
    if WRITE_TRAIN_ONLY and "train" in available:
        train_gdf = build_footprints(
            df[df["split"] == "train"].copy(), PATCH_SIZE_M, CRS_TARGET
        )
        train_path = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_train_only.gpkg")
        train_gdf.to_file(train_path, driver="GPKG")
        print(f"Wrote {len(train_gdf)} training footprints to {train_path}")

    print(
        "\nDone. In QGIS, style by the 'split' attribute to distinguish "
        "train / val / test."
    )


if __name__ == "__main__":
    main()
