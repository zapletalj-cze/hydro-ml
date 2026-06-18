"""
Fix Metadata: Add source_idx to existing patches_metadata.csv
==============================================================

Adds the missing source_idx column to patches_metadata.csv files
without regenerating any patches. Uses the GPKG files saved during
patch generation (segments_categorized.gpkg, patch_centers_positive.gpkg,
patch_centers_negative.gpkg).

Why this is needed
------------------
The original 01_generate_patches_*.py did not write source_idx into
patches_metadata.csv. As a consequence, the training notebook had to
fall back to splitting by comid, which leaks data — multiple levees
sharing one MERIT reach (same comid) ended up in different splits.

This fix recovers source_idx by joining:
  patch_id -> segment_id (from patch_centers_positive.gpkg)
  segment_id -> source_idx (from segments_categorized.gpkg)

Negative patches inherit source_idx from their parent positive patch
via source_positive_id.

Author:   Jakub Zapletal
Date:     2026-04-13
Version:  0.1
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import geopandas as gpd

# ------------------------------------------------------------
# Inputs — adjust per region
# ------------------------------------------------------------
REGIONS = {
    "PL": Path(
        r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_PL"
    ),
    "NL": Path(
        r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_NL"
    ),
}


def fix_metadata(region_dir):
    """
    Add source_idx column to patches_metadata.csv in the given region directory.
    Writes the fixed file as patches_metadata_fixed.csv next to the original.
    """
    metadata_path = region_dir / "patches_metadata.csv"
    segments_path = region_dir / "segments_categorized.gpkg"
    pos_centers_path = region_dir / "patch_centers_positive.gpkg"
    neg_centers_path = region_dir / "patch_centers_negative.gpkg"

    metadata = pd.read_csv(metadata_path)
    segments = gpd.read_file(segments_path)
    pos_centers = gpd.read_file(pos_centers_path)
    neg_centers = gpd.read_file(neg_centers_path)

    # Positive patches: patch_id -> segment_id -> source_idx
    pos_lookup = pos_centers[["patch_id", "segment_id"]].merge(
        segments[["segment_id", "source_idx"]],
        on="segment_id",
        how="left",
    )[["patch_id", "source_idx"]]

    # Negative patches: source_positive_id -> source_idx (via positive lookup)
    neg_lookup = neg_centers[["patch_id", "source_positive_id"]].merge(
        pos_lookup.rename(columns={"patch_id": "source_positive_id"}),
        on="source_positive_id",
        how="left",
    )[["patch_id", "source_idx"]]

    # Combine
    all_lookup = pd.concat([pos_lookup, neg_lookup], ignore_index=True)

    # Merge into metadata
    fixed = metadata.merge(all_lookup, on="patch_id", how="left")

    # Sanity: how many patches got source_idx
    n_total = len(fixed)
    n_with_source = fixed["source_idx"].notna().sum()
    n_missing = n_total - n_with_source

    print(f"Region: {region_dir.name}")
    print(f"  Total patches:        {n_total}")
    print(f"  With source_idx:      {n_with_source}")
    print(f"  Missing source_idx:   {n_missing}")

    if n_missing > 0:
        # Show a sample of the failures
        missing_sample = fixed[fixed["source_idx"].isna()]["patch_id"].head(5).tolist()
        print(f"  Sample missing IDs:   {missing_sample}")

    output_path = region_dir / "patches_metadata_fixed.csv"
    fixed.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print()


if __name__ == "__main__":
    for region, region_dir in REGIONS.items():
        fix_metadata(region_dir)
