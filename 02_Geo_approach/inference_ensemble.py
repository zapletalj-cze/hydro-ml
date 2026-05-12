"""
Ensemble inference: Average probability rasters from N models and postprocess
to vector levee detections.

This script consumes the intermediate GeoTIFF probability rasters produced by
inference_aoi.py (one per trained model) and produces a single ensembled GPKG
of detected levee centerlines.

Pipeline:
    1. Load N probability rasters (must share shape, geotransform, CRS).
    2. Compute pixel-wise average (simple, unweighted).
    3. Rebuild river corridor mask from AOI + MERIT (same as single-model run).
    4. Threshold + morphological cleanup + skeletonize + vectorize.
    5. Save as GPKG (EPSG:2180).

Shared helpers (corridor building, postprocessing, etc.) are imported from
inference_aoi.py — both scripts must live in the same directory.

Author:   Jakub Zapletal
Date:     2026-05-08
Version:  0.1
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np

from osgeo import gdal
gdal.UseExceptions()

# Reuse helpers from inference_aoi.py (must be in same folder)
from interference import (
    load_aoi_polygon,
    build_river_corridor,
    rasterize_corridor_mask,
    postprocess_to_vector,
    CRS_TARGET,
    PROB_THRESHOLD,
    MIN_COMPONENT_PX,
    CLOSING_RADIUS_PX,
    MIN_LINE_LENGTH_M,
)


# ============================================================
# CONFIG
# ============================================================

# Probability rasters produced by inference_aoi.py (one per model).
# All three must come from the same AOI + MERIT config so they share
# raster shape, geotransform and CRS.
PROB_TIF_PATHS = [
    Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\inference_output\detected_levees_resnet_unet_prob.tif"),
    Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\inference_output\detected_levees_segformer_prob.tif"),
    Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\inference_output\detected_levees_deeplabv3plus_prob.tif"),
]

# AOI and MERIT — needed to recompute the corridor mask
AOI_PATH         = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\aoi_inference\aoi_polygon.gpkg")
MERIT_PATH       = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\source\merit\merit_hydro_pl.gpkg")

# Output
OUTPUT_GPKG          = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\inference_output\detected_levees_ensemble.gpkg")
OUTPUT_ENSEMBLE_TIF  = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_prob.tif")

# Geotransform comparison tolerance (in meters) for compatibility check.
# Should be much smaller than PATCH_RES_M (10 m).
GT_TOLERANCE_M       = 0.01


# ============================================================
# RASTER LOADING + COMPATIBILITY
# ============================================================

def load_prob_raster(path):
    """
    Load a probability raster.
    Returns (array, geotransform, projection_wkt).
    """
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open {path}")
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None
    return arr, gt, proj


def check_compatible(rasters):
    """
    Verify that all rasters have identical shape and geotransform (within tol).
    Raises RuntimeError on mismatch.
    """
    ref_shape = rasters[0][0].shape
    ref_gt    = rasters[0][1]

    for i, (arr, gt, _) in enumerate(rasters[1:], start=1):
        if arr.shape != ref_shape:
            raise RuntimeError(
                f"Raster {i} shape {arr.shape} differs from reference {ref_shape}"
            )
        for j, (a, b) in enumerate(zip(gt, ref_gt)):
            if abs(a - b) > GT_TOLERANCE_M:
                raise RuntimeError(
                    f"Raster {i} geotransform[{j}] = {a} differs from reference {b}"
                )


def save_geotiff(arr, geotransform, output_path):
    """Save float32 array as compressed GeoTIFF, EPSG:2180."""
    height_px, width_px = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(output_path), width_px, height_px, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"],
    )
    ds.SetGeoTransform(geotransform)
    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(CRS_TARGET)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(-1.0)
    ds.FlushCache()
    ds = None


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Number of models: {len(PROB_TIF_PATHS)}")
    for p in PROB_TIF_PATHS:
        print(f"  {p}")
    print(f"Output GPKG:    {OUTPUT_GPKG}")
    print(f"Output ens TIF: {OUTPUT_ENSEMBLE_TIF}")
    print()

    # 1. Load probability rasters
    print("Loading probability rasters...")
    rasters = [load_prob_raster(p) for p in PROB_TIF_PATHS]
    for i, (arr, _, _) in enumerate(rasters):
        print(f"  Model {i+1}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")

    # 2. Check compatibility
    print("Checking raster compatibility...")
    check_compatible(rasters)
    ref_gt = rasters[0][1]
    print("  All rasters have matching shape and geotransform")

    # 3. Average
    print("Computing pixel-wise average...")
    stack = np.stack([arr for arr, _, _ in rasters], axis=0)  # (N, H, W)
    prob_avg = stack.mean(axis=0)
    print(f"  Ensemble probability: min={prob_avg.min():.3f}, max={prob_avg.max():.3f}, mean={prob_avg.mean():.3f}")

    # 4. Save ensemble probability raster (useful for inspection / thresholding tuning)
    print(f"Saving ensemble probability raster to {OUTPUT_ENSEMBLE_TIF}...")
    OUTPUT_ENSEMBLE_TIF.parent.mkdir(parents=True, exist_ok=True)
    save_geotiff(prob_avg, ref_gt, OUTPUT_ENSEMBLE_TIF)

    # 5. Rebuild river corridor mask
    # Override path constants in inference_aoi module so its helpers see our AOI/MERIT
    import inference_aoi as ia
    ia.AOI_PATH   = AOI_PATH
    ia.MERIT_PATH = MERIT_PATH

    print("Loading AOI + MERIT corridor...")
    aoi_geom = load_aoi_polygon()
    corridor_geom, merit_in_aoi = build_river_corridor(aoi_geom)
    print(f"  AOI area:       {aoi_geom.area / 1e6:.1f} km²")
    print(f"  Corridor area:  {corridor_geom.area / 1e6:.1f} km² ({corridor_geom.area / aoi_geom.area * 100:.1f}%)")
    print(f"  MERIT reaches:  {len(merit_in_aoi)}")

    print("Rasterizing corridor mask...")
    corridor_mask = rasterize_corridor_mask(corridor_geom, ref_gt, prob_avg.shape)

    # 6. Postprocess (threshold -> cleanup -> skeleton -> vectorize)
    print("Postprocessing (threshold -> skeleton -> vectorize)...")
    detected = postprocess_to_vector(prob_avg, corridor_mask, ref_gt)
    print(f"  Detected lines: {len(detected)}")
    if len(detected) > 0:
        print(f"  Total length:   {detected['length_m'].sum() / 1000:.1f} km")
        print(f"  Mean length:    {detected['length_m'].mean():.0f} m")
        print(f"  Max length:     {detected['length_m'].max():.0f} m")

    # 7. Save
    print(f"Saving GPKG to {OUTPUT_GPKG}...")
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    detected.to_file(OUTPUT_GPKG, driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()
