"""
Prepare a binary water-mask raster from an OSM water vector.

Rasterizes OSM water geometries onto the reference DSM grid (aligned extent,
resolution and CRS). The resulting 0/1 GeoTIFF is read as the 7th input
channel (water) by BOTH the patch generator and the inference pipeline via
patch_io.read_window with NEAREST resampling, so the mask stays 0/1.

Run once per region (PL, NL, CZ, ...). No rasterio / fiona: GDAL for rasters,
GeoPandas + pyogrio for the vector.

Author:   Jakub Zapletal
Date:     2026-06-18
Version:  0.2   (simplified from prepare_water_distance: binary mask only)
"""

import warnings

warnings.filterwarnings("ignore")

import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd

from osgeo import gdal, osr

gdal.UseExceptions()


# ============================================================
# CONFIG
# ============================================================

# OSM water vector (polygons preferred: natural=water, landuse=reservoir,
# water=river/riverbank; line waterways are buffered, see WATERWAY_LINE_BUFFER_M).
WATER_VECTOR_PATH = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\source\pl\osm_water_pl.gpkg"
)
WATER_LAYER = None  # layer name, or None to use the first layer

# Reference raster defines the output grid: extent, resolution, CRS, alignment.
# Use the SAME DSM the patch generator / inference reads.
REFERENCE_RASTER = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif"
)

OUTPUT_WATER_TIF = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\source\pl\water_mask_pl.tif"
)

WATERWAY_LINE_BUFFER_M = (
    5.0  # buffer line geometries (narrow rivers) by this; 0 ignores lines
)
ALL_TOUCHED = True  # burn every pixel a geometry touches (catches thin features)


# ============================================================
# HELPERS
# ============================================================


def reference_grid(ref_path):
    """Read grid geometry (geotransform, projection, size, EPSG) from a raster."""
    ds = gdal.Open(str(ref_path))
    if ds is None:
        raise RuntimeError(f"Cannot open reference raster: {ref_path}")
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    w, h = ds.RasterXSize, ds.RasterYSize

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    epsg = srs.GetAuthorityCode(None)
    ds = None

    return {
        "gt": gt,
        "proj": proj,
        "epsg": int(epsg) if epsg else None,
        "width": w,
        "height": h,
        "xres": gt[1],
        "yres": -gt[5],
    }


def load_water_in_grid_crs(vec_path, layer, target_epsg, line_buffer_m):
    """Load water vector, reproject to the grid CRS, buffer any line geometries."""
    gdf = gpd.read_file(vec_path, layer=layer) if layer else gpd.read_file(vec_path)
    if gdf.crs is None:
        raise RuntimeError("Water vector has no CRS; set one before running.")
    if target_epsg is not None and gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    is_line = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    if is_line.any() and line_buffer_m and line_buffer_m > 0:
        gdf.loc[is_line, "geometry"] = gdf.loc[is_line, "geometry"].buffer(
            line_buffer_m
        )

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) == 0:
        raise RuntimeError("No usable water geometries after loading/buffering.")
    return gdf


def rasterize_binary_to_tif(gdf, grid, out_path, all_touched):
    """Rasterize water geometries to a 0/1 GeoTIFF aligned to the reference grid."""
    tmp = (
        Path(tempfile.gettempdir())
        / f"water_tmp_{abs(hash(str(gdf.total_bounds)))}.gpkg"
    )
    gdf[["geometry"]].to_file(tmp, driver="GPKG")

    mem = gdal.GetDriverByName("MEM").Create(
        "", grid["width"], grid["height"], 1, gdal.GDT_Byte
    )
    mem.SetGeoTransform(grid["gt"])
    mem.SetProjection(grid["proj"])
    gdal.Rasterize(mem, str(tmp), burnValues=[1], allTouched=all_touched)
    arr = mem.GetRasterBand(1).ReadAsArray()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdal.GetDriverByName("GTiff").CreateCopy(
        str(out_path), mem, options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    mem = None

    try:
        tmp.unlink()
    except OSError:
        pass
    return arr


# ============================================================
# MAIN
# ============================================================


def main():
    grid = reference_grid(REFERENCE_RASTER)
    print(
        f"Reference grid: {grid['width']}x{grid['height']} px @ {grid['xres']:.1f} m, "
        f"EPSG:{grid['epsg']}"
    )

    gdf = load_water_in_grid_crs(
        WATER_VECTOR_PATH, WATER_LAYER, grid["epsg"], WATERWAY_LINE_BUFFER_M
    )
    print(f"Water features: {len(gdf)}")

    arr = rasterize_binary_to_tif(gdf, grid, OUTPUT_WATER_TIF, ALL_TOUCHED)
    print(f"Water pixels: {int(arr.sum()):,} / {arr.size:,} ({arr.mean() * 100:.2f}%)")
    print(f"Saved water mask: {OUTPUT_WATER_TIF}")


if __name__ == "__main__":
    main()
