"""
Shared raster patch I/O for the levee-detection pipeline.

Both the patch generator and the inference scripts (04 / 04b) read raster
windows and compute TPI through THESE functions, so that a patch is
constructed bit-identically at training and inference time. Centralising the
code removes the latent mismatch that existed when the generator (rasterio,
TPI mode 'nearest') and inference (GDAL, TPI mode 'reflect') each kept their
own copy.

No rasterio / fiona: GDAL only.

Author:   Jakub Zapletal
Date:     2026-06-18
Version:  0.1
"""

import numpy as np
from scipy.ndimage import uniform_filter

from osgeo import gdal, gdalconst
gdal.UseExceptions()


def read_window(ds, bbox, target_pixels, resample=gdalconst.GRA_Bilinear):
    """
    Read a square window from an open GDAL dataset, resampled to
    target_pixels x target_pixels.

    bbox: (xmin, ymin, xmax, ymax) in the dataset CRS.
    Out-of-bounds areas are zero-filled. Returns a float32 array.

    Use resample=gdalconst.GRA_NearestNeighbour for categorical rasters such
    as the binary water mask, so the values stay 0/1 instead of being blurred.
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

    # Fully outside the raster -> zeros
    if (col_off >= raster_xsize or row_off >= raster_ysize
            or col_off + col_size <= 0 or row_off + row_size <= 0):
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    read_col = max(0, col_off)
    read_row = max(0, row_off)
    read_col_size = min(col_size - (read_col - col_off), raster_xsize - read_col)
    read_row_size = min(row_size - (read_row - row_off), raster_ysize - read_row)

    if read_col_size <= 0 or read_row_size <= 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    band = ds.GetRasterBand(1)

    # Fully inside -> single resampled read
    if (read_col == col_off and read_row == row_off
            and read_col_size == col_size and read_row_size == row_size):
        return band.ReadAsArray(
            read_col, read_row, read_col_size, read_row_size,
            buf_xsize=target_pixels, buf_ysize=target_pixels,
            resample_alg=resample,
        ).astype(np.float32)

    # Partially out of bounds -> read what exists, place into a zero-padded output
    sub_target_w = int(round(target_pixels * read_col_size / col_size))
    sub_target_h = int(round(target_pixels * read_row_size / row_size))
    if sub_target_w == 0 or sub_target_h == 0:
        return np.zeros((target_pixels, target_pixels), dtype=np.float32)

    sub = band.ReadAsArray(
        read_col, read_row, read_col_size, read_row_size,
        buf_xsize=sub_target_w, buf_ysize=sub_target_h,
        resample_alg=resample,
    ).astype(np.float32)

    out = np.zeros((target_pixels, target_pixels), dtype=np.float32)
    out_col_off = max(0, min(int(round(target_pixels * (read_col - col_off) / col_size)),
                             target_pixels - 1))
    out_row_off = max(0, min(int(round(target_pixels * (read_row - row_off) / row_size)),
                             target_pixels - 1))
    # Clamp both destination slice and source so independent rounding can never
    # produce a shape mismatch for edge patches.
    h = min(sub_target_h, target_pixels - out_row_off)
    w = min(sub_target_w, target_pixels - out_col_off)
    out[out_row_off:out_row_off + h, out_col_off:out_col_off + w] = sub[:h, :w]
    return out


def compute_tpi(z, radius_px, mode="reflect"):
    """
    Topographic Position Index: elevation minus the mean of a (2*r+1) window.
    'mode' is fixed to 'reflect' across generation and inference for consistency.
    """
    size = 2 * radius_px + 1
    return (z - uniform_filter(z, size=size, mode=mode)).astype(np.float32)


def patch_geotransform(center_x, center_y, patch_size_m, patch_res_m):
    """
    GDAL geotransform for a patch centered on (center_x, center_y).
    Used to rasterize the per-patch label onto the same grid the channels use.
    """
    half = patch_size_m / 2.0
    return (center_x - half, patch_res_m, 0.0, center_y + half, 0.0, -patch_res_m)
