"""
Shared GIS helpers (vector I/O, raster windows, rasterization).

Author: Jakub Zapletal
Date:   2026-04-02
"""

import numpy as np
import geopandas as gpd

# GDAL is only needed by the Raster class; the Vector part runs without it.
try:
    from osgeo import gdal, ogr, osr, gdalconst

    gdal.UseExceptions()
except ImportError:
    gdal = ogr = osr = gdalconst = None


class Vector:
    @staticmethod
    def load_vector(path, bbox=None, target_epsg=None):
        """
        Load a vector file into a GeoDataFrame.
        :param path: path to vector file
        :param bbox: optional (xmin, ymin, xmax, ymax) filter, in the file CRS
        :param target_epsg: optional EPSG code to reproject to
        :return: GeoDataFrame
        """
        if bbox is not None:
            gdf = gpd.read_file(path, bbox=tuple(bbox))
        else:
            gdf = gpd.read_file(path)
        if target_epsg is not None:
            gdf = Vector.reproject_vector(gdf, target_epsg)
        return gdf

    @staticmethod
    def save_vector(gdf, path, driver="GPKG"):
        """Save a GeoDataFrame to file."""
        gdf.to_file(path, driver=driver)

    @staticmethod
    def reproject_vector(gdf, epsg_out):
        """Reproject to epsg_out; assigns the CRS if missing."""
        if gdf.crs is None:
            return gdf.set_crs(epsg=epsg_out)
        if gdf.crs.to_epsg() != epsg_out:
            return gdf.to_crs(epsg=epsg_out)
        return gdf

    @staticmethod
    def drop_empty_geometries(gdf):
        """Remove rows with missing or empty geometry."""
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    @staticmethod
    def clip_vector(gdf, clipping_area):
        """Clip a GeoDataFrame by a polygon GeoDataFrame."""
        return gpd.clip(gdf, clipping_area)

    @staticmethod
    def explode_to_lines(gdf):
        """Keep line geometries only, explode multiparts, drop zero-length rows."""
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        return gdf[gdf.geometry.length > 0].reset_index(drop=True)


class Raster:
    @staticmethod
    def open_raster(path):
        """Open a raster file, raise if it cannot be read."""
        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(f"Could not open raster: {path}")
        return ds

    @staticmethod
    def point_in_raster(ds, x, y):
        """True when (x, y) lies inside the raster extent."""
        gt = ds.GetGeoTransform()
        minx, maxy = gt[0], gt[3]
        maxx = minx + ds.RasterXSize * gt[1]
        miny = maxy + ds.RasterYSize * gt[5]
        return (minx <= x <= maxx) and (miny <= y <= maxy)

    @staticmethod
    def _resample_alg(name):
        algs = {
            "bilinear": gdalconst.GRA_Bilinear,
            "nearest": gdalconst.GRA_NearestNeighbour,
            "cubic": gdalconst.GRA_Cubic,
        }
        return algs[name]

    @staticmethod
    def read_window(ds, bbox, target_pixels, resample="bilinear"):
        """
        Read a square window from an open raster dataset, resampled to
        target_pixels x target_pixels. Out-of-bounds areas are zero-filled.
        :param ds: open GDAL dataset
        :param bbox: (xmin, ymin, xmax, ymax) in the dataset CRS
        :param target_pixels: output size in pixels
        :param resample: "bilinear" | "nearest" | "cubic" (nearest for categorical)
        :return: float32 array (target_pixels, target_pixels)
        """
        resample_alg = Raster._resample_alg(resample)
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

        # Fully outside the raster
        if (
            col_off >= raster_xsize
            or row_off >= raster_ysize
            or col_off + col_size <= 0
            or row_off + row_size <= 0
        ):
            return np.zeros((target_pixels, target_pixels), dtype=np.float32)

        read_col = max(0, col_off)
        read_row = max(0, row_off)
        read_col_size = min(col_size - (read_col - col_off), raster_xsize - read_col)
        read_row_size = min(row_size - (read_row - row_off), raster_ysize - read_row)

        if read_col_size <= 0 or read_row_size <= 0:
            return np.zeros((target_pixels, target_pixels), dtype=np.float32)

        band = ds.GetRasterBand(1)

        # Fully inside: single resampled read
        if (
            read_col == col_off
            and read_row == row_off
            and read_col_size == col_size
            and read_row_size == row_size
        ):
            return band.ReadAsArray(
                read_col,
                read_row,
                read_col_size,
                read_row_size,
                buf_xsize=target_pixels,
                buf_ysize=target_pixels,
                resample_alg=resample_alg,
            ).astype(np.float32)

        # Partially outside: read the valid part into a zero-padded output
        sub_target_w = int(round(target_pixels * read_col_size / col_size))
        sub_target_h = int(round(target_pixels * read_row_size / row_size))
        if sub_target_w == 0 or sub_target_h == 0:
            return np.zeros((target_pixels, target_pixels), dtype=np.float32)

        sub = band.ReadAsArray(
            read_col,
            read_row,
            read_col_size,
            read_row_size,
            buf_xsize=sub_target_w,
            buf_ysize=sub_target_h,
            resample_alg=resample_alg,
        ).astype(np.float32)

        out = np.zeros((target_pixels, target_pixels), dtype=np.float32)
        out_col_off = max(
            0,
            min(
                int(round(target_pixels * (read_col - col_off) / col_size)),
                target_pixels - 1,
            ),
        )
        out_row_off = max(
            0,
            min(
                int(round(target_pixels * (read_row - row_off) / row_size)),
                target_pixels - 1,
            ),
        )
        # Clamp both slices so rounding cannot produce a shape mismatch
        h = min(sub_target_h, target_pixels - out_row_off)
        w = min(sub_target_w, target_pixels - out_col_off)
        out[out_row_off : out_row_off + h, out_col_off : out_col_off + w] = sub[:h, :w]
        return out

    @staticmethod
    def compute_tpi(z, radius_px, mode="reflect"):
        """Topographic Position Index: z minus the mean of a (2r+1) window."""
        from scipy.ndimage import uniform_filter

        size = 2 * radius_px + 1
        return (z - uniform_filter(z, size=size, mode=mode)).astype(np.float32)

    @staticmethod
    def patch_geotransform(center_x, center_y, patch_size_m, patch_res_m):
        """Geotransform for a square patch centered on (center_x, center_y)."""
        half = patch_size_m / 2.0
        return (center_x - half, patch_res_m, 0.0, center_y + half, 0.0, -patch_res_m)

    @staticmethod
    def rasterize_geometries(
        geoms, geotransform, shape, epsg, burn_value=1, all_touched=False
    ):
        """
        Rasterize shapely geometries onto a target grid.
        :param geoms: iterable of shapely geometries
        :param geotransform: GDAL geotransform of the target grid
        :param shape: (rows, cols) or a single int for a square grid
        :param epsg: EPSG code of the grid
        :param burn_value: value burned into covered cells
        :param all_touched: burn all touched cells, not only cell centers
        :return: uint8 array (rows, cols)
        """
        if isinstance(shape, int):
            rows, cols = shape, shape
        else:
            rows, cols = shape

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)

        target = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Byte)
        target.SetGeoTransform(geotransform)
        target.SetProjection(srs.ExportToWkt())

        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if geoms:
            drv = ogr.GetDriverByName("Memory")
            vds = drv.CreateDataSource("mem")
            layer = vds.CreateLayer("geoms", srs, ogr.wkbUnknown)
            defn = layer.GetLayerDefn()
            for g in geoms:
                feat = ogr.Feature(defn)
                feat.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
                layer.CreateFeature(feat)
                feat = None
            options = ["ALL_TOUCHED=TRUE"] if all_touched else []
            gdal.RasterizeLayer(
                target, [1], layer, burn_values=[burn_value], options=options
            )
            vds = None

        arr = target.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        target = None
        return arr

    @staticmethod
    def save_array(
        path,
        array,
        geotransform,
        epsg,
        nodata=None,
        data_type=None,
        options=("COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"),
    ):
        """
        Save a 2D array as a GeoTIFF.
        :param path: output file path
        :param array: 2D numpy array
        :param geotransform: GDAL geotransform
        :param epsg: EPSG code of the grid
        :param nodata: optional nodata value
        :param data_type: GDAL data type, float32 when not set
        """
        if data_type is None:
            data_type = gdal.GDT_Float32
        rows, cols = array.shape

        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(path), cols, rows, 1, data_type, options=list(options))
        ds.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(array)
        if nodata is not None:
            ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds.FlushCache()
        ds = None
