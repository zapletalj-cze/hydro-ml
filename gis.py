# Standard library imports
from osgeo import gdal, osr, ogr
import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import os
from os.path import join, exists
import shutil
import shapely
from shapely.geometry import Polygon, box, LineString, Point
from shapely.wkt import loads
from shapely.ops import nearest_points, unary_union, linemerge
from math import floor, ceil, sqrt, acos, degrees
import pyflwdir


# glob_params = Parameters()
gdal.UseExceptions()
gdal.SetCacheMax(4 * pow(1024, 3))


class Files:
    @staticmethod
    def set_workspace(path, folders):
        if not exists(path):
            os.makedirs(path)
        for folder in folders:
            folder_path = join(path, folder)
            if not exists(folder_path):
                os.makedirs(folder_path)

    @staticmethod
    def get_value_from_dict(item, key, default=None):
        return item.get(key, default)

    def flush_temp(self):
        try:
            temp_path = os.path.join(self.model_root, "temp")
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                print(f"Temp folder '{temp_path}' has been flushed.")
            else:
                print(f"Temp folder '{temp_path}' does not exist.")
        except Exception as e:
            print(f"Failed to flush temp folder: {e}")

    def flush_temp_subdir(self, subdir_path):
        try:
            temp_path = os.path.join(self.model_root, "temp")
            target_path = os.path.join(temp_path, subdir_path)

            if os.path.exists(target_path) and os.path.isdir(target_path):
                shutil.rmtree(target_path)
                print(f"Subdirectory '{subdir_path}' in temp has been flushed.")
            else:
                print(f"Subdirectory '{subdir_path}' does not exist in temp.")
        except Exception as e:
            print(f"Failed to flush subdirectory '{subdir_path}' in temp: {e}")

    @staticmethod
    def get_root_path():
        current_path = os.path.dirname(os.path.abspath(__file__))
        while os.path.basename(current_path) != "IF-HydroSim":
            current_path = os.path.dirname(current_path)
        return current_path

    @staticmethod
    def split_list(input_list, num_sublists):
        """
        Split a list into a defined number of sublists.
        """
        avg = len(input_list) / float(num_sublists)
        out_list = []
        index_l = 0

        while index_l < len(input_list):
            out_list.append(input_list[int(index_l) : int(index_l + avg)])
            index_l += avg

        return out_list

    @staticmethod
    def split_dict(input_dict, num_sublists):
        """
        Split a dictionary into a defined number of sublists of (key, value) pairs.
        """
        items = list(input_dict.items())
        avg = len(items) / float(num_sublists)
        sublists = []
        index_l = 0

        while index_l < len(items):
            sublists.append(items[int(index_l) : int(index_l + avg)])
            index_l += avg

        list_dict = []
        for sublist in sublists:
            list_dict.append(dict(sublist))
        return list_dict


class Raster:
    def __init__(self):
        self.extent_epsg = None
        self.snap_raster = None
        self.snap = False

    @staticmethod
    def to_array(path, band_number=1):
        """
        Read raster as TIF to numpy array
        :param path: path to raster file
        :return: numpy array
        """
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {path}")
        array = ds.GetRasterBand(band_number).ReadAsArray()
        return array

    @staticmethod
    def to_array_mem(ds_raster, band_number=1):
        """
        Read raster as MEM raster to numpy array
        :param ds_raster: MEM raster dataset
        :return: numpy array
        """
        array = ds_raster.GetRasterBand(band_number).ReadAsArray()
        return array

    @staticmethod
    def save_raster(output_path, ds):
        """
        Saves GDAL Dataset to file as TIF
        ::param output_path: path to output raster file
        :param ds: existing GDAL Dataset to save
        """
        driver = gdal.GetDriverByName("GTiff")
        new_ds = driver.Create(
            output_path,
            ds.RasterXSize,
            ds.RasterYSize,
            ds.RasterCount,
            ds.GetRasterBand(1).DataType,
            options=["COMPRESS=ZSTD", "PREDICTOR=2", "TILED=YES"],
        )
        new_ds.SetGeoTransform(ds.GetGeoTransform())
        new_ds.SetProjection(ds.GetProjection())
        for i in range(1, ds.RasterCount + 1):
            band = new_ds.GetRasterBand(i)
            band.WriteArray(ds.GetRasterBand(i).ReadAsArray())
        new_ds.FlushCache()
        new_ds = None

    def clip_by_vector(
        in_raster: str,
        out_raster: str,
        vector_file: str,
        epsg_out: int,
        cell_size_out,
        data_type,
        CPU_AVAILABLE=16,
        no_data=-9999,
        bounds=None,
        interpolation="nearest",
    ):
        """
        Clips raster by vector file and reprojects to new projection and resolution
        :param in_raster: input raster file
        :param out_raster: output raster file
        :param vector_file: vector file for clipping
        :param epsg_out: EPSG code of output projection
        :param cell_size_out: output cell size
        :param data_type: output data type in GDAL format
        """
        ds_in = gdal.Open(in_raster)
        if ds_in is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        srs_in = osr.SpatialReference()
        srs_in.ImportFromWkt(ds_in.GetProjection())
        srs_out = osr.SpatialReference()
        srs_out.ImportFromEPSG(epsg_out)
        options = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=vector_file,
            xRes=cell_size_out,
            yRes=cell_size_out,
            dstSRS=srs_out,
            srcSRS=srs_in,
            outputBounds=bounds,
            resampleAlg=interpolation,
            srcNodata=ds_in.GetRasterBand(1).GetNoDataValue(),
            dstNodata=no_data,
            outputType=data_type,
            creationOptions=[
                "COMPRESS=ZSTD",
                # "PREDICTOR=2",
                "TILED=YES",
                "BIGTIFF=YES",
            ],
            warpOptions=[
                f"NUM_THREADS={CPU_AVAILABLE}",
            ],
        )
        gdal.Warp(out_raster, ds_in, options=options)
        ds_in = None

    def clip_by_vector_mem(
        in_raster: str,
        vector_file,
        epsg_out: int,
        cell_size_out,
        data_type,
        CPU_AVAILABLE=1,
        no_data=-9999,
        bounds=None,
    ):
        """
        Clips raster by vector file and reprojects to new projection and resolution
        :param in_raster: input raster file
        :param vector_file: vector file path (str) or OGR datasource object
        :param epsg_out: EPSG code of output projection
        :param cell_size_out: output cell size
        :param data_type: output data type in GDAL format
        """
        ds_in = gdal.Open(in_raster)
        if ds_in is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        srs_in = osr.SpatialReference()
        srs_in.ImportFromWkt(ds_in.GetProjection())
        srs_out = osr.SpatialReference()
        srs_out.ImportFromEPSG(epsg_out)

        # Handle both file paths and OGR datasource objects
        if isinstance(vector_file, str):
            warp_options = gdal.WarpOptions(
                format="MEM",
                cutlineDSName=vector_file,
                xRes=cell_size_out,
                yRes=cell_size_out,
                dstSRS=srs_out,
                srcSRS=srs_in,
                outputBounds=bounds,
                srcNodata=ds_in.GetRasterBand(1).GetNoDataValue(),
                dstNodata=no_data,
                outputType=data_type,
                multithread=True,
            )
        else:
            warp_options = gdal.WarpOptions(
                format="MEM",
                cutlineLayer=vector_file,
                xRes=cell_size_out,
                yRes=cell_size_out,
                dstSRS=srs_out,
                srcSRS=srs_in,
                outputBounds=bounds,
                srcNodata=ds_in.GetRasterBand(1).GetNoDataValue(),
                dstNodata=no_data,
                outputType=data_type,
                multithread=True,
            )
        mem_ds = gdal.Warp("", ds_in, options=warp_options)
        ds_in = None
        return mem_ds

    @staticmethod
    def from_array_mem(
        array: np.array,
        output_raster: str = None,
        source_raster: str = None,
        geotransform: list = None,
        size_xy: list = None,
        epsg_out: int = None,
        data_type=None,
        nodata=None,
        raster_driver="GTiff",
    ):
        """
        Converts array to raster file or in memory raster
        :param array: numpy array
        :param output_raster: path to output raster file
        :param source_raster: path to source raster file for configuration
        :param geotransform: geotransform of output raster
        :param epsg_out: projection of output raster as EPSG code`
        :param data_type: data type of output raster, use gdal_dt from helpers.py
        :param no_data: no data value of output raster
        :param raster_driver: driver for output raster, supported GTiff and MEM
        """
        # Set projection for output raster
        projection = osr.SpatialReference()
        if epsg_out is None:
            try:
                epsg_out = Raster().extent_epsg
            except Exception:
                epsg_out = Raster.get_raster_info(source_raster, ["projection_epsg"])[
                    "projection_epsg"
                ]
        projection.ImportFromEPSG(epsg_out)
        projection = projection.ExportToWkt()
        # Get necessary parameters from source raster
        info_required = []
        if geotransform is None:
            info_required.append("geotransform")
        if data_type is None:
            info_required.append("data_type")
        if nodata is None:
            info_required.append("nodata")
        if size_xy is None:
            info_required.append("size")
        if len(info_required) > 0:
            if source_raster is None:
                raise ValueError(
                    "Path to output raster file must be provided if configuration is not set."
                )
        info = Raster.get_raster_info(source_raster, info_required)
        for key, value in info.items():
            if key == "geotransform":
                geotransform = value
            elif key == "data_type":
                data_type = value
            elif key == "nodata":
                nodata = value if value is not None else -9999
            elif key == "size":
                size_xy = value
        if geotransform is None:
            geotransform = info["geotransform"]
        if data_type is None:
            data_type = info["data_type"]
        if nodata is None:
            nodata = info["nodata"]
        if size_xy is None:
            size_xy = info["size"]
        # Create output raster
        driver = gdal.GetDriverByName(raster_driver)
        if raster_driver == "MEM":
            ds = driver.Create("", size_xy[0], size_xy[1], 1, data_type)
        else:
            ds = driver.Create(
                output_raster,
                size_xy[0],
                size_xy[1],
                1,
                data_type,
                options=["COMPRESS=ZSTD", "PREDICTOR=2", "TILED=YES"],
            )
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        ds.GetRasterBand(1).WriteArray(array)
        ds.GetRasterBand(1).SetNoDataValue(nodata)
        if raster_driver == "MEM":
            return ds
        ds.FlushCache()
        ds = None

    def create_snap_raster(self):
        """
        Create a snap raster for the extent, 2x2 pixels
        :return: path to snap raster
        """
        path = join(self.model_root, "config", "grid")
        Files.set_workspace(join(self.model_root, "config"), ["grid"])
        snap_raster = join(path, "snap_raster.tif")
        if not exists(snap_raster):
            xmin = self.extent[0]
            ymin = self.extent[2]
            cell_size = self.default_resolution
            data = np.ones((2, 2), dtype=np.uint8)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(snap_raster, 2, 2, 1, gdal.GDT_Byte)
            ds.SetGeoTransform(
                (xmin, cell_size, 0, ymin + 2 * cell_size, 0, -cell_size)
            )
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(self.extent_epsg)
            ds.SetProjection(srs.ExportToWkt())
            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            ds = None
        self.snap_raster = snap_raster

    @staticmethod
    def get_raster_nodata(in_raster: str, band_number: int = 1):
        """
        Get raster nodata value
        :param in_raster: input raster file
        :param band_number: band number to get the nodata value from
        :return: nodata value
        """
        ds = gdal.Open(in_raster)
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        band = ds.GetRasterBand(band_number)
        no_data = band.GetNoDataValue()
        return no_data

    @staticmethod
    def raster_to_vect(
        in_raster,
        out_vector,
        layer_name,
        raster_val_name,
        epsg,
        filter_values: list = None,
        area_threshold=None,
    ):
        """
        Converts vecror to raster using Polygonize function, expected:
        - single band raster file
        - categorical or binary raster
        :in raster: input raster file
        :out_vector: output vector file
        :raster_val_name: name of the raster value field in output vector
        :layer_name: same as basename of vector file
        :epsg: EPSG code of output vector
        """
        # Set vector driver for final output
        fe = os.path.splitext(out_vector)[1]
        if fe == ".gpkg":
            drv = ogr.GetDriverByName("GPKG")
        elif fe == ".shp":
            if len(raster_val_name) > 8:
                raster_val_name = raster_val_name[:8]
                print(
                    f"Field name too long, truncating to 8 characters: {raster_val_name}"
                )
            drv = ogr.GetDriverByName("ESRI Shapefile")
        elif fe == ".geojson":
            drv = ogr.GetDriverByName("GeoJSON")
        else:
            raise ValueError(f"Unsupported output vector format: {fe}")

        # Set inmemory driver
        dst_layername = layer_name
        drv_mem = ogr.GetDriverByName("Memory")  # Still using Memory instead of MEM
        dst_ds = drv_mem.CreateDataSource("in_memory")

        # If input raster is INMEMORY, use datasource directly
        try:
            drv_raster = in_raster.GetDriver()
            drivername = drv_raster.ShortName
        except Exception:
            pass

        if drivername == "MEM":
            ds_raster = in_raster
        else:
            ds_raster = gdal.Open(in_raster)
        band = ds_raster.GetRasterBand(1)
        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(f"EPSG:{epsg}")

        dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)

        # Polygonize inmemory, way faster than direct writing
        fld = ogr.FieldDefn(raster_val_name, ogr.OFTInteger)
        dst_layer.CreateField(fld)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(raster_val_name)
        gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)
        if filter_values is not None:
            if len(filter_values) == 1:
                filter_values = 2 * filter_values  # TEMPORARY FIX?
        dst_layer.SetAttributeFilter(
            "{} IN {}".format(raster_val_name, tuple(filter_values))
        )

        # filter on area if set
        if area_threshold is not None:
            dst_layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
            # option A using transaction per row
            dst_layer.StartTransaction()
            for feature in dst_layer:
                geom = feature.GetGeometryRef()
                area = geom.GetArea()
                feature.SetField("area", area)
                dst_layer.SetFeature(feature)
            dst_layer.CommitTransaction()
            dst_layer.SetAttributeFilter("area > {}".format(area_threshold))
        gdal.VectorTranslate(out_vector, dst_ds)
        dst_ds = None
        ds_raster = None
        return True

    @staticmethod
    def raster_to_mem_vector(
        in_raster,
        raster_val_name,
        epsg,
        filter_values: list = None,
        area_threshold=None,
    ):
        """
        Converts vecror to raster using Polygonize function, expected:
        - single band raster file
        - categorical or binary raster
        :in raster: input raster file
        :raster_val_name: name of the raster value field in output vector
        :layer_name: same as basename of vector file
        :epsg: EPSG code of output vector
        return ogr memory vector
        """
        # Set vector driver for final output
        drv = ogr.GetDriverByName("MEM")
        # Set inmemory driver
        dst_ds = drv.CreateDataSource("in_memory")
        dst_layername = "MEM layer"
        try:
            drv_rast = in_raster.GetDriver()
            drivername = drv_rast.ShortName
        except Exception:
            pass

        if drivername == "MEM":
            ds = in_raster
        else:
            ds = gdal.Open(in_raster)
        band = ds.GetRasterBand(1)
        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(f"EPSG:{epsg}")
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)

        # Polygonize inmemory, way faster than direct writing
        fld = ogr.FieldDefn(raster_val_name, ogr.OFTInteger)
        dst_layer.CreateField(fld)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(raster_val_name)
        gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)
        if filter_values is not None:
            if len(filter_values) == 1:
                filter_values = 2 * filter_values  # TEMPORARY FIX?
        dst_layer.SetAttributeFilter(
            "{} IN {}".format(raster_val_name, tuple(filter_values))
        )

        # filter on area if set
        if area_threshold is not None:
            dst_layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
            # option A using transaction per row
            dst_layer.StartTransaction()
            for feature in dst_layer:
                geom = feature.GetGeometryRef()
                area = geom.GetArea()
                feature.SetField("area", area)
                dst_layer.SetFeature(feature)
            dst_layer.CommitTransaction()
            dst_layer.SetAttributeFilter("area > {}".format(area_threshold))
        ds = None
        return dst_ds

    ### VRT manipulation
    @staticmethod
    def raster_list_to_vrt(in_rasters: list, output_vrt, no_data, resolution="highest"):
        """
        Creates a VRT file from a list of raster files
        :param in_rasters: list of raster files
        :param output_vrt: path to output VRT file
        :param no_data: nodata value to set in the VRT file and source rasters
        :param resolution: resolution of the output VRT file, see gdal.BuildVRTOptions
        """
        options = gdal.BuildVRTOptions(
            resolution=resolution, srcNodata=no_data, VRTNodata=no_data
        )
        gdal.BuildVRT(output_vrt, in_rasters, options=options)
        return output_vrt

    @staticmethod
    def raster_to_vrt(in_raster, out_vrt_dataset, no_data=-9999):
        """
        Converts a raster to a VRT (Virtual Raster) file and sets a NoData value if specified.
        :param in_raster: Input raster file path.
        :param out_vrt_dataset: Output VRT file path.
        :param no_data: NoData value to set in the VRT file.
        """
        # Open the input raster file
        src_ds = gdal.Open(in_raster, gdal.GA_Update)
        if src_ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")

        # Create a VRT dataset
        vrt_options = gdal.BuildVRTOptions(srcNodata=no_data)
        vrt_ds = gdal.BuildVRT(out_vrt_dataset, [src_ds], options=vrt_options)
        if vrt_ds is None:
            raise RuntimeError(f"Failed to create VRT file: {out_vrt_dataset}")

        vrt_ds.FlushCache()
        vrt_ds = None

    @staticmethod
    def update_nodata_vrt(vrt_file, no_data):
        """
        Updates the NoData value in an existing VRT file.
        Assumes the VRT file has only one raster band.
        :vrt_file: Path to the VRT file.
        :no_data: New NoData value to set.
        """
        vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
        if vrt_ds is None:
            raise FileNotFoundError(f"Could not open VRT file: {vrt_file}")
        if vrt_ds is None:
            raise FileNotFoundError(f"Could not open VRT file: {vrt_file}")

        band = vrt_ds.GetRasterBand(1)
        if band is None:
            raise RuntimeError("No raster band found in the VRT file.")
        # set Nodata value
        band.SetNoDataValue(no_data)
        vrt_ds = None
        vrt_ds.FlushCache()

    @staticmethod
    def setVRT_ColorInterp(raster_path, color_interp=gdal.GCI_GrayIndex):
        """
        Set color interpretation of a VRT file/raster
        :param raster_path: path to VRT/raster file
        :param color_interp: color interpretation
        """
        ds = gdal.Open(raster_path, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        band.SetColorInterpretation(color_interp)
        ds.FlushCache()
        ds = None

    @staticmethod
    def getVRT_ColorInterp(vrt_path):
        """
        Get color interpretation of a VRT file
        :param vrt_path: path to VRT file
        """
        try:
            with gdal.Open(vrt_path) as ds:
                band = ds.GetRasterBand(1)
                color_interp = band.GetColorInterpretation()
                return color_interp
        except Exception as e:
            print(f"Failed to get color interpretation: {e}")
            return None

    @staticmethod
    def get_extent_raster(in_raster: str):
        """
        Extracts extent of raster file
        :param in_raster: input raster file
        :return extent as [xmin, xmax, ymin, ymax]
        """
        ds = gdal.Open(in_raster)
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        geotransform = ds.GetGeoTransform()
        x_min = geotransform[0]
        x_max = geotransform[0] + ds.RasterXSize * geotransform[1]
        y_max = geotransform[3]
        y_min = geotransform[3] - ds.RasterYSize * abs(geotransform[5])
        extent = [x_min, x_max, y_min, y_max]
        return extent

    @staticmethod
    def get_extent_raster_mem(in_raster):
        """
        Extracts extent of raster file
        :param in_raster: input raster file
        :return extent as [xmin, xmax, ymin, ymax]
        """
        ds = in_raster
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        geotransform = ds.GetGeoTransform()
        x_min = geotransform[0]
        x_max = geotransform[0] + ds.RasterXSize * geotransform[1]
        y_max = geotransform[3]
        y_min = geotransform[3] - ds.RasterYSize * abs(geotransform[5])
        extent = [x_min, x_max, y_min, y_max]
        return extent

    @staticmethod
    def get_raster_info(in_raster: str, info: list):
        """
        Extracts raster info
        :param in_raster: input raster file
        :param info: raster properties to extract, supported [extent, nodata, cell_size, projection_epsg, geotransform, data_type, size]
        :return dict with info
        """
        ds = gdal.Open(in_raster)
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        export = {}
        if isinstance(info, str):
            info = [info]
        if "extent" in info:
            extent = Raster.get_extent_raster(in_raster)
            export["extent"] = extent
        if "nodata" in info:
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            export["nodata"] = nodata
        if "cell_size" in info:
            geotransform = ds.GetGeoTransform()
            cell_size = abs(geotransform[1])
            export["cell_size"] = cell_size
        if "size" in info:
            size = [ds.RasterXSize, ds.RasterYSize]
            export["size"] = size
        if "projection_epsg" in info:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjection())
            export["projection_epsg"] = int(srs.GetAttrValue("AUTHORITY", 1))
        if "geotransform" in info:
            geotransform = ds.GetGeoTransform()
            export["geotransform"] = geotransform
        if "data_type" in info:
            data_type = ds.GetRasterBand(1).DataType
            export["data_type"] = data_type
        return export

    def create_empty_raster(
        output_path: str,
        extent: list,
        cell_size: float,
        epsg_code: int,
        nodata_value: float,
        gdal_dtype: int = gdal.GDT_Int32,
    ) -> gdal.Dataset:
        min_x, max_x, min_y, max_y = extent
        width = int((max_x - min_x) // cell_size + 1)
        height = int((max_y - min_y) // cell_size + 1)
        gt = (min_x, cell_size, 0, max_y, 0, -cell_size)
        raster_driver = gdal.GetDriverByName("GTiff")
        options = [
            "TFW=YES",
            "BIGTIFF=YES",
            "TILED=YES",
            "NUM_THREADS=ALL_CPUS",
            "COMPRESS=ZSTD",
        ]
        dst_ds = raster_driver.Create(
            output_path,
            width,
            height,
            1,
            gdal_dtype,
            options=options,
        )
        if dst_ds is None:
            raise RuntimeError(f"Failed to create GDAL dataset for {output_path}")
        dst_ds.SetGeoTransform(gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        crs_wkt = srs.ExportToWkt()
        dst_ds.SetProjection(crs_wkt)
        band_dst = dst_ds.GetRasterBand(1)
        band_dst.SetNoDataValue(float(nodata_value))
        return dst_ds

    @staticmethod
    def get_raster_info_mem(in_raster, info: list):
        """
        Extracts raster info
        :param in_raster: dataset of in memory raster
        :param info: raster properties to extract, supported [extent, nodata, cell_size, projection_epsg, geotransform, data_type, size]
        :return dict with info
        """
        ds = in_raster
        if ds is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        export = {}
        if isinstance(info, str):
            info = [info]
        if "extent" in info:
            extent = Raster.get_extent_raster_mem(in_raster)
            export["extent"] = extent
        if "nodata" in info:
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            export["nodata"] = nodata
        if "cell_size" in info:
            geotransform = ds.GetGeoTransform()
            cell_size = abs(geotransform[1])
            export["cell_size"] = cell_size
        if "size" in info:
            size = [ds.RasterXSize, ds.RasterYSize]
            export["size"] = size
        if "projection_epsg" in info:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjection())
            export["projection_epsg"] = int(srs.GetAttrValue("AUTHORITY", 1))
        if "geotransform" in info:
            geotransform = ds.GetGeoTransform()
            export["geotransform"] = geotransform
        if "data_type" in info:
            data_type = ds.GetRasterBand(1).DataType
            export["data_type"] = data_type
        return export

    def reproject_raster(
        self,
        in_raster: str,
        out_raster: str,
        epsg_out: int,
        cell_size_out=0,
        data_type=gdal.GDT_Float32,
        interpolation="nearest",
        CPU_AVAILABLE=4,
        no_data=-9999,
        snap=False,
        extent=None,
        in_memory=False,
    ):
        """
        Reprojects raster to new projection and resolution
        :param in_raster: input raster file
        :param out_raster: output raster file
        :param epsg_out: EPSG code of output projection
        :param cell_size_out: output cell size
        :param data_type: output data type in GDAL format
        :param snap: snap extent to snap raster
        :param extent: extent of output raster
        """
        ds_in = gdal.Open(in_raster)
        if in_memory:
            out_raster = None  # in memory raster
        if ds_in is None:
            raise FileNotFoundError(f"Could not open input raster: {in_raster}")
        srs_in = osr.SpatialReference()
        srs_in.ImportFromWkt(ds_in.GetProjection())
        srs_out = osr.SpatialReference()
        srs_out.ImportFromEPSG(epsg_out)

        if cell_size_out == 0:
            cell_size_out = self.default_resolution
        if snap:
            if extent is None:
                extent = Raster.get_extent_raster(in_raster)
            extent_snapped = Vector.align_extent_to_snap(
                extent,
                snap_raster=self.snap_raster,
                cell_size=cell_size_out,
                optimize=True,
            )
            extent = extent_snapped
        bounds = [extent[0], extent[2], extent[1], extent[3]]
        if in_memory:
            format = "MEM"
            creation_options = []
        else:
            format = "GTiff"
            creation_options = [
                "COMPRESS=ZSTD",
                "PREDICTOR=2",
                "TILED=YES",
                "BIGTIFF=YES",
            ]
        options = gdal.WarpOptions(
            format=format,
            outputBounds=bounds,
            xRes=cell_size_out,
            yRes=cell_size_out,
            dstSRS=srs_out,
            srcSRS=srs_in,
            srcNodata=ds_in.GetRasterBand(1).GetNoDataValue(),
            dstNodata=no_data,
            outputType=data_type,
            resampleAlg=interpolation,
            warpOptions=[
                f"NUM_THREADS={CPU_AVAILABLE}",
            ],
            creationOptions=creation_options,
        )
        if in_memory:
            ds_out = gdal.Warp("", ds_in, options=options)
            return ds_out  # return in memory raster
        else:
            gdal.Warp(out_raster, ds_in, options=options)

        ds_in = None

    def rasterize_to_new_raster(
        self,
        out_raster,
        in_vector,
        value,
        cell_size,
        extent,
        format="GTiff",
        nodata=-9999,
        data_type=gdal.GDT_Float32,
        all_touched=False,
    ):
        vector = None
        try:
            vector = ogr.Open(in_vector)
        except Exception:
            drv = in_vector.GetDriver()
            drv_name = drv.ShortName
            if drv_name == "MEM" or drv_name == "Memory":
                vector = in_vector
        if vector is None:
            raise FileNotFoundError(
                f"Input vector file {in_vector} could not be opened."
            )

        # Create the output raster
        driver = gdal.GetDriverByName(format)
        if driver is None:
            raise ValueError(f"Driver {format} not found.")

        # Calculate raster dimensions
        x_min, x_max, y_min, y_max = extent
        cols = int((x_max - x_min) / cell_size)
        rows = int((y_max - y_min) / cell_size)
        if out_raster is None:
            out_raster_ds = driver.Create("", cols, rows, 1, data_type)
        else:
            out_raster_ds = driver.Create(out_raster, cols, rows, 1, data_type)
        if out_raster_ds is None:
            raise IOError(f"Could not create output raster file {out_raster}")

        # Set the spatial reference and geotransform
        out_raster_ds.SetGeoTransform((x_min, cell_size, 0, y_max, 0, -cell_size))
        out_raster_ds.SetProjection(vector.GetLayer(0).GetSpatialRef().ExportToWkt())

        # Set nodata value
        out_band = out_raster_ds.GetRasterBand(1)
        out_band.SetNoDataValue(nodata)

        options = []
        if all_touched:
            options.append("ALL_TOUCHED=TRUE")
        extent_str = f"{x_min},{y_min},{x_max},{y_max}"
        options.append(f"OUTPUT_BOUNDS={extent_str}")

        gdal.RasterizeLayer(
            out_raster_ds, [1], vector.GetLayer(), burn_values=[value], options=options
        )

        out_band.FlushCache()
        out_raster_ds.FlushCache()
        if format == "MEM":
            return out_raster_ds

    @staticmethod
    def get_nearest_cell_center(gt, point_x, point_y):
        """Return center of the cell containing the point
        :param gt: geotransform of the raster
        :param point_x: x coordinate of the point
        :param point_y: y coordinate of the point

        """
        x_origin, y_origin, cell_size_x, cell_size_y = gt[0], gt[3], gt[1], gt[5]
        snapped_x = (
            x_origin
            + cell_size_x * floor((point_x - x_origin) / cell_size_x)
            + cell_size_x / 2
        )
        snapped_y = (
            y_origin
            + cell_size_y * floor((point_y - y_origin) / cell_size_y)
            + cell_size_y / 2
        )
        return snapped_x, snapped_y

    @staticmethod
    def extract_window(
        band, px, py, window_size=3, safe_mode=False, mask_value=-999999
    ):
        """
        Extract a window around a pixel in a raster band
        :param band: raster band
        :param px: x coordinate of the center pixel
        :param py: y coordinate of the center pixel
        :param window_size: size of the window
        :param safe_mode: if True, return None if window is out of bounds
        :return: window as numpy array

        """
        # if window size is odd, return error
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        half_window = window_size // 2
        start_px = px - half_window
        start_py = py - half_window
        try:
            window = band.ReadAsArray(start_px, start_py, window_size, window_size)
        except Exception:
            if safe_mode:
                return None
            else:
                window = np.full((window_size, window_size), mask_value)
        return window

    def snap_point_on_array_pos(x, y, window_size):
        center_x_index = int(window_size / 2)
        center_y_index = int(window_size / 2)
        dif_x = x - center_x_index
        dif_y = y - center_y_index
        return dif_x, dif_y


class Vector:
    @staticmethod
    def load_vector(path, bbox=None, layer=None):
        """
        Load vector file to GeoDataFrame. Supports parquet files and OGR-supported formats.
        If multiple layers exist and no specific layer is requested, loads all layers into one GeoDataFrame.
        
        :param path: Path to vector file
        :param bbox: Optional bounding box tuple (xmin, ymin, xmax, ymax)
        :param layer: Optional specific layer name to load
        :return: GeoDataFrame or None if loading fails
        """
        if path.endswith(".parquet") or path.endswith(".pqt"):
            try:
                gdf = gpd.read_parquet(path)
                return gdf
            except Exception as e:
                print(f"Unable to load {path} to GeoDataFrame: {e}")
                return None
        else:
            try:
                # Use OGR to list available layers
                ds = ogr.Open(path)
                if ds is None:
                    raise FileNotFoundError(f"Could not open file: {path}")
                
                layer_count = ds.GetLayerCount()
                layers_to_load = []
                
                # Determine which layers to load
                if layer is not None:
                    # Load only the specified layer
                    lyr = ds.GetLayerByName(layer)
                    if lyr is None:
                        raise ValueError(f"Layer '{layer}' not found in {path}")
                    layers_to_load = [layer]
                else:
                    # Load all available layers
                    layers_to_load = [ds.GetLayer(i).GetName() for i in range(layer_count)]
                
                # Load each layer using geopandas
                gdfs = []
                for lyr_name in layers_to_load:
                    try:
                        if bbox:
                            gdf = gpd.read_file(path, layer=lyr_name, engine="pyogrio", bbox=tuple(bbox))
                        else:
                            gdf = gpd.read_file(path, layer=lyr_name, engine="pyogrio")
                        
                        if not gdf.empty:
                            gdfs.append(gdf)
                    except Exception as e:
                        print(f"Unable to load layer '{lyr_name}' from {path}: {e}")
                        continue
                
                ds = None  # Close the datasource
                
                if not gdfs:
                    print(f"No layers could be loaded from {path}")
                    return None
                
                # Combine all layers into a single geodataframe
                if len(gdfs) == 1:
                    return gdfs[0]
                else:
                    # Ensure all geodataframes have the same CRS before concatenating
                    target_crs = gdfs[0].crs
                    for i, gdf in enumerate(gdfs[1:], 1):
                        if gdf.crs != target_crs:
                            gdfs[i] = gdf.to_crs(target_crs)
                    
                    combined_gdf = pd.concat(gdfs, ignore_index=True)
                    return combined_gdf
            
            except Exception as e:
                print(f"Unable to load {path} to GeoDataFrame: {e}")
                return None

    @staticmethod
    def save_vector(gdf, path, gpd_driver="GPKG"):
        try:
            gdf.to_file(path, driver=gpd_driver)
        except Exception as e:
            print(f"Unable to save GeoDataFrame to {path}: {e}")

    @staticmethod
    def save_vector_to_parquet(gdf, path):
        try:
            gdf.to_parquet(path, engine="pyarrow")
        except Exception as e:
            print(f"Unable to save GeoDataFrame to Parquet file: {path}: {e}")

    @staticmethod
    def reproject_vector(gdf, epsg_out):
        if gdf.crs.to_epsg() == epsg_out:
            return gdf
        try:
            gdf_reprojected = gdf.to_crs(epsg=epsg_out)
            return gdf_reprojected
        except Exception as e:
            print(f"Unable to reproject GeoDataFrame: {e}")
            return None

    @staticmethod
    def buffer_vector(gdf: gpd.GeoDataFrame, buffer_size: float, dissolve=False):
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame.")

        try:
            gdf["geometry"] = gdf["geometry"].buffer(buffer_size)

            if dissolve:
                gdf = gdf.dissolve()

            return gdf
        except Exception as e:
            raise RuntimeError(f"Unable to buffer GeoDataFrame: {e}")

    @staticmethod
    def select_by_attribute_categorical(
        gdf: gpd.GeoDataFrame, attribute: str, values: list, remove=False
    ):
        """
        Filter a GeoDataFrame based on categorical values from list of values.

        :param gdf: gpd.GeoDataFrame, the input GeoDataFrame
        :param attribute: str, the name of the column to filter on
        :param values: list, the values to select or remove
        :param remove: bool, if True, removes rows matching the values; otherwise, keep them - default
        :return: gpd.GeoDataFrame, the filtered GeoDataFrame
        """
        if isinstance(values, str):
            values = [values]
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame.")
        if attribute not in gdf.columns:
            raise ValueError(f"Attribute '{attribute}' not found in the GeoDataFrame.")
        if remove:
            gdf = gdf[~gdf[attribute].isin(values)]
        else:
            gdf = gdf[gdf[attribute].isin(values)]
        return gdf

    @staticmethod
    def filter_by_extent(gdf: gpd.GeoDataFrame, extent: list):
        """
        :param gdf - Geodataframe to be filtered
        :param extent - list of coordinates [xmin, xmax, ymin, ymax]
        """
        # Define the extent as a box (bounding box)
        extent_box = box(extent[0], extent[2], extent[1], extent[3])
        gdf_filtered = gdf[gdf.geometry.intersects(extent_box)]
        return gdf_filtered

    @staticmethod
    def merge_geodataframes(gdf_list: list):
        """
        :param gdf_list
        """
        gdf_merged = pd.concat(gdf_list, ignore_index=True)
        return gdf_merged

    @staticmethod
    def list_to_txt(input_list, path, name):
        output_file = join(path, name)
        with open(output_file, "w") as output:
            output.write("\n".join(map(str, input_list)))

    @staticmethod
    def convert_extent_to_polygon(extent):
        """
        :param extent - list of coordinates [xmin, xmax, ymin, ymax]
        :return shapely.Polygon object
        """
        coordinates = (
            (extent[0], extent[2]),
            (extent[1], extent[2]),
            (extent[1], extent[3]),
            (extent[0], extent[3]),
            (extent[0], extent[2]),
        )
        return Polygon(coordinates)

    @staticmethod
    def convert_bounds_to_polygon(bounds):
        """
        :param extent - list of coordinates [xmin, ymin, xmax, ymax]
        :return shapely.Polygon object
        """
        coordinates = (
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
            (bounds[0], bounds[1]),
        )
        return Polygon(coordinates)

    @staticmethod
    def convert_extent_to_ogr_geometry(extent):
        """
        :param extent - list of coordinates [xmin, xmax, ymin, ymax]
        :return ogr.Geometry object
        """
        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint_2D(extent[0], extent[2])
        ring.AddPoint_2D(extent[1], extent[2])
        ring.AddPoint_2D(extent[1], extent[3])
        ring.AddPoint_2D(extent[0], extent[3])
        ring.AddPoint_2D(extent[0], extent[2])

        # Create polygon
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        return polygon

    @staticmethod
    def ext1_to_ext2(source_ext, source_epsg, out_epsg):
        polygon = Vector.convert_extent_to_polygon(source_ext)
        gdf = gpd.GeoDataFrame(
            index=[0], crs=f"epsg:{source_epsg}", geometry=[polygon]
        ).to_crs(epsg=out_epsg)
        bounds = gdf.total_bounds
        return [bounds[0], bounds[2], bounds[1], bounds[3]]

    def bounds1_to_bounds2(bounds, source_epsg, out_epsg):
        polygon = Vector.convert_bounds_to_polygon(bounds)
        gdf = gpd.GeoDataFrame(
            index=[0], crs=f"epsg:{source_epsg}", geometry=[polygon]
        ).to_crs(epsg=out_epsg)
        bounds = gdf.total_bounds
        return bounds

    def progress_callback(complete, message, data):
        percent = int(complete * 100)  # round to integer percent
        data.update(percent)  # update the progressbar
        return 1

    def select_files_in_extent(list_files, extent):
        """
        select files where extents intersects
        :return list of file intersecting given extent
        """
        extent_polygon = Vector.convert_extent_to_polygon(extent)
        filtered_list = []
        for file in list_files:
            compare_polygon = Vector.convert_extent_to_polygon(
                Vector.get_extent_vector(file)
            )
            if shapely.intersects(extent_polygon, compare_polygon):
                filtered_list.append(file)
        return filtered_list

    def get_epsg_from_vector(file):
        """
        extracts epsg of polygon file
        :param file vector dataset or geopandas.GeodataFrame
        :return extent as epsg code as int
        """
        if isinstance(file, gpd.GeoDataFrame):
            return int(file.crs.to_epsg())

        elif os.path.splitext(file)[1] in [".shp", ".gpkg", ".geojson"]:
            data = ogr.Open(file)
            layer = data.GetLayer()
            spatial_ref = layer.GetSpatialRef()
            if spatial_ref.IsGeographic() == 1:  # this is a geographic srs
                proj_type = "GEOGCS"
                return int(spatial_ref.GetAuthorityCode(proj_type))
            else:  # this is a projected srs
                proj_type = "PROJCS"
                return int(spatial_ref.GetAuthorityCode(proj_type))
        else:
            print(f"File: {file} is not in supported format.")

    def clip_polygon(data, clipping_area):
        """
        :param data vector dataset
        :type data gpd.Geodataframe or string
        :param clipping_area vector dataset
        :type clipping_area gpd.Geodataframe or string
        if geometries are not in the same projection, CRS from param clipping area would be applied
        """
        if isinstance(data, gpd.GeoDataFrame):
            pass
        else:
            data = gpd.read_file(data)
        if isinstance(clipping_area, gpd.GeoDataFrame):
            pass
        else:
            clipping_area = gpd.read_file(clipping_area)

        data_epsg = data.crs.to_epsg()
        clipping_area_epsg = clipping_area.crs.to_epsg()
        if data_epsg != clipping_area_epsg:
            data = data.to_crs(clipping_area_epsg)
        clipped_data = gpd.clip(data, clipping_area)
        return clipped_data

    def fix_geometry(data, compare=False):
        """
        :param data vector dataset
        :type data gpd.Geodataframe or string
        :param compare if True, function will return number of rows where geometry was fixed
        :type compare bool
        :return fixed geometries
        Returns fixed geometries for geodataframe or vector geodataset.
        """
        if isinstance(data, gpd.GeoDataFrame):
            pass
        elif os.path.splitext(data)[1] in [".shp", ".gpkg", ".geojson"]:
            data = gpd.read_file(data)
        else:
            raise ValueError(f"File: {data} is not in supported format.")
        if compare:
            data["copy_geometry"] = data["geometry"]
        g = data["geometry"]
        g_v = g.make_valid()
        data["geometry"] = g_v
        if compare:
            data["changed"] = np.where(data["copy_geometry"] == data["geometry"], 0, 1)
            geom_change = data["changed"].sum()
            if geom_change > 0:
                print(f"Geometries were fixed in {geom_change} rows")
            data.drop(columns=["copy_geometry", "changed"], inplace=True)
        return data

    def align_extent_to_snap(extent, snap_raster, cell_size=0, optimize=False):
        """
        :param extent - list of coordinates [left_x, right_x, bottom_y, upper_y] (mandatory)
        :param snap_raster - reference grid for snapping (mandatory) or its extent
        :param cell_size - raster resolution, mandatory if optimize is True (mandatory if optimize is True)
        :param optimize - if true, bounding box will be resized in accordance with raster original extent (optional)
        :return extent snapped to snap raster
        """
        ds_snap = gdal.Open(snap_raster)
        geotransform_snap = list(ds_snap.GetGeoTransform())
        if (
            geotransform_snap[1] > 0
            and (extent[0] - geotransform_snap[0]) > 0
            or geotransform_snap[1] < 0
            and ((extent[0] - geotransform_snap[0]) < 0)
        ):
            meth_x = floor
        else:
            meth_x = ceil

        if (
            geotransform_snap[5] > 0
            and (extent[3] - geotransform_snap[3]) > 0
            or geotransform_snap[5] < 0
            and ((extent[3] - geotransform_snap[3]) < 0)
        ):
            meth_y = floor
        else:
            meth_y = ceil

        x_left = (
            geotransform_snap[0]
            + meth_x((extent[0] - geotransform_snap[0]) / geotransform_snap[1])
            * geotransform_snap[1]
        )
        y_upper = (
            geotransform_snap[3]
            + meth_y((extent[3] - geotransform_snap[3]) / geotransform_snap[5])
            * geotransform_snap[5]
        )
        x_right = (
            x_left
            + ceil((extent[1] - x_left) / geotransform_snap[1]) * geotransform_snap[1]
        )
        y_lower = (
            y_upper
            + ceil((extent[2] - y_upper) / geotransform_snap[5]) * geotransform_snap[5]
        )
        if optimize:
            x_left = floor((extent[0] - x_left) / cell_size) * cell_size + x_left
            x_right = ceil((extent[1] - x_right) / cell_size) * cell_size + x_right
            y_lower = floor((extent[2] - y_lower) / cell_size) * cell_size + y_lower
            y_upper = ceil((extent[3] - y_upper) / cell_size) * cell_size + y_upper

        extent_snapped = [x_left, x_right, y_lower, y_upper]
        return extent_snapped

    @staticmethod
    def get_extent_layer(layer):
        extent = layer.GetExtent()
        return [extent[0], extent[1], extent[2], extent[3]]

    @staticmethod
    def get_extent_vector(file):
        """
        Extracts extent of polygon file
        :param file vector dataset or geopandas.GeoDataFrame
        :return extent as [xmin, xmax, ymin, ymax]
        """
        if isinstance(file, gpd.GeoDataFrame):
            bounds = file.total_bounds
            return [bounds[0], bounds[2], bounds[1], bounds[3]]
        elif os.path.splitext(file)[1] in [".shp", ".gpkg", ".geojson"]:
            data = ogr.Open(file)
            layer = data.GetLayer()
            extent = Vector.get_extent_layer(layer)
            return extent
        else:
            raise ValueError(f"File: {file} is not in supported format.")

    def get_convex_hull(file, out_epsg, out_type="shapely_polygon"):
        """
        based on: https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#save-the-convex-hull-of-all-geometry-from-an-input-layer-to-an-output-layer
        extracts extent of convex hull in specified projection
        :param file vector dataset or gpd.GeodataFrame
        :type file str or object
        :param out_epsg epsg code
        :type out_epsg int
        :param out_type defines output's data_type (shapely.Polygon or ogr.Geometry)
        :return convex hull as Polygon
        """
        if isinstance(file, (gpd.GeoDataFrame, gpd.GeoSeries)):
            convex_hull = Polygon(file.convex_hull.get_coordinates())

        elif os.path.splitext(file)[1] in [".shp", ".gpkg", ".geojson"]:
            data = ogr.Open(file)
            layer = data.GetLayer()
            geomcol = ogr.Geometry(ogr.wkbGeometryCollection)
            for feature in layer:
                geomcol.AddGeometry(feature.GetGeometryRef())

            # Calculate convex hull
            convexhull = geomcol.ConvexHull()
            sourceprj = layer.GetSpatialRef()
            targetprj = osr.SpatialReference()
            targetprj.ImportFromEPSG(out_epsg)
            transform = osr.CoordinateTransformation(sourceprj, targetprj)
            convexhull.Transform(transform)
            if out_type == "shapely_polygon":
                convex_hull = loads(convexhull.ExportToWkt())
        else:
            print(f"Warning: File {file} is not in supported format.")
            return None
        return convex_hull

    @staticmethod
    def extract_intersection(gdf_lines1, gdf_lines2):
        """
        Extracts intersection of two line layers
        :param gdf_lines1: first line layer
        :param gdf_lines2: second line layer
        :return: gpd.GeoDataFrame with intersection
        """
        if gdf_lines1.crs != gdf_lines2.crs:
            Exception("CRS of both layers must be the same")
        intersections = gdf_lines1.unary_union.intersection(gdf_lines2.unary_union)
        # Filter only Point geometries
        try:
            points = [geom for geom in intersections.geoms if geom.geom_type == "Point"]
        except Exception:
            points = []
        return points

    @staticmethod
    def load_ogr_ds_to_geodataframe(ogr_ds):
        gdf = (pyogrio.read_dataframe(ogr_ds),)
        return gdf

    def sjoin_nearest(gdf1, gdf2, distance=0.1):
        """
        Spatial join of two geodataframe on nearest predicate
        :param gdf1: gpd.GeoDataFrame
        :param gdf2: gpd.GeoDataFrame with points
        :param distance: float, maximum distance to join
        :return: gpd.GeoDataFrame with points and lines
        """
        gdf_joined = gpd.sjoin_nearest(
            gdf1, gdf2, how="inner", max_distance=distance, distance_col="temp_dist"
        )
        return gdf_joined

    def merge_lines_on_pseudonodes(gdf_lines):
        """
        Merges lines on pseudonodes
        BEWARE: this function return geodataframe with geometry only
        :param gdf_lines: gpd.GeoDataFrame with Linestring as geometry
        :return: gpd.GeoDataFrame with merged lines
        """
        crs_orig = gdf_lines.crs
        if len(gdf_lines.index) > 1:
            geom_merged = linemerge(unary_union(gdf_lines.geometry))
            gdf_lines_merged = (
                gpd.GeoDataFrame(geometry=[geom_merged])
                .explode()
                .reset_index()
                .set_crs(crs_orig)
            )
        else:
            gdf_lines_merged = gdf_lines.copy()
        return gdf_lines_merged


class Geometry:
    # from StackOverflow to be tested
    def find_nearest_line(point, lines):
        return min(lines, key=lambda line: line.distance(point))

    def extract_points_on_line(line, point, dist: list):
        proj_point = nearest_points(line, point)[1]
        line_length = line.length
        distances = [
            line.project(proj_point) - dist[0],
            line.project(proj_point) + dist[1],
        ]

        start_dist = max(0, distances[0])
        end_dist = min(line_length, distances[1])
        start_point = line.interpolate(start_dist)
        end_point = line.interpolate(end_dist)
        return start_point, end_point

    def create_points_on_line(line, distance):
        points = []
        for i in np.arange(0, line.length, distance):
            point = line.interpolate(i)
            points.append(point)
        return points

    def construct_point(x, y):
        return Point(x, y)

    def construct_line(points):
        return LineString(points)

    def calculate_angle(p1, p2, p3):
        a = sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        b = sqrt((p3.x - p2.x) ** 2 + (p3.y - p2.y) ** 2)
        c = sqrt((p3.x - p1.x) ** 2 + (p3.y - p1.y) ** 2)
        if a == 0 or b == 0:
            return 180
        angle = acos((a**2 + b**2 - c**2) / (2 * a * b))
        return degrees(angle)

    def close_holes(poly: Polygon) -> Polygon:
        """
        Close polygon holes by limitation to the exterior ring.

        :param poly: Input shapely Polygon
        :return: Polygon geometry

            :Example:

            df.geometry.apply(lambda p: close_holes(p))

        """
        if poly.interiors:
            return Polygon(list(poly.exterior.coords))
        else:
            return poly

    def create_fishnet(extent, cell_size, block_size, overlap):
        """
        Creates regular square grid
        :param extent: list of coordinates [xmin, xmax, ymin, ymax]
        :param cell_size: size of the cell in meters
        :param block_size: size of the edge in number of cells
        :param overlap: overlap between cells in meters
        :return: list of shapely.Polygon objects representing the grid cells
        """
        xmin, xmax, ymin, ymax = extent
        block_size = block_size * cell_size
        x, y = (xmin, ymin)
        geom_array = []
        while y <= ymax:
            while x <= xmax:
                geom = Polygon(
                    [
                        (x, y),
                        (x, y + block_size),
                        (x + block_size, y + block_size),
                        (x + block_size, y),
                        (x, y),
                    ]
                )
                geom_array.append(geom)
                x += block_size - overlap
            x = xmin
            y += block_size - overlap
        return geom_array


class CulvertsSpecific:
    @staticmethod
    def create_fishnet(extent, cell_size, block_size, overlap):
        """
        Creates regular square grid
        :param extent: list of coordinates [xmin, xmax, ymin, ymax]
        :param cell_size: size of the cell in meters
        :param block_size: size of the edge in number of cells
        :param overlap: overlap between cells in meters
        :return: list of shapely.Polygon objects representing the grid cells
        """
        xmin, xmax, ymin, ymax = extent
        block_size = block_size * cell_size
        x, y = (xmin, ymin)
        geom_array = []
        while y <= ymax:
            while x <= xmax:
                geom = Polygon(
                    [
                        (x, y),
                        (x, y + block_size),
                        (x + block_size, y + block_size),
                        (x + block_size, y),
                        (x, y),
                    ]
                )
                geom_array.append(geom)
                x += block_size - overlap
            x = xmin
            y += block_size - overlap
        return geom_array


class Hydro:
    @staticmethod
    def fill_depression_wang(dtm_array):
        filled_dtm, d8 = pyflwdir.dem.fill_depressions(elevtn=dtm_array)
        return filled_dtm

    def flow_accumulation(d8):
        flow_acc = pyflwdir.flow_accumulation(d8)
        return flow_acc


class MosaicRasters:
    """
    Class for creating mosaic for rasters, min added in 1.0.3

    :param rasters: List of input rasters
    :param output: Output file as tif or any other
    :param cell_size: Output cell size
    :param epsg: EPSG code is integer
    :param data_type: data type as gdal.GDT_Int16 or other, gdal.GDT_Float32
    :param no_data: Output no data value
    :param progress: Print progress sof merging


    :return:   String with saved tif

    """

    def __init__(
        self,
        rasters: list,
        output: str,
        cell_size: int | float = None,
        epsg: int = None,
        data_type=gdal.GDT_Int16,
        no_data: int | float = -32768,
        compress: bool = True,
        progress: bool = True,
        extent: list = None,
        compress_type=None,
        round_extent: bool = False,
    ):
        if compress_type is None:
            self.compress_type = ["COMPRESS=LZW"]
        else:
            self.compress_type = compress_type
        self.rasters = rasters
        self.output = output
        self.cell_size = cell_size
        if not self.cell_size:
            self.cell_size = self.get_resolution(rasters[0])[0]
        self.data_type = data_type
        self.no_data = no_data
        self.epsg = epsg
        self.progress = progress
        self.round_extent = round_extent

        if extent is not None:
            self.extent = extent
            self.bands = 1
        else:
            self.extent, self.bands = self._get_max_extent()

        if self.epsg is None:
            self.epsg = self.get_epsg(rasters[0])
        self.compress = compress
        self.dst = self._initialize_raster()
        self.dst = None
        self.dst = gdal.Open(self.output, gdal.GA_Update)

    @staticmethod
    def get_epsg(raster: str) -> int:
        """
        Extracting epsg if not specified on init
        """
        d = gdal.Open(raster)
        proj = osr.SpatialReference(wkt=d.GetProjection())
        return proj.GetAttrValue("AUTHORITY", 1)

    @staticmethod
    def get_resolution(raster):
        """
        Method which will return restolution in x and y direction

        :param raster: Raster file as string
        :return: X, Y cell size
        """
        src = gdal.Open(raster)
        gt = src.GetGeoTransform()
        cell_size_x = gt[1]
        cell_size_y = gt[5]
        src = None
        return cell_size_x, cell_size_y

    def _get_max_extent(self):
        """
        Private method which will get extent from all rasters

        :return: [total_xmin, total_xmax, total_ymin, total_ymax]
        """
        total_xmax, total_xmin, total_ymax, total_ymin = [], [], [], []
        bands_total = []
        for file in self.rasters:
            src = gdal.Open(file, gdal.GA_ReadOnly)

            bands_total.append(src.RasterCount)
            upx, xres, xskew, upy, yskew, yres = src.GetGeoTransform()

            ## no of cols/rows
            cols = src.RasterXSize
            rows = src.RasterYSize
            ## calculation of raster extent
            xmin = upx + 0 * xres + rows * xskew
            ymin = upy + 0 * yskew + rows * yres

            xmax = upx + cols * xres + 0 * xskew
            ymax = upy + cols * yskew + 0 * yres

            total_xmax.append(xmax)
            total_xmin.append(xmin)
            total_ymax.append(ymax)
            total_ymin.append(ymin)
            src = None
        if self.round_extent:
            return [
                round(min(total_xmin), 2),
                round(max(total_xmax), 2),
                round(min(total_ymin), 2),
                round(max(total_ymax), 2),
            ], max(bands_total)
        else:
            return [
                min(total_xmin),
                max(total_xmax),
                min(total_ymin),
                max(total_ymax),
            ], max(bands_total)

    def _initialize_raster(self):
        """
        Private method which will create output file
        """
        extent = self.extent
        if self.progress:
            print(extent)
        tiff_width = int((extent[1] - extent[0]) // self.cell_size)
        tiff_height = int((extent[3] - extent[2]) // self.cell_size)
        if self.progress:
            print(f"Size of new raster is {tiff_width} x {tiff_height}")

        gt = extent[0], self.cell_size, 0, extent[3], 0, -self.cell_size

        raster = gdal.GetDriverByName("GTiff")
        if self.compress:
            options = [
                "TFW=YES",
                "BIGTIFF=YES",
                "TILED=YES",
                "NUM_THREADS=ALL_CPUS",
            ]
            options.extend(self.compress_type)
            dst_ds = raster.Create(
                self.output,
                int(tiff_width),
                int(tiff_height),
                self.bands,
                self.data_type,
                options=options,
            )
        else:
            dst_ds = raster.Create(
                self.output,
                int(tiff_width),
                int(tiff_height),
                self.bands,
                self.data_type,
                options=["TFW=YES", "BIGTIFF=YES", "TILED=YES"],
            )
        ## setting upper left corner and resolution, y has to be minus
        dst_ds.SetGeoTransform(gt)

        srs = osr.SpatialReference()
        if isinstance(self.epsg, int):
            srs.ImportFromEPSG(self.epsg)
        elif "EPSG" in self.epsg or "epsg" in self.epsg:
            srs.ImportFromEPSG(self.epsg)
        elif "ESRI" in self.epsg:
            srs.ImportFromESRI(self.epsg.split(":")[1])
        elif "proj" in self.epsg:
            srs.ImportFromProj4(self.epsg)
        else:
            srs.ImportFromWkt([self.epsg])

        srs = srs.ExportToWkt()
        dst_ds.SetProjection(srs)
        band = dst_ds.GetRasterBand(1)

        ## we can change no data value if needed, this one is UnsignedInteger16Bit
        band.SetNoDataValue(self.no_data)

        # array = np.zeros((int(tiff_height), int(tiff_width)))
        # band.WriteArray(array)

        return dst_ds

    def mosaic_maximum(self):
        """
        Calling this method on the class will start the mosaicing of maximum per all rasters.

        :return: None

        :Example:

        ::

         MosaicRasters(*arg).mosaic_maximum()
        """
        for ix, file in enumerate(self.rasters):
            try:
                if self.progress:
                    print(f"{ix + 1} out of {len(self.rasters)}")

                ## write raster into array
                src = gdal.Open(file)

                for band_number in range(src.RasterCount):
                    band_dst = self.dst.GetRasterBand(band_number + 1)
                    band = src.GetRasterBand(band_number + 1)
                    desc = band.GetDescription()
                    band_dst.SetDescription(desc)

                    no_data_src = band.GetNoDataValue()

                    cols = src.RasterXSize
                    rows = src.RasterYSize

                    ## upper left corner
                    gt = src.GetGeoTransform()
                    # print(gt)
                    # print(dst.GetGeoTransform())
                    minx = gt[0]
                    maxy = gt[3]

                    y, x = maxy - self.extent[3], self.extent[0] - minx

                    locy, locx = (
                        int(abs(y / self.cell_size)),
                        int(abs(x / self.cell_size)),
                    )

                    # print(f'\tRequesting array of size {rows} x {cols} at location {locx} {locy}')

                    ## comparing if array already written is smaller or not
                    array_src = band.ReadAsArray(0, 0, cols, rows)
                    array_src = np.where(
                        (array_src == no_data_src), self.no_data, array_src
                    )

                    array_dst = band_dst.ReadAsArray(locx, locy, cols, rows)

                    array_to_write = np.where(
                        array_dst != self.no_data,
                        np.where(
                            array_src != self.no_data,
                            np.where(array_dst < array_src, array_src, array_dst),
                            array_dst,
                        ),
                        np.where(array_src != self.no_data, array_src, self.no_data),
                    )

                    self.dst.GetRasterBand(band_number + 1).WriteArray(
                        array_to_write, locx, locy
                    )
                src = None
            except Exception as e:
                print(e)

        self.dst = None
        return self.output

    def mosaic_minimum(self):
        """
        :Example:

        ::

         MosaicRasters(*arg).mosaic_minimum()

        """

        for ix, file in enumerate(self.rasters):
            try:
                if self.progress:
                    print(f"{ix + 1} out of {len(self.rasters)}")
                ## write raster into array
                src = gdal.Open(file)

                for band_number in range(src.RasterCount):
                    band_dst = self.dst.GetRasterBand(band_number + 1)
                    band = src.GetRasterBand(band_number + 1)
                    no_data_src = band.GetNoDataValue()

                    cols = src.RasterXSize
                    rows = src.RasterYSize

                    ## upper left corner
                    gt = src.GetGeoTransform()
                    # print(gt)
                    # print(dst.GetGeoTransform())
                    minx = gt[0]
                    maxy = gt[3]

                    y, x = maxy - self.extent[3], self.extent[0] - minx

                    locy, locx = (
                        int(abs(y / self.cell_size)),
                        int(abs(x / self.cell_size)),
                    )

                    # print(f'\tRequesting array of size {rows} x {cols} at location {locx} {locy}')

                    ## comparing if array already written is smaller or not
                    array_src = band.ReadAsArray(0, 0, cols, rows)
                    array_src = np.where(
                        (array_src == no_data_src), self.no_data, array_src
                    )

                    array_dst = band_dst.ReadAsArray(locx, locy, cols, rows)
                    ## if DEST raster has data
                    array_to_write = np.where(
                        array_dst != self.no_data,
                        ## if also src has data
                        np.where(
                            array_src != self.no_data,
                            ## if output is higher than small
                            np.where(
                                array_dst > array_src,
                                ## put small
                                array_src,
                                ## leave as was
                                array_dst,
                            ),
                            ## if src is no_data, put data from big
                            array_dst,
                        ),
                        ## if no_data in dest but data in src
                        np.where(
                            array_src != self.no_data,
                            ## leave as src small
                            array_src,
                            ## else put no_data if all is no_data
                            self.no_data,
                        ),
                    )

                    self.dst.GetRasterBand(band_number + 1).WriteArray(
                        array_to_write, locx, locy
                    )
            except Exception as e:
                print(e)
        self.dst = None
        return self.output

    def mosaic_in_order(self):
        """
        Mosaic rasters in order as they are provided in list.
        First raster will be on top, last on the bottom.

        :Example:

        ::

         MosaicRasters(*arg).mosaic_in_order()

        """

        for ix, file in enumerate(reversed(self.rasters)):
            if self.progress:
                print(f"{ix + 1} out of {len(self.rasters)}")

            ## write raster into array
            src = gdal.Open(file)
            for band_number in range(src.RasterCount):
                band_dst = self.dst.GetRasterBand(band_number + 1)

                band = src.GetRasterBand(band_number + 1)
                no_data_src = band.GetNoDataValue()

                cols = src.RasterXSize
                rows = src.RasterYSize

                ## upper left corner
                gt = src.GetGeoTransform()
                # print(gt)
                # print(dst.GetGeoTransform())
                minx = gt[0]
                maxy = gt[3]

                y, x = maxy - self.extent[3], self.extent[0] - minx

                locy, locx = int(abs(y / self.cell_size)), int(abs(x / self.cell_size))

                # print(f'\tRequesting array of size {rows} x {cols} at location {locx} {locy}')

                ## comparing if array already written is smaller or not
                array_src = band.ReadAsArray(0, 0, cols, rows)
                array_src = np.where(
                    (array_src == no_data_src), self.no_data, array_src
                )

                array_dst = band_dst.ReadAsArray(locx, locy, cols, rows)
                ## if DEST raster has data
                array_to_write = np.where(
                    array_dst != self.no_data,
                    ## if also src has data
                    np.where(array_src != self.no_data, array_src, array_dst),
                    ## if no_data in dest but data in src
                    np.where(
                        array_src != self.no_data,
                        ## leave as src small
                        array_src,
                        ## else put no_data if all is no_data
                        self.no_data,
                    ),
                )

                self.dst.GetRasterBand(band_number + 1).WriteArray(
                    array_to_write, locx, locy
                )
        self.dst = None
        return self.output

    def mosaic_sum(self):
        """
        Mosaic rasters as a sum.

        :Example:

        ::

         MosaicRasters(*arg).mosaic_sum()

        """

        for ix, file in enumerate(reversed(self.rasters)):
            if self.progress:
                print(f"{ix + 1} out of {len(self.rasters)}")

            ## write raster into array
            src = gdal.Open(file)
            for band_number in range(src.RasterCount):
                band_dst = self.dst.GetRasterBand(band_number + 1)

                band = src.GetRasterBand(band_number + 1)
                no_data_src = band.GetNoDataValue()
                no_data_dst = self.dst.GetRasterBand(band_number + 1).GetNoDataValue()

                cols = src.RasterXSize
                rows = src.RasterYSize

                ## upper left corner
                gt = src.GetGeoTransform()
                # print(gt)
                # print(dst.GetGeoTransform())
                minx = gt[0]
                maxy = gt[3]

                y, x = maxy - self.extent[3], self.extent[0] - minx

                locy, locx = int(abs(y / self.cell_size)), int(abs(x / self.cell_size))

                print(
                    f"\tRequesting array of size {rows} x {cols} at location {locx} {locy}"
                )

                ## comparing if array already written is smaller or not
                array_src = band.ReadAsArray(0, 0, cols, rows)
                array_src = np.where((array_src == no_data_src), 0, array_src)

                array_dst = band_dst.ReadAsArray(locx, locy, cols, rows)
                array_dst = np.where(array_dst == no_data_dst, 0, array_dst)
                ## if DEST raster has data
                array_to_write = array_dst + array_src
                array_to_write = np.where(
                    array_to_write == 0, self.no_data, array_to_write
                )
                self.dst.GetRasterBand(band_number + 1).WriteArray(
                    array_to_write, locx, locy
                )

        self.dst = None
        return self.output
