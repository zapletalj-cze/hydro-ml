"""script creates zsh lines for tuflow using user lines of culverts/dikes/channels
- it adds an elevation values from DTM
   INPUT: 1) shp with lines  - culverts, channels, levees
          2) DTM
          3) min/max/mean specification for zonal statistics
          4) the size of buffer ot the vertices for the zonal stats.
    OUTPUT: 1) zsh line _K.shp
            2) zsh points with elevation _P

            output will be in located into sub-folder ./zsh_outputs
"""

# Import system modules
import os, geopandas as gpd, multiprocessing as mp
import shapely
from osgeo import gdal
from shapely.geometry import Point
from pathlib import Path
import numpy as np
import warnings

gdal.UseExceptions()
## INPUTS ############################
type_zsh_line = "culvert"  ### 'culvert' | 'levee' | 'channel'
no_cores = 15  ## number of available PC cores - 1
RASTER_IS_INTEGER = True  # is DEM integer? True = divide by 100.0


## path to line files - shp file with dikes/levees, channels, culverts
## list of shapefiles, where the calculation should be performed
list_zsh_line_sources = [
    # r"b:\01_Projects\157_Canada_Flood_v3\01_MD\01_HAZARD\05_Defences\02_levees\O02\O02_levees_240202.gpkg"
    # r'b:\01_Projects\157_Canada_Flood_v3\01_MD\01_HAZARD\05_Defences\03_culverts\S06\S06_001_culverts_L_1m_250827_Lida.gpkg'
    # r'v:\01_Projects\157_Canada_Flood_v3\01_MD\98_Users\MSa\06_TU_Pluvial\cities\Halifax\zsh_culverts\E02_culverts_halifax_251210.gpkg'
    r"v:\01_Projects\157_Canada_Flood_v3\01_MD\98_Users\MSa\06_TU_Pluvial\cities\Quebec\culverts_merge\q15_culverts_lida_add_msa_251212.gpkg"
]
## DTM that is used
# default_dtmPath = r"\\eupraappp073\e$\01_Projects\157_CanadaFlood_v3\01_MD\01_HAZARD\01_DTM\1m\S\_Mosaic\S06_S27_Edmonton_1m_crop.tif"
# default_dtmPath = r'V:\01_Projects\157_Canada_Flood_v3\01_MD\98_Users\MSa\06_TU_Pluvial\cities\Halifax\dtm_recalculated\DTM_WB_buildings_1m_Halifax.tif'
default_dtmPath = r"V:\01_Projects\157_Canada_Flood_v3\01_MD\98_Users\MSa\06_TU_Pluvial\cities\Quebec\dtm_recalculated\DTM_WB_buildings_1m_Quebec.tif"
# default_dtmPath = r'B:\01_Projects\154_Poland_Flood_v3\01_MD\01_HAZARD\01_DTM\PL_DEM_10m_borders_20220624.tif'
###############################################


## variables - do not change
id_field = "Id"  ## not change

## specification:
typeOfPoints_forElevation = (
    "END" if type_zsh_line == "culvert" else "ALL"
)  ## 'ALL' - all vertices of polyline / 'END' - only endpoints are used as ZSH points
buffer_size_dtm_multiply_mid = 3  ## e.g 1.5 for CAN PL; 2 means 2xCellSize ; for all vertices is better to use bigger value than 2
buffer_size_dtm_multiply_start_end = 3  ##first and end, dont change
statistics = (
    "MAX" if type_zsh_line == "levee" else "MIN"
)  ## 'MIN' (for channels etc.) / 'MAX' ( for levees etc.) / 'MEAN'


def AddRemovefield_zsh(zshF, NotRemoveOlderValue, NotRemoveOlderValue_ID):
    ## function add necessary filed fo zsh layer type (plus field ID for zonal statistic) and remove unnecessary fields
    ## IN: zshF - full pass to zsh shapefile
    ##    NotRemoveOlderValue - keep older value for necessary fields except Id
    ##    NotRemoveOlderValue_ID - 'y' keep older "Id" value

    print("Update of ZSH fields done. File: " + zshF)


def get_local_value(dem, point, buffer):
    """
    Method for getting matrix from DTM. Offsets the point to upper left corner. Reads buffer*2 cells and does statistics.
    :param dem: DEM for extraction
    :param point: point for extraction
    :param buffer: buffer for statistics multiple of cells 3 = 3*3 cells
    :return: value at the point
    """
    src = gdal.Open(dem, gdal.GA_ReadOnly)
    gt = src.GetGeoTransform()
    no_data = src.GetRasterBand(1).GetNoDataValue()
    data_type = src.GetRasterBand(1).DataType

    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * src.RasterXSize
    miny = maxy + gt[5] * src.RasterYSize

    point_x, point_y = point.x, point.y
    if (minx <= point_x <= maxx) and (miny <= point_y <= maxy):
        loc_x = int((point_x - minx) / gt[1])
        loc_y = int((point_y - maxy) / gt[5])
        value_center = src.GetRasterBand(1).ReadAsArray(loc_x, loc_y, 1, 1)[0][0]
        if buffer == 1:
            loc_x = loc_x
            loc_y = loc_y
        else:
            loc_x -= buffer
            loc_y -= buffer

        array = src.GetRasterBand(1).ReadAsArray(loc_x, loc_y, buffer * 2, buffer * 2)
        vals = np.unique(array)
        if len(vals) == 1:
            if vals[0] == no_data:
                warnings.warn("There is only no_data around levee extraction point")

        array = np.ma.masked_equal(array, no_data)
        if value_center == no_data:
            warnings.warn("There is no_data on some levee extraction point")
            value_center = -9999
        else:
            if RASTER_IS_INTEGER:
                if data_type in (
                    gdal.GDT_Int32,
                    gdal.GDT_Int16,
                    gdal.GDT_Byte,
                    gdal.GDT_UInt16,
                    gdal.GDT_UInt32,
                ):
                    value_center = value_center / 100.0
                else:
                    raise Exception(
                        "Raster is not integer, but RASTER_IS_INTEGER is set to True"
                    )
            else:
                if data_type in (
                    gdal.GDT_Int32,
                    gdal.GDT_Int16,
                    gdal.GDT_Byte,
                    gdal.GDT_UInt16,
                    gdal.GDT_UInt32,
                ):
                    raise Exception(
                        "Raster is integer, but RASTER_IS_INTEGER is set to False"
                    )
                else:
                    value_center = value_center

        if statistics == "MIN":
            value = np.ma.min(array)
        elif statistics == "MAX":
            value = np.ma.max(array)
        elif statistics == "MEAN":
            value = np.ma.mean(array)

        del src
        if isinstance(value, np.ma.core.MaskedConstant):
            value = -9999
        else:
            if RASTER_IS_INTEGER:
                value = value / 100.0

        return value, value_center
    else:
        raise Exception("Point not inside raster")


def line_pool(line):
    """
    Method for iterating over line, and extracting value.
    :param line: Polyline
    :return: Tuple with Point coordinates, Z value and line ID
    """
    nodes = []
    line_no = line[0]
    line = line[1]
    # print(line)
    if isinstance(line, shapely.geometry.MultiLineString):
        list_of_points = []
        for one_line in line.geoms:
            list_of_points.extend(list(one_line.coords))
    else:
        list_of_points = list(line.coords)
    ## iterate over points in line

    for index, point in enumerate(list_of_points):
        if typeOfPoints_forElevation.upper() == "ALL":
            ## if first/last point use different buffer size
            if index == 0 or index == len(list_of_points) - 1:
                buffer = buffer_size_dtm_multiply_start_end
            else:
                buffer = buffer_size_dtm_multiply_mid
                ## call method for values extraction
            value, center_value = get_local_value(default_dtmPath, Point(point), buffer)
            nodes.append((Point(point), value, line_no, center_value))

        elif typeOfPoints_forElevation.upper() == "END":
            if index == 0 or index == len(list_of_points) - 1:
                buffer = buffer_size_dtm_multiply_start_end
                value, center_value = get_local_value(
                    default_dtmPath, Point(point), buffer
                )
                nodes.append((Point(point), value, line_no, center_value))

    return nodes


def get_points(zsh_file_l):
    """
    Method which get all points from line (or endpoints) and get Min/Max/Mean from raster
    :param zsh_file_l: Line layer
    :return: GeoDataframe
    """
    gdf = gpd.read_file(zsh_file_l)
    gdf_line = gdf.copy()
    crs = gdf.crs

    array = gdf.geometry.to_numpy()
    array = [(index, item) for index, item in enumerate(array)]
    nodes = []
    print("\tIterating over lines and points")
    ## iterate over line with multiple processes
    with mp.Pool(no_cores if len(array) > no_cores else len(array)) as pool:
        for result in pool.imap(line_pool, array):
            nodes.extend(result)
    # for item in array:
    #     nodes.append(line_pool(item))
    gdf = gpd.GeoDataFrame(
        [col for col in nodes], columns=["geometry", "Z", "Id", "centerZ"]
    )
    gdf.crs = crs
    return gdf, gdf_line


def add_fields(gdf_final):
    ## adding fields to dataframe, same as pandas, change to '' if text or 0 if number
    columns = ["Z", "dZ", "Shape_Widt", "Shape_Opti", "pCFW", "pFLC", "centerZ"]
    for column in columns:
        if column not in gdf_final.columns:
            if column == "Shape_Opti":
                gdf_final[column] = ""
            elif column == "Shape_Widt":
                if "width" in gdf_final.columns:
                    gdf_final[column] = gdf_final["width"]
                else:
                    gdf_final[column] = 0.0
            else:
                gdf_final[column] = 0.0
    gdf_final["Z"] = np.where(gdf_final["Z"] == 0.0, 0.01, gdf_final["Z"])
    gdf_final = gdf_final[
        [
            "geometry",
            "Z",
            "dZ",
            "Shape_Widt",
            "Shape_Opti",
            "pCFW",
            "pFLC",
            "Id",
            "centerZ",
        ]
    ]
    return gdf_final


if __name__ == "__main__":
    ## script body
    for zsh_file_l in list_zsh_line_sources:
        print("Started processing")
        gdf, gdf_line = get_points(zsh_file_l)

        gdf = add_fields(gdf)
        gdf_line["Id"] = gdf_line.index.astype(int)

        gdf_line = add_fields(gdf_line)
        gdf_line["Z"] = -99999.0

        output_location = os.path.join(os.path.dirname(zsh_file_l), "zsh_output")
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        output_line_name = (
            os.path.basename(Path(zsh_file_l).stem) + "_L.shp"
        )  # .#replace('.shp', '_L.shp')
        output_line = os.path.join(output_location, output_line_name)

        output_point_name = (
            os.path.basename(Path(zsh_file_l).stem) + "_P.shp"
        )  # .replace('.shp', '_P.shp')
        output_point = os.path.join(output_location, output_point_name)

        gdf.to_file(output_point)
        gdf_line.to_file(output_line)

        print(f"zsh points: {output_point}")
        print(f"zsh lines: {output_line}")
        print("Saved, done")
