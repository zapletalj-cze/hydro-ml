import gc
import os
import shutil
import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal
import multiprocessing as mp
from shapely.geometry import Point, MultiLineString, LineString, box, Polygon
from shapely.ops import linemerge, split
from ifgis.raster import SampleRaster, save_array
from gis import Hydro, Vector, Raster, Geometry
from numba import njit
import collections
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math
from typing import List
from scipy.spatial import cKDTree


def filter_roads(gdf_roads, filter_field, filter):
    """
    Filter roads based on attribute values
    """
    gdf_roads = gdf_roads[gdf_roads[filter_field].isin(filter)]
    # TUNNELS AND BRIDGES REMOVED
    gdf_roads = Vector.select_by_attribute_categorical(
        gdf=gdf_roads, attribute="tunnel", values="T", remove=True
    )
    gdf_roads = Vector.select_by_attribute_categorical(
        gdf=gdf_roads, attribute="bridge", values="T", remove=True
    )
    return gdf_roads


def get_points_along_geometry(
    roads: str, interval, filter_field, filter, CELL_SIZE, EPSG
):
    """
    Get points along geometry
    UPDATE - remove tunnels and bridges
    """
    df_roads = gpd.read_file(roads).to_crs(epsg=EPSG)
    df_roads = filter_roads(df_roads, filter_field, filter)
    lines = df_roads.geometry.values.tolist()

    inlets = []
    for line in lines:
        distances = np.arange(interval, line.length, interval)
        distances = np.append(distances, line.length)
        points = [line.interpolate(distance) for distance in distances]
        inlets.extend(points)
    inlets = gpd.GeoDataFrame(geometry=inlets, crs=f"EPSG:{EPSG}")
    inlets.geometry = inlets.geometry.buffer(CELL_SIZE)
    inlets = inlets.dissolve()
    inlets = inlets.explode(ignore_index=True)
    inlets.geometry = inlets.geometry.centroid
    inlets = inlets.reset_index()
    inlets["pointid"] = inlets.index
    return inlets, df_roads


def snap_to_min(band, point, gt, size):
    x_end, y_end = point.x, point.y
    x_end_s, y_end_s = Raster.get_nearest_cell_center(gt, point_x=x_end, point_y=y_end)
    x_end_l = int((x_end_s - gt[0]) / gt[1])
    y_end_l = int((y_end_s - gt[3]) / gt[5])
    dem_window = Raster.extract_window(
        band,
        x_end_l,
        y_end_l,
        window_size=size,
        mask_value=999999,
        safe_mode=False,
    )
    dem_window_min = np.min(dem_window)
    if dem_window_min != -999999 or dem_window_min != 999999:
        dem_window_min_ind = np.argwhere(np.where(dem_window == dem_window_min, 1, 0))[
            0
        ]
        y_pos, x_pos = dem_window_min_ind[0], dem_window_min_ind[1]
        dif_x, dif_y = Raster.snap_point_on_array_pos(x_pos, y_pos, size)
        x_end_s_shift, y_end_s_shift = (
            x_end_s + dif_x * gt[1],
            y_end_s + dif_y * gt[5],
        )
    else:
        x_end_s_shift, y_end_s_shift, dem_window_min = None, None, None
    return x_end_s_shift, y_end_s_shift, dem_window_min


def remove_outlets_buildings(
    outlets: gpd.GeoDataFrame, buildings, out_file: str, bounds=None, bounds_epsg=None
):
    """
    Remove outlets that intersect with buildings.

    Args:
        outlets (gpd.GeoDataFrame): GeoDataFrame of outlet points.
        buildings (str): Path to the building GeoJSON/Shapefile.
        out_file (str): Path to save the filtered outlets GeoDataFrame.

    Returns:
        str: Path to the output file.
    """
    # TODO: ALLOW TO LOAD FROM LIST OF FILES/FILE
    if isinstance(buildings, list):
        buildings_gdfs = []
        for building in buildings:
            crs_build = Vector.get_epsg_from_vector(building)
            if crs_build != bounds_epsg:
                bounds_load = Vector.bounds1_to_bounds2(
                    bounds, from_epsg=bounds_epsg, to_epsg=crs_build
                )
                gdf_temp = Vector.load_vector(building, bbox=tuple(bounds_load)).to_crs(
                    epsg=bounds_epsg
                )
            else:
                bounds_load = bounds
                gdf_temp = Vector.load_vector(building, bbox=tuple(bounds_load))
            buildings_gdfs.append(gdf_temp)
        buildings_gdf = pd.concat(buildings_gdfs, ignore_index=True)
        buildings_gdf = gpd.GeoDataFrame(buildings_gdf, crs=buildings_gdfs[0].crs)
    else:
        crs_build = Vector.get_epsg_from_vector(buildings)
        if crs_build != bounds_epsg:
            bounds_load = Vector.bounds1_to_bounds2(
                bounds, from_epsg=bounds_epsg, to_epsg=crs_build
            )
            buildings_gdf = Vector.load_vector(
                buildings, bbox=tuple(bounds_load)
            ).to_crs(epsg=bounds_epsg)
        else:
            bounds_load = bounds
            buildings_gdf = Vector.load_vector(buildings, bbox=tuple(bounds_load))
    original_outlets_gdf = Vector.load_vector(outlets)
    outlets_cols = original_outlets_gdf.columns.tolist()
    print(
        f"\tOutlets filtering on buildings, initial count: {len(original_outlets_gdf)}"
    )
    joined_outlets = gpd.sjoin(
        original_outlets_gdf, buildings_gdf, how="left", predicate="intersects"
    )
    filtered_outlets = joined_outlets[joined_outlets["index_right"].isna()]
    filtered_outlets = filtered_outlets.drop(columns=["index_right"])
    filtered_outlets = filtered_outlets.reset_index(drop=True)
    outlets_final = filtered_outlets[outlets_cols]
    print(f"\tOutlets filtering on buildings, final count: {len(outlets_final)}")
    outlets_final.to_file(out_file)
    return out_file


def create_inlets_selection(
    points: gpd.GeoDataFrame, water_network: list, cell_size: int, out_file: str
):
    """
    Remove inlets inside river bodies
    """
    if not os.path.isfile(out_file):
        dfs = [gpd.read_file(water, bbox=points) for water in water_network]
        df = pd.concat(dfs)
        df.geometry = df.geometry.buffer(cell_size * 2)
        if "index_right" in points.columns:
            points = points.drop(columns=["index_right"])
        if "index_right" in df.columns:
            df = df.drop(columns=["index_right"])
        inlets_in_wb = gpd.sjoin(points, df, how="left", predicate="within")
        results = inlets_in_wb[inlets_in_wb["index_right"].isna()]
        results = results[["geometry", "pointid"]]
        results.geometry = results.geometry.buffer(cell_size)
        results = results.dissolve()
        results = results.explode(ignore_index=True)
        results.geometry = results.geometry.centroid
        results = results.reset_index()
        results.to_file(out_file)
    return out_file


def _snap(point: Point, dtm: str) -> Point:
    """
    Private method, which will snap vertices to lowest cell in surrouding area
    :param segment: One segment of river network
    :return: New snapped line
    """
    new_point = get_min_point_from_array(point, dtm=dtm)
    return new_point


def snap_gdf_to_dtm(gdf, transform):
    """ """

    origin_x, pixel_width, _, origin_y, _, pixel_height = transform

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    cols = np.floor((x - origin_x) / pixel_width)
    rows = np.floor((y - origin_y) / pixel_height)
    snapped_x = origin_x + (cols + 0.5) * pixel_width
    snapped_y = origin_y + (rows + 0.5) * pixel_height
    snapped_geometry = [Point(px, py) for px, py in zip(snapped_x, snapped_y)]
    snapped_gdf = gdf.copy()
    snapped_gdf.geometry = snapped_geometry
    return snapped_gdf


def get_min_point_from_array(point: Point, dtm: str, buffer: int = 1) -> Point:
    """
    Method for getting matrix from DTM array. Offsets the point to upper left corner.
    Reads buffer*2 cells and returns min value.
    :param point: point for extraction
    :param buffer: buffer for statistics
    :return: location in the raster of the min value in the coordinate system
    """
    src = gdal.OpenShared(dtm)
    gt = src.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    x_size = src.RasterXSize
    y_size = src.RasterYSize
    maxx = minx + gt[1] * x_size
    miny = maxy + gt[5] * y_size
    no_data = src.GetRasterBand(1).GetNoDataValue()

    point_x, point_y = point.x, point.y
    # calculate location inside the array and move it a bit to the left
    if (minx <= (point_x - 1 * gt[1]) <= maxx) and (
        miny <= (point_y + 1 * gt[1]) <= maxy
    ):
        loc_x = int((point_x - minx) / gt[1])
        loc_y = int((point_y - maxy) / gt[5])
        loc_x -= buffer
        loc_y -= buffer
        try:
            array = src.GetRasterBand(1).ReadAsArray(loc_x, loc_y, 3, 3)
            array = np.ma.masked_equal(array, no_data)

            index_of_value = np.unravel_index(array.argmin(), array.shape)
            loc_x += index_of_value[1]
            loc_y += index_of_value[0]
            del src
            return Point(
                minx + loc_x * gt[1] + gt[1] / 2,
                maxy - loc_y * gt[1] - gt[1] / 2,
            )
        except Exception:
            # print(f"Error reading array for point {point}, error: {e}")
            del src
            return point

    else:
        del src
        return point


def move_inlets(inlets: str, out_file, cell_size, epsg, dtm):
    """
    Move inlets to the lowest point in DTM in 3x3 window
    """
    if isinstance(inlets, str):
        inlets = gpd.read_file(inlets)
    inlets = snap_gdf_to_dtm(
        inlets,
        transform=tuple(
            Raster.get_raster_info(dtm, info=["geotransform"])["geotransform"]
        ),
    )
    args_list = [(inlet.geometry, dtm) for _, inlet in inlets.iterrows()]

    # Use ThreadPoolExecutor for parallel calls WITHOUT __main__ guard
    with ThreadPoolExecutor(max_workers=12) as executor:
        inlets_snapped = list(executor.map(lambda args: _snap(*args), args_list))

    inlets = gpd.GeoDataFrame(geometry=inlets_snapped, crs=f"EPSG:{epsg}")

    inlets.geometry = inlets.geometry.buffer(cell_size)
    inlets = inlets.dissolve()
    inlets = inlets.explode(ignore_index=True)
    inlets.geometry = inlets.geometry.centroid
    inlets = inlets.reset_index()
    inlets = inlets.explode(index_parts=True)

    inlets = inlets.reset_index(drop=True)
    inlets["pointid"] = inlets.index
    if out_file:
        inlets.to_file(out_file)
        return out_file
    else:
        return inlets


def move_outlets(outlets: str, out_file, epsg, cell_size, dtm):
    """
    Move outlets to the lowest point in DTM in 3x3 window
    """
    if isinstance(outlets, str):
        outlets = gpd.read_file(outlets)

    args_list = [(outlet.geometry, dtm) for _, outlet in outlets.iterrows()]

    # Use ThreadPoolExecutor for parallel calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        outlets_snapped = list(executor.map(lambda args: _snap(*args), args_list))

    outlets = gpd.GeoDataFrame(geometry=outlets_snapped, crs=f"EPSG:{epsg}")

    outlets.geometry = outlets.geometry.buffer(cell_size)
    outlets = outlets.dissolve()
    outlets = outlets.explode(ignore_index=True)
    outlets.geometry = outlets.geometry.centroid
    outlets = outlets.reset_index()
    outlets = outlets.explode(index_parts=True)

    outlets = outlets.reset_index(drop=True)
    outlets["outlet_id"] = outlets.index
    if out_file:
        outlets.to_file(out_file)
        return out_file
    else:
        return outlets


def split_polygon_equal_area(
    polygon: Polygon, target_area: float = 4000
) -> List[Polygon]:
    total_area = polygon.area
    num_parts = max(1, math.ceil(total_area / target_area))
    grid_dim = math.ceil(math.sqrt(num_parts))
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    cell_width = width / grid_dim
    cell_height = height / grid_dim
    parts = []
    for i in range(grid_dim):
        for j in range(grid_dim):
            cell_box = box(
                minx + i * cell_width,
                miny + j * cell_height,
                minx + (i + 1) * cell_width,
                miny + (j + 1) * cell_height,
            )
            intersection = polygon.intersection(cell_box)
            if not intersection.is_empty and intersection.area > 0:
                if intersection.geom_type == "Polygon":
                    parts.append(intersection)
                elif intersection.geom_type == "MultiPolygon":
                    parts.extend(list(intersection.geoms))
    return parts


def create_outlets(rn, waterbody, interval, out_file, epsg, cell_size):
    df_rn = gpd.read_file(rn)
    if "fclass" in df_rn.columns:  # changed 260130
        df_rn = df_rn[df_rn["fclass"].isin(["river", ""])]
    if "tunnel" in df_rn.columns:
        df_rn = df_rn[df_rn["tunnel"] != 1]
    elif "underground" in df_rn.columns:
        df_rn = df_rn[df_rn["underground"] != 1]
    else:
        pass
    geoms = df_rn.union_all()
    if isinstance(geoms, MultiLineString):
        geoms = linemerge(geoms)
    cols = {
        "geometry": [geoms],
    }
    df_rn = gpd.GeoDataFrame(cols, crs=f"EPSG:{epsg}")
    df_rn = df_rn.explode()

    lines = df_rn.geometry.values.tolist()

    outlets = []
    for line in lines:
        if line.length < 300:
            continue
        distances = np.arange(interval, line.length, interval)
        points = [line.interpolate(distance) for distance in distances]
        outlets.extend(points)
    df_outlets = gpd.GeoDataFrame(geometry=outlets, crs=f"EPSG:{epsg}")
    if os.path.isfile(waterbody):
        df_wb = gpd.read_file(waterbody)
        df_wb = df_wb.dissolve().explode()
        df_wb = df_wb[df_wb.geometry.area > 800]
        df_wb_s = df_wb[df_wb.geometry.area <= 15000]
        df_wb_m = df_wb[(df_wb.geometry.area > 15000) & (df_wb.geometry.area <= 45000)]
        df_wb_l = df_wb[
            (df_wb.geometry.area > 45000) & (df_wb.geometry.area <= 3500000)
        ]
        df_wb_xl = df_wb[df_wb.geometry.area > 3500000]
        wb_m = []
        for idx_wb_m, row_wb_m in df_wb_m.iterrows():
            parts = split_polygon_equal_area(
                row_wb_m.geometry, row_wb_m.geometry.area / 3
            )
            for part in parts:
                wb_m.append(part)
        df_wb_m = gpd.GeoDataFrame(geometry=wb_m, crs=df_wb.crs)
        wb_l = []
        for idx_wb_l, row_wb_l in df_wb_l.iterrows():
            parts = split_polygon_equal_area(
                row_wb_l.geometry, row_wb_l.geometry.area / 10
            )
            for part in parts:
                wb_l.append(part)
        df_wb_l = gpd.GeoDataFrame(geometry=wb_l, crs=df_wb.crs)
        wb_xl = []
        for idx_wb_xl, row_wb_xl in df_wb_xl.iterrows():
            parts = split_polygon_equal_area(row_wb_xl.geometry, 10000)
            for part in parts:
                wb_xl.append(part)
        df_wb_xl = gpd.GeoDataFrame(geometry=wb_xl, crs=df_wb.crs)
        df_wb = pd.concat([df_wb_s, df_wb_m, df_wb_l, df_wb_xl])
        df_wb.geometry = df_wb.geometry.representative_point()
        df_result = pd.concat([df_outlets, df_wb])
    else:
        df_result = df_outlets
    df_result.geometry = df_result.geometry.buffer(cell_size * 2)
    df_result = df_result.dissolve()
    df_result = df_result.explode(index_parts=True)
    df_result.geometry = df_result.geometry.representative_point()
    df_result["outlet_id"] = range(1, len(df_result) + 1)
    df_result.to_file(out_file)
    return out_file


def cleanup(temp):
    for the_file in os.listdir(temp):
        file_path = os.path.join(temp, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


## computation is memory heavy, if we have 40000 inlets and 3000 outlets (easily in a small city)
## we need at least 16 GB of free RAM, 40000 * 3000 = alot of combinations
## all we do is create a spatial matrix csv with SAGA cmd (distance matrix)
## filter out inlets below outlets
## clean layer based on distance and height from outlet
## merge inlets and outlets, create index and other required fields
## input are at the bottom
## 40000 inlets and 4000 outlets = 21 minutes
## 40000 inlets and 2000 outlets = 21 minutes
## 40000 inlets and 300 outlets = 3-5 minutes


def extract(dem, inlets, outlets, temp):
    print("\tExtracting DEM to points:")
    inlets = SampleRaster(
        inlets,
        dem,
        os.path.join(temp, os.path.basename(inlets)),
        "height_inlet",
        multiprocess=False,
    ).process_whole()

    print("\tOutlets")
    outlets = SampleRaster(
        outlets,
        dem,
        os.path.join(temp, os.path.basename(outlets)),
        "height_outlet",
        multiprocess=False,
    ).process_whole()
    return inlets, outlets


def extract_inlets(dem, inlets, temp):
    print("\tExtracting DEM to points:")
    print("\tInlets")
    inlets = SampleRaster(
        inlets,
        dem,
        None,
        "dtm_value",
        multiprocess=False,
    ).process_whole()
    return inlets


def _distance_matrix(
    source, near, field_left="id", field_right="id", outlets_to_remove=None
):
    """
    Compute Euclidean distance matrix between two point GeoDataFrames.

    :param source: input points layer (GeoDataFrame or path), id field required
    :param near: input near layer (GeoDataFrame or path), id field required
    :param field_left: name of ID field in source
    :param field_right: name of ID field in near
    :param outlets_to_remove: iterable of IDs in 'near' to be excluded
    :return: pandas DataFrame with distances, index=source IDs, columns=near IDs
    """
    if not isinstance(source, gpd.GeoDataFrame):
        source = gpd.read_file(source)
    if not isinstance(near, gpd.GeoDataFrame):
        near = gpd.read_file(near)

    # Work on copies to avoid side effects on original GDFs
    source = source.copy()
    near = near.copy()

    if outlets_to_remove:
        near = near[~near[field_right].isin(outlets_to_remove)]

    source["ID_POINT"] = source[field_left]
    near["ID_NEAR"] = near[field_right]

    # Extract coordinates
    source_list = source.geometry.values.tolist()
    near_list = near.geometry.values.tolist()

    x1 = np.asarray([item.x for item in source_list])
    y1 = np.asarray([item.y for item in source_list])
    x2 = np.asarray([item.x for item in near_list])
    y2 = np.asarray([item.y for item in near_list])

    # Build distance matrix (squared differences, then sqrt in-place)
    x_i = x1[:, np.newaxis]
    x_j = x2[np.newaxis, :]
    y_i = y1[:, np.newaxis]
    y_j = y2[np.newaxis, :]

    d = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
    np.sqrt(d, out=d)

    df_matrix = pd.DataFrame(d, index=source["ID_POINT"], columns=near["ID_NEAR"])

    return df_matrix


def connect(inlets, outlets, out_file, dem, temp, epsg):
    inlets, outlets = extract(dem, inlets, outlets, temp)
    print("\tSpatial joinning outlets to inlets")
    # os.environ['PATH'] += ';' + os.path.join(qgs_pth,'saga-ltr')
    inlets = gpd.read_file(inlets)
    inlets = inlets.sample(frac=1, random_state=42).reset_index(drop=True)  # ranodmized
    outlets = gpd.read_file(outlets)
    print(inlets.columns, outlets.columns)
    # 10/09/2025 - newly added this part
    inlets.reset_index(drop=True, inplace=True)
    inlets["pointid"] = inlets.index
    outlets.reset_index(drop=True, inplace=True)
    outlets["outlet_id"] = outlets.index
    outlet_id_counts_master = outlets["outlet_id"].value_counts().to_dict()

    # 10/09/2025 - END newly added this part
    # TODO: use numpy from https://stackoverflow.com/questions/58713739/distance-matrix-between-two-point-layers - DONE
    # TODO: set data types correctly -
    # inlets_joined = inlets.sjoin_nearest(outlets, distance_col="outlet_distance")
    len_inlets = len(inlets)
    # Divide inlets into chunks of 10,000 for processing
    chunk_size = 1000
    inlet_chunks = []
    inlets_c = inlets.copy()
    for i in range(0, len_inlets, chunk_size):
        chunk = inlets_c.iloc[i : i + chunk_size]
        inlet_chunks.append(chunk)

    chunk_container = []
    # Use tqdm to show progress for chunk processing
    outlets_removed = 0
    for i, chunk in enumerate(tqdm(inlet_chunks, desc="Processing inlet chunks")):
        outlets_remove = [
            outlet_id
            for outlet_id, count in outlet_id_counts_master.items()
            if count > 400
        ]
        outlets_removed = len(outlets_remove)
        df_joinned = _distance_matrix(
            chunk,
            outlets,
            field_left="pointid",
            field_right="outlet_id",
            outlets_to_remove=outlets_remove,
        )
        df_joinned["pointid"] = df_joinned.index
        df_joinned = df_joinned.melt(
            id_vars=["pointid"], var_name="outlet_id", value_name="outlet_distance"
        )

        df_joinned = pd.merge(
            df_joinned,
            chunk[["pointid", "height_inlet", "Number_of", "geometry"]],
            on="pointid",
            how="left",
        )

        inlets_joined = pd.merge(
            df_joinned,
            outlets[["outlet_id", "height_outlet"]],
            on="outlet_id",
            how="left",
        )
        inlets_joined = gpd.GeoDataFrame(
            inlets_joined, geometry="geometry", crs=f"EPSG:{epsg}"
        )
        # print(df_joinned, df_joinned.columns)
        # print(f"Number of inlets before filtering: {len(inlets_joined.index)}")
        final_df = inlets_joined[
            inlets_joined["height_outlet"] < inlets_joined["height_inlet"]
        ].copy()
        final_df = final_df.sort_values(
            by=["pointid", "outlet_distance"], ascending=[1, 1]
        )
        final_df.drop_duplicates(subset=["pointid"], keep="first", inplace=True)

        final_df.sort_values(
            by=["outlet_id", "height_inlet"], ascending=[1, 0], inplace=True
        )
        final_df.reset_index(drop=True, inplace=True)
        final_df["VP_Sur_Index"] = (
            final_df.index
        )  # TODO check this, most likely wrong and needs to be descending for each group
        final_df["VP_Network_ID"] = final_df["outlet_id"].astype(int)
        outlet_id_counts_slave = final_df["VP_Network_ID"].value_counts().to_dict()
        final_df["Type"] = "I"
        final_df["Inlet_Type"] = "Can_No_7"
        final_df["Conn_No"] = 4
        final_df["Number_of"] = final_df["Number_of"]
        final_df["Lag_Approach"] = ""
        final_df["Lag_Value"] = np.nan
        final_df["ZIn"] = final_df["height_inlet"]
        final_df["ZOut"] = final_df["height_outlet"]
        for outlet_id in outlet_id_counts_slave:
            if outlet_id in outlet_id_counts_master:
                outlet_id_counts_master[outlet_id] = (
                    outlet_id_counts_master[outlet_id]
                    + outlet_id_counts_slave[outlet_id]
                )

        chunk_container.append(final_df)
    final_df = pd.concat(chunk_container, ignore_index=True)
    final_df.loc[final_df["Type"] == "O", "ZOut"] = final_df.loc[
        final_df["Type"] == "O", "ZIn"
    ]
    final_df = final_df.sort_values(by=["VP_Network_ID", "ZIn"], ascending=[True, True])
    final_df["VP_Sur_Index"] = final_df.groupby("VP_Network_ID").cumcount() + 1
    final_df = final_df.reset_index(drop=True)

    outlets["ID"] = 0
    outlets["Type"] = "O"
    outlets["VP_Network_ID"] = outlets["outlet_id"].astype(int)
    outlets["Inlet_Type"] = "0"
    outlets["Conn_No"] = 0
    outlets["VP_Sur_Index"] = 0
    outlets["ZOut"] = outlets["height_outlet"]
    add_sur_index(final_df, outlets, out_file)


def reconnect(inlets, outlets, out_file, dem, temp, epsg, id_min_inlet, id_min_outlet):
    inlets, outlets = extract(dem, inlets, outlets, temp)
    print("\tSpatial joinning outlets to inlets")
    # os.environ['PATH'] += ';' + os.path.join(qgs_pth,'saga-ltr')
    inlets = gpd.read_file(inlets)
    outlets = gpd.read_file(outlets)
    # print(inlets.head(50))
    # print(outlets.head(50))
    print(inlets.columns, outlets.columns)
    # 10/09/2025 - newly added this part
    inlets.reset_index(drop=True, inplace=True)
    inlets["pointid"] = inlets.index + id_min_inlet
    inlets["Number_of"] = 0
    outlets.reset_index(drop=True, inplace=True)
    outlets["outlet_id"] = outlets.index + id_min_outlet
    # 10/09/2025 - END newly added this part
    # TODO: use numpy from https://stackoverflow.com/questions/58713739/distance-matrix-between-two-point-layers - DONE
    # TODO: set data types correctly -
    # inlets_joined = inlets.sjoin_nearest(outlets, distance_col="outlet_distance")
    df_joinned = _distance_matrix(
        inlets, outlets, field_left="pointid", field_right="outlet_id"
    )
    df_joinned.index = df_joinned.index.astype(int)
    df_joinned["pointid"] = df_joinned.index
    df_joinned = df_joinned.melt(
        id_vars=["pointid"],
        var_name="outlet_id",
        value_name="outlet_distance",
    )

    df_joinned["outlet_id"] = df_joinned["outlet_id"].astype(int)
    df_joinned = pd.merge(
        df_joinned,
        inlets[["pointid", "height_inlet", "Number_of", "geometry"]],
        on="pointid",
        how="left",
    )
    inlets_joined = pd.merge(
        df_joinned, outlets[["outlet_id", "height_outlet"]], on="outlet_id", how="left"
    )
    inlets_joined = gpd.GeoDataFrame(
        inlets_joined, geometry="geometry", crs=f"EPSG:{epsg}"
    )
    # print(df_joinned, df_joinned.columns)
    # print(f"Number of inlets before filtering: {len(inlets_joined.index)}")
    final_df = inlets_joined[
        inlets_joined["height_outlet"] < inlets_joined["height_inlet"]
    ]
    final_df = final_df.sort_values(by=["pointid", "outlet_distance"], ascending=[1, 1])
    final_df.drop_duplicates(subset=["pointid"], keep="first", inplace=True)
    print(f"\tNumber of inlets after filtering: {len(final_df.index)}")

    final_df.sort_values(
        by=["outlet_id", "height_inlet"], ascending=[1, 0], inplace=True
    )
    final_df.reset_index(drop=True, inplace=True)
    final_df["VP_Sur_Index"] = (
        final_df.index
    )  # TODO check this, most likely wrong and needs to be descending for each group
    final_df["VP_Network_ID"] = final_df["outlet_id"].astype(int)
    final_df["Type"] = "I"
    final_df["Inlet_Type"] = "Can_No_7"
    final_df["Conn_No"] = 4
    final_df["Number_of"] = final_df["Number_of"]
    final_df["Lag_Approach"] = ""
    final_df["Lag_Value"] = np.nan

    final_df_sorted = final_df.sort_values("height_inlet", ascending=1).groupby(
        "VP_Network_ID"
    )
    final_df["VP_Sur_Index"] = final_df_sorted.cumcount() + 1

    outlets["ID"] = 0
    outlets["Type"] = "O"
    outlets["VP_Network_ID"] = outlets["outlet_id"].astype(int)
    outlets["Inlet_Type"] = "0"
    outlets["Conn_No"] = 0
    outlets["VP_Sur_Index"] = 0
    redo_sur_index(final_df, outlets, out_file)


def add_sur_index(inlets, outlets, out_file):
    # TODO: SET DATA TYPES CORRECTLY
    print("\tFinalizing file")
    items = [inlets, outlets]
    fields = {
        "VP_QMax": 10,
        "Width": 2,
        "Conn_2D": "SX",
        "pBlockage": 0,
    }
    df = pd.concat(items)
    for field in fields:
        df[field] = fields[field]
    df["ID"] = df.index
    fields = [
        "ID",
        "Type",
        "VP_Network_ID",
        "Inlet_Type",
        "VP_Sur_Index",
        "VP_QMax",
        "Width",
        "Conn_2D",
        "Conn_No",
        "pBlockage",
        "Number_of",
        "Lag_Approach",
        "Lag_Value",
        "ZIn",
        "ZOut",
        "geometry",
    ]

    df = df[fields]
    df["ID"] = np.where(df["Type"] == "I", 1000000 + df["ID"], df["ID"])
    df["ID"] = df["ID"].astype(str)
    df["Type"] = df["Type"].astype(str)
    df["VP_Network_ID"] = df["VP_Network_ID"].astype(
        int
    )  # differs from my version VP_Network_ID vs VPNetwork
    df["Inlet_Type"] = df["Inlet_Type"].astype(str)
    df["VP_Sur_Index"] = df["VP_Sur_Index"].astype(float)
    df["VP_QMax"] = df["VP_QMax"].astype(
        float
    )  # differs from my version VP_QMax vs QMax
    df["Width"] = df["Width"].astype(float)
    df["Conn_2D"] = df["Conn_2D"].astype(str)
    df["Conn_No"] = df["Conn_No"].astype(int)
    df["pBlockage"] = df["pBlockage"].astype(float)
    df.to_file(out_file)
    # Create an empty GeoDataFrame with the same columns and dtypes as df
    columns = df.columns
    dtypes = {col: df[col].dtype for col in columns if col != "geometry"}
    df_adjusted = gpd.GeoDataFrame(
        {
            col: pd.Series(dtype=dtypes.get(col, "object"))
            for col in columns
            if col != "geometry"
        },
        geometry=pd.Series(dtype="geometry"),
        crs=df.crs,
    )
    # Add the first item from df to df_adjusted
    df_adjusted = pd.concat([df_adjusted, df.iloc[[0]]], ignore_index=True)
    df_adjusted["ID"] = "77777777"
    df_adjusted["VP_Network_ID"] = 77777778
    df_adjusted["VP_Network_ID"] = df_adjusted["VP_Network_ID"].astype(int)
    df_adjusted["Type"] = "A"
    out_file_adjusted = out_file.replace(".gpkg", "_adjustments.gpkg")
    df_adjusted.to_file(out_file_adjusted)
    # Adjust lenght of all fields using ogr


def redo_sur_index(inlets, outlets, out_file):
    # TODO: SET DATA TYPES CORRECTLY
    print("\tFinalizing file")
    items = [inlets, outlets]
    fields = {
        "VP_QMax": 10,
        "Width": 2,
        "Conn_2D": "SX",
        "pBlockage": 0,
    }
    df = pd.concat(items)
    for field in fields:
        df[field] = fields[field]
    df["ID"] = df.index
    fields = [
        "ID",
        "Type",
        "VP_Network_ID",
        "Inlet_Type",
        "VP_Sur_Index",
        "VP_QMax",
        "Width",
        "Conn_2D",
        "Conn_No",
        "pBlockage",
        "Number_of",
        "Lag_Approach",
        "Lag_Value",
        "ZIn",
        "ZOut",
        "geometry",
    ]

    df = df[fields]
    df["ID"] = np.where(df["Type"] == "I", 1000000 + df["ID"], df["ID"])
    df["ID"] = df["ID"].astype(str)
    df["Type"] = df["Type"].astype(str)
    df["VP_Network_ID"] = df["VP_Network_ID"].astype(
        int
    )  # differs from my version VP_Network_ID vs VPNetwork
    df["Inlet_Type"] = df["Inlet_Type"].astype(str)
    df["VP_Sur_Index"] = df["VP_Sur_Index"].astype(float)
    df["VP_QMax"] = df["VP_QMax"].astype(
        float
    )  # differs from my version VP_QMax vs QMax
    df["Width"] = df["Width"].astype(float)
    df["Conn_2D"] = df["Conn_2D"].astype(str)
    df["Conn_No"] = df["Conn_No"].astype(int)
    df["pBlockage"] = df["pBlockage"].astype(float)
    df.to_file(out_file)
    # Create an empty GeoDataFrame with the same columns and dtypes as df
    columns = df.columns
    dtypes = {col: df[col].dtype for col in columns if col != "geometry"}
    df_adjusted = gpd.GeoDataFrame(
        {
            col: pd.Series(dtype=dtypes.get(col, "object"))
            for col in columns
            if col != "geometry"
        },
        geometry=pd.Series(dtype="geometry"),
        crs=df.crs,
    )
    # Add the first item from df to df_adjusted
    df_adjusted = pd.concat([df_adjusted, df.iloc[[0]]], ignore_index=True)
    df_adjusted["ID"] = "77777777"
    df_adjusted["VP_Network_ID"] = "77777778"
    df_adjusted["Type"] = "A"
    out_file_adjusted = out_file.replace(".gpkg", "_adjustments.gpkg")
    df_adjusted.to_file(out_file_adjusted)
    # Adjust lenght of all fields using ogr


def clip_layer_by_domain(layer, code, out):
    # geopandas clip whole layer by another geodataframe
    code = gpd.read_file(code)

    df = gpd.read_file(layer, bbox=code)
    df = gpd.clip(df, code, keep_geom_type=True)

    df.to_file(out)
    return out


def checkFile(file):
    if os.path.isfile(file):
        return True
    else:
        return False


@njit
def subtractions(a, b):
    return a - b


def generate_points_grid(xmin, xmax, ymin, ymax, step, epsg=3979):
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    points = [Point(x, y) for x in xs for y in ys]
    gdf = gpd.GeoDataFrame(geometry=points, epsg=epsg)
    return gdf


def get_depressions(dtm):
    """
    :param dtm: path-like string
    :return: path to filled DTM
    """
    dtm_array = gdal.Open(dtm).ReadAsArray()
    # no_data = Raster.get_raster_info(dtm, info="nodata")["nodata"]
    filled_dtm = Hydro.fill_depression_wang(dtm_array)
    depressions_array = subtractions(filled_dtm, dtm_array)
    # path_depression = rf"D:/temp/VPs/temp/{os.path.basename(dtm)[:-4]}_depressions.tif"
    path_depression = rf"/vsimem/{os.path.basename(dtm)[:-4]}_depressions.tif"
    save_array(
        array=depressions_array, output=path_depression, sample_raster=dtm
    )  # export grid
    return path_depression


def filter_depressions(depressions_path, out_depressions_path, area_threshold=350):
    """
    :param depressions_path: path to the depressions vector layer
    :param area_threshold: minimum size of depressions to keep
    :return: path to the filtered depressions vector layer
    """
    epsg_code = Raster.get_raster_info(depressions_path, info="projection_epsg")[
        "projection_epsg"
    ]
    depressions_map = gdal.Open(depressions_path).ReadAsArray()
    depressions_map = np.where(depressions_map > 0, 1, 0)
    depression_map_raster = Raster.from_array_mem(
        array=depressions_map,
        source_raster=depressions_path,
        epsg_out=epsg_code,
        raster_driver="MEM",
    )
    depressions_map = None
    Raster.raster_to_vect(
        in_raster=depression_map_raster,
        out_vector=out_depressions_path,
        layer_name="depressions",
        raster_val_name="is_depression",
        epsg=epsg_code,
        filter_values=[1],
        area_threshold=area_threshold,
    )
    depression_map_raster = None
    gc.collect()
    return out_depressions_path


def create_new_inlets(
    dem, depression_vector_path, gdf_roads, interval, out_file, epsg, cell_size
):
    """
    Densify inlets in previous version
    """
    new_inlets = []
    gdf_depressions = Vector.load_vector(depression_vector_path)
    gdf_depressions = gdf_depressions[gdf_depressions.geometry.area < 500000]
    gdf_depressions = gdf_depressions.explode(ignore_index=True)
    gdf_depressions_roads = gpd.sjoin(
        gdf_depressions, gdf_roads, how="inner", predicate="intersects"
    )
    gdf_depressions_roads.drop(columns=["index_right"], inplace=True)
    gdf_roads = gdf_roads[
        gdf_roads.geometry.apply(
            lambda geom: geom.geom_type in ["LineString", "MultiLineString"]
        )
    ]
    gdf_depressions = gdf_depressions[
        gdf_depressions.geometry.apply(
            lambda geom: geom.geom_type in ["Polygon", "MultiPolygon"]
        )
    ]
    gdf_roads_depressions = gpd.overlay(gdf_roads, gdf_depressions, how="intersection")
    # Retain only road segments longer than 50 meters
    gdf_roads_depressions = gdf_roads_depressions[
        gdf_roads_depressions.geometry.length > 10
    ]
    gdf_depressions_roads = gdf_depressions_roads.dissolve().explode()
    gdf_depressions_roads.reset_index(drop=True, inplace=True)
    # this creates new points along roads specifically where it intersects with depressions
    road_subset = gdf_roads_depressions[
        gdf_roads_depressions.geometry.apply(
            lambda geom: geom.geom_type in ["LineString", "MultiLineString"]
        )
    ]
    lines = road_subset.geometry.values.tolist()
    inlets_depression = []
    for line in lines:
        distances = np.arange(interval, line.length, interval)
        distances = np.append(distances, line.length)
        points = [line.interpolate(distance) for distance in distances]
        inlets_depression.extend(points)

    inlets_depression = gpd.GeoDataFrame(geometry=inlets_depression, crs=f"EPSG:{epsg}")
    inlets_depression.geometry = inlets_depression.geometry.buffer(cell_size)
    inlets_depression = inlets_depression.dissolve()
    inlets_depression = inlets_depression.explode(ignore_index=True)
    inlets_depression.geometry = inlets_depression.geometry.centroid
    if "index_right" in inlets_depression.columns:
        inlets_depression.drop(columns=["index_right"], inplace=True)
    if "index_right" in gdf_depressions_roads.columns:
        gdf_depressions_roads.drop(columns=["index_right"], inplace=True)
    inlets_depression = gpd.sjoin(
        inlets_depression,
        gdf_depressions_roads,
        how="inner",
        predicate="intersects",
    )  # retain only points within depressions!!
    inlets_depression = inlets_depression.reset_index()
    inlets_depression["pointid"] = inlets_depression.index
    new_inlets.append(inlets_depression)
    gdf_inlets_new = pd.concat(new_inlets)
    gdf_inlets_new = SampleRaster(
        gdf_inlets_new, dem, output=None, col_name="dtm_value", multiprocess=False
    ).process_whole()

    gdf_inlets_new = gdf_inlets_new.reset_index(drop=True)
    gdf_inlets_new.sort_values(by="dtm_value", inplace=True)
    gdf_inlets_new.to_file(out_file)
    return out_file


def filter_inlets(inlets: gpd.GeoDataFrame, cell_size: float, increase_capacity=False):
    if isinstance(inlets, str):
        gdf = gpd.read_file(inlets)
    else:
        gdf = inlets.copy()
    gdf = gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    if "pointid" not in gdf.columns:
        gdf["pointid"] = gdf.index

    if increase_capacity:
        if "Number_of" not in gdf.columns:
            gdf["Number_of"] = 1
        else:
            gdf["Number_of"] = gdf["Number_of"].fillna(1)

    original_count = len(gdf)
    print(f"Starting with {original_count} inlets.")
    gdf = gdf.sort_values("dtm_value", ascending=True).reset_index(drop=True)
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    tree = cKDTree(coords)
    removed = [False] * len(gdf)
    capacity_map = gdf["Number_of"].values.copy() if increase_capacity else None
    threshold = cell_size * 2.5
    for i in range(len(gdf)):
        if removed[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], threshold)
        for n_idx in neighbors:
            if n_idx == i or removed[n_idx]:
                continue
            removed[n_idx] = True
            if increase_capacity:
                capacity_map[i] += capacity_map[n_idx]

    if increase_capacity:
        gdf["Number_of"] = capacity_map

    gdf_filtered = gdf[~pd.Series(removed)].copy()

    print(
        f"Filtered to {len(gdf_filtered)} inlets. (Removed {original_count - len(gdf_filtered)})"
    )
    return gdf_filtered


def filter_inlets_deprecated(
    inlets: gpd.GeoDataFrame, cell_size: float, increase_capacity=False
):
    if isinstance(inlets, str):
        gdf_inlets = gpd.read_file(inlets)
    else:
        gdf_inlets = inlets
    gdf_inlets = gdf_inlets.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    inlet_columns = ["pointid", "geometry", "dtm_value"]
    if increase_capacity:
        inlet_columns.append("Number_of")
        if "Number_of" not in gdf_inlets.columns:
            gdf_inlets["Number_of"] = 1
    print(f"Starting with {len(gdf_inlets)} inlets.")
    iteration = 0
    while True:
        iteration += 1
        protected_ids = set()
        to_remove_ids = set()
        add_values = {}
        gdf_paired = gpd.sjoin_nearest(
            gdf_inlets,
            gdf_inlets,
            how="inner",
            distance_col="dist",
            max_distance=cell_size * 2.5,
            exclusive=True,
        )
        print(f"\t\t Resolving: {len(gdf_paired.index)}")

        gdf_paired = gdf_paired[gdf_paired["dist"] < cell_size * 2.5]
        gdf_paired.sort_values("dtm_value_left", ascending=True, inplace=True)
        if len(gdf_paired) == 0:
            print("No more close inlets found.")
            break
        if iteration > 20:
            to_remove_ids = set(gdf_paired["pointid_right"].unique())
            gdf_inlets = gdf_inlets[~gdf_inlets["pointid"].isin(to_remove_ids)].copy()
            continue
        for idx, row in gdf_paired.iterrows():
            id_left = row["pointid_left"]
            id_right = row["pointid_right"]
            dtm_value_left = row["dtm_value_left"]
            dtm_value_right = row["dtm_value_right"]
            if dtm_value_left <= dtm_value_right:
                if id_right not in protected_ids and id_right not in to_remove_ids:
                    if increase_capacity:
                        num_right = row["Number_of_right"]
                        if id_left in add_values:
                            add_values[id_left] += num_right
                        else:
                            add_values[id_left] = num_right
                    to_remove_ids.add(id_right)
                    if id_left not in protected_ids:
                        protected_ids.add(id_left)
        gdf_inlets = gdf_inlets[~gdf_inlets["pointid"].isin(to_remove_ids)].copy()
        if increase_capacity:
            for pid, add_val in add_values.items():
                gdf_inlets.loc[gdf_inlets["pointid"] == pid, "Number_of"] += add_val
    print(f"Filtered to {len(gdf_inlets)} inlets.")
    return gdf_inlets


def filter_inlets_deprecated(inlets, cell_size, increase_capacity=False):
    """
    TODO: Filtration based on spatial join (suggesting: by pair, remove always one with lower DTM value)

    """
    if isinstance(inlets, str):
        gdf_inlets = gpd.read_file(inlets)
    else:
        gdf_inlets = inlets
    inlet_columns = ["pointid", "geometry", "dtm_value"]
    inlet_list = []
    counts = []
    gdf_inlets = gdf_inlets.dropna(subset=["dtm_value"])
    gdf_buffered = gdf_inlets.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(cell_size * 1, cap_style=3)
    gdf_dissolved = gdf_buffered.dissolve().explode()
    gdf_dissolved = gdf_dissolved[["pointid", "geometry"]].reset_index(drop=True)
    gdf_inlets = gdf_inlets[inlet_columns].reset_index(drop=True)
    gdf_inlets_joined = gpd.sjoin(
        gdf_inlets, gdf_dissolved, predicate="intersects", how="inner"
    )
    unique_clusters = gdf_inlets_joined["index_right"].unique()

    for cluster_id in unique_clusters:
        gdf_cluster = gdf_inlets_joined[gdf_inlets_joined["index_right"] == cluster_id]
        lowest_point = gdf_cluster.loc[gdf_cluster["dtm_value"].idxmin()]
        points = len(gdf_cluster.index)
        points = int(points) - 1
        counts.append(points)
        inlet_list.append(lowest_point)
    if inlet_list:
        gdf_inlets_filtered = pd.DataFrame(inlet_list)
        gdf_inlets_filtered = gpd.GeoDataFrame(
            gdf_inlets_filtered, geometry="geometry", crs=gdf_inlets.crs
        )
        gdf_inlets_filtered = gdf_inlets_filtered.drop(
            columns=["index_right", "pointid_right"]
        )
        gdf_inlets_filtered = gdf_inlets_filtered.rename(
            columns={"pointid_left": "pointid"}
        )
        gdf_inlets_filtered = gdf_inlets_filtered[inlet_columns]
        if increase_capacity:
            gdf_inlets_filtered["Number_of"] = counts
            gdf_inlets_filtered["Number_of"] = gdf_inlets_filtered["Number_of"]
        else:
            gdf_inlets_filtered["Number_of"] = 0

    else:
        gdf_inlets_filtered = gpd.GeoDataFrame(
            columns=inlet_columns, geometry="geometry", crs=gdf_inlets.crs
        )
    print(f"\tNumber of inlets AFTER clustering: {len(gdf_inlets_filtered.index)}")
    return gdf_inlets_filtered


def extract_window(
    band, px, py, window_size=3, safe_mode=False, mask_value=-999999
) -> np.ndarray:
    """
    Extracts a window around a pixel in a raster band.
    """
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
            return np.full((window_size, window_size), mask_value)
    return window


def create_builtup_area_mask(
    materials_file, extent, threshold, overlap, window_size, epsg
) -> gpd.GeoDataFrame:
    # materials handling copy-pasted from pluvial grid selection
    """
    Filters selected or all inlets based on landuse raster (materials)
    """
    import tqdm

    DRAINAGE_CODES = [
        50,
        51,
        58,
    ]  # drainage codes for industrial, buildings and parking lots - Canada Flood
    cell_size = Raster.get_raster_info(materials_file, info="cell_size")[
        "cell_size"
    ]  # expected that materials are on 10 m
    """extent, cell_size, block_size, overlap"""
    fishnet = Geometry.create_fishnet(
        extent=extent, cell_size=cell_size, block_size=window_size, overlap=overlap
    )
    ds_materials = gdal.Open(materials_file)
    band_materials = ds_materials.GetRasterBand(1)
    gt_materials = ds_materials.GetGeoTransform()
    gdf_densenet = gpd.GeoDataFrame(geometry=fishnet, crs=epsg)
    for idx_dense, row_dense in tqdm.tqdm(
        gdf_densenet.iterrows(), total=len(gdf_densenet)
    ):
        centroid = row_dense.geometry.centroid
        x_start_s, y_start_s = Raster.get_nearest_cell_center(
            gt_materials, point_x=centroid.x, point_y=centroid.y
        )
        x_start_l = int((x_start_s - gt_materials[0]) / gt_materials[1])
        y_start_l = int((y_start_s - gt_materials[3]) / gt_materials[5])
        array_materials = extract_window(
            band_materials,
            px=x_start_l,
            py=y_start_l,
            window_size=window_size,
            safe_mode=True,
            mask_value=0,
        )
        array_materials = np.isin(array_materials, DRAINAGE_CODES).astype(int)
        pct_materials = np.sum(array_materials) / array_materials.size
        gdf_densenet.at[idx_dense, "Materials"] = pct_materials
    gdf_densenet = gdf_densenet[gdf_densenet["Materials"] > threshold]
    gdf_densenet = gdf_densenet.dissolve().explode(ignore_index=True)

    return gdf_densenet


def split_lines_points(gdf_lines, gdf_points):
    split_lines = []
    for idx, line in gdf_lines.iterrows():
        touching_points = gdf_points[gdf_points.geometry.intersects(line.geometry)]
        if not touching_points.empty:
            multi_point = touching_points.geometry.union_all()
            splitted = split(line.geometry, multi_point)
            for segment in splitted.geoms if hasattr(splitted, "geoms") else [splitted]:
                props = {k: line[k] for k in gdf_lines.columns if k != "geometry"}
                split_lines.append({"geometry": segment, **props})
        else:
            props = {k: line[k] for k in gdf_lines.columns}
            split_lines.append(props)

    gdf_split_lines = gpd.GeoDataFrame(split_lines, crs=gdf_lines.crs)
    gdf_split_lines.reset_index(drop=True, inplace=True)
    return gdf_split_lines


def get_line_intersections(gdf_lines):
    """
    Extracts all intersections from provided line dataset
    """
    bounds = gdf_lines.total_bounds
    extent = bounds[0], bounds[2], bounds[1], bounds[3]
    fishnet = Geometry.create_fishnet(
        extent=extent, cell_size=1, block_size=1000, overlap=100
    )
    intersections = []
    for fish in fishnet:
        subset = gdf_lines.cx[
            fish.bounds[0] : fish.bounds[2], fish.bounds[1] : fish.bounds[3]
        ]
        lines = subset.geometry.tolist()
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                inter = line1.intersection(line2)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    intersections.append(inter)
                elif inter.geom_type == "MultiPoint":
                    intersections.extend([pt for pt in inter.geoms])
    intersections = list(set(intersections))  # Remove duplicates
    gdf_intersections = gpd.GeoDataFrame(geometry=intersections, crs=gdf_lines.crs)
    return gdf_intersections
