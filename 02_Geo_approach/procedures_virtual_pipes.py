from gis import Vector
from helpers import Parameters
import warnings
import os
import datetime
import pandas as pd
from osgeo import gdal
import sys
import geopandas as gpd
from math import pi, sqrt
from osgeo import ogr, gdal
from gis import Raster, Vector
import numpy as np
from shapely.geometry import Point
from ifgis.raster import SampleRaster

from ifgis.raster import extract_by_mask_rasterized
from lib_00_vp_processing_script import (
    filter_depressions,
    move_inlets,
    move_outlets,
    create_outlets,
    create_inlets_selection,
    get_points_along_geometry,
    connect,
    clip_layer_by_domain,
    get_depressions,
    create_new_inlets,
    filter_inlets,
    extract_inlets,
    remove_outlets_buildings,
    create_builtup_area_mask,
    filter_roads,
    get_line_intersections,
    split_lines_points,
    snap_to_min,
)


class VirtualPipes:
    """
    Class for virtual pipes creation for pluvial flood modelling of cities.
    TODO: Filtration task will become iterative pairwise process removing always one of the points and increasing the capacity of the other
    """

    def __init__(self):
        self.yaml_path = None
        self.general_parameters = None
        self.virtual_pipes_parameters = None
        self.waterbodies_parameters = None
        self.waterbodies_path = None
        self.dtm_path = None
        self.domain_name = None
        self.mask_path = None
        self.vp_processing = None
        self.results_dir = None

    def set_general_parameters(self, path_yaml):
        """Set General parameters from a YAML file."""
        self.yaml_path = path_yaml
        self.general_parameters = Parameters.load_local_parameters(
            self.yaml_path, "general_parameters"
        )

    def set_virtual_pipes_parameters(self, path_yaml):
        """Set Virtual Pipes parameters from a YAML file."""
        self.yaml_path = path_yaml
        self.virtual_pipes_parameters = Parameters.load_local_parameters(
            self.yaml_path, "virtual_pipes_parameters"
        )

    def set_waterbodies_parameters(self, path_yaml):
        """Set Waterbodies parameters from a YAML file."""
        self.yaml_path = path_yaml
        self.waterbodies_parameters = Parameters.load_local_parameters(
            self.yaml_path, "waterbodies_parameters"
        )

    def set_projection(self):
        if self.general_parameters is None:
            raise ValueError(
                "General parameters not set. Run set_general_parameters() first."
            )
        self.projection = Parameters.get_local_parameter(
            self.general_parameters, "default_epsg_code"
        )

    def set_waterbodies_path(self, path):
        self.waterbodies_path = path

    def set_rn_path(self, path):
        self.rn_path = path

    def set_tunnels_path(self, path):
        self.tunnels_path = path

    def set_dtm_path(self, path):
        self.dtm_path = path

    def set_domain_name(self, name):
        self.domain_name = name

    def set_results_dir(self, path):
        self.results_dir = path

    def set_mask_path(self, path):
        self.mask_path = path

    def create_virtual_pipes(self):
        ogr_drivers = [ogr.GetDriver(i).GetName() for i in range(ogr.GetDriverCount())]
        memory_driver = None
        if "MEM" in ogr_drivers:
            memory_driver = "MEM"
        elif "Memory" in ogr_drivers:
            memory_driver = "Memory"
        else:
            raise RuntimeError("No in-memory vector driver found in GDAL/OGR.")
        print(f"\nCreating Virtual Pipes for {self.domain_name}")
        # FIRST CREATE CLIP FILE TO self.domain_name
        self.projection = Parameters.get_local_parameter(
            self.general_parameters, "default_epsg_code"
        )
        in_folder = os.path.join(self.results_dir)
        out_folder = os.path.join(in_folder, "model", "gis")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        code_file = os.path.join(out_folder, f"2d_clip_{self.domain_name}_R.gpkg")
        gdf_mask = Vector.load_vector(self.mask_path)
        if gdf_mask.crs.to_epsg() != self.projection:
            gdf_mask = gdf_mask.to_crs(epsg=self.projection)
        gdf_mask.to_file(code_file, driver="GPKG", index=False)
        # BY-PASS ALL HARDCODED PATHS - LOAD FROM PARAMETERS
        RN = self.rn_path
        TUNNELS = self.tunnels_path
        WB = self.waterbodies_path
        DTM = self.dtm_path
        ROADS = self.virtual_pipes_parameters.get("roads", None)
        INDUSTRIAL_ESTATES = self.virtual_pipes_parameters.get("industrial_estates")
        BUILDINGS = self.virtual_pipes_parameters.get("buildings", None)
        PARKING = self.virtual_pipes_parameters.get("parking", None)
        MANNING = self.virtual_pipes_parameters.get("manning", None)
        ADDRESS_POINTS = self.virtual_pipes_parameters.get("address_points", None)
        CELL_SIZE = self.virtual_pipes_parameters.get("cell_size", None)
        interval_inlets = self.virtual_pipes_parameters.get("interval_inlets", None)
        interval_outlets = self.virtual_pipes_parameters.get("interval_outlets", None)
        FIELD_FOR_FILTERING = self.virtual_pipes_parameters.get(
            "field_for_filtering", None
        )
        SNAP_INLET_TO_LOWEST = self.virtual_pipes_parameters.get(
            "snap_inlet_to_lowest", None
        )
        FILTER_BUILDINGS = self.virtual_pipes_parameters.get("filter_buildings", None)
        filter_a = self.virtual_pipes_parameters.get("filter_a", None)
        filter_b = self.virtual_pipes_parameters.get("filter_b", None)
        filter_c = self.virtual_pipes_parameters.get("filter_c", None)

        required_vars = {
            "WB": WB,
            "DTM": DTM,
            "RN": RN,
            "ROADS": ROADS,
            "INDUSTRIAL_ESTATES": INDUSTRIAL_ESTATES,
            "BUILDINGS": BUILDINGS,
            "PARKING": PARKING,
            "MANNING": MANNING,
            "ADDRESS_POINTS": ADDRESS_POINTS,
            "CELL_SIZE": CELL_SIZE,
            "interval_inlets": interval_inlets,
            "interval_outlets": interval_outlets,
            "FIELD_FOR_FILTERING": FIELD_FOR_FILTERING,
            "SNAP_INLET_TO_LOWEST": SNAP_INLET_TO_LOWEST,
            "FILTER_BUILDINGS": FILTER_BUILDINGS,
            "filter_a": filter_a,
            "filter_b": filter_b,
            "filter_c": filter_c,
        }

        def shift_point_randomly(point, max_shift=CELL_SIZE):
            import shapely
            import random

            # Only allow shifts in cardinal directions: left, right, up, down
            direction = random.choice(["left", "right", "up", "down"])
            if direction == "left":
                dx, dy = -max_shift, 0
            elif direction == "right":
                dx, dy = max_shift, 0
            elif direction == "up":
                dx, dy = 0, max_shift
            else:  # 'down'
                dx, dy = 0, -max_shift
            return shapely.geometry.Point(point.x + dx, point.y + dy)

        missing = [name for name, value in required_vars.items() if value is None]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        date = datetime.datetime.now().strftime("%y%m%d")
        out_name = f"1d_pit_{self.domain_name}_{str(CELL_SIZE)}m_{date}_P.gpkg"

        if os.path.isfile(code_file) and not os.path.isfile(
            os.path.join(out_folder, out_name)
        ):
            out_part_layers = os.path.join(in_folder, "model", "gis", "virtual_pipes")
            os.makedirs(out_part_layers, exist_ok=True)

            VP_TEMP_DIR = self.virtual_pipes_parameters.get("temp_vp_dir", r"D:\temp\VPs")
            temp = os.path.join(VP_TEMP_DIR, "temp")
            os.makedirs(temp, exist_ok=True)
            roads = clip_layer_by_domain(
                ROADS, code_file, os.path.join(temp, f"roads_{self.domain_name}.gpkg")
            )

            rn = clip_layer_by_domain(
                RN, code_file, os.path.join(temp, f"rn_{self.domain_name}.gpkg")
            )
            gdf_rn = gpd.read_file(rn)
            # Retain only values where fclass is "river", "stream", or NULL
            if "fclass" in gdf_rn.columns:
                gdf_rn = gdf_rn[
                    gdf_rn["fclass"].isin(["river", "stream"])
                    | gdf_rn["fclass"].isnull()
                ]
                gdf_rn.to_file(rn, driver="GPKG", index=False)
            wb = clip_layer_by_domain(
                WB, code_file, os.path.join(temp, f"wb_{self.domain_name}.gpkg")
            )
            dtm = extract_by_mask_rasterized(
                DTM, code_file, os.path.join(temp, f"dtm_{self.domain_name}.tif")
            )
            print("\tExtracting depressions")
            start = datetime.datetime.now()
            depressions_path = get_depressions(dtm)
            out_depressions_path = rf"D:\90_PersonalFoldlers\JZa\Workspaces\Pluvial\cities\_depressions\depressions_{self.domain_name}.gpkg"
            filtered_depressions_path = filter_depressions(
                depressions_path=depressions_path,
                out_depressions_path=out_depressions_path,
                area_threshold=250,
            )
            # add new inlets to depressions
            print(f"\tFiltered depressions saved to {filtered_depressions_path}")

            def pp_compactness(geom):
                p = geom.length
                a = geom.area
                return (4 * pi * a) / (p * p)

            gdf_depressions_filtered = Vector.load_vector(filtered_depressions_path)
            gdf_depressions_filtered["compactness"] = pp_compactness(
                gdf_depressions_filtered.geometry
            )
            # Retain depressions with area < 1500 and compactness > 0.25
            gdf_depressions_filtered = gdf_depressions_filtered[
                (gdf_depressions_filtered.geometry.area < 30000)
                & (gdf_depressions_filtered["compactness"] > 0.1)
            ]
            gdf_wb = Vector.load_vector(wb)
            # Remove depressions that intersect with waterbodies
            gdf_depressions_filtered = gpd.sjoin(
                gdf_depressions_filtered, gdf_wb, how="left", predicate="intersects"
            )
            gdf_depressions_filtered = gdf_depressions_filtered[
                gdf_depressions_filtered.index_right.isna()
            ]
            gdf_depressions_filtered = gdf_depressions_filtered.drop(
                columns=["index_right"]
            )
            bbox_depress = gdf_depressions_filtered.total_bounds
            gdf_address_points = Vector.load_vector(
                ADDRESS_POINTS, bbox=tuple(bbox_depress)
            )
            gdf_depressions_filtered = gpd.sjoin_nearest(
                gdf_depressions_filtered,
                gdf_address_points,
                how="left",
                distance_col="dist",
            )
            gdf_depressions_filtered = gdf_depressions_filtered[
                gdf_depressions_filtered["dist"] < 175
            ]
            gdf_depressions_filtered["geom_wkt"] = (
                gdf_depressions_filtered.geometry.to_wkt()
            )
            gdf_depressions_filtered = gdf_depressions_filtered.drop_duplicates(
                subset=["geom_wkt"]
            )
            gdf_depressions_filtered = gdf_depressions_filtered.drop(
                columns=["geom_wkt"]
            )
            dtm_cell_size = Raster.get_raster_info(dtm, ["cell_size"])["cell_size"]
            gdf_depressions_filtered.reset_index(drop=True, inplace=True)
            print(
                f"\t Processing {len(gdf_depressions_filtered.index)} depressions for inlet densification"
            )
            gdf_depressions_filtered.to_file(
                os.path.join(temp, f"00_depressions_filtered_final_{self.domain_name}.gpkg")
            )
            # Get DTM info once outside the loop for efficiency
            dtm_geotransform = Raster.get_raster_info(dtm, ["geotransform"])[
                "geotransform"
            ]
            dtm_dt = Raster.get_raster_info(dtm, ["data_type"])["data_type"]

            points = []
            values = []
            """
            ADD INLETS IN DEPRSSION CLOSE TO BUILDINGS
            """
            for idx_depression, row_depression in gdf_depressions_filtered.iterrows():
                # Get depression bounds in standard format (minx, miny, maxx, maxy)
                bounds_depression = row_depression.geometry.bounds
                extent_depression = [
                    bounds_depression[0],
                    bounds_depression[2],
                    bounds_depression[1],
                    bounds_depression[3],
                ]

                extent_snapped = Vector.align_extent_to_snap(
                    extent_depression,
                    snap_raster=dtm,
                    cell_size=CELL_SIZE,
                )
                # Extend snap extent by 50 cells (100m if CELL_SIZE=2)
                extent_snapped = [
                    extent_snapped[0] - 50 * CELL_SIZE,
                    extent_snapped[1] + 50 * CELL_SIZE,
                    extent_snapped[2] - 50 * CELL_SIZE,
                    extent_snapped[3] + 50 * CELL_SIZE,
                ]

                # Create in-memory datasource for depression geometry
                drv = ogr.GetDriverByName(memory_driver)
                dst_ds = drv.CreateDataSource("in_memory")
                dest_srs = ogr.osr.SpatialReference()
                dest_srs.ImportFromEPSG(self.projection)

                dst_lyr = dst_ds.CreateLayer(
                    "depression", dest_srs, geom_type=ogr.wkbPolygon
                )
                geom = ogr.CreateGeometryFromWkt(
                    str(gdf_depressions_filtered["geometry"].values[idx_depression])
                )

                feature = ogr.Feature(dst_lyr.GetLayerDefn())
                feature.SetGeometry(geom)
                dst_lyr.CreateFeature(feature)

                # Clip DTM to depression bounds
                bounds_snapped = [
                    extent_snapped[0],
                    extent_snapped[2],
                    extent_snapped[1],
                    extent_snapped[3],
                ]

                dtm_clip = Raster.clip_by_vector_mem(
                    in_raster=dtm,
                    vector_file=dst_ds,
                    epsg_out=self.projection,
                    cell_size_out=dtm_cell_size,
                    data_type=dtm_dt,
                    CPU_AVAILABLE=4,
                    no_data=0,
                    bounds=bounds_snapped,
                )

                # Rasterize depression polygon
                r = Raster()
                geos_depression_selected_raster = r.rasterize_to_new_raster(
                    None,
                    dst_ds,
                    value=1,
                    cell_size=dtm_cell_size,
                    extent=extent_snapped,
                    format="MEM",
                    nodata=0,
                    data_type=gdal.GDT_Byte,
                    all_touched=False,
                )

                # Convert to arrays for analysis
                dtm_depression_array = Raster.to_array_mem(dtm_clip)
                depression_array = Raster.to_array_mem(geos_depression_selected_raster)
                # Find minimum DTM value within depression polygon
                if np.any(depression_array == 1):
                    mask = depression_array == 1
                    depression_area = row_depression.geometry.area

                    # Get values within depression
                    masked_values = dtm_depression_array[mask]

                    # Determine how many lowest points to extract
                    num_points = (
                        5
                        if 750 < depression_area <= 1500
                        else (10 if depression_area > 1500 else 1)
                    )
                    num_points = min(num_points, len(masked_values))

                    # Find indices of the lowest values
                    lowest_indices = np.argsort(masked_values)[:num_points]

                    # Get geotransform for coordinate conversion
                    gt = Raster.get_raster_info_mem(dtm_clip, ["geotransform"])[
                        "geotransform"
                    ]

                    # Process each of the lowest points
                    for rank, masked_idx in enumerate(lowest_indices):
                        # Find the actual array indices for this masked value
                        array_indices = np.where(mask)
                        min_index = (
                            array_indices[0][masked_idx],
                            array_indices[1][masked_idx],
                        )

                        row, col = min_index
                        min_dtm_value = dtm_depression_array[row, col]
                        min_x = gt[0] + col * gt[1] + (row + 1) * gt[2] + (gt[1] / 2)
                        min_y = gt[3] + col * gt[4] + (row + 1) * gt[5] - (gt[5] / 2)
                        point = Point(min_x, min_y)
                        points.append(point)
                        values.append(min_dtm_value)

                # Clean up datasources to prevent memory leaks
                dst_lyr = None
                dst_ds = None

            gdf_inlets_depressions = gpd.GeoDataFrame(
                geometry=points, crs=self.projection
            )
            gdf_inlets_depressions["dtm_value"] = values
            gdf_inlets_depressions = gdf_inlets_depressions.drop_duplicates(
                subset=["geometry"]
            )
            gdf_inlets_depressions.reset_index(drop=True, inplace=True)
            gdf_inlets_depressions["pointid"] = gdf_inlets_depressions.index + 1
            gdf_inlets_depressions = gdf_inlets_depressions[
                ["pointid", "geometry", "dtm_value"]
            ]
            gdf_inlets_depressions = filter_inlets(
                gdf_inlets_depressions, cell_size=CELL_SIZE / 2, increase_capacity=False
            )
            gdf_inlets_depressions = filter_inlets(
                gdf_inlets_depressions, cell_size=CELL_SIZE, increase_capacity=False
            )

            inlets_depressions = os.path.join(VP_TEMP_DIR, f"USE01_Depression_{self.domain_name}.gpkg")

            gdf_inlets_depressions.to_file(
                inlets_depressions,
                driver="GPKG",
                index=False,
            )

            print("\tProcessing tunnels and underground RN sections")
            tunnels = clip_layer_by_domain(
                TUNNELS,
                code_file,
                os.path.join(temp, f"tunnels_{self.domain_name}.gpkg"),
            )
            gdf_tunnels = Vector.load_vector(tunnels)
            gdf_tunnels["type"] = "tunnel"
            # For each tunnel, extract vertices: all except last are inlets, last is outlet
            PRIORITY_INLETS = {}
            PRIORITY_OUTLETS = {}
            ds_dtm = gdal.Open(dtm)
            band_dem = ds_dtm.GetRasterBand(1)
            gdf_tunnels_rn = gpd.read_file(RN)
            gdf_tunnels_rn = gpd.sjoin(
                gdf_tunnels_rn, gdf_mask[["geometry"]], how="left", predicate="within"
            )
            # Filter gdf_tunnels_rn for tunnel=1 or underground=1 if those columns exist
            if "tunnel" in gdf_tunnels_rn.columns:
                gdf_tunnels_rn = gdf_tunnels_rn[gdf_tunnels_rn["tunnel"] == 1]
            elif "underground" in gdf_tunnels_rn.columns:
                gdf_tunnels_rn = gdf_tunnels_rn[gdf_tunnels_rn["underground"] == 1]
            gdf_tunnels_rn = Vector.merge_lines_on_pseudonodes(gdf_tunnels_rn)
            gdf_tunnels_rn = gdf_tunnels_rn[gdf_tunnels_rn.geometry.length > 25]
            gdf_tunnels_rn["type"] = "tunnel_rn"
            gdf_tunnels_rn.reset_index(drop=True, inplace=True)
            gdf_tunnels = pd.concat([gdf_tunnels, gdf_tunnels_rn], ignore_index=True)
            gdf_tunnels.reset_index(drop=True, inplace=True)
            gdf_tunnels_rn = gdf_tunnels_rn.iloc[0:0].copy()

            for _, row in gdf_tunnels.iterrows():
                geom = row.geometry
                type = row["type"]
                if geom is None or geom.is_empty or geom.geom_type != "LineString":
                    print(
                        "Warning: Invalid geometry in tunnels layer, skipping feature."
                    )
                    continue
                coords = list(geom.coords)
                if len(coords) < 2:
                    continue
                outlet_not_snapped = Point(coords[-1])
                if type == "tunnel":
                    x, y, z = snap_to_min(
                        band_dem, outlet_not_snapped, dtm_geotransform, size=5
                    )
                    if x is None or y is None or z is None:
                        continue
                    OUTLET_SNAPPED = Point(x, y)
                    DTM_VALUE_OUTLET = z

                    # Measure distance to each of PRIORITY_OUTLETS
                    if PRIORITY_OUTLETS:
                        outlet_keys = list(PRIORITY_OUTLETS.keys())
                        distances = [
                            OUTLET_SNAPPED.distance(PRIORITY_OUTLETS[k][0])
                            for k in outlet_keys
                        ]
                        min_dist = min(distances)
                        if min_dist < CELL_SIZE * 2.5:
                            closest_index = outlet_keys[distances.index(min_dist)]
                        else:
                            new_key = (
                                max(PRIORITY_OUTLETS.keys()) + 1
                                if PRIORITY_OUTLETS
                                else 0
                            )
                            PRIORITY_OUTLETS[new_key] = (
                                OUTLET_SNAPPED,
                                DTM_VALUE_OUTLET,
                            )
                            closest_index = new_key
                    else:
                        PRIORITY_OUTLETS[0] = (OUTLET_SNAPPED, DTM_VALUE_OUTLET)
                        closest_index = 0
                else:
                    x, y, z = snap_to_min(
                        band_dem, outlet_not_snapped, dtm_geotransform, size=7
                    )
                    if x is None or y is None or z is None:
                        continue
                    OUTLET_SNAPPED = Point(x, y)
                    DTM_VALUE_OUTLET = z

                    # Measure distance to each of PRIORITY_OUTLETS
                    if PRIORITY_OUTLETS:
                        outlet_keys = list(PRIORITY_OUTLETS.keys())
                        distances = [
                            OUTLET_SNAPPED.distance(PRIORITY_OUTLETS[k][0])
                            for k in outlet_keys
                        ]
                        min_dist = min(distances)
                        if min_dist < CELL_SIZE * 2.5:
                            closest_index = outlet_keys[distances.index(min_dist)]
                        else:
                            new_key = (
                                max(PRIORITY_OUTLETS.keys()) + 1
                                if PRIORITY_OUTLETS
                                else 0
                            )
                            PRIORITY_OUTLETS[new_key] = (
                                OUTLET_SNAPPED,
                                DTM_VALUE_OUTLET,
                            )
                            closest_index = new_key
                    else:
                        PRIORITY_OUTLETS[0] = (OUTLET_SNAPPED, DTM_VALUE_OUTLET)
                        closest_index = 0

                # Link all points except the last one to the priority outlet using a dictionary
                if type == "tunnel":
                    for pt in coords[:-1]:
                        inlet_point = Point(pt)
                        x, y, z = snap_to_min(
                            band_dem, inlet_point, dtm_geotransform, size=3
                        )
                        if x is None or y is None or z is None:
                            continue
                        inlet_point = Point(x, y)
                        DTM_VALUE_INLET = z
                        if PRIORITY_INLETS:
                            distances = [
                                inlet_point.distance(existing_inlet_key)
                                for existing_inlet_key in PRIORITY_INLETS.keys()
                            ]
                            min_dist = min(distances)
                            if min_dist > 2.5 * CELL_SIZE:
                                # Store both the outlet info and the inlet DTM value
                                PRIORITY_INLETS[inlet_point] = {
                                    "outlet": PRIORITY_OUTLETS[closest_index],
                                    "dtm_value_inlet": DTM_VALUE_INLET,
                                }
                            else:
                                print(
                                    "\t\tInlet close to existing priority inlet, skipping."
                                )
                        else:
                            # First inlet, add it directly
                            PRIORITY_INLETS[inlet_point] = {
                                "outlet": PRIORITY_OUTLETS[closest_index],
                                "dtm_value_inlet": DTM_VALUE_INLET,
                            }
                else:
                    x, y, z = snap_to_min(
                        band_dem, Point(coords[0]), dtm_geotransform, size=5
                    )
                    if x is None or y is None or z is None:
                        continue
                    inlet_point = Point(x, y)
                    DTM_VALUE_INLET = z
                    if PRIORITY_INLETS:
                        distances = [
                            inlet_point.distance(existing_inlet_key)
                            for existing_inlet_key in PRIORITY_INLETS.keys()
                        ]
                        min_dist = min(distances)
                        if min_dist > 2.5 * CELL_SIZE:
                            # Store both the outlet info and the inlet DTM value
                            PRIORITY_INLETS[inlet_point] = {
                                "outlet": PRIORITY_OUTLETS[closest_index],
                                "dtm_value_inlet": DTM_VALUE_INLET,
                            }
                        else:
                            print(
                                "\t\tInlet close to existing priority inlet, skipping."
                            )
                    else:
                        # First inlet, add it directly
                        PRIORITY_INLETS[inlet_point] = {
                            "outlet": PRIORITY_OUTLETS[closest_index],
                            "dtm_value_inlet": DTM_VALUE_INLET,
                        }

            print("\tProcessing RN tunnels/underground sections")

            for _, row in gdf_tunnels_rn.iterrows():
                geom = row.geometry
                if geom.geom_type == "MultiLineString":
                    geom = max(geom.geoms, key=lambda geom_part: geom_part.length)

                if geom is None or geom.is_empty or geom.geom_type != "LineString":
                    print(geom.geom_type)
                    print(
                        "Warning: Invalid geometry in tunnels layer, skipping feature."
                    )
                    continue
                coords = list(geom.coords)
                if len(coords) < 2:
                    continue

                # Snap both endpoints and determine direction by elevation
                x0, y0, z0 = snap_to_min(
                    band_dem, Point(coords[0]), dtm_geotransform, size=5
                )
                xn, yn, zn = snap_to_min(
                    band_dem, Point(coords[-1]), dtm_geotransform, size=5
                )
                if x0 is None or xn is None:
                    continue

                # Outlet = lower elevation end, Inlet = higher elevation end
                if z0 <= zn:
                    # First vertex is lower → it's the outlet, last is inlet
                    OUTLET_SNAPPED = Point(x0, y0)
                    DTM_VALUE_OUTLET = z0
                    inlet_point = Point(xn, yn)
                    DTM_VALUE_INLET = zn
                else:
                    # Last vertex is lower → it's the outlet, first is inlet
                    OUTLET_SNAPPED = Point(xn, yn)
                    DTM_VALUE_OUTLET = zn
                    inlet_point = Point(x0, y0)
                    DTM_VALUE_INLET = z0

                # Skip if outlet is not meaningfully lower than inlet
                if DTM_VALUE_OUTLET >= DTM_VALUE_INLET:
                    continue

                # Measure distance to each of PRIORITY_OUTLETS
                if PRIORITY_OUTLETS:
                    outlet_keys = list(PRIORITY_OUTLETS.keys())
                    distances = [
                        OUTLET_SNAPPED.distance(PRIORITY_OUTLETS[k][0])
                        for k in outlet_keys
                    ]
                    min_dist = min(distances)
                    if min_dist < CELL_SIZE * 2.5:
                        closest_index = outlet_keys[distances.index(min_dist)]
                    else:
                        new_key = max(PRIORITY_OUTLETS.keys()) + 1
                        PRIORITY_OUTLETS[new_key] = (OUTLET_SNAPPED, DTM_VALUE_OUTLET)
                        closest_index = new_key
                else:
                    new_key = (
                        max(PRIORITY_OUTLETS.keys()) + 1 if PRIORITY_OUTLETS else 0
                    )
                    PRIORITY_OUTLETS[new_key] = (OUTLET_SNAPPED, DTM_VALUE_OUTLET)
                    closest_index = new_key

                # Check inlet proximity to existing inlets
                if PRIORITY_INLETS:
                    distances = [
                        inlet_point.distance(existing_inlet_key)
                        for existing_inlet_key in PRIORITY_INLETS.keys()
                    ]
                    min_dist = min(distances)
                    if min_dist <= 2.5 * CELL_SIZE:
                        continue  # too close to existing inlet, skip

                if closest_index in PRIORITY_OUTLETS:
                    PRIORITY_INLETS[inlet_point] = {
                        "outlet": PRIORITY_OUTLETS[closest_index],
                        "dtm_value_inlet": DTM_VALUE_INLET,
                    }

            gdf_priority_inlets = gpd.GeoDataFrame(
                geometry=list(PRIORITY_INLETS.keys()), crs=3979
            )
            gdf_priority_inlets["INLET_ID"] = gdf_priority_inlets.index + 1
            # Create reverse mapping from outlet tuple to its ID
            outlet_to_id = {id(v): k for k, v in PRIORITY_OUTLETS.items()}
            # Extract outlet geometry and DTM value for each inlet
            gdf_priority_inlets["OutletID"] = [
                next(
                    (k for k, v in PRIORITY_OUTLETS.items() if v == inlet["outlet"]),
                    None,
                )
                for inlet in PRIORITY_INLETS.values()
            ]
            gdf_priority_inlets["ZIn"] = [
                v["dtm_value_inlet"] for v in PRIORITY_INLETS.values()
            ]
            gdf_priority_inlets["ZOut"] = [
                v["outlet"][1] for v in PRIORITY_INLETS.values()
            ]
            gdf_priority_outlets = gpd.GeoDataFrame(
                geometry=[val[0] for val in PRIORITY_OUTLETS.values()],
                crs=3979,
            )
            gdf_priority_outlets["ZOut"] = [val[1] for val in PRIORITY_OUTLETS.values()]
            gdf_priority_outlets["OutletID"] = gdf_priority_outlets.index

            if not gdf_priority_inlets.empty and not gdf_priority_outlets.empty:
                joined = gpd.sjoin_nearest(
                    gdf_priority_outlets,
                    gdf_priority_inlets[["INLET_ID", "geometry", "OutletID"]],
                    how="left",
                    distance_col="dist_to_inlet",
                    max_distance=3 * CELL_SIZE,
                )

                # Keep only close matches
                close_pairs = joined[joined["dist_to_inlet"] <= 15]
                close_pairs = close_pairs[["OutletID_left", "OutletID_right"]]
                close_pairs = close_pairs.sort_values(by="OutletID_left")

                close_pairs.rename(
                    columns={
                        "OutletID_left": "flows_from",
                        "OutletID_right": "flows_to",
                    },
                    inplace=True,
                )

                # Initial mapping: from outlet -> nearer outlet
                mapping = dict(zip(close_pairs["flows_from"], close_pairs["flows_to"]))

                def find_final_dest(node, mapping, cache):
                    if node in cache:
                        return cache[node]
                    visited = set()
                    current = node
                    while current in mapping and current not in visited:
                        visited.add(current)
                        nxt = mapping[current]
                        # protect against NaNs
                        if pd.isna(nxt):
                            break
                        current = nxt
                    cache[node] = current
                    return current

                # Collapse chains: A->B->C becomes A->C, B->C
                cache = {}
                close_pairs["flows_to"] = close_pairs["flows_from"].apply(
                    lambda x: find_final_dest(x, mapping, cache)
                )
                mapping_final = dict(
                    zip(close_pairs["flows_from"], close_pairs["flows_to"])
                )

                # --- NEW PART 1: ensure mapping_final only maps to existing outlets ---
                # existing outlets BEFORE removal of 'flows_from'
                remaining_outlet_ids = set(gdf_priority_outlets["OutletID"].unique())

                mapping_final = {
                    src: dst
                    for src, dst in mapping_final.items()
                    if pd.notna(dst) and dst in remaining_outlet_ids
                }
                # ----------------------------------------------------------------------

                # Apply mapping to priority inlets (remap OutletID to final destination)
                gdf_priority_inlets["OutletID"] = gdf_priority_inlets["OutletID"].apply(
                    lambda x: mapping_final.get(x, x)
                )

                # Remove only the original 'from' outlets (sources of chains)
                gdf_priority_outlets = gdf_priority_outlets.loc[
                    ~gdf_priority_outlets["OutletID"].isin(
                        close_pairs["flows_from"].values
                    )
                ]

                # --- NEW PART 2: remap inlets that still point to non-existing outlets ---
                # After removal above, recompute valid outlet IDs
                valid_outlet_ids = set(gdf_priority_outlets["OutletID"].unique())

                # Inlets whose OutletID is NOT in the final outlets
                mask_orphan = ~gdf_priority_inlets["OutletID"].isin(valid_outlet_ids)
                orphans = gdf_priority_inlets.loc[mask_orphan].copy()

                if not orphans.empty:
                    # Join orphan inlets to nearest remaining outlets
                    remap_join = gpd.sjoin_nearest(
                        orphans[["INLET_ID", "geometry", "OutletID"]],
                        gdf_priority_outlets[["OutletID", "geometry"]],
                        how="left",
                        distance_col="dist_to_outlet",
                    )

                    # For each orphan inlet INLET_ID, take nearest OutletID_right
                    remap_pairs = remap_join[["INLET_ID", "OutletID_right"]].dropna(
                        subset=["OutletID_right"]
                    )
                    remap_pairs = remap_pairs.rename(
                        columns={"OutletID_right": "new_OutletID"}
                    )

                    # Build mapping from INLET_ID -> new OutletID
                    orphan_remap = dict(
                        zip(remap_pairs["INLET_ID"], remap_pairs["new_OutletID"])
                    )

                    # Apply back to the full gdf_priority_inlets, keyed by INLET_ID
                    gdf_priority_inlets["OutletID"] = gdf_priority_inlets.apply(
                        lambda row: orphan_remap.get(row["INLET_ID"], row["OutletID"]),
                        axis=1,
                    )
                # -----------------------------------------------------------------------
                # Debug check
                gdf_priority_inlets["OutletID"] = gdf_priority_inlets["OutletID"].apply(
                    lambda x: mapping_final.get(x, x)
                )

                # Remove only the original 'from' outlets (sources of chains)
                gdf_priority_outlets = gdf_priority_outlets[
                    ~gdf_priority_outlets["OutletID"].isin(
                        close_pairs["flows_from"].values
                    )
                ]
                gdf_priority_inlets = gdf_priority_inlets.drop_duplicates(
                    subset=["geometry"]
                )
                gdf_priority_inlets.reset_index(drop=True, inplace=True)
                gdf_priority_inlets["ID"] = gdf_priority_inlets.index + 1 + 5000000
                gdf_priority_inlets["ID"] = gdf_priority_inlets["ID"].astype(str)
                gdf_priority_inlets["Type"] = "I"
                gdf_priority_inlets["Inlet_Type"] = "StormOutletNo1"
                gdf_priority_inlets["VP_Network_ID"] = gdf_priority_inlets[
                    "OutletID"
                ].astype(int)
                gdf_priority_inlets["VP_Network_ID"] = (
                    gdf_priority_inlets["VP_Network_ID"] + 7000000
                )
                gdf_priority_inlets.sort_values(
                    by=["VP_Network_ID", "ZIn"], ascending=[True, False], inplace=True
                )
                gdf_priority_inlets["VP_Sur_Index"] = gdf_priority_inlets.groupby(
                    "VP_Network_ID"
                ).cumcount()
                gdf_priority_inlets["VP_QMax"] = 10.0
                gdf_priority_inlets["Width"] = 2.0
                gdf_priority_inlets["Conn_2D"] = "SX"
                gdf_priority_inlets["Conn_No"] = 4
                gdf_priority_inlets["pBlockage"] = 0.0
                gdf_priority_inlets["Number_of"] = 1
                gdf_priority_inlets["Lag_Approach"] = "None"
                gdf_priority_inlets["Lag_Value"] = 0.0
                gdf_priority_inlets["ZIn"] = gdf_priority_inlets["ZIn"].astype(int)
                gdf_priority_inlets["ZOut"] = gdf_priority_inlets["ZOut"].astype(int)
                gdf_priority_inlets["if_type"] = "tunnel"

                gdf_priority_outlets["VP_Network_ID"] = gdf_priority_outlets[
                    "OutletID"
                ].astype(int)
                gdf_priority_outlets["VP_Network_ID"] = (
                    gdf_priority_outlets["VP_Network_ID"] + 7000000
                )
                gdf_priority_outlets["ID"] = gdf_priority_outlets[
                    "VP_Network_ID"
                ].astype(str)
                gdf_priority_outlets["Type"] = "O"
                gdf_priority_outlets["Inlet_Type"] = "0"
                gdf_priority_outlets["VP_Sur_Index"] = 0
                gdf_priority_outlets["VP_QMax"] = 10.0
                gdf_priority_outlets["Width"] = 2.0
                gdf_priority_outlets["Conn_2D"] = "SX"
                gdf_priority_outlets["Conn_No"] = 0
                gdf_priority_outlets["pBlockage"] = 0.0
                gdf_priority_outlets["Number_of"] = 0
                gdf_priority_outlets["Lag_Approach"] = "None"
                gdf_priority_outlets["Lag_Value"] = 0.0
                gdf_priority_outlets["ZIn"] = None
                gdf_priority_outlets["if_type"] = "tunnel"

                gdf_priority_outlets = SampleRaster(
                    points=gdf_priority_outlets,
                    raster=dtm,
                    output=None,
                    col_name="ZOut",
                ).process_whole()
                gdf_priority_outlets = gdf_priority_outlets[
                    [
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
                        "if_type",
                        "geometry",
                    ]
                ]

                gdf_priority_inlets = SampleRaster(
                    points=gdf_priority_inlets,
                    raster=dtm,
                    output=None,
                    col_name="ZIn",
                ).process_whole()
                gdf_priority_inlets["ZIn"] = (
                    gdf_priority_inlets["ZIn"].fillna(0).astype(int)
                )
                gdf_priority_inlets["ZOut"] = gdf_priority_inlets["VP_Network_ID"].map(
                    gdf_priority_outlets.set_index("VP_Network_ID")["ZOut"]
                )
                gdf_priority_inlets["ZOut"] = (
                    gdf_priority_inlets["ZOut"].fillna(0).astype(int)
                )
                gdf_priority_inlets = gdf_priority_inlets[
                    [
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
                        "if_type",
                        "geometry",
                    ]
                ]
            gdf_all_points = pd.concat(
                [gdf_priority_inlets, gdf_priority_outlets], ignore_index=True
            )
            # Snap these points
            gdf_joined = gpd.sjoin_nearest(
                gdf_all_points,
                gdf_all_points,
                how="left",
                distance_col="dist",
                exclusive=True,
            )
            gdf_joined = gdf_joined[gdf_joined["dist"] <= 2.5 * CELL_SIZE]
            gdf_joined[["ID_left", "ID_right", "dist"]]
            duplicates = gdf_all_points[
                gdf_all_points.duplicated(subset=["geometry"], keep=False)
            ].copy()

            duplicates["geom_key"] = duplicates.geometry.to_wkt()
            joined = duplicates.merge(
                duplicates, on="geom_key", suffixes=("_left", "_right")
            )
            joined = joined[joined["ID_left"] != joined["ID_right"]]

            joined = joined[joined["ID_left"] < joined["ID_right"]]

            gdf_duplicates = gpd.GeoDataFrame(
                joined[["ID_left", "ID_right"]].copy(),
                geometry=joined["geometry_left"],
                crs=gdf_all_points.crs,
            )
            gdf_to_resolve = (
                pd.concat([gdf_duplicates, gdf_joined])
                .drop_duplicates()
                .reset_index(drop=True)
            )
            print(f"Resolving {len(gdf_to_resolve)} points close to each other...")
            iteration = 0
            while not gdf_to_resolve.empty:
                iteration += 1
                gdf_joined = gpd.sjoin_nearest(
                    gdf_all_points,
                    gdf_all_points,
                    how="left",
                    distance_col="dist",
                    exclusive=True,
                )

                duplicates = gdf_all_points[
                    gdf_all_points.duplicated(subset=["geometry"], keep=False)
                ].copy()

                duplicates["geom_key"] = duplicates.geometry.to_wkt()

                joined = duplicates.merge(
                    duplicates, on="geom_key", suffixes=("_left", "_right")
                )

                joined = joined[joined["ID_left"] != joined["ID_right"]]

                joined = joined[joined["ID_left"] < joined["ID_right"]]

                gdf_duplicates = gpd.GeoDataFrame(
                    joined[["ID_left", "ID_right"]].copy(),
                    geometry=joined["geometry_left"],
                    crs=gdf_all_points.crs,
                )
                gdf_joined = gdf_joined[gdf_joined["dist"] <= 2.5 * CELL_SIZE]
                gdf_joined[["ID_left", "ID_right", "dist"]]
                gdf_to_resolve = (
                    pd.concat([gdf_duplicates, gdf_joined])
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                print(f"Resolving {len(gdf_to_resolve)} points close to each other...")
                if iteration > 15:
                    gdf_all_points = gdf_all_points[
                        ~gdf_all_points["ID"].isin(gdf_to_resolve["ID_left"])
                    ]
                    continue
                for idx, row in gdf_to_resolve.iterrows():
                    left_geom = gdf_all_points.loc[
                        gdf_all_points["ID"] == row["ID_left"], "geometry"
                    ].values[0]
                    right_geom = gdf_all_points.loc[
                        gdf_all_points["ID"] == row["ID_right"], "geometry"
                    ].values[0]
                    gdf_all_points.loc[
                        gdf_all_points["ID"] == row["ID_left"], "geometry"
                    ] = shift_point_randomly(left_geom, 5)
                    # gdf_all_points.loc[
                    #     gdf_all_points["ID"] == row["ID_right"], "geometry"
                    # ] = shift_point_randomly(right_geom, 5)

            gdf_points_priority = gdf_all_points.copy()
            gdf_points_priority.to_file(
                os.path.join(VP_TEMP_DIR, f"USE00_TUNNELS_{self.domain_name}.gpkg"),
                driver="GPKG",
                index=False,
            )
            print("\tGenerating points")
            # UPDATE DONE - filtering bridges and tunnels, and will return path only for relevant road types - Points as for version 1.0
            filter = filter_a + filter_b + filter_c
            filter = list(set(filter))
            gdf_roads = Vector.load_vector(roads)
            gdf_roads = filter_roads(gdf_roads, FIELD_FOR_FILTERING, filter)
            gdf_intersections = get_line_intersections(gdf_roads)
            gdf_roads_split = split_lines_points(
                gdf_lines=gdf_roads, gdf_points=gdf_intersections
            )
            # gdf_roads_split = gdf_roads_split[gdf_roads_split.geometry.length > 15]
            gdf_roads_split.to_file(roads)
            points_a, gdf_roads_filter_a = get_points_along_geometry(
                roads=roads,
                interval=interval_inlets,
                filter_field=FIELD_FOR_FILTERING,
                filter=filter_a,
                CELL_SIZE=CELL_SIZE,
                EPSG=self.projection,
            )
            points_b, gdf_roads_filter_b = get_points_along_geometry(
                roads=roads,
                interval=interval_inlets,
                filter_field=FIELD_FOR_FILTERING,
                filter=filter_b,
                CELL_SIZE=CELL_SIZE,
                EPSG=self.projection,
            )
            points_c, gdf_roads_filter_c = get_points_along_geometry(
                roads=roads,
                interval=interval_inlets,
                filter_field=FIELD_FOR_FILTERING,
                filter=filter_c,
                CELL_SIZE=CELL_SIZE,
                EPSG=self.projection,
            )
            points_d, gdf_roads_filter_d = get_points_along_geometry(
                roads=roads,
                interval=interval_inlets,
                filter_field=FIELD_FOR_FILTERING,
                filter=filter,
                CELL_SIZE=CELL_SIZE,
                EPSG=self.projection,
            )
            gdf_roads_filter_d = None
            gdf_industrial_estates = Vector.load_vector(
                INDUSTRIAL_ESTATES, bbox=tuple(gdf_roads.total_bounds)
            )
            gdf_industrial_estates = Vector.fix_geometry(gdf_industrial_estates)
            points_d = points_d.clip(
                gdf_industrial_estates
            )  # clipping only within built-up area
            points_d.to_file(
                os.path.join(temp, "industrial_points.gpkg"), driver="GPKG", index=False
            )

            # As for now it loads roads specified in filter_a and adds roads from filter_b (unclassified) that are within built-up areas
            bounds = gdf_roads_filter_b.total_bounds
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
            mask = create_builtup_area_mask(
                materials_file=MANNING,
                extent=extent,
                threshold=0.25,
                window_size=11,
                overlap=50,
                epsg=self.projection,
            )
            gdf_roads_filter_b = gdf_roads_filter_b.clip(mask)
            gdf_roads_filter_b = gdf_roads_filter_b[
                gdf_roads_filter_b.geometry.length > 15
            ]
            points_b = points_b.clip(mask)  # clipping only within built-up area
            print("\tFiltering service inlet points.")
            bbox_address = gdf_roads_filter_c.total_bounds
            gdf_address_points = Vector.load_vector(
                ADDRESS_POINTS, bbox=tuple(bbox_address)
            )
            gdf_parking = Vector.load_vector(PARKING, bbox=tuple(bbox_address))
            gdf_address_points.geometry = gdf_address_points.buffer(100)
            gdf_address_points = gdf_address_points.dissolve().explode()
            points_c = points_c.clip(gdf_address_points)
            gdf_roads_filter_c = gdf_roads_filter_c.clip(gdf_address_points)
            # Remove points_c that intersect with gdf_parking (difference)
            points_c = points_c.overlay(gdf_parking, how="difference")
            gdf_roads_filter_c = gdf_roads_filter_c.overlay(
                gdf_parking, how="difference"
            )
            gdf_roads_filter_c = gdf_roads_filter_c[
                gdf_roads_filter_c.geometry.length > 15
            ]
            points_a["source"] = "a"
            points_b["source"] = "b"
            points_c["source"] = "c"
            points_d["source"] = "d"
            points = pd.concat(
                [points_a, points_b, points_c, points_d], ignore_index=True
            )
            points = points.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

            gdf_roads_filtered = pd.concat(
                [gdf_roads_filter_a, gdf_roads_filter_b, gdf_roads_filter_c],
                ignore_index=True,
            )

            print(
                f"\tCreating basic inlets at predefined interval of: {interval_inlets} m"
            )
            inlets = create_inlets_selection(
                points=points,
                water_network=[wb, rn],
                cell_size=CELL_SIZE,
                out_file=os.path.join(VP_TEMP_DIR, f"01_inlets_{self.domain_name}.gpkg"),
            )
            print("\tMoving inlets to hazard")
            inlets = move_inlets(
                inlets=inlets,
                out_file=os.path.join(temp, f"USE02_inlets_normal_{self.domain_name}.gpkg"),
                cell_size=CELL_SIZE,
                epsg=self.projection,
                dtm=dtm,
            )
            inlets = Vector.load_vector(inlets)
            inlets = extract_inlets(
                dem=dtm,
                inlets=inlets,
                temp=VP_TEMP_DIR,
            )
            inlets = inlets[["pointid", "dtm_value", "geometry"]]
            inlets = filter_inlets(inlets, CELL_SIZE, increase_capacity=False)
            inlets.to_file(
                os.path.join(VP_TEMP_DIR, f"USE02_normal_inlets_{self.domain_name}.gpkg"),
                driver="GPKG",
                index=False,
            )
            inlets = os.path.join(VP_TEMP_DIR, f"USE02_normal_inlets_{self.domain_name}.gpkg")

            print("\tCreating densified inlets within depressions")
            inlets_densified = create_new_inlets(
                dem=dtm,
                depression_vector_path=filtered_depressions_path,
                gdf_roads=gdf_roads_filtered,
                interval=interval_inlets / 3,
                out_file=os.path.join(temp, f"02_densified_inlets{self.domain_name}.gpkg"),
                epsg=self.projection,
                cell_size=CELL_SIZE,
            )
            inlets_densified = Vector.load_vector(inlets_densified)
            # apply cleanup on densified inlets
            inlets_densified = create_inlets_selection(
                points=inlets_densified,
                water_network=[wb, rn],
                cell_size=CELL_SIZE,
                out_file=os.path.join(temp, f"03_inlets_densified_selected_{self.domain_name}.gpkg"),
            )

            inlets_densified = move_inlets(
                inlets=inlets_densified,
                out_file=os.path.join(temp, f"04_densified_inlets_snapped_{self.domain_name}.gpkg"),
                cell_size=CELL_SIZE,
                epsg=self.projection,
                dtm=dtm,
            )

            inlets_densified = extract_inlets(
                dem=dtm,
                inlets=inlets_densified,
                temp=VP_TEMP_DIR,
            )

            inlets_densified = filter_inlets(
                inlets_densified, CELL_SIZE, increase_capacity=False
            )
            inlets_densified.to_file(
                os.path.join(VP_TEMP_DIR, f"USE03_densified_{self.domain_name}_snapped_filtered.gpkg"),
                driver="GPKG",
                index=False,
            )
            print("\tAdding new inlets that will drain water from parking lots")
            gdf_roads_parking = gdf_roads_split.clip(gdf_parking)
            gdf_roads_parking = gdf_roads_parking[
                gdf_roads_parking.geometry.length > 10
            ]

            inlets_densified_parking = create_new_inlets(
                dem=dtm,
                depression_vector_path=filtered_depressions_path,
                gdf_roads=gdf_roads_parking,
                interval=interval_inlets / 5,
                out_file=os.path.join(temp, f"02_densified_inlets_PARKING_{self.domain_name}.gpkg"),
                epsg=self.projection,
                cell_size=CELL_SIZE,
            )
            inlets_densified_parking = Vector.load_vector(inlets_densified_parking)
            # apply cleanup on densified inlets
            inlets_densified_parking = create_inlets_selection(
                points=inlets_densified_parking,
                water_network=[wb, rn],
                cell_size=CELL_SIZE,
                out_file=os.path.join(temp, f"03_inlets_densified_selected_PARKING_{self.domain_name}.gpkg"),
            )

            inlets_densified_parking = move_inlets(
                inlets=inlets_densified_parking,
                out_file=os.path.join(VP_TEMP_DIR, f"04_densified_inlets_snapped_PARKING_{self.domain_name}.gpkg"),
                cell_size=CELL_SIZE,
                epsg=self.projection,
                dtm=dtm,
            )

            inlets_densified_parking = extract_inlets(
                dem=dtm,
                inlets=inlets_densified_parking,
                temp=VP_TEMP_DIR,
            )

            inlets_densified_parking = filter_inlets(
                inlets_densified_parking, CELL_SIZE, increase_capacity=False
            )
            inlets_densified_parking.to_file(
                os.path.join(VP_TEMP_DIR, f"USE04_parking_{self.domain_name}.gpkg"),
                driver="GPKG",
                index=False,
            )

            print("\tCreating outlets on rn and waterbodies")

            outlets = create_outlets(
                rn=rn,
                waterbody=wb,
                interval=interval_outlets,
                out_file=os.path.join(
                    out_part_layers, f"outlets_{self.domain_name}.gpkg"
                ),
                epsg=self.projection,
                cell_size=CELL_SIZE,
            )

            outlets = move_outlets(
                outlets=outlets,
                out_file=os.path.join(
                    out_part_layers, f"outlets_{self.domain_name}_snapped.gpkg"
                ),
                epsg=self.projection,
                cell_size=CELL_SIZE,
                dtm=dtm,
            )
            if FILTER_BUILDINGS:
                bounds = gdf_mask.total_bounds
                outlets = remove_outlets_buildings(
                    outlets,
                    BUILDINGS,
                    out_file=os.path.join(VP_TEMP_DIR, f"07_outlets_{self.domain_name}_snapped_filtered.gpkg"),
                    bounds=bounds,
                    bounds_epsg=self.projection,
                )
            else:
                pass
            inlets = Vector.load_vector(inlets)
            inlets_depressions = Vector.load_vector(inlets_depressions)
            # CHECK FILE REMOVE
            gdf_all_inlets = pd.concat(
                [
                    inlets,
                    inlets_densified,
                    inlets_densified_parking,
                    inlets_depressions,
                ],
                ignore_index=True,
            )
            gdf_all_inlets = gdf_all_inlets.drop_duplicates(
                subset=["geometry"]
            ).reset_index(drop=True)
            print(
                f"\tCleaning inlets, total count before cleaning: {len(gdf_all_inlets.index)}"
            )
            gdf_all_inlets = filter_inlets(
                gdf_all_inlets, CELL_SIZE, increase_capacity=True
            )
            gdf_all_inlets["type"] = "inlet"
            gdf_all_outlets = gpd.read_file(outlets)
            gdf_all_outlets["type"] = "outlet"
            gdf_all_points = pd.concat(
                [gdf_all_inlets, gdf_all_outlets], ignore_index=True
            )
            gdf_all_points = gdf_all_points.reset_index(drop=True)
            gdf_all_points["pointid"] = gdf_all_points.index + 1  # you already do this

            gdf_left = gdf_all_points.rename(columns={"pointid": "pointid_left"})
            gdf_right = gdf_all_points.rename(columns={"pointid": "pointid_right"})

            gdf_joined = gpd.sjoin_nearest(
                gdf_left,
                gdf_right,
                how="left",
                distance_col="dist",
                exclusive=True,
            )

            gdf_joined = gdf_joined[gdf_joined["dist"] <= 2.5 * CELL_SIZE]
            gdf_joined = gdf_joined[
                ["pointid_left", "pointid_right", "dist", "geometry"]
            ]
            duplicates = gdf_all_points[
                gdf_all_points.duplicated(subset=["geometry"], keep=False)
            ].copy()

            duplicates["geom_key"] = duplicates.geometry.to_wkt()
            joined = duplicates.merge(
                duplicates, on="geom_key", suffixes=("_left", "_right")
            )
            joined = joined[joined["pointid_left"] != joined["pointid_right"]]

            joined = joined[joined["pointid_left"] < joined["pointid_right"]]

            gdf_duplicates = gpd.GeoDataFrame(
                joined[["pointid_left", "pointid_right"]].copy(),
                geometry=joined["geometry_left"],
                crs=gdf_all_points.crs,
            )
            gdf_to_resolve = (
                pd.concat([gdf_duplicates, gdf_joined])
                .drop_duplicates()
                .reset_index(drop=True)
            )

            while not gdf_to_resolve.empty:
                gdf_left = gdf_all_points.rename(columns={"pointid": "pointid_left"})
                gdf_right = gdf_all_points.rename(columns={"pointid": "pointid_right"})

                gdf_joined = gpd.sjoin_nearest(
                    gdf_left,
                    gdf_right,
                    how="left",
                    distance_col="dist",
                    exclusive=True,
                )

                gdf_joined = gdf_joined[gdf_joined["dist"] <= 2.5 * CELL_SIZE]
                gdf_joined = gdf_joined[
                    ["pointid_left", "pointid_right", "dist", "geometry"]
                ]
                duplicates = gdf_all_points[
                    gdf_all_points.duplicated(subset=["geometry"], keep=False)
                ].copy()

                duplicates["geom_key"] = duplicates.geometry.to_wkt()
                joined = duplicates.merge(
                    duplicates, on="geom_key", suffixes=("_left", "_right")
                )
                joined = joined[joined["pointid_left"] != joined["pointid_right"]]

                joined = joined[joined["pointid_left"] < joined["pointid_right"]]

                gdf_duplicates = gpd.GeoDataFrame(
                    joined[["pointid_left", "pointid_right"]].copy(),
                    geometry=joined["geometry_left"],
                    crs=gdf_all_points.crs,
                )
                gdf_to_resolve = (
                    pd.concat([gdf_duplicates, gdf_joined])
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                print(f"Resolving {len(gdf_to_resolve)} points close to each other...")
                for idx, row in gdf_to_resolve.iterrows():
                    left_geom = gdf_all_points.loc[
                        gdf_all_points["pointid"] == row["pointid_left"], "geometry"
                    ].values[0]
                    right_geom = gdf_all_points.loc[
                        gdf_all_points["pointid"] == row["pointid_right"], "geometry"
                    ].values[0]
                    gdf_all_points.loc[
                        gdf_all_points["pointid"] == row["pointid_left"], "geometry"
                    ] = shift_point_randomly(left_geom, 5)
                    gdf_all_points.loc[
                        gdf_all_points["pointid"] == row["pointid_right"], "geometry"
                    ] = shift_point_randomly(right_geom, 5)
            gdf_inlets = gdf_all_points[gdf_all_points["type"] == "inlet"].copy()
            gdf_inlets = gdf_inlets.reset_index(drop=True)
            gdf_outlets = gdf_all_points[gdf_all_points["type"] == "outlet"].copy()
            gdf_outlets = gdf_outlets.reset_index(drop=True)

            print(f"Columns of inlets: {gdf_inlets.columns}")

            if not gdf_inlets.empty and not gdf_outlets.empty:
                # 1) Remove inlets that are within 30m of any outlet
                joined = gpd.sjoin_nearest(
                    gdf_inlets,
                    gdf_outlets,
                    how="left",
                    distance_col="dist_to_outlet",
                    max_distance=5 * CELL_SIZE,
                )

                # Indices of inlets where dist_to_outlet is NaN OR > 20
                keep_idx = joined.index[
                    joined["dist_to_outlet"].isna()
                    | (joined["dist_to_outlet"] > 2.5 * CELL_SIZE)
                ]
                gdf_inlets = gdf_inlets.loc[keep_idx].copy()
                gdf_inlets = gdf_inlets.reset_index(drop=True)

                # 2) Load priority points
                gdf_points_priority = Vector.load_vector(
                    os.path.join(VP_TEMP_DIR, f"USE00_TUNNELS_{self.domain_name}.gpkg")
                )

                # 3) Remove inlets within 15m of a priority point
                joined_inlets_prio = gpd.sjoin_nearest(
                    gdf_inlets,
                    gdf_points_priority,
                    how="left",
                    distance_col="dist_to_priority",
                    max_distance=3 * CELL_SIZE,
                )

                keep_idx_prio_inlets = joined_inlets_prio.index[
                    joined_inlets_prio["dist_to_priority"].isna()
                    | (joined_inlets_prio["dist_to_priority"] > CELL_SIZE * 2.5)
                ]
                gdf_inlets = gdf_inlets.loc[keep_idx_prio_inlets].copy()
                gdf_inlets = gdf_inlets.reset_index(drop=True)

                # 4) Remove outlets within 15m of a priority point
                joined_outlets_prio = gpd.sjoin_nearest(
                    gdf_outlets,
                    gdf_points_priority,
                    how="left",
                    distance_col="dist_to_priority",
                    max_distance=3 * CELL_SIZE,
                )

                keep_idx_prio_outlets = joined_outlets_prio.index[
                    joined_outlets_prio["dist_to_priority"].isna()
                    | (joined_outlets_prio["dist_to_priority"] > CELL_SIZE * 2.5)
                ]
                gdf_outlets = gdf_outlets.loc[keep_idx_prio_outlets].copy()
                gdf_outlets = gdf_outlets.reset_index(drop=True)

            # Output paths
            outlets = os.path.join(VP_TEMP_DIR, f"08_outlets_{self.domain_name}_cleaned.gpkg")
            inlets = os.path.join(VP_TEMP_DIR, f"08_inlets_{self.domain_name}_cleaned.gpkg")

            gdf_inlets.to_file(
                inlets,
                driver="GPKG",
                index=False,
            )
            gdf_outlets.to_file(
                outlets,
                driver="GPKG",
                index=False,
            )
            connect(
                inlets,
                outlets,
                os.path.join(out_folder, out_name),
                dtm,
                temp,
                epsg=self.projection,
            )
            outfile = os.path.join(out_folder, out_name)
            gdf_outfile = gpd.read_file(outfile)
            gdf_outfile["if_type"] = "normal"
            gdf_priority = gpd.read_file(
                os.path.join(VP_TEMP_DIR, f"USE00_TUNNELS_{self.domain_name}.gpkg")
            )
            gdf_priority.sort_values(
                by=["VP_Network_ID", "ZIn"], ascending=[True, False], inplace=True
            )
            gdf_all_points = pd.concat([gdf_priority, gdf_outfile], ignore_index=True)
            gdf = gdf_all_points.copy()
            outlets = gdf[gdf["Type"] == "O"][["VP_Network_ID", "geometry"]].set_index(
                "VP_Network_ID"
            )

            # Function to calculate distance to outlet
            def distance_to_outlet(row):
                if row["Type"] == "O":
                    return 0.0  # Outlets have distance 0

                # Check if outlet exists for this VP_Network_ID
                if row["VP_Network_ID"] not in outlets.index:
                    return None  # Return None if no matching outlet

                outlet_geom = outlets.loc[row["VP_Network_ID"], "geometry"]
                return row["geometry"].distance(outlet_geom)

            # Apply the function to create the distance column
            gdf["dist"] = gdf.apply(distance_to_outlet, axis=1)
            gdf["dist"] = gdf["dist"].round(2)

            gdf.loc[gdf["Type"] == "O", "ZIn"] = gdf.loc[gdf["Type"] == "O", "ZOut"]
            gdf = gdf.sort_values(by=["VP_Network_ID", "ZIn"], ascending=[True, True])
            gdf["VP_Sur_Index"] = gdf.groupby("VP_Network_ID").cumcount()
            gdf = gdf.reset_index(drop=True)
            # Remove outlets that have no inlets in their network
            # For each VP_Network_ID, check if there's at least one inlet
            inlets_per_network = gdf[gdf["Type"] == "I"].groupby("VP_Network_ID").size()
            valid_networks_with_inlets = set(inlets_per_network.index)
            # Keep all inlets and outlets that have inlets in their network
            gdf = gdf[
                (gdf["Type"] == "I")
                | (
                    (gdf["Type"] == "O")
                    & (gdf["VP_Network_ID"].isin(valid_networks_with_inlets))
                )
            ]
            gdf.loc[(gdf["Type"] == "I") & (gdf["Number_of"] == 0), "Number_of"] = (
                1  # Minimum capacity is 1
            )
            gdf.loc[(gdf["Type"] == "I") & (gdf["Number_of"] > 12), "Number_of"] = (
                12  # Capacity max restricted to 12
            )
            gdf["Lag_Approach"] = "None"
            gdf["Lag_Value"] = 0.0
            gdf = gdf.reset_index(drop=True)

            gdf_left = gdf.rename(columns={"ID": "ID_left"})
            gdf_right = gdf.rename(columns={"ID": "ID_right"})
            if "level_0" in gdf_left.columns:
                gdf_left = gdf_left.drop(columns=["level_0"])
            if "level_0" in gdf_right.columns:
                gdf_right = gdf_right.drop(columns=["level_0"])

            gdf_joined = gpd.sjoin_nearest(
                gdf_left,
                gdf_right,
                how="left",
                distance_col="dist",
                exclusive=True,
            )

            gdf_joined = gdf_joined[gdf_joined["dist"] <= CELL_SIZE * 2.5]
            gdf_joined = gdf_joined[["ID_left", "ID_right", "dist", "geometry"]]
            duplicates = gdf_all_points[
                gdf_all_points.duplicated(subset=["geometry"], keep=False)
            ].copy()

            duplicates["geom_key"] = duplicates.geometry.to_wkt()
            joined = duplicates.merge(
                duplicates, on="geom_key", suffixes=("_left", "_right")
            )
            joined = joined[joined["ID_left"] != joined["ID_right"]]

            joined = joined[joined["ID_left"] < joined["ID_right"]]

            gdf_duplicates = gpd.GeoDataFrame(
                joined[["ID_left", "ID_right"]].copy(),
                geometry=joined["geometry_left"],
                crs=gdf_all_points.crs,
            )
            gdf_to_resolve = (
                pd.concat([gdf_duplicates, gdf_joined])
                .drop_duplicates()
                .reset_index(drop=True)
            )

            while not gdf_to_resolve.empty:
                gdf_left = gdf.rename(columns={"ID": "ID_left"})
                gdf_right = gdf.rename(columns={"ID": "ID_right"})

                gdf_joined = gpd.sjoin_nearest(
                    gdf_left,
                    gdf_right,
                    how="left",
                    distance_col="dist",
                    exclusive=True,
                )

                gdf_joined = gdf_joined[gdf_joined["dist"] <= CELL_SIZE * 2.5]
                gdf_joined = gdf_joined[["ID_left", "ID_right", "dist", "geometry"]]
                duplicates = gdf[gdf.duplicated(subset=["geometry"], keep=False)].copy()

                duplicates["geom_key"] = duplicates.geometry.to_wkt()
                joined = duplicates.merge(
                    duplicates, on="geom_key", suffixes=("_left", "_right")
                )
                joined = joined[joined["ID_left"] != joined["ID_right"]]

                joined = joined[joined["ID_left"] < joined["ID_right"]]

                gdf_duplicates = gpd.GeoDataFrame(
                    joined[["ID_left", "ID_right"]].copy(),
                    geometry=joined["geometry_left"],
                    crs=gdf_all_points.crs,
                )
                gdf_to_resolve = (
                    pd.concat([gdf_duplicates, gdf_joined])
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                print(f"Resolving {len(gdf_to_resolve)} points close to each other...")
                for idx, row in gdf_to_resolve.iterrows():
                    left_geom = gdf.loc[gdf["ID"] == row["ID_left"], "geometry"].values[
                        0
                    ]
                    right_geom = gdf.loc[
                        gdf["ID"] == row["ID_right"], "geometry"
                    ].values[0]
                    gdf.loc[gdf["ID"] == row["ID_left"], "geometry"] = (
                        shift_point_randomly(left_geom, 5)
                    )
                    gdf.loc[gdf["ID"] == row["ID_right"], "geometry"] = (
                        shift_point_randomly(right_geom, 5)
                    )

            # 1) Define target dtypes
            int_cols = [
                "VP_Network_ID",
                "Conn_No",
                "Number_of",
            ]
            float_cols = [
                "VP_Sur_Index",
                "VP_QMax",
                "Width",
                "pBlockage",
                "Lag_Value",
                "ZIn",
                "ZOut",
                "dist",
            ]

            str_cols = [
                "ID",
                "Type",
                "Inlet_Type",
                "Conn_2D",
                "Lag_Approach",
                "if_type",
            ]

            # 2) Convert to numeric (float) first with errors='coerce' to find bad values
            for col in int_cols:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

            for col in float_cols:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

            # 3) (Optional) inspect rows with invalid integer values before forcing int
            #    These are rows where integer columns became NaN:
            bad_int_mask = gdf[int_cols].isna().any(axis=1)
            if bad_int_mask.any():
                print("Rows with non-convertible values in integer columns:")
                print(gdf.loc[bad_int_mask, int_cols + ["ID"]].head())

            # 4) Fill NaN in integer columns with a default value (e.g., 0) before casting
            for col in int_cols:
                gdf[col] = gdf[col].fillna(0)

            # 5) Now safely cast types
            for col in int_cols:
                gdf[col] = gdf[col].astype(int)

            for col in float_cols:
                gdf[col] = gdf[col].astype(float)

            for col in str_cols:
                gdf[col] = gdf[col].astype(str)

            gdf.to_file(
                outfile,
                driver="GPKG",
                index=False,
            )
            print(
                f"\tFile for adjustments saved to:  {os.path.join(out_folder, out_name.replace('.gpkg','_adjusted.gpkg'))}"
            )
            print("\tDo not forget to cleanup temp path")
            print("\tDone in {} s".format(datetime.datetime.now() - start))

        else:
            print(f"\t{self.domain_name} does not exist or is already created SKIPPING")
