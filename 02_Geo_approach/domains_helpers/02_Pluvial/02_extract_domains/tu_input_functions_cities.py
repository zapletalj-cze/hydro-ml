import os
import glob
import csv

os.environ["USE_PYGEOS"] = "0"

import shutil
import fnmatch
from datetime import date
import sys
from osgeo import gdal
import pyogrio
import geopandas as gpd
from ifgis.raster import (
    get_array_from_raster,
    save_array_with_type,
    extract_by_mask_rasterized,
    get_resolution,
    get_extent,
    snap_extent,
    rasterize,
)
from ifgis.vector import select_by_location, buffer, select_by_attribute, merge
from shapely import make_valid
from shapely.geometry import LineString, Point
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

import tu_data_cities as tu_data
import yaml

tu_sources = tu_data.tu_sources
workspace = tu_data.workspace
config_variables = tu_sources.config_variables

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_template_root() -> str:
    peril_folder = tu_sources.tuflow_directory[tu_data.tu_parameters["Q_peril"]]
    candidates = []

    # Prefer common template folders across all configured root paths
    # (V:, D:, network shares, etc.), not just domain_directory_source.
    for _, root_hazard in tu_sources.d_root_path_net.items():
        candidates.append(
            os.path.join(root_hazard, peril_folder, "02_common", "Shared", "template")
        )
        candidates.append(
            os.path.join(root_hazard, peril_folder, "01_common", "Shared", "template")
        )

    # Backward-compatible candidates.
    candidates.extend(
        [
            os.path.join(tu_data.path_common_inputs, "Shared", "template"),
            os.path.join(SCRIPT_DIR, "Shared", "template"),
            os.path.join(os.path.dirname(SCRIPT_DIR), "Shared", "template"),
        ]
    )

    # Keep order but avoid duplicate checks.
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm not in seen:
            seen.add(norm)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Template root not found. Checked: " + "; ".join(ordered_candidates)
    )


TEMPLATE_ROOT = _resolve_template_root()


##msa
class City_parameters:
    def __init__(self, name, dtm_resolution, mesh_resolution):
        self.name = name
        self.dtm_resolution = dtm_resolution
        self.mesh_resolution = mesh_resolution

    def __repr__(self):
        return (
            f"City_parameters(name='{self.name}', "
            f"dtm_resolution={self.dtm_resolution}, "
            f"mesh_resolution={self.mesh_resolution})"
        )


def tu_input_names(workspace, domain):
    """
    Method which will get output file names for fluvial
    """
    code_vec = os.path.join(
        workspace, domain, "model", "gis", "2d_code_" + domain + "_R.gpkg"
    )
    rf_vec = os.path.join(
        workspace, domain, "model", "gis", "2d_rf_" + domain + "_R.gpkg"
    )
    bc_in_vec_R = os.path.join(
        workspace, domain, "model", "gis", "2d_sa_IN_" + domain + "_R.gpkg"
    )
    bc_in_vec_L = os.path.join(
        workspace, domain, "model", "gis", "2d_sa_IN_" + domain + "_L.gpkg"
    )

    ## TUFLOW rasters input names - specification:
    dtm_grid_flt = os.path.join(
        workspace, domain, "model", "grid", "dtm_" + domain + ".tif"
    )
    mat_grid_flt = os.path.join(
        workspace, domain, "model", "grid", "2d_mat_" + domain + ".tif"
    )
    soil_grid_flt = os.path.join(
        workspace, domain, "model", "grid", "2d_soil_" + domain + ".tif"
    )

    ## msa mozna pak pridat i vp resp.1d_pit
    # pit_vec = os.path.join(
    #     workspace, domain, "model", "gis", "1d_pit_" + domain + "_P.gpkg"
    # )

    return (
        code_vec,
        rf_vec,
        bc_in_vec_R,
        bc_in_vec_L,
        dtm_grid_flt,
        mat_grid_flt,
        soil_grid_flt,
        # pit_vec
    )


def create_tuflow_dir_structure(workspace, domain):
    """
    Create workspace structure
    """
    os.makedirs(os.path.join(workspace, domain, "model", "grid"), exist_ok=True)
    os.makedirs(os.path.join(workspace, domain, "runs"), exist_ok=True)


def get_DTM_metadata(dtm_grid_tif: str):
    """
    Method which will get meta data from DTM
    """
    cell_size_snap = get_resolution(config_variables["snap_raster"])[0]
    extent = snap_extent(
        get_extent(dtm_grid_tif),
        get_extent(config_variables["snap_raster"]),
        cell_size_snap,
    )

    src = gdal.Open(dtm_grid_tif, gdal.GA_ReadOnly)
    gt = src.GetGeoTransform()
    ncols = src.RasterXSize
    nrows = src.RasterYSize
    cellsize = gt[1]
    xllcorner = gt[0]
    y_max = gt[3]
    yllcorner = y_max + (
        src.RasterYSize * gt[-1]
    )  ## calculating bottom y, metadata includes only upper y

    # return ncols, nrows, xllcorner, yllcorner, cellsize
    return (
        ncols + 20,
        nrows + 20,
        round(float(extent[0]), 2) - cell_size_snap,
        round(float(extent[2]), 2) - cell_size_snap,
        cellsize,
    )


def extract_dem_add(
    workspace, domain, code_buff, d_DTM_add_path, key, value, watershed=None, city=None
):
    """
    Method, which will add some value to dtm
    """
    output = os.path.join(workspace, domain, "model", "grid", f"{key}_{domain}.tif")
    tmp_file = rf"/vsimem/dtm_{key}_{domain}_tmp.tif"

    ##msa dodelat
    # print(f'output: {output}')
    # print(city)
    # print(key)
    # print(d_DTM_add_path[key][city.name])
    if city is not None:
        if str(city.dtm_resolution) in d_DTM_add_path[key][city.name]:
            in_file = d_DTM_add_path[key][city.name][str(city.dtm_resolution)]
            print(in_file)

        else:
            print(f"\n... DTM {key} not found for {city.name} in {d_DTM_add_path[key]}")
            return

    elif watershed is not None:
        if (
            str(tu_data.tu_parameters["pluvial_resolution"])
            in d_DTM_add_path[key][watershed]
        ):
            in_file = d_DTM_add_path[key][watershed][
                tu_data.tu_parameters["pluvial_resolution"]
            ]

        else:
            print(f"\n... DTM {key} not found for {watershed} in {d_DTM_add_path[key]}")
            return
    else:
        if str(tu_data.tu_parameters["pluvial_resolution"]) in d_DTM_add_path[key]:
            in_file = d_DTM_add_path[key][tu_data.tu_parameters["pluvial_resolution"]]
        else:
            print(f"\n... DTM {key} not found for {domain} in {d_DTM_add_path[key]}")
            return

    extract_by_mask_rasterized(
        raster=in_file,
        mask=code_buff,
        output=tmp_file,
        save_empty=True,
    )
    if value["multiplication_factor"] == 1:
        gdal.Translate(
            output,
            tmp_file,
            creationOptions=["COMPRESS=DEFLATE", "NUM_THREADS=8", "ZLEVEL=12"],
        )
    else:
        ## reading field
        array, no_data = get_array_from_raster(tmp_file)
        ## re-typing
        array_as_float = array.astype(float)
        ## multiply, np.where(condition, true, false)
        multiplied = np.where(
            array_as_float != no_data,
            array_as_float * value["multiplication_factor"],
            no_data,
        )
        save_array_with_type(
            multiplied,
            output,
            sample_raster=tmp_file,
            data_type=gdal.GDT_Float32,
            no_data=no_data,
            creation_options=[
                "COMPRESS=DEFLATE",
                "NUM_THREADS=ALL_CPUS",
                "ZLEVEL=12",
            ],
        )

    gdal.Unlink(tmp_file)


## msa add for cities
def extract_pits_to_domain(
    workspace,
    domain,
    # tu_zsh_lines_switch,
    path_in_pit_VP_P,
    output_resolution,
):
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")

    ## msa 260217 - zmena vyberu VP od RF
    rf = os.path.join(workspace, domain, "model", "gis", f"2d_rf_{domain}_R.gpkg")

    area = domain.split("_")[0]

    os.makedirs(os.path.join(workspace, domain, "model", "gis"), exist_ok=True)

    ## output layers
    out_pit_ga00_name = "1d_pit_" + domain + "_GA00_P"
    out_pit_ga50_name = "1d_pit_" + domain + "_GA50_P"
    out_tunnel_name = "1d_pit_" + domain + "_tunnel_P"
    out_tunnel_50_name = "1d_pit_" + domain + "_tunnel_50_P"

    out_pit_ga00 = os.path.join(
        workspace, domain, "model", "gis", out_pit_ga00_name + ".gpkg"
    )
    out_pit_ga50 = os.path.join(
        workspace, domain, "model", "gis", out_pit_ga50_name + ".gpkg"
    )

    out_tunnel = os.path.join(
        workspace, domain, "model", "gis", out_tunnel_name + ".gpkg"
    )

    out_tunnel_50 = os.path.join(
        workspace, domain, "model", "gis", out_tunnel_50_name + ".gpkg"
    )

    if os.path.isfile(out_pit_ga00):
        gdal.Unlink(out_pit_ga00)  # delete file if exists
    if os.path.isfile(out_pit_ga50):
        gdal.Unlink(out_pit_ga50)  # delete file if exists
    if os.path.isfile(out_tunnel):
        gdal.Unlink(out_tunnel)  # delete file if exists
    if os.path.isfile(out_tunnel_50):
        gdal.Unlink(out_tunnel_50)  # delete file if exists

    # ## funkni verze 2025-12-04
    # overlay_neg = buffer(
    #     input=code, distance=-50, field="distance", output=None, driver="GPKG"
    # )

    ## gdf_p = select_by_location(
    ##     select_from=d_path_in_pit_VP_P,
    ##     overlay=code,
    ##     output=None,
    ##     method="intersect",
    ## )

    # ## funkni verze 2025-12-04
    # gdf_p = select_by_location(
    #     select_from=path_in_pit_VP_P,
    #     overlay=overlay_neg,
    #     output=None,
    #     method="intersect",
    # )

    ##testovaci verze 2025-12-04
    overlay_outflow = buffer(
        input=code, distance=-50, field="distance", output=None, driver="GPKG"
    )
    # msa zaloha pristupu overlay pro inlets, zmena vyberu VP od RF
    # overlay_inflow = buffer(
    #     input=code, distance=-50, field="distance", output=None, driver="GPKG"
    # )

    if os.path.exists(rf):
        overlay_inflow = buffer(
            input=rf, distance=1500, field="distance", output=None, driver="GPKG"
        ).dissolve()

    else:
        print(
            f"\n... RF layer not found for {domain} at {rf}, using code buffer for inflow selection"
        )
        # overlay_inflow = buffer(
        #     input=code, distance=-30, field="distance", output=None, driver="GPKG"
        # )

    ## msa 260217 - zmena vyberu VP od RF
    # overlay_outflow = buffer(
    #     input=code, distance=-50, field="distance", output=None, driver="GPKG"
    # )

    ## vezme min VP bodu a to ruzne pro inlets a outlets
    gdf_p_ouflow = select_by_attribute(
        input=path_in_pit_VP_P, field="Type", method=0, value="O"
    )
    gdf_p_ouflow = gdf_p_ouflow[gdf_p_ouflow["if_type"] != "tunnel"]

    gdf_p_ouflow_sel = select_by_location(
        select_from=gdf_p_ouflow,
        overlay=overlay_outflow,
        output=None,
        method="intersect",
    )

    gdf_p_inflow = select_by_attribute(
        input=path_in_pit_VP_P, field="Type", method=0, value="I"
    )
    gdf_p_inflow = gdf_p_inflow[gdf_p_inflow["if_type"] != "tunnel"]

    # pro Jakuba - nasledujici zakonmntovana cast mela nejdriv vybrat inflow VP body v okoli RF polygonu a tuhle selekci oriznout polygonem negativniho bufferu od code.

    gdf_p_inflow_sel_rf = select_by_location(
        select_from=gdf_p_inflow,
        overlay=overlay_inflow,
        output=None,
        method="intersect",
    )
    if "index_right" in gdf_p_inflow_sel_rf.columns:
        gdf_p_inflow_sel_rf = gdf_p_inflow_sel_rf.drop(columns=["index_right"])

    # ## msa 270217 - zmena vyberu VP od RF, pro inlets se pouzije buffer od RF, pro outlets buffer od code, protoze outlets jsou blize k code nez k RF
    gdf_p_inflow_sel = select_by_location(
        select_from=gdf_p_inflow_sel_rf,
        overlay=overlay_outflow,
        output=None,
        method="intersect",
    )
    if "index_right" in gdf_p_inflow_sel.columns:
        gdf_p_inflow_sel = gdf_p_inflow_sel.drop(columns=["index_right"])
    # gdf_p_inflow_sel= select_by_location(
    #     select_from=gdf_p_inflow_sel,
    #     overlay=overlay_outflow,
    #     output=None,
    #     method="intersect",
    # )
    # if "index_right" in gdf_p_inflow_sel.columns:
    #     gdf_p_inflow_sel = gdf_p_inflow_sel.drop(columns=["index_right"])

    ## parameters for setup of VP's attributes
    Q_max_inlet_pipes = 0.05  ##0.05 (50l/s) pro klasicky inlet; je to nejaky kompromis
    default_number_of_inlets_per_outlet = 400  ## defaultni pocet inletu pro jeden outlet, pro ktery se bude nastavovat Qmax 0.05 m3/s; pokud bude v nejakem outletu mene inletu, bude se jim nastavit vsem Qmax 0.05 m3/s, pokud bude v nejakem outletu vice inletu, bude se jim nastavit vsem Qmax 0.05 m3/s, ale bude to znamenat, ze pro ten outlet bude nastaveno Qmax ;
    ## pouziva kuba pro linkovani iletu na outlet
    Q_max_outlet_pipes = 10  ##10 m3/s for outflow (9m3s odpovida cca v 2m/s a prumer trouby 2m), je to nejaky kompromis, pro GA50 se to bude redukovat na tretinu tj.3 m3/s
    Q_max_outlet_pipes_GA50 = (
        Q_max_outlet_pipes * 0.5
    ) / 2  ## redukce napulku blokaci potrubni site a na dalsi pulku za polovicni natok do inpets. outflow s inlet zustava stejny jako u GA00, aby konzistentni s nizsimi rp
    Q_max_outlet_tunnel = 40  ##10 m3/s for outflow (9m3s odpovida cca v 2m/s a prumer trouby 2m), je to nejaky kompromis, pro GA50 se to bude redukovat na tretinu tj.3 m3/s

    if not gdf_p_inflow_sel.empty:
        gdf_p_inflow_sel["VP_QMax"] = (
            Q_max_inlet_pipes  ##0.05 (60l/s) pro klasicky inlet
        )

    if not gdf_p_ouflow_sel.empty:
        gdf_p_ouflow_sel["VP_QMax"] = Q_max_outlet_pipes  ##12 m3/s for outflow

    gdf_p = merge([gdf_p_ouflow_sel, gdf_p_inflow_sel])

    # gdf_p["Lag_Approach"] = 'None'  ## 'None' 'Decay'

    # Check for invalid Lag_Approach values
    invalid_lag = gdf_p[~gdf_p["Lag_Approach"].isin(["None", "Decay"])]
    if not invalid_lag.empty:
        invalid_ids = invalid_lag["VP_Network_ID"].unique()
        gdf_p.loc[gdf_p["VP_Network_ID"].isin(invalid_ids), "Lag_Approach"] = "None"

    # Check for consistent Lag_Approach and Lag_Value within each VP_Network_ID group
    grouped = gdf_p.groupby("VP_Network_ID")
    for name, group in grouped:
        if group["Lag_Approach"].nunique() > 1 or group["Lag_Value"].nunique() > 1:
            gdf_p.loc[gdf_p["VP_Network_ID"] == name, "Lag_Approach"] = "None"

    if not gdf_p.empty:
        ## save full open VP
        gdf_p.to_file(out_pit_ga00, driver="GPKG", layer=out_pit_ga00_name)

    ## for GA50
    if not gdf_p_inflow_sel.empty:
        #     gdf_p_inflow_sel["VP_QMax"] = 0.05  ##0.05 (50l/s) pro klasicky inlet - zustava stejne pro ga50
        gdf_p_inflow_sel["pBlockage"] = 0.75

    if not gdf_p_ouflow_sel.empty:
        gdf_p_ouflow_sel["VP_QMax"] = (
            Q_max_outlet_pipes_GA50  ##redukce z 10 m3/s na 2.5 for outflow pro ga50
        )

        # ## save full open VP with 50% blockage
        # gdf_p_inflow_sel["pBlockage"] = 0.5
        # gdf_p_ouflow_sel["VP_QMax"] = 2   ##from 10 in GA00 to 2 in GA50

    gdf_p50 = merge([gdf_p_ouflow_sel, gdf_p_inflow_sel])

    gdf_p50["Lag_Approach"] = "None"  ## 'None' 'Decay'

    # # Check for invalid Lag_Approach values
    # invalid_lag_50 = gdf_p50[~gdf_p50['Lag_Approach'].isin(['None', 'Decay'])]
    # if not invalid_lag_50.empty:
    #     invalid_ids_50 = invalid_lag_50['VP_Network_ID'].unique()
    #     gdf_p50.loc[gdf_p50['VP_Network_ID'].isin(invalid_ids_50), 'Lag_Approach'] = 'None'
    #
    # # Check for consistent Lag_Approach and Lag_Value within each VP_Network_ID group
    # grouped_50 = gdf_p50.groupby('VP_Network_ID')
    # for name, group in grouped_50:
    #     if group['Lag_Approach'].nunique() > 1 or group['Lag_Value'].nunique() > 1:
    #         gdf_p50.loc[gdf_p50['VP_Network_ID'] == name, 'Lag_Approach'] = 'None'

    if not gdf_p50.empty:
        gdf_p50.to_file(out_pit_ga50, driver="GPKG", layer=out_pit_ga50_name)
    else:
        print("         - no VP's pits for the domain")

    ## create tunnels layer
    gdf_p_tunnel = select_by_attribute(
        input=path_in_pit_VP_P, field="if_type", method=0, value="tunnel"
    )

    overlay_tunnel = buffer(
        input=code, distance=-50, field="distance", output=None, driver="GPKG"
    )

    gdf_p_tunnel_sel = select_by_location(
        select_from=gdf_p_tunnel,
        overlay=overlay_tunnel,
        output=None,
        method="intersect",
    )

    if not gdf_p_tunnel_sel.empty:
        # gdf_p_tunnel_sel["Conn_No"] = -1  ## add msa 2025-12-29 to set only one cell for inlets and outlets; enable add point zsh fo inlets in waterbody
        gdf_p_tunnel_sel["VP_QMax"] = Q_max_outlet_tunnel  ##velky prutok pro koryta
        gdf_p_tunnel_sel["Inlet_Type"] = "Tunnel_culvert_2x1"  ##velky prutok pro koryta

        ## save full open tunnel (as vp)
        gdf_p_tunnel_sel.to_file(out_tunnel, driver="GPKG", layer=out_tunnel_name)

        ## save open VP with 50% blockage
        gdf_p_tunnel_sel["pBlockage"] = 0.5
        gdf_p_tunnel_sel.to_file(out_tunnel_50, driver="GPKG", layer=out_tunnel_50_name)

    else:
        print("         - no pits's pits for the domain")


##add by msa 2025-12-29, add zsh polygon for dtm editing
def extract_dtm_zsh_to_domain(
    workspace,
    domain,
    path_in_dtm_zsh_R,
):
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")
    # area = domain.split("_")[0]

    os.makedirs(os.path.join(workspace, domain, "model", "gis", "dtm"), exist_ok=True)

    ## output layers
    out_zsh = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_polygon_" + domain + "_R.gpkg"
    )
    if os.path.isfile(out_zsh):
        gdal.Unlink(out_zsh)  # delete file if exists

    gdf_r = select_by_location(
        select_from=path_in_dtm_zsh_R,
        overlay=code,
        output=None,
        method="intersect",
    )

    if not gdf_r.empty:
        gdf_r = gdf_r.explode(ignore_index=True, index_parts=True)
        gdf_r["Shape_Widt"] = 0

        gdf_r_part = gdf_r[gdf_r["Z"].notnull()].copy()
        gdf_r_part2 = gdf_r[
            gdf_r["Z"].isnull()
        ].copy()  ## add vertices with max.distance half of meshsize

        gdf_r_part["Shape_Opti"] = "NO MERGE"  ## vezme se vsude vyska z Z
        gdf_r_part2["Shape_Opti"] = "MERGE ALL"  ## vsude se vezme vyska terenu z dtm

        gdf_r = pd.concat([gdf_r_part, gdf_r_part2])
        gdf_r.geometry = gdf_r.geometry.apply(lambda x: make_valid(x))
        # gdf_r.geometry = gdf_l.geometry.apply(
        #     lambda x: LineString([(p[0], p[1]) for p in x.coords])
        # )
        gdf_r.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_polygon_" + domain + "_R")

    else:
        print("         - no zsh polygons for the domain")


##add by msa 2026-01-30, add zsh line add for dtm editing
def extract_dtm_zsh_L_to_domain(
    workspace,
    domain,
    path_in_dtm_zsh_L,
    output_resolution,
):
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")
    # area = domain.split("_")[0]

    os.makedirs(os.path.join(workspace, domain, "model", "gis", "dtm"), exist_ok=True)

    ## output layers
    out_zsh = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_add_" + domain + "_L.gpkg"
    )
    if os.path.isfile(out_zsh):
        gdal.Unlink(out_zsh)  # delete file if exists
    gdf_r = select_by_location(
        select_from=path_in_dtm_zsh_L,
        overlay=code,
        output=None,
        method="intersect",
    )

    if not gdf_r.empty:
        gdf_r = gdf_r.explode(ignore_index=True, index_parts=True)
        gdf_r["Shape_Widt"] = round(float(output_resolution) * (2**0.5))

        gdf_r_part = gdf_r[gdf_r["Z"].notnull()].copy()
        # gdf_r_part2 = gdf_r[gdf_r["Z"].isnull()].copy()  ## add vertices with max.distance half of meshsize

        # gdf_r_part["Shape_Opti"] = 'NO MERGE' ## vezme se vsude vyska z Z
        # gdf_r_part2["Shape_Opti"] = 'MERGE ALL'  ## vsude se vezme vyska terenu z dtm

        # gdf_r = pd.concat([gdf_r_part, gdf_r_part2])
        gdf_r_part.geometry = gdf_r.geometry.apply(lambda x: make_valid(x))
        # gdf_r.geometry = gdf_l.geometry.apply(
        #     lambda x: LineString([(p[0], p[1]) for p in x.coords])
        # )
        gdf_r_part.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_add_" + domain + "_L")

    else:
        print("         - no zsh ADD line for the domain")


def extract_culvert_to_domain(
    workspace,
    domain,
    tu_zsh_lines_switch,
    d_path_in_zsh_culvert_L,
    d_path_in_zsh_culvert_P,
    output_resolution,
):
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")
    area = domain.split("_")[0]

    os.makedirs(os.path.join(workspace, domain, "model", "gis", "dtm"), exist_ok=True)

    ## output layers
    out_zsh = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_culvert_" + domain + ".gpkg"
    )
    if os.path.isfile(out_zsh):
        gdal.Unlink(out_zsh)  # delete file if exists

    gdf_l = select_by_location(
        select_from=d_path_in_zsh_culvert_L,
        overlay=code,
        output=None,
        method="intersect",
    )

    if not gdf_l.empty:
        gdf_l = gdf_l.explode(ignore_index=True, index_parts=True)

        ### uprava msa 260128
        # 1. Set 'p_CWF' = 1 where 'Shape_Widt' > 0
        gdf_l.loc[gdf_l["Shape_Widt"] > 0, "pCFW"] = 1

        # 2. Set 'Shape_Widt' = 7 where 'Shape_Widt' is null or 0
        gdf_l.loc[
            gdf_l["Shape_Widt"].isnull() | (gdf_l["Shape_Widt"] == 0), "Shape_Widt"
        ] = round(float(output_resolution) * (2**0.5))

        # 3. Set 'p_CWF' = 0.7 where 'p_CWF' is null or 0
        gdf_l.loc[gdf_l["pCFW"].isnull() | (gdf_l["pCFW"] == 0), "pCFW"] = float(
            tu_zsh_lines_switch["zsh_culvert_CFW"]
        )
        gdf_l.loc[gdf_l["pFLC"].isnull() | (gdf_l["pFLC"] == 0), "pFLC"] = float(
            tu_zsh_lines_switch["zsh_culvert_FLC"]
        )

        ### original pred upravou msa 260128
        # gdf_l = gdf_l.explode(ignore_index=True, index_parts=True)
        # gdf_l_part = gdf_l[gdf_l["Shape_Widt"] == 0 ].copy()
        # gdf_l = gdf_l[gdf_l["Shape_Widt"] != 0].copy()
        #
        # gdf_l_part["Shape_Widt"] = round(float(output_resolution) * (2**0.5))
        # gdf_l = pd.concat([gdf_l, gdf_l_part])

        ## msa change 260128 - vypoustim, moc nastaveni dodelat napriste, aby se davali hodnoty jen tamkde uz nein nic zadaneho
        # gdf_l["pCFW"] = float(tu_zsh_lines_switch["zsh_culvert_CFW"])
        # gdf_l["pFLC"] = float(tu_zsh_lines_switch["zsh_culvert_FLC"])

        gdf_l["Z"] = float(-99999)
        gdf_l.geometry = gdf_l.geometry.apply(lambda x: make_valid(x))
        gdf_l.geometry = gdf_l.geometry.apply(
            lambda x: LineString([(p[0], p[1]) for p in x.coords])
        )
        gdf_l.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_culvert_{domain}_L")

        overlay = buffer(
            input=gdf_l, distance=0.001, field="distance", output=None, driver="GPKG"
        )
        gdf_p = select_by_location(
            select_from=d_path_in_zsh_culvert_P,
            overlay=overlay,
            output=None,
            method="intersect",
            driver="GPKG",
        )
        if not gdf_p.empty:
            gdf_p["dZ"] = float(tu_zsh_lines_switch["zsh_culvert_dZ"])
            gdf_p.geometry = gdf_p.geometry.apply(lambda x: make_valid(x))
            gdf_p.geometry = gdf_p.geometry.apply(lambda x: Point(x.x, x.y))

            gdf_p.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_culvert_{domain}_P")
        else:
            print("         - no valid points for culvert for the domain")
    else:
        print("         - no culvert for the domain")


def extract_channel_to_domain(
    workspace,
    domain,
    tu_zsh_lines_switch,
    d_path_in_zsh_channel_L,
    d_path_in_zsh_channel_P,
    output_resolution,
):
    import warnings

    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")
    area = domain.split("_")[0]
    in_zsh_L = d_path_in_zsh_channel_L
    in_zsh_P = d_path_in_zsh_channel_P

    os.makedirs(os.path.join(workspace, domain, "model", "gis", "dtm"), exist_ok=True)

    ## output layers
    out_zsh = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_channel_" + domain + ".gpkg"
    )
    if os.path.isfile(out_zsh):
        gdal.Unlink(out_zsh)

    gdf_l = select_by_location(
        select_from=in_zsh_L,
        overlay=code,
        output=None,
        method="intersect",
    )
    if not gdf_l.empty:
        gdf_l_part = gdf_l[gdf_l["Shape_Widt"] == 0].copy()
        gdf_l = gdf_l[gdf_l["Shape_Widt"] != 0].copy()

        gdf_l_part["Shape_Widt"] = round(float(output_resolution) * (2**0.5))
        gdf_l = pd.concat([gdf_l, gdf_l_part])
        gdf_l["Z"] = float(-99999)
        gdf_l.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_channel_{domain}_L")

        overlay = buffer(
            input=gdf_l, distance=0.001, field="distance", output=None, driver="GPKG"
        )
        gdf_p = select_by_location(
            select_from=in_zsh_P,
            overlay=overlay,
            output=None,
            method="intersect",
        )
        if not gdf_p.empty:
            gdf_p["dZ"] = float(tu_zsh_lines_switch["zsh_channel_dZ"])
            gdf_p["Z"] = np.where(
                (gdf_p["Z"] + gdf_p["dZ"]) == 0, gdf_p["Z"] + 0.001, gdf_p["Z"]
            )
            gdf_p.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_channel_{domain}_P")
        else:
            print("         - no valid points for channel for the domain")
    else:
        print("         - no channel for the domain")


def extract_levee_to_domain(
    workspace,
    domain,
    tu_zsh_lines_switch,
    in_zsh_L,
    in_zsh_P,
    output_resolution,
):
    import warnings

    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")

    os.makedirs(os.path.join(workspace, domain, "model", "gis", "dtm"), exist_ok=True)

    ## output layers
    out_zsh = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_levee_" + domain + ".gpkg"
    )
    if os.path.isfile(out_zsh):
        gdal.Unlink(out_zsh)

    gdf_l = select_by_location(
        select_from=in_zsh_L,
        overlay=code,
        output=None,
        method="intersect",
    )
    if not gdf_l.empty:
        gdf_l_part = gdf_l[gdf_l["Shape_Widt"] == 0].copy()
        gdf_l = gdf_l[gdf_l["Shape_Widt"] != 0].copy()

        gdf_l_part["Shape_Widt"] = round(float(output_resolution) * (2**0.5))
        gdf_l = pd.concat([gdf_l, gdf_l_part])
        gdf_l["Z"] = float(-99999)
        gdf_l.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_levee_{domain}_L")

        overlay = buffer(
            input=gdf_l, distance=0.001, field="distance", output=None, driver="GPKG"
        )
        gdf_p = select_by_location(
            select_from=in_zsh_P,
            overlay=overlay,
            output=None,
            method="intersect",
        )
        if not gdf_p.empty:
            gdf_p["dZ"] = float(tu_zsh_lines_switch["zsh_levee_dZ"])
            gdf_p.to_file(out_zsh, driver="GPKG", layer=f"2d_zsh_levee_{domain}_P")

        else:
            print("         - no valid points for levees for the domain")
    else:
        print("         - no levee for the domain")


def extract_bc_dbase_to_local(
    workspace, domain, path_bc_dbase_inputs, Q_peril, bc_files
):

    bc_dbase_common_path = os.path.join(path_bc_dbase_inputs, domain.split("_")[0])
    bc_dbase_local_path = os.path.join(workspace, domain, "bc_dbase")
    bc_dbase_files = [
        "bc_dbase_R",
        "bc_dbase_L",
        "bc_dbase_C",
        "bc_dbase_rf",
    ]
    bc_definition_file = list(
        set(bc_files) ^ set(bc_dbase_files)
    )  # HT0 and others...- to be find in bc_dbase csv to filter out
    # list_bc = os.listdir(bc_dbase_common_path)
    # for name in glob.glob(bc_dbase_common_path):
    #     print(name)
    os.makedirs(bc_dbase_local_path, exist_ok=True)

    names = []
    if Q_peril == "fl":
        bc_sa_R = os.path.join(
            workspace, domain, "model", "gis", f"2d_sa_IN_{domain}_R.gpkg"
        )
        if get_bc_dbase_target(bc_sa_R, Q_peril):
            names.append(get_bc_dbase_target(bc_sa_R, Q_peril))
        bc_sa_L = os.path.join(
            workspace, domain, "model", "gis", f"2d_sa_IN_{domain}_L.gpkg"
        )
        if get_bc_dbase_target(bc_sa_L, Q_peril):
            names.append(get_bc_dbase_target(bc_sa_L, Q_peril))
        names = [val for sublist in names for val in sublist]

        if not names:
            print(" ... There is no valid SA_IN gpkg.", end="")

        for bc_dbase_type in bc_dbase_files:
            for source_name in names:
                try:
                    source = os.path.join(
                        bc_dbase_common_path, bc_dbase_type, source_name + ".csv"
                    )
                    target = os.path.join(
                        bc_dbase_local_path, bc_dbase_type, source_name + ".csv"
                    )
                    remove_TU_files(target)
                    if not os.path.isfile(source):
                        if "bc_dbase_C" in bc_dbase_type:
                            print(
                                f" ... There is no valid SA_IN name {source_name} for {bc_dbase_type}.",
                                end="",
                            )
                        elif "bc_dbase_L" in bc_dbase_type and "stream" in source_name:
                            print(
                                f" ... There is no valid SA_IN name {source_name} for {bc_dbase_type}.",
                                end="",
                            )
                        elif (
                            "bc_dbase_R" in bc_dbase_type
                            and not "stream" in source_name
                        ):
                            print(
                                f" ... There is no valid SA_IN name {source_name} for {bc_dbase_type}.",
                                end="",
                            )
                    else:
                        if os.path.isfile(source):
                            if "bc_dbase_C" in bc_dbase_type:
                                shutil.copy2(source, target)
                            elif (
                                "bc_dbase_L" in bc_dbase_type
                                and "stream" in source_name
                            ):
                                shutil.copy2(source, target)
                            elif (
                                "bc_dbase_R" in bc_dbase_type
                                and not "stream" in source_name
                            ):
                                shutil.copy2(source, target)
                except:
                    print(f" ... There is no valid SA_IN name {source_name}.", end="")

        names_bc_content = bc_definition_file + names
        for source_name in bc_files:
            source = os.path.join(bc_dbase_common_path, source_name + ".csv")
            if os.path.exists(source):
                target = os.path.join(bc_dbase_local_path, source_name + ".csv")
                remove_TU_files(target)
                if source_name in [
                    "bc_dbase_R",
                    "bc_dbase_L",
                    "bc_dbase_C",
                    "bc_dbase_rf",
                ]:
                    filter_csvfile(source, target, names_bc_content)
                else:
                    shutil.copy2(source, target)

    if Q_peril in ("pl", "pl_city"):
        # bc_dbase_common_path = os.path.join(path_bc_dbase_inputs, "IDF")
        ## msa , mozna bude potreba pouzit natvrdo zadanou estu pro IDF; pokud je lokace IDF v jine casti adresaroveho stromu

        bc_dbase_common_path = (
            tu_sources.bc_dbase_common_path_idf
            if Q_peril == "pl_city"
            else os.path.join(path_bc_dbase_inputs, "IDF")
        )

        idf_list = os.listdir(bc_dbase_common_path)
        rf_R = os.path.join(workspace, domain, "model", "gis", f"2d_rf_{domain}_R.gpkg")

        if get_bc_dbase_target(rf_R, Q_peril):
            names.append(get_bc_dbase_target(rf_R, Q_peril))
        names = [val for sublist in names for val in sublist]
        if not names:
            print(" ... There is no valid rf shapefile.", end="")

        for source_name in names:
            try:
                for filename in fnmatch.filter(idf_list, f"{source_name}_*.csv"):
                    source = os.path.join(bc_dbase_common_path, filename)
                    target = os.path.join(bc_dbase_local_path, filename)
                    remove_TU_files(target)
                    shutil.copy2(source, target)
            except:
                print(f" ... There is no valid rf name {source_name}_*.", end="")

        names_bc_content = bc_definition_file + names
        for source_name in bc_files:
            source = os.path.join(bc_dbase_common_path, source_name + ".csv")
            if os.path.exists(source):
                target = os.path.join(bc_dbase_local_path, source_name + ".csv")
                remove_TU_files(target)
                if source_name in [
                    "bc_dbase_R",
                    "bc_dbase_L",
                    "bc_dbase_C",
                    "bc_dbase_rf",
                ]:
                    string_names_bc_content = [str(int) for int in names_bc_content]
                    filter_csvfile(source, target, string_names_bc_content)
                else:
                    shutil.copy2(source, target)


def get_bc_dbase_target(shape, Q_peril):
    print(f" ... Extracting names from {shape} for {Q_peril} BC dbase.")
    if os.path.exists(shape):
        df = gpd.read_file(shape)
        if Q_peril == "fl":
            return df["Name"].to_list()
        if Q_peril == "pl" or Q_peril == "rf" or Q_peril == "pl_city":
            df["ID_RF"] = df["ID_RF"].astype(str)
            return df["ID_RF"].to_list()


def remove_TU_files(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)


def filter_csvfile(input_file, output_file, filter_list):
    df = pd.read_csv(input_file, low_memory=False)
    df["Name"] = df["Name"].astype(str)
    df = df[df["Name"].isin(filter_list)]
    df.to_csv(output_file, index=False)


def zpts_features_order(domain, list_zsh_reverse_domain):
    if domain in list_zsh_reverse_domain:
        set_zpts_order = "levee_first"
    else:
        set_zpts_order = "levee_last"
    return set_zpts_order


def copy_domain_to_ts(workspace, domain, workspace_ts, back_up=True):
    folder_to_compute = os.path.join(workspace, domain)

    destination_folder = os.path.join(workspace_ts, domain.split("_")[0], domain)

    ignore = shutil.ignore_patterns("*.lock")

    if os.path.exists(folder_to_compute):
        if os.path.exists(destination_folder):
            if back_up:
                back_up_folder = os.path.join(
                    os.path.dirname(destination_folder), "_backUp_" + str(date.today())
                )

                if not os.path.exists(back_up_folder):
                    os.makedirs(back_up_folder)

                back_up_domain_path = os.path.join(back_up_folder, domain)

                try:
                    if os.path.exists(back_up_domain_path):
                        shutil.rmtree(back_up_domain_path)
                        shutil.move(destination_folder, back_up_domain_path)

                        shutil.copytree(
                            folder_to_compute, destination_folder, ignore=ignore
                        )
                        print("...done")
                        return True

                    else:
                        shutil.rmtree(destination_folder)
                        shutil.copytree(
                            folder_to_compute, destination_folder, ignore=ignore
                        )
                        print("...done")
                        return True
                except:
                    print(
                        "...Unable to backup files. Please backup/delete folder manually. Than copy domain again"
                    )
                    return False
        else:
            shutil.copytree(folder_to_compute, destination_folder, ignore=ignore)

            print("...done")
            return True
    else:
        print(
            f"...Skip. Domain does not exist on {os.path.dirname(folder_to_compute)}."
        )
        return False


def check_drainage(workspace, domain):
    """
    Check if drainage is present in the domain
    """
    drainage_file_1 = os.path.join(
        workspace, domain, "model", "grid", f"2d_drain_layer_1_{domain}.tif"
    )
    drainage_file_2 = os.path.join(
        workspace, domain, "model", "grid", f"2d_drain_layer_2_{domain}.tif"
    )
    drain_1, drain_2 = False, False
    if os.path.isfile(drainage_file_1):
        drain_1 = True
    if os.path.isfile(drainage_file_2):
        drain_2 = True
    return drain_1, drain_2


## msa mayby check for VP will be necessary
def check_vp(workspace, domain):
    """
    Check if drainage is present in the domain
    """
    vp_file = os.path.join(
        workspace, domain, "model", "gis", "1d_pit_" + domain + "_P.gpkg"
    )

    vp_1 = False
    if os.path.isfile(vp_file):
        vp_1 = True
    return vp_1


## msa add 2025-12-29 for config file
def check_dtm_zsh(workspace, domain):
    """
    Check if zsh polygon is present in the domain
    """
    zsh_file = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_polygon_" + domain + "_R.gpkg"
    )

    zsh_1 = False
    if os.path.isfile(zsh_file):
        zsh_1 = True
    return zsh_1


## msa add 2025-12-29 for config file
def check_dtm_zsh_L(workspace, domain):
    """
    Check if zsh add line is present in the domain
    """
    zsh_file = os.path.join(
        workspace, domain, "model", "gis", "dtm", "2d_zsh_add_" + domain + "_L.gpkg"
    )

    zsh_2 = False
    if os.path.isfile(zsh_file):
        zsh_2 = True
    return zsh_2


def get_min_value(domain, workspace):
    """Extract min value from DTM

    :param domain: Domain name
    :param workspace: Workspace path
    """
    files = glob.glob(
        os.path.join(workspace, domain, "model", "grid", f"dtm_{domain}*tif")
    )
    mins = []
    if files:
        for file in files:
            # get min value from raster using gdal
            ds = gdal.Open(file, gdal.GA_ReadOnly)
            if ds is not None:
                band = ds.GetRasterBand(1)
                stats = band.GetStatistics(True, True)
                min_value = stats[0]  # min value
                ds = None
                if min_value is not None:
                    mins.append(min_value)

    if mins:
        min_value = min(mins)
        return round(min_value, 1)
    return None


def write_tgc_file(
    domain,
    workspace,
    dtms_combinations,
    restart=False,
    main_resolution=None,
    mesh_parameters_grid=None,
):
    tgc_file_name = os.path.join(workspace, domain, "model", domain + ".tgc")
    coarsest_resolution = sorted(dtms_combinations)[0]
    if main_resolution is not None:
        coarsest_resolution = main_resolution
    if mesh_parameters_grid is None:
        dtm_grid = os.path.join(
            workspace,
            domain,
            "model",
            "grid",
            f"dtm_{domain}_{str(coarsest_resolution).zfill(2)}.tif",
        )
        mesh_parameters_grid = dtm_grid

    drain_1, drain_2 = check_drainage(workspace, domain)

    ncols, nrows, xllcorner, yllcorner, cellsize = get_DTM_metadata(
        mesh_parameters_grid
    )
    shifted = float(xllcorner) - 200
    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{str(int(cellsize))}m", "model")
        )
    )
    try:
        min_value = get_min_value(domain, workspace)  # NASTAVIT IWL PODLE MAILU
        if min_value is not None:
            zpts = min_value - 2.5
            iwl = min_value - 2
        else:
            zpts = tu_data.tu_parameters["tu_values_default"]["Set_Zpts"]
            iwl = tu_data.tu_parameters["tu_values_default"]["Set_IWL"]
    except Exception:
        zpts = tu_data.tu_parameters["tu_values_default"]["Set_Zpts"]
        iwl = tu_data.tu_parameters["tu_values_default"]["Set_IWL"]
    template = environment_loader.get_template("domain.tgc")
    content = template.render(
        domain=domain,
        x_min=xllcorner,
        y_min=yllcorner,
        x_min_shift=shifted,
        ncols=ncols * cellsize,
        nrows=nrows * cellsize,
        cellsize=cellsize,
        restart=restart,
        soil=tu_data.tu_parameters["tu_values_default"]["Set_Soil"],
        soil_other=tu_data.tu_parameters["tu_values_default"]["Set_Soil_Other"],
        zpts=zpts,
        iwl=iwl,
        mat=tu_data.tu_parameters["tu_values_default"]["Set_Mat"],
        drainage_layer_1=drain_1,
        drainage_layer_2=drain_2,
    )
    with open(tgc_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tcf_local_inputs(
    domain,
    workspace,
    is_levee=False,
    is_channel=False,
    is_culvert=False,
    is_zsh_R=False,
    is_zsh_L=False,
    zpt_order="auto",
    mesh_resolution=tu_data.tu_parameters["pluvial_resolution"],
):
    tcf_file_name = os.path.join(
        workspace, domain, "model", f"LocalInputs_{domain}.tcf"
    )
    is_channel_check, is_levee_check, is_culvert_check = check_culverts_levees_channels(
        domain, workspace
    )
    is_wb, is_buildings, is_cwf = check_additional_layers(domain, workspace)
    is_zsh_polygon_check = check_dtm_zsh(workspace, domain)
    is_zsh_L_check = check_dtm_zsh_L(workspace, domain)  ## check linii pro add vysky
    print(f"is_zsh_polygon_check {is_zsh_polygon_check}")
    print(f"is_zsh_L_check {is_zsh_polygon_check}")

    print(f"is_buildings {is_buildings}")
    print(f"is_cwf {is_cwf}")

    if is_levee_check or is_levee:
        is_levee = True
    if is_channel_check or is_channel:
        is_channel = True
    if is_culvert_check or is_culvert:
        is_culvert = True
    if is_zsh_polygon_check or is_zsh_R:
        is_zsh_R = True
    if is_zsh_L_check or is_zsh_L:
        is_zsh_L = True
    if any(
        zpt_order.lower() in x for x in ["auto", "levee_last", "levees_last", "default"]
    ):
        order = False
        print("\n\t\tOrder for levees first is False", end="")
    else:
        order = True
    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "model")
        )
    )
    template = environment_loader.get_template("LocalInputs_domain.tcf")
    content = template.render(
        domain=domain,
        levee=is_levee,
        channel=is_channel,
        culvert=is_culvert,
        order=order,
        wb=is_wb,
        buildings=is_buildings,
        cwf_buildings=is_cwf,
        zsh_R=is_zsh_R,
        zsh_L=is_zsh_L,
    )
    with open(tcf_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_ecf_file(
    domain,
    workspace,
    mesh_resolution=tu_data.tu_parameters["pluvial_resolution"],
    is_vp00=False,
    is_vp50=False,
    is_tnl=False,
    is_tnl_50=False,
):

    is_vpga00, is_vpga50, is_tunnel, is_tunnel_50 = check_vpga00_vpga50_tunnel(
        domain, workspace
    )
    is_pit00 = False
    is_pit50 = False
    if is_vpga00 or is_vp00:
        is_pit00 = True
    if is_vpga50 or is_vp50:
        is_pit50 = True
    if is_tunnel or is_tnl:
        is_tunnel = True
    if is_tunnel_50 or is_tnl_50:
        is_tunnel_50 = True

    template_dir = os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "runs")
    template_name = "domain.ecf"
    print(f"[DEBUG] ECF template source: {os.path.join(template_dir, template_name)}")

    for g in ["GA00", "GA50", "tunnel", "tunnel_50"]:
        ecf_file_name = os.path.join(workspace, domain, "runs", f"{domain}_{g}.ecf")
        print(f"[DEBUG] ECF output target ({g}): {ecf_file_name}")
        environment_loader = Environment(loader=FileSystemLoader(template_dir))
        template = environment_loader.get_template(template_name)

        content = ""
        if g == "GA00":
            content = template.render(
                domain=domain, ga=g, vp=is_pit00, tunnel=is_tunnel
            )
        elif g == "GA50":
            content = template.render(
                domain=domain, ga=g, vp=is_pit50, tunnel=is_tunnel
            )
        elif g == "tunnel":
            content = template.render(domain=domain, ga=g, vp=False, tunnel=is_tunnel)
        elif g == "tunnel_50":
            content = template.render(
                domain=domain, ga=g, vp=False, tunnel=is_tunnel_50
            )
        with open(ecf_file_name, "w") as domain_tfc:
            domain_tfc.write(content)


def write_tbc_file(
    domain, workspace, mesh_resolution=tu_data.tu_parameters["pluvial_resolution"]
):
    tbc_file_name = os.path.join(workspace, domain, "model", f"{domain}.tbc")
    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{mesh_resolution}m", "model")
        )
    )
    template = environment_loader.get_template("domain.tbc")
    content = template.render(
        domain=domain,
    )
    with open(tbc_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def check_culverts_levees_channels(domain, workspace) -> [bool, bool, bool]:
    folder = os.path.join(workspace, domain, "model", "gis", "dtm")
    channel = os.path.join(folder, "2d_zsh_channel_" + domain + ".gpkg")
    levee = os.path.join(folder, "2d_zsh_levee_" + domain + ".gpkg")
    culvert = os.path.join(folder, "2d_zsh_culvert_" + domain + ".gpkg")
    is_channel, is_levee, is_culvert = False, False, False
    if os.path.isfile(channel):
        is_channel = True
    if os.path.isfile(levee):
        is_levee = True
    if os.path.isfile(culvert):
        is_culvert = True
    return is_channel, is_levee, is_culvert


def check_vpga00_vpga50_tunnel(domain, workspace) -> [bool, bool, bool]:
    folder = os.path.join(workspace, domain, "model", "gis")
    vp_pit_ga00 = os.path.join(folder, "1d_pit_" + domain + "_GA00_P.gpkg")
    vp_pit_ga50 = os.path.join(folder, "1d_pit_" + domain + "_GA50_P.gpkg")
    vp_tunnel = os.path.join(folder, "1d_pit_" + domain + "_tunnel_P.gpkg")
    vp_tunnel_50 = os.path.join(folder, "1d_pit_" + domain + "_tunnel_50_P.gpkg")

    is_vp_ga00, is_vp_ga50, is_tunnel, is_tunnel_50 = False, False, False, False
    if os.path.isfile(vp_pit_ga00):
        is_vp_ga00 = True
    if os.path.isfile(vp_pit_ga50):
        is_vp_ga50 = True
    if os.path.isfile(vp_tunnel):
        is_tunnel = True
    if os.path.isfile(vp_tunnel_50):
        is_tunnel_50 = True
    return vp_pit_ga00, vp_pit_ga50, is_tunnel, is_tunnel_50


def write_restart_file(
    domain, workspace, mesh_resolution=tu_data.tu_parameters["pluvial_resolution"]
):
    restart_file_name = os.path.join(workspace, domain, "runs", f"Restart_{domain}.tcf")
    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "runs")
        )
    )
    template = environment_loader.get_template("Restart_domain.tcf")
    content = template.render(
        domain=domain,
    )
    with open(restart_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tef_file(
    domain,
    workspace,
    watershed,
    mesh_resolution=tu_data.tu_parameters["pluvial_resolution"],
):
    tef_file_name = os.path.join(workspace, domain, "runs", f"{domain}.tef")
    template_dir = os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "runs")
    template_name = "domain.tef"
    print(f"[DEBUG] TEF template source: {os.path.join(template_dir, template_name)}")
    print(f"[DEBUG] TEF output target: {tef_file_name}")
    environment_loader = Environment(loader=FileSystemLoader(template_dir))
    template = environment_loader.get_template(template_name)
    content = template.render(
        domain=domain,
        area=watershed,
        # soil_file=os.path.basename(config_variables["soil_file"]),
    )

    content = content.lstrip("\ufeff")
    with open(tef_file_name, "w", encoding="utf-8", newline="") as domain_tfc:
        domain_tfc.write(content)


def write_tcf_file(
    domain,
    workspace,
    quadtree=False,
    mesh_resolution=tu_data.tu_parameters["pluvial_resolution"],
):
    tcf_file_name = os.path.join(workspace, domain, "runs", f"{domain}.tcf")
    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "runs")
        )
    )
    template = environment_loader.get_template("domain.tcf")
    content = template.render(
        domain=domain,
        epsg=config_variables["epsg"],
        quadtree=quadtree,
        tuflow_version=config_variables["tuflow_version"],
    )
    with open(tcf_file_name, "w") as domain_tfc:
        domain_tfc.write(content)

    if quadtree:
        quadtree_file_name = os.path.join(
            workspace, domain, "model", f"Quadtree_{domain}.tcf"
        )
        environment_loader = Environment(
            loader=FileSystemLoader(
                os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "model")
            )
        )
        template = environment_loader.get_template("Quadtree_domain.tcf")
        content = template.render(
            domain=domain, tuflow_version=config_variables["tuflow_version"]
        )
        with open(quadtree_file_name, "w") as domain_tfc:
            domain_tfc.write(content)

        qcf_file_name = os.path.join(
            workspace, domain, "model", f"quadtree_{domain}.qcf"
        )
        environment_loader = Environment(
            loader=FileSystemLoader(
                os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "model")
            )
        )
        template = environment_loader.get_template("domain.qcf")
        content = template.render(domain=domain)
        with open(qcf_file_name, "w") as domain_tfc:
            domain_tfc.write(content)


def extract_dtms(
    domain, dtm_grid, q_peril: dict, code_buff, DTM_switch, watershed=None, city=None
):
    area = domain.split("_")[0]
    area_key = area[0]
    dtm_path = None
    if q_peril["peril"] == "fl":
        for resolution in tu_sources.DTMs_combination[DTM_switch]:
            resolution = str(resolution)
            print(f"       - extracting DTM, {resolution}m", end="")
            if str(area_key) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key)][resolution]
            elif str(area_key)[0] in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key[0])][resolution]
            else:
                raise FileNotFoundError(
                    f"DTM path not found, check {area_key} or {str(area_key)[0]}"
                )
            src = gdal.OpenEx(dtm_path)
            data_type = src.GetRasterBand(1).DataType
            del src
            if data_type in (
                gdal.GDT_Int32,
                gdal.GDT_Int16,
                gdal.GDT_Byte,
                gdal.GDT_UInt16,
                gdal.GDT_UInt32,
            ):
                raster = extract_by_mask_rasterized(
                    raster=dtm_path,
                    mask=code_buff,
                    output=f"/vsimem/{domain}.tif",
                )
                array, no_data = get_array_from_raster(raster)
                array = array.astype(float)
                array = np.where(array != no_data, array / 100.0, no_data)
                save_array_with_type(
                    array,
                    dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                    raster,
                    gdal.GDT_Float32,
                    no_data,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
                gdal.Unlink(f"/vsimem/{domain}.tif")

            else:
                extract_by_mask_rasterized(
                    # raster=d_DTM_path[f"{area_key}_{str(resolution).zfill(2)}"],
                    raster=dtm_path,
                    mask=code_buff,
                    output=dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
            print("... done")
    elif q_peril["peril"] == "pl":
        resolution = q_peril["resolution"]
        if q_peril["resolution"] not in q_peril["allowed_resolutions"]:
            raise ValueError(
                f"Resolution {q_peril['resolution']} is not specified in reference file!"
            )

        print(f"       - extracting DTM, {resolution}m", end="")
        if tu_data.tu_parameters["tu_values_default"]["Select_DTM_by"] == "domain":
            if str(area_key) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key)][resolution]
            elif str(area_key)[0] in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key[0])][resolution]
            else:
                raise FileNotFoundError(
                    f"DTM path not found, check {area_key} or {str(area_key)[0]}"
                )
        elif tu_data.tu_parameters["tu_values_default"]["Select_DTM_by"] == "watershed":
            if str(watershed) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(watershed)][resolution]
            elif str(watershed)[0] in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(watershed[0])][resolution]
            else:
                raise FileNotFoundError(
                    f"DTM path not found, check {watershed} or {str(watershed)[0]}"
                )
        src = gdal.OpenEx(dtm_path)
        data_type = src.GetRasterBand(1).DataType
        del src
        if data_type in (
            gdal.GDT_Int32,
            gdal.GDT_Int16,
            gdal.GDT_Byte,
            gdal.GDT_UInt16,
            gdal.GDT_UInt32,
        ):
            x_res, y_res = get_resolution(dtm_path)
            bounds = pyogrio.read_info(code_buff)["total_bounds"]
            bounds = snap_extent(
                [bounds[0], bounds[2], bounds[1], bounds[3]],
                get_extent(dtm_path),
                x_res,
            )
            raster = f"/vsimem/{domain}.tif"
            gdal.Warp(
                raster,
                dtm_path,
                options=gdal.WarpOptions(
                    outputBounds=[bounds[0], bounds[2], bounds[1], bounds[3]],
                    xRes=x_res,
                    yRes=abs(y_res),
                    resampleAlg="nearest",
                ),
            )

            array, no_data = get_array_from_raster(raster)
            array = array.astype(float)
            array = np.where(array != no_data, array / 100.0, no_data)
            save_array_with_type(
                array,
                dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                raster,
                gdal.GDT_Float32,
                no_data,
                creation_options=[
                    "COMPRESS=DEFLATE",
                    "NUM_THREADS=ALL_CPUS",
                    "ZLEVEL=12",
                ],
            )
            gdal.Unlink(f"/vsimem/{domain}.tif")

        else:
            x_res, y_res = get_resolution(dtm_path)
            bounds = pyogrio.read_info(code_buff)["total_bounds"]
            bounds = snap_extent(
                [bounds[0], bounds[2], bounds[1], bounds[3]],
                get_extent(dtm_path),
                get_resolution(dtm_path)[0],
            )

            gdal.Warp(
                dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                dtm_path,
                options=gdal.WarpOptions(
                    outputBounds=[bounds[0], bounds[2], bounds[1], bounds[3]],
                    xRes=x_res,
                    yRes=abs(y_res),
                    resampleAlg="nearest",
                    creationOptions=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                ),
            )

    elif q_peril["peril"] == "pl_city":
        resolution = str(city.dtm_resolution)
        # if q_peril["resolution"] not in q_peril["allowed_resolutions"]:
        #     raise ValueError(
        #         f"Resolution {q_peril['resolution']} is not specified in reference file!"
        #     )

        print(f"       - extracting DTM, {resolution}m", end="")
        if tu_data.tu_parameters["tu_values_default"]["Select_DTM_by"] == "city":
            if str(city.name) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(city.name)][resolution]
                # elif str(watershed)[0] in tu_sources.d_DTM_path:
                #     dtm_path = tu_sources.d_DTM_path[str(city.name[0])][resolution]
                print(f"\nCity DTM path: {dtm_path}")
            else:
                raise FileNotFoundError(f"\nDTM path not found, check {city.name} ")
        elif tu_data.tu_parameters["tu_values_default"]["Select_DTM_by"] == "domain":
            if str(area_key) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key)][resolution]
            elif str(area_key)[0] in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(area_key[0])][resolution]
            else:
                raise FileNotFoundError(
                    f"\nDTM path not found, check {area_key} or {str(area_key)[0]}"
                )
        elif tu_data.tu_parameters["tu_values_default"]["Select_DTM_by"] == "watershed":
            if str(watershed) in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(watershed)][resolution]
            elif str(watershed)[0] in tu_sources.d_DTM_path:
                dtm_path = tu_sources.d_DTM_path[str(watershed[0])][resolution]
            else:
                raise FileNotFoundError(
                    f"\nDTM path not found, check {watershed} or {str(watershed)[0]}"
                )
        src = gdal.OpenEx(dtm_path)
        data_type = src.GetRasterBand(1).DataType
        del src
        if data_type in (
            gdal.GDT_Int32,
            gdal.GDT_Int16,
            gdal.GDT_Byte,
            gdal.GDT_UInt16,
            gdal.GDT_UInt32,
        ):
            x_res, y_res = get_resolution(dtm_path)
            bounds = pyogrio.read_info(code_buff)["total_bounds"]
            bounds = snap_extent(
                [bounds[0], bounds[2], bounds[1], bounds[3]],
                get_extent(dtm_path),
                x_res,
            )
            raster = f"/vsimem/{domain}.tif"
            gdal.Warp(
                raster,
                dtm_path,
                options=gdal.WarpOptions(
                    outputBounds=[bounds[0], bounds[2], bounds[1], bounds[3]],
                    xRes=x_res,
                    yRes=abs(y_res),
                    resampleAlg="nearest",
                ),
            )

            array, no_data = get_array_from_raster(raster)
            array = array.astype(float)
            array = np.where(array != no_data, array / 100.0, no_data)
            save_array_with_type(
                array,
                dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                raster,
                gdal.GDT_Float32,
                no_data,
                creation_options=[
                    "COMPRESS=DEFLATE",
                    "NUM_THREADS=ALL_CPUS",
                    "ZLEVEL=12",
                ],
            )
            gdal.Unlink(f"/vsimem/{domain}.tif")

        else:
            x_res, y_res = get_resolution(dtm_path)
            bounds = pyogrio.read_info(code_buff)["total_bounds"]
            bounds = snap_extent(
                [bounds[0], bounds[2], bounds[1], bounds[3]],
                get_extent(dtm_path),
                get_resolution(dtm_path)[0],
            )

            gdal.Warp(
                dtm_grid.replace(".tif", f"_{str(resolution).zfill(2)}.tif"),
                dtm_path,
                options=gdal.WarpOptions(
                    outputBounds=[bounds[0], bounds[2], bounds[1], bounds[3]],
                    xRes=x_res,
                    yRes=abs(y_res),
                    resampleAlg="nearest",
                    creationOptions=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                ),
            )

        print("... done")


## msa 2025-11-26 add 'city' as key, with abreviation of city's name, witch is consider and 'DTM_city_resolution' with info with resolution of DTM have to be used for the city
def get_dtm_switch(
    domain: str,
    domains_dict: dict,
) -> tuple:

    # domains_dict can be either:
    # 1) mapping area-prefix -> excel path (from yaml), or
    # 2) already-loaded domain table {domain_id: {...}} for a single area.
    if domain in domains_dict:
        area_domains = domains_dict
    else:
        area_domains = read_domains_file(domains_dict, domain.split("_")[0])

    if domain in area_domains:
        resolutions = [
            int(area_domains[domain]["DTM_Resolution1"]),
            int(area_domains[domain]["DTM_Resolution2"]),
        ]

        if all(
            item in area_domains[domain]
            for item in ("City", "DTM_city_Resolution", "Mesh_city_Resolution")
        ):
            city = City_parameters(
                area_domains[domain]["City"],
                int(area_domains[domain]["DTM_city_Resolution"]),
                int(area_domains[domain]["Mesh_city_Resolution"]),
            )
        else:
            city = None

        ##msa debug
        print(resolutions)
        print(city)

        res_combo = []
        for res in resolutions:
            if float(res) != 0.0:
                if ".5" in str(res):
                    res = str(float(res)).zfill(3)
                    res_combo.append(res)
                elif ".0" in str(res):
                    res = str(int(float(res))).zfill(2)
                    res_combo.append(res)
                else:
                    res_combo.append(str(res).zfill(2))
        res = f"D{'_'.join(res_combo)}"
        if "inflow" in area_domains[domain]:
            inflow_type = str(area_domains[domain]["inflow"])
            if inflow_type.upper() in ["R", "C", "L"]:
                inflow_type = inflow_type.upper()
            else:
                inflow_type = None
        else:
            inflow_type = None
        if "watershed" in area_domains[domain]:
            watershed = area_domains[domain]["watershed"]
        else:
            watershed = None

        ##mse debug
        # print(res, resolutions[0], inflow_type, watershed, resolutions, city)
        return res, resolutions[0], inflow_type, watershed, resolutions, city
    else:
        raise FileNotFoundError(
            f"Domain {domain} is not in list of domains with resolution"
        )


def write_bat_file(
    workspace, domain, multi_scenario, multi_scenario_path, manual_scenario
):
    bat_file = os.path.join(workspace, domain, "runs", f"run_simulations_{domain}.bat")
    remove_TU_files(bat_file)

    with open(bat_file, "w", newline="\n") as bat_file:
        bat_file.write(
            f"set TUFLOWEXE=c:\\TUFLOW\\Releases\\{tu_sources.tuflow_version}\\TUFLOW_iSP_w64.exe\n"
        )
        bat_file.write(
            r'set RUN=start "TUFLOW" /wait "%TUFLOWEXE%" -b -acf -x' + "\n\n"
        )
        if multi_scenario is None or multi_scenario_path is None:
            raise FileNotFoundError(
                "Multiscenario specified, but file was not provided"
            )
        if multi_scenario == "manual":
            df = pd.DataFrame.from_dict(manual_scenario, orient="index")
            df.reset_index(inplace=True)
            df.rename(columns={"index": "RP"}, inplace=True)
        else:
            scenario = os.path.join(multi_scenario_path, multi_scenario + ".csv")
            df = pd.read_csv(scenario)
        for _, item in df.iterrows():
            one_line = (
                f"%RUN% -s1 {item['start']} "
                f"-s2 {item['end']} "
                f"-s3 {item['model']} "
                f"-s4 {item['dtm']} "
                f"-s5 {item['output']} "
                f"-s6 {item['outputsize']} "
                f"-s7 {item['restart']} "
                f"-e1 {item['Inf_Soil']} "
                f"-e2 {item['Duration']} "
                f"-e3 {item['RP']} "
                f"{domain}.tcf\n"
            )
            bat_file.write(one_line)
        bat_file.write("rem pause")


def write_bat_test_files(
    domain,
    workspace,
    watershed,
    tuflow_version,
    mesh_resolution=tu_data.tu_parameters["pluvial_resolution_mesh"],
):
    bat_file_exec_name = os.path.join(workspace, domain, "runs", "run_sim_exec.bat")
    bat_file_test_name = os.path.join(workspace, domain, "runs", "run_sim_test.bat")

    environment_loader = Environment(
        loader=FileSystemLoader(
            os.path.join(TEMPLATE_ROOT, f"{str(mesh_resolution)}m", "runs")
        )
    )
    template = environment_loader.get_template("run_sim_exec.bat")
    content = template.render(
        domain=domain,
        area=watershed,
        tu_version=tuflow_version,
        # soil_file=os.path.basename(config_variables["soil_file"]),
    )
    with open(bat_file_exec_name, "w") as domain_tfc:
        domain_tfc.write(content)

    template = environment_loader.get_template("run_sim_test.bat")
    content = template.render(
        domain=domain,
        area=watershed,
        tu_version=tuflow_version,
        # soil_file=os.path.basename(config_variables["soil_file"]),
    )
    with open(bat_file_test_name, "w") as domain_tfc2:
        domain_tfc2.write(content)


def get_zpts_order_list(user_dirs_levee_first, area):
    try:
        if os.path.isfile(user_dirs_levee_first[area]):
            with open(user_dirs_levee_first[area], encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                list_domain = list(reader)
            return [val for sublist in list_domain for val in sublist]
        else:
            list_domain = []
            return list_domain
    except:
        list_domain = []
        return list_domain


def create_scenario_input_multi(
    domain,
    peril_path,
    workspace,
    tuflow_yaml_path,
    multi_scenario,
    project,
    project_start_yaml,
    tuflow_version,
    precision="iSP",
    auto_restart=False,
    max_time=200,
    time_shift=10,
    create_without_domain=False,
    inflow_type=None,
    other_values: dict = {"resolution": "30"},
):
    final_path = [
        workspace,
        domain,
    ]

    root_folder = os.path.join(*final_path)
    area = domain.split("_")[0]

    workspace_yaml = os.path.join(tuflow_yaml_path, project, "yaml", area)
    os.makedirs(workspace_yaml, exist_ok=True)
    domain_scenario_target = os.path.join(
        workspace_yaml, f"{domain}_{other_values['resolution']}.yaml"
    )

    if (
        not os.path.exists(os.path.join(workspace, domain))
        and not create_without_domain
    ):
        print(
            "   ...Skip. Domain does not exist on the TUFLOW server. Nothing to compute."
        )
        return False

    else:
        domain_dict = {
            "domain": domain,
            "data_source": root_folder,
            "project": project,
            "peril_path": peril_path,
            "project_start_yaml": project_start_yaml,
            "start_scenario": multi_scenario,
            "auto_restart": auto_restart,
            "max_time": max_time,
            "time_shift": time_shift,
            "tuflow_version": tuflow_version,
            "tuflow_precision": precision,
        }
        if inflow_type is not None:
            domain_dict["inflow_type"] = inflow_type
            print(
                f"... Inflow type is set to {inflow_type}, as specified in reference excel",
                end="",
            )
        if other_values is not None:
            domain_dict.update(other_values)
        if domain_dict["start_scenario"] == "manual":
            domain_dict["scenario"] = tu_data.tu_parameters["manual_scenario"]
        else:
            df = pd.read_csv(
                os.path.join(tu_sources.multi_scenario_path, f"{multi_scenario}.csv")
            )
            df.set_index("RP", inplace=True)
            df_dict = df.to_dict(orient="index")
            domain_dict["scenario"] = df_dict
        with open(domain_scenario_target, "w") as out_file:
            yaml.dump(domain_dict, out_file, default_flow_style=False, sort_keys=False)

        print("... created")
        return True


def is_quadtree(
    scenario="multi", multi_scenario=None, multi_scenario_path=None
) -> bool:
    if multi_scenario != "manual":
        if multi_scenario is None or multi_scenario_path is None:
            raise FileNotFoundError(
                "Multiscenario specified, but file was not provided"
            )
        scenario = os.path.join(multi_scenario_path, multi_scenario + ".csv")
        scenario_set = set(pd.read_csv(scenario)["model"].values.tolist())
        if "Q" in scenario_set or "T" in scenario_set:
            return True
        else:
            return False
    else:
        return False


def filter_list_dirs(
    user_dirs, default_area, user_dirs_condition_values
):  ## select only names appropriate with query in  'f_condition'
    tmp_list_dir = []
    for id in user_dirs:
        if is_tu_domain(id, default_area) is True:
            if (
                filter_dirs_condition(int(id.split("_")[1]), user_dirs_condition_values)
                is True
            ):
                tmp_list_dir.append(id)
    return tmp_list_dir


def filter_dirs_condition(id_domain, user_dirs_condition_values):
    lower = user_dirs_condition_values[0]
    upper = user_dirs_condition_values[1]
    if (id_domain > lower and id_domain < upper) is True:
        return True


def is_tu_domain(name, default_area):
    if fnmatch.fnmatch(name, default_area + "_*"):
        return True


def get_user_dirs(workspace, area):
    print("Domain directories are NOT defined by user. \r\n")
    users_dirs = [i for i in os.listdir(workspace) if is_tu_domain(i, area) is True]
    return users_dirs


def ignore_underscore_dirs(dir, names):
    # Only ignore directories (not files) that start with '_'
    return {
        name
        for name in names
        if os.path.isdir(os.path.join(dir, name)) and name.startswith("_")
    }


def write_common_input_model(workspace, domain, folder_common_inputs_model):
    out_folder = os.path.join(workspace, domain, "common")
    os.makedirs(out_folder, exist_ok=True)

    # print(f'folder_common_inputs_model: {folder_common_inputs_model}')
    # shutil.copytree(folder_common_inputs_model, out_folder, dirs_exist_ok=True)
    shutil.copytree(
        folder_common_inputs_model,
        out_folder,
        dirs_exist_ok=True,
        ignore=ignore_underscore_dirs,
    )
    # copytree(folder_common_inputs_model, out_folder)


def output_size(
    scenario="multi", multi_scenario=None, multi_scenario_path=None
) -> bool:
    if multi_scenario != "manual":
        if multi_scenario is None or multi_scenario_path is None:
            raise FileNotFoundError(
                "Multiscenario specified, but file was not provided"
            )
        scenario = os.path.join(multi_scenario_path, multi_scenario + ".csv")
        scenario_set = list(set(pd.read_csv(scenario)["outputsize"].values.tolist()))
        res = float(scenario_set[0][2:])
        return res
    else:
        df = pd.DataFrame.from_dict(
            tu_data.tu_parameters["manual_scenario"], orient="index"
        )
        df.reset_index(inplace=True)
        df.rename(columns={"index": "RP"}, inplace=True)
        scenario_set = list(set(df["outputsize"].values.tolist()))
        res = float(scenario_set[0][2:])
        return res


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def check_additional_layers(domain, workspace):
    folder = os.path.join(workspace, domain, "model", "grid")

    wb = os.path.join(folder, "dtm_wb_" + domain + ".tif")
    buildings = os.path.join(folder, "dtm_buildings_" + domain + ".tif")
    cwf = os.path.join(folder, "cwf_buildings_" + domain + ".tif")
    is_wb, is_buildings, is_cwf = False, False, False
    if os.path.isfile(wb):
        is_wb = True
    if os.path.isfile(buildings):
        is_buildings = True
    if os.path.isfile(cwf):
        is_cwf = True

    return is_wb, is_buildings, is_cwf


def read_domains_file(domains_file, area):
    area = str(area)
    excel_path = domains_file[area[0]]

    print(f"Trying to load: {excel_path}")
    print(area)
    with pd.ExcelFile(excel_path) as xls:
        df = pd.read_excel(xls, sheet_name=area)

    df["Domain"] = df["Domain"].astype(str).str.strip()
    df.fillna(0, inplace=True)
    df.set_index("Domain", inplace=True)

    return df.to_dict("index")


def change_2d_mat(
    workspace: str,
    domain: str,
    value_mat: int = 80,
    size: int = 30,
):
    area = domain.split("_")
    area = area[0]
    channels = os.path.join(
        workspace, domain, "model", "gis", "dtm", f"2d_zsh_channel_{domain}.gpkg"
    )
    culverts = os.path.join(
        workspace, domain, "model", "gis", "dtm", f"2d_zsh_culvert_{domain}.gpkg"
    )
    mat_file = os.path.join(workspace, domain, "model", "grid", f"2d_mat_{domain}.tif")
    extent = get_extent(mat_file)
    mat_resolution = get_resolution(mat_file)[0]

    df = None
    if os.path.isfile(channels) and os.path.isfile(culverts):
        df = pd.concat(
            [
                gpd.read_file(channels, layer=f"2d_zsh_channel_{domain}_L"),
                gpd.read_file(culverts, layer=f"2d_zsh_culvert_{domain}_L"),
            ]
        )
    elif os.path.isfile(channels):
        df = gpd.read_file(channels, layer=f"2d_zsh_channel_{domain}_L")
    elif os.path.isfile(culverts):
        df = gpd.read_file(culverts, layer=f"2d_zsh_culvert_{domain}_L")
    else:
        print("\tNo channels or culverts provided, skipping 2d_mat change.")
        return
    if df.empty:
        print("\tNo channels or culverts found, skipping 2d_mat change.")
        return
    # buffer the channel/culvert to a specified width
    df.geometry = df.geometry.buffer(size / 2)
    df["value"] = 1
    df = df[["value", "geometry"]]
    tmp_folder = os.path.join(workspace, domain, "model", "gis")
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_file = os.path.join(tmp_folder, f"2d_channel_culvert_{domain}.gpkg")
    tmp_tif = f"/vsimem/channel_{domain}.tif"
    df.to_file(tmp_file)

    if os.path.isfile(mat_file):
        print("\tChannels and/or culverts are provided, changing 2d_mat.")

        channel_tif = rasterize(
            tmp_file,
            "value",
            tmp_tif,
            mat_resolution,
            1,
            0,
            extent=extent,
            all_touched=True,
        )
        array_channel, no_data_channel = get_array_from_raster(channel_tif)
        array_mat, no_data_mat = get_array_from_raster(mat_file)

        array = np.where(
            array_channel != no_data_channel,
            np.where(array_mat != no_data_mat, value_mat, no_data_mat),
            array_mat,
        )

        src = gdal.Open(mat_file, gdal.GA_Update)
        src.GetRasterBand(1).WriteArray(array)
        del src

        gdal.Unlink(tmp_tif)
        gdal.Unlink(tmp_file)


def _split_line_point_sources(paths):
    """
    Split input files into culvert line/point files.
    Supports both .shp and .gpkg and suffixes *_L / *_P.
    """
    culverts_l = []
    culverts_p = []

    for value in paths:
        if value is None:
            continue
        path = str(value).strip()
        if not path:
            continue

        path_upper = path.upper()
        if "_L." in path_upper:
            culverts_l.append(path)
        elif "_P." in path_upper:
            culverts_p.append(path)

    if len(culverts_l) != len(culverts_p):
        raise ValueError(
            "Culvert source configuration is invalid: the number of *_L and *_P files does not match. "
            f"Found {len(culverts_l)} line files and {len(culverts_p)} point files."
        )

    return culverts_l, culverts_p


def get_culverts(
    culvert_source,
    watershed=None,
    Q_peril=None,
    city_name=None,
    source_label="zsh_culvert",
):
    """
    Resolve ZSH line source files (culverts/channels/levees).

    Supported input:
    1) Direct list/tuple of line+point files, e.g.:
       ["..._L.gpkg", "..._P.gpkg"]
    """
    if culvert_source in [None, "None", "", []]:
        return [], []

    if isinstance(culvert_source, (list, tuple)):
        return _split_line_point_sources(culvert_source)

    if isinstance(culvert_source, str):
        raise ValueError(
            f"YAML indirection is not supported anymore for {source_label}. "
        )

    raise TypeError(f"Unsupported culvert source type: {type(culvert_source)}.")


def check_culverts_inside_domain(culverts_l, culverts_p, workspace, domain):
    """
    Check if the culverts are inside the domain.
    """
    folder = os.path.join(workspace, domain, "model", "gis")
    # clip_2d = os.path.join(folder, f"2d_clip_{domain}_R.gpkg")
    clip_2d = os.path.join(folder, f"2d_code_{domain}_R.gpkg")

    if not os.path.isfile(clip_2d):
        raise FileNotFoundError(
            f"Clip file {clip_2d} not found. Please create it before extracting culverts."
        )
    clip = gpd.read_file(clip_2d)
    dfs_culverts_l = []
    dfs_culverts_p = []

    print(f"culverts: ")
    print(culverts_l)

    for index, file in enumerate(culverts_l):
        df_l = gpd.read_file(file, bbox=clip)
        if not df_l.empty:
            df_p = gpd.read_file(culverts_p[index])
            dfs_culverts_l.append(df_l)
            dfs_culverts_p.append(df_p)
    if not dfs_culverts_l:
        print(f"\t\t-No culverts found inside the domain {domain} in any file.")
        return None, None
        # raise ValueError(f"No culverts found inside the domain {domain} in any file.")

    df_l = pd.concat(dfs_culverts_l, ignore_index=True)
    df_p = pd.concat(dfs_culverts_p, ignore_index=True)
    return df_l, df_p


def get_zsh_order(area):
    list_zsh_reverse_domain = get_zpts_order_list(
        tu_sources.user_dirs_levee_first, area
    )
    return list_zsh_reverse_domain


# domains_with_dtm_resolution = read_domains_file(tu_sources.domains_list, area)

if __name__ == "__main__":
    pass
