import os
import csv
import re

os.environ["USE_PYGEOS"] = "0"

import shutil
import fnmatch
from datetime import date

from osgeo import gdal
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
from ifgis.vector import select_by_location, buffer
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tu_input_sources import config_variables

_template_model = os.path.join(config_variables["template_path"], "template", "model")
_template_runs = os.path.join(config_variables["template_path"], "template", "runs")


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

    return (
        code_vec,
        rf_vec,
        bc_in_vec_R,
        bc_in_vec_L,
        dtm_grid_flt,
        mat_grid_flt,
        soil_grid_flt,
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


def extract_dem_add(workspace, domain, code_buff, d_DTM_add_path, key, value):
    """
    Method, which will add some value to dtm
    """
    output = os.path.join(workspace, domain, "model", "grid", f"{key}_{domain}.tif")
    tmp_file = rf"/vsimem/dtm_{key}_{domain}_tmp.tif"
    extract_by_mask_rasterized(
        raster=d_DTM_add_path[key],
        mask=code_buff,
        output=tmp_file,
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
            creation_options=["COMPRESS=DEFLATE", "NUM_THREADS=ALL_CPUS", "ZLEVEL=12"],
        )

    gdal.Unlink(tmp_file)


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
        gdf_l_part = gdf_l[gdf_l["Shape_Widt"] == 0].copy()
        gdf_l = gdf_l[gdf_l["Shape_Widt"] != 0].copy()

        gdf_l_part["Shape_Widt"] = round(float(output_resolution) * (2**0.5))
        gdf_l = pd.concat([gdf_l, gdf_l_part])
        gdf_l["pCFW"] = float(tu_zsh_lines_switch["zsh_culvert_CFW"])
        gdf_l["pFLC"] = float(tu_zsh_lines_switch["zsh_culvert_FLC"])
        gdf_l["Z"] = float(-99999)
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
    d_path_in_zsh_levee_L,
    d_path_in_zsh_levee_P,
    output_resolution,
):
    import warnings

    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    code = os.path.join(workspace, domain, "model", "gis", f"2d_code_{domain}_R.gpkg")
    area = domain.split("_")[0]
    in_zsh_L = d_path_in_zsh_levee_L
    in_zsh_P = d_path_in_zsh_levee_P

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

    if Q_peril == "pl":
        bc_dbase_common_path = os.path.join(path_bc_dbase_inputs, "IDF")
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
    if os.path.exists(shape):
        df = gpd.read_file(shape)
        if Q_peril == "fl":
            return df["Name"].to_list()
        if Q_peril == "pl" or Q_peril == "rf":
            df["ID_RF"] = df["ID_RF"].astype(str)
            return df["ID_RF"].to_list()


def remove_TU_files(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)


def filter_csvfile(input_file, output_file, filter_list):
    df = pd.read_csv(input_file)
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


def write_tgc_file(domain, workspace, dtms_combinations, restart=False):
    tgc_file_name = os.path.join(workspace, domain, "model", domain + ".tgc")
    coarsest_resolution = sorted(dtms_combinations)[-1]
    dtm_grid = os.path.join(
        workspace,
        domain,
        "model",
        "grid",
        f"dtm_{domain}_{str(coarsest_resolution).zfill(2)}.tif",
    )
    ncols, nrows, xllcorner, yllcorner, cellsize = get_DTM_metadata(dtm_grid)
    shifted = float(xllcorner) - 200
    environment_loader = Environment(loader=FileSystemLoader(_template_model))
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
    )
    with open(tgc_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tcf_local_inputs(
    domain,
    workspace,
    is_levee=False,
    is_channel=False,
    is_culvert=False,
    zpt_order="auto",
):
    tcf_file_name = os.path.join(
        workspace, domain, "model", f"LocalInputs_{domain}.tcf"
    )
    is_channel_check, is_levee_check, is_culvert_check = check_culverts_levees_channels(
        domain, workspace
    )
    is_wb, is_buildings = check_additional_layers(domain, workspace)
    if is_levee_check or is_levee:
        is_levee = True
    if is_channel_check or is_channel:
        is_channel = True
    if is_culvert_check or is_culvert:
        is_culvert = True
    if any(
        zpt_order.lower() in x for x in ["auto", "levee_last", "levees_last", "default"]
    ):
        order = False
        print("\n\t\tOrder for levees first is False", end="")
    else:
        order = True
    environment_loader = Environment(loader=FileSystemLoader(_template_model))
    template = environment_loader.get_template("LocalInputs_domain.tcf")
    content = template.render(
        domain=domain,
        levee=is_levee,
        channel=is_channel,
        culvert=is_culvert,
        order=order,
        wb=is_wb,
        buildings=is_buildings,
    )
    with open(tcf_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_ecf_file(domain, workspace):
    ecf_file_name = os.path.join(workspace, domain, "runs", f"{domain}.ecf")
    environment_loader = Environment(loader=FileSystemLoader(_template_runs))
    template = environment_loader.get_template("domain.ecf")
    content = template.render(
        domain=domain,
    )
    with open(ecf_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tbc_file(domain, workspace):
    tbc_file_name = os.path.join(workspace, domain, "model", f"{domain}.tbc")
    environment_loader = Environment(loader=FileSystemLoader(_template_model))
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


def write_restart_file(domain, workspace):
    restart_file_name = os.path.join(workspace, domain, "runs", f"Restart_{domain}.tcf")
    environment_loader = Environment(loader=FileSystemLoader(_template_runs))
    template = environment_loader.get_template("Restart_domain.tcf")
    content = template.render(
        domain=domain,
    )
    with open(restart_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tef_file(domain, workspace, watershed):
    tef_file_name = os.path.join(workspace, domain, "runs", f"{domain}.tef")
    environment_loader = Environment(loader=FileSystemLoader(_template_runs))
    template = environment_loader.get_template("domain.tef")
    content = template.render(
        domain=domain,
        area=watershed,
        # soil_file=os.path.basename(config_variables["soil_file"]),
    )
    with open(tef_file_name, "w") as domain_tfc:
        domain_tfc.write(content)


def write_tcf_file(domain, workspace, quadtree=False):
    tcf_file_name = os.path.join(workspace, domain, "runs", f"{domain}.tcf")
    environment_loader = Environment(loader=FileSystemLoader(_template_runs))
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
        environment_loader = Environment(loader=FileSystemLoader(_template_model))
        template = environment_loader.get_template("Quadtree_domain.tcf")
        content = template.render(
            domain=domain, tuflow_version=config_variables["tuflow_version"]
        )
        with open(quadtree_file_name, "w") as domain_tfc:
            domain_tfc.write(content)

        qcf_file_name = os.path.join(
            workspace, domain, "model", f"quadtree_{domain}.qcf"
        )
        environment_loader = Environment(loader=FileSystemLoader(_template_model))
        template = environment_loader.get_template("domain.qcf")
        content = template.render(domain=domain)
        with open(qcf_file_name, "w") as domain_tfc:
            domain_tfc.write(content)


def get_dtm_switch(
    domain: str,
    domains_dict: dict,
) -> tuple:
    domains_dict = read_domains_file(domains_dict, domain.split("_")[0])
    if domain in domains_dict:
        resolutions = [
            domains_dict[domain]["DTM_Resolution1"],
            domains_dict[domain]["DTM_Resolution2"],
        ]
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
        if "inflow" in domains_dict[domain]:
            inflow_type = str(domains_dict[domain]["inflow"])
            if inflow_type.upper() in ["R", "C", "L"]:
                inflow_type = inflow_type.upper()
            else:
                inflow_type = None
        else:
            inflow_type = None
        if "watershed" in domains_dict[domain]:
            watershed = domains_dict[domain]["watershed"]
        else:
            watershed = None
        return res, resolutions[0], inflow_type, watershed
    else:
        raise FileNotFoundError(
            f"Domain {domain} is not in list of domains with resolution"
        )


def write_bat_file(workspace, domain, tu_values_default, tu_bat_variables):
    bat_file = os.path.join(workspace, domain, "runs", f"run_simulations_{domain}.bat")
    remove_TU_files(bat_file)

    with open(bat_file, "w", newline="\n") as bat_file:
        bat_file.write(
            f"set TUFLOWEXE=c:\\TUFLOW\\Releases\\{tu_values_default['TUFLOW_release']}\\TUFLOW_iSP_w64.exe\n"
        )
        bat_file.write(
            r'set RUN=start "TUFLOW" /wait "%TUFLOWEXE%" -b -acf -x' + "\n\n"
        )
        for event in tu_bat_variables["events"]["infiltration"]:
            for inflow in tu_bat_variables["events"]["inflow"]:
                for rp in tu_bat_variables["events"]["rp"]:
                    one_line = (
                        f"%RUN% -s1 {tu_bat_variables['scenarios']['start_time']} "
                        f"-s2 {tu_bat_variables['scenarios']['end_time']} "
                        f"-s3 {tu_bat_variables['scenarios']['method']} "
                        f"-s4 {tu_bat_variables['scenarios']['dtm_res']} "
                        f"-s5 {tu_bat_variables['scenarios']['output_time']} "
                        f"-s6 {tu_bat_variables['scenarios']['output_size']} "
                        f"-s7 {tu_bat_variables['scenarios']['restart']} "
                        f"-e1 {event} "
                        f"-e2 {inflow} "
                        f"-e3 {rp} "
                        f"{domain}.tcf\n"
                    )

                    bat_file.write(one_line)
        bat_file.write("rem pause")


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


def create_scenario_input_manual(
    domain,
    workspace_ts,
    manual_scenario,
    create_without_domain=False,
) -> bool:
    workspace_ts_area = os.path.join(
        workspace_ts, domain.split("_")[0]
    )  # , f'{output_resolution}m')
    workspace_ts_batlist = os.path.join(
        workspace_ts, "03_bat"
    )  # , f'{output_resolution}m')
    os.makedirs(workspace_ts_batlist, exist_ok=True)
    domain_scenario_ts_target = os.path.join(workspace_ts_batlist, domain + ".csv")

    if (
        not os.path.exists(os.path.join(workspace_ts_area, domain))
        and not create_without_domain
    ):  # False:
        print(
            " ...Skip. Domain does not exist on the TUFLOW server. Nothing to compute."
        )
        return f"{domain} - Domain does not exist on the TUFLOW server."

    else:
        with open(domain_scenario_ts_target, "w", newline="\n") as domain_csv:
            headers = [
                "Domain",
                "RP",
                "Duration",
                "Inf_Soil",
                "start",
                "end",
                "model",
                "dtm",
                "output",
                "outputsize",
                "restart",
            ]
            list_rps = manual_scenario["events"]["rp"]
            list_durations = manual_scenario["events"]["inflow"]
            list_inf = manual_scenario["events"]["infiltration"]
            start_time = manual_scenario["scenarios"]["start_time"]
            end_time = manual_scenario["scenarios"]["end_time"]
            method = manual_scenario["scenarios"]["method"]
            restart = manual_scenario["scenarios"]["restart"]
            output_time = manual_scenario["scenarios"]["output_time"]
            output_size = manual_scenario["scenarios"]["output_size"]
            dtm_res = manual_scenario["scenarios"]["dtm_res"]

            scenario_writer = csv.writer(domain_csv, delimiter=",")
            scenario_writer.writerow(headers)
            for rp in list_rps:
                for duration in list_durations:
                    for infiltration in list_inf:
                        scenario_writer.writerow(
                            [
                                domain,
                                rp,
                                duration,
                                infiltration,
                                start_time,
                                end_time,
                                method,
                                dtm_res,
                                output_time,
                                output_size,
                                restart,
                            ]
                        )
        print("... created")
        return True


def create_scenario_input_multi(
    domain,
    workspace_ts,
    multi_scenario,
    multi_scenario_path,
    create_without_domain=False,
    inflow_type=None,
):
    workspace_ts_area = os.path.join(
        workspace_ts, domain.split("_")[0]
    )  # , f'{output_resolution}m')
    workspace_ts_batlist = os.path.join(
        workspace_ts, "03_bat"
    )  # , f'{output_resolution}m')
    if not os.path.exists(workspace_ts_batlist):
        os.makedirs(workspace_ts_batlist)
    domain_scenario_ts_target = os.path.join(workspace_ts_batlist, domain + ".csv")
    domain_scenario_ts_source = os.path.join(
        multi_scenario_path, multi_scenario + ".csv"
    )

    if (
        not os.path.exists(os.path.join(workspace_ts_area, domain))
        and not create_without_domain
    ):
        print(
            "   ...Skip. Domain does not exist on the TUFLOW server. Nothing to compute."
        )
        return False

    else:
        df = pd.read_csv(domain_scenario_ts_source)
        df["Domain"] = domain
        if inflow_type is not None:
            df["Duration"] = inflow_type
            print(
                f"... Inflow type is set to {inflow_type}, as specified in reference excel",
                end="",
            )
        df.to_csv(domain_scenario_ts_target, index=False)

        print("... created")
        return True


def is_quadtree(
    scenario, tu_bat_variables, multi_scenario=None, multi_scenario_path=None
) -> bool:
    if scenario == "multi":
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
        method = tu_bat_variables["scenarios"]["method"]
        if "Q" in method or "T" in method:
            return True
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
    if re.fullmatch(re.escape(default_area) + r"_[0-9]{8,12}", name):
        return True


def get_user_dirs(workspace, area):
    print("Domain directories are NOT defined by user. \r\n")
    users_dirs = [i for i in os.listdir(workspace) if is_tu_domain(i, area) is True]
    return users_dirs


def write_common_input_model(workspace, domain, folder_common_inputs_model):
    out_folder = os.path.join(workspace, domain, "common")
    os.makedirs(out_folder, exist_ok=True)
    shutil.copytree(folder_common_inputs_model, out_folder, dirs_exist_ok=True)
    # copytree(folder_common_inputs_model, out_folder)


def output_size(
    scenario, tu_bat_variables, multi_scenario=None, multi_scenario_path=None
) -> bool:
    if scenario == "multi":
        if multi_scenario is None or multi_scenario_path is None:
            raise FileNotFoundError(
                "Multiscenario specified, but file was not provided"
            )
        scenario = os.path.join(multi_scenario_path, multi_scenario + ".csv")
        scenario_set = list(set(pd.read_csv(scenario)["outputsize"].values.tolist()))
        res = float(scenario_set[0][2:])
        return res
    else:
        scenario_set = tu_bat_variables["scenarios"]["output_size"]
        res = float(scenario_set[2:])
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
    folder = os.path.join(workspace, domain, "model", "gis", "dtm")

    wb = os.path.join(folder, "dtm_wb_" + domain + ".gpkg")
    buildings = os.path.join(folder, "dtm_buildings_" + domain + ".gpkg")
    is_wb, is_buildings = False, False
    if os.path.isfile(wb):
        is_wb = True
    if os.path.isfile(buildings):
        is_buildings = True
    return is_wb, is_buildings


_domains_file_cache = {}


def create_scenario_yaml(
    domain,
    workspace,
    tuflow_yaml_path,
    multi_scenario,
    multi_scenario_path,
    project,
    peril_path,
    project_start_yaml,
    tuflow_version,
    precision="iSP",
    auto_restart=False,
    max_time=200,
    time_shift=10,
    resolution="05",
    create_without_domain=False,
    inflow_type=None,
    schema=None,
):
    import yaml as _yaml

    area = domain.split("_")[0]
    workspace_yaml = os.path.join(tuflow_yaml_path, project, "yaml", area)
    os.makedirs(workspace_yaml, exist_ok=True)
    domain_yaml_target = os.path.join(workspace_yaml, f"{domain}_{resolution}.yaml")

    if (
        not os.path.exists(os.path.join(workspace, domain))
        and not create_without_domain
    ):
        print("   ...Skip. Domain does not exist. Nothing to write.")
        return False

    root_folder = os.path.join(workspace, domain)
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
        "resolution": resolution,
        "schema": schema,
    }
    if inflow_type is not None:
        domain_dict["inflow_type"] = inflow_type
        print(
            f"... Inflow type is set to {inflow_type}, as specified in reference excel",
            end="",
        )

    scenario_csv = os.path.join(multi_scenario_path, f"{multi_scenario}.csv")
    df = pd.read_csv(scenario_csv)
    df.set_index("RP", inplace=True)
    domain_dict["scenario"] = df.to_dict(orient="index")

    with open(domain_yaml_target, "w") as out_file:
        _yaml.dump(domain_dict, out_file, default_flow_style=False, sort_keys=False)

    print("... created")
    return True


def read_domains_file(domains_file, area):
    excel_path = domains_file[str(area)[0]]
    cache_key = (excel_path, str(area))
    if cache_key in _domains_file_cache:
        return _domains_file_cache[cache_key]
    df = pd.read_excel(excel_path, sheet_name=str(area))
    df.fillna(0, inplace=True)
    df.set_index("Domain", inplace=True)
    result = df.to_dict("index")
    _domains_file_cache[cache_key] = result
    return result


def change_2d_mat(
    workspace: str,
    domain: str,
    d_path_in_zsh_channel_L: dict,
    d_path_in_zsh_culvert_L: dict,
    value_mat: int = 80,
    size: int = 30,
):
    area = domain.split("_")
    area = area[0]
    channels = d_path_in_zsh_channel_L
    culverts = d_path_in_zsh_culvert_L

    mat_file = os.path.join(workspace, domain, "model", "grid", f"2d_mat_{domain}.tif")
    extent = get_extent(mat_file)
    mat_resolution = get_resolution(mat_file)[0]

    df = None
    if os.path.isfile(channels) and os.path.isfile(culverts):
        df = pd.concat([gpd.read_file(channels), gpd.read_file(culverts)])
    elif os.path.isfile(channels):
        df = gpd.read_file(channels)
    elif os.path.isfile(culverts):
        df = gpd.read_file(culverts)
    else:
        print("\tNo channels or culverts provided, skipping 2d_mat change.")
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


if __name__ == "__main__":
    file = r"b:\01_Projects\997_SmallProjects\35_TUFLOW_NEWSETUP\01_MD\01_HAZARD\06_TUFLOW\domains.xlsx"
    dict = read_domains_file(file, "W01")
    print(dict)

    print(get_dtm_switch("W01_0002", dict))
