import gc
import os

import numpy as np
from osgeo import gdal

from tu_input_functions import (
    buffer,
    change_2d_mat,
    copy_domain_to_ts,
    create_scenario_input_manual,
    create_scenario_input_multi,
    create_scenario_yaml,
    create_tuflow_dir_structure,
    extract_bc_dbase_to_local,
    extract_channel_to_domain,
    extract_culvert_to_domain,
    extract_dem_add,
    extract_levee_to_domain,
    get_dtm_switch,
    is_quadtree,
    output_size,
    tu_input_names,
    write_bat_file,
    write_common_input_model,
    write_ecf_file,
    write_restart_file,
    write_tbc_file,
    write_tcf_file,
    write_tcf_local_inputs,
    write_tef_file,
    write_tgc_file,
    zpts_features_order,
)
from tu_data import (
    Q_peril,
    area,
    area_key,
    code_buffer,
    copy_domain_to_ts_errlog,
    create_scenario_input_errlog,
    domain_directory_source,
    list_zsh_reverse_domain,
    multi_scenario,
    multi_scenario_path,
    path_bc_dbase_inputs,
    path_common_inputs,
    rest_value,
    temp,
    tu_bat_variables_fl,
    tu_bat_variables_rf,
    tu_values_default,
    tu_zsh_lines_switch,
    tu_what_to_do,
    use_scenario,
    user_dirs,
    workspace_ts,
    yaml_auto_restart,
    yaml_max_time,
    yaml_peril_path,
    yaml_precision,
    yaml_project,
    yaml_schema,
    yaml_time_shift,
)
from ifgis.raster import (
    extract_by_mask_rasterized,
    convert_dtype,
    get_array_from_raster,
    save_array_with_type,
)
from tu_input_sources import (
    DTMs_combination,
    d_DTM_add_path,
    d_DTM_add_specification,
    d_DTM_path,
    d_manning_bldns_path,
    d_manning_path,
    d_path_in_zsh_channel_L,
    d_path_in_zsh_channel_P,
    d_path_in_zsh_culvert_L,
    d_path_in_zsh_culvert_P,
    d_path_in_zsh_levee_L,
    d_path_in_zsh_levee_P,
    d_root_path_net,
    d_soil_draiange_path_default_rasters,
    d_soil_path,
    domains_list,
    tuflow_directory,
    tuflow_version_cfg,
    tuflow_yaml_path_cfg,
)

os.chdir(os.path.dirname(__file__))
gdal.UseExceptions()
# --------------------------------------------------
#
# version230611
#
# --------------------------------------------------


class TuflowInputs:
    def __init__(self, domain: str, workspace: str):
        self.process_domain(domain, workspace)

    @staticmethod
    def process_domain(domain, workspace):
        if not os.path.exists(os.path.join(workspace, domain)):
            print(f"       - domain does not exist: {os.path.join(workspace, domain)}")
            return
        else:
            print(f"    {domain}")
            create_tuflow_dir_structure(workspace, domain)

        ## defining order of the zsh lines
        DTM_switch, main_resolution, inflow_type, watershed = get_dtm_switch(
            domain, domains_list
        )
        main_resolution = str(main_resolution)
        if tu_values_default["Set_Zpts_Order"] == "auto":
            tu_values_default["Set_Zpts_Order"] = zpts_features_order(
                domain, list_zsh_reverse_domain
            )

        ## defining tuflow gis features - paths and names
        (
            code_vec,
            rf_vec,
            bc_in_vec_R,
            bc_in_vec_L,
            dtm_grid,
            mat_grid,
            soil_grid,
        ) = tu_input_names(workspace, domain)
        if Q_peril == "fl":
            variables = tu_bat_variables_fl
        else:
            variables = tu_bat_variables_rf
        output_res = output_size(
            use_scenario, variables, multi_scenario, multi_scenario_path
        )
        ### extracting zsh culverts
        if tu_what_to_do["Q_Do_ZSH_Culverts"] == "y":
            print("       - extracting culverts")
            if area in d_path_in_zsh_culvert_L:
                in_zsh_L = d_path_in_zsh_culvert_L[area]
                if isinstance(in_zsh_L, dict):
                    in_zsh_L = in_zsh_L[str(main_resolution)]
            elif area[0] in d_path_in_zsh_culvert_L:
                in_zsh_L = d_path_in_zsh_culvert_L[area[0]]
                if isinstance(in_zsh_L, dict):
                    in_zsh_L = in_zsh_L[str(main_resolution)]
            else:
                raise FileNotFoundError(
                    f"No culvert_L file specified for the area: {area} or: {area[0]} and Q_Do_ZSH_Culverts = y"
                )
            if area in d_path_in_zsh_culvert_P:
                in_zsh_P = d_path_in_zsh_culvert_P[area]
                if isinstance(in_zsh_P, dict):
                    in_zsh_P = in_zsh_P[str(main_resolution)]
            elif area[0] in d_path_in_zsh_culvert_P:
                in_zsh_P = d_path_in_zsh_culvert_P[area[0]]
                if isinstance(in_zsh_P, dict):
                    in_zsh_P = in_zsh_P[str(main_resolution)]
            else:
                raise FileNotFoundError(
                    f"No culvert_P file specified for the area: {area} or: {area[0]} and Q_Do_ZSH_Culverts = y"
                )
            ## the source must be valid

            extract_culvert_to_domain(
                workspace,
                domain,
                tu_zsh_lines_switch,
                in_zsh_L,
                in_zsh_P,
                output_res,
            )

        ### extracting zsh channels
        if tu_what_to_do["Q_Do_ZSH_Channels"] == "y":
            print("       - extracting channels")
            if area in d_path_in_zsh_channel_L:
                in_channel_L = d_path_in_zsh_channel_L[area]
                if isinstance(in_channel_L, dict):
                    in_channel_L = in_channel_L[str(main_resolution)]
            if area in d_path_in_zsh_channel_P:
                in_channel_P = d_path_in_zsh_channel_P[area]
                if isinstance(in_channel_P, dict):
                    in_channel_P = in_channel_P[str(main_resolution)]
            if os.path.exists(in_channel_L) and os.path.exists(
                in_channel_P
            ):  ## the source must be valid
                extract_channel_to_domain(
                    workspace,
                    domain,
                    tu_zsh_lines_switch,
                    in_channel_L,
                    in_channel_P,
                    output_res,
                )
            else:
                print(
                    "       - source for channels is not defined correctly or not exists!!!"
                )

        ### extracting zsh levees
        if tu_what_to_do["Q_Do_ZSH_Levees"] == "y":
            print("       - extracting levees")
            if area in d_path_in_zsh_levee_L:
                in_levee_L = d_path_in_zsh_levee_L[area]
                if isinstance(in_levee_L, dict):
                    in_levee_L = in_levee_L[str(main_resolution)]
            if area in d_path_in_zsh_levee_P:
                in_levee_P = d_path_in_zsh_levee_P[area]
                if isinstance(in_levee_P, dict):
                    in_levee_P = in_levee_P[str(main_resolution)]
            if os.path.exists(in_levee_L) and os.path.exists(
                in_levee_P
            ):  ## the source must be valid
                extract_levee_to_domain(
                    workspace,
                    domain,
                    tu_zsh_lines_switch,
                    in_levee_L,
                    in_levee_P,
                    output_res,
                )
            else:
                print(
                    "       - source for levees is not defined correctly or not exists!!!"
                )
        #
        if not os.path.exists(code_vec):
            raise FileNotFoundError(
                f"       - code vector file does not exist: {code_vec}"
            )
        code_buff = os.path.join(temp, "2d_code_buff_" + domain + "_R.gpkg")
        buffer(
            input=code_vec,
            distance=10 * int(code_buffer),
            field="distance",
            output=code_buff,
            driver="GPKG",
        )

        ### extract DTM
        if tu_what_to_do["Q_Do_DTM"] == "y":
            for resolution in DTMs_combination[DTM_switch]:
                resolution = str(resolution)
                print(f"       - extracting DTM, {resolution}m", end="")
                if str(area_key) in d_DTM_path:
                    dtm_path = d_DTM_path[str(area_key)][resolution]
                elif str(area_key)[0] in d_DTM_path:
                    dtm_path = d_DTM_path[str(area_key[0])][resolution]
                else:
                    raise FileNotFoundError(
                        f"DTM path not found, check {area_key} or {str(area_key)[0]}"
                    )
                src = gdal.OpenEx(dtm_path)
                data_type = src.GetRasterBand(1).DataType
                src = None
                _dtm_output = dtm_grid.replace(
                    ".tif", f"_{str(resolution).zfill(2)}.tif"
                )
                if os.path.exists(_dtm_output):
                    os.remove(_dtm_output)
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
                        _dtm_output,
                        raster,
                        gdal.GDT_Float32,
                        no_data,
                        creation_options=[
                            "COMPRESS=DEFLATE",
                            "NUM_THREADS=ALL_CPUS",
                            "ZLEVEL=12",
                        ],
                    )
                    raster = None  # close before unlinking vsimem to prevent GC crash
                    gc.collect()
                    gdal.Unlink(f"/vsimem/{domain}.tif")

                else:
                    extract_by_mask_rasterized(
                        # raster=d_DTM_path[f"{area_key}_{str(resolution).zfill(2)}"],
                        raster=dtm_path,
                        mask=code_buff,
                        output=_dtm_output,
                        creation_options=[
                            "COMPRESS=DEFLATE",
                            "NUM_THREADS=ALL_CPUS",
                            "ZLEVEL=12",
                        ],
                    )
                print("... done")

        ### extract DTM - others
        if tu_what_to_do["Q_Do_DTM_add"] == "y":
            if any(d_DTM_add_specification):
                for key, value in d_DTM_add_specification.items():
                    print("       - extracting additional rasters: " + key, end="")
                    extract_dem_add(
                        workspace, domain, code_buff, d_DTM_add_path, key, value
                    )
                    print("... done")

        ### extract landuse for manning
        if tu_what_to_do["Q_Do_Manning"] == "y":
            print("       - extracting Manning", end="")
            tmp_mat = f"/vsimem/{os.path.basename(mat_grid)}"

            if Q_peril == "pl":
                if (
                    d_manning_bldns_path.get(area_key[0]) is None
                    or d_manning_bldns_path.get(area_key[0]) == "None"
                ):
                    manning_file = d_manning_path[str(main_resolution)]
                else:
                    manning_file = d_manning_bldns_path
                extract_by_mask_rasterized(
                    raster=manning_file,
                    mask=code_buff,
                    output=tmp_mat,
                )  # , debug = True)
                convert_dtype(
                    tmp_mat,
                    mat_grid,
                    gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
                gc.collect()
                gdal.Unlink(tmp_mat)

            else:
                extract_by_mask_rasterized(
                    raster=d_manning_path[main_resolution],
                    mask=code_buff,
                    output=tmp_mat,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )  # , debug = True)
                convert_dtype(
                    tmp_mat,
                    mat_grid,
                    gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
                gc.collect()
                gdal.Unlink(tmp_mat)

            print("... done")

        ### extract soil for infiltration
        if tu_what_to_do["Q_Do_Soil"] == "y":
            print("       - extracting Soil", end="")
            for soil_name in d_soil_path[area_key]:
                soil_raster = d_soil_path[area_key][soil_name]
                tmp_soil = f"/vsimem/soil_name.tif"
                extract_by_mask_rasterized(
                    raster=soil_raster,
                    mask=code_buff,
                    output=tmp_soil,
                )  # , debug = True)
                convert_dtype(
                    tmp_soil,
                    soil_grid.replace(".tif", f"_{soil_name}.tif"),
                    gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
            print("... done")

        if tu_what_to_do["Q_Do_Drainage"] == "y":
            print("       - extracting Soil drainage   ", end="")
            soil_drainage_raster = d_soil_draiange_path_default_rasters[area_key[0]]
            tmp_soil = f"/vsimem/soil_drainage_name.tif"
            extract_by_mask_rasterized(
                raster=soil_drainage_raster,
                mask=code_buff,
                output=tmp_soil,
                save_empty=True,
            )  # , debug = True)
            convert_dtype(
                tmp_soil,
                soil_grid.replace("2d_soil", f"2d_soil_drain"),
                gdal.GDT_Float32,
                creation_options=[
                    "COMPRESS=DEFLATE",
                    "NUM_THREADS=ALL_CPUS",
                    "ZLEVEL=12",
                ],
            )
            print("... done")
        os.remove(code_buff)
        # # print(time.monotonic() - start)

        ### extract hydrology - bcdbase files from commoninputs into the local domain folder
        if tu_what_to_do["Q_Do_BC_Dbase_local"] == "y":
            print("       - extracting hydrology", end="")
            bc_files = [
                "HT0",
                "HT9999",
                "HTm020",
                "HTm015",
                "HTm010",
                "HTm005",
                "HT0005",
                "HT0010",
                "HT0015",
                "HT0020",
                "bc_dbase_R",
                "bc_dbase_L",
                "bc_dbase_C",
                "bc_dbase_rf",
            ]
            extract_bc_dbase_to_local(
                workspace, domain, path_bc_dbase_inputs, Q_peril, bc_files
            )
            print("... done")

        if tu_what_to_do["Q_Do_TUFLOW_files"] == "y":
            print("       - writing TUFLOW text inputs", end="")

            ### write all tuflow txt files:
            # restart = False
            # if Q_peril == "fl":
            #     restart = True
            is_levee, is_channel, is_culvert = False, False, False
            write_tgc_file(domain, workspace, DTMs_combination[DTM_switch])

            write_tcf_local_inputs(
                domain,
                workspace,
                is_levee,
                is_channel,
                is_culvert,
                tu_values_default["Set_Zpts_Order"],
            )
            write_ecf_file(domain, workspace)
            write_tbc_file(domain, workspace)
            # if restart:
            write_restart_file(domain, workspace)
            write_tef_file(domain, workspace, watershed)

            quadtree = is_quadtree(
                use_scenario, variables, multi_scenario, multi_scenario_path
            )

            write_tcf_file(domain, workspace, quadtree)
            write_common_input_model(workspace, domain, path_common_inputs)

            ### write bat file - to be able to run tuflow manually domain per domain.
            if Q_peril == "pl" or Q_peril == "rf":
                write_bat_file(
                    workspace,
                    domain,
                    tu_values_default,
                    tu_bat_variables_rf,
                )

            if Q_peril == "fl":
                write_bat_file(
                    workspace,
                    domain,
                    tu_values_default,
                    tu_bat_variables_fl,
                )

            print("... done")
            print("       - zsh order: " + tu_values_default["Set_Zpts_Order"])

        if (
            tu_what_to_do["Q_Do_Create_Scenario"] == "y"
            and tu_what_to_do["Q_Do_CopyDomains_to_server"] == "y"
        ):
            print("       - copying domain folder to the server", end="")
            b = copy_domain_to_ts(workspace, domain, workspace_ts, back_up=True)
            if b == False:
                copy_domain_to_ts_errlog.append(f"{domain} - Domain didnt copy")
                create_scenario_input_errlog.append(
                    f"{domain} - Scenario didnt created on the TUFLOW server."
                )

            if not b == False:
                if use_scenario == "manual":
                    manual_scenario = (
                        tu_bat_variables_fl if Q_peril == "fl" else tu_bat_variables_rf
                    )
                    print("       - writing scenario (manual) for TUFLOW", end="")
                    c = create_scenario_input_manual(
                        domain,
                        workspace_ts,
                        manual_scenario,
                        create_without_domain=False,
                    )

                if use_scenario == "multi":
                    print("       - writing scenario (multi) for TUFLOW", end="")
                    c = create_scenario_input_multi(
                        domain,
                        workspace_ts,
                        multi_scenario,
                        multi_scenario_path,
                        create_without_domain=False,
                        inflow_type=inflow_type,
                    )
                    if c == False:
                        create_scenario_input_errlog.append(
                            f"{domain} - Scenario didnt created on the TUFLOW server."
                        )

        if (
            tu_what_to_do["Q_Do_Create_Scenario"] == "y"
            and tu_what_to_do["Q_Do_CopyDomains_to_server"] == "n"
        ):
            if use_scenario == "manual":
                manual_scenario = (
                    tu_bat_variables_fl if Q_peril == "fl" else tu_bat_variables_rf
                )
                print("       - writing scenario (manual) for TUFLOW", end="")
                c = create_scenario_input_manual(
                    domain,
                    workspace_ts,
                    manual_scenario,
                    create_without_domain=False,
                )

            if use_scenario == "multi":
                print("       - writing scenario (multi) for TUFLOW", end="")
                c = create_scenario_input_multi(
                    domain,
                    workspace_ts,
                    multi_scenario,
                    multi_scenario_path,
                    create_without_domain=False,
                    inflow_type=inflow_type,
                )
                if c == False:
                    create_scenario_input_errlog.append(
                        f"{domain} - Scenario didnt created on the TUFLOW server."
                    )

        if (
            tu_what_to_do["Q_Do_Create_Scenario"] == "n"
            and tu_what_to_do["Q_Do_CopyDomains_to_server"] == "y"
        ):
            print("       - copying domain folder to the server", end="")
            b = copy_domain_to_ts(workspace, domain, workspace_ts, back_up=True)
            if b == False:
                copy_domain_to_ts_errlog.append(f"{domain} - Domain didnt copy")

        if tu_what_to_do["Q_Do_Create_YAML"] == "y":
            print("       - writing domain YAML", end="")
            create_scenario_yaml(
                domain=domain,
                workspace=workspace,
                tuflow_yaml_path=tuflow_yaml_path_cfg,
                multi_scenario=multi_scenario,
                multi_scenario_path=multi_scenario_path,
                project=yaml_project,
                peril_path=yaml_peril_path,
                tuflow_version=tuflow_version_cfg,
                precision=yaml_precision,
                auto_restart=yaml_auto_restart,
                max_time=yaml_max_time,
                time_shift=yaml_time_shift,
                resolution=f"{main_resolution}m",
                create_without_domain=False,
                schema=yaml_schema,
            )

        if not "y" in [values for key, values in tu_what_to_do.items()]:
            print("       - nothing to do.")
        if tu_what_to_do["Q_Do_Mannig_Culvert_channel"] == "y":
            if area in d_path_in_zsh_channel_L:
                in_channel_L = d_path_in_zsh_channel_L[area]
                if isinstance(d_path_in_zsh_channel_L, dict):
                    in_channel_L = d_path_in_zsh_channel_L[str(main_resolution)]
            else:
                in_channel_L = ""
            if area in d_path_in_zsh_culvert_L:
                in_zsh_L = d_path_in_zsh_culvert_L[area]
                if isinstance(in_zsh_L, dict):
                    in_zsh_L = in_zsh_L[str(main_resolution)]
            else:
                in_zsh_L = ""
            change_2d_mat(
                workspace,
                domain,
                in_channel_L,
                in_zsh_L,
                tu_values_default["channel_value"],
                tu_values_default["channel_width"],
            )
        ## reset user defined zsh order to the default
        tu_values_default["Set_Zpts_Order"] = rest_value  ## don't change


if __name__ == "__main__":
    workspace_in = os.path.join(
        d_root_path_net[domain_directory_source],
        tuflow_directory[Q_peril],
    )
    is_merit = False
    try:
        area = int(float(area))
        is_merit = True
    except Exception as e:
        area = area

    failed_domains = []
    for domain in user_dirs:
        if is_merit:
            workspace = os.path.join(workspace_in, domain.split("_")[1][:2])
        else:
            workspace = os.path.join(workspace_in, area)

        try:
            TuflowInputs(domain, workspace)
        except BaseException as e:
            import traceback

            print(f"\n    ERROR in domain {domain}:")
            traceback.print_exc()
            failed_domains.append(domain)

    if failed_domains:
        print(f"\nFailed domains ({len(failed_domains)}): {failed_domains}")
