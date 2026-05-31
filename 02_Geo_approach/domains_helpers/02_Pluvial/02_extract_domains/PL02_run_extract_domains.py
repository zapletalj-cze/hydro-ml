import os
import time
import traceback
import pandas as pd
import tu_input_functions_cities as tu_input_functions
import tu_data_cities as tu_data
from ifgis.raster import (
    extract_by_mask_rasterized,
    convert_dtype,
)

tu_input_functions.os.chdir(tu_input_functions.os.path.dirname(__file__))
tu_input_functions.gdal.UseExceptions()
# --------------------------------------------------
#
# version230611
#
# --------------------------------------------------
tu_sources = tu_data.tu_sources
tu_parameters = tu_data.tu_parameters

user_dirs = pd.read_csv(tu_parameters["user_dirs"])["DomainID"].tolist()
areas = set([domain.split("_")[0] for domain in user_dirs])
list_zsh_reverse_domain = []

for area in areas:
    list_zsh_reverse_domain.extend(tu_input_functions.get_zsh_order(area))


class TuflowInputs:
    def __init__(self, domain: str, workspace: str, domains_table: dict = None):
        self.area = domain.split("_")[0]
        self.domains_table = domains_table
        self.process_domain(domain, workspace)

    def process_domain(self, domain, workspace):
        if not tu_input_functions.os.path.exists(
            tu_input_functions.os.path.join(workspace, domain)
        ):
            print("       - domain does not exist.")
        else:
            print(f"    {domain}")
            tu_input_functions.create_tuflow_dir_structure(workspace, domain)

        ## defining order of the zsh lines
        if tu_parameters["tu_values_default"]["Set_Zpts_Order"] == "auto":
            tu_parameters["tu_values_default"]["Set_Zpts_Order"] = (
                tu_input_functions.zpts_features_order(domain, list_zsh_reverse_domain)
            )
        domains_source = (
            self.domains_table
            if self.domains_table is not None
            else tu_sources.domains_list
        )
        DTM_switch, main_resolution, inflow_type, watershed, resolutions, city = (
            tu_input_functions.get_dtm_switch(domain, domains_source)
        )
        print(
            f"       - DTM switch: {DTM_switch}, main resolution: {main_resolution}, inflow type: {inflow_type}, watershed: {watershed}, city: {city.name}"
        )
        if tu_parameters["Q_peril"] == "pl_city":
            main_resolution = (
                city.mesh_resolution
            )  ## change for cities for se;eted resolution of mesh.some data like material suposed to be in the mesh resolution
        main_resolution = str(main_resolution)
        ## defining tuflow gis features - paths and names
        (
            code_vec,
            rf_vec,
            bc_in_vec_R,
            bc_in_vec_L,
            dtm_grid,
            mat_grid,
            soil_grid,
        ) = tu_input_functions.tu_input_names(workspace, domain)

        output_res = tu_input_functions.output_size(
            multi_scenario=tu_parameters["multi_scenario"],
            multi_scenario_path=tu_sources.multi_scenario_path,
        )
        ## msa pridani VP
        # ### extracting pits for VP
        if tu_parameters["tu_what_to_do"]["Q_Do_VP"]:
            print("       - extracting pits for VP")
            # try:
            in_file_pit_P = tu_sources.d_VP_pit_points[city.name][
                str(city.mesh_resolution)
            ]

            ## the source must be valid
            tu_input_functions.extract_pits_to_domain(
                workspace,
                domain,
                # tu_zsh_lines_switch,
                in_file_pit_P,
                output_res,
            )

        ### extracting zsh culverts
        if tu_parameters["tu_what_to_do"]["Q_Do_ZSH_Culverts"]:
            print("       - extracting culverts")
            # try:
            mesh_resolution = (
                str(city.mesh_resolution)
                if tu_parameters["Q_peril"] == "pl_city"
                else tu_parameters["pluvial_resolution"]
            )
            yaml_culverts = tu_sources.d_path_in_zsh_culvert[mesh_resolution]
            # print (yaml_culverts)
            if tu_parameters["Q_peril"] == "pl_city":
                culverts_l, culverts_p = tu_input_functions.get_culverts(
                    yaml_culverts,
                    watershed,
                    tu_parameters["Q_peril"],
                    city.name,
                    source_label="zsh_culvert",
                )
            else:
                culverts_l, culverts_p = tu_input_functions.get_culverts(
                    yaml_culverts,
                    watershed,
                    tu_parameters["Q_peril"],
                    source_label="zsh_culvert",
                )

            in_zsh_L, in_zsh_P = tu_input_functions.check_culverts_inside_domain(
                culverts_l, culverts_p, workspace, domain
            )

            ## the source must be valid
            if in_zsh_L is not None and in_zsh_P is not None:
                tu_input_functions.extract_culvert_to_domain(
                    workspace,
                    domain,
                    tu_parameters["tu_zsh_lines_switch"],
                    in_zsh_L,
                    in_zsh_P,
                    output_res,
                )

        ### extracting zsh channels
        if tu_parameters["tu_what_to_do"]["Q_Do_ZSH_Channels"]:
            print("       - extracting channels")
            try:
                yaml_channels = tu_sources.d_path_in_zsh_channel[
                    tu_parameters["pluvial_resolution"]
                ]
                channels_l, channels_p = tu_input_functions.get_culverts(
                    yaml_channels, watershed, source_label="zsh_channel"
                )
                in_channel_L, in_channel_P = (
                    tu_input_functions.check_culverts_inside_domain(
                        channels_l, channels_p, workspace, domain
                    )
                )
            except KeyError:
                raise KeyError(
                    f"ZSH channel for resolution {tu_parameters['pluvial_resolution']}m not defined in the configuration file."
                )

            if (
                in_channel_L is not None and in_channel_P is not None
            ):  ## the source must be valid
                tu_input_functions.extract_channel_to_domain(
                    workspace,
                    domain,
                    tu_parameters["tu_zsh_lines_switch"],
                    in_channel_L,
                    in_channel_P,
                    output_res,
                )
            else:
                print(
                    "       - source for channels is not defined correctly or not exists!!!"
                )

        ### extracting zsh levees
        if tu_parameters["tu_what_to_do"]["Q_Do_ZSH_Levees"]:
            print("       - extracting levees")
            try:
                # yaml_levees = tu_sources.d_path_in_zsh_levee[
                #     tu_parameters["pluvial_resolution"]
                # ]
                mesh_resolution = (
                    str(city.mesh_resolution)
                    if tu_parameters["Q_peril"] == "pl_city"
                    else tu_parameters["pluvial_resolution"]
                )
                yaml_levees = tu_sources.d_path_in_zsh_levee[mesh_resolution]

                # print (yaml_culverts)
                if tu_parameters["Q_peril"] == "pl_city":
                    levees_l, levees_p = tu_input_functions.get_culverts(
                        yaml_levees,
                        watershed,
                        tu_parameters["Q_peril"],
                        city.name,
                        source_label="zsh_levee",
                    )
                else:
                    levees_l, levees_p = tu_input_functions.get_culverts(
                        yaml_levees,
                        watershed,
                        tu_parameters["Q_peril"],
                        source_label="zsh_levee",
                    )

                levees_l, levees_p = tu_input_functions.check_culverts_inside_domain(
                    levees_l, levees_p, workspace, domain
                )
            except KeyError:
                raise KeyError(
                    f"ZSH levee for resolution {tu_parameters['pluvial_resolution']}m not defined in the configuration file."
                )

            if (
                levees_l is not None and levees_p is not None
            ):  ## the source must be valid
                tu_input_functions.extract_levee_to_domain(
                    workspace,
                    domain,
                    tu_parameters["tu_zsh_lines_switch"],
                    levees_l,
                    levees_p,
                    output_res,
                )
            else:
                print(
                    "       - source for levees is not defined correctly or not exists!!!"
                )
        #
        code_buff = tu_input_functions.os.path.join(
            tu_parameters["temp"], "2d_code_buff_" + domain + "_R.gpkg"
        )
        tu_input_functions.buffer(
            input=code_vec,
            distance=10 * int(tu_parameters["code_buffer"]),
            field="distance",
            output=code_buff,
            driver="GPKG",
        )
        # code_buff = os.path.join(temp, '2d_code_buff_' + domain + '_R.shp')
        # buffer(input=code_vec, distance=10*int(output_resolution), field='distance', output=code_buff, driver='ESRI Shapefile')

        # # import time
        # # start = time.monotonic()
        ### extracting zsh polygon for DTM editing
        if tu_parameters["tu_what_to_do"]["Q_Do_ZSH_dtm_polygon"]:
            print("       - extracting zsh polygon for editing of DTM")

            try:
                if tu_parameters["Q_peril"] == "pl_city":
                    in_dtm_zsh_R = tu_sources.d_DTM_zsh_polygon_path[city.name]
                else:
                    in_dtm_zsh_R = tu_sources.d_DTM_zsh_polygon_path[watershed]

            except KeyError:
                raise KeyError(
                    f"ZSH polygon for DTM for city {city.name} or watershed {watershed} not defined in the configuration file (d_DTM_zsh_polygon_path)."
                )

            ## the source must be valid
            if os.path.exists(in_dtm_zsh_R):
                tu_input_functions.extract_dtm_zsh_to_domain(
                    workspace, domain, in_dtm_zsh_R
                )
                print("  ... done")
            else:
                raise KeyError(
                    f"ZSH polygon {in_dtm_zsh_R} \n  for DTM for city {city.name} or watershed {watershed} does not exist (variable: d_DTM_zsh_polygon_path)."
                )

            ### extracting zsh ADD L for DTM editing
        if tu_parameters["tu_what_to_do"]["Q_Do_ZSH_dtm_add_L"]:
            print("       - extracting zsh ADD L for editing of DTM")

            try:
                if tu_parameters["Q_peril"] == "pl_city":
                    in_dtm_zsh_L = tu_sources.d_DTM_zsh_add_L_path[city.name]
                else:
                    in_dtm_zsh_L = tu_sources.d_DTM_zsh_add_L_path[watershed]

            except KeyError:
                raise KeyError(
                    f"ZSH ADD L line  for DTM for city {city.name} or watershed {watershed} not defined in the configuration file (d_DTM_zsh_add_L_path)."
                )

            ## the source must be valid
            if os.path.exists(in_dtm_zsh_L):
                tu_input_functions.extract_dtm_zsh_L_to_domain(
                    workspace,
                    domain,
                    in_dtm_zsh_L,
                    output_res,
                )
                print("  ... done")
            else:
                raise KeyError(
                    f"ZSH add L line {in_dtm_zsh_L} \n  for DTM for city {city.name} or watershed {watershed} does not exist (variable: d_DTM_zsh_add_L_path)."
                )

        ### extract DTM
        ## msa debug print
        # print(tu_parameters["tu_what_to_do"]["Q_Do_DTM"])
        if tu_parameters["tu_what_to_do"]["Q_Do_DTM"]:
            if tu_parameters["Q_peril"] == "pl_city":
                peril_dict = {
                    "peril": tu_parameters["Q_peril"],
                    "resolution": city.dtm_resolution,
                    "allowed_resolutions": [str(city.dtm_resolution)],
                }

                # msa debug vymaz
                print(f"city: {str(city.name)}")
                dtm_path = tu_sources.d_DTM_path[str(city.name)][
                    tu_parameters["pluvial_resolution"]
                ]
                print(dtm_path)

                tu_input_functions.extract_dtms(
                    domain=domain,
                    dtm_grid=dtm_grid,
                    q_peril=peril_dict,
                    code_buff=code_buff,
                    DTM_switch=DTM_switch,
                    watershed=watershed,
                    city=city,
                )
            else:
                if tu_parameters["Q_peril"] == "pl":
                    peril_dict = {
                        "peril": tu_parameters["Q_peril"],
                        "resolution": tu_parameters["pluvial_resolution"],
                        "allowed_resolutions": [str(res) for res in resolutions],
                    }
                else:
                    peril_dict = {"peril": tu_parameters["Q_peril"]}
                ## msa debug print
                # print(peril_dict)
                tu_input_functions.extract_dtms(
                    domain=domain,
                    dtm_grid=dtm_grid,
                    q_peril=peril_dict,
                    code_buff=code_buff,
                    DTM_switch=DTM_switch,
                    watershed=watershed,
                )

        ### extract DTM - others
        if tu_parameters["tu_what_to_do"]["Q_Do_DTM_add"]:
            if any(tu_sources.d_DTM_add_specification):
                for (
                    key,
                    value,
                ) in tu_sources.d_DTM_add_specification.items():
                    print("       - extracting additional rasters: " + key, end="")
                    if tu_parameters["Q_peril"] == "pl_city":
                        tu_input_functions.extract_dem_add(
                            workspace,
                            domain,
                            code_buff,
                            tu_sources.d_DTM_add_path,
                            key,
                            value,
                            watershed=watershed,
                            city=city,
                        )
                    else:
                        tu_input_functions.extract_dem_add(
                            workspace,
                            domain,
                            code_buff,
                            tu_sources.d_DTM_add_path,
                            key,
                            value,
                            watershed=watershed,
                        )
                    print("... done")

        ### extract landuse for manning
        if tu_parameters["tu_what_to_do"]["Q_Do_Manning"]:
            print("       - extracting Manning", end="")
            # tmp_mat = f"/vsimem/{tu_input_functions.os.path.basename(mat_grid)}"
            tmp_mat = rf"/vsimem/mat_{domain}_tmp.tif"

            if tu_parameters["Q_peril"] == "pl_city":
                if city is not None:
                    if (
                        str(city.mesh_resolution)
                        in tu_sources.d_manning_path_local_raster[city.name]
                    ):

                        manning_file = tu_sources.d_manning_path_local_raster[
                            city.name
                        ][str(city.mesh_resolution)]

                    else:
                        print(
                            f"\n... Material not found for {city.name} in tu_sources.d_manning_path"
                        )
                        return
                extract_by_mask_rasterized(
                    raster=manning_file,
                    mask=code_buff,
                    output=tmp_mat,
                )  # , debug = True)
                convert_dtype(
                    tmp_mat,
                    mat_grid,
                    tu_input_functions.gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )

            elif tu_parameters["Q_peril"] == "pl":
                if (
                    tu_sources.d_manning_bldns_path.get(self.area[0]) is None
                    or tu_sources.d_manning_bldns_path.get(self.area[0]) == "None"
                ):
                    manning_file = tu_sources.d_manning_path[
                        tu_parameters["pluvial_resolution"]
                    ]
                else:
                    manning_file = tu_sources.d_manning_bldns_path
                extract_by_mask_rasterized(
                    raster=manning_file,
                    mask=code_buff,
                    output=tmp_mat,
                )  # , debug = True)
                convert_dtype(
                    tmp_mat,
                    mat_grid,
                    tu_input_functions.gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )

            else:
                extract_by_mask_rasterized(
                    raster=tu_sources.d_manning_path[main_resolution],
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
                    tu_input_functions.gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )

            print("... done")

        ### extract soil for infiltration
        if tu_parameters["tu_what_to_do"]["Q_Do_Soil"]:
            print("       - extracting Soil", end="")
            area_soil = tu_sources.d_soil_path.get(self.area)
            if area_soil is None:
                raise KeyError(
                    f"Area {self.area} not found in d_soil_path. "
                    f"Available sample keys: {list(tu_sources.d_soil_path.keys())[:10]}"
                )

            soil_res_key = str(tu_parameters["pluvial_resolution"])
            soil_layers = area_soil.get(soil_res_key)
            if soil_layers is None:
                # Be tolerant to yaml int keys if they slip through as integers.
                try:
                    soil_layers = area_soil.get(int(soil_res_key))
                except ValueError:
                    soil_layers = None

            if soil_layers is None:
                raise KeyError(
                    f"Soil resolution '{soil_res_key}' not found for area {self.area}. "
                    f"Available soil resolution keys: {list(area_soil.keys())}"
                )

            for soil_name in soil_layers:
                soil_raster = soil_layers[soil_name]
                tmp_soil = "/vsimem/soil_name.tif"
                extract_by_mask_rasterized(
                    raster=soil_raster,
                    mask=code_buff,
                    output=tmp_soil,
                )  # , debug = True)
                convert_dtype(
                    tmp_soil,
                    soil_grid.replace(".tif", f"_{soil_name}.tif"),
                    tu_input_functions.gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
            print("... done")

        if tu_parameters["tu_what_to_do"]["Q_Do_Drainage"]:
            print("       - extracting Soil drainage   ", end="")
            if tu_parameters["tu_values_default"]["Select_drainage_by"] == "domain":
                soil_layers = tu_sources.d_soil_draiange_path_default_rasters[
                    self.area[0][tu_parameters["pluvial_resolution"]]
                ]
            elif (
                tu_parameters["tu_values_default"]["Select_drainage_by"] == "watershed"
            ):
                soil_layers = tu_sources.d_soil_draiange_path_default_rasters[
                    watershed
                ][tu_parameters["pluvial_resolution"]]

            tmp_soil = "/vsimem/soil_drainage_name.tif"
            for layer in soil_layers:
                soil_drainage_raster = soil_layers[layer]
                extract_by_mask_rasterized(
                    raster=soil_drainage_raster,
                    mask=code_buff,
                    output=tmp_soil,
                    save_empty=True,
                )  # , debug = True)
                convert_dtype(
                    tmp_soil,
                    soil_grid.replace("2d_soil", f"2d_drain_{layer}"),
                    tu_input_functions.gdal.GDT_Float32,
                    creation_options=[
                        "COMPRESS=DEFLATE",
                        "NUM_THREADS=ALL_CPUS",
                        "ZLEVEL=12",
                    ],
                )
            print("... done")
        tu_input_functions.os.remove(code_buff)
        # # print(time.monotonic() - start)

        ### extract hydrology - bcdbase files from commoninputs into the local domain folder
        if tu_parameters["tu_what_to_do"]["Q_Do_BC_Dbase_local"]:
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

            path_bc_dbase_inputs = (
                tu_sources.bc_dbase_common_path_idf
                if tu_parameters["Q_peril"] == "pl_city"
                else tu_data.path_bc_dbase_inputs
            )

            tu_input_functions.extract_bc_dbase_to_local(
                workspace,
                domain,
                path_bc_dbase_inputs,
                tu_parameters["Q_peril"],
                bc_files,
            )
            print("... done")

        if tu_parameters["tu_what_to_do"]["Q_Do_TUFLOW_files"]:
            print("       - writing TUFLOW text inputs", end="")

            ### write all tuflow txt files:
            # restart = False
            # if Q_peril == "fl":
            #     restart = True
            is_levee, is_channel, is_culvert, is_zsh_polygon, is_zsh_L = (
                False,
                False,
                False,
                False,
                False,
            )
            if tu_parameters["Q_peril"] == "pl_city":
                dtm_resolution = str(city.mesh_resolution)

            elif tu_parameters["Q_peril"] == "pl":
                dtm_resolution = tu_parameters["pluvial_resolution"]
            else:
                dtm_resolution = main_resolution

            tu_input_functions.write_tgc_file(
                domain,
                workspace,
                tu_sources.DTMs_combination[DTM_switch],
                main_resolution=dtm_resolution,
                mesh_parameters_grid=(
                    mat_grid if tu_parameters["Q_peril"] == "pl_city" else None
                ),  ## da se jiny grid nez dtm - material nebo treba soil viz.tu_input_names mat_grid,soil_grid,dtm_grid nebo full path to a grid
                ## msa 2026-02-14 zmena z mat na soil [pr 10m pod 5m (soil ma res.30m)
                # mesh_parameters_grid = mat_grid.replace(".tif", "_layer_0_30.tif").replace('2d_mat_','2d_soil_') if tu_parameters[
                #                                        "Q_peril"] == "pl_city" else None,  ## da se jiny grid nez dtm - material nebo treba soil viz.tu_input_names mat_grid,soil_grid,dtm_grid nebo full path to a grid
            )

            tu_input_functions.write_tcf_local_inputs(
                domain,
                workspace,
                is_levee,
                is_channel,
                is_culvert,
                is_zsh_polygon,
                is_zsh_L,
                tu_parameters["tu_values_default"]["Set_Zpts_Order"],
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )
            tu_input_functions.write_ecf_file(
                domain,
                workspace,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )
            tu_input_functions.write_tbc_file(
                domain,
                workspace,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )
            # if restart:
            tu_input_functions.write_restart_file(
                domain,
                workspace,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )
            tu_input_functions.write_tef_file(
                domain,
                workspace,
                watershed,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )

            quadtree = tu_input_functions.is_quadtree(
                multi_scenario=tu_parameters["multi_scenario"],
                multi_scenario_path=tu_sources.multi_scenario_path,
            )

            tu_input_functions.write_tcf_file(
                domain,
                workspace,
                quadtree,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )
            tu_input_functions.write_common_input_model(
                workspace, domain, tu_data.path_common_inputs
            )

            ### write bat file - to be able to run tuflow manually domain per domain.

            ### tu_sources.tuflow_version
            tu_input_functions.write_bat_test_files(
                domain,
                workspace,
                watershed,
                tu_sources.tuflow_version,
                mesh_resolution=(
                    city.mesh_resolution
                    if tu_parameters["Q_peril"] == "pl_city"
                    else None
                ),
            )

            tu_input_functions.write_bat_file(
                workspace,
                domain,
                tu_parameters["multi_scenario"],
                tu_sources.multi_scenario_path,
                manual_scenario=tu_parameters.get("manual_scenario"),
            )

            print("... done")
            print(
                "       - zsh order: "
                + tu_parameters["tu_values_default"]["Set_Zpts_Order"]
            )

        if tu_parameters["tu_what_to_do"]["Q_Do_Create_Yaml"]:
            print("       - writing scenario (multi) for TUFLOW", end="")
            other_values = {}
            if tu_parameters["Q_peril"][:2] == "pl":
                auto_restart = False
            else:
                auto_restart = tu_parameters["tu_values_default"]["Auto_Restart"]
                other_values["peril"] = tu_parameters["Q_peril"]
                other_values["resolution"] = f"{main_resolution}m"
            if tu_parameters["Q_peril"][:2] == "pl":
                other_values["peril"] = tu_parameters["Q_peril"]
                # other_values["resolution"] = f"{tu_parameters['pluvial_resolution']}m"

                other_values["resolution"] = (
                    f"{city.mesh_resolution}m"
                    if tu_parameters["Q_peril"] == "pl_city"
                    else f"{tu_parameters['pluvial_resolution']}m"
                )

                other_values["schema"] = (
                    f"{tu_sources.config_variables['database_schema']}.pluvial.Calculation"
                )

            c = tu_input_functions.create_scenario_input_multi(
                domain=domain,
                workspace=workspace,
                peril_path=tu_sources.tuflow_directory[tu_parameters["Q_peril"]],
                tuflow_yaml_path=tu_sources.tuflow_yaml_path,
                multi_scenario=tu_parameters["multi_scenario"],
                project=tu_parameters["project"],
                project_start_yaml=tu_parameters["project_start_scenario"],
                tuflow_version=tu_sources.tuflow_version,
                precision=tu_parameters["tu_values_default"]["Precision"],
                auto_restart=auto_restart,
                max_time=tu_parameters["tu_values_default"]["Max_Time"],
                time_shift=tu_parameters["tu_values_default"]["Time_Shift"],
                create_without_domain=False,
                inflow_type=inflow_type,
                other_values=other_values,
            )
            if not c:
                tu_parameters["create_scenario_input_errlog"].append(
                    f"{domain} - Scenario didnt created on the TUFLOW server."
                )

        if True not in [
            values for key, values in tu_parameters["tu_what_to_do"].items()
        ]:
            print("       - nothing to do.")
        if tu_parameters["tu_what_to_do"]["Q_Do_Mannig_Culvert_channel"]:
            tu_input_functions.change_2d_mat(
                workspace,
                domain,
                tu_parameters["tu_values_default"]["Channel_value"],
                tu_parameters["tu_values_default"]["Channel_width"],
            )
        ## reset user defined zsh order to the default
        tu_parameters["tu_values_default"]["Set_Zpts_Order"] = tu_data.rest_value
        ## don't change


def multiprocessing_helper(domain):
    area = domain.split("_")[0]
    if tu_parameters["Q_peril"] == "pl":
        workspace = os.path.join(
            tu_data.workspace,
            f"{tu_parameters['pluvial_resolution']}m",
            area,
        )
    else:
        workspace = os.path.join(tu_data.workspace, area)
    try:
        domains_table = tu_input_functions.read_domains_file(
            tu_sources.domains_list, area
        )
        TuflowInputs(domain, workspace, domains_table)
    except Exception as e:
        return f"{domain},error: {e}"


if __name__ == "__main__":
    # get today date
    start = time.strftime("%Y%m%d", time.localtime())
    if not tu_parameters["use_multiprocessing"]:
        errors = []
        current_area = None
        current_domains_table = None
        for domain in user_dirs:
            area = domain.split("_")[0]
            if area != current_area:
                current_domains_table = tu_input_functions.read_domains_file(
                    tu_sources.domains_list, area
                )
                current_area = area
            if tu_parameters["Q_peril"] == "pl_city":
                workspace = os.path.join(
                    ## msa pluvial_resolution_mesh intead of pluvial_resolution
                    tu_data.workspace,
                    f"{tu_parameters['pluvial_resolution_mesh']}m",
                    area,
                )

                ###msa debug print
                print(f"workspace for domain {domain}: {workspace}")

            elif tu_parameters["Q_peril"] == "pl":
                workspace = os.path.join(
                    ## msa pluvial_resolution_mesh intead of pluvial_resolution
                    tu_data.workspace,
                    area,
                    f"{tu_parameters['pluvial_resolution_mesh']}m",
                )
                ###msa debug print
                print(f"workspace for domain {domain}: {workspace}")
            else:
                workspace = os.path.join(tu_data.workspace, area)
            try:
                TuflowInputs(domain, workspace, current_domains_table)
            except Exception as e:
                print(f"       - ERROR {domain}: {e}")
                traceback.print_exc()
                errors.append(f"{domain},error: {e}")
        if errors:
            folder = os.path.dirname(tu_sources.tuflow_yaml_path)
            ## msa kde je log file
            print(f'\nlog file {os.path.join(folder, f"tu_extract_log_{start}.txt")}')
            with open(
                os.path.join(folder, f"tu_extract_log_{start}.txt"), "w"
            ) as log_file:
                for result in errors:
                    if isinstance(result, str):
                        log_file.write(result + "\n")

    else:
        import multiprocessing as mp

        pool = mp.Pool(int(float(tu_parameters["multiprocessing_cores"])))
        results = pool.map(multiprocessing_helper, user_dirs)
        pool.close()
        pool.join()
        if results:
            folder = os.path.dirname(tu_sources.tuflow_yaml_path)

            with open(
                os.path.join(folder, f"tu_extract_log_{start}.txt"), "w"
            ) as log_file:
                for result in results:
                    if isinstance(result, str):
                        log_file.write(result + "\n")

    ##msa
    print("\n ######## All done ########")
