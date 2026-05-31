"""
v210531 - (c) msa ;
    zmena u drsnosti. Variabilni na rp. upraveno v tcf
    pridano d_fl_rp_mat_file
            d_pl_rp_mat_file
            selected_n_man_rp_files
"""

# reading config_yaml file
import yaml
import os
from string import ascii_lowercase as alc


def my_construct_mapping(self, node, deep=False):
    data = self.construct_mapping_org(node, deep)
    return {(str(key) if isinstance(key, int) else key): data[key] for key in data}


class TuInputSources:
    def __init__(self, main_setup_yaml):
        yaml.SafeLoader.construct_mapping_org = yaml.SafeLoader.construct_mapping
        yaml.SafeLoader.construct_mapping = my_construct_mapping
        master_inputs_file = main_setup_yaml
        with open(
            os.path.join(os.path.dirname(__file__), master_inputs_file), "r"
        ) as file:
            print(f"Loadfing from: {file}")
            self.config_variables = yaml.safe_load(file)
        ## List of Areas
        list_of_areas = [
            f"{letter.upper()}{str(number).zfill(2)}"
            for letter in alc
            for number in range(0, 100)
        ]
        self.prj_file = self.config_variables["prj_file"]
        self.snap_raster = self.config_variables["snap_raster"]

        ## AREA: dictionary which contains region code (like "W51") and folder name of this region (like "W51_Toronto_Area")
        self.d_AreaFolderName = {k: k for k in list_of_areas}

        ## dictionary which contains paths to source DTM RASTER file for each region
        self.d_DTM_path = self.config_variables["d_DTM_path"]

        ## dictionary which contains paths to ZSH polygond for editing holes and gaps in DTM raster for each region
        self.d_DTM_zsh_R_path = self.config_variables["d_DTM_zsh_polygon_path"]

        ## dictionary which contains paths to ZSH add L lines  for editing in DTM raster for each region
        self.d_DTM_zsh_L_path = self.config_variables["d_DTM_zsh_add_L_path"]

        ## MANNING: root paths to folder with source of RASTER files with materials/landuse layers
        self.d_manning_path_default_raster = self.config_variables["manning_default"]
        self.d_manning_path_bldns_raster = self.config_variables[
            "manning_buildings_default"
        ]
        self.d_manning_path_local_raster = self.config_variables[
            "manning_path_local_raster"
        ]

        self.d_manning_path = self.d_manning_path_default_raster
        self.d_manning_bldns_path = self.d_manning_path_bldns_raster

        ## SOIL: root paths to folder with source of RASTER files with soil layers - for pluvial
        self.d_soil_path_default_rasters = self.config_variables["soil_file"]
        self.d_soil_path = {k: self.d_soil_path_default_rasters for k in list_of_areas}

        ## DRAINAGE:
        self.d_soil_draiange_path_default_rasters = self.config_variables[
            "layer_drainage"
        ]

        ## msa pridavam promenou co se pouzije lib_1_tu_extract
        # ##Virtual Pipes
        self.d_VP_pit_points = self.config_variables["VP_pit_points"]

        ## msa pridavam promenou co se pouzije lib_1_tu_extract
        # ##zsh dtm polygons
        self.d_DTM_zsh_polygon_path = self.config_variables["d_DTM_zsh_polygon_path"]
        # ##zsh dtm add L
        self.d_DTM_zsh_add_L_path = self.config_variables["d_DTM_zsh_add_L_path"]

        ### msa DB source for IDF, if it is in different folders tree (for cities)
        self.bc_dbase_common_path_idf = self.config_variables[
            "bc_dbase_common_path_idf"
        ]

        #### WHERE the order of culvert and levees should be not default (reversed) - default is: culverts -> channell
        self.user_dirs_levee_first = self.config_variables["user_dirs_levee_first"]

        ### ZSH culverts

        self.d_path_in_zsh_culvert = self.config_variables["zsh_culvert"]

        ### ZSH channels
        self.d_path_in_zsh_channel = self.config_variables["zsh_channel"]

        ### ZSH levees
        self.d_path_in_zsh_levee = self.config_variables["zsh_levee"]

        self.tuflow_directory = {
            "fl": "06_TUFLOW",
            "rf": "06_TUFLOW",
            "pl": "06_TU_Pluvial",  # '06_TU_Pluvial'
            "pl_city": "06_TUFLOW_PL",  # '06_TU_Pluvial'
        }
        self.d_root_path_net = self.config_variables["root_path"]
        self.tuflow_yaml_path = self.config_variables["tuflow_yaml_path"]

        ###########################
        ## ADDITIONAL INPUTS : DEMs
        ## keys in d_DTM_add_specification and d_DTM_add_path must be the same
        self.d_DTM_add_path = self.config_variables["DTM_add_path"]
        ## d_DTM_add_path - use only the keys for RASTER you want to extract for domains
        self.d_DTM_add_specification = self.config_variables["DTM_add_specification"]

        self.DTMs_combination = self.config_variables["DTMs_combination"]
        self.folder_common_inputs_model = self.config_variables[
            "folder_common_inputs_default"
        ]

        ## msa nevim zda je nasledujici promena potreba i pro yaml, uvidi se pri uprave dalsich casti
        ## bc_dbase_common_path_idf = config_variables["bc_dbase_common_path_idf"]

        self.domains_list = self.config_variables["domains_list"]

        # multiscenario path
        self.multi_scenario_path = self.config_variables["scenarios_folder"]
        self.tuflow_version = self.config_variables["tuflow_version"]


class TuParameters:
    def __init__(self, tu_what_to_do_yaml):
        # yaml.SafeLoader.construct_mapping_org = yaml.SafeLoader.construct_mapping
        # yaml.SafeLoader.construct_mapping = my_construct_mapping
        master_inputs_file = tu_what_to_do_yaml
        with open(
            os.path.join(os.path.dirname(__file__), master_inputs_file), "r"
        ) as file:
            self.config_variables = yaml.safe_load(file)
