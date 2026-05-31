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


yaml.SafeLoader.construct_mapping_org = yaml.SafeLoader.construct_mapping
yaml.SafeLoader.construct_mapping = my_construct_mapping

master_inputs_file = "main_setup_fluvial.yaml"
with open(os.path.join(os.path.dirname(__file__), master_inputs_file), "r") as file:
    config_variables = yaml.safe_load(file)
## List of Areas
list_of_areas = [
    f"{letter.upper()}{str(number).zfill(2)}"
    for letter in alc
    for number in range(0, 100)
]
prj_file = config_variables["prj_file"]
snap_raster = config_variables["snap_raster"]

## AREA: dictionary which contains region code (like "W51") and folder name of this region (like "W51_Toronto_Area")
d_AreaFolderName = {k: k for k in list_of_areas}

## dictionary which contains paths to source DTM RASTER file for each region
d_DTM_path = config_variables["d_DTM_path"]

## MANNING: root paths to folder with source of RASTER files with materials/landuse layers
d_manning_path_default_raster = config_variables["manning_default"]
d_manning_path_bldns_raster = config_variables["manning_buildings_default"]

d_manning_path = d_manning_path_default_raster
d_manning_bldns_path = d_manning_path_bldns_raster


## SOIL: root paths to folder with source of RASTER files with soil layers - for pluvial
d_soil_path_default_rasters = config_variables["soil_file"]
d_soil_path = {k: d_soil_path_default_rasters for k in list_of_areas}

## DRAINAGE:
d_soil_draiange_path_default_rasters = config_variables["layer_drainage"]

#### WHERE the order of culvert and levees should be not default (reversed) - default is: culverts -> channell
user_dirs_levee_first = config_variables["user_dirs_levee_first"]


### ZSH culverts

d_path_in_zsh_culvert_L = config_variables["zsh_culvert_L"]
d_path_in_zsh_culvert_P = config_variables["zsh_culvert_P"]

### ZSH channels
d_path_in_zsh_channel_L = config_variables["zsh_channel_L"]
d_path_in_zsh_channel_P = config_variables["zsh_channel_P"]

### ZSH levees
d_path_in_zsh_levee_L = config_variables["zsh_levee_L"]
d_path_in_zsh_levee_P = config_variables["zsh_levee_P"]


tuflow_directory = {
    "fl": "06_TUFLOW_FL",
    "rf": "06_TUFLOW",
    "pl": "06_TUFLOW_PL",  # '06_TU_Pluvial'
}
d_root_path_net = config_variables["root_path"]


###########################
## ADDITIONAL INPUTS : DEMs
## keys in d_DTM_add_specification and d_DTM_add_path must be the same
d_DTM_add_path = config_variables["DTM_add_path"]
## d_DTM_add_path - use only the keys for RASTER you want to extract for domains
d_DTM_add_specification = config_variables["DTM_add_specification"]

DTMs_combination = config_variables["DTMs_combination"]
folder_common_inputs_model = config_variables["folder_common_inputs_default"]

domains_list = config_variables["domains_list"]

## fixed path to Jinja2 template files
template_path = config_variables["template_path"]

## fixed paths to scenario CSV folders (independent of domain_directory_source)
multi_scenario_path_cfg = config_variables["multi_scenario_path"]

## YAML scenario generation parameters
tuflow_yaml_path_cfg = config_variables.get("tuflow_yaml_path")
yaml_project = config_variables.get("project")
yaml_peril_path = config_variables.get("peril_path")
yaml_project_start_yaml = config_variables.get("project_start_scenario")
yaml_precision = config_variables.get("precision", "iSP")
yaml_auto_restart = config_variables.get("auto_restart", False)
yaml_max_time = config_variables.get("max_time", 200)
yaml_time_shift = config_variables.get("time_shift", 10)
yaml_schema = config_variables.get("schema", None)
tuflow_version_cfg = config_variables.get("tuflow_version")
