### start of the script body. User variable:
import os
from tu_input_functions import (
    get_zpts_order_list,
    filter_list_dirs,
    get_user_dirs,
    read_domains_file,
)
from tu_input_sources import (
    d_root_path_net,
    tuflow_directory,
    user_dirs_levee_first,
    folder_common_inputs_model,
    domains_list,
)

### Choose basic source, area, resolution and peril for TUFLOW computation:

domain_directory_source = "B"  ## 'B' - Source location of input domains - other options e.g. 'D_local', 'P005_e'
area = "M06"  ## tuflow area name: 'A42', 'W42'...
code_buffer = "30"  ##
Q_peril = "fl"  ## 'fl'|'rf'|'pl': 'fl' - fluvial, 'pl' - pluvial
TUFLOW_server = "P005_d"  ## nickname of th TUFLOW server - where domain will be send for computation

temp = r"d:\01_Projects\157_Canada_Flood_v3\01_MD\01_HAZARD\01_DTM\tmp"

### Update domain by raster and hydrology, TUFLOW text files creation, create scenarios, sending to the computational server
### JV - set for fluvial
tu_what_to_do = {
    "Q_Do_DTM": "y",  ## create DTM raster - set "y"
    "Q_Do_DTM_add": "n",  ## create DTM raster from additional layers, e.g. buildings, waterbodies...
    "Q_Do_Manning": "y",  ## create material (2d_mat* layer) - set "y"
    "Q_Do_Soil": "n",  ## create soil (2d_soil* layer) - set "y"
    "Q_Do_Drainage": "n",  ## create drainage (2d_soil_drain* layer) - set "y"
    "Q_Do_BC_Dbase_local": "y",  ## create copy of the hydrology into given domain, the hydrology inputs must exist in the CommonInput folder on the source-drive
    "Q_Do_ZSH_Levees": "n",  ## create local ZSH lines based on the global layer: levees
    "Q_Do_ZSH_Channels": "n",  ## create local ZSH lines based on the global layer: channel
    "Q_Do_ZSH_Culverts": "n",  ## create local ZSH lines based on the global layer: culverts
    "Q_Do_Mannig_Culvert_channel": "n",  ## change value in 2d_mat file for culverts and channels, 2d_mat must already exist in the domain
    #
    "Q_Do_TUFLOW_files": "y",  ## create all TUFLOW text files - e.g. .tcf, .tgc....
    #
    "Q_Do_CopyDomains_to_server": "y",  ## copy all domains to chosen server for the computation
    "Q_Do_Create_Scenario": "y",  ## crete a scenario to start the computation on the server
}

### USER defined domains for modification:
user_dirs = [
    "M06_74000027",
    "M06_74000028",
    # "M06_74000029",
    # "M06_74000253",
    # "M06_74000271",
    # "M06_74000329",
    # "M06_7400040002",
    "M06_74000468",
    "M06_74001512",
    "M06_74001709",
    "M06_74001717",
    # "M03_7400004302",
    # "M03_74000051",
    # "M03_74000426",
    # "M03_74000465",
    # "M03_74000487",
    # "M03_74000494",
    # "M03_74000508",
    # "M03_74000716",
    # "M03_7400079102",
    # "M03_74001719",
    # "M03_74001729",
]
user_dirs_condition_values = [
    00000000,
    9999999999,
]  # 4000] ## to filter dirs in between two values in the list; to filter creeks use  [5000,10000] |


### Select scenarios for TUFLOW domain computation:
use_scenario = "multi"  ## multi | manual - use predefined CSV scenario in 03_bat/multi folder, or manual created scenario using pattern below

# multi_scenario = "fl_RPall_C_HPC_30m_main_RP"  ## name of template saved in '06_TUFLOW\03_batLists'fl_RPall_C_HPC_30m
# multi_scenario = "fl_RPall_C_HPC_30m_secondary_RP"
# multi_scenario = "fl_RPall_R_HPC_30m_main_RP "
# multi_scenario = "fl_RPall_R_HPC_30m_secondary_RP "
# multi_scenario = "fl_RPall_C_HPC_10m_main_RP"
# multi_scenario = "fl_RPall_C_HPC_10m_secondary_RP"
# multi_scenario = "fl_RPall_R_HPC_10m_main_RP"
# multi_scenario = "fl_RPall_R_HPC_10m_secondary_RP"
# multi_scenario = "fl_RPall_L_HPC_30m_main_RP "
multi_scenario = "fl_RPall_R_HPC_30m_TBa"

# multi_scenario = 'fl_RPall_major_lp'   'fl_RPall_major_lp_rest'  #'fl_RPall_major_lp_5_1h_10k'

if "secondary" in multi_scenario:
    tu_what_to_do["Q_Do_CopyDomains_to_server"] = "n"
print(tu_what_to_do)

### Modify parameters of ZSH lines extracted into domains
tu_zsh_lines_switch = {
    ##CULVERTS:
    "zsh_culvert_CFW": 0.7,  # (interval 0-100% - reduction of the capacity)
    "zsh_culvert_FLC": 0,
    "zsh_culvert_dZ": 0,
    ##CHANNELS:
    "zsh_channel_dZ": 0,  # -0.75,
    ##LEVEES:
    "zsh_levee_dZ": 0.0,
}

### TUFLOW defaults - changeable by user:
tu_values_default = {
    "TUFLOW_release": "2023-03-AF",
    "bc_dbase_location": "local",  ## 'CommonInputs', 'local' - in domain =>
    "Set_Zpts": -0.5,  ## for TGC command Set Zpts == -0.5 (default elevation of terrain)
    "Set_Mat": 1,  ## for TGC command Set Mat == 501 (default material ID)
    "Set_Soil": 1,
    ## for TGC command Set Soil == 501 (default soil ID) - Set up value is necessary only if soil is used
    "Set_IWL": -10,  ## for TGC command Set IWL == 501  (default initial water level)
    "Set_Zpts_Order": "auto",
    ## 'auto' - order by the list = preferred option!; #'default' = 'levees_last'; 'levees_first'
    "Q_Use_VirtualPipes": "y",
    "channel_value": 80,  # value that will be set to 2d_mat file for the channel/culvert
    "channel_width": 30,  # width of the channel/culvert for 2d mat file
    ## default: 'y' = 'if_available' - Use VP only, but only if available; "no" - never use the VP
}

### SCRIPT BODY - no need to change
### modify  only for new project - if there is a need to have a different setup
###

## temp - folder - local!
if not os.path.exists(temp):
    os.makedirs(temp)

### configuration for batfiles
tu_bat_variables_fl = {
    "scenarios": {
        "start_time": "s0",
        "end_time": "e40",
        "method": "H00",
        "dtm_res": "D05",  # edit by DTM
        "output_time": "o10",
        "output_size": "ms5",  # edit by DTM
        "restart": "Hu",
    },
    "events": {
        "infiltration": ["I"],
        "inflow": ["C"],
        "rp": [
            "f00005",
            "f00020",
            "f00050",
            "f00100",
            "f00200",
            "f00500",
            "f01000",
            "f10000",
        ],
    },
}

### configuration for batfiles
tu_bat_variables_rf = {
    "scenarios": {
        "start_time": "s0",
        "end_time": "e24",
        "method": "Q00",
        "dtm_res": "D01",
        "output_time": "o24",
        "output_size": "ms10",
        "restart": "Hs",
    },
    "events": {
        "infiltration": ["I00", "I50", "I90"],
        "inflow": ["d360", "d1440"],
        "rp": [
            "r00005",
            "r00020",
            "r00050",
            "r00100",
            "r00200",
            "r00500",
            "r01000",
            "r10000",
        ],
    },
}

# 'Inf0' - no infiltration; no soil
# 'I' for NO rainfall simulation. Just short scenario for using for river flood modelling without any infiltration
# 'GA50' - 50% of saturation; infiltration by Green Ampt method
# 'GA00' - 0% of saturation; infiltration by Green Ampt method

### various defaults for various perils
tu_features_switch = {
    "BC_SA_Q_by": (
        "y" if Q_peril == "fl" else "n"
    ),  ## pluvial doesnt use SA_IN polygon/stream
    "PO_Q_by": ("y" if Q_peril == "fl" else "n"),  ## pluvial doesnt use PO line
    "BC_RF_Q_by": "2d_rf",  #'2d_rf' #'2d_code'
    "Soil_Q_by": (
        "n" if Q_peril == "fl" else "y"
    ),  ## use soil for computation? pluvial doesnt use soil
}

## defining tuflow directory - source of domain on the main input sources server
workspace = os.path.join(
    d_root_path_net[domain_directory_source], tuflow_directory[Q_peril], area[:3]
)
#                         f'{output_resolution}m')

## defining tuflow server directory - the path on the tuflow server whre the computation will be performed
workspace_ts = os.path.join(d_root_path_net[TUFLOW_server], tuflow_directory[Q_peril])

## path to the list of scenarios - the paterns
multi_scenario_path = os.path.join(
    d_root_path_net[domain_directory_source],
    tuflow_directory[Q_peril],
    "03_bat",
)

area_key = area  # + '_' + output_resolution

## path to the CommonInputs directory on the source server
path_common_inputs = os.path.join(
    folder_common_inputs_model,
    tuflow_directory[Q_peril],
    "01_common",
)
path_bc_dbase_inputs = os.path.join(
    folder_common_inputs_model, tuflow_directory[Q_peril], "04_bc_dbase"
)

if "user_dirs" not in locals() or not user_dirs:
    user_dirs = get_user_dirs(workspace=workspace, area=area)
else:
    print("Domain directories are defined by user!!!")

user_dirs = filter_list_dirs(user_dirs, area, user_dirs_condition_values)

## looking for the list where the user specified in that some domains should have a reverse order of the zsh lines
## default is channel/culverts - levees, reverse is levees - channel/culverts
list_zsh_reverse_domain = get_zpts_order_list(user_dirs_levee_first, area)
rest_value = tu_values_default["Set_Zpts_Order"]  # don't change.

copy_domain_to_ts_errlog = []  ## for log
create_scenario_input_errlog = []  ## for log
domains_with_dtm_resolution = read_domains_file(domains_list, area)
