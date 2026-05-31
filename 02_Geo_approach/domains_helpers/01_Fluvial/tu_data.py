"""
Load the fluvial extract-pipeline run configuration from tu_what_to_do.yaml and
reproduce the derived values that 01_tu_extract_run.py consumes.

This replaces the former tu_what_to_do.py: the user-editable settings now live in
tu_what_to_do.yaml, while the (unchanged) derived "script body" lives here.
"""

import os
import yaml

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
    multi_scenario_path_cfg,
    # YAML scenario-generation parameters - re-exported so 01_tu_extract_run.py
    # can keep importing them from this module.
    yaml_auto_restart,
    yaml_max_time,
    yaml_peril_path,
    yaml_precision,
    yaml_project,
    yaml_project_start_yaml,
    yaml_schema,
    yaml_time_shift,
)

### Load user configuration ----------------------------------------------------
with open(
    os.path.join(os.path.dirname(__file__), "tu_what_to_do.yaml"), "r"
) as _file:
    _cfg = yaml.safe_load(_file)

domain_directory_source = _cfg["domain_directory_source"]
area = _cfg["area"]
code_buffer = _cfg["code_buffer"]
Q_peril = _cfg["Q_peril"]
TUFLOW_server = _cfg["TUFLOW_server"]
temp = _cfg["temp"]
use_scenario = _cfg["use_scenario"]
multi_scenario = _cfg["multi_scenario"]
user_dirs = _cfg.get("user_dirs") or []
user_dirs_condition_values = _cfg["user_dirs_condition_values"]
tu_what_to_do = _cfg["tu_what_to_do"]
tu_zsh_lines_switch = _cfg["tu_zsh_lines_switch"]
tu_values_default = _cfg["tu_values_default"]
tu_bat_variables_fl = _cfg["tu_bat_variables_fl"]
tu_bat_variables_rf = _cfg["tu_bat_variables_rf"]

# Copy-to-server is disabled: the extract pipeline no longer pushes domains to a
# TUFLOW server. Use FL03_tuflow_test_run.py (local CPU test run) followed by
# FL04_copy_to_distribution.py instead. Forced off here so it cannot run even if
# the YAML sets it to "y".
tu_what_to_do["Q_Do_CopyDomains_to_server"] = "n"
print(tu_what_to_do)

### SCRIPT BODY - no need to change --------------------------------------------
### modify only for new project - if there is a need to have a different setup

## temp - folder - local!
if not os.path.exists(temp):
    os.makedirs(temp)

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

## defining tuflow server directory - the path on the tuflow server where the computation will be performed
workspace_ts = os.path.join(d_root_path_net[TUFLOW_server], tuflow_directory[Q_peril])

## path to the list of scenarios - always from the fixed path defined in YAML
multi_scenario_path = multi_scenario_path_cfg[Q_peril]

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

if not user_dirs:
    user_dirs = get_user_dirs(workspace=workspace, area=area)
else:
    print("Domain directories are defined by user!!!")

user_dirs = filter_list_dirs(user_dirs, area, user_dirs_condition_values)

## looking for the list where the user specified that some domains should have a reverse order of the zsh lines
## default is channel/culverts - levees, reverse is levees - channel/culverts
list_zsh_reverse_domain = get_zpts_order_list(user_dirs_levee_first, area)
rest_value = tu_values_default["Set_Zpts_Order"]  # don't change.

copy_domain_to_ts_errlog = []  ## for log
create_scenario_input_errlog = []  ## for log
domains_with_dtm_resolution = read_domains_file(domains_list, area)
