### start of the script body. User variable:
import os
from tu_input_sources_cities import TuInputSources, TuParameters

### Choose basic source, area, resolution and peril for TUFLOW computation:

tu_parameters = TuParameters("tu_what_to_do_city_yaml.yaml").config_variables
if tu_parameters.get("is_test_run", False):
    if not tu_parameters.get("multiscenario_test"):
        raise KeyError("multiscenario_test must be set when is_test_run is True")
    tu_parameters["multi_scenario"] = tu_parameters["multiscenario_test"]

main_setup_yaml = tu_parameters.get("main_setup_yaml", "main_setup_pluvial_cities.yaml")
tu_sources = TuInputSources(main_setup_yaml)


def _first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


## temp - folder - local!
if not os.path.exists(tu_parameters["temp"]):
    os.makedirs(tu_parameters["temp"])


## defining tuflow directory - source of domain on the main input sources server
workspace = os.path.join(
    tu_sources.d_root_path_net[tu_parameters["domain_directory_source"]],
    tu_sources.tuflow_directory[tu_parameters["Q_peril"]],
)


## path to the list of scenarios - the paterns

## path to the CommonInputs directory on the source server
use_peril_for_common_inputs = (
    "pl" if tu_parameters["Q_peril"] == "pl_city" else tu_parameters["Q_peril"]
)

path_common_inputs = os.path.join(
    tu_sources.folder_common_inputs_model,
    tu_sources.tuflow_directory[use_peril_for_common_inputs],
    "01_common",
)

root_hazard = tu_sources.d_root_path_net[tu_parameters["domain_directory_source"]]
peril_folder = tu_sources.tuflow_directory[tu_parameters["Q_peril"]]

path_common_inputs = _first_existing_path(
    [
        path_common_inputs,
        os.path.join(tu_sources.folder_common_inputs_model, "01_common"),
        tu_sources.folder_common_inputs_model,
        os.path.join(root_hazard, peril_folder, "02_common"),
        os.path.join(root_hazard, peril_folder, "01_common"),
    ]
)

path_bc_dbase_inputs = _first_existing_path(
    [
        os.path.join(
            tu_sources.folder_common_inputs_model,
            tu_sources.tuflow_directory[use_peril_for_common_inputs],
            "04_bc_dbase",
        ),
        os.path.join(root_hazard, peril_folder, "04_bc_dbase"),
        os.path.join(root_hazard, peril_folder, "02_common", "04_bc_dbase"),
    ]
)


## looking for the list where the user specified in that some domains should have a reverse order of the zsh lines
## default is channel/culverts - levees, reverse is levees - channel/culverts
rest_value = tu_parameters["tu_values_default"]["Set_Zpts_Order"]  # don't change.

copy_domain_to_ts_errlog = []  ## for log
create_scenario_input_errlog = []  ## for log
