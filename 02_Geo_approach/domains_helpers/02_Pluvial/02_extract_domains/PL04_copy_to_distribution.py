"""
This script automates the process of copying TUFLOW domain folders and associated YAML configuration files
from a local source to a designated storage machine for flood modeling runs. It ensures that each domain's
data is distributed to the correct machine, handles manual restarts, and manages overwriting of existing data.
Main Features:
- Reads a list of domains to process from a CSV file.
- For each domain, parses its YAML configuration file.
- Checks if the domain folder already exists on any storage machine.
- Copies the domain folder to the specified storage machine, with options to overwrite or reuse existing data.
- Validates the presence of required TSF log files before copying.
- Handles manual restart scenarios by updating the YAML configuration.
- Moves processed YAML files to a tagged archive folder.
- Logs any issues encountered during processing to a CSV file.
Classes:
    CopyRunStart:
        Handles the logic for copying domain folders, creating YAML files on the distribution machine,
        checking for existing folders, validating TSF files, and archiving processed YAML files.
Configuration:
- Paths and machine mappings are defined at the top of the script.
- Manual restart scenarios can be configured via the MANUAL_RESTART dictionary.
Usage:
Run the script as a standalone program. It will process all domains listed in the specified CSV file,
copy their data as needed, and log any issues encountered.
"""

import os
import glob
import shutil
import time
import yaml
import pandas as pd

# pick a machine, where data is stored for TUFLOW distribution, must be some workstation, not you personal machine!!!!
DATA_STORAGE = "p134_e"
# path where the local yaml files are stored, output from 01_tu_extract.py
TUFLOW_YAML_PATH = r"D:\01_Projects\900_FL_Europe\01_MD\01_HAZARD\06_TUFLOW_PL\_start_yaml\900_FL_Europe\yaml"
OVERWRITE_ON_STORAGE = (
    True  # If True, will overwrite existing folders on storage machines
)

# List of domains to process, csv with DomainID
DOMAINS_TO_DO = r"V:\01_Projects\900_FL_Europe\01_MD\01_HAZARD\06_TUFLOW_PL\_dtd_PL\_cities\N05_pluvial_reference.csv"


RESOLUTION = "10m"  # Resolution of the domains, used to find the yaml files
FLUVIAL_PLUVIAL = "pl"  # Fluvial or Pluvial, used to find the yaml files
TEST_RPS = {
    "pl": ["f00005"],
}  # RPs to test run, can be changed
# time.sleep(3670)
MANUAL_RESTART = {
    "restarting": False,  # Flag to indicate if manual restart is needed - use with domains to do and fluvial only!!!!
    "scenario": {
        "f00005": {
            "Duration": "R",
            "Inf_Soil": "I",
            "dtm": "D10",
            "start": "s40",
            "end": "e100",
            "model": "H00",
            "output": "o10",
            "outputsize": "ms10",
            "restart": "He",
        },
    },  # New scenario to use for manual restart
}
# ------------------------- other inputs, no need to change-----------------------------------
DISTRIBUTION_MACHINE_PATH = r"\\EUPRAAPPP134\01_Projects2\999_TUFLOW_Start\_start_yaml"
machines = {
    "EUPRAAPPP006": "01_Projects",
    "EUPRAAPPP008": "01_Projects",
    "EUPRAAPPP032": "01_Projects",
    "EUPRAAPPP033": "01_Projects",
    "EUPRAAPPP041": "01_Projects",
    "EUPRAAPPP051": "01_Projects3",
    "EUPRAAPPP062": "01_Projects2",
    "EUPRAAPPP098": "01_Projects",
    "EUPRAAPPP099": "01_Projects",
    "EUPRAAPPP100": "01_Projects",
    "EUPRAAPPP105": "01_Projects",
    "EUPRAAPPP119": "01_Projects",
    "EUPRAAPPP120": "01_Projects",
    "EUPRAAPPP121": "01_Projects",
    "EUPRAAPPP132": "01_Projects",
    "EUPRAAPPP133": "01_Projects",
    "EUPRAAPPP134": "01_Projects",
    "EUPRAAPPP135": "01_Projects",
    "EUPRAAPPP136": "01_Projects",
}
peril_path = {
    "fl": "fluvial",
    "pl": "pluvial",
    "pl_city": "pluvial",
}
machines_storage = {
    "p062_e": {"shared_path": r"\\EUPRAAPPP062\01_projects", "drive_letter": "e"},
    "p009": {"shared_path": r"\\EUPRAAPPP009\01_projects", "drive_letter": "d"},
    "p006_d": {"shared_path": r"\\EUPRAAPPP006\01_projects", "drive_letter": "d"},
    "p006_e": {"shared_path": r"\\EUPRAAPPP006\01_projects2", "drive_letter": "e"},
    "p134_e": {"shared_path": r"\\EUPRAAPPP134\01_projects2", "drive_letter": "e"},
}


class CopyRunStart:
    def __init__(self, yaml_file_name: str):
        self.yaml_file_name = yaml_file_name
        self.parse_yaml_file()
        exists_dict = self.check_existing_folder()
        if not self.check_tsf():
            raise Exception(f"{self.data['domain']}, No test run done!")
        else:
            self.data["test_run_done"] = True
        if sum(exists_dict.values()) > 1:
            raise Exception(
                f"Multiple machines have the same folder for domain {self.data['domain']}: {exists_dict}, please delete the folder on machines."
            )
        elif sum(exists_dict.values()) == 1 and MANUAL_RESTART["restarting"]:
            # return the one machine that has the folder
            for machine, exists in exists_dict.items():
                if exists:
                    self.data["machine"] = machine
            self.copy_domain()
        elif sum(exists_dict.values()) == 1 and not MANUAL_RESTART["restarting"]:
            raise Exception(
                f"Machine has the same folder for domain {self.data['domain']}: {exists_dict}, please delete the folder on the machine."
            )
        elif sum(exists_dict.values()) == 0 and MANUAL_RESTART["restarting"]:
            raise Exception(
                f"No machine has the folder for domain {self.data['domain']}: {exists_dict}, manual restart exepects it on some machine!"
            )
        else:
            self.copy_domain()

    def parse_yaml_file(self) -> dict:
        """
        Parse the YAML file and return its content as a dictionary.

        :return: Dictionary containing the parsed YAML data.
        """
        try:
            with open(self.yaml_file_name, "r") as file:
                self.data = yaml.safe_load(file)
        except Exception as e:
            print(f"Error reading YAML file {self.yaml_file_name}: {e}")
            self.data = {}

    def copy_domain(
        self,
    ):
        source_folder = self.data["data_source"]
        if os.path.exists(source_folder):
            data_destination = machines_storage[DATA_STORAGE]["shared_path"]
            if "region" in self.data:
                data_destination = os.path.join(
                    data_destination,
                    self.data["project"],
                    "01_MD",
                    "01_HAZARD",
                    self.data["region"],
                    (
                        "06_TU_Pluvial"
                        if self.data["peril_path"] == "06_TU_Pluvial_city_yaml"
                        else self.data["peril_path"]
                    ),
                    self.data["domain"].split("_")[0],
                    self.data["resolution"],
                    self.data["domain"],
                )
            else:
                data_destination = os.path.join(
                    data_destination,
                    self.data["project"],
                    "01_MD",
                    "01_HAZARD",
                    (
                        "06_TU_Pluvial"
                        if self.data["peril_path"] == "06_TU_Pluvial_city_yaml"
                        else self.data["peril_path"]
                    ),
                    self.data["domain"].split("_")[0],
                    self.data["resolution"],
                    self.data["domain"],
                )
            self.data["data_source"] = data_destination
            if not OVERWRITE_ON_STORAGE:
                if os.path.exists(data_destination):
                    print(
                        f"Destination folder {data_destination} already exists, reusing data"
                    )
                else:
                    shutil.copytree(
                        source_folder,
                        data_destination,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*log*", "*results*"),
                    )

            else:
                if os.path.exists(data_destination):
                    print(
                        f"Destination folder {data_destination} already exists, but OVERWRITE_ON_STORAGE is set to True. Overwriting whole domain."
                    )
                    shutil.rmtree(data_destination)
                shutil.copytree(
                    source_folder,
                    data_destination,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*log*", "*results*"),
                )

            self.create_yaml_on_machine()
        else:
            raise FileNotFoundError(f"Source folder {source_folder} does not exist.")

    def create_yaml_on_machine(
        self,
    ):
        """
        Create a YAML file on the specified machine with the provided data.
        """
        print(
            f"Creating YAML file on distribution machine...\n\t {self.data['domain']}"
        )
        if MANUAL_RESTART["restarting"]:
            path = "manual_restart"
            self.data["scenario"] = MANUAL_RESTART["scenario"]
            self.data["start_scenario"] = "manual_restart"
        else:
            path = peril_path[self.data["peril"]]
        ## use only high end cards
        if self.data["peril"] == "pl_city":
            yaml_path = os.path.join(
                DISTRIBUTION_MACHINE_PATH, self.data["project"] + "_HR", path
            )
        else:
            yaml_path = os.path.join(
                DISTRIBUTION_MACHINE_PATH, self.data["project"], path
            )
        os.makedirs(yaml_path, exist_ok=True)

        yaml_path = os.path.join(yaml_path, os.path.basename(self.yaml_file_name))

        self.data["manual_restart"] = MANUAL_RESTART["restarting"]

        ## zmena peril_patf in distribuion yaml
        tmp_peril_path = self.data["peril_path"]
        self.data["peril_path"] = (
            "06_TU_Pluvial"
            if self.data["peril_path"] == "06_TU_Pluvial_city_yaml"
            else self.data["peril_path"]
        )

        with open(yaml_path, "w") as file:
            yaml.dump(self.data, file, default_flow_style=False)

        ## zpet puvodni hodnota peril path
        self.data["peril_path"] = tmp_peril_path

        self._move_sent_file(self.yaml_file_name, "_copied_to_distribution")

        print(f"YAML file created at {yaml_path}")

    def check_tsf(self) -> bool:
        df_parameters = pd.DataFrame.from_dict(self.data["scenario"], orient="index")
        df_parameters.reset_index(inplace=True)
        df_parameters.rename(columns={"index": "RP"}, inplace=True)

        tsf_files = []
        # print(df_parameters)

        for _, row in df_parameters.iterrows():
            if row["RP"] in TEST_RPS[FLUVIAL_PLUVIAL]:
                log_folder = os.path.join(self.data["data_source"], "runs", "log")
                tsf_file = f"{domain}_{row['Inf_Soil']}+{row['Duration']}+{row['RP']}_{row['start']}+{row['end']}+{row['model']}+{row['dtm']}+{row['output']}+{row['outputsize']}+{row['restart']}+CPU.tsf"
                tsf_path = os.path.join(log_folder, tsf_file)
                if os.path.isfile(tsf_path):
                    tsf_files.append(tsf_path)

        if len(tsf_files) == len(TEST_RPS[FLUVIAL_PLUVIAL]):
            return True
        return False

    def check_existing_folder(self) -> bool:
        """
        Check if the folder for the given domain exists on the specified machine.

        :param data: Dictionary containing the domain, root folder, machine, and other parameters.
        :type data: dict
        :return: True if the folder exists, False otherwise.
        :rtype: bool
        """
        area = self.data["domain"].split("_")[0]
        exists = {}
        for machine in machines:
            in_folder_machine = os.path.join(
                "\\\\",
                machine,
                machines[machine],
                self.data["project"],
                "01_MD",
                "01_HAZARD",
                self.data["peril_path"],
                area,
                self.data["domain"],
            )
            if os.path.exists(in_folder_machine):
                exists[machine] = 1
            else:
                exists[machine] = 0
        # check if multiple machines have the same folder
        return exists

    @staticmethod
    def _move_sent_file(file, tag) -> None:
        """
        Copy the file to the sent_to_queue folder with the current date as a subfolder.
        :param file: File to be moved.
        :param tag: Tag to be added to the folder name.
        :return: None
        """
        current_date = time.strftime("%Y%m%d")
        out_folder = os.path.join(os.path.dirname(TUFLOW_YAML_PATH), tag, current_date)
        os.makedirs(out_folder, exist_ok=True)
        shutil.move(file, os.path.join(out_folder, os.path.basename(file)))
        # check if file dir is empty and remove it if so
        if not os.listdir(os.path.dirname(file)):
            os.rmdir(os.path.dirname(file))


if __name__ == "__main__":

    today = time.strftime("%Y%m%d_%H%M")
    issues = []
    domains_to_do = pd.read_csv(DOMAINS_TO_DO)["DomainID"].tolist()
    for domain in domains_to_do:

        area = domain.split("_")[0]
        yaml_folder = os.path.join(TUFLOW_YAML_PATH, area)
        yaml_file = os.path.join(yaml_folder, f"{domain}_{RESOLUTION}.yaml")
        if os.path.isfile(yaml_file):
            print(f"Processing domain: {domain}")
            try:
                copy_run_start = CopyRunStart(yaml_file)
            except Exception as e:
                issues.append(f"{domain},{e}")

    if issues:

        file_path = os.path.join(TUFLOW_YAML_PATH, f"issues_{today}.csv")
        print(f"Saving issues to {file_path}")
        with open(file_path, "w", newline="\n") as file:
            for issue in issues:
                file.write(issue + "\n")
    else:
        print("No issues encountered during processing.")
