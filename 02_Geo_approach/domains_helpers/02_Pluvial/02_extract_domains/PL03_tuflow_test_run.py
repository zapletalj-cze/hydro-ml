import os
import yaml
import pandas as pd
import subprocess
import multiprocessing as mp

DOMAINS_TO_DO = r"R:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\_dtd_PL\N05_pluvial_reference.csv"



RESOLUTION = "10m"  # Resolution of the domains, used to find the yaml files
IN_FOLDER = r"D:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL"
YAML_FOLDER = r"D:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\_start_yaml\173_Nordics_flood\yaml"
TUFLOW_EXE = r"c:\TUFLOW\Releases\2026-0-1\TUFLOW_iSP_w64.exe"
FLUVIAL_PLUVIAL = "pl_city"

df = pd.read_csv(DOMAINS_TO_DO)
domains = df.DomainID.tolist()

TEST_RPS = {
    "pl_city": [
        "r00005",
        # "r00020",
        # "r00050",
        # "r00100",
        # "r00200",
        # "r00500",
        # "r01000",
        # "r10000",
    ],
}  # 5,


def start_domain(yaml_file):
    print(f"Starting with: {yaml_file}")
    with open(yaml_file, "r") as file:
        body = yaml.safe_load(file)
        domain_dict = body
        domain = domain_dict["domain"]
        root_path = domain_dict["data_source"]
        scenario = domain_dict["scenario"]

        ##msa
        print(f"  ..start {domain}")

        # create "start_bat" from the scenario which is defined in the yaml
        df_parameters = pd.DataFrame.from_dict(scenario, orient="index")
        df_parameters.reset_index(inplace=True)
        df_parameters.rename(columns={"index": "RP"}, inplace=True)

        return_codes = []

        for _, row in df_parameters.iterrows():
            path_to_run = os.path.join(root_path, "runs")
            tsf_file = f"{domain}_{row['Inf_Soil']}+{row['Duration']}+{row['RP']}_{row['start']}+{row['end']}+{row['model']}+{row['dtm']}+{row['output']}+{row['outputsize']}+{row['restart']}+CPU.tsf"
            if (
                not os.path.isfile(os.path.join(path_to_run, "log", tsf_file))
                and row["RP"] in TEST_RPS[FLUVIAL_PLUVIAL]
            ):
                print(row)
                try:

                    tcf_file = domain + ".tcf"
                    os.chdir(path_to_run)
                    ## msa zmenil jsem pocet jader z 2 na 20 -.80%, uvidim co 24 (-nt24) vyuziva cca 95% na p121 pro rychlejsi testovani.26 bylo moc
                    cmd = (
                        f"{TUFLOW_EXE} -b -acf -t -nlc -nt10 "
                        f"-s1 {row['start']} "
                        f"-s2 {row['end']} "
                        f"-s3 {row['model']} "
                        f"-s4 {row['dtm']} "
                        f"-s5 {row['output']} "
                        f"-s6 {row['outputsize']} "
                        f"-s7 {row['restart']} "
                        f"-s8 CPU "
                        f"-e1 {row['Inf_Soil']} "
                        f"-e2 {row['Duration']} "
                        f"-e3 {row['RP']} "
                        f"{tcf_file} "
                        f"> {domain}_{row['RP']}_test_run.log"
                    )

                    ## start of TUFLOW
                    p = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )

                    retval = p.wait()
                    return_codes.append(retval)
                    if os.path.exists(os.path.join(path_to_run, "log", tsf_file)):
                        print(f"  .. {domain},rp {row['RP']}; tsf is ok")
                    else:
                        print(
                            f"  .. {domain},rp {row['RP']}; tsf is MISSING!!!; {retval}"
                        )

                except Exception as e:
                    print(f"Error running TUFLOW for {domain} with RP {row['RP']}: {e}")
                    return_codes.append(-1)
            else:
                if row["RP"] in TEST_RPS[FLUVIAL_PLUVIAL]:
                    print(f"  .. {domain},rp {row['RP']}; tsf already exists, skipping")


if __name__ == "__main__":
    domains_to_check = []
    for domain in domains:
        area = domain.split("_")[0]
        yaml_folder = os.path.join(YAML_FOLDER, area)
        yaml_file = os.path.join(yaml_folder, f"{domain}_{RESOLUTION}.yaml")
        if os.path.isfile(yaml_file):
            domains_to_check.append(yaml_file)

    pool = mp.Pool(processes=5)
    for i, _ in enumerate(pool.imap(start_domain, domains_to_check)):
        print(f"Domain {i + 1}/{len(domains_to_check)} processed")
    pool.close()
    pool.join()
