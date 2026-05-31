import yaml
import os
import geopandas as gpd
import pandas as pd
import multiprocessing as mp
import sqlalchemy
from sqlalchemy.orm import sessionmaker, declarative_base
from shapely.geometry import Point, LineString, MultiLineString, MultiPolygon
from shapely.ops import linemerge
import sys

DOMAINS_TO_DO = r"R:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\_dtd_PL\N05_pluvial_reference.csv"  # reference created using 01_Extract_FLPL
OUT_FOLDER_DOMAINS = r"D:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\\"  # dir on your local machine where the domains will be created
PL_RESOLUTION = 10  # do not change

with open(
    os.path.join(os.path.dirname(__file__), "yaml_1_grid_creation_parameters.yaml"), "r"
) as stream:
    try:
        parameters_grid = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise exc


rf_fields = {"f1": 1.0, "f2": 1.0}
flags = {
    "Type": "HT",
    "Flags": "",
    "Name": "HT0",
    "f": 1.0,
    "d": 0,
    "td": 0,
    "a": 0,
    "b": 0,
}

df_idf_grid = gpd.read_file(parameters_grid["idf_grid"])  ## finish later

df_idf_grid.to_crs(parameters_grid["epsg"], inplace=True)
df_idf_grid["INDEX_RC"] = df_idf_grid["IDFID"]


def split_files(item):
    """
    Method which will split fishnet into polygons
    :param item: One row of GeoDataFrame
    """
    # df_watersheds = gpd.read_file(vector)
    # for index, item in df_watersheds.iterrows():
    domain = str(item["DomainID"])
    pl_resolution = (
        str(PL_RESOLUTION) + "m"
        if not isinstance(PL_RESOLUTION, str)
        else PL_RESOLUTION
    )
    out_folder = os.path.join(
        OUT_FOLDER_DOMAINS, pl_resolution, domain[:3], domain, "model", "gis"
    )
    out_clip = os.path.join(out_folder, f"2d_clip_{domain}_R.gpkg")
    out_code = os.path.join(out_folder, f"2d_code_{domain}_R.gpkg")
    out_rf = os.path.join(out_folder, f"2d_rf_{domain}_R.gpkg")
    bc_out_file = os.path.join(out_folder, f"2d_bc_OUT_{domain}_L.gpkg")
    if (
        not os.path.isfile(out_clip)
        or not os.path.isfile(out_code)
        or not os.path.isfile(out_rf)
    ):
        os.makedirs(out_folder, exist_ok=True)
        df_one_watershed = gpd.GeoDataFrame(item).transpose()
        df_one_watershed.set_geometry("geometry", inplace=True)
        df_one_watershed.crs = df_idf_grid.crs
        df_rf = df_one_watershed.copy()
        # buffer rf file
        df_one_watershed["geometry"] = df_one_watershed["geometry"].buffer(
            parameters_grid["clip_size"], cap_style=2, join_style=2
        )

        df_one_watershed.to_file(out_clip)
        df_2d_code = df_one_watershed.copy()
        # buffer rf to make code
        df_2d_code["geometry"] = df_2d_code["geometry"].buffer(
            parameters_grid["code_size"], cap_style=2, join_style=2
        )

        # add required fields to code
        df_2d_code["Code"] = 1
        df_2d_code["DomainName"] = domain
        df_2d_code = df_2d_code[["Code", "DomainName", "geometry"]]
        df_2d_code.to_file(out_code)

        # extract IDF grid to rf file
        df_2d_rf = gpd.overlay(df_idf_grid, df_rf, how="intersection")
        for field in rf_fields:
            df_2d_rf[field] = rf_fields[field]

        df_2d_rf["ID_RF"] = df_2d_rf[parameters_grid["idf_grid_column"]]
        df_2d_rf = df_2d_rf[["ID_RF", "f1", "f2", "INDEX_RC", "geometry"]]
        df_2d_rf.to_file(out_rf)

    # create boundary condition for the domain
    extract_domain(domain)

    if (
        os.path.isfile(out_code)
        and os.path.isfile(out_clip)
        and os.path.isfile(out_rf)
        and os.path.isfile(bc_out_file)
    ):
        return domain


def extract_domain(domain: str):
    """
    Method which will extract boundary condition based on river network and DTM
    :param domain: Domain ID as str
    :return:
    """
    domain_folder = parameters_grid["out_folder_domains"]
    pl_resolution = (
        str(PL_RESOLUTION) + "m"
        if not isinstance(PL_RESOLUTION, str)
        else PL_RESOLUTION
    )
    gis_folder = os.path.join(
        domain_folder, pl_resolution, domain[:3], domain, "model", "gis"
    )
    code_file = os.path.join(gis_folder, f"2d_code_{domain}_R.gpkg")
    bc_out_file = os.path.join(gis_folder, f"2d_bc_OUT_{domain}_L.gpkg")
    if not os.path.isfile(bc_out_file):
        df_code = gpd.read_file(code_file)
        # df_rn = gpd.read_file(rn, bbox=df_code)

        df_code.geometry = df_code.geometry.boundary

        # intersect code with rn and on the points create bc_out with a buffer
        try:
            segments = []
            bc_out_length = parameters_grid["bc_out_size"]
            for line in df_code.geometry.tolist():
                if isinstance(line, MultiLineString):
                    for l in line.geoms:
                        if l.length < bc_out_length * 10:
                            segments.append(l)
                        else:
                            segmentized = l.segmentize(bc_out_length)
                            for i, coord in enumerate(segmentized.coords):
                                if i == 0:
                                    prev_point = coord
                                else:
                                    split_line = LineString(
                                        [
                                            Point(prev_point[0], prev_point[1]),
                                            Point(coord[0], coord[1]),
                                        ]
                                    )
                                    prev_point = coord
                                    segments.append(split_line)

                else:
                    segmentized = line.segmentize(bc_out_length)

                    for i, coord in enumerate(segmentized.coords):
                        if i == 0:
                            prev_point = coord
                        else:
                            split_line = LineString(
                                [
                                    Point(prev_point[0], prev_point[1]),
                                    Point(coord[0], coord[1]),
                                ]
                            )
                            prev_point = coord
                            segments.append(split_line)
            multiplier = 1
            while len(segments) > 253:
                for index, line_split in enumerate(segments):
                    if (
                        line_split.length < bc_out_length * multiplier
                        and not isinstance(line_split, MultiLineString)
                    ):
                        # if line is too short, merge it with next from list and replace it
                        if index + 1 < len(segments) - 1:
                            if not isinstance(segments[index + 1], MultiLineString):
                                line = linemerge([line_split, segments[index + 1]])
                                segments[index + 1] = line
                                segments.pop(index)
                multiplier += 0.5
            df_bc_out = gpd.GeoDataFrame(geometry=segments, crs=df_code.crs)

            for col in flags:
                df_bc_out[col] = flags[col]
            df_bc_out.to_file(bc_out_file)

        except Exception as e:
            print(e)
            return gpd.GeoDataFrame


def create_structure_with_files(grid: str):
    df_watersheds = gpd.read_file(grid)

    df = pd.read_csv(DOMAINS_TO_DO)
    domains_to_do = df.DomainID.values.tolist()
    df_watersheds = df_watersheds[df_watersheds["DomainID"].isin(domains_to_do)]
    done = []
    grids = [item for index, item in df_watersheds.iterrows()]

    if not parameters_grid["multicore"]:
        # --------single core---------------
        for i, item in enumerate(grids):
            done.append(split_files(item))
            sys.stderr.write(
                "\r\tDomains created and copied {0:%}".format(i / len(grids))
            )
    else:
        # --------multi core---------------
        with mp.Pool(int(mp.cpu_count() * 0.5)) as pool:
            for i, domain in enumerate(pool.imap(split_files, grids)):
                sys.stderr.write(
                    "\r\tDomains created and copied {0:%}".format(i / len(grids))
                )
                done.append(domain)
            pool.close()
            pool.join()


if __name__ == "__main__":
    create_structure_with_files(parameters_grid["out_grid"])
