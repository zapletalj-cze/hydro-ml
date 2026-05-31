import os
import geopandas as gpd
import pandas as pd
from pandas.io.formats import excel
import glob

GRID = r"R:\01_Projects\900_FL_Europe\01_MD\01_HAZARD\06_TUFLOW_PL\05_domains\grid\EUROPE_GRID_3035_8km_ISO3.gpkg" # expected in European dir
OUT = r"R:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\_reference"
SELECTED_DOMAINS = r"R:\01_Projects\173_Nordics_flood\01_MD\01_HAZARD\06_TUFLOW_PL\_dtd_PL\N05_pluvial_reference.csv"
CITY_NAME = "N05"
CELL_SIZE_DTM_DEFAULT = 1


ALL_COLS = [
    "Domain",
    "DTM_Resolution1",
    "DTM_Resolution2",
    "watershed",
    "DTM_city_Resolution",
    "Mesh_city_Resolution",
    "City",
]
PR_COLS = [
    "Domain",
    "DTM_Resolution1",
    "DTM_Resolution2",
    "City",
    "DTM_city_Resolution",
    "Mesh_city_Resolution",
]


def write_excel(filename, sheetname, dataframe):
    excel.ExcelFormatter.header_style = None
    if not os.path.isfile(filename):
        dataframe[ALL_COLS].to_excel(filename, sheet_name=str(sheetname), index=False)
    else:
        try:
            df = pd.read_excel(filename, sheet_name=str(sheetname))
            cols_exuist = [c for c in PR_COLS if c in df.columns]
            df = df[cols_exuist]
            # merge only watershed (no overlapping columns) then join preserved cols
            base = dataframe[["Domain", "watershed"]]
            merged = pd.merge(base, df, on="Domain", how="left")
            defaults = dataframe.set_index("Domain")
            for col in [
                "DTM_Resolution1",
                "DTM_Resolution2",
                "DTM_city_Resolution",
                "Mesh_city_Resolution",
                "City",
            ]:
                if col not in merged.columns:
                    merged[col] = dataframe[col].values
                else:
                    mask = merged[col].isna()
                    merged.loc[mask, col] = merged.loc[mask, "Domain"].map(
                        defaults[col]
                    )
            dataframe = merged[ALL_COLS]
        except Exception as e:
            dataframe = dataframe[ALL_COLS]
        with pd.ExcelWriter(
            filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            dataframe.to_excel(writer, sheet_name=str(sheetname), index=False)


def prepare_grid(grid):
    selected = pd.read_csv(SELECTED_DOMAINS)["DomainID"].tolist()
    df_grid = gpd.read_file(grid)
    df_grid["WS"] = ""
    df_grid = df_grid[["DomainID", "WS", "Area"]]
    df_grid.rename(columns={"DomainID": "Domain"}, inplace=True)
    df_grid = df_grid[df_grid["Domain"].isin(selected)]
    df_grid["DTM_Resolution1"] = CELL_SIZE_DTM_DEFAULT
    df_grid["DTM_Resolution2"] = ""
    df_grid["DTM_city_Resolution"] = CELL_SIZE_DTM_DEFAULT
    df_grid["Mesh_city_Resolution"] = 10
    df_grid["City"] = CITY_NAME

    groups = df_grid.groupby("Area")
    for name, group in groups:
        group["watershed"] = group["WS"]
        group = group[
            [
                "Domain",
                "DTM_Resolution1",
                "DTM_Resolution2",
                "watershed",
                "DTM_city_Resolution",
                "Mesh_city_Resolution",
                "City",
            ]
        ]
        write_excel(
            os.path.join(OUT, f"{name[0]}_EUFL_Tuflow_domains.xlsx"), name, group
        )


if __name__ == "__main__":
    prepare_grid(GRID)
