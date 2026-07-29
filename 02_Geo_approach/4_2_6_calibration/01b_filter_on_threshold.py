"""Filters crest_points_2d.gpkg to the sel_030 selection and writes a crest
file for script 15 (column h_te_ortho, EPSG:2180). Prints counts."""

from pathlib import Path
import geopandas as gpd

IN_GPKG  = Path(__file__).parent / "diagnostics_ch4" / "crest_points_2d.gpkg"
OUT_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_crest_2d_th030.gpkg")

def main():
    g = gpd.read_file(IN_GPKG, engine="pyogrio")
    n_all = len(g)
    g = g[g["sel_030"] == 1].copy() if g["sel_030"].dtype != bool \
        else g[g["sel_030"]].copy()
    g = g.rename(columns={"h_ortho": "h_te_ortho"})
    g = g[["geometry", "h_te_ortho", "prominence2d", "n_ref"]]
    if g.crs is None:
        g = g.set_crs(epsg=2180)
    g.to_file(OUT_GPKG, driver="GPKG")
    print(f"candidates: {n_all} | selected at 0.3: {len(g)}")
    print(f"written: {OUT_GPKG}")

if __name__ == "__main__":
    main()