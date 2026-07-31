"""
Paired SFINCS models (baseline vs detected levees), steady RP100 discharge.
The two models share grid, elevation, roughness, mask and forcing; model B
adds the detected levees as weirs, so any flood difference is due to levees.

Author: Jakub Zapletal
Date:   2026-04-27
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
from hydromt_sfincs import SfincsModel

from toolset import sfincs

# ============================================================
# CONFIG
# ============================================================

# --- Inputs -------------------------------------------------
AOI_GPKG        = Path("data/domain.gpkg")            # model domain polygon
DSM_TIF         = Path("data/dsm_10m.tif")            # elevation, EGM2008, 10 m
WORLDCOVER_TIF  = Path("data/worldcover.tif")         # ESA WorldCover classes
DISCHARGE_GPKG  = Path("data/bc_upstream.gpkg")       # points: index, q_rp100[, name]
LEVEES_GPKG     = Path("data/levee_segments_z.gpkg")  # levee segments with crest z

# Downstream outflow polygon; None makes the whole active-domain edge outflow
OUTFLOW_GPKG    = Path("data/bc_downstream.gpkg")

# --- Output -------------------------------------------------
OUT_ROOT = Path("output/model_RP100")

# --- Grid ---------------------------------------------------
CRS_EPSG = 2180
RES_M    = 10.0

# --- Levee crest --------------------------------------------
Z_COLUMN   = "z"     # crest elevation column on the levee lines (EGM2008 m)
DZ_DEFAULT = 1.8     # fallback crest = DSM + DZ_DEFAULT where z is missing;
                     # 1.8 m = median dz over DSM from the ATL08 crest points
WEIR_CD    = 0.38    # weir discharge coefficient (par1), SFINCS default

# --- Rebuild control ----------------------------------------
SKIP_EXISTING = True  # leave an already built model untouched

# --- Steady-flow run ----------------------------------------
SIM_HOURS   = 70                  # constant-Q run length; check steadiness
TSTART      = "20260101 000000"   # sfincs.inp datetime format
OUTPUT_DT_S = 3600                # map output interval [s]

# --- Local SFINCS executable --------------------------------
EXE_PATH        = Path("sfincs/sfincs.exe")
RUN_AFTER_BUILD = False

# Manning reclass table, written next to this script on first run
RECLASS_CSV = Path(__file__).parent / "worldcover_manning.csv"

# ESA WorldCover class -> Manning n (identical for both models)
WORLDCOVER_MANNING = {
    10: 0.120,   # tree cover
    20: 0.050,   # shrubland
    30: 0.034,   # grassland
    40: 0.040,   # cropland
    50: 0.100,   # built-up
    60: 0.030,   # bare / sparse vegetation
    70: 0.030,   # snow and ice
    80: 0.030,   # permanent water bodies
    90: 0.050,   # herbaceous wetland
    95: 0.070,   # mangroves
    100: 0.035,  # moss and lichen
}


# ============================================================
# MODEL BUILD
# ============================================================


def build_model(root, with_levees, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv):
    """Build one SFINCS model; both variants share every call except the
    weirs block, so the levees are the only difference."""
    label = "WITH levees" if with_levees else "baseline"

    if SKIP_EXISTING and sfincs.model_is_built(root):
        print(f"\n=== skipping {label} (already built) -> {root}")
        print("    set SKIP_EXISTING = False to rebuild from scratch")
        return None

    print(f"\n=== building {label} -> {root}")
    root.mkdir(parents=True, exist_ok=True)

    sf = SfincsModel(root=str(root), mode="w", write_gis=True)

    # 1. grid over the AOI (regular, no rotation, 10 m)
    sf.grid.create_from_region(
        region={"geom": str(AOI_GPKG)},
        res=RES_M,
        crs=CRS_EPSG,
        rotated=False,
        align=True,
    )

    # 2. run control first: discharge create() clips series to the model time
    t0 = pd.to_datetime(TSTART, format="%Y%m%d %H%M%S")
    t1 = t0 + pd.Timedelta(hours=SIM_HOURS)
    fmt = "%Y%m%d %H%M%S"
    sf.config.update({
        "tref":     t0.strftime(fmt),
        "tstart":   t0.strftime(fmt),
        "tstop":    t1.strftime(fmt),
        "dtout":    OUTPUT_DT_S,
        "dtmaxout": SIM_HOURS * 3600,   # zsmax over the whole run
        "alpha":    0.5,
    })

    # 3. elevation
    sf.elevation.create(elevation_list=[{"elevation": str(DSM_TIF)}])

    # 4. active mask + outflow boundary
    sf.mask.create_active(include_polygon=str(AOI_GPKG), all_touched=True)
    if OUTFLOW_GPKG is not None:
        sf.mask.create_boundary(btype="outflow",
                                include_polygon=str(OUTFLOW_GPKG),
                                all_touched=True)
    else:
        # no include_polygon -> every edge cell of the active domain
        sf.mask.create_boundary(btype="outflow")

    # 5. roughness from ESA WorldCover (fallback n where lulc has nodata)
    sf.roughness.create(
        roughness_list=[{"lulc": str(WORLDCOVER_TIF),
                         "reclass_table": str(reclass_csv)}],
        manning_land=0.04,
        manning_sea=0.04,   # irrelevant for a fluvial reach; keep equal to land
    )

    # 6. constant RP100 discharge (steady flow)
    sf.discharge_points.create(locations=dis_gdf, timeseries=dis_ts)

    # 7. the only difference between the two models
    if with_levees:
        if len(levees_z) > 0:
            print(f"  weirs from crest z: {len(levees_z)} segments")
            sf.weirs.create(locations=levees_z, par1=WEIR_CD)
        if len(levees_noz) > 0:
            print(f"  weirs from DSM+{DZ_DEFAULT} m: {len(levees_noz)} segments")
            sf.weirs.create(locations=levees_noz, dep=str(DSM_TIF),
                            dz=DZ_DEFAULT, par1=WEIR_CD)
        if len(levees_z) == 0 and len(levees_noz) == 0:
            raise ValueError("No levee segments to schematize as weirs")

    # 8. write everything (sfincs.inp last, incl. component files)
    sf.write()
    if with_levees:
        if len(levees_noz) > 0:
            print(f"  WARNING: {len(levees_noz)} segments without z are not in "
                  f"the manual weirfile")
        sfincs.force_weirfile_in_inp(root, levees_z, WEIR_CD)
    sfincs.write_run_bat(root, EXE_PATH)
    print(f"  written: {root}")
    return sf


# ============================================================
# MAIN
# ============================================================


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Inputs:")
    reclass_csv = sfincs.write_reclass_table(RECLASS_CSV, WORLDCOVER_MANNING)
    dis_gdf = sfincs.load_discharge_points(DISCHARGE_GPKG, CRS_EPSG)
    dis_ts = sfincs.constant_timeseries(dis_gdf, TSTART, SIM_HOURS)
    levees_z, levees_noz = sfincs.prepare_levees(
        LEVEES_GPKG, AOI_GPKG, CRS_EPSG, Z_COLUMN, DZ_DEFAULT
    )

    roots = [OUT_ROOT / "sfincs_baseline", OUT_ROOT / "sfincs_levees"]
    build_model(roots[0], False, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv)
    build_model(roots[1], True, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv)

    if RUN_AFTER_BUILD:
        print("\nRunning models locally...")
        for r in roots:
            if SKIP_EXISTING and (r / "sfincs_map.nc").exists():
                print(f"  skipping run {r.name} (sfincs_map.nc exists)")
                continue
            sfincs.run_model(r, EXE_PATH)
    else:
        print("\nBoth models built. Run them via run.bat in each folder, or:")
        for r in roots:
            print(f'  pushd "{r}" && "{EXE_PATH}" & popd')
        print("Or set RUN_AFTER_BUILD = True to launch both from this script.")

    print("\nSteadiness check: compare zsmax at ~0.75*T and T; if they differ,")
    print("extend SIM_HOURS. Compare runs via zsmax difference (levees - baseline).")


if __name__ == "__main__":
    main()
