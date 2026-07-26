"""
Paired SFINCS models: baseline vs detected levees (fluvial, steady flow, RP100)
===============================================================================

Builds TWO HydroMT-SFINCS models that are identical in every respect (grid,
elevation, roughness, mask, boundaries, discharge) and differ in exactly one
thing: model B contains the detected levees as weir structures. Any difference
in the simulated flood is therefore attributable to the levees alone.

    <OUT_ROOT>/sfincs_baseline      no levees
    <OUT_ROOT>/sfincs_levees        detected levees as weirs (crest z, overtopping)

Design:
    - grid:        EPSG:2180, regular, 10 m resolution (cell = DSM pixel)
    - elevation:   Copernicus GLO-30 resampled DSM, 10 m, EGM2008 orthometric
    - roughness:   ESA WorldCover 10 m via reclass table (written next to script)
    - forcing:     point discharges, constant Q = RP100 (steady state reached by
                   running a constant hydrograph for SIM_HOURS)
    - boundary:    downstream outflow cells in the mask (mask=3)
    - structures:  detected levee lines as weirs; crest from column Z_COLUMN,
                   fallback: crest = DSM + DZ_DEFAULT where z is missing

Discharge input format (prepared by the user):
    GPKG in EPSG:2180, point layer with columns:
        index    unique integer id
        q_rp100  discharge [m3/s]
        name     optional
    Points must lie on the river centreline a few cells inside the active mask.

Environment (current stack, verified against hydromt_sfincs v2.0.0 docs):
    pip install "hydromt_sfincs>=2.0.0"      # pulls hydromt v1 (component API)
    SFINCS kernel: run locally via the Windows executable (no Docker).
        Download the SFINCS release (sfincs.exe + its DLLs) from Deltares and
        set EXE_PATH below. The kernel reads sfincs.inp from its working
        directory, so each model is run with cwd = its own folder. Windows
        resolves the DLLs from the executable's own directory, so the exe can
        stay in its release folder.

v2.0.0 API notes (component architecture replaces the old setup_* methods):
    setup_grid_from_region   -> sf.grid.create_from_region(region=...)
    setup_dep                -> sf.elevation.create(elevation_list=[{"elevation": ...}])
    setup_mask_active        -> sf.mask.create_active(include_polygon=...)
    setup_mask_bounds        -> sf.mask.create_boundary(btype="outflow", ...)
    setup_manning_roughness  -> sf.roughness.create(roughness_list=[{"lulc": ..., "reclass_table": ...}])
    setup_discharge_forcing  -> sf.discharge_points.create(locations=..., timeseries=...)
    setup_structures(weir)   -> sf.weirs.create(locations=..., dep=..., dz=...)
    setup_config             -> sf.config.update({...})   # Pydantic-validated dict

Two ordering facts baked into this script (from the v2 source):
    1) discharge_points.create() clips the timeseries to the model time window,
       so config tstart/tstop MUST be set before the discharge call.
    2) the roughness reclass CSV is read with index_col=0 and the Manning values
       must sit in a column literally named "N" (lulc class is the index).

Note: hydromt uses rasterio internally; this script itself does not import
rasterio or fiona.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from hydromt_sfincs import SfincsModel

# ============================================================
# CONFIG
# ============================================================

# --- Inputs -------------------------------------------------
AOI_GPKG        = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\domain\domain.gpkg")            # model domain polygon, EPSG:2180
DSM_TIF         = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\dtm\COP_DSM_10m_Wistula.tif")      # elevation, EGM2008, 10 m
WORLDCOVER_TIF  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\landuse\ESA_WorldCover_2021_2180_c.tif")   # ESA WorldCover classes
DISCHARGE_GPKG  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\bc_upstream\bc_upstream.gpkg")  # see format above
LEVEES_GPKG     = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\levees\levee_segments_z.gpkg")  # levee segments with z (script 15)

# Optional: polygon marking the downstream edge where water may leave the
# domain (outflow). If None, the WHOLE edge of the active domain becomes
# outflow (confirmed behaviour of mask.create_boundary without include_polygon),
# which is acceptable for a valley reach but less controlled.
OUTFLOW_GPKG    = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\bc_downstream\2d_bcdownstream.gpkg")

# --- Output -------------------------------------------------
OUT_ROOT = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100")

# --- Grid ---------------------------------------------------
CRS_EPSG = 2180
RES_M    = 10.0

# --- Levee crest --------------------------------------------
Z_COLUMN   = "z"     # crest elevation column on the levee lines (EGM2008 metres)
DZ_DEFAULT = 0.9     # fallback: crest = DSM + DZ_DEFAULT where z is missing.
                     # 0.9 m = median dz over DSM measured from the ATL08 crest
                     # points after removing structures and vegetation, NOT an
                     # arbitrary value. Normally unused: script 15 assigns z to
                     # every segment, so the fallback path stays empty.
WEIR_CD    = 0.6     # weir discharge coefficient (par1), SFINCS default

# --- Rebuild control ----------------------------------------
# True: a model whose sfincs.inp + sfincs.dep already exist is left untouched,
# so a failed second model can be retried without rebuilding the first.
SKIP_EXISTING = True

# --- Steady-flow run ----------------------------------------
SIM_HOURS   = 66                  # constant-Q run length; check steadiness
TSTART      = "20260101 000000"   # sfincs.inp datetime format
OUTPUT_DT_S = 3600                # map output interval [s]

# --- Local SFINCS executable (no Docker) --------------------
# Full path to the SFINCS Windows kernel. A run.bat is written into each model
# folder regardless; set RUN_AFTER_BUILD=True to also launch both runs here.
EXE_PATH        = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS\SFINCS_2026_01_release\SFINCS_v2.4.0_Galibier_release_exe\sfincs.exe")
RUN_AFTER_BUILD = False

# Manning reclass table is written next to this script on first run.
# IMPORTANT: hydromt reads it with index_col=0 and requires a column named "N".
RECLASS_CSV = Path(__file__).parent / "worldcover_manning.csv"

# ESA WorldCover class -> Manning n (standard literature values; keep IDENTICAL
# between both models so roughness never enters the comparison)
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
# HELPERS
# ============================================================

def write_reclass_table(path):
    """WorldCover -> Manning reclass table in the exact layout hydromt expects:
    lulc class as the first (index) column, values in a column named 'N'."""
    df = pd.DataFrame(
        {"lulc": list(WORLDCOVER_MANNING.keys()),
         "N": [WORLDCOVER_MANNING[k] for k in WORLDCOVER_MANNING]}
    )
    df.to_csv(path, index=False)
    print(f"  reclass table: {path}")
    return path


def load_discharge_points(path):
    """Load and validate the user-prepared RP100 discharge points."""
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        gdf.set_crs(epsg=CRS_EPSG, inplace=True)
    elif gdf.crs.to_epsg() != CRS_EPSG:
        gdf = gdf.to_crs(epsg=CRS_EPSG)
    for col in ("index", "q_rp100"):
        if col not in gdf.columns:
            raise ValueError(f"Discharge GPKG must contain column '{col}'")
    gdf["index"] = gdf["index"].astype(int)
    if gdf["index"].duplicated().any():
        raise ValueError("Discharge point 'index' values must be unique")
    print(f"  discharge points: {len(gdf)} "
          f"(Q total {gdf['q_rp100'].sum():.0f} m3/s)")
    return gdf


def constant_timeseries(gdf, tstart, hours):
    """Constant RP100 hydrograph per point over the simulation window.
    Columns are the integer point ids (v2 coerces timeseries columns to int)."""
    t0 = pd.to_datetime(tstart, format="%Y%m%d %H%M%S")
    times = pd.date_range(t0, t0 + pd.Timedelta(hours=hours), freq="1h")
    data = {int(idx): np.full(len(times), float(q))
            for idx, q in zip(gdf["index"], gdf["q_rp100"])}
    df = pd.DataFrame(data, index=times)
    df.index.name = "time"
    return df


def prepare_levees(path):
    """Load detected levees, clip to the model domain, split by crest-z availability."""
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        gdf.set_crs(epsg=CRS_EPSG, inplace=True)
    elif gdf.crs.to_epsg() != CRS_EPSG:
        gdf = gdf.to_crs(epsg=CRS_EPSG)

    # Clip to the model domain FIRST. Segments outside it pass the len() guard
    # but vanish inside hydromt's own masking, which then raises
    # "GeoDataFrame has no data after masking". Clipping also avoids handing
    # the builder thousands of segments the model will never see.
    n_before = len(gdf)
    domain = gpd.read_file(AOI_GPKG, engine="pyogrio")
    if domain.crs is None:
        domain.set_crs(epsg=CRS_EPSG, inplace=True)
    elif domain.crs.to_epsg() != CRS_EPSG:
        domain = domain.to_crs(epsg=CRS_EPSG)
    gdf = gpd.clip(gdf, domain)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.length > 0].reset_index(drop=True)
    print(f"  levees clipped to domain: {len(gdf)} of {n_before} segments "
          f"({gdf.geometry.length.sum() / 1000:.1f} km)")
    if len(gdf) == 0:
        raise ValueError(
            "No levee segments inside the model domain - check that the levee "
            "file matches the modelled river (Wisla vs Odra)")

    if Z_COLUMN in gdf.columns:
        if Z_COLUMN != "z":
            gdf = gdf.rename(columns={Z_COLUMN: "z"})
        has_z = gdf["z"].notna()
    else:
        gdf["z"] = np.nan
        has_z = pd.Series(False, index=gdf.index)

    gdf_z = gdf[has_z].copy()
    gdf_noz = gdf[~has_z].drop(columns=["z"]).copy()
    print(f"  levees: {len(gdf)} lines | crest z present: {len(gdf_z)} "
          f"({100 * len(gdf_z) / max(len(gdf), 1):.0f} %) | "
          f"fallback DSM+{DZ_DEFAULT} m: {len(gdf_noz)}")
    return gdf_z, gdf_noz


# ============================================================
# LOCAL RUN (no Docker)
# ============================================================

def write_run_bat(root, exe_path):
    """Write a run.bat that runs the kernel from inside the model folder.
    %~dp0 makes it work regardless of where it is launched from."""
    bat = root / "run.bat"
    bat.write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        f"\"{exe_path}\" > sfincs_log.txt 2>&1\r\n",
        encoding="ascii",
    )
    print(f"  run script: {bat}")
    return bat


def run_model(root, exe_path):
    """Run one model synchronously with cwd = the model folder (so the kernel
    finds sfincs.inp and writes outputs there)."""
    import subprocess
    if not Path(exe_path).exists():
        raise FileNotFoundError(f"SFINCS executable not found: {exe_path}")
    print(f"  running {root.name} ...")
    with open(root / "sfincs_log.txt", "w") as log:
        subprocess.run([str(exe_path)], cwd=str(root),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"  finished {root.name} (see sfincs_log.txt)")


# ============================================================
# MODEL BUILD
# ============================================================

def model_is_built(root):
    """A model counts as built when sfincs.inp and the elevation grid exist."""
    return (root / "sfincs.inp").exists() and (root / "sfincs.dep").exists()


def write_weirfile_manual(root, levees_z, filename="sfincs.weir"):
    """Last-resort writer: SFINCS structure (tekal) format, one block per
    segment, columns x y z_crest par1(Cd). Used only when hydromt wrote no
    weir file at all."""
    path = root / filename
    n_struct = 0
    with open(path, "w") as f:
        for i, row in levees_z.reset_index(drop=True).iterrows():
            geom = row.geometry
            z = float(row["z"])
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            f.write(f"weir{i:05d}\n")
            f.write(f"{len(coords)} 4\n")
            for x, y in coords:
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {WEIR_CD:.2f}\n")
            n_struct += 1
    print(f"  manually wrote {n_struct} weir structures -> {path.name}")
    return filename


def verify_weirs_in_inp(root, levees_z):
    """Authoritative weirfile: ALWAYS rewrite sfincs.weir manually from the
    segment z values and force the inp reference. hydromt v2 weirs.create was
    observed to write z=0 for every vertex (ignoring the z column), which the
    kernel prunes as below-bed -> '0 structure u/v points found'."""
    import re
    inp_path = root / "sfincs.inp"
    inp = inp_path.read_text()

    weir_name = write_weirfile_manual(root, levees_z)

    zs = levees_z["z"].astype(float)
    print(f"  weir crest z: min {zs.min():.2f} m, median {zs.median():.2f} m, "
          f"max {zs.max():.2f} m (must be terrain-level, NOT 0)")

    if re.search(r"^\s*weirfile\s*=", inp, flags=re.M):
        inp = re.sub(r"^\s*weirfile\s*=\s*\S+", f"weirfile           = {weir_name}",
                     inp, flags=re.M)
    else:
        inp = inp.rstrip() + f"\nweirfile           = {weir_name}\n"
    inp_path.write_text(inp)
    print(f"  weirfile forced to manual '{weir_name}' in sfincs.inp")


def build_model(root, with_levees, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv):
    """Build one SFINCS model with the v2 component API. Both variants share
    every call except the weirs block, so the levees are the only difference."""
    label = "WITH levees" if with_levees else "baseline"

    if SKIP_EXISTING and model_is_built(root):
        print(f"\n=== skipping {label} (already built) -> {root}")
        print("    set SKIP_EXISTING = False to rebuild from scratch")
        return None

    print(f"\n=== building {label} -> {root}")
    root.mkdir(parents=True, exist_ok=True)

    sf = SfincsModel(root=str(root), mode="w", write_gis=True)

    # --- 1. grid over the AOI (regular, no rotation, 10 m, EPSG:2180) ---
    sf.grid.create_from_region(
        region={"geom": str(AOI_GPKG)},
        res=RES_M,
        crs=CRS_EPSG,
        rotated=False,
        align=True,
    )

    # --- 2. run control FIRST: discharge create() clips series to model time ---
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

    # --- 3. elevation (key renamed to "elevation" in v2) ---
    sf.elevation.create(elevation_list=[{"elevation": str(DSM_TIF)}])

    # --- 4. active mask + outflow boundary ---
    sf.mask.create_active(include_polygon=str(AOI_GPKG), all_touched=True)
    if OUTFLOW_GPKG is not None:
        sf.mask.create_boundary(btype="outflow",
                                include_polygon=str(OUTFLOW_GPKG),
                                all_touched=True)
    else:
        # no include_polygon -> every edge cell of the active domain (verified
        # in the v2 mask source); acceptable for a valley reach
        sf.mask.create_boundary(btype="outflow")

    # --- 5. roughness from ESA WorldCover (fallback n where lulc has nodata) ---
    sf.roughness.create(
        roughness_list=[{"lulc": str(WORLDCOVER_TIF),
                         "reclass_table": str(reclass_csv)}],
        manning_land=0.04,
        manning_sea=0.04,   # irrelevant for a fluvial reach; keep equal to land
    )

    # --- 6. constant RP100 discharge (steady flow) ---
    sf.discharge_points.create(locations=dis_gdf, timeseries=dis_ts)

    # --- 7. the ONLY difference between the two models ---
    if with_levees:
        if len(levees_z) > 0:
            # crest taken from the 'z' column (EGM2008, from ATL08 workflow)
            print(f"  weirs from crest z: {len(levees_z)} segments")
            sf.weirs.create(locations=levees_z, par1=WEIR_CD)
        if len(levees_noz) > 0:
            # crest sampled from the DSM along the line, raised by DZ_DEFAULT
            print(f"  weirs from DSM+{DZ_DEFAULT} m: {len(levees_noz)} segments")
            sf.weirs.create(locations=levees_noz, dep=str(DSM_TIF),
                            dz=DZ_DEFAULT, par1=WEIR_CD)
        if len(levees_z) == 0 and len(levees_noz) == 0:
            raise ValueError("No levee segments to schematize as weirs")

    # --- 8. write everything (sfincs.inp written last, incl. component files) ---
    sf.write()
    if with_levees:
        if len(levees_noz) > 0:
            print(f"  WARNING: {len(levees_noz)} segments without z are NOT in "
                  f"the manual weirfile (script 15 should assign z to all)")
        verify_weirs_in_inp(root, levees_z)
    write_run_bat(root, EXE_PATH)
    print(f"  written: {root}")
    return sf


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Inputs:")
    reclass_csv = write_reclass_table(RECLASS_CSV)
    dis_gdf = load_discharge_points(DISCHARGE_GPKG)
    dis_ts = constant_timeseries(dis_gdf, TSTART, SIM_HOURS)
    levees_z, levees_noz = prepare_levees(LEVEES_GPKG)

    roots = [OUT_ROOT / "sfincs_baseline", OUT_ROOT / "sfincs_levees"]
    build_model(roots[0], False, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv)
    build_model(roots[1], True, dis_gdf, dis_ts, levees_z, levees_noz, reclass_csv)

    if RUN_AFTER_BUILD:
        print("\nRunning models locally...")
        for r in roots:
            if SKIP_EXISTING and (r / "sfincs_map.nc").exists():
                print(f"  skipping run {r.name} (sfincs_map.nc exists)")
                continue
            run_model(r, EXE_PATH)
    else:
        print("\nBoth models built. Run them locally without Docker by either")
        print("double-clicking run.bat in each folder, or from a shell:")
        for r in roots:
            print(f'  pushd "{r}" && "{EXE_PATH}" & popd')
        print("Or set RUN_AFTER_BUILD = True to launch both from this script.")

    print("\nSteadiness check: compare zsmax at ~0.75*T and T; if they differ,")
    print("extend SIM_HOURS. Compare runs via zsmax difference (levees - baseline).")


if __name__ == "__main__":
    main()