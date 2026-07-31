"""
SFINCS helper tools: input preparation, weir file writing, local kernel runs.

Author: Jakub Zapletal
Date:   2026-04-05
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .gis import Vector


def write_reclass_table(path, class_to_manning):
    """Land-use -> Manning reclass table; hydromt reads it with index_col=0
    and expects the values in a column named 'N'."""
    df = pd.DataFrame(
        {"lulc": list(class_to_manning.keys()),
         "N": [class_to_manning[k] for k in class_to_manning]}
    )
    df.to_csv(path, index=False)
    print(f"  reclass table: {path}")
    return path


def load_discharge_points(path, epsg):
    """Load and validate discharge points (columns: index, q_rp100)."""
    gdf = Vector.load_vector(path, target_epsg=epsg)
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
    """Constant hydrograph per point; columns are the integer point ids."""
    t0 = pd.to_datetime(tstart, format="%Y%m%d %H%M%S")
    times = pd.date_range(t0, t0 + pd.Timedelta(hours=hours), freq="1h")
    data = {int(idx): np.full(len(times), float(q))
            for idx, q in zip(gdf["index"], gdf["q_rp100"])}
    df = pd.DataFrame(data, index=times)
    df.index.name = "time"
    return df


def prepare_levees(path, domain_path, epsg, z_column="z", dz_default=None):
    """Load levee lines, clip to the model domain, split by crest-z presence.
    Returns (gdf_with_z, gdf_without_z)."""
    gdf = Vector.load_vector(path, target_epsg=epsg)

    # Clip to the domain first; segments outside it make hydromt fail with
    # "GeoDataFrame has no data after masking"
    n_before = len(gdf)
    domain = Vector.load_vector(domain_path, target_epsg=epsg)
    gdf = Vector.clip_vector(gdf, domain)
    gdf = Vector.explode_to_lines(gdf)
    print(f"  levees clipped to domain: {len(gdf)} of {n_before} segments "
          f"({gdf.geometry.length.sum() / 1000:.1f} km)")
    if len(gdf) == 0:
        raise ValueError(
            "No levee segments inside the model domain - check that the levee "
            "file matches the modelled river")

    if z_column in gdf.columns:
        if z_column != "z":
            gdf = gdf.rename(columns={z_column: "z"})
        has_z = gdf["z"].notna()
    else:
        gdf["z"] = np.nan
        has_z = pd.Series(False, index=gdf.index)

    gdf_z = gdf[has_z].copy()
    gdf_noz = gdf[~has_z].drop(columns=["z"]).copy()
    print(f"  levees: {len(gdf)} lines | crest z present: {len(gdf_z)} "
          f"({100 * len(gdf_z) / max(len(gdf), 1):.0f} %) | "
          f"fallback DSM+{dz_default} m: {len(gdf_noz)}")
    return gdf_z, gdf_noz


def model_is_built(root):
    """A model counts as built when sfincs.inp and the elevation grid exist."""
    return (root / "sfincs.inp").exists() and (root / "sfincs.dep").exists()


def write_run_bat(root, exe_path):
    """Write a run.bat that runs the kernel from inside the model folder."""
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
    """Run one model synchronously with cwd = the model folder."""
    if not Path(exe_path).exists():
        raise FileNotFoundError(f"SFINCS executable not found: {exe_path}")
    print(f"  running {root.name} ...")
    with open(root / "sfincs_log.txt", "w") as log:
        subprocess.run([str(exe_path)], cwd=str(root),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"  finished {root.name} (see sfincs_log.txt)")


def write_weirfile_manual(root, levees_z, weir_cd, filename="sfincs.weir"):
    """Write the SFINCS structure (tekal) weir file: one block per segment,
    columns x y z_crest par1(Cd)."""
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
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {weir_cd:.2f}\n")
            n_struct += 1
    print(f"  manually wrote {n_struct} weir structures -> {path.name}")
    return filename


def force_weirfile_in_inp(root, levees_z, weir_cd):
    """Rewrite sfincs.weir from the segment z values and force the inp
    reference; hydromt weirs.create writes z=0 for every vertex, which the
    kernel prunes as below-bed ('0 structure u/v points found')."""
    inp_path = root / "sfincs.inp"
    inp = inp_path.read_text()

    weir_name = write_weirfile_manual(root, levees_z, weir_cd)

    zs = levees_z["z"].astype(float)
    print(f"  weir crest z: min {zs.min():.2f} m, median {zs.median():.2f} m, "
          f"max {zs.max():.2f} m")

    if re.search(r"^\s*weirfile\s*=", inp, flags=re.M):
        inp = re.sub(r"^\s*weirfile\s*=\s*\S+", f"weirfile           = {weir_name}",
                     inp, flags=re.M)
    else:
        inp = inp.rstrip() + f"\nweirfile           = {weir_name}\n"
    inp_path.write_text(inp)
    print(f"  weirfile set to '{weir_name}' in sfincs.inp")
