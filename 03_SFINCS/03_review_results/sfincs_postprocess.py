
"""SFINCS postprocess in one pass: (1) georeferenced depth GeoTIFFs from
sfincs_map.nc, (2) steadiness check of both runs, (3) baseline vs levees
comparison statistics, (4) light thesis-style figures + sfincs_summary.json
"""

import warnings
warnings.filterwarnings("ignore")

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal, osr

gdal.UseExceptions()

# ---- CONFIG -----------------------------------------------------------------
MODEL_ROOT = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100")
RUN_BASELINE = MODEL_ROOT / "sfincs_baseline"
RUN_LEVEES   = MODEL_ROOT / "sfincs_levees"

FALLBACK_EPSG = 2180
MIN_DEPTH_M   = 0.05     # wet cell / nodata cut for depth tifs
STEADY_DZ_M   = 0.05     # steady if max hourly |dz| over wet cells < this
DIFF_MIN_M    = 0.05     # a cell counts as changed if |dz| exceeds this
NODATA        = -9999.0

OUT_DIR = Path(__file__).parent / "diagnostics_ch4"

INK, SUB, GRID, SPINE = "#1A1A1A", "#555555", "#E5E7EB", "#BBBBBB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelcolor": INK, "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "xtick.color": SUB, "ytick.color": SUB, "text.color": INK,
    "legend.fontsize": 9.5,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


# ---- shared low-level helpers (proven in scripts 16/26) ---------------------

def parse_inp(inp_path):
    text = inp_path.read_text()
    def get(key, cast=float, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(\S+)", text, flags=re.M)
        if m is None:
            if default is not None:
                return default
            raise KeyError(f"'{key}' not in {inp_path}")
        return cast(m.group(1))
    return {"x0": get("x0"), "y0": get("y0"), "dx": get("dx"),
            "dy": get("dy"), "mmax": get("mmax", int),
            "nmax": get("nmax", int), "epsg": get("epsg", int, FALLBACK_EPSG)}


def read_var(nc_path, var):
    ds = gdal.Open(f'NETCDF:"{nc_path}":{var}')
    arr = ds.ReadAsArray().astype(np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    fill = ds.GetRasterBand(1).GetNoDataValue()
    if fill is not None:
        arr[arr == fill] = np.nan
    arr[np.abs(arr) > 1e20] = np.nan
    ds = None
    return arr


def orient(arr2d, x2d, y2d):
    if np.nanmax(np.abs(np.diff(x2d, axis=0))) > \
       np.nanmax(np.abs(np.diff(x2d, axis=1))):
        arr2d, x2d, y2d = arr2d.T, x2d.T, y2d.T
    if x2d[0, 0] > x2d[0, -1]:
        arr2d, x2d, y2d = arr2d[:, ::-1], x2d[:, ::-1], y2d[:, ::-1]
    if y2d[0, 0] < y2d[-1, 0]:
        arr2d, x2d, y2d = arr2d[::-1, :], x2d[::-1, :], y2d[::-1, :]
    return arr2d, x2d, y2d


def write_gtiff(path, arr2d, inp):
    rows, cols = arr2d.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"])
    top = inp["y0"] + inp["nmax"] * inp["dy"]
    ds.SetGeoTransform((inp["x0"], inp["dx"], 0.0, top, 0.0, -inp["dy"]))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(inp["epsg"])
    ds.SetProjection(srs.ExportToWkt())
    out = np.where(np.isfinite(arr2d), arr2d, NODATA).astype(np.float32)
    b = ds.GetRasterBand(1); b.WriteArray(out); b.SetNoDataValue(NODATA)
    ds.FlushCache(); ds = None
    print(f"  saved {path.name}")


# ---- per-run processing -----------------------------------------------------

def process_run(name, root):
    nc = root / "sfincs_map.nc"
    inp = parse_inp(root / "sfincs.inp")
    area_m2 = inp["dx"] * inp["dy"]

    x2d = read_var(nc, "x")[0]
    y2d = read_var(nc, "y")[0]
    zb = read_var(nc, "zb")[0]
    zsmax = np.nanmax(read_var(nc, "zsmax"), axis=0)

    hmax = zsmax - zb
    hmax[hmax < MIN_DEPTH_M] = np.nan
    hmax_o, _, _ = orient(hmax.copy(), x2d.copy(), y2d.copy())
    write_gtiff(root / f"hmax_{name}.tif", hmax_o, inp)
    zs_o, _, _ = orient(zsmax.copy(), x2d.copy(), y2d.copy())
    write_gtiff(root / f"zsmax_{name}.tif", zs_o, inp)

    wet = np.isfinite(hmax)
    stats = {
        "wet_area_km2": float(wet.sum() * area_m2 / 1e6),
        "max_depth_m": float(np.nanmax(hmax)) if wet.any() else None,
        "mean_depth_m": float(np.nanmean(hmax)) if wet.any() else None,
        "volume_mil_m3": float(np.nansum(hmax) * area_m2 / 1e6 / 1e3)
                         if wet.any() else None,   # 10^6 m3
    }

    # steadiness on hourly zs
    zs = read_var(nc, "zs")
    rows = []
    for t in range(zs.shape[0]):
        depth_t = zs[t] - zb
        wet_t = depth_t > MIN_DEPTH_M
        if t == 0:
            dmed = dmax = np.nan
        else:
            both = wet_t | ((zs[t-1] - zb) > MIN_DEPTH_M)
            d = np.abs(zs[t] - zs[t-1])[both]
            dmed = float(np.nanmedian(d)) if d.size else np.nan
            dmax = float(np.nanmax(d)) if d.size else np.nan
        rows.append({"run": name, "hour": t,
                     "wet_area_km2": float(wet_t.sum() * area_m2 / 1e6),
                     "dz_median_m": dmed, "dz_max_m": dmax})
    df = pd.DataFrame(rows)
    last = df.iloc[-1]
    stats["sim_hours"] = int(zs.shape[0] - 1)
    stats["last_hour_dz_max_m"] = float(last["dz_max_m"])
    stats["last_hour_dz_median_m"] = float(last["dz_median_m"])
    stats["steady"] = bool(last["dz_max_m"] < STEADY_DZ_M)
    print(f"  {name}: wet {stats['wet_area_km2']:.1f} km2, "
          f"last dz_max {stats['last_hour_dz_max_m']:.3f} m -> "
          f"{'STEADY' if stats['steady'] else 'NOT steady'}")
    return stats, df, hmax, zsmax, zb, inp, area_m2


# ---- comparison -------------------------------------------------------------

def compare(hb, hl, zsb, zsl, area_m2, inp):
    wet_b = np.isfinite(hb)
    wet_l = np.isfinite(hl)
    protected = wet_b & ~wet_l          # wet without levees, dry with them
    newly_wet = wet_l & ~wet_b
    both = wet_b & wet_l
    dz = np.where(both, zsl - zsb, np.nan)
    changed = both & (np.abs(dz) > DIFF_MIN_M)

    comp = {
        "protected_area_km2": float(protected.sum() * area_m2 / 1e6),
        "newly_wet_area_km2": float(newly_wet.sum() * area_m2 / 1e6),
        "wet_area_diff_km2": float((wet_l.sum() - wet_b.sum()) * area_m2 / 1e6),
        "changed_area_km2": float(changed.sum() * area_m2 / 1e6),
        "dz_median_changed_m": float(np.nanmedian(dz[changed]))
                               if changed.any() else None,
        "dz_min_m": float(np.nanmin(dz)) if both.any() else None,
        "dz_max_m": float(np.nanmax(dz)) if both.any() else None,
    }
    # difference raster for maps (levees - baseline water level, common wet)
    dz_o, _, _ = orient(dz.copy(),
                        read_var_x[0].copy(), read_var_y[0].copy())
    write_gtiff(OUT_DIR / "dz_levees_minus_baseline.tif", dz_o, inp)
    return comp


# ---- figures ----------------------------------------------------------------

def figures(dfs):
    allrows = pd.concat(dfs, ignore_index=True)
    allrows.to_csv(OUT_DIR / "steadiness.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, ls in (("baseline", "-"), ("levees", "--")):
        sub = allrows[allrows["run"] == name]
        ax.plot(sub["hour"], sub["dz_max_m"], lw=1.6, color=INK, ls=ls,
                label="bez hrází" if name == "baseline" else "s hrázemi")
    ax.axhline(STEADY_DZ_M, color=SUB, ls=":", lw=1.1,
               label=f"práh ustálení {STEADY_DZ_M} m")
    ax.set_yscale("log")
    ax.set_xlabel("Čas simulace [h]")
    ax.set_ylabel("Max. hodinová změna hladiny [m]")
    ax.legend(frameon=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_steadiness.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, ls in (("baseline", "-"), ("levees", "--")):
        sub = allrows[allrows["run"] == name]
        ax.plot(sub["hour"], sub["wet_area_km2"], lw=1.6, color=INK, ls=ls,
                label="bez hrází" if name == "baseline" else "s hrázemi")
    ax.set_xlabel("Čas simulace [h]")
    ax.set_ylabel("Zatopená plocha [km²]")
    ax.legend(frameon=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_wet_area.png")
    plt.close(fig)
    print("  saved fig_steadiness.png, fig_wet_area.png")


# ---- main -------------------------------------------------------------------

read_var_x = read_var_y = None

def main():
    global read_var_x, read_var_y
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("baseline:")
    sb, dfb, hb, zsb, zbb, inp_b, area = process_run("baseline", RUN_BASELINE)
    print("levees:")
    sl, dfl, hl, zsl, zbl, inp_l, _ = process_run("levees", RUN_LEVEES)

    if (inp_b["x0"], inp_b["y0"], inp_b["mmax"], inp_b["nmax"]) != \
       (inp_l["x0"], inp_l["y0"], inp_l["mmax"], inp_l["nmax"]):
        raise RuntimeError("Runs are on different grids - comparison invalid")

    read_var_x = read_var(RUN_BASELINE / "sfincs_map.nc", "x")
    read_var_y = read_var(RUN_BASELINE / "sfincs_map.nc", "y")
    comp = compare(hb, hl, zsb, zsl, area, inp_b)

    summary = {"baseline": sb, "levees": sl, "comparison": comp,
               "thresholds": {"min_depth_m": MIN_DEPTH_M,
                              "steady_dz_m": STEADY_DZ_M,
                              "diff_min_m": DIFF_MIN_M}}
    (OUT_DIR / "sfincs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("  saved sfincs_summary.json")

    figures([dfb, dfl])
    print("Done.")


if __name__ == "__main__":
    main()