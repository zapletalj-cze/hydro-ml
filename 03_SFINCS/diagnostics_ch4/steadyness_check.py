"""Convergence check of SFINCS runs. Reads hourly zs from sfincs_map.nc,
computes per-hour water-level change over wet cells and wet-area evolution,
writes steadiness.csv + two figures (thesis style, no titles)."""

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()

# ---- CONFIG -----------------------------------------------------------------
SFINCS_ROOT = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model")
RUNS = {
    "baseline 65 h":  SFINCS_ROOT / "model_RP100" / "sfincs_baseline",
    "s hrazemi 65 h": SFINCS_ROOT / "model_RP100" / "sfincs_levees",
    # extended runs - adjust folder names to your 70 h pair:
    "baseline 70 h":  SFINCS_ROOT / "model_RP100_70h" / "sfincs_baseline",
    "s hrazemi 70 h": SFINCS_ROOT / "model_RP100_70h" / "sfincs_levees",
}
MIN_DEPTH_M   = 0.05    # wet cell = depth above this
STEADY_DZ_M   = 0.05    # steady if max hourly |dz| over wet cells < this
OUT_DIR = Path(__file__).parent / "diagnostics_ch4"

INK, SUB, GRID, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
COLORS = ["#6B7280", "#0E7C7B", "#9CA3AF", "#C2410C"]
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


def cell_area_m2(root):
    inp = (root / "sfincs.inp").read_text()
    def g(k):
        m = re.search(rf"^\s*{k}\s*=\s*(\S+)", inp, flags=re.M)
        return float(m.group(1))
    return g("dx") * g("dy")


def analyse(name, root):
    nc = root / "sfincs_map.nc"
    if not nc.exists():
        print(f"  SKIP {name}: {nc} not found")
        return None
    zs = read_var(nc, "zs")
    zb = read_var(nc, "zb")[0]
    area = cell_area_m2(root)
    n_t = zs.shape[0]

    rows = []
    for t in range(n_t):
        depth = zs[t] - zb
        wet = depth > MIN_DEPTH_M
        wet_km2 = wet.sum() * area / 1e6
        if t == 0:
            dz_med = dz_max = np.nan
        else:
            both = wet | ((zs[t-1] - zb) > MIN_DEPTH_M)
            d = np.abs(zs[t] - zs[t-1])[both]
            dz_med = float(np.nanmedian(d)) if d.size else np.nan
            dz_max = float(np.nanmax(d)) if d.size else np.nan
        rows.append({"run": name, "hour": t, "wet_area_km2": wet_km2,
                     "dz_median_m": dz_med, "dz_max_m": dz_max})
    df = pd.DataFrame(rows)
    last = df.iloc[-1]
    steady = last["dz_max_m"] < STEADY_DZ_M
    print(f"  {name}: T={n_t-1} h, wet {last['wet_area_km2']:.1f} km2, "
          f"last-hour dz max {last['dz_max_m']:.3f} m -> "
          f"{'STEADY' if steady else 'NOT steady'}")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = [d for name, root in RUNS.items()
           if (d := analyse(name, root)) is not None]
    if not dfs:
        raise RuntimeError("no runs analysed")
    allrows = pd.concat(dfs, ignore_index=True)
    allrows.to_csv(OUT_DIR / "steadiness.csv", index=False)

    # fig 1: hourly max |dz| over wet cells, log scale, threshold line
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for (name, _), c in zip(RUNS.items(), COLORS):
        sub = allrows[allrows["run"] == name]
        if len(sub):
            ax.plot(sub["hour"], sub["dz_max_m"], lw=1.8, color=c, label=name)
    ax.axhline(STEADY_DZ_M, color=INK, ls=":", lw=1.2,
               label=f"práh ustálení {STEADY_DZ_M} m")
    ax.set_yscale("log")
    ax.set_xlabel("Čas simulace [h]")
    ax.set_ylabel("Max. hodinová změna hladiny [m]")
    ax.legend(frameon=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_steadiness_dz.png")
    plt.close(fig)

    # fig 2: wet area evolution
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for (name, _), c in zip(RUNS.items(), COLORS):
        sub = allrows[allrows["run"] == name]
        if len(sub):
            ax.plot(sub["hour"], sub["wet_area_km2"], lw=1.8, color=c,
                    label=name)
    ax.set_xlabel("Čas simulace [h]")
    ax.set_ylabel("Zatopená plocha [km²]")
    ax.legend(frameon=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_wet_area.png")
    plt.close(fig)

    print("written:", OUT_DIR / "steadiness.csv")
    print("written:", OUT_DIR / "fig_steadiness_dz.png")
    print("written:", OUT_DIR / "fig_wet_area.png")


if __name__ == "__main__":
    main()