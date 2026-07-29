"""Validation of levee crest heights against the national 1 m DTM (read-only).
Compares: (1) ATL08 crest points vs DTM crest, (2) segment z by assignment
method vs DTM crest, (3) GLO-30 vs DTM crest (body share). Estimates the
KRON86 vs EGM2008 vertical offset empirically on non-crest low-vegetation
points and removes it. Outputs dtm_validation.json, two CSVs, two figures."""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()

# ---- CONFIG -----------------------------------------------------------------
DTM_TIF = Path(r"D:\PATH\TO\DTM_PL_1m_KRON86.tif")   # 500 GB mosaic, EPSG:2180 - FILL IN
CREST_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_terrain_heights_updated_crest.gpkg")
SEGMENTS_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z.gpkg")

CREST_WIN_M   = 5.0    # crest sampling: max DTM within this radius (local ridge)
GROUND_WIN_M  = 1.0    # ground sampling: median DTM within this radius
SEG_STEP_M    = 10.0   # densification step along segments
CANOPY_MAX_M  = 2.0    # non-crest points below this canopy define the offset
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


# ---- read-only DTM sampler --------------------------------------------------

class DtmSampler:
    def __init__(self, path):
        self.ds = gdal.Open(str(path), gdal.GA_ReadOnly)   # never written
        self.gt = self.ds.GetGeoTransform()
        self.band = self.ds.GetRasterBand(1)
        self.nod = self.band.GetNoDataValue()
        self.nx, self.ny = self.ds.RasterXSize, self.ds.RasterYSize

    def window(self, x, y, radius_m):
        px = (x - self.gt[0]) / self.gt[1]
        py = (y - self.gt[3]) / self.gt[5]
        r = max(1, int(round(radius_m / abs(self.gt[1]))))
        x0, y0 = int(px) - r, int(py) - r
        w = h = 2 * r + 1
        if x0 < 0 or y0 < 0 or x0 + w > self.nx or y0 + h > self.ny:
            return None
        arr = self.band.ReadAsArray(x0, y0, w, h).astype(np.float64)
        if self.nod is not None:
            arr[arr == self.nod] = np.nan
        arr[np.abs(arr) > 1e10] = np.nan
        return arr

    def crest(self, x, y):
        a = self.window(x, y, CREST_WIN_M)
        return float(np.nanmax(a)) if a is not None and np.isfinite(a).any() else np.nan

    def ground(self, x, y):
        a = self.window(x, y, GROUND_WIN_M)
        return float(np.nanmedian(a)) if a is not None and np.isfinite(a).any() else np.nan


def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def block(diffs):
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return {"n": int(d.size), "bias_mean_m": float(d.mean()),
            "median_m": float(np.median(d)),
            "mae_m": float(np.mean(np.abs(d))),
            "p20_m": float(np.percentile(d, 20)),
            "p80_m": float(np.percentile(d, 80))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dtm = DtmSampler(DTM_TIF)
    report = {"config": {"crest_window_m": CREST_WIN_M,
                         "ground_window_m": GROUND_WIN_M,
                         "vertical_datum_dtm": "PL-KRON86-NH",
                         "offset_note": "constant offset estimated on "
                         "non-crest low-vegetation points and removed"}}

    # ---- points ----
    pts = gpd.read_file(CREST_GPKG, engine="pyogrio")
    if pts.crs is None or pts.crs.to_epsg() != 2180:
        pts = pts.set_crs(epsg=2180) if pts.crs is None else pts.to_crs(epsg=2180)
    c_h = pick(pts.columns, "h_te_ortho", "h_ortho", "h", "height_ortho")
    c_crest = pick(pts.columns, "crest_flag", "is_crest", "crest")
    c_canopy = pick(pts.columns, "canopy_height", "canopy", "veg_height")
    c_dsm = pick(pts.columns, "dsm", "dsm_z", "z_dsm", "glo30")
    if c_h is None:
        raise KeyError(f"no ortho-height column found in {CREST_GPKG.name}; "
                       f"columns: {list(pts.columns)}")

    rows = []
    for _, r in pts.iterrows():
        x, y = r.geometry.x, r.geometry.y
        is_crest = bool(r[c_crest]) if c_crest else False
        dtm_val = dtm.crest(x, y) if is_crest else dtm.ground(x, y)
        rows.append({"x": x, "y": y, "is_crest": is_crest,
                     "h_atl08": float(r[c_h]),
                     "canopy": float(r[c_canopy]) if c_canopy else np.nan,
                     "dsm": float(r[c_dsm]) if c_dsm else np.nan,
                     "dtm": dtm_val})
    pdf = pd.DataFrame(rows)

    # empirical datum offset: DTM(KRON86) - ATL08(EGM2008) on bare non-crest pts
    bare = pdf[(~pdf["is_crest"]) & np.isfinite(pdf["dtm"])]
    if c_canopy:
        bare = bare[bare["canopy"] < CANOPY_MAX_M]
    offset = float(np.nanmedian(bare["dtm"] - bare["h_atl08"])) if len(bare) \
        else 0.0
    report["datum_offset_m"] = offset
    report["datum_offset_n"] = int(len(bare))
    pdf["dtm_egm"] = pdf["dtm"] - offset
    pdf.to_csv(OUT_DIR / "dtm_points.csv", index=False)

    crest_pts = pdf[pdf["is_crest"] & np.isfinite(pdf["dtm_egm"])]
    diff_pts = crest_pts["h_atl08"] - crest_pts["dtm_egm"]
    report["atl08_crest_vs_dtm"] = block(diff_pts)
    if c_dsm:
        report["glo30_vs_dtm_crest"] = block(crest_pts["dsm"]
                                             - crest_pts["dtm_egm"])

    # ---- segments ----
    seg = gpd.read_file(SEGMENTS_GPKG, engine="pyogrio")
    if seg.crs is None or seg.crs.to_epsg() != 2180:
        seg = seg.set_crs(epsg=2180) if seg.crs is None else seg.to_crs(epsg=2180)
    c_z = pick(seg.columns, "z")
    c_m = pick(seg.columns, "method")
    srows = []
    for _, r in seg.iterrows():
        geom = r.geometry
        if geom is None or geom.length == 0:
            continue
        n = max(2, int(geom.length // SEG_STEP_M) + 1)
        samples = [dtm.crest(p.x, p.y) for p in
                   (geom.interpolate(d) for d in np.linspace(0, geom.length, n))]
        samples = [s for s in samples if np.isfinite(s)]
        if not samples:
            continue
        crest_dtm = float(np.median(samples)) - offset
        srows.append({"method": str(r[c_m]) if c_m else "n/a",
                      "length_m": float(geom.length),
                      "z_segment": float(r[c_z]),
                      "crest_dtm_egm": crest_dtm,
                      "diff": float(r[c_z]) - crest_dtm})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT_DIR / "dtm_segments.csv", index=False)

    by_method = {}
    for m, g in sdf.groupby("method"):
        b = block(g["diff"])
        if b:
            b["length_km"] = float(g["length_m"].sum() / 1000.0)
            by_method[m] = b
    report["segments_by_method"] = by_method
    report["segments_all"] = block(sdf["diff"])

    (OUT_DIR / "dtm_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    d = np.asarray(diff_pts, float)
    d = d[np.isfinite(d)]
    ax.hist(d, bins=40, color="#C9C9C9", edgecolor=INK, linewidth=0.5)
    ax.axvline(0, color=INK, lw=0.9)
    ax.axvline(np.median(d), color=SUB, ls="--", lw=0.9,
               label=f"medián {np.median(d):+.2f} m")
    ax.set_xlabel("Rozdíl výšky koruny ATL08 a DTM [m]")
    ax.set_ylabel("Počet bodů")
    ax.legend(frameon=False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_dtm_hist.png")
    plt.close(fig)

    order = [m for m in ("measured", "levee_dz", "global_dz")
             if m in by_method] or list(by_method)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    med = [by_method[m]["median_m"] for m in order]
    lo = [by_method[m]["median_m"] - by_method[m]["p20_m"] for m in order]
    hi = [by_method[m]["p80_m"] - by_method[m]["median_m"] for m in order]
    ax.bar(range(len(order)), med, yerr=[lo, hi], capsize=3,
           color="#8A8A8A", edgecolor=INK, linewidth=0.6, width=0.55,
           error_kw={"elinewidth": 0.8, "ecolor": INK})
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Chyba segmentové kóty vůči DTM [m]")
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_dtm_by_method.png")
    plt.close(fig)

    print(f"offset (KRON86-EGM2008, empirical): {offset:+.2f} m "
          f"from {report['datum_offset_n']} points")
    print("written: dtm_validation.json, dtm_points.csv, dtm_segments.csv,")
    print("         fig_dtm_hist.png, fig_dtm_by_method.png")


if __name__ == "__main__":
    main()