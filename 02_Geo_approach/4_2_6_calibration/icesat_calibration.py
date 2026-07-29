"""Crest selection by the four simple rules + threshold sweep vs the 1 m DTM.
Input: point export within 200 m of reference levees. Rules: candidate within
30 m of detected levee axis; 2D prominence = height minus median of reference
ring 50-200 m (reference = points >30 m from any levee, canopy < 2 m, >= 10
points required); structure cap 8 m. Sweeps thresholds and validates each
selection against the DTM crest. Outputs JSON + CSV + new crest GPKG."""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()

# ---- CONFIG -----------------------------------------------------------------
POINTS_GPKG   = Path(r"D:\PATH\TO\atl08_points_200m.gpkg")        # FILL: your 200 m export
DTM_TIF       = Path(r"D:\PATH\TO\DTM_PL_1m_KRON86.tif")          # FILL: same as script 28
SEGMENTS_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z.gpkg")
BDOT_GPKG     = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg")  # optional, reference levees

CAND_DIST_M   = 30.0    # rule 1: candidate within this distance of detected axis
RING_MIN_M    = 50.0    # rule 2: reference ring (adjusted to the 200 m export)
RING_MAX_M    = 200.0
LEVEE_EXCL_M  = 30.0    # reference excludes points this close to any levee
CANOPY_MAX_M  = 2.0
MIN_REF_PTS   = 10      # rule 4: decision requires at least this many refs
PROM_CAP_M    = 8.0     # rule 3b: structures (bridges) above this are dropped
THRESHOLDS    = [0.3, 0.5, 0.7]

CREST_WIN_M   = 5.0
GROUND_WIN_M  = 1.0
OFFSET_SAMPLE = 5000
OUT_DIR = Path(__file__).parent / "diagnostics_ch4"

INK, SUB, GRID, SPINE = "#1A1A1A", "#555555", "#E5E7EB", "#BBBBBB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelcolor": INK, "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "xtick.color": SUB, "ytick.color": SUB, "text.color": INK,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


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
        a = self.band.ReadAsArray(x0, y0, w, h).astype(np.float64)
        if self.nod is not None:
            a[a == self.nod] = np.nan
        a[np.abs(a) > 1e10] = np.nan
        return a

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


def to2180(g):
    if g.crs is None:
        return g.set_crs(epsg=2180)
    return g.to_crs(epsg=2180) if g.crs.to_epsg() != 2180 else g


def dist_to_lines(pts, lines_gdf, max_d):
    """Distance from each point to the nearest line (inf beyond max_d)."""
    j = gpd.sjoin_nearest(pts[["geometry"]], lines_gdf[["geometry"]],
                          how="left", max_distance=max_d,
                          distance_col="d")
    d = j.groupby(j.index)["d"].min()
    out = np.full(len(pts), np.inf)
    out[d.index.values] = d.values
    return out


def stats_block(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return {"n": int(d.size), "median_m": float(np.median(d)),
            "mae_m": float(np.mean(np.abs(d))),
            "p20_m": float(np.percentile(d, 20)),
            "p80_m": float(np.percentile(d, 80))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"rules": {"cand_dist_m": CAND_DIST_M,
                        "ring_m": [RING_MIN_M, RING_MAX_M],
                        "levee_excl_m": LEVEE_EXCL_M,
                        "canopy_max_m": CANOPY_MAX_M,
                        "min_ref_pts": MIN_REF_PTS,
                        "prom_cap_m": PROM_CAP_M}}

    pts = to2180(gpd.read_file(POINTS_GPKG, engine="pyogrio"))
    seg = to2180(gpd.read_file(SEGMENTS_GPKG, engine="pyogrio"))
    c_h = pick(pts.columns, "h_te_ortho", "h_ortho", "h", "height_ortho")
    c_can = pick(pts.columns, "canopy_height", "canopy", "veg_height", "canopy_h")
    if c_h is None:
        raise KeyError(f"no ortho height column; columns: {list(pts.columns)}")
    report["input"] = {"n_points": int(len(pts)),
                       "height_col": c_h, "canopy_col": c_can}
    pts = pts.reset_index(drop=True)
    h = pd.to_numeric(pts[c_h], errors="coerce").values
    canopy = pd.to_numeric(pts[c_can], errors="coerce").values if c_can \
        else np.zeros(len(pts))

    # distances
    d_det = dist_to_lines(pts, seg, RING_MAX_M + 50)
    d_any = d_det.copy()
    if BDOT_GPKG.exists():
        bdot = to2180(gpd.read_file(BDOT_GPKG, engine="pyogrio"))
        d_bdot = dist_to_lines(pts, bdot, RING_MAX_M + 50)
        d_any = np.minimum(d_any, d_bdot)

    xy = np.c_[pts.geometry.x.values, pts.geometry.y.values]
    is_ref = (d_any > LEVEE_EXCL_M) & (canopy < CANOPY_MAX_M) & np.isfinite(h)
    is_cand = (d_det <= CAND_DIST_M) & np.isfinite(h)
    report["n_reference"] = int(is_ref.sum())
    report["n_candidates"] = int(is_cand.sum())

    # datum offset on a reference subsample
    dtm = DtmSampler(DTM_TIF)
    ref_idx = np.where(is_ref)[0]
    rng = np.random.default_rng(42)
    sub = rng.choice(ref_idx, size=min(OFFSET_SAMPLE, len(ref_idx)),
                     replace=False)
    diffs = []
    for i in sub:
        v = dtm.ground(xy[i, 0], xy[i, 1])
        if np.isfinite(v):
            diffs.append(v - h[i])
    if len(diffs) < 200:
        (OUT_DIR / "crest_selection_sweep.json").write_text(
            json.dumps(report, indent=2))
        raise RuntimeError(f"offset from only {len(diffs)} points")
    offset = float(np.median(diffs))
    report["datum_offset_m"] = offset
    report["datum_offset_n"] = int(len(diffs))
    print(f"datum offset: {offset:+.2f} m from {len(diffs)} points")

    # 2D prominence for candidates
    tree = cKDTree(xy[is_ref])
    href = h[is_ref]
    prom = np.full(len(pts), np.nan)
    nref = np.zeros(len(pts), int)
    cand_idx = np.where(is_cand)[0]
    for i in cand_idx:
        idx = tree.query_ball_point(xy[i], RING_MAX_M)
        if not idx:
            continue
        dd = np.linalg.norm(xy[is_ref][idx] - xy[i], axis=1)
        ring = [j for j, dj in zip(idx, dd) if dj >= RING_MIN_M]
        nref[i] = len(ring)
        if len(ring) >= MIN_REF_PTS:
            prom[i] = h[i] - float(np.median(href[ring]))

    decided = is_cand & np.isfinite(prom)
    report["n_undecided"] = int(is_cand.sum() - decided.sum())

    # sweep + DTM validation per threshold
    sweep_rows = []
    dtm_cache = {}
    for thr in THRESHOLDS:
        sel = decided & (prom >= thr) & (prom <= PROM_CAP_M)
        diffs = []
        for i in np.where(sel)[0]:
            if i not in dtm_cache:
                dtm_cache[i] = dtm.crest(xy[i, 0], xy[i, 1])
            v = dtm_cache[i]
            if np.isfinite(v):
                diffs.append(h[i] - (v - offset))
        b = stats_block(diffs) or {}
        row = {"threshold": thr, "n_selected": int(sel.sum()), **{
            f"dtm_{k}": v for k, v in b.items()}}
        sweep_rows.append(row)
        print(row)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_DIR / "crest_sweep.csv", index=False)
    report["sweep"] = sweep_rows

    # export new candidate set with prominence and per-threshold flags
    out = pts.loc[cand_idx, ["geometry"]].copy()
    out["h_ortho"] = h[cand_idx]
    out["prominence2d"] = prom[cand_idx]
    out["n_ref"] = nref[cand_idx]
    for thr in THRESHOLDS:
        out[f"sel_{int(thr*100):03d}"] = (
            np.isfinite(prom[cand_idx]) & (prom[cand_idx] >= thr)
            & (prom[cand_idx] <= PROM_CAP_M))
    out.to_file(OUT_DIR / "crest_points_2d.gpkg", driver="GPKG")

    (OUT_DIR / "crest_selection_sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].bar([str(t) for t in sweep["threshold"]], sweep["n_selected"],
                color="#8A8A8A", edgecolor=INK, linewidth=0.6, width=0.55)
    axes[0].set_xlabel("Práh převýšení [m]")
    axes[0].set_ylabel("Počet korunních bodů")
    if "dtm_mae_m" in sweep:
        axes[1].plot(sweep["threshold"], sweep["dtm_mae_m"], color=INK,
                     lw=1.0, marker="o", markersize=3)
        axes[1].set_xlabel("Práh převýšení [m]")
        axes[1].set_ylabel("MAE vůči koruně z DTM [m]")
    for ax in axes:
        for sd in ("left", "bottom"):
            ax.spines[sd].set_color(SPINE)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_crest_sweep.png")
    plt.close(fig)

    print("written: crest_selection_sweep.json, crest_sweep.csv,")
    print("         crest_points_2d.gpkg, fig_crest_sweep.png")


if __name__ == "__main__":
    main()