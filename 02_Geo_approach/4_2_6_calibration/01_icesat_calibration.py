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
POINTS_GPKG   = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\at08_terrain_near_levees.gpkg")
DTM_TIF       = Path(r"B:\01_Projects\154_Poland_Flood_v3\01_MD\01_HAZARD\01_DTM\1m\Poland_dem_1m.tif")
SEGMENTS_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z.gpkg")
BDOT_GPKG     = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg")  # optional, reference levees

CAND_DIST_M   = 30.0    # rule 1: candidate within this distance of detected axis
RING_MIN_M    = 50.0    # rule 2: reference ring (adjusted to the 200 m export)
RING_MAX_M    = 200.0
LEVEE_EXCL_M  = 30.0    # reference excludes points this close to any levee
CANOPY_MAX_M  = 2.0
MIN_REF_PTS   = 10      # rule 4: decision requires at least this many refs
PROM_CAP_M    = 8.0     # rule 3b: structures (bridges) above this are dropped
THRESHOLDS    = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

CREST_WIN_M   = 5.0
GROUND_WIN_M  = 1.0
GRID_KM        = 10.0   # offset grid cell size [km], whole Poland
MIN_CELL_N     = 100    # cells with fewer valid diffs are not mapped
CORRIDOR_KM    = 30.0   # cells within this distance of the axis feed the correction
PROFILE_BIN_KM = 25.0   # bin size of the diff-vs-X and diff-vs-Y profiles
N_WORKERS      = 12     # parallel DTM readers (threads, each with own handle)
CHUNK          = 20000  # points per work unit; each unit saved to the temp dir
OUT_DIR  = Path(__file__).parent / "diagnostics_ch4"   # figures + json + csv
OUT_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\crest_points_2d.gpkg")  # set where the point layer should go

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

    # datum offset from ALL reference points, parallel chunked DTM reads with
    # resume: every finished chunk lands in the temp dir and is never redone
    from shapely.ops import unary_union, nearest_points
    from shapely.geometry import Point as _P
    axis = unary_union(list(seg.geometry.values))

    ref_idx = np.where(np.isfinite(h))[0]
    step = GRID_KM * 1000.0
    cell_of = (np.floor(xy[ref_idx, 0] / step).astype(int) * 100000
               + np.floor(xy[ref_idx, 1] / step).astype(int))
    order = np.argsort(cell_of, kind="stable")   # cell-sorted: cache-friendly

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tmp_dir = OUT_DIR / "offset_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tls = threading.local()

    def worker(k, ois):
        part = tmp_dir / f"chunk_{k:05d}.npz"
        if part.exists():
            return k, "cached"
        if not hasattr(tls, "dtm"):
            tls.dtm = DtmSampler(DTM_TIF)     # one GDAL handle per thread
        out = np.full(len(ois), np.nan)
        for j, oi in enumerate(ois):
            i = ref_idx[oi]
            out[j] = tls.dtm.ground(xy[i, 0], xy[i, 1])
        np.savez_compressed(part, oi=np.asarray(ois), val=out)
        return k, "done"

    chunks = [order[c:c + CHUNK] for c in range(0, len(order), CHUNK)]
    print(f"offset: {len(ref_idx)} points in {len(chunks)} chunks, "
          f"{N_WORKERS} workers, temp: {tmp_dir}")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(worker, k, ois) for k, ois in enumerate(chunks)]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 5 == 0 or done == len(chunks):
                print(f"  {done} / {len(chunks)} chunks")

    diffs = np.full(len(ref_idx), np.nan)
    for k in range(len(chunks)):
        part = np.load(tmp_dir / f"chunk_{k:05d}.npz")
        vals = part["val"]
        ois = part["oi"]
        ok = np.isfinite(vals)
        diffs[ois[ok]] = vals[ok] - h[ref_idx[ois[ok]]]

    dtm = DtmSampler(DTM_TIF)   # main-thread handle for the later crest sampling
    valid = np.isfinite(diffs)
    pxs, pys = xy[ref_idx, 0][valid], xy[ref_idx, 1][valid]
    cell_ids = cell_of[valid]
    diffs = diffs[valid]

    block_stats = []
    for c in np.unique(cell_ids):
        d_c = diffs[cell_ids == c]
        if len(d_c) >= MIN_CELL_N:
            cx = (c // 100000) * step + step / 2
            cy = (c % 100000) * step + step / 2
            block_stats.append({"cell": int(c), "x_m": float(cx),
                                "y_m": float(cy), "n": int(len(d_c)),
                                "median_m": float(np.median(d_c)),
                                "dist_axis_km":
                                    float(_P(cx, cy).distance(axis)) / 1000.0})
    if len(diffs) < 200:
        (OUT_DIR / "crest_selection_sweep.json").write_text(
            json.dumps(report, indent=2))
        raise RuntimeError(f"offset from only {len(diffs)} points")
    corridor_cells = [b["cell"] for b in block_stats
                      if b["dist_axis_km"] <= CORRIDOR_KM]
    in_corr = np.isin(cell_ids, corridor_cells)
    offset_pl = float(np.median(diffs))
    offset = float(np.median(diffs[in_corr])) if in_corr.any() else offset_pl
    report["datum_offset_m"] = offset               # corridor, used as correction
    report["datum_offset_poland_m"] = offset_pl     # country-wide evidence
    report["datum_offset_n"] = int(in_corr.sum())
    report["datum_offset_n_poland"] = int(len(diffs))
    report["datum_offset_spread"] = {
        "std_m": float(diffs.std()),
        "p20_m": float(np.percentile(diffs, 20)),
        "p80_m": float(np.percentile(diffs, 80)),
        "share_within_0_2_m": float((np.abs(diffs - offset_pl) <= 0.2).mean()),
        "share_within_0_5_m": float((np.abs(diffs - offset_pl) <= 0.5).mean()),
    }
    report["datum_offset_cells"] = block_stats
    meds = [b["median_m"] for b in block_stats]
    if meds:
        report["datum_offset_cell_range_m"] = float(max(meds) - min(meds))

    # (a) Kruskal-Wallis: do cells share one distribution of differences?
    from scipy import stats as sps
    groups = [diffs[cell_ids == b["cell"]] for b in block_stats]
    if len(groups) >= 3:
        kw = sps.kruskal(*groups)
        report["offset_kruskal"] = {"H": float(kw.statistic),
                                    "p": float(kw.pvalue)}
    # (b) weighted OLS trend of cell medians on x, y (per 100 km)
    if len(block_stats) >= 5:
        bx = np.array([b["x_m"] for b in block_stats]) / 1e5
        by = np.array([b["y_m"] for b in block_stats]) / 1e5
        bm = np.array([b["median_m"] for b in block_stats])
        bw = np.sqrt(np.array([b["n"] for b in block_stats], float))
        A = np.c_[np.ones(len(bm)), bx, by] * bw[:, None]
        yv = bm * bw
        coef, res, rank, _ = np.linalg.lstsq(A, yv, rcond=None)
        dof = len(bm) - 3
        if dof > 0 and res.size:
            cov = (float(res[0]) / dof) * np.linalg.inv(A.T @ A)
            se = np.sqrt(np.diag(cov))
            tv = coef / se
            pv = 2 * (1 - sps.t.cdf(np.abs(tv), dof))
            report["offset_trend_per_100km"] = {
                "east_m": float(coef[1]), "east_p": float(pv[1]),
                "north_m": float(coef[2]), "north_p": float(pv[2])}

    pd.DataFrame({"x": pxs, "y": pys, "cell": cell_ids,
                  "dtm_minus_atl08_m": diffs}).to_csv(
        OUT_DIR / "datum_offset_points.csv", index=False)
    print(f"offset corridor {offset:+.2f} m (n {int(in_corr.sum())}), "
          f"Poland {offset_pl:+.2f} m (n {len(diffs)}), "
          f"cell range {report.get('datum_offset_cell_range_m', 0):.2f} m")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    lim = max(1.0, float(np.percentile(np.abs(diffs - offset), 99)) + 0.3)
    ax.hist(np.clip(diffs, offset - lim, offset + lim),
            bins=60, color="#C9C9C9", edgecolor=INK, linewidth=0.4)
    ax.axvline(0, color=INK, lw=0.9, label="nulový rozdíl")
    ax.axvline(offset, color=SUB, ls="--", lw=1.0,
               label=f"medián {offset:+.2f} m")
    ax.set_xlabel("Rozdíl DTM a terénní výšky ATL08 [m]")
    ax.set_ylabel("Počet bodů")
    ax.legend(frameon=False)
    for sdn in ("left", "bottom"):
        ax.spines[sdn].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_datum_offset_hist.png")
    plt.close(fig)

    if block_stats:
        from matplotlib import colors as mcolors
        # raster map of cell medians (pcolormesh, NaN = white)
        step_km = GRID_KM
        cxs = np.array([b["x_m"] for b in block_stats]) / 1000
        cys = np.array([b["y_m"] for b in block_stats]) / 1000
        cms = np.array([b["median_m"] for b in block_stats])
        ix = np.round((cxs - cxs.min()) / step_km).astype(int)
        iy = np.round((cys - cys.min()) / step_km).astype(int)
        raster = np.full((iy.max() + 1, ix.max() + 1), np.nan)
        raster[iy, ix] = cms
        xe = cxs.min() - step_km / 2 + np.arange(ix.max() + 2) * step_km
        ye = cys.min() - step_km / 2 + np.arange(iy.max() + 2) * step_km
        lim = max(0.05, float(np.nanmax(np.abs(raster))))
        fig, ax = plt.subplots(figsize=(6.8, 6.4))
        cmap = plt.get_cmap("Greys").copy()
        cmap.set_bad("white")
        pm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(raster), cmap=cmap,
                           norm=mcolors.Normalize(vmin=-lim, vmax=lim),
                           edgecolors="none")
        geoms = axis.geoms if hasattr(axis, "geoms") else [axis]
        for g in geoms:
            arr = np.asarray(g.coords)
            ax.plot(arr[:, 0] / 1000, arr[:, 1] / 1000, color=INK, lw=0.8)
        cb = fig.colorbar(pm, ax=ax, shrink=0.75)
        cb.set_label("Medián rozdílu DTM a ATL08 [m]")
        ax.set_xlabel("X (EPSG:2180) [km]")
        ax.set_ylabel("Y (EPSG:2180) [km]")
        ax.set_aspect("equal")
        ax.grid(False)
        for sdn in ("left", "bottom"):
            ax.spines[sdn].set_color(SPINE)
        ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_datum_offset_map.png")
        plt.close(fig)

        # profiles of the difference along X and along Y (all points)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
        for ax2, coord, lab in ((axes[0], pxs, "X (EPSG:2180) [km]"),
                                (axes[1], pys, "Y (EPSG:2180) [km]")):
            ck = coord / 1000
            bins = np.arange(ck.min(), ck.max() + PROFILE_BIN_KM,
                             PROFILE_BIN_KM)
            mids, med, lo, hi = [], [], [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = (ck >= b0) & (ck < b1)
                if m.sum() >= MIN_CELL_N:
                    mids.append((b0 + b1) / 2)
                    med.append(np.median(diffs[m]))
                    lo.append(np.percentile(diffs[m], 20))
                    hi.append(np.percentile(diffs[m], 80))
            ax2.fill_between(mids, lo, hi, color="#DDDDDD",
                             label="p20 az p80")
            ax2.plot(mids, med, color=INK, lw=1.1, label="median")
            ax2.axhline(0, color=SUB, ls=":", lw=0.9)
            ax2.set_xlabel(lab)
            for sdn in ("left", "bottom"):
                ax2.spines[sdn].set_color(SPINE)
            ax2.tick_params(length=0)
        axes[0].set_ylabel("Rozdil DTM a ATL08 [m]")
        axes[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_datum_offset_profiles.png")
        plt.close(fig)

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

    # sweep + DTM validation per threshold; the DTM crest is sampled at the
    # nearest point ON the detected axis (union of segments), not at the ATL08
    # point, so slope-toe points are judged against the true crest
    sweep_rows = []
    dtm_cache = {}
    for thr in THRESHOLDS:
        sel = decided & (prom >= thr) & (prom <= PROM_CAP_M)
        diffs = []
        for i in np.where(sel)[0]:
            if i not in dtm_cache:
                on_axis = nearest_points(_P(xy[i, 0], xy[i, 1]), axis)[1]
                dtm_cache[i] = dtm.crest(on_axis.x, on_axis.y)
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

    # per-point scatter: prominence vs difference to the axis-anchored DTM
    # crest, over ALL decided candidates (not only selected ones)
    scat_prom, scat_diff = [], []
    for i in np.where(decided)[0]:
        if i not in dtm_cache:
            on_axis = nearest_points(_P(xy[i, 0], xy[i, 1]), axis)[1]
            dtm_cache[i] = dtm.crest(on_axis.x, on_axis.y)
        v = dtm_cache[i]
        if np.isfinite(v):
            scat_prom.append(prom[i])
            scat_diff.append(h[i] - (v - offset))
    pd.DataFrame({"prominence2d": scat_prom, "diff_dtm_m": scat_diff}
                 ).to_csv(OUT_DIR / "crest_points_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(scat_prom, scat_diff, s=5, facecolors="none",
               edgecolors="#8A8A8A", linewidths=0.4, alpha=0.5, zorder=2)
    ax.axhline(0, color=INK, lw=0.9, zorder=3)
    sp = np.asarray(scat_prom); sd = np.asarray(scat_diff)
    bins = np.arange(0.0, min(8.0, np.nanmax(sp)) + 0.25, 0.25)
    bmid, bmed = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (sp >= b0) & (sp < b1)
        if m.sum() >= 15:
            bmid.append((b0 + b1) / 2)
            bmed.append(np.median(sd[m]))
    ax.plot(bmid, bmed, color=INK, lw=1.2, zorder=4,
            label="medián po intervalech")
    for thr in (0.2, 0.3, 0.5):
        ax.axvline(thr, color="#BBBBBB", ls=":", lw=0.9, zorder=1)
    ax.set_xlabel("Plošná prominence [m]")
    ax.set_ylabel("Rozdíl vůči koruně z DTM [m]")
    ax.legend(frameon=False)
    for sdn in ("left", "bottom"):
        ax.spines[sdn].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_crest_scatter.png")
    plt.close(fig)

    # export new candidate set with prominence and per-threshold flags
    out = pts.loc[cand_idx, ["geometry"]].copy()
    out["h_ortho"] = h[cand_idx]
    out["prominence2d"] = prom[cand_idx]
    out["n_ref"] = nref[cand_idx]
    for thr in THRESHOLDS:
        out[f"sel_{int(thr*100):03d}"] = (
            np.isfinite(prom[cand_idx]) & (prom[cand_idx] >= thr)
            & (prom[cand_idx] <= PROM_CAP_M))
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT_GPKG, driver="GPKG")

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
    print("         crest_points_validation.csv, fig_crest_scatter.png,")
    print("         crest_points_2d.gpkg, fig_crest_sweep.png")


if __name__ == "__main__":
    main()