"""
Levee crest elevation per segment from ATL08 crest points
=========================================================

Splits the detected levee lines into segments (~SEGMENT_LEN_M) and assigns each
segment a crest elevation for the SFINCS weir schematization, using the ATL08
crest points (output of 00_ICESAT_ATL08_query.py, crest_flag subset).

Height semantics (important):
    - Segments WITH enough measured points get the ABSOLUTE crest elevation
      directly: median of h_te_ortho (EGM2008) of their points. No dz involved,
      fewest error terms.
    - dz is only the FALLBACK quantity for unmeasured segments, defined as
      crest minus DSM at the point (how much the DSM underestimates the crest).
      Fallback crest = median DSM along the segment + dz. Prominence (crest vs
      floodplain) must NOT be used as dz here: the DSM along the levee line
      already contains the smoothed levee, so adding prominence would count the
      levee twice and overestimate the crest.

Aggregation: median as the main estimate (the hydraulically relevant weak spot
is the LOW point, so a high percentile would overstate protection); a
conservative 20th-percentile variant is written alongside (z_cons) for a
sensitivity run.

Three-level fallback per segment (method column):
    z_measured   >= MIN_POINTS_SEGMENT points on the segment
    dz_levee     levee as a whole has >= MIN_POINTS_LEVEE points
    dz_default   no usable points on the levee -> DZ_DEFAULT

Outputs (OUTPUT_GPKG + siblings):
    levee_segments_z.gpkg      segments with: levee_id, seg_id, length_m,
                               n_pts, method, dz_used, dsm_med, z, z_cons
    levee_segments_summary.csv share of levee length per method + dz stats
    profiles/*.png             QA crest profiles for the longest levees

Builder hookup: every segment carries 'z', so in 11_build_sfincs_models.py the
whole file goes through the Z_COLUMN="z" path (no dz call needed). For the
conservative sensitivity run point Z_COLUMN at "z_cons".

Dependencies: geopandas+pyogrio, shapely, GDAL, pandas, matplotlib.
No rasterio, no fiona.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

# ============================================================
# CONFIG
# ============================================================

DETECTED_LEVEES_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\inference_outputs\detected_levees_Wisla.gpkg")  # keep your previous value if it differs
CREST_POINTS_GPKG    = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_crest_2d_th030.gpkg")  # 2D selection, threshold 0.3 (script 30)
DSM_TIF              = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\dtm\COP_DSM_10m_Wistula.tif")  # EGM2008, EPSG:2180

OUTPUT_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z_v2.gpkg")  # v2: old file stays for comparison

CRS_METRIC = 2180

SEGMENT_LEN_M      = 300.0   # target segment length
MIN_TAIL_FRAC      = 0.4     # trailing piece shorter than this fraction merges back
MAX_DIST_M         = 30.0    # max point-to-segment distance for assignment
MIN_POINTS_SEGMENT = 3       # measured-z requires at least this many points
MIN_POINTS_LEVEE   = 3       # levee-level dz requires at least this many points
DZ_MIN             = -0.3    # points with dz below this are dropped (vegetation)
DZ_DEFAULT         = None    # None -> median dz of all kept points (printed)
CONSERVATIVE_PCT   = 20      # percentile for the conservative variant

H_COLUMN = "h_te_ortho"      # crest elevation column on the points (EGM2008)

DSM_SAMPLE_STEP_M = 20.0     # DSM sampling step along each segment
N_PROFILE_FIGURES = 3        # QA profiles for the N longest levees

# ============================================================
# CORE LOGIC (import-safe without GDAL/geopandas; unit-testable)
# ============================================================

def split_line(line, seg_len=SEGMENT_LEN_M, min_tail_frac=MIN_TAIL_FRAC):
    """Split a LineString into consecutive segments of ~seg_len metres.
    A trailing piece shorter than min_tail_frac*seg_len merges into the
    previous segment so no tiny stubs are produced."""
    L = line.length
    if L <= seg_len * (1 + min_tail_frac):
        return [line]
    n_full = int(L // seg_len)
    cuts = [i * seg_len for i in range(n_full + 1)]
    tail = L - cuts[-1]
    if tail < min_tail_frac * seg_len:
        cuts[-1] = L            # merge tail into the last segment
    elif tail > 0:
        cuts.append(L)
    return [substring(line, cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def assign_points_to_segments(seg_geoms, pt_geoms, max_dist=MAX_DIST_M):
    """Nearest-segment assignment with a distance cap.
    Returns for each point the segment index or -1."""
    tree = STRtree(seg_geoms)
    out = np.full(len(pt_geoms), -1, dtype=int)
    for i, pt in enumerate(pt_geoms):
        j = int(tree.nearest(pt))
        if seg_geoms[j].distance(pt) <= max_dist:
            out[i] = j
    return out


def resolve_segments(seg_table, pts_table, dz_default=DZ_DEFAULT,
                     min_pts_seg=MIN_POINTS_SEGMENT,
                     min_pts_levee=MIN_POINTS_LEVEE,
                     cons_pct=CONSERVATIVE_PCT):
    """Decide z / z_cons / method per segment via the three-level fallback.

    seg_table: DataFrame with [seg_idx, levee_id, dsm_med]
    pts_table: DataFrame with [seg_idx, levee_id, h, dz]  (assigned points only)
    Returns seg_table with added columns n_pts, method, dz_used, z, z_cons.
    """
    seg = seg_table.copy()
    seg["n_pts"] = 0
    seg["method"] = "dz_default"
    seg["dz_used"] = float(dz_default)
    seg["z"] = np.nan
    seg["z_cons"] = np.nan

    by_seg = pts_table.groupby("seg_idx") if len(pts_table) else None
    by_levee = pts_table.groupby("levee_id") if len(pts_table) else None
    levee_dz_med, levee_dz_cons, levee_n = {}, {}, {}
    if by_levee is not None:
        for lid, g in by_levee:
            levee_n[lid] = len(g)
            levee_dz_med[lid] = float(np.median(g["dz"]))
            levee_dz_cons[lid] = float(np.percentile(g["dz"], cons_pct))

    for i in seg.index:
        sidx = seg.at[i, "seg_idx"]
        lid = seg.at[i, "levee_id"]
        dsm_med = seg.at[i, "dsm_med"]

        g = by_seg.get_group(sidx) if (by_seg is not None
                                       and sidx in by_seg.groups) else None
        n = 0 if g is None else len(g)
        seg.at[i, "n_pts"] = n

        if n >= min_pts_seg:
            seg.at[i, "method"] = "z_measured"
            seg.at[i, "z"] = float(np.median(g["h"]))
            seg.at[i, "z_cons"] = float(np.percentile(g["h"], cons_pct))
            seg.at[i, "dz_used"] = float(np.median(g["dz"]))
        elif levee_n.get(lid, 0) >= min_pts_levee:
            seg.at[i, "method"] = "dz_levee"
            seg.at[i, "dz_used"] = levee_dz_med[lid]
            seg.at[i, "z"] = dsm_med + levee_dz_med[lid]
            seg.at[i, "z_cons"] = dsm_med + levee_dz_cons[lid]
        else:
            seg.at[i, "z"] = dsm_med + dz_default
            seg.at[i, "z_cons"] = dsm_med + dz_default
    return seg


# ============================================================
# DSM SAMPLER (GDAL, windowed bilinear; no rasterio)
# ============================================================

def make_dsm_sampler(dsm_path, expect_epsg=CRS_METRIC):
    """Windowed bilinear sampler f(x, y) -> elevation (NaN on nodata/outside).
    Requires the DSM to be in the same metric CRS as the levees."""
    from osgeo import gdal, osr
    gdal.UseExceptions()
    ds = gdal.Open(str(dsm_path))
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    code = srs.GetAuthorityCode(None)
    if code is None or int(code) != int(expect_epsg):
        raise RuntimeError(
            f"DSM CRS (EPSG:{code}) != EPSG:{expect_epsg}; reproject the DSM first.")
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    inv_gt = gdal.InvGeoTransform(ds.GetGeoTransform())
    nx, ny = ds.RasterXSize, ds.RasterYSize

    def sample(x, y):
        col = inv_gt[0] + inv_gt[1] * x + inv_gt[2] * y
        row = inv_gt[3] + inv_gt[4] * x + inv_gt[5] * y
        cf, rf = col - 0.5, row - 0.5
        c0, r0 = int(np.floor(cf)), int(np.floor(rf))
        if c0 < 0 or r0 < 0 or c0 + 1 >= nx or r0 + 1 >= ny:
            return float("nan")
        block = band.ReadAsArray(c0, r0, 2, 2).astype(np.float64)
        if nodata is not None:
            block[block == nodata] = np.nan
        dc, dr = cf - c0, rf - r0
        w = np.array([[(1 - dc) * (1 - dr), dc * (1 - dr)],
                      [(1 - dc) * dr, dc * dr]])
        v = np.nansum(block * w.T * 0 + block * np.array([[ (1-dc)*(1-dr), dc*(1-dr)],
                                                          [ (1-dc)*dr,     dc*dr   ]]))
        return float(v) if np.isfinite(block).all() else float(np.nansum(
            np.where(np.isfinite(block), block * np.array([[(1-dc)*(1-dr), dc*(1-dr)],
                                                           [(1-dc)*dr,     dc*dr]]), 0.0)))
    return sample


def dsm_median_along(segment, sampler, step=DSM_SAMPLE_STEP_M):
    """Median DSM along a segment, sampled every `step` metres."""
    n = max(2, int(segment.length // step) + 1)
    ds = np.linspace(0.0, segment.length, n)
    vals = []
    for d in ds:
        p = segment.interpolate(d)
        v = sampler(p.x, p.y)
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


# ============================================================
# QA PROFILE FIGURES
# ============================================================

INK, SECOND, GRID = "#1A1A1A", "#555555", "#E5E7EB"
C_Z, C_DSM, C_PT = "#1A1A1A", "#9A9A9A", "#1A1A1A"
METHOD_ALPHA = {"z_measured": 1.0, "dz_levee": 0.55, "dz_default": 0.30}


def profile_figure(seg_levee, pts_levee, levee_id, out_path):
    """Crest profile of one levee: DSM along segments, resolved z per segment
    (opacity by method), measured points on top."""
    seg = seg_levee.sort_values("chain0")
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=200)
    for _, r in seg.iterrows():
        ax.plot([r["chain0"], r["chain1"]], [r["dsm_med"], r["dsm_med"]],
                color=C_DSM, lw=1.0)
        ax.plot([r["chain0"], r["chain1"]], [r["z"], r["z"]],
                color=C_Z, lw=1.2, alpha=METHOD_ALPHA.get(r["method"], 1.0),
                solid_capstyle="butt")
    if len(pts_levee):
        ax.scatter(pts_levee["chain"], pts_levee["h"], s=9,
                   facecolors="white", edgecolors=C_PT, linewidths=0.7,
                   zorder=5, label="body ATL08 (koruna)")
    ax.plot([], [], color=C_Z, lw=1.2, label="kóta segmentu")
    ax.plot([], [], color=C_DSM, lw=1.0, label="DSM podél segmentu")
    ax.set_xlabel("Staničení podél hráze [m]", color=INK)
    ax.set_ylabel("Výška (EGM2008) [m]", color=INK)
    ax.grid(color=GRID, lw=0.7, axis="y")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color("#BBBBBB")
    ax.tick_params(colors=SECOND, length=0)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    import geopandas as gpd

    out_dir = OUTPUT_GPKG.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load levees ----
    levees = gpd.read_file(DETECTED_LEVEES_GPKG, engine="pyogrio")
    if levees.crs is None:
        levees.set_crs(epsg=CRS_METRIC, inplace=True)
    elif levees.crs.to_epsg() != CRS_METRIC:
        levees = levees.to_crs(epsg=CRS_METRIC)
    levees = levees.explode(index_parts=False).reset_index(drop=True)
    levees = levees[levees.geometry.geom_type == "LineString"]
    print(f"levees: {len(levees)} lines, total {levees.length.sum()/1000:.1f} km")

    # ---- load crest points ----
    pts = gpd.read_file(CREST_POINTS_GPKG, engine="pyogrio")
    if pts.crs is None:
        pts.set_crs(epsg=4326, inplace=True)
    pts = pts.to_crs(epsg=CRS_METRIC)
    if "crest_flag" in pts.columns:
        pts = pts[pts["crest_flag"].astype(bool)].copy()
    if H_COLUMN not in pts.columns:
        raise ValueError(f"Points file lacks '{H_COLUMN}'")
    pts = pts[np.isfinite(pts[H_COLUMN].values)].reset_index(drop=True)
    print(f"crest points: {len(pts)}")

    # ---- segmentation ----
    seg_geoms, seg_rows = [], []
    for lid, line in zip(levees.index, levees.geometry):
        chain = 0.0
        for s in split_line(line):
            seg_rows.append({"seg_idx": len(seg_geoms), "levee_id": int(lid),
                             "length_m": s.length,
                             "chain0": chain, "chain1": chain + s.length})
            seg_geoms.append(s)
            chain += s.length
    seg_table = pd.DataFrame(seg_rows)
    print(f"segments: {len(seg_table)} (target {SEGMENT_LEN_M:.0f} m)")

    # ---- DSM sampling ----
    sampler = make_dsm_sampler(DSM_TIF)
    seg_table["dsm_med"] = [dsm_median_along(g, sampler) for g in seg_geoms]
    n_bad = int((~np.isfinite(seg_table["dsm_med"])).sum())
    if n_bad:
        print(f"  WARNING: {n_bad} segments without DSM coverage")

    # ---- assign points, compute per-point dz over DSM ----
    assign = assign_points_to_segments(seg_geoms, list(pts.geometry))
    keep = assign >= 0
    pts = pts[keep].reset_index(drop=True)
    assign = assign[keep]
    pts_table = pd.DataFrame({
        "seg_idx": assign,
        "levee_id": seg_table.loc[assign, "levee_id"].values,
        "h": pts[H_COLUMN].values,
        "geom": list(pts.geometry.values),
    })
    pts_table["dsm_at_pt"] = [sampler(p.x, p.y) for p in pts_table["geom"]]
    pts_table = pts_table[np.isfinite(pts_table["dsm_at_pt"])]
    pts_table["dz"] = pts_table["h"] - pts_table["dsm_at_pt"]
    n_before = len(pts_table)
    pts_table = pts_table[pts_table["dz"] >= DZ_MIN].reset_index(drop=True)
    print(f"dz tolerance {DZ_MIN} m: kept {len(pts_table)} of {n_before}")
    # chainage of each point along its levee, for the QA profiles
    chain = []
    for sidx, p in zip(pts_table["seg_idx"], pts_table["geom"]):
        r = seg_table.loc[seg_table["seg_idx"] == sidx].iloc[0]
        chain.append(r["chain0"] + seg_geoms[int(sidx)].project(p))
    pts_table["chain"] = chain
    pts_table = pts_table.drop(columns=["geom"])

    print(f"assigned points (<= {MAX_DIST_M:.0f} m, DSM ok): {len(pts_table)}")
    if len(pts_table):
        print(f"  dz over DSM: median {pts_table['dz'].median():+.2f} m, "
              f"p20 {np.percentile(pts_table['dz'], 20):+.2f} m, "
              f"p80 {np.percentile(pts_table['dz'], 80):+.2f} m")

    # ---- resolve z per segment ----
    dz_default = (float(np.median(pts_table["dz"])) if DZ_DEFAULT is None
                  and len(pts_table) else (DZ_DEFAULT or 0.9))
    print(f"global dz default: {dz_default:+.2f} m "
          f"({'median of kept points' if DZ_DEFAULT is None else 'fixed'})")
    seg_table = resolve_segments(seg_table, pts_table, dz_default=dz_default)

    # ---- outputs ----
    gdf = gpd.GeoDataFrame(seg_table, geometry=seg_geoms,
                           crs=f"EPSG:{CRS_METRIC}")
    gdf.to_file(OUTPUT_GPKG, driver="GPKG", engine="pyogrio")
    print(f"\nsaved {OUTPUT_GPKG}")

    total = seg_table["length_m"].sum()
    summary = (seg_table.groupby("method")["length_m"].sum() / total * 100
               ).round(1).rename("percent_length").reset_index()
    summary_path = out_dir / "levee_segments_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"saved {summary_path.name}")
    print(summary.to_string(index=False))

    # ---- QA profiles for the longest levees ----
    prof_dir = out_dir / "profiles"
    prof_dir.mkdir(exist_ok=True)
    longest = (seg_table.groupby("levee_id")["length_m"].sum()
               .sort_values(ascending=False).head(N_PROFILE_FIGURES).index)
    for lid in longest:
        profile_figure(seg_table[seg_table["levee_id"] == lid],
                       pts_table[pts_table["levee_id"] == lid],
                       lid, prof_dir / f"profile_levee_{lid}.png")
    print(f"saved {len(longest)} QA profiles to {prof_dir}")
    print("Done.")


if __name__ == "__main__":
    main()