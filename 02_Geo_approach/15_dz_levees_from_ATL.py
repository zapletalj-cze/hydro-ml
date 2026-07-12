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

Point filters before aggregation:
    dz > DZ_MAX   bridge decks and similar structures (verified visually)
    dz < DZ_MIN   ATL08 ground under vegetation vs DSM canopy top; a crest
                  point cannot sit below the SURFACE model (datum verified OK
                  on bare surfaces), so such points are not crests
    optional canopy raster filter (CANOPY_TIF): drop points under vegetation
                  taller than CANOPY_MAX_M regardless of dz

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
    fig_crest_points_diagnostics.png  dz distribution + distance vs dz,
                               removal reasons highlighted
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

DETECTED_LEVEES_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\predictions_eval\levees_predicted_Odra.gpkg")
CREST_POINTS_GPKG    = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_terrain_heights_updated_crest.gpkg")  # from 00 script
DSM_TIF              = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c_10m.tif")              # EGM2008, EPSG:2180

OUTPUT_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z.gpkg")  # segments with z, dz_used, method, dsm_med, n_pts

CRS_METRIC = 2180

SEGMENT_LEN_M      = 1000.0   # target segment length
MIN_TAIL_FRAC      = 0.4     # trailing piece shorter than this fraction merges back
MAX_DIST_M         = 30.0    # max point-to-segment distance for assignment
MIN_POINTS_SEGMENT = 3       # measured-z requires at least this many points
MIN_POINTS_LEVEE   = 3       # levee-level dz requires at least this many points
DZ_DEFAULT         = 0.9     # global fallback [m]
DZ_MAX             = 8.0     # cap on dz over DSM [m]: points above are bridge
                             # decks and similar structures, not levee crests
                             # (verified visually on the tallest candidates)
DZ_MIN             = -0.3    # lower tolerance [m]: a crest point cannot sit
                             # below the SURFACE model; more negative dz means
                             # ATL08 ground under vegetation vs DSM canopy top
                             # (datum verified OK on bare surfaces), so such
                             # points are dropped from the dz sample
CONSERVATIVE_PCT   = 20      # percentile for the conservative variant

# Optional canopy filter: sample the canopy-height raster at each point and
# drop points under vegetation taller than CANOPY_MAX_M regardless of dz.
# Set CANOPY_TIF = None to disable (the DZ_MIN tolerance then does the work).
CANOPY_TIF   = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_2180.tif"          # e.g. Path(r"D:\...\canopy_height_2180.tif")
CANOPY_MAX_M = 2.0

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
    Requires the raster to be in the same metric CRS as the levees."""
    from osgeo import gdal, osr
    gdal.UseExceptions()
    ds = gdal.Open(str(dsm_path))
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    code = srs.GetAuthorityCode(None)
    if code is None or int(code) != int(expect_epsg):
        raise RuntimeError(
            f"Raster CRS (EPSG:{code}) != EPSG:{expect_epsg}; reproject first.")
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
                      [(1 - dc) * dr,       dc * dr]])
        ok = np.isfinite(block)
        if not ok.any():
            return float("nan")
        wsum = w[ok].sum()
        if wsum <= 0:
            return float("nan")
        return float((block[ok] * w[ok]).sum() / wsum)

    sample._ds = ds  # keep the GDAL dataset alive (else band handle is freed)
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
# QA FIGURES
# ============================================================

INK, SECOND, GRID = "#1F2937", "#6B7280", "#E5E7EB"
C_Z, C_DSM, C_PT = "#0E7C7B", "#6B7280", "#C2410C"
C_LOW, C_VEG = "#2B6CB0", "#5B8C5A"
METHOD_ALPHA = {"z_measured": 1.0, "dz_levee": 0.55, "dz_default": 0.30}

REASON_STYLE = {
    "kept":       (C_Z,   "ponecháno"),
    "structure":  (C_PT,  "konstrukce (dz > strop)"),
    "below_dsm":  (C_LOW, "pod DSM (vegetace/niva)"),
    "vegetation": (C_VEG, "vegetace (rastr)"),
}


def diagnostics_figure(pts_all, dz_min, dz_max, max_dist, out_path):
    """Three-panel QA/thesis figure over the assigned crest points:
    (a) boxplot of retained dz, (b) histogram of dz by removal reason with
    median/limits, (c) distance-to-levee vs dz by removal reason."""
    kept = pts_all["reason"] == "kept"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=200,
                             gridspec_kw={"width_ratios": [0.8, 1.4, 1.6]})

    # (a) boxplot of retained dz
    ax = axes[0]
    bp = ax.boxplot([pts_all.loc[kept, "dz"].values], widths=0.5,
                    patch_artist=True, showmeans=True,
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor=SECOND,
                                    markeredgecolor="none", alpha=0.4),
                    medianprops=dict(color=INK, linewidth=1.5),
                    whiskerprops=dict(color=SECOND, linewidth=1.1),
                    capprops=dict(color=SECOND, linewidth=1.1),
                    meanprops=dict(marker="D", markersize=5,
                                   markerfacecolor="white",
                                   markeredgecolor=C_Z, markeredgewidth=1.3))
    bp["boxes"][0].set(facecolor=C_Z, alpha=0.20, edgecolor=C_Z, linewidth=1.5)
    ax.set_xticks([])
    ax.set_ylabel("dz nad DSM [m]", color=INK)
    ax.set_title("Rozdělení dz (ponecháno)", fontsize=10.5, color=INK)

    # (b) histogram of dz by reason
    ax = axes[1]
    lo = float(np.floor(pts_all["dz"].min()))
    hi = float(np.ceil(pts_all["dz"].max()))
    bins = np.linspace(lo, hi, 50)
    for reason, (color, label) in REASON_STYLE.items():
        m = pts_all["reason"] == reason
        if m.any():
            ax.hist(pts_all.loc[m, "dz"], bins=bins, color=color, alpha=0.55,
                    edgecolor="none", label=label)
    med = float(pts_all.loc[kept, "dz"].median())
    ax.axvline(med, color=INK, lw=1.4, ls=(0, (5, 2)),
               label=f"medián {med:.2f} m")
    ax.axvline(dz_max, color=C_PT, lw=1.2, ls=":")
    ax.axvline(dz_min, color=C_LOW, lw=1.2, ls=":")
    ax.set_xlabel("dz nad DSM [m]", color=INK)
    ax.set_ylabel("Počet bodů", color=INK)
    ax.set_title("Histogram dz", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5)

    # (c) distance to levee vs dz by reason
    ax = axes[2]
    for reason, (color, label) in REASON_STYLE.items():
        m = pts_all["reason"] == reason
        if m.any():
            ax.scatter(pts_all.loc[m, "dist_m"], pts_all.loc[m, "dz"],
                       s=8 if reason == "kept" else 10, color=color,
                       alpha=0.45 if reason == "kept" else 0.8,
                       edgecolors="none", label=label)
    ax.axhline(dz_max, color=C_PT, lw=1.2, ls=":")
    ax.axhline(dz_min, color=C_LOW, lw=1.2, ls=":")
    ax.set_xlim(0, max_dist)
    ax.set_xlabel("Vzdálenost od detekované hráze [m]", color=INK)
    ax.set_ylabel("dz nad DSM [m]", color=INK)
    ax.set_title("Vzdálenost od hráze vs dz", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5)

    for ax in axes:
        ax.grid(color=GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=SECOND, length=0)
    fig.suptitle("Korunní body ATL08 vůči DSM a detekovaným hrázím",
                 y=1.02, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def profile_figure(seg_levee, pts_levee, levee_id, out_path):
    """Crest profile of one levee: DSM along segments, resolved z per segment
    (opacity by method), measured points on top."""
    seg = seg_levee.sort_values("chain0")
    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=200)
    ax.plot([], [])
    for _, r in seg.iterrows():
        ax.plot([r["chain0"], r["chain1"]], [r["dsm_med"], r["dsm_med"]],
                color=C_DSM, lw=1.6)
        ax.plot([r["chain0"], r["chain1"]], [r["z"], r["z"]],
                color=C_Z, lw=2.6, alpha=METHOD_ALPHA.get(r["method"], 1.0),
                solid_capstyle="butt")
    if len(pts_levee):
        ax.scatter(pts_levee["chain"], pts_levee["h"], s=14, color=C_PT,
                   zorder=5, label="body ATL08 (koruna)")
    ax.set_xlabel("Staničení podél hráze [m]", color=INK)
    ax.set_ylabel("Výška (EGM2008) [m]", color=INK)
    ax.set_title(f"Hráz {levee_id}: koruna po segmentech "
                 f"(plná = měřeno, slabší = dz fallback), DSM šedě",
                 fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=SECOND, length=0)
    if len(pts_levee):
        ax.legend(frameon=False, fontsize=8.5)
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
    })
    pts_table["dsm_at_pt"] = [sampler(p.x, p.y) for p in pts.geometry]
    pts_table = pts_table[np.isfinite(pts_table["dsm_at_pt"])]
    pts_table["dz"] = pts_table["h"] - pts_table["dsm_at_pt"]
    # distance of each point to its assigned segment (diagnostics)
    pts_table["dist_m"] = [seg_geoms[int(s)].distance(p)
                           for s, p in zip(pts_table["seg_idx"],
                                           pts.geometry[pts_table.index])]
    # chainage of each point along its levee, for the QA profiles
    chain = []
    for sidx, p in zip(pts_table["seg_idx"], pts.geometry[pts_table.index]):
        r = seg_table.loc[seg_table["seg_idx"] == sidx].iloc[0]
        chain.append(r["chain0"] + seg_geoms[int(sidx)].project(p))
    pts_table["chain"] = chain
    print(f"assigned points (<= {MAX_DIST_M:.0f} m, DSM ok): {len(pts_table)}")

    # ---- point filters: structures above, vegetation below ----
    pts_all = pts_table.copy()                    # kept for diagnostics figure
    pts_all["reason"] = "kept"

    # optional canopy filter: drop points under tall vegetation regardless of dz
    if CANOPY_TIF is not None:
        canopy_sampler = make_dsm_sampler(CANOPY_TIF)
        canopy_h = np.array([canopy_sampler(p.x, p.y)
                             for p in pts.geometry[pts_table.index]])
        veg = np.isfinite(canopy_h) & (canopy_h > CANOPY_MAX_M)
        pts_all.loc[pts_table.index[veg], "reason"] = "vegetation"
        pts_table = pts_table[~veg]
        print(f"  removed {int(veg.sum())} points under vegetation "
              f"> {CANOPY_MAX_M:.1f} m (canopy raster)")

    capped = pts_table["dz"] > DZ_MAX
    low = pts_table["dz"] < DZ_MIN
    pts_all.loc[pts_table.index[capped], "reason"] = "structure"
    pts_all.loc[pts_table.index[low], "reason"] = "below_dsm"
    n_capped, n_low = int(capped.sum()), int(low.sum())
    pts_table = pts_table[~capped & ~low].reset_index(drop=True)
    print(f"  removed {n_capped} points with dz > {DZ_MAX:.1f} m "
          f"(bridge decks / structures)")
    print(f"  removed {n_low} points with dz < {DZ_MIN:.1f} m "
          f"(ATL08 ground under canopy vs DSM top; not crest)")
    if len(pts_table):
        print(f"  dz over DSM ({len(pts_table)} pts): "
              f"median {pts_table['dz'].median():+.2f} m, "
              f"p20 {np.percentile(pts_table['dz'], 20):+.2f} m, "
              f"p80 {np.percentile(pts_table['dz'], 80):+.2f} m")

    # ---- resolve z per segment ----
    seg_table = resolve_segments(seg_table, pts_table)

    # ---- diagnostics figure (all assigned points, removal reasons) ----
    diagnostics_figure(pts_all, DZ_MIN, DZ_MAX, MAX_DIST_M,
                       out_dir / "fig_crest_points_diagnostics.png")
    print(f"saved fig_crest_points_diagnostics.png")

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