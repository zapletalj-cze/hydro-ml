"""
Training patch generation for levee detection (Poland, large rivers only).

Author: Jakub Zapletal
Date:   2026-04-06
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from shapely.ops import substring, unary_union
from shapely.prepared import prep
from tqdm import tqdm

from gis import Vector, Raster

# ============================================================
# CONFIG
# ============================================================

# ------- Input paths ----------------------------------------
BDOT_GPKG = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg"
MERIT_GPKG = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\riv_pfaf_2x_MERIT_Hydro_v07_Basin_flip.gpkg"
COPDEM_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif"
CANOPY_HEIGHT_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_2180.tif"
CANOPY_HEIGHT_SD_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_SD_2180.tif"
# Binary water mask on the same grid as the DSM
WATER_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\source\pl\water_mask_pl.tif"

OUTPUT_DIR = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_PL"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------- Spatial reference ----------------------------------
TARGET_CRS = "EPSG:2180"  # PUWG 1992
TARGET_EPSG = 2180
POLAND_BBOX_WGS84 = (14.0, 49.0, 24.2, 55.0)  # MERIT file is in WGS84


# ------- MERIT reach attribute columns ----------------------
COMID_COL = "COMID"
NEXTDOWN_COL = "NextDownID"
UPAREA_COL = "uparea"


# ------- River / segment selection --------------------------
MIN_UPAREA_KM2 = 2000  # keep only segments on rivers at least this large
SEGMENT_LENGTH_M = 500
MIN_LEVEE_LENGTH_M = 100
MAX_DIST_TO_REACH_M = 500


# ------- Geographic hold-out (by drainage basin) ------------
# Run with REPORT_BASINS_ONLY = True first to print the basin table, then set
# TRAIN_BASINS to the outlet COMIDs kept for training (rest of PL = test).
REPORT_BASINS_ONLY = True
TRAIN_BASINS = None  # list of outlet COMIDs; None keeps all basins


# ------- Patch geometry -------------------------------------
PATCH_SIZE_PX = 256
PATCH_RES_M = 10
PATCH_SIZE_M = PATCH_SIZE_PX * PATCH_RES_M  # 2560 m


# ------- Label rasterization --------------------------------
LEVEE_BUFFER_M = 15


# ------- Negative sampling ----------------------------------
RIVER_BUFFER_M = 500  # corridor half-width around large reaches
NEG_EXCLUSION_BUFFER_M = 100  # min distance of a negative from any levee
NEG_POS_RATIO = 3  # negatives per positive
NEG_TRIES_FACTOR = 40  # rejection-sampling budget = ratio * n_pos * this


# ------- DSM derivatives ------------------------------------
TPI_RADII_PX = [5, 10, 15]


# ------- Reproducibility ------------------------------------
SEED = 42


# Channel order written into each .npz (label added separately)
CHANNEL_KEYS = [
    "dsm",
    "tpi_r5",
    "tpi_r10",
    "tpi_r15",
    "canopy_height",
    "canopy_height_sd",
    "water",
]


# ============================================================
# VECTOR INPUTS
# ============================================================


def load_bdot_levees(path, target_epsg):
    gdf = Vector.load_vector(path, target_epsg=target_epsg)
    return Vector.drop_empty_geometries(gdf)


def load_merit_reaches(path, bbox_wgs84, target_epsg):
    """Read MERIT reaches clipped to the Poland bbox (file CRS), reproject."""
    return Vector.load_vector(path, bbox=bbox_wgs84, target_epsg=target_epsg)


# ============================================================
# LEVEE SEGMENTATION
# ============================================================


def segment_line(line, segment_length, min_length):
    if line.length < min_length:
        return []
    n_segments = max(1, round(line.length / segment_length))
    actual_length = line.length / n_segments
    segments = []
    for i in range(n_segments):
        seg = substring(line, i * actual_length, (i + 1) * actual_length)
        if seg.length > 0:
            segments.append(seg)
    return segments


def segment_levees(gdf_levees, segment_length, min_length):
    records = []
    for idx, row in gdf_levees.iterrows():
        geom = row.geometry
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            continue
        for part in parts:
            for seg in segment_line(part, segment_length, min_length):
                records.append(
                    {"source_idx": idx, "geometry": seg, "length_m": seg.length}
                )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf_levees.crs)
    gdf["segment_id"] = range(len(gdf))
    return gdf


def assign_reach_to_segments(gdf_segments, gdf_reaches, max_dist):
    """Attach the nearest MERIT reach attributes to each segment; keep the
    highest-uparea reach within max_dist to handle parallel reaches."""
    cols = [COMID_COL, UPAREA_COL, NEXTDOWN_COL, "geometry"]
    joined = gpd.sjoin(
        gdf_segments,
        gdf_reaches[cols],
        how="left",
        predicate="dwithin",
        distance=max_dist,
    )
    joined = joined[joined[COMID_COL].notna()].copy()
    joined = joined.sort_values(UPAREA_COL, ascending=False)
    joined = joined.drop_duplicates(subset="segment_id", keep="first")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


# ============================================================
# DRAINAGE BASINS
# ============================================================


def trace_basins(gdf_reaches):
    """Group reaches into basins by following NextDownID to a terminal outlet
    (NextDownID <= 0 or outside the clipped set). Returns {COMID -> outlet COMID}."""
    comids = (
        pd.to_numeric(gdf_reaches[COMID_COL], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .tolist()
    )
    nxts = (
        pd.to_numeric(gdf_reaches[NEXTDOWN_COL], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .tolist()
    )
    next_of = dict(zip(comids, nxts))
    comid_set = set(comids)

    basin_of = {}
    for start in comids:
        path, seen, cur = [], set(), int(start)
        while True:
            if cur in basin_of:
                outlet = basin_of[cur]
                break
            if cur in seen:  # cycle in data, stop here
                outlet = cur
                break
            seen.add(cur)
            path.append(cur)
            nd = next_of.get(cur)
            if nd is None or nd <= 0 or nd not in comid_set:
                outlet = cur
                break
            cur = nd
        for c in path:
            basin_of[c] = outlet
    return basin_of


def basin_outlet_locations(gdf_reaches, basin_ids):
    """Representative outlet location per basin, for the report."""
    locs = {}
    for bid in basin_ids:
        sub = gdf_reaches[gdf_reaches[COMID_COL].astype(np.int64) == int(bid)]
        if len(sub) > 0:
            pt = sub.geometry.iloc[0].representative_point()
            locs[bid] = (round(pt.x, 1), round(pt.y, 1))
        else:
            locs[bid] = (None, None)
    return locs


def filter_and_tag(gdf_segments, basin_of, min_uparea):
    gdf = gdf_segments[gdf_segments[UPAREA_COL] >= min_uparea].copy()
    gdf["basin_id"] = (
        pd.to_numeric(gdf[COMID_COL], errors="coerce").astype("Int64").map(basin_of)
    )
    gdf = gdf[gdf["basin_id"].notna()].copy()
    gdf["basin_id"] = gdf["basin_id"].astype(np.int64)
    return gdf


def print_basin_report(gdf_segments, gdf_reaches, min_uparea, output_dir, basin_of_all):
    """Print and save a per-basin summary used to choose the training subset."""
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= min_uparea].copy()
    big["basin_id"] = (
        pd.to_numeric(big[COMID_COL], errors="coerce").astype("Int64").map(basin_of_all)
    )

    rows = []
    for bid, seg_sub in gdf_segments.groupby("basin_id"):
        reaches_sub = big[big["basin_id"] == bid]
        rows.append(
            {
                "basin_id": int(bid),
                "n_positive_segments": len(seg_sub),
                "n_reaches_ge_thresh": len(reaches_sub),
                "levee_length_km": round(seg_sub["length_m"].sum() / 1000.0, 1),
            }
        )
    report = pd.DataFrame(rows).sort_values("n_positive_segments", ascending=False)

    locs = basin_outlet_locations(gdf_reaches, report["basin_id"].tolist())
    report["outlet_x"] = report["basin_id"].map(lambda b: locs[b][0])
    report["outlet_y"] = report["basin_id"].map(lambda b: locs[b][1])

    print(
        "\n================ BASIN REPORT (uparea >= "
        f"{min_uparea} km^2) ================"
    )
    print(report.to_string(index=False))
    print(
        f"\nTotal basins: {len(report)} | "
        f"total positive segments: {int(report['n_positive_segments'].sum())}"
    )
    print("Pick TRAIN_BASINS from the basin_id column; the rest of PL is held out.\n")

    report.to_csv(output_dir / "basin_report.csv", index=False)
    return report


# ============================================================
# PATCH CENTERS
# ============================================================


def generate_positive_centers(gdf_segments):
    gdf = gdf_segments.copy()
    gdf["geometry"] = gdf.geometry.interpolate(0.5, normalized=True)
    gdf["patch_type"] = "positive"
    gdf["patch_id"] = [f"pos_{i:07d}" for i in range(len(gdf))]
    return gdf


def build_corridor(gdf_reaches, min_uparea, buffer_m, basin_of=None, train_basins=None):
    """Union of large reaches buffered by buffer_m, optionally train basins only."""
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= min_uparea].copy()
    if train_basins is not None and basin_of is not None:
        big["basin_id"] = (
            pd.to_numeric(big[COMID_COL], errors="coerce").astype("Int64").map(basin_of)
        )
        big = big[big["basin_id"].isin(set(train_basins))]
    if len(big) == 0:
        raise RuntimeError("No reaches in corridor (check threshold / train basins).")
    return unary_union(big.geometry.buffer(buffer_m).tolist())


def sample_corridor_negatives(corridor, exclusion, n, rng, tries_factor):
    """Uniformly sample n points inside the corridor, outside the exclusion zone."""
    minx, miny, maxx, maxy = corridor.bounds
    pc, pe = prep(corridor), prep(exclusion)
    pts, tries, cap = [], 0, max(1, n) * tries_factor
    while len(pts) < n and tries < cap:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if pc.contains(p) and not pe.contains(p):
            pts.append(p)
    return pts


def generate_negative_centers(
    n_positive,
    corridor,
    gdf_levees,
    exclusion_buffer,
    ratio,
    rng,
    tries_factor,
    gdf_reaches,
    basin_of,
):
    """Sample corridor-wide negatives away from any levee; negatives on water
    are kept as hard "water, not levee" examples."""
    # Only levees near the corridor can affect sampling inside it
    near = gdf_levees[gdf_levees.intersects(corridor.buffer(exclusion_buffer))]
    if len(near) > 0:
        exclusion = unary_union(near.geometry.buffer(exclusion_buffer).tolist())
    else:
        exclusion = Polygon()

    n_neg = int(round(n_positive * ratio))
    pts = sample_corridor_negatives(corridor, exclusion, n_neg, rng, tries_factor)

    gdf_neg = gpd.GeoDataFrame(
        {"patch_type": ["negative"] * len(pts)},
        geometry=pts,
        crs=f"EPSG:{TARGET_EPSG}",
    )

    # Attach basin_id via the nearest large reach (used for split grouping)
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= MIN_UPAREA_KM2][
        [COMID_COL, "geometry"]
    ].copy()
    if len(gdf_neg) > 0 and len(big) > 0:
        nearest = gpd.sjoin_nearest(gdf_neg, big, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        gdf_neg["comid"] = nearest[COMID_COL]
        gdf_neg["basin_id"] = gdf_neg["comid"].map(
            lambda c: basin_of.get(int(c)) if pd.notna(c) else np.nan
        )
    else:
        gdf_neg["comid"] = np.nan
        gdf_neg["basin_id"] = np.nan

    gdf_neg["patch_id"] = [f"neg_{i:07d}" for i in range(len(gdf_neg))]
    return gdf_neg


# ============================================================
# RASTER CHANNEL EXTRACTION
# ============================================================


def extract_channels(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds, water_ds):
    """Return dict of channel arrays for one patch center, None if outside DSM."""
    if not Raster.point_in_raster(dsm_ds, cx, cy):
        return None

    half = PATCH_SIZE_M / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)

    dsm = Raster.read_window(dsm_ds, bbox, PATCH_SIZE_PX, "bilinear")
    if np.all(dsm == 0):  # fully out-of-bounds fill
        return None

    canopy = Raster.read_window(canopy_ds, bbox, PATCH_SIZE_PX, "bilinear")
    canopy_sd = Raster.read_window(canopy_sd_ds, bbox, PATCH_SIZE_PX, "bilinear")
    water = Raster.read_window(water_ds, bbox, PATCH_SIZE_PX, "nearest")
    water = (water > 0.5).astype(np.float32)  # keep strict 0/1

    channels = {
        "dsm": dsm.astype(np.float32),
        "canopy_height": canopy.astype(np.float32),
        "canopy_height_sd": canopy_sd.astype(np.float32),
        "water": water,
    }
    for r in TPI_RADII_PX:
        channels[f"tpi_r{r}"] = Raster.compute_tpi(dsm, r)
    return channels


# ============================================================
# BUILD PATCHES + SAVE
# ============================================================


def build_and_save(gdf_centers, gdf_levees, output_dir):
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    levees_sindex = gdf_levees.sindex

    dsm_ds = Raster.open_raster(COPDEM_TIFF)
    canopy_ds = Raster.open_raster(CANOPY_HEIGHT_TIFF)
    canopy_sd_ds = Raster.open_raster(CANOPY_HEIGHT_SD_TIFF)
    water_ds = Raster.open_raster(WATER_TIFF)

    metadata_rows = []

    for _, row in tqdm(
        gdf_centers.iterrows(), total=len(gdf_centers), desc="Building patches"
    ):
        cx, cy = row.geometry.x, row.geometry.y

        channels = extract_channels(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds, water_ds)
        if channels is None:
            continue

        gt = Raster.patch_geotransform(cx, cy, PATCH_SIZE_M, PATCH_RES_M)

        # Label: levees intersecting the patch bbox, buffered, rasterized
        half = PATCH_SIZE_M / 2.0
        patch_bounds = (cx - half, cy - half, cx + half, cy + half)
        cand_idx = list(levees_sindex.intersection(patch_bounds))
        buffered = []
        if cand_idx:
            patch_box = box(*patch_bounds)
            cand = gdf_levees.iloc[cand_idx]
            relevant = cand[cand.geometry.intersects(patch_box.buffer(LEVEE_BUFFER_M))]
            buffered = [g.buffer(LEVEE_BUFFER_M) for g in relevant.geometry]
        label = Raster.rasterize_geometries(
            buffered, gt, PATCH_SIZE_PX, TARGET_EPSG, all_touched=True
        )

        patch_id = row["patch_id"]
        out = {k: channels[k] for k in CHANNEL_KEYS}
        out["label"] = label
        np.savez_compressed(patches_dir / f"{patch_id}.npz", **out)

        metadata_rows.append(
            {
                "patch_id": patch_id,
                "patch_type": row["patch_type"],
                "category": row.get("category", None),
                "center_x": cx,
                "center_y": cy,
                "source_idx": row.get("source_idx", None),
                "uparea": row.get(UPAREA_COL, np.nan),
                "comid": row.get("comid", None),
                "basin_id": row.get("basin_id", None),
                "n_label_px": int(label.sum()),
                "npz_path": str(
                    (patches_dir / f"{patch_id}.npz").relative_to(output_dir)
                ),
            }
        )

    dsm_ds = canopy_ds = canopy_sd_ds = water_ds = None

    meta = pd.DataFrame(metadata_rows)
    meta.to_csv(output_dir / "patches_metadata.csv", index=False)
    print(
        f"\nSaved {len(meta)} patches "
        f"({(meta['patch_type'] == 'positive').sum()} pos, "
        f"{(meta['patch_type'] == 'negative').sum()} neg) to {patches_dir}"
    )
    return meta


def categorize_uparea(u):
    """Coarse size class kept for per-category reporting downstream."""
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    return "L" if u >= 10000 else "M"


# ============================================================
# MAIN
# ============================================================


def main():
    rng = np.random.default_rng(SEED)

    # Vector inputs
    gdf_levees = load_bdot_levees(BDOT_GPKG, TARGET_EPSG)
    gdf_reaches = load_merit_reaches(MERIT_GPKG, POLAND_BBOX_WGS84, TARGET_EPSG)

    # Segment levees + attach the nearest reach
    gdf_segments = segment_levees(gdf_levees, SEGMENT_LENGTH_M, MIN_LEVEE_LENGTH_M)
    gdf_segments = assign_reach_to_segments(
        gdf_segments, gdf_reaches, MAX_DIST_TO_REACH_M
    )

    # Basins + uparea filter + basin tag
    basin_of = trace_basins(gdf_reaches)
    gdf_segments = filter_and_tag(gdf_segments, basin_of, MIN_UPAREA_KM2)
    gdf_segments["category"] = gdf_segments[UPAREA_COL].apply(categorize_uparea)

    if len(gdf_segments) == 0:
        raise RuntimeError("No segments left after uparea filter; lower MIN_UPAREA_KM2?")

    print_basin_report(gdf_segments, gdf_reaches, MIN_UPAREA_KM2, OUTPUT_DIR, basin_of)
    Vector.save_vector(gdf_segments, OUTPUT_DIR / "segments_filtered.gpkg")

    if REPORT_BASINS_ONLY:
        print(
            "REPORT_BASINS_ONLY is True -> stopping. "
            "Set TRAIN_BASINS and REPORT_BASINS_ONLY=False to generate patches."
        )
        return

    # Geographic hold-out: keep only training basins
    if TRAIN_BASINS is not None:
        before = len(gdf_segments)
        gdf_segments = gdf_segments[
            gdf_segments["basin_id"].isin(set(TRAIN_BASINS))
        ].copy()
        print(
            f"Geographic hold-out: kept {len(gdf_segments)}/{before} positive segments "
            f"in basins {TRAIN_BASINS}"
        )
        if len(gdf_segments) == 0:
            raise RuntimeError("TRAIN_BASINS selected no segments; check basin ids.")

    # Patch centers
    gdf_pos = generate_positive_centers(gdf_segments)

    corridor = build_corridor(
        gdf_reaches, MIN_UPAREA_KM2, RIVER_BUFFER_M, basin_of, TRAIN_BASINS
    )
    gdf_neg = generate_negative_centers(
        len(gdf_pos),
        corridor,
        gdf_levees,
        NEG_EXCLUSION_BUFFER_M,
        NEG_POS_RATIO,
        rng,
        NEG_TRIES_FACTOR,
        gdf_reaches,
        basin_of,
    )
    print(
        f"Positives: {len(gdf_pos)} | negatives: {len(gdf_neg)} "
        f"(target ratio {NEG_POS_RATIO}:1)"
    )

    Vector.save_vector(gdf_pos, OUTPUT_DIR / "patch_centers_positive.gpkg")
    Vector.save_vector(gdf_neg, OUTPUT_DIR / "patch_centers_negative.gpkg")

    # Combine centers
    keep_cols = [
        "geometry",
        "patch_type",
        "patch_id",
        "category",
        "source_idx",
        UPAREA_COL,
        COMID_COL,
        "comid",
        "basin_id",
    ]
    pos = gdf_pos.reindex(columns=[c for c in keep_cols if c in gdf_pos.columns])
    neg = gdf_neg.reindex(columns=[c for c in keep_cols if c in gdf_neg.columns])
    gdf_all = gpd.GeoDataFrame(
        pd.concat([pos, neg], ignore_index=True),
        geometry="geometry",
        crs=f"EPSG:{TARGET_EPSG}",
    )

    # Coalesce positives' COMID and negatives' comid into one column so every
    # patch carries a reach id for the train/val/test grouping downstream
    if "comid" not in gdf_all.columns:
        gdf_all["comid"] = np.nan
    if COMID_COL in gdf_all.columns:
        gdf_all["comid"] = gdf_all["comid"].fillna(gdf_all[COMID_COL])

    # Extract channels, rasterize labels, save
    build_and_save(gdf_all, gdf_levees, OUTPUT_DIR)


if __name__ == "__main__":
    main()
