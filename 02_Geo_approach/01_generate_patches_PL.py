"""
Levee Detection - Patch Generation Pipeline (v0.2)
==================================================

Generates training patches for levee detection from Copernicus DSM, restricted
to large rivers (upstream area >= MIN_UPAREA_KM2) and to a geographic subset of
drainage basins, so the model is NOT trained on the whole country and the held
out basins give an honest within-country generalization test.

Changes vs v0.1:
  - GDAL + GeoPandas(pyogrio) everywhere (no rasterio / fiona).
  - Raster channels read through the SHARED patch_io.read_window, and TPI through
    patch_io.compute_tpi, so patches match the inference pipeline bit-for-bit.
  - Hard upstream-area filter (MIN_UPAREA_KM2) instead of S/M/L sampling.
  - Drainage basins traced from NextDownID; a per-basin report is printed and the
    training subset is chosen via TRAIN_BASINS (geographic hold-out).
  - Corridor-wide negatives: sampled anywhere in the large-river corridor
    (reaches buffered by RIVER_BUFFER_M), excluded only near ANY levee. Negatives
    that fall on water are KEPT on purpose (hard "water, not levee" examples).
  - 7th channel: binary water mask (from prepare_water_mask.py), read NEAREST.
  - Metadata gains a basin_id column.

Two-step use:
  1. REPORT_BASINS_ONLY = True  -> prints the basin table and exits (pick basins).
  2. Set TRAIN_BASINS = [...] and REPORT_BASINS_ONLY = False -> generates patches.

Author:   Jakub Zapletal
Date:     2026-06-18
Version:  0.2
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

from osgeo import gdal, ogr, osr, gdalconst
gdal.UseExceptions()

import patch_io   # shared read_window + compute_tpi + patch_geotransform


# ============================================================
# CONFIG
# ============================================================

# ------- Input paths ----------------------------------------
BDOT_GPKG             = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg"
MERIT_GPKG            = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\riv_pfaf_2x_MERIT_Hydro_v07_Basin_flip.gpkg"
COPDEM_TIFF           = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif"
CANOPY_HEIGHT_TIFF    = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_2180.tif"
CANOPY_HEIGHT_SD_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\data\CanopyHeight\Poland\reprojected\ETH_GlobalCanopyHeight_10m_2020_Poland_Map_SD_2180.tif"
# Binary water mask produced by prepare_water_mask.py (same grid as the DSM).
WATER_TIFF            = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\source\pl\water_mask_pl.tif"

OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_PL")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------- Spatial reference ----------------------------------
TARGET_CRS        = "EPSG:2180"            # PUWG 1992
TARGET_EPSG       = 2180
POLAND_BBOX_WGS84 = (14.0, 49.0, 24.2, 55.0)   # MERIT is read in its native CRS (WGS84) then reprojected


# ------- MERIT reach attribute columns ----------------------
COMID_COL    = "COMID"
NEXTDOWN_COL = "NextDownID"
UPAREA_COL   = "uparea"


# ------- River / segment selection --------------------------
MIN_UPAREA_KM2      = 2000     # keep only segments on rivers at least this large
SEGMENT_LENGTH_M    = 500
MIN_LEVEE_LENGTH_M  = 100
MAX_DIST_TO_REACH_M = 500      # tightened from 1000 to reduce uparea mis-inheritance at the threshold


# ------- Geographic hold-out (by drainage basin) ------------
# First run with REPORT_BASINS_ONLY = True to print the basin table, then set
# TRAIN_BASINS to the outlet COMIDs you want for TRAINING (rest of PL = test).
REPORT_BASINS_ONLY = True
TRAIN_BASINS       = None      # e.g. [12345, 67890]; None = keep all basins


# ------- Patch geometry -------------------------------------
PATCH_SIZE_PX = 256
PATCH_RES_M   = 10
PATCH_SIZE_M  = PATCH_SIZE_PX * PATCH_RES_M    # 2560 m


# ------- Label rasterization --------------------------------
LEVEE_BUFFER_M = 15


# ------- Corridor-wide negative sampling --------------------
RIVER_BUFFER_M         = 500   # corridor half-width around large reaches (matches inference)
NEG_EXCLUSION_BUFFER_M = 100   # negatives must be at least this far from ANY levee
NEG_POS_RATIO          = 3     # negatives per positive
NEG_TRIES_FACTOR       = 40    # rejection-sampling attempt budget = ratio * n_pos * this


# ------- DSM derivatives ------------------------------------
TPI_RADII_PX = [5, 10, 15]


# ------- Reproducibility ------------------------------------
SEED = 42


# Channel order written into each .npz (label added separately)
CHANNEL_KEYS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd", "water"]


# ============================================================
# SECTION 2: Load BDOT levees and MERIT reaches
# ============================================================

def load_bdot_levees(path, target_crs):
    gdf = gpd.read_file(path)
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def load_merit_reaches(path, bbox_wgs84, target_crs):
    """Read MERIT reaches clipped to the Poland bbox (file is in WGS84), reproject."""
    gdf = gpd.read_file(path, bbox=bbox_wgs84)   # pyogrio interprets bbox in the file CRS
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


# ============================================================
# SECTION 3: Cut levees into fixed-length segments
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
                records.append({"source_idx": idx, "geometry": seg, "length_m": seg.length})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf_levees.crs)
    gdf["segment_id"] = range(len(gdf))
    return gdf


# ============================================================
# SECTION 4: Assign upstream area (and NextDownID) to segments
# ============================================================

def assign_reach_to_segments(gdf_segments, gdf_reaches, max_dist):
    """
    Attach the nearest large-enough MERIT reach attributes to each segment.
    Keeps the highest-uparea reach within max_dist (handles parallel reaches).
    """
    cols = [COMID_COL, UPAREA_COL, NEXTDOWN_COL, "geometry"]
    joined = gpd.sjoin(
        gdf_segments, gdf_reaches[cols],
        how="left", predicate="dwithin", distance=max_dist,
    )
    joined = joined[joined[COMID_COL].notna()].copy()
    joined = joined.sort_values(UPAREA_COL, ascending=False)
    joined = joined.drop_duplicates(subset="segment_id", keep="first")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


# ============================================================
# SECTION 5: Trace drainage basins from NextDownID
# ============================================================

def trace_basins(gdf_reaches):
    """
    Group reaches into drainage basins by following NextDownID downstream to a
    terminal outlet. A reach is an outlet when NextDownID is a terminal sentinel
    (<= 0) or points outside the clipped set (river leaves the bbox). Returns a
    dict {COMID -> basin_id (outlet COMID)}. Memoized, with a cycle guard.
    """
    comids = pd.to_numeric(gdf_reaches[COMID_COL], errors="coerce").fillna(-1).astype(np.int64).tolist()
    nxts = pd.to_numeric(gdf_reaches[NEXTDOWN_COL], errors="coerce").fillna(-1).astype(np.int64).tolist()
    next_of = dict(zip(comids, nxts))
    comid_set = set(comids)

    basin_of = {}
    for start in comids:
        path, seen, cur = [], set(), int(start)
        while True:
            if cur in basin_of:
                outlet = basin_of[cur]
                break
            if cur in seen:                      # cycle in data -> stop here
                outlet = cur
                break
            seen.add(cur)
            path.append(cur)
            nd = next_of.get(cur)
            if nd is None or nd <= 0 or nd not in comid_set:
                outlet = cur                     # terminal, or flows out of the clip
                break
            cur = nd
        for c in path:
            basin_of[c] = outlet
    return basin_of


def basin_outlet_locations(gdf_reaches, basin_ids):
    """Approximate outlet location (representative point) per basin, for the report."""
    locs = {}
    for bid in basin_ids:
        sub = gdf_reaches[gdf_reaches[COMID_COL].astype(np.int64) == int(bid)]
        if len(sub) > 0:
            pt = sub.geometry.iloc[0].representative_point()
            locs[bid] = (round(pt.x, 1), round(pt.y, 1))
        else:
            locs[bid] = (None, None)
    return locs


# ============================================================
# SECTION 6: Filter by upstream area + attach basin_id
# ============================================================

def filter_and_tag(gdf_segments, basin_of, min_uparea):
    gdf = gdf_segments[gdf_segments[UPAREA_COL] >= min_uparea].copy()
    gdf["basin_id"] = pd.to_numeric(gdf[COMID_COL], errors="coerce").astype("Int64").map(basin_of)
    gdf = gdf[gdf["basin_id"].notna()].copy()
    gdf["basin_id"] = gdf["basin_id"].astype(np.int64)
    return gdf


def print_basin_report(gdf_segments, gdf_reaches, min_uparea, output_dir, basin_of_all):
    """Print and save a per-basin summary so the training subset can be chosen."""
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= min_uparea].copy()
    big["basin_id"] = pd.to_numeric(big[COMID_COL], errors="coerce").astype("Int64").map(basin_of_all)

    rows = []
    for bid, seg_sub in gdf_segments.groupby("basin_id"):
        reaches_sub = big[big["basin_id"] == bid]
        rows.append({
            "basin_id": int(bid),
            "n_positive_segments": len(seg_sub),
            "n_reaches_ge_thresh": len(reaches_sub),
            "levee_length_km": round(seg_sub["length_m"].sum() / 1000.0, 1),
        })
    report = pd.DataFrame(rows).sort_values("n_positive_segments", ascending=False)

    locs = basin_outlet_locations(gdf_reaches, report["basin_id"].tolist())
    report["outlet_x"] = report["basin_id"].map(lambda b: locs[b][0])
    report["outlet_y"] = report["basin_id"].map(lambda b: locs[b][1])

    print("\n================ BASIN REPORT (uparea >= "
          f"{min_uparea} km^2) ================")
    print(report.to_string(index=False))
    print(f"\nTotal basins: {len(report)} | "
          f"total positive segments: {int(report['n_positive_segments'].sum())}")
    print("Pick TRAIN_BASINS from the basin_id column; the rest of PL is held out.\n")

    report.to_csv(output_dir / "basin_report.csv", index=False)
    return report


# ============================================================
# SECTION 7: Positive patch centers
# ============================================================

def generate_positive_centers(gdf_segments):
    gdf = gdf_segments.copy()
    gdf["geometry"] = gdf.geometry.interpolate(0.5, normalized=True)
    gdf["patch_type"] = "positive"
    gdf["patch_id"] = [f"pos_{i:07d}" for i in range(len(gdf))]
    return gdf


# ============================================================
# SECTION 8: Corridor-wide negative patch centers
# ============================================================

def build_corridor(gdf_reaches, min_uparea, buffer_m, basin_of=None, train_basins=None):
    """Union of large reaches buffered by buffer_m, optionally restricted to train basins."""
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= min_uparea].copy()
    if train_basins is not None and basin_of is not None:
        big["basin_id"] = pd.to_numeric(big[COMID_COL], errors="coerce").astype("Int64").map(basin_of)
        big = big[big["basin_id"].isin(set(train_basins))]
    if len(big) == 0:
        raise RuntimeError("No reaches in corridor (check threshold / train basins).")
    return unary_union(big.geometry.buffer(buffer_m).tolist())


def sample_corridor_negatives(corridor, exclusion, n, rng, tries_factor):
    """Uniformly sample n points inside the corridor but outside the exclusion zone."""
    minx, miny, maxx, maxy = corridor.bounds
    pc, pe = prep(corridor), prep(exclusion)
    pts, tries, cap = [], 0, max(1, n) * tries_factor
    while len(pts) < n and tries < cap:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if pc.contains(p) and not pe.contains(p):
            pts.append(p)
    return pts


def generate_negative_centers(n_positive, corridor, gdf_levees,
                              exclusion_buffer, ratio, rng, tries_factor,
                              gdf_reaches, basin_of):
    """
    Sample corridor-wide negatives. Exclusion is ALL levees buffered (so a
    negative never lands on a levee of any size). Negatives on water are kept.
    basin_id is attached via the nearest large reach for record-keeping.
    """
    # Exclusion = ANY levee near the corridor, buffered. Negatives are only
    # sampled inside the corridor, so levees farther than the buffer cannot
    # affect them; restricting keeps the union cheap.
    near = gdf_levees[gdf_levees.intersects(corridor.buffer(exclusion_buffer))]
    if len(near) > 0:
        exclusion = unary_union(near.geometry.buffer(exclusion_buffer).tolist())
    else:
        exclusion = Polygon()   # empty -> nothing excluded

    n_neg = int(round(n_positive * ratio))
    pts = sample_corridor_negatives(corridor, exclusion, n_neg, rng, tries_factor)

    gdf_neg = gpd.GeoDataFrame(
        {"patch_type": ["negative"] * len(pts)},
        geometry=pts, crs=f"EPSG:{TARGET_EPSG}",
    )

    # Attach basin_id via nearest large reach (record-keeping + split grouping).
    big = gdf_reaches[gdf_reaches[UPAREA_COL] >= MIN_UPAREA_KM2][[COMID_COL, "geometry"]].copy()
    if len(gdf_neg) > 0 and len(big) > 0:
        nearest = gpd.sjoin_nearest(gdf_neg, big, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]   # one row per negative (ties)
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
# SECTION 9: Raster channel extraction (shared read_window)
# ============================================================

def open_raster(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return ds


def center_in_raster(ds, cx, cy):
    gt = ds.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx = minx + ds.RasterXSize * gt[1]
    miny = maxy + ds.RasterYSize * gt[5]
    return (minx <= cx <= maxx) and (miny <= cy <= maxy)


def extract_channels(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds, water_ds):
    """Return dict of channel arrays for one patch center, or None if outside DSM."""
    if not center_in_raster(dsm_ds, cx, cy):
        return None

    half = PATCH_SIZE_M / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)

    dsm = patch_io.read_window(dsm_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    if np.all(dsm == 0):           # fully out-of-bounds fill
        return None

    canopy    = patch_io.read_window(canopy_ds,    bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    canopy_sd = patch_io.read_window(canopy_sd_ds, bbox, PATCH_SIZE_PX, gdalconst.GRA_Bilinear)
    water     = patch_io.read_window(water_ds,     bbox, PATCH_SIZE_PX, gdalconst.GRA_NearestNeighbour)
    water = (water > 0.5).astype(np.float32)   # enforce strict 0/1

    channels = {
        "dsm": dsm.astype(np.float32),
        "canopy_height": canopy.astype(np.float32),
        "canopy_height_sd": canopy_sd.astype(np.float32),
        "water": water,
    }
    for r in TPI_RADII_PX:
        channels[f"tpi_r{r}"] = patch_io.compute_tpi(dsm, r)
    return channels


# ============================================================
# SECTION 10: Per-patch label rasterization (GDAL)
# ============================================================

def rasterize_label(buffered_geoms, geotransform, size_px, srs_wkt):
    """Rasterize buffered levee polygons onto the patch grid -> uint8 0/1 array."""
    target = gdal.GetDriverByName("MEM").Create("", size_px, size_px, 1, gdal.GDT_Byte)
    target.SetGeoTransform(geotransform)
    target.SetProjection(srs_wkt)

    if buffered_geoms:
        drv = ogr.GetDriverByName("Memory")
        vds = drv.CreateDataSource("mem")
        srs = osr.SpatialReference()
        srs.ImportFromWkt(srs_wkt)
        layer = vds.CreateLayer("lev", srs, ogr.wkbPolygon)
        defn = layer.GetLayerDefn()
        for g in buffered_geoms:
            if g is None or g.is_empty:
                continue
            feat = ogr.Feature(defn)
            feat.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
            layer.CreateFeature(feat)
            feat = None
        gdal.RasterizeLayer(target, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        vds = None

    arr = target.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    target = None
    return arr


# ============================================================
# SECTION 11: Build patches + save
# ============================================================

def build_and_save(gdf_centers, gdf_levees, output_dir):
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(TARGET_EPSG)
    srs_wkt = srs.ExportToWkt()

    levees_sindex = gdf_levees.sindex

    dsm_ds       = open_raster(COPDEM_TIFF)
    canopy_ds    = open_raster(CANOPY_HEIGHT_TIFF)
    canopy_sd_ds = open_raster(CANOPY_HEIGHT_SD_TIFF)
    water_ds     = open_raster(WATER_TIFF)

    metadata_rows = []

    for _, row in tqdm(gdf_centers.iterrows(), total=len(gdf_centers), desc="Building patches"):
        cx, cy = row.geometry.x, row.geometry.y

        channels = extract_channels(cx, cy, dsm_ds, canopy_ds, canopy_sd_ds, water_ds)
        if channels is None:
            continue

        gt = patch_io.patch_geotransform(cx, cy, PATCH_SIZE_M, PATCH_RES_M)

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
        label = rasterize_label(buffered, gt, PATCH_SIZE_PX, srs_wkt)

        # Save .npz
        patch_id = row["patch_id"]
        out = {k: channels[k] for k in CHANNEL_KEYS}
        out["label"] = label
        np.savez_compressed(patches_dir / f"{patch_id}.npz", **out)

        metadata_rows.append({
            "patch_id":   patch_id,
            "patch_type": row["patch_type"],
            "category":   row.get("category", None),
            "center_x":   cx,
            "center_y":   cy,
            "source_idx": row.get("source_idx", None),
            "uparea":     row.get(UPAREA_COL, np.nan),
            "comid":      row.get("comid", None),
            "basin_id":   row.get("basin_id", None),
            "n_label_px": int(label.sum()),
            "npz_path":   str((patches_dir / f"{patch_id}.npz").relative_to(output_dir)),
        })

    dsm_ds = canopy_ds = canopy_sd_ds = water_ds = None

    meta = pd.DataFrame(metadata_rows)
    meta.to_csv(output_dir / "patches_metadata.csv", index=False)
    print(f"\nSaved {len(meta)} patches "
          f"({(meta['patch_type'] == 'positive').sum()} pos, "
          f"{(meta['patch_type'] == 'negative').sum()} neg) to {patches_dir}")
    return meta


def categorize_uparea(u):
    """Coarse label kept only so downstream per-category reporting still works."""
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    return "L" if u >= 10000 else "M"


# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(SEED)

    # Section 2: load inputs (vectors)
    gdf_levees  = load_bdot_levees(BDOT_GPKG, TARGET_CRS)
    gdf_reaches = load_merit_reaches(MERIT_GPKG, POLAND_BBOX_WGS84, TARGET_CRS)

    # Section 3-4: segment + attach nearest reach
    gdf_segments = segment_levees(gdf_levees, SEGMENT_LENGTH_M, MIN_LEVEE_LENGTH_M)
    gdf_segments = assign_reach_to_segments(gdf_segments, gdf_reaches, MAX_DIST_TO_REACH_M)

    # Section 5-6: basins + uparea filter + basin tag
    basin_of = trace_basins(gdf_reaches)
    gdf_segments = filter_and_tag(gdf_segments, basin_of, MIN_UPAREA_KM2)
    gdf_segments["category"] = gdf_segments[UPAREA_COL].apply(categorize_uparea)

    if len(gdf_segments) == 0:
        raise RuntimeError("No segments left after uparea filter; lower MIN_UPAREA_KM2?")

    # Basin report (always printed)
    print_basin_report(gdf_segments, gdf_reaches, MIN_UPAREA_KM2, OUTPUT_DIR, basin_of)
    gdf_segments.to_file(OUTPUT_DIR / "segments_filtered.gpkg", driver="GPKG")

    if REPORT_BASINS_ONLY:
        print("REPORT_BASINS_ONLY is True -> stopping. "
              "Set TRAIN_BASINS and REPORT_BASINS_ONLY=False to generate patches.")
        return

    # Geographic hold-out: keep only training basins
    if TRAIN_BASINS is not None:
        before = len(gdf_segments)
        gdf_segments = gdf_segments[gdf_segments["basin_id"].isin(set(TRAIN_BASINS))].copy()
        print(f"Geographic hold-out: kept {len(gdf_segments)}/{before} positive segments "
              f"in basins {TRAIN_BASINS}")
        if len(gdf_segments) == 0:
            raise RuntimeError("TRAIN_BASINS selected no segments; check basin ids.")

    # Section 7: positive centers
    gdf_pos = generate_positive_centers(gdf_segments)

    # Section 8: corridor-wide negatives
    corridor = build_corridor(gdf_reaches, MIN_UPAREA_KM2, RIVER_BUFFER_M, basin_of, TRAIN_BASINS)
    gdf_neg = generate_negative_centers(
        len(gdf_pos), corridor, gdf_levees,
        NEG_EXCLUSION_BUFFER_M, NEG_POS_RATIO, rng, NEG_TRIES_FACTOR,
        gdf_reaches, basin_of,
    )
    print(f"Positives: {len(gdf_pos)} | negatives: {len(gdf_neg)} "
          f"(target ratio {NEG_POS_RATIO}:1)")

    gdf_pos.to_file(OUTPUT_DIR / "patch_centers_positive.gpkg", driver="GPKG")
    gdf_neg.to_file(OUTPUT_DIR / "patch_centers_negative.gpkg", driver="GPKG")

    # Combine centers
    keep_cols = ["geometry", "patch_type", "patch_id", "category",
                 "source_idx", UPAREA_COL, COMID_COL, "comid", "basin_id"]
    pos = gdf_pos.reindex(columns=[c for c in keep_cols if c in gdf_pos.columns])
    neg = gdf_neg.reindex(columns=[c for c in keep_cols if c in gdf_neg.columns])
    gdf_all = gpd.GeoDataFrame(pd.concat([pos, neg], ignore_index=True),
                               geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")

    # Coalesce positives' COMID and negatives' comid into one 'comid' column,
    # so every patch has a reach id for the train/val/test grouping downstream.
    if "comid" not in gdf_all.columns:
        gdf_all["comid"] = np.nan
    if COMID_COL in gdf_all.columns:
        gdf_all["comid"] = gdf_all["comid"].fillna(gdf_all[COMID_COL])

    # Section 9-11: extract channels, rasterize labels, save
    build_and_save(gdf_all, gdf_levees, OUTPUT_DIR)


if __name__ == "__main__":
    main()