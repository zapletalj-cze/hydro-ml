"""
Patch-generation tools: levee segmentation, basin tracing, corridor sampling,
raster channel extraction and patch export.

Author: Jakub Zapletal
Date:   2026-04-03
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from shapely.ops import substring, unary_union
from shapely.prepared import prep
from tqdm import tqdm

from .gis import Raster


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


def assign_reach_to_segments(
    gdf_segments, gdf_reaches, max_dist, comid_col, uparea_col, nextdown_col
):
    """Attach the nearest reach attributes to each segment; keep the
    highest-uparea reach within max_dist to handle parallel reaches."""
    cols = [comid_col, uparea_col, nextdown_col, "geometry"]
    joined = gpd.sjoin(
        gdf_segments,
        gdf_reaches[cols],
        how="left",
        predicate="dwithin",
        distance=max_dist,
    )
    joined = joined[joined[comid_col].notna()].copy()
    joined = joined.sort_values(uparea_col, ascending=False)
    joined = joined.drop_duplicates(subset="segment_id", keep="first")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


# ============================================================
# DRAINAGE BASINS
# ============================================================


def trace_basins(gdf_reaches, comid_col, nextdown_col):
    """Group reaches into basins by following NextDownID to a terminal outlet
    (id <= 0 or outside the clipped set). Returns {COMID -> outlet COMID}."""
    comids = (
        pd.to_numeric(gdf_reaches[comid_col], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .tolist()
    )
    nxts = (
        pd.to_numeric(gdf_reaches[nextdown_col], errors="coerce")
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


def basin_outlet_locations(gdf_reaches, basin_ids, comid_col):
    """Representative outlet location per basin, for the report."""
    locs = {}
    for bid in basin_ids:
        sub = gdf_reaches[gdf_reaches[comid_col].astype(np.int64) == int(bid)]
        if len(sub) > 0:
            pt = sub.geometry.iloc[0].representative_point()
            locs[bid] = (round(pt.x, 1), round(pt.y, 1))
        else:
            locs[bid] = (None, None)
    return locs


def filter_and_tag(gdf_segments, basin_of, min_uparea, comid_col, uparea_col):
    gdf = gdf_segments[gdf_segments[uparea_col] >= min_uparea].copy()
    gdf["basin_id"] = (
        pd.to_numeric(gdf[comid_col], errors="coerce").astype("Int64").map(basin_of)
    )
    gdf = gdf[gdf["basin_id"].notna()].copy()
    gdf["basin_id"] = gdf["basin_id"].astype(np.int64)
    return gdf


def print_basin_report(
    gdf_segments, gdf_reaches, min_uparea, output_dir, basin_of_all, comid_col, uparea_col
):
    """Print and save a per-basin summary used to choose the training subset."""
    big = gdf_reaches[gdf_reaches[uparea_col] >= min_uparea].copy()
    big["basin_id"] = (
        pd.to_numeric(big[comid_col], errors="coerce").astype("Int64").map(basin_of_all)
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

    locs = basin_outlet_locations(gdf_reaches, report["basin_id"].tolist(), comid_col)
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
    print("Pick TRAIN_BASINS from the basin_id column; the rest is held out.\n")

    report.to_csv(output_dir / "basin_report.csv", index=False)
    return report


def categorize_uparea(u, large_threshold=10000):
    """Coarse size class kept for per-category reporting downstream."""
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    return "L" if u >= large_threshold else "M"


# ============================================================
# PATCH CENTERS
# ============================================================


def generate_positive_centers(gdf_segments):
    gdf = gdf_segments.copy()
    gdf["geometry"] = gdf.geometry.interpolate(0.5, normalized=True)
    gdf["patch_type"] = "positive"
    gdf["patch_id"] = [f"pos_{i:07d}" for i in range(len(gdf))]
    return gdf


def build_corridor(
    gdf_reaches,
    min_uparea,
    buffer_m,
    comid_col,
    uparea_col,
    basin_of=None,
    train_basins=None,
):
    """Union of large reaches buffered by buffer_m, optionally train basins only."""
    big = gdf_reaches[gdf_reaches[uparea_col] >= min_uparea].copy()
    if train_basins is not None and basin_of is not None:
        big["basin_id"] = (
            pd.to_numeric(big[comid_col], errors="coerce").astype("Int64").map(basin_of)
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
    gdf_reaches,
    basin_of,
    epsg,
    min_uparea,
    comid_col,
    uparea_col,
    exclusion_buffer,
    ratio,
    rng,
    tries_factor,
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
        crs=f"EPSG:{epsg}",
    )

    # Attach basin_id via the nearest large reach (used for split grouping)
    big = gdf_reaches[gdf_reaches[uparea_col] >= min_uparea][
        [comid_col, "geometry"]
    ].copy()
    if len(gdf_neg) > 0 and len(big) > 0:
        nearest = gpd.sjoin_nearest(gdf_neg, big, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        gdf_neg["comid"] = nearest[comid_col]
        gdf_neg["basin_id"] = gdf_neg["comid"].map(
            lambda c: basin_of.get(int(c)) if pd.notna(c) else np.nan
        )
    else:
        gdf_neg["comid"] = np.nan
        gdf_neg["basin_id"] = np.nan

    gdf_neg["patch_id"] = [f"neg_{i:07d}" for i in range(len(gdf_neg))]
    return gdf_neg


# ============================================================
# RASTER CHANNEL EXTRACTION + PATCH EXPORT
# ============================================================


def extract_channels(cx, cy, rasters, patch_size_px, patch_size_m, tpi_radii):
    """
    Channel arrays for one patch center, None if outside the DSM.
    :param rasters: dict of open datasets with keys
                    dsm, canopy_height, canopy_height_sd, water
    """
    if not Raster.point_in_raster(rasters["dsm"], cx, cy):
        return None

    half = patch_size_m / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)

    dsm = Raster.read_window(rasters["dsm"], bbox, patch_size_px, "bilinear")
    if np.all(dsm == 0):  # fully out-of-bounds fill
        return None

    canopy = Raster.read_window(rasters["canopy_height"], bbox, patch_size_px, "bilinear")
    canopy_sd = Raster.read_window(
        rasters["canopy_height_sd"], bbox, patch_size_px, "bilinear"
    )
    water = Raster.read_window(rasters["water"], bbox, patch_size_px, "nearest")
    water = (water > 0.5).astype(np.float32)  # keep strict 0/1

    channels = {
        "dsm": dsm.astype(np.float32),
        "canopy_height": canopy.astype(np.float32),
        "canopy_height_sd": canopy_sd.astype(np.float32),
        "water": water,
    }
    for r in tpi_radii:
        channels[f"tpi_r{r}"] = Raster.compute_tpi(dsm, r)
    return channels


def build_and_save(
    gdf_centers,
    gdf_levees,
    output_dir,
    raster_paths,
    channel_keys,
    patch_size_px,
    patch_res_m,
    levee_buffer_m,
    tpi_radii,
    epsg,
    uparea_col,
):
    """Extract channels, rasterize labels and save .npz patches + metadata CSV.
    :param raster_paths: dict of file paths with keys
                         dsm, canopy_height, canopy_height_sd, water
    """
    patch_size_m = patch_size_px * patch_res_m
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    levees_sindex = gdf_levees.sindex
    rasters = {k: Raster.open_raster(p) for k, p in raster_paths.items()}

    metadata_rows = []

    for _, row in tqdm(
        gdf_centers.iterrows(), total=len(gdf_centers), desc="Building patches"
    ):
        cx, cy = row.geometry.x, row.geometry.y

        channels = extract_channels(cx, cy, rasters, patch_size_px, patch_size_m, tpi_radii)
        if channels is None:
            continue

        gt = Raster.patch_geotransform(cx, cy, patch_size_m, patch_res_m)

        # Label: levees intersecting the patch bbox, buffered, rasterized
        half = patch_size_m / 2.0
        patch_bounds = (cx - half, cy - half, cx + half, cy + half)
        cand_idx = list(levees_sindex.intersection(patch_bounds))
        buffered = []
        if cand_idx:
            patch_box = box(*patch_bounds)
            cand = gdf_levees.iloc[cand_idx]
            relevant = cand[cand.geometry.intersects(patch_box.buffer(levee_buffer_m))]
            buffered = [g.buffer(levee_buffer_m) for g in relevant.geometry]
        label = Raster.rasterize_geometries(
            buffered, gt, patch_size_px, epsg, all_touched=True
        )

        patch_id = row["patch_id"]
        out = {k: channels[k] for k in channel_keys}
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
                "uparea": row.get(uparea_col, np.nan),
                "comid": row.get("comid", None),
                "basin_id": row.get("basin_id", None),
                "n_label_px": int(label.sum()),
                "npz_path": str(
                    (patches_dir / f"{patch_id}.npz").relative_to(output_dir)
                ),
            }
        )

    for k in rasters:
        rasters[k] = None

    meta = pd.DataFrame(metadata_rows)
    meta.to_csv(output_dir / "patches_metadata.csv", index=False)
    print(
        f"\nSaved {len(meta)} patches "
        f"({(meta['patch_type'] == 'positive').sum()} pos, "
        f"{(meta['patch_type'] == 'negative').sum()} neg) to {patches_dir}"
    )
    return meta
