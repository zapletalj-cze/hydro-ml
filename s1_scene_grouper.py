"""
s1_scene_grouper.py
───────────────────
Builds coverage groups from preprocessed Sentinel-1 GeoTIFFs using a
greedy set-cover algorithm.

Input:  preprocessed GeoTIFFs produced by pyroSAR (EPSG:2180, sigma0)
Output:
  1. Tile-index GeoPackages  (asc_tile_index.gpkg, desc_tile_index.gpkg)
     → load in QGIS for visual QC; each scene is coloured by group
  2. scene_groups.json       → machine-readable input for Section 2
  3. Console summary

Algorithm
---------
Groups are built one at a time from a shared pool of scenes.
Within each group:
  1. Pick the scene that contributes the most new AOI coverage (largest
     intersection with AOI minus what the group already covers).
  2. Add it to the group; update the running group union.
  3. Repeat until group coverage >= MIN_GROUP_COVERAGE.
Scenes in the group are removed from the pool; the next group starts
fresh from the remaining scenes.

Selection at each step deliberately favours tiles with MINIMAL overlap
with the existing group — i.e. tiles that add the most NEW area.

Partial groups (pool exhausted before reaching MIN_GROUP_COVERAGE) are
included in the JSON but flagged in the tile index and console output.

Output JSON structure
---------------------
{
  "asc": {
    "grp_000": ["C:/.../scene1_VV_grd_elp.tif", ...],
    "grp_001": ["C:/.../scene4_VV_grd_elp.tif", ...],
  },
  "desc": { ... }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box
from shapely.ops import unary_union

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ASC_DIR  = Path(r'C:/data/processed/ascending')
DESC_DIR = Path(r'C:/data/processed/descending')
AOI_PATH = Path(r'C:/data/aoi/aoi.gpkg')
OUT_DIR  = Path(r'C:/data/model')

POLARIZATION     = 'VV'    # VV used for footprints; VH has identical extent
EPSG             = 2180    # PL-1992 — hardcoded, consistent with pyroSAR output

# A group is considered complete once its union covers this fraction of the AOI.
# Scenes may individually cover any fraction.
MIN_GROUP_COVERAGE = 0.80

# ══════════════════════════════════════════════════════════════════════════════
# Scene collection
# ══════════════════════════════════════════════════════════════════════════════

gdal.UseExceptions()


def collect_scenes(directory: Path, polarization: str) -> list[Path]:
    """Returns sorted list of pyroSAR output GeoTIFFs for the given polarization."""
    for pattern in [
        f'*_{polarization}_sigma0-elp.tif',
        f'*_{polarization}_gamma0-elp.tif',
        f'*_{polarization}_grd_elp.tif',
        f'*{polarization}*.tif',
    ]:
        scenes = sorted(directory.glob(pattern))
        if scenes:
            return scenes
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Tile index
# ══════════════════════════════════════════════════════════════════════════════

def read_scene_footprint(path: Path) -> tuple[float, float, float, float]:
    """
    Returns (xmin, ymin, xmax, ymax) from GeoTIFF header metadata only.
    No pixel data is read — fast even for BigTIFF scenes.
    """
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    gt    = ds.GetGeoTransform()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    xmin  = gt[0]
    ymax  = gt[3]
    xmax  = xmin + ncols * gt[1]
    ymin  = ymax + nrows * gt[5]   # gt[5] is negative for north-up rasters
    ds    = None
    return xmin, ymin, xmax, ymax


def build_tile_index(directory: Path, polarization: str) -> gpd.GeoDataFrame:
    """
    Builds a spatial tile index GeoDataFrame for all scenes in directory.
    CRS is set to EPSG:2180 — matches pyroSAR output, no reprojection needed.
    """
    scenes = collect_scenes(directory, polarization)
    if not scenes:
        raise FileNotFoundError(
            f'No {polarization} scenes found in {directory}. '
            f'Check ASC_DIR / DESC_DIR and POLARIZATION settings.'
        )
    print(f'  Found {len(scenes)} scenes in {directory.name}')

    records = []
    skipped = 0
    for path in scenes:
        try:
            xmin, ymin, xmax, ymax = read_scene_footprint(path)
            records.append({
                'path':     str(path),
                'name':     path.stem,
                'geometry': box(xmin, ymin, xmax, ymax),
            })
        except Exception as exc:
            print(f'  Warning: skipping {path.name} — {exc}')
            skipped += 1

    if skipped:
        print(f'  Skipped {skipped} scene(s) due to read errors.')

    return gpd.GeoDataFrame(records, crs=f'EPSG:{EPSG}')


# ══════════════════════════════════════════════════════════════════════════════
# Greedy set-cover grouping
# ══════════════════════════════════════════════════════════════════════════════

def build_coverage_groups(
    tile_index: gpd.GeoDataFrame,
    aoi_geom,
    min_coverage: float = MIN_GROUP_COVERAGE,
) -> tuple[list[list[int]], list[int]]:
    """
    Builds coverage groups via greedy set cover.

    Each group is grown by repeatedly selecting the scene from the
    remaining pool that adds the most new AOI area to the group:

        gain(i) = area( geoms[i] ∩ AOI ) - area( geoms[i] ∩ group_union ∩ AOI )

    This is equivalent to preferring tiles with minimal overlap with the
    current group contents.  Ties are broken by scene index (stable sort).

    A group is closed once its union covers >= min_coverage of the AOI,
    or when no remaining scene adds any new coverage.  All scenes in a
    closed group are removed from the pool.

    Returns
    -------
    groups : list[list[int]]
        Each inner list contains the row indices (into tile_index) of one
        group, in the order they were added.
    partial : list[int]
        Row indices of scenes that could not be assigned to any complete
        group (pool exhausted before reaching min_coverage in the last
        group).  These are included as a final partial group if non-empty.
    """
    geoms    = list(tile_index.geometry)
    aoi_area = aoi_geom.area

    # Pre-compute each scene's intersection with the AOI (used repeatedly).
    # Stored as Shapely geometry so we can re-intersect with group_union later.
    aoi_intersections = [g.intersection(aoi_geom) for g in geoms]
    aoi_inter_areas   = [g.area for g in aoi_intersections]

    remaining = list(range(len(geoms)))   # indices into tile_index
    groups: list[list[int]] = []

    while remaining:
        group: list[int]  = []
        group_union       = None   # Shapely geometry, grown incrementally
        group_cov         = 0.0

        while group_cov < min_coverage:
            # Score every remaining scene by new AOI area it would add
            best_idx  = None
            best_gain = -1.0

            for i in remaining:
                if aoi_inter_areas[i] == 0:
                    continue   # scene does not intersect AOI at all — skip

                if group_union is None:
                    gain = aoi_inter_areas[i]
                else:
                    already = aoi_intersections[i].intersection(group_union).area
                    gain    = aoi_inter_areas[i] - already

                if gain > best_gain:
                    best_gain = gain
                    best_idx  = i

            if best_idx is None or best_gain <= 0:
                break   # no remaining scene adds new coverage — stop growing

            # Add best scene to group
            group.append(best_idx)
            remaining.remove(best_idx)

            if group_union is None:
                group_union = geoms[best_idx]
            else:
                group_union = group_union.union(geoms[best_idx])

            group_cov = group_union.intersection(aoi_geom).area / aoi_area

        if group:
            groups.append(group)

    # Any leftover scenes that produced no complete group stay as partial
    # (remaining is empty at this point because all scenes were consumed
    #  inside the while-loop; the last group may be partial).
    return groups, []


# ══════════════════════════════════════════════════════════════════════════════
# AOI helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_aoi(aoi_path: Path):
    """Returns AOI as a single dissolved Shapely geometry in EPSG:2180."""
    aoi = gpd.read_file(aoi_path, engine='pyogrio')
    if aoi.crs and aoi.crs.to_epsg() != EPSG:
        aoi = aoi.to_crs(epsg=EPSG)
    return unary_union(aoi.geometry)


def coverage_fraction(geoms: list, aoi_geom) -> float:
    """Fraction of the AOI area covered by the union of geoms."""
    if not geoms:
        return 0.0
    covered = unary_union(geoms).intersection(aoi_geom).area
    return covered / aoi_geom.area if aoi_geom.area > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def process_direction(
    directory: Path,
    direction: str,
    aoi_geom,
) -> tuple[dict[str, list[str]], gpd.GeoDataFrame]:
    """
    Builds tile index, runs greedy coverage grouping, annotates the
    GeoDataFrame, and returns (group_dict, annotated_tile_index).
    """
    print(f'\n── {direction.upper()} ─────────────────────────────────────────────')

    tile_index = build_tile_index(directory, POLARIZATION)

    print(f'  Building coverage groups (min AOI coverage: {MIN_GROUP_COVERAGE * 100:.0f}%)...')
    groups, _ = build_coverage_groups(tile_index, aoi_geom)
    print(f'  Groups built : {len(groups)}')
    print()

    all_geoms    = list(tile_index.geometry)
    group_labels = ['unassigned'] * len(tile_index)
    group_dict: dict[str, list[str]] = {}

    for g_idx, row_indices in enumerate(groups):
        label  = f'grp_{g_idx:03d}'
        geoms  = [all_geoms[i] for i in row_indices]
        cov    = coverage_fraction(geoms, aoi_geom)
        ext    = unary_union(geoms).bounds
        paths  = tile_index.iloc[row_indices]['path'].tolist()
        is_partial = cov < MIN_GROUP_COVERAGE

        tile_label = f'{label}_partial' if is_partial else label
        for i in row_indices:
            group_labels[i] = tile_label

        group_dict[label] = paths

        status = f'⚠  PARTIAL {cov * 100:5.1f}%' if is_partial else f'   {cov * 100:5.1f}%'
        print(
            f'  {label} : {status}  '
            f'{len(paths):3d} scenes  '
            f'x: {ext[0]:.0f} – {ext[2]:.0f}  '
            f'y: {ext[1]:.0f} – {ext[3]:.0f}'
        )

    tile_index = tile_index.copy()
    tile_index['group'] = group_labels

    # Overall coverage from all groups combined
    all_assigned = [i for grp in groups for i in grp]
    total_cov = coverage_fraction([all_geoms[i] for i in all_assigned], aoi_geom)
    print(f'\n  Combined AOI coverage : {total_cov * 100:.1f}%')
    if total_cov < 0.99:
        print(f'  ⚠  Coverage < 99% — possible gap in scene archive.')

    return group_dict, tile_index


def main(dry_run: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('s1_scene_grouper')
    print(f'  AOI      : {AOI_PATH}')
    print(f'  ASC dir  : {ASC_DIR}')
    print(f'  DESC dir : {DESC_DIR}')
    print(f'  Out dir  : {OUT_DIR}')
    print(f'  Dry run  : {dry_run}')

    aoi_geom = load_aoi(AOI_PATH)
    print(f'\n  AOI area : {aoi_geom.area / 1e6:.1f} km²')

    all_groups: dict[str, dict[str, list[str]]] = {}

    for direction, directory in [('asc', ASC_DIR), ('desc', DESC_DIR)]:
        group_dict, tile_index = process_direction(directory, direction, aoi_geom)
        all_groups[direction] = group_dict

        gpkg_path = OUT_DIR / f'{direction}_tile_index.gpkg'
        tile_index.to_file(gpkg_path, driver='GPKG', engine='pyogrio')
        print(f'  Tile index -> {gpkg_path}')

    # ── Grand summary ──────────────────────────────────────────────────────
    print('\n══ Summary ══════════════════════════════════════════════════')
    total_groups = 0
    for direction, groups in all_groups.items():
        n_scenes = sum(len(v) for v in groups.values())
        print(f'  {direction.upper()}: {len(groups)} groups,  {n_scenes} scenes total')
        total_groups += len(groups)
    print(f'\n  Total groups : {total_groups}')
    print('═════════════════════════════════════════════════════════════')

    json_path = OUT_DIR / 'scene_groups.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_groups, f, indent=2, ensure_ascii=False)
    print(f'\n  Grouping JSON -> {json_path}')

    if dry_run:
        print('\n  Dry-run preview (first 2 groups per direction):')
        preview: dict = {}
        for direction, groups in all_groups.items():
            items = list(groups.items())[:2]
            preview[direction] = {
                k: [Path(p).name for p in v[:3]] + (['...'] if len(v) > 3 else [])
                for k, v in items
            }
        print(json.dumps(preview, indent=2))


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build AOI-coverage groups from preprocessed S1 GeoTIFFs.'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print summary and JSON preview only — write no files.',
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
