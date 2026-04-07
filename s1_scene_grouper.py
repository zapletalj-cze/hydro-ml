"""
s1_scene_grouper.py
───────────────────
Groups preprocessed Sentinel-1 GeoTIFFs by orbit track based on
spatial footprint similarity (IoU).

Input:  preprocessed GeoTIFFs produced by pyroSAR (EPSG:2180, sigma0)
Output:
  1. Tile-index GeoPackages  (asc_tile_index.gpkg, desc_tile_index.gpkg)
     → load in QGIS for visual QC
  2. scene_groups.json       → machine-readable input for Section 2
  3. Console summary

Output JSON structure
---------------------
{
  "asc": {
    "grp_000": ["C:/.../scene1_VV_grd_elp.tif", ...],
    "grp_001": ["C:/.../scene4_VV_grd_elp.tif", ...],
    ...
  },
  "desc": {
    "grp_000": [...],
    ...
  }
}

Each group = one orbit track = scenes to be temporally averaged together.
Groups from different tracks are mosaicked after averaging.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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

POLARIZATION  = 'VV'   # VV used for footprints; VH has identical extent
EPSG          = 2180   # PL-1992 — hardcoded, consistent with pyroSAR output

# Scenes with IoU >= this threshold are treated as the same orbit track.
# Same-track Sentinel-1 repeats: IoU typically > 0.97.
# Adjacent-track overlap:        IoU typically  0.05 – 0.25.
IOU_THRESHOLD = 0.85

# Groups whose union footprint covers less than this fraction of the AOI
# are excluded from the JSON output and flagged in the tile index.
# Individual scenes within a group may have any coverage.
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
# IoU-based grouping via Union-Find
# ══════════════════════════════════════════════════════════════════════════════

def _iou(geom_a, geom_b) -> float:
    inter = geom_a.intersection(geom_b).area
    union = geom_a.union(geom_b).area
    return inter / union if union > 0 else 0.0


def group_by_footprint(
    gdf: gpd.GeoDataFrame,
    iou_threshold: float = IOU_THRESHOLD,
) -> list[list[int]]:
    """
    Groups rows of gdf by spatial footprint similarity using Union-Find.

    Two scenes are placed in the same group when their bounding-box IoU
    exceeds iou_threshold.  Grouping is transitive: if A~B and B~C then
    A, B, C all land in the same group even if IoU(A, C) < threshold.

    Returns a list of groups, each group being a sorted list of row indices
    into gdf.  Groups are sorted by the x-centroid of their union extent
    (west → east), which matches the natural left-to-right mosaic order.

    Complexity: O(n²) pairwise IoU — acceptable for n ≤ ~500 scenes.
    """
    n     = len(gdf)
    geoms = list(gdf.geometry)

    # ── Union-Find with path compression ──────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(geoms[i], geoms[j]) >= iou_threshold:
                union(i, j)

    # Collect buckets
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)

    groups = [sorted(v) for v in buckets.values()]

    # Sort groups west → east by x-centroid of their union footprint
    def x_centroid(row_indices: list[int]) -> float:
        return unary_union([geoms[i] for i in row_indices]).centroid.x

    return sorted(groups, key=x_centroid)


# ══════════════════════════════════════════════════════════════════════════════
# AOI coverage diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def load_aoi(aoi_path: Path):
    """Returns AOI as a single dissolved Shapely geometry in EPSG:2180."""
    aoi = gpd.read_file(aoi_path, engine='pyogrio')
    if aoi.crs and aoi.crs.to_epsg() != EPSG:
        aoi = aoi.to_crs(epsg=EPSG)
    return unary_union(aoi.geometry)


def coverage_fraction(geoms: list, aoi_geom) -> float:
    """Fraction of the AOI area covered by the union of geoms."""
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
    Builds tile index, groups scenes by footprint, annotates the GeoDataFrame,
    and returns (group_dict, annotated_tile_index).
    """
    print(f'\n── {direction.upper()} ─────────────────────────────────────────────')

    tile_index = build_tile_index(directory, POLARIZATION)
    groups     = group_by_footprint(tile_index)

    print(f'  Orbit tracks detected : {len(groups)}')
    print(f'  IoU threshold         : {IOU_THRESHOLD}')
    print()

    # Annotate tile index with group label
    group_labels = [''] * len(tile_index)
    for g_idx, row_indices in enumerate(groups):
        label = f'grp_{g_idx:03d}'
        for i in row_indices:
            group_labels[i] = label
    tile_index = tile_index.copy()
    tile_index['group'] = group_labels

    # Build output dict + per-group console summary
    all_geoms        = list(tile_index.geometry)
    group_dict: dict[str, list[str]] = {}
    n_excluded_groups = 0

    for g_idx, row_indices in enumerate(groups):
        label      = f'grp_{g_idx:03d}'
        paths      = tile_index.iloc[row_indices]['path'].tolist()
        geoms      = [all_geoms[i] for i in row_indices]
        cov        = coverage_fraction(geoms, aoi_geom)
        ext        = unary_union(geoms).bounds

        if cov < MIN_GROUP_COVERAGE:
            # Group does not meet coverage threshold — exclude from output
            n_excluded_groups += 1
            for i in row_indices:
                group_labels[i] = f'{label}_excluded'
            print(
                f'  {label} : ⚠  EXCLUDED  {len(paths):3d} scenes  '
                f'AOI coverage: {cov * 100:5.1f}% < {MIN_GROUP_COVERAGE * 100:.0f}%  '
                f'x: {ext[0]:.0f} – {ext[2]:.0f}'
            )
        else:
            group_dict[label] = paths
            print(
                f'  {label} :    {len(paths):3d} scenes  '
                f'AOI coverage: {cov * 100:5.1f}%  '
                f'x: {ext[0]:.0f} – {ext[2]:.0f}  '
                f'y: {ext[1]:.0f} – {ext[3]:.0f}'
            )

    if n_excluded_groups:
        print(f'\n  ⚠  Groups excluded (coverage < {MIN_GROUP_COVERAGE * 100:.0f}%): {n_excluded_groups}')

    # Combined coverage of accepted groups only
    accepted_indices = [
        i
        for g_idx, row_indices in enumerate(groups)
        if f'grp_{g_idx:03d}' in group_dict
        for i in row_indices
    ]
    accepted_geoms = [all_geoms[i] for i in accepted_indices]
    total_cov = coverage_fraction(accepted_geoms, aoi_geom) if accepted_geoms else 0.0
    print(f'\n  Combined AOI coverage (accepted groups) : {total_cov * 100:.1f}%')
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

        if not dry_run:
            gpkg_path = OUT_DIR / f'{direction}_tile_index.gpkg'
            tile_index.to_file(gpkg_path, driver='GPKG', engine='pyogrio')
            print(f'  Tile index -> {gpkg_path}')

    # ── Grand summary ──────────────────────────────────────────────────────
    print('\n══ Summary ══════════════════════════════════════════════════')
    total_groups = 0
    for direction, groups in all_groups.items():
        n_scenes = sum(len(v) for v in groups.values())
        print(f'  {direction.upper()}: {len(groups)} tracks,  {n_scenes} scenes total')
        total_groups += len(groups)
    print(f'\n  Mosaic strips after temporal averaging : {total_groups}')
    print('═════════════════════════════════════════════════════════════')

    if not dry_run:
        json_path = OUT_DIR / 'scene_groups.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_groups, f, indent=2, ensure_ascii=False)
        print(f'\n  Grouping JSON -> {json_path}')
    else:
        # Print a readable preview without full paths
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
        description='Group preprocessed S1 GeoTIFFs by orbit track (IoU).'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print summary and JSON preview only — write no files.',
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
