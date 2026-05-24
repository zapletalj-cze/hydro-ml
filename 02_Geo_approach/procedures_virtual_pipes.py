"""
procedures_virtual_pipes.py
===========================

Pipeline pro tvorbu Virtual Pipes (1D pit network) pro pluviální flood
modelování. Tato verze je strukturovaná tak, aby maximalizovala čitelnost
a snadnost ladění:

* `VirtualPipes.create_virtual_pipes()` je tenký orchestrátor; veškerá
  logika je rozdělena do explicitně pojmenovaných metod.
* Generování bodů ze silnic běží **po tilech** (`DomainTiler`) — namísto
  načtení celého rozsáhlého území najednou se zpracovává v překrývajících
  se blocích. Snazší ladění, menší peak memory, lepší logy.
* Pro malá vodní tělesa, která hrozí přetížením 2D modelu, se generují
  **WB drainage links** (`generate_wb_drainage_links`) — krátké linie z
  hrany malého WB do velkého WB. Tyto linky jsou zařazeny mezi priority
  tunnels a běží zbytkem priority pipeline beze změny.
* Snap na výstupní pravidelný grid se vynucením 3×3 minimálního odstupu
  zajišťuje `GridSnapper`. Sloupcová vazba inlet → outlet
  (`VP_Network_ID`) zůstává nedotčená.
* Používají se moderní geopandas vzory: `union_all`, `dissolve`,
  `pd.concat`. Žádné `unary_union` ani `.append` geometrií.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import datetime
import shutil
import tempfile
from math import pi
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import nearest_points

from gis import Raster, Vector
from helpers import Parameters
from ifgis.raster import SampleRaster, extract_by_mask_rasterized
from lib_00_vp_processing_script import (
    clip_layer_by_domain,
    connect,
    create_builtup_area_mask,
    create_inlets_selection,
    create_new_inlets,
    create_outlets,
    extract_inlets,
    filter_depressions,
    filter_inlets,
    filter_roads,
    get_depressions,
    get_line_intersections,
    move_inlets,
    move_outlets,
    remove_outlets_buildings,
    snap_to_min,
    split_lines_points,
)


# ===========================================================================
# Grid snapping (replaces the legacy iterative shift_point_randomly loops)
# ===========================================================================
class GridSnapper:
    """
    Snap points to a regular grid and enforce minimum spacing such that
    no two points lie within a 3x3 cell neighborhood (Chebyshev distance >= 2).

    Operates on integer cell indices => O(N) total cost, deterministic,
    idempotent. Does NOT modify any attribute columns, only geometry;
    therefore preserves VP_Network_ID / OutletID linkage between inlets and
    their outlets across the snap operation.
    """

    def __init__(self, origin_x, origin_y, cell_size, epsg, max_search_rings=5):
        self.ox = float(origin_x)
        self.oy = float(origin_y)
        self.cs = float(cell_size)
        self.epsg = int(epsg)
        self.max_search_rings = int(max_search_rings)

    @classmethod
    def from_raster(cls, raster_path, cell_size, epsg, max_search_rings=5):
        gt = Raster.get_raster_info(raster_path, ["geotransform"])["geotransform"]
        return cls(gt[0], gt[3], cell_size, epsg, max_search_rings)

    def xy_to_cell(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        i = np.floor((x - self.ox) / self.cs).astype(np.int64)
        j = np.floor((self.oy - y) / self.cs).astype(np.int64)
        return i, j

    def cell_to_xy(self, i, j):
        x = self.ox + (np.asarray(i) + 0.5) * self.cs
        y = self.oy - (np.asarray(j) + 0.5) * self.cs
        return x, y

    def snap_and_deduplicate(
        self,
        gdf: gpd.GeoDataFrame,
        priority_col: Optional[str] = None,
        priority_order: Optional[list] = None,
        relocate: bool = True,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            out = gdf.copy()
            out.attrs["dropped"] = 0
            return out

        if priority_col is not None and priority_col in gdf.columns:
            if priority_order is not None:
                rank = {v: r for r, v in enumerate(priority_order)}
                gdf = gdf.assign(
                    _grid_snap_rank=gdf[priority_col].map(rank).fillna(10**9)
                )
                gdf = gdf.sort_values("_grid_snap_rank", kind="stable").drop(
                    columns="_grid_snap_rank"
                )
            else:
                gdf = gdf.sort_values(priority_col, kind="stable")
        gdf = gdf.reset_index(drop=True)

        xs = gdf.geometry.x.to_numpy()
        ys = gdf.geometry.y.to_numpy()
        i_ideal, j_ideal = self.xy_to_cell(xs, ys)

        occupied: set = set()
        out_i = np.empty(len(gdf), dtype=np.int64)
        out_j = np.empty(len(gdf), dtype=np.int64)
        keep = np.zeros(len(gdf), dtype=bool)
        max_rings = self.max_search_rings if relocate else 0

        for k in range(len(gdf)):
            ci, cj = int(i_ideal[k]), int(j_ideal[k])
            placed = False
            for ring in range(max_rings + 1):
                for di, dj in self._ring_offsets(ring):
                    ni, nj = ci + di, cj + dj
                    if self._neighborhood_free(occupied, ni, nj):
                        occupied.add((ni, nj))
                        out_i[k], out_j[k] = ni, nj
                        keep[k] = True
                        placed = True
                        break
                if placed:
                    break

        idx = np.where(keep)[0]
        out = gdf.iloc[idx].copy()
        nx, ny = self.cell_to_xy(out_i[idx], out_j[idx])
        out.geometry = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(nx, ny)],
            crs=f"EPSG:{self.epsg}",
            index=out.index,
        )
        out.attrs["dropped"] = int(len(gdf) - len(out))
        return out.reset_index(drop=True)

    @staticmethod
    def _neighborhood_free(occupied: set, ci: int, cj: int) -> bool:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if (ci + di, cj + dj) in occupied:
                    return False
        return True

    @staticmethod
    def _ring_offsets(ring: int):
        if ring == 0:
            yield (0, 0)
            return
        for di in range(-ring, ring + 1):
            yield (di, -ring)
            yield (di, ring)
        for dj in range(-ring + 1, ring):
            yield (-ring, dj)
            yield (ring, dj)


def _resample_z_after_snap(
    gdf: gpd.GeoDataFrame,
    dtm: str,
    type_col: str = "Type",
    inlet_value: str = "I",
    outlet_value: str = "O",
    network_col: str = "VP_Network_ID",
) -> gpd.GeoDataFrame:
    """
    Re-sample DTM at the post-snap geometries.
    For outlets: writes ZOut.
    For inlets:  writes ZIn, then copies the matching outlet's ZOut into the
                 inlet rows of the same VP_Network_ID.
    """
    if gdf.empty or type_col not in gdf.columns:
        return gdf
    out = gdf.copy()
    mask_o = out[type_col] == outlet_value
    mask_i = out[type_col] == inlet_value

    if mask_o.any() and "ZOut" in out.columns:
        sub = out.loc[mask_o, ["geometry"]].copy()
        sub["ZOut"] = 0
        sub = SampleRaster(
            points=sub, raster=dtm, output=None, col_name="ZOut"
        ).process_whole()
        out.loc[mask_o, "ZOut"] = sub["ZOut"].values

    if mask_i.any() and "ZIn" in out.columns:
        sub = out.loc[mask_i, ["geometry"]].copy()
        sub["ZIn"] = 0
        sub = SampleRaster(
            points=sub, raster=dtm, output=None, col_name="ZIn"
        ).process_whole()
        out.loc[mask_i, "ZIn"] = sub["ZIn"].values

    if (
        mask_o.any()
        and mask_i.any()
        and network_col in out.columns
        and "ZOut" in out.columns
    ):
        outlet_zout = (
            out.loc[mask_o, [network_col, "ZOut"]]
            .dropna(subset=[network_col])
            .drop_duplicates(subset=[network_col])
            .set_index(network_col)["ZOut"]
        )
        propagated = out.loc[mask_i, network_col].map(outlet_zout)
        idx_valid = propagated.dropna().index
        out.loc[idx_valid, "ZOut"] = propagated.loc[idx_valid].values
    return out


# ===========================================================================
# Domain tiling (block-based processing for road point generation)
# ===========================================================================
class DomainTiler:
    """
    Iterate a rectangular domain as a grid of overlapping square tiles.

    Tiles cover the full domain; each tile is expanded by `overlap` on every
    side so that linear features straddling a tile boundary are visible to
    both neighbouring tiles. Final deduplication across the overlap zones is
    delegated to the downstream `GridSnapper` / `filter_inlets` steps.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: float,
        overlap: float,
        crs,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.tile_size = float(tile_size)
        self.overlap = float(overlap)
        self.crs = crs
        self._tiles = list(self._build_tiles())

    def _build_tiles(self):
        nx = max(1, int(np.ceil((self.maxx - self.minx) / self.tile_size)))
        ny = max(1, int(np.ceil((self.maxy - self.miny) / self.tile_size)))
        for ix in range(nx):
            for iy in range(ny):
                core_minx = self.minx + ix * self.tile_size
                core_miny = self.miny + iy * self.tile_size
                core_maxx = min(core_minx + self.tile_size, self.maxx)
                core_maxy = min(core_miny + self.tile_size, self.maxy)
                buf_minx = core_minx - self.overlap
                buf_miny = core_miny - self.overlap
                buf_maxx = core_maxx + self.overlap
                buf_maxy = core_maxy + self.overlap
                buffered_bounds = (buf_minx, buf_miny, buf_maxx, buf_maxy)
                buffered_poly = box(buf_minx, buf_miny, buf_maxx, buf_maxy)
                yield (ix, iy), buffered_bounds, buffered_poly

    def iter_tiles(self):
        yield from self._tiles

    def __len__(self) -> int:
        return len(self._tiles)


# ===========================================================================
# Waterbody drainage links (NEW: small WB -> large WB)
# ===========================================================================
def generate_wb_drainage_links(
    gdf_wb: gpd.GeoDataFrame,
    small_wb_area_max: float = 15000.0,
    large_wb_area_min: float = 100000.0,
    crs=None,
) -> gpd.GeoDataFrame:
    """
    Build LineString drainage links from each "small" waterbody to its
    nearest "large" waterbody. The resulting GeoDataFrame is ready to be
    concatenated into the tunnels GeoDataFrame and processed by the priority
    pipeline (first vertex = INLET on small WB edge, last vertex = OUTLET
    inside the large WB).

    Outlet location is the large WB's `representative_point()`, which Shapely
    guarantees lies inside the polygon (unlike the centroid, which can fall
    outside concave polygons).
    """
    if crs is None:
        crs = gdf_wb.crs

    empty = gpd.GeoDataFrame(
        {"type": [], "small_wb_idx": [], "large_wb_idx": []},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    if gdf_wb is None or gdf_wb.empty:
        return empty

    work = gdf_wb[["geometry"]].copy()
    work = work.explode(ignore_index=True)
    work = work[work.geometry.is_valid & ~work.geometry.is_empty].copy()
    if work.empty:
        return empty
    work["area"] = work.geometry.area

    small = work[work["area"] <= small_wb_area_max].reset_index(drop=True)
    large = work[work["area"] >= large_wb_area_min].reset_index(drop=True)
    if small.empty or large.empty:
        return empty

    # representative_point is guaranteed inside the polygon
    large = large.assign(rep_pt=large.geometry.representative_point())

    small_indexed = small.reset_index(drop=True)
    small_indexed["small_wb_idx"] = small_indexed.index + 1

    large_indexed = large.reset_index(drop=True)
    large_indexed["large_wb_idx"] = large_indexed.index + 1
    large_for_join = large_indexed[["geometry", "large_wb_idx"]].copy()

    paired = gpd.sjoin_nearest(
        small_indexed,
        large_for_join,
        how="left",
        distance_col="_pair_dist",
    )
    paired = paired.dropna(subset=["large_wb_idx"]).copy()
    if paired.empty:
        return empty
    paired["large_wb_idx"] = paired["large_wb_idx"].astype(int)

    rep_pt_by_idx = large_indexed.set_index("large_wb_idx")["rep_pt"]

    links_geom = []
    keep_rows = []
    for _, row in paired.iterrows():
        outlet_pt: Point = rep_pt_by_idx.loc[row["large_wb_idx"]]
        try:
            boundary = row.geometry.boundary
            if boundary.is_empty:
                continue
            inlet_pt, _ = nearest_points(boundary, outlet_pt)
        except Exception:
            continue
        if inlet_pt.equals(outlet_pt):
            continue
        links_geom.append(LineString([inlet_pt, outlet_pt]))
        keep_rows.append((row["small_wb_idx"], row["large_wb_idx"]))

    if not links_geom:
        return empty

    small_idx_arr, large_idx_arr = zip(*keep_rows)
    return gpd.GeoDataFrame(
        {
            "type": ["wb_drainage"] * len(links_geom),
            "small_wb_idx": list(small_idx_arr),
            "large_wb_idx": list(large_idx_arr),
        },
        geometry=list(links_geom),
        crs=crs,
    )


# ===========================================================================
# VirtualPipes
# ===========================================================================
class VirtualPipes:
    """
    Orchestrates creation of the 1D pit (Virtual Pipe) network used as the
    storm drainage component of pluvial flood models.

    Top-level entry point is `create_virtual_pipes()`; it delegates each
    stage to a focused private method so the high-level flow stays readable.

    Constants (offsets) for IDs:
    --------------------------------
    Regular network IDs are produced by the `connect()` step at the model
    level (`outlet_id`). Priority (tunnel + WB drainage) network IDs are
    shifted by +7_000_000 so the two pools never collide. Priority inlet
    string IDs are shifted by +5_000_000.
    """

    PRIORITY_NETWORK_OFFSET = 7_000_000
    PRIORITY_INLET_ID_OFFSET = 5_000_000

    # ------------------------------------------------------------------
    # Construction and setters (public API unchanged)
    # ------------------------------------------------------------------
    def __init__(self):
        self.yaml_path = None
        self.general_parameters = None
        self.virtual_pipes_parameters = None
        self.waterbodies_parameters = None
        self.waterbodies_path = None
        self.dtm_path = None
        self.domain_name = None
        self.mask_path = None
        self.vp_processing = None
        self.results_dir = None

    def set_general_parameters(self, path_yaml):
        self.yaml_path = path_yaml
        self.general_parameters = Parameters.load_local_parameters(
            self.yaml_path, "general_parameters"
        )

    def set_virtual_pipes_parameters(self, path_yaml):
        self.yaml_path = path_yaml
        self.virtual_pipes_parameters = Parameters.load_local_parameters(
            self.yaml_path, "virtual_pipes_parameters"
        )

    def set_waterbodies_parameters(self, path_yaml):
        self.yaml_path = path_yaml
        self.waterbodies_parameters = Parameters.load_local_parameters(
            self.yaml_path, "waterbodies_parameters"
        )

    def set_projection(self):
        if self.general_parameters is None:
            raise ValueError(
                "General parameters not set. Run set_general_parameters() first."
            )
        self.projection = Parameters.get_local_parameter(
            self.general_parameters, "default_epsg_code"
        )

    def set_waterbodies_path(self, path):  self.waterbodies_path = path
    def set_rn_path(self, path):           self.rn_path = path
    def set_tunnels_path(self, path):      self.tunnels_path = path
    def set_dtm_path(self, path):          self.dtm_path = path
    def set_domain_name(self, name):       self.domain_name = name
    def set_results_dir(self, path):       self.results_dir = path
    def set_mask_path(self, path):         self.mask_path = path

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------
    def create_virtual_pipes(self):
        """
        Run the full Virtual Pipes pipeline for the configured domain.

        All intermediate files (mask, clipped layers, partial inlet/outlet
        outputs, the connect() result) live inside a per-run scratch
        directory which is deleted in the `finally` block. The only file
        left behind is the final pit network at `self._out_path`.

        Stages:
            1.  _setup ........................ validate, paths, scratch dir
            2.  _prepare_input_layers ......... clip RN/WB/roads/DTM (in-memory)
            3.  _init_grid_snapper ............ output grid configuration
            4.  _generate_depression_inlets ... lowest cells per depression
            5.  _generate_priority_points ..... tunnels + RN + WB drainage
            6.  _snap_priority ................ grid snap + Z resample
            7.  _generate_road_inlets ......... tiled point generation
            8.  _densify_in_depressions ....... extra inlets in depressions
            9.  _densify_in_parking ........... extra inlets on parking lots
            10. _generate_outlets ............. on RN and WBs
            11. _snap_general ................. inlets + outlets pre-connect
            12. _post_clean_against_priority .. remove near-priority duplicates
            13. _run_connect .................. compute inlet -> outlet links
            14. _snap_final ................... merge with priority, final snap
            15. _save_output .................. final GPKG with dtypes
        """
        print(f"\nCreating Virtual Pipes for {self.domain_name}")
        start = datetime.datetime.now()

        if not self._setup():
            return
        if self._already_done():
            print(f"\t{self.domain_name} already exists. SKIPPING")
            self._cleanup_scratch()
            return

        try:
            self._prepare_input_layers()
            self._init_grid_snapper()

            self._generate_depression_inlets()
            self._generate_priority_points()
            self._snap_priority()

            self._generate_road_inlets_tiled()
            self._densify_in_depressions()
            self._densify_in_parking()
            self._generate_outlets()

            self._snap_general()
            self._post_clean_against_priority()

            self._run_connect()
            self._snap_final()
            self._save_output()
        finally:
            self._cleanup_scratch()

        print(f"\tDone in {datetime.datetime.now() - start}")

    def _cleanup_scratch(self):
        """Remove all temp scratch files for this run, if any."""
        scratch = getattr(self, "_session_tmp", None)
        if scratch and os.path.isdir(scratch):
            try:
                shutil.rmtree(scratch, ignore_errors=True)
                print(f"\tCleaned scratch dir: {scratch}")
            except Exception as exc:
                print(f"\tWarning: scratch cleanup failed: {exc}")

    # ------------------------------------------------------------------
    # 1. Setup: validate parameters, derive paths
    # ------------------------------------------------------------------
    def _setup(self) -> bool:
        """
        Validate parameters, materialise the domain mask into scratch,
        derive working paths. Returns False if pre-conditions cannot be
        met (caller will skip run).

        Scratch layout
        --------------
        A per-run directory `vp_<domain>_<timestamp>` is created under the
        configured `temp_vp_dir`. EVERY intermediate file (mask, clipped
        layers, partial inlet/outlet outputs, the connect() result) goes
        in here and is deleted at the end of the run. Only the final pit
        network GPKG at `self._out_path` is persisted.
        """
        ogr_drivers = [
            ogr.GetDriver(i).GetName() for i in range(ogr.GetDriverCount())
        ]
        if "MEM" in ogr_drivers:
            self._memory_driver = "MEM"
        elif "Memory" in ogr_drivers:
            self._memory_driver = "Memory"
        else:
            raise RuntimeError("No in-memory vector driver found in GDAL/OGR.")

        self.projection = Parameters.get_local_parameter(
            self.general_parameters, "default_epsg_code"
        )
        if self.projection is None:
            raise ValueError("default_epsg_code is missing in general_parameters")

        in_folder = os.path.join(self.results_dir)
        self._out_folder = os.path.join(in_folder, "model", "gis")
        os.makedirs(self._out_folder, exist_ok=True)

        # Per-run scratch directory under the configured temp root.
        temp_root = self.virtual_pipes_parameters.get(
            "temp_vp_dir", r"D:\temp\VPs"
        )
        os.makedirs(temp_root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._session_tmp = os.path.join(
            temp_root, f"vp_{self.domain_name}_{stamp}"
        )
        os.makedirs(self._session_tmp, exist_ok=True)

        # The mask must be on disk because several helpers
        # (extract_by_mask_rasterized) and OGR-level operations accept
        # a path only. Goes into scratch -> removed at run end.
        self._code_file = os.path.join(
            self._session_tmp, f"2d_clip_{self.domain_name}_R.gpkg"
        )
        gdf_mask = Vector.load_vector(self.mask_path)
        if gdf_mask.crs.to_epsg() != self.projection:
            gdf_mask = gdf_mask.to_crs(epsg=self.projection)
        gdf_mask.to_file(self._code_file, driver="GPKG", index=False)
        self._gdf_mask = gdf_mask

        self._params = self._collect_required_parameters()

        self._cell_size = self._params["CELL_SIZE"]
        date = datetime.datetime.now().strftime("%y%m%d")
        self._out_name = (
            f"1d_pit_{self.domain_name}_{self._cell_size}m_{date}_P.gpkg"
        )
        self._out_path = os.path.join(self._out_folder, self._out_name)

        return os.path.isfile(self._code_file)

    def _collect_required_parameters(self) -> dict:
        """Pull all required parameters from the YAML; raise if any missing."""
        p = self.virtual_pipes_parameters
        params = {
            "WB":                   self.waterbodies_path,
            "DTM":                  self.dtm_path,
            "RN":                   self.rn_path,
            "TUNNELS":              self.tunnels_path,
            "ROADS":                p.get("roads"),
            "INDUSTRIAL_ESTATES":   p.get("industrial_estates"),
            "BUILDINGS":            p.get("buildings"),
            "PARKING":              p.get("parking"),
            "MANNING":              p.get("manning"),
            "ADDRESS_POINTS":       p.get("address_points"),
            "CELL_SIZE":            p.get("cell_size"),
            "interval_inlets":      p.get("interval_inlets"),
            "interval_outlets":     p.get("interval_outlets"),
            "FIELD_FOR_FILTERING":  p.get("field_for_filtering"),
            "SNAP_INLET_TO_LOWEST": p.get("snap_inlet_to_lowest"),
            "FILTER_BUILDINGS":     p.get("filter_buildings"),
            "filter_a":             p.get("filter_a"),
            "filter_b":             p.get("filter_b"),
            "filter_c":             p.get("filter_c"),
        }
        missing = [k for k, v in params.items() if v is None]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        return params

    def _already_done(self) -> bool:
        return os.path.isfile(self._out_path)

    # ------------------------------------------------------------------
    # 2. Prepare input layers (clip RN/WB/roads/DTM to domain)
    # ------------------------------------------------------------------
    def _prepare_input_layers(self):
        """
        Clip input vector / raster layers to the domain. All resulting
        intermediate files live in the per-run scratch directory.

        We keep three things on disk in scratch because downstream helpers
        from lib_00 accept paths only: RN, WB, and the clipped roads.
        Everything else (mask GDF, in-memory inlet/outlet GDFs, priority
        points) is held in memory until final save.
        """
        p = self._params
        self._roads_path = clip_layer_by_domain(
            p["ROADS"], self._code_file,
            os.path.join(self._session_tmp, f"roads_{self.domain_name}.gpkg"),
        )
        self._rn_path = clip_layer_by_domain(
            p["RN"], self._code_file,
            os.path.join(self._session_tmp, f"rn_{self.domain_name}.gpkg"),
        )
        gdf_rn = gpd.read_file(self._rn_path)
        if "fclass" in gdf_rn.columns:
            gdf_rn = gdf_rn[
                gdf_rn["fclass"].isin(["river", "stream"])
                | gdf_rn["fclass"].isnull()
            ]
            gdf_rn.to_file(self._rn_path, driver="GPKG", index=False)

        self._wb_path = clip_layer_by_domain(
            p["WB"], self._code_file,
            os.path.join(self._session_tmp, f"wb_{self.domain_name}.gpkg"),
        )
        self._dtm = extract_by_mask_rasterized(
            p["DTM"], self._code_file,
            os.path.join(self._session_tmp, f"dtm_{self.domain_name}.tif"),
        )
        self._dtm_geotransform = Raster.get_raster_info(
            self._dtm, ["geotransform"]
        )["geotransform"]
        self._dtm_cell_size = Raster.get_raster_info(
            self._dtm, ["cell_size"]
        )["cell_size"]
        self._dtm_dt = Raster.get_raster_info(
            self._dtm, ["data_type"]
        )["data_type"]

    # ------------------------------------------------------------------
    # 3. Init grid snapper
    # ------------------------------------------------------------------
    def _init_grid_snapper(self):
        p = self.virtual_pipes_parameters
        self._output_grid_cell_size = p.get(
            "output_grid_cell_size", self._cell_size
        )
        self._max_search_rings = p.get("snap_max_search_rings", 5)
        self._snapper = GridSnapper.from_raster(
            raster_path=self._dtm,
            cell_size=self._output_grid_cell_size,
            epsg=self.projection,
            max_search_rings=self._max_search_rings,
        )
        print(
            f"\tOutput grid: {self._output_grid_cell_size} m "
            f"(DTM origin, EPSG:{self.projection}, "
            f"max relocation = {self._max_search_rings} cells)"
        )


    # ------------------------------------------------------------------
    # 4. Depression inlets (densify in detected depressions)
    # ------------------------------------------------------------------
    def _generate_depression_inlets(self):
        """
        For each filtered depression polygon, place 1-10 inlet points at
        the lowest DTM cells inside it. The number of points scales with
        the depression area:
            <= 750 m^2  -> 1 point
            <= 1500 m^2 -> 5 points
             > 1500 m^2 -> 10 points

        Result is stored in memory as `self._gdf_inlets_depressions`.
        `filter_depressions` writes to scratch because it expects a path;
        we keep its path on `self._filtered_depressions_path` because
        `create_new_inlets` (called later) also needs a path.
        """
        print("\tExtracting depressions")
        depressions_path = get_depressions(self._dtm)
        out_depressions_path = os.path.join(
            self._session_tmp, f"depressions_{self.domain_name}.gpkg"
        )
        filtered_depressions_path = filter_depressions(
            depressions_path=depressions_path,
            out_depressions_path=out_depressions_path,
            area_threshold=250,
        )
        self._filtered_depressions_path = filtered_depressions_path

        gdf = Vector.load_vector(filtered_depressions_path)
        gdf["compactness"] = self._polsby_popper_compactness(gdf.geometry)
        gdf = gdf[(gdf.geometry.area < 30000) & (gdf["compactness"] > 0.1)]

        gdf_wb = Vector.load_vector(self._wb_path)
        gdf = gpd.sjoin(gdf, gdf_wb, how="left", predicate="intersects")
        gdf = gdf[gdf.index_right.isna()].drop(columns=["index_right"])

        bbox = gdf.total_bounds
        gdf_addr = Vector.load_vector(
            self._params["ADDRESS_POINTS"], bbox=tuple(bbox)
        )
        gdf = gpd.sjoin_nearest(gdf, gdf_addr, how="left", distance_col="dist")
        gdf = gdf[gdf["dist"] < 175]

        gdf["_geom_wkb"] = gdf.geometry.apply(lambda g: g.wkb)
        gdf = gdf.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"])
        gdf = gdf.reset_index(drop=True)

        print(
            f"\tProcessing {len(gdf)} depressions for inlet densification"
        )
        points, values = self._extract_lowest_points_per_depression(gdf)

        gdf_inlets = gpd.GeoDataFrame(
            geometry=points, crs=self.projection
        )
        gdf_inlets["dtm_value"] = values
        gdf_inlets = gdf_inlets.drop_duplicates(subset=["geometry"])
        gdf_inlets = gdf_inlets.reset_index(drop=True)
        gdf_inlets["pointid"] = gdf_inlets.index + 1
        gdf_inlets = gdf_inlets[["pointid", "geometry", "dtm_value"]]

        gdf_inlets = filter_inlets(
            gdf_inlets, cell_size=self._cell_size / 2, increase_capacity=False
        )
        gdf_inlets = filter_inlets(
            gdf_inlets, cell_size=self._cell_size, increase_capacity=False
        )

        self._gdf_inlets_depressions = gdf_inlets

    @staticmethod
    def _polsby_popper_compactness(geom_series: gpd.GeoSeries) -> pd.Series:
        """4*pi*A / P^2 -- 1.0 for a circle, < 1 for any other shape."""
        p = geom_series.length
        a = geom_series.area
        return (4 * pi * a) / (p * p)

    def _extract_lowest_points_per_depression(self, gdf_depressions):
        """
        For each depression, build a small DTM clip aligned to the grid,
        rasterise the polygon, then pick the N lowest DTM cells inside.

        Bug fix vs. previous version: np.where(mask) is computed once per
        depression instead of once per output point.
        """
        points = []
        values = []

        for _, row in gdf_depressions.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
            extent = Vector.align_extent_to_snap(
                extent, snap_raster=self._dtm, cell_size=self._cell_size,
            )
            # Pad by 50 cells so that the depression interior is not clipped
            extent = [
                extent[0] - 50 * self._cell_size,
                extent[1] + 50 * self._cell_size,
                extent[2] - 50 * self._cell_size,
                extent[3] + 50 * self._cell_size,
            ]
            bounds_snapped = [extent[0], extent[2], extent[1], extent[3]]

            dst_ds = self._make_inmemory_polygon_ds(row.geometry)

            dtm_clip = Raster.clip_by_vector_mem(
                in_raster=self._dtm,
                vector_file=dst_ds,
                epsg_out=self.projection,
                cell_size_out=self._dtm_cell_size,
                data_type=self._dtm_dt,
                CPU_AVAILABLE=4,
                no_data=0,
                bounds=bounds_snapped,
            )
            r = Raster()
            dep_raster = r.rasterize_to_new_raster(
                None, dst_ds,
                value=1, cell_size=self._dtm_cell_size, extent=extent,
                format="MEM", nodata=0, data_type=gdal.GDT_Byte,
                all_touched=False,
            )

            dtm_arr = Raster.to_array_mem(dtm_clip)
            dep_arr = Raster.to_array_mem(dep_raster)

            if not np.any(dep_arr == 1):
                dst_ds = None
                continue

            mask = dep_arr == 1
            masked_values = dtm_arr[mask]
            depression_area = row.geometry.area
            if 750 < depression_area <= 1500:
                num_points = 5
            elif depression_area > 1500:
                num_points = 10
            else:
                num_points = 1
            num_points = min(num_points, len(masked_values))

            # Hoist np.where out of the inner loop -- mask is constant here.
            rows_arr, cols_arr = np.where(mask)
            lowest_idx = np.argsort(masked_values)[:num_points]

            gt = Raster.get_raster_info_mem(dtm_clip, ["geotransform"])[
                "geotransform"
            ]
            for m_idx in lowest_idx:
                r0, c0 = int(rows_arr[m_idx]), int(cols_arr[m_idx])
                v = float(dtm_arr[r0, c0])
                x = gt[0] + c0 * gt[1] + (r0 + 1) * gt[2] + (gt[1] / 2)
                y = gt[3] + c0 * gt[4] + (r0 + 1) * gt[5] - (gt[5] / 2)
                points.append(Point(x, y))
                values.append(v)

            dst_ds = None

        return points, values

    def _make_inmemory_polygon_ds(self, geom):
        """Wrap a shapely polygon in an in-memory OGR DataSource."""
        drv = ogr.GetDriverByName(self._memory_driver)
        ds = drv.CreateDataSource("in_memory")
        srs = ogr.osr.SpatialReference()
        srs.ImportFromEPSG(self.projection)
        lyr = ds.CreateLayer("polygon", srs, geom_type=ogr.wkbPolygon)
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetGeometry(ogr.CreateGeometryFromWkt(geom.wkt))
        lyr.CreateFeature(feat)
        return ds


    # ------------------------------------------------------------------
    # 5. Priority points: tunnels, underground RN, WB drainage links
    # ------------------------------------------------------------------
    def _generate_priority_points(self):
        """
        Build the priority (tunnel-like) inlet / outlet network.

        Sources combined into a single LineString GeoDataFrame:
            * provided tunnels layer (type="tunnel")
            * underground sections of the road network (type="tunnel_rn")
            * NEW: drainage links from small to large waterbodies
              (type="wb_drainage")

        For each line, vertices are snapped to the lowest DTM cell in a
        small window, then assigned to PRIORITY_INLETS or PRIORITY_OUTLETS
        dicts. Close outlets are merged (chain collapsing). The result is
        materialised as gdf_priority_inlets and gdf_priority_outlets.
        """
        print("\tProcessing tunnels and underground RN sections")
        gdf_tunnels = self._collect_tunnel_lines()
        if gdf_tunnels.empty:
            print("\tNo tunnel-like lines to process; skipping priority stage.")
            self._gdf_priority_inlets = self._empty_priority_gdf()
            self._gdf_priority_outlets = self._empty_priority_gdf()
            return

        # ds_dtm is opened once and threaded into snap_to_min via its band
        ds_dtm = gdal.Open(self._dtm)
        band_dem = ds_dtm.GetRasterBand(1)

        priority_inlets: dict = {}
        priority_outlets: dict = {}

        for _, row in gdf_tunnels.iterrows():
            geom = row.geometry
            t = row["type"]
            if geom is None or geom.is_empty:
                print("\t\tWarning: empty geometry; skipping.")
                continue
            if geom.geom_type == "MultiLineString":
                # For tunnel_rn the original code kept the longest part
                geom = max(geom.geoms, key=lambda g: g.length)
            if geom.geom_type != "LineString":
                print(f"\t\tWarning: unsupported geom_type {geom.geom_type}; skipping.")
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            if t == "tunnel":
                self._process_tunnel_line(
                    coords, band_dem, priority_inlets, priority_outlets,
                    outlet_window=5, inlet_window=3, source_type="tunnel",
                )
            elif t == "tunnel_rn":
                self._process_tunnel_rn_line(
                    coords, band_dem, priority_inlets, priority_outlets,
                    window=5, source_type="tunnel_rn",
                )
            elif t == "wb_drainage":
                # First vertex on small WB boundary = INLET,
                # last vertex inside large WB = OUTLET. Explicit, no
                # elevation-based reordering.
                self._process_tunnel_line(
                    coords, band_dem, priority_inlets, priority_outlets,
                    outlet_window=5, inlet_window=3,
                    source_type="wb_drainage",
                )
            else:
                print(f"\t\tWarning: unknown type '{t}'; skipping.")
                continue

        ds_dtm = None  # close the GDAL dataset

        self._priority_inlets = priority_inlets
        self._priority_outlets = priority_outlets

        self._gdf_priority_inlets, self._gdf_priority_outlets = (
            self._priority_dicts_to_gdf()
        )
        self._collapse_outlet_chains()
        self._finalize_priority_attributes()

    def _empty_priority_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
            crs=f"EPSG:{self.projection}",
        )

    def _collect_tunnel_lines(self) -> gpd.GeoDataFrame:
        """
        Merge configured tunnels, underground RN sections, and waterbody
        drainage links into a single LineString GeoDataFrame in the target
        CRS, with a `type` column distinguishing the three sources.
        """
        # Provided tunnel layer (already a vector file)
        tunnels_clipped = clip_layer_by_domain(
            self._params["TUNNELS"], self._code_file,
            os.path.join(self._session_tmp, f"tunnels_{self.domain_name}.gpkg"),
        )
        gdf_tunnels = Vector.load_vector(tunnels_clipped)
        if not gdf_tunnels.empty:
            gdf_tunnels = gdf_tunnels[["geometry"]].copy()
            gdf_tunnels["type"] = "tunnel"

        # Underground sections of the road network
        gdf_rn = gpd.read_file(self._params["RN"])
        gdf_rn = gpd.sjoin(
            gdf_rn, self._gdf_mask[["geometry"]],
            how="left", predicate="within",
        )
        if "tunnel" in gdf_rn.columns:
            gdf_rn = gdf_rn[gdf_rn["tunnel"] == 1]
        elif "underground" in gdf_rn.columns:
            gdf_rn = gdf_rn[gdf_rn["underground"] == 1]
        else:
            gdf_rn = gdf_rn.iloc[0:0]
        if not gdf_rn.empty:
            gdf_rn = Vector.merge_lines_on_pseudonodes(gdf_rn)
            gdf_rn = gdf_rn[gdf_rn.geometry.length > 25].copy()
            gdf_rn = gdf_rn[["geometry"]].copy()
            gdf_rn["type"] = "tunnel_rn"

        # NEW: waterbody drainage links
        gdf_wb = Vector.load_vector(self._wb_path)
        small_max = self.virtual_pipes_parameters.get(
            "small_wb_area_max", 15000.0
        )
        large_min = self.virtual_pipes_parameters.get(
            "large_wb_area_min", 100000.0
        )
        gdf_wb_drainage = generate_wb_drainage_links(
            gdf_wb,
            small_wb_area_max=small_max,
            large_wb_area_min=large_min,
            crs=f"EPSG:{self.projection}",
        )
        # Drop bookkeeping columns that other tunnel rows lack to keep concat
        # schemas clean.
        if not gdf_wb_drainage.empty:
            gdf_wb_drainage = gdf_wb_drainage[["geometry", "type"]].copy()
        print(
            f"\tTunnel-like inputs: provided tunnels={len(gdf_tunnels)}, "
            f"underground RN={len(gdf_rn)}, "
            f"WB drainage links={len(gdf_wb_drainage)} "
            f"(small<= {small_max} m^2, large>= {large_min} m^2)"
        )

        pieces = [g for g in (gdf_tunnels, gdf_rn, gdf_wb_drainage) if not g.empty]
        if not pieces:
            return self._empty_priority_gdf().assign(type=[])
        combined = pd.concat(pieces, ignore_index=True)
        combined = gpd.GeoDataFrame(
            combined, geometry="geometry", crs=f"EPSG:{self.projection}"
        )
        return combined.reset_index(drop=True)

    def _process_tunnel_line(
        self, coords, band_dem,
        priority_inlets, priority_outlets,
        outlet_window, inlet_window, source_type,
    ):
        """Tunnel / WB-drainage: last vertex is OUTLET, others are INLETs."""
        outlet_pt_raw = Point(coords[-1])
        snapped = snap_to_min(
            band_dem, outlet_pt_raw, self._dtm_geotransform, size=outlet_window
        )
        x, y, z = snapped
        if x is None or y is None or z is None:
            return
        outlet_pt = Point(x, y)
        outlet_z = z

        closest_outlet_id = self._register_outlet(
            outlet_pt, outlet_z, priority_outlets,
        )

        for raw in coords[:-1]:
            xi, yi, zi = snap_to_min(
                band_dem, Point(raw), self._dtm_geotransform,
                size=inlet_window,
            )
            if xi is None or yi is None or zi is None:
                continue
            inlet_pt = Point(xi, yi)
            if not self._inlet_is_far_enough(inlet_pt, priority_inlets):
                print("\t\tInlet close to existing priority inlet, skipping.")
                continue
            priority_inlets[inlet_pt] = {
                "outlet_id": closest_outlet_id,
                "dtm_value_inlet": zi,
                "source_type": source_type,
            }

    def _process_tunnel_rn_line(
        self, coords, band_dem,
        priority_inlets, priority_outlets,
        window, source_type,
    ):
        """
        Underground RN: snap both endpoints; the lower-elevation end is the
        OUTLET, the higher-elevation end is the INLET. Skip if the outlet is
        not meaningfully lower than the inlet.
        """
        x0, y0, z0 = snap_to_min(
            band_dem, Point(coords[0]), self._dtm_geotransform, size=window,
        )
        xn, yn, zn = snap_to_min(
            band_dem, Point(coords[-1]), self._dtm_geotransform, size=window,
        )
        if x0 is None or xn is None:
            return
        if z0 <= zn:
            outlet_pt, outlet_z = Point(x0, y0), z0
            inlet_pt, inlet_z = Point(xn, yn), zn
        else:
            outlet_pt, outlet_z = Point(xn, yn), zn
            inlet_pt, inlet_z = Point(x0, y0), z0
        if outlet_z >= inlet_z:
            return  # no drainage gradient

        outlet_id = self._register_outlet(outlet_pt, outlet_z, priority_outlets)
        if not self._inlet_is_far_enough(inlet_pt, priority_inlets):
            return
        priority_inlets[inlet_pt] = {
            "outlet_id": outlet_id,
            "dtm_value_inlet": inlet_z,
            "source_type": source_type,
        }

    def _register_outlet(self, outlet_pt: Point, outlet_z: float,
                         priority_outlets: dict) -> int:
        """
        Return an integer key into priority_outlets for this outlet. If a
        previously-registered outlet lies within CELL_SIZE * 2.5, reuse its
        key (incremental dedup, O(N) per call). Otherwise add a new entry.
        """
        threshold = self._cell_size * 2.5
        if priority_outlets:
            keys = list(priority_outlets.keys())
            dists = [outlet_pt.distance(priority_outlets[k][0]) for k in keys]
            min_d = min(dists)
            if min_d < threshold:
                return keys[dists.index(min_d)]
            new_key = max(keys) + 1
        else:
            new_key = 0
        priority_outlets[new_key] = (outlet_pt, outlet_z)
        return new_key

    def _inlet_is_far_enough(self, inlet_pt: Point,
                              priority_inlets: dict) -> bool:
        """True if no existing priority inlet lies within 2.5 cells."""
        threshold = self._cell_size * 2.5
        if not priority_inlets:
            return True
        dists = [inlet_pt.distance(k) for k in priority_inlets.keys()]
        return min(dists) > threshold


    def _priority_dicts_to_gdf(self):
        """
        Materialise the priority_inlets and priority_outlets dicts into a
        pair of GeoDataFrames. Uses the explicit dict keys as OutletID so
        the link between an inlet and its outlet survives the conversion.

        Bug fix vs. previous version: priority_inlets stores `outlet_id`
        directly (instead of the outlet *tuple* by reference), eliminating
        the previous O(N*M) "find key by value" scan.
        """
        pi_dict = self._priority_inlets
        po_dict = self._priority_outlets

        crs = f"EPSG:{self.projection}"

        if not pi_dict and not po_dict:
            empty = self._empty_priority_gdf()
            return empty, empty

        # Inlets
        inlet_geoms = list(pi_dict.keys())
        if inlet_geoms:
            inlet_attrs = list(pi_dict.values())
            gdf_inlets = gpd.GeoDataFrame(
                {
                    "OutletID": [v["outlet_id"] for v in inlet_attrs],
                    "ZIn":      [v["dtm_value_inlet"] for v in inlet_attrs],
                    "ZOut":     [po_dict[v["outlet_id"]][1]
                                 for v in inlet_attrs],
                    "if_type":  [v.get("source_type", "tunnel")
                                 for v in inlet_attrs],
                },
                geometry=inlet_geoms,
                crs=crs,
            )
            gdf_inlets["INLET_ID"] = gdf_inlets.index + 1
        else:
            gdf_inlets = self._empty_priority_gdf()
            gdf_inlets["OutletID"] = pd.Series(dtype=int)
            gdf_inlets["ZIn"]     = pd.Series(dtype=float)
            gdf_inlets["ZOut"]    = pd.Series(dtype=float)
            gdf_inlets["if_type"] = pd.Series(dtype=str)
            gdf_inlets["INLET_ID"] = pd.Series(dtype=int)

        # Outlets -- iterate keys explicitly (don't rely on RangeIndex match)
        outlet_keys = list(po_dict.keys())
        if outlet_keys:
            gdf_outlets = gpd.GeoDataFrame(
                {
                    "OutletID": outlet_keys,
                    "ZOut":     [po_dict[k][1] for k in outlet_keys],
                },
                geometry=[po_dict[k][0] for k in outlet_keys],
                crs=crs,
            )
        else:
            gdf_outlets = self._empty_priority_gdf()
            gdf_outlets["OutletID"] = pd.Series(dtype=int)
            gdf_outlets["ZOut"]     = pd.Series(dtype=float)

        return gdf_inlets, gdf_outlets

    def _collapse_outlet_chains(self):
        """
        Merge outlets that lie close to another outlet's inlet vertex.
        Pattern A -> B -> C (where each `->` is a "physical proximity"
        edge) is collapsed so all inlets that previously drained to A or B
        now drain to C.

        Source outlets (the "froms" of every chain edge) are removed from
        gdf_priority_outlets. Any inlets that end up pointing to a removed
        outlet are remapped to the nearest surviving outlet (orphan fix).
        """
        gdf_inlets = self._gdf_priority_inlets
        gdf_outlets = self._gdf_priority_outlets
        if gdf_inlets.empty or gdf_outlets.empty:
            return

        joined = gpd.sjoin_nearest(
            gdf_outlets,
            gdf_inlets[["INLET_ID", "geometry", "OutletID"]],
            how="left",
            distance_col="dist_to_inlet",
            max_distance=3 * self._cell_size,
        )
        # 15 m hard cutoff (preserved from original logic) so the chain
        # collapse only fires for truly co-located outlet/inlet pairs.
        close = joined[joined["dist_to_inlet"] <= 15]
        if close.empty:
            return

        close = close[["OutletID_left", "OutletID_right"]].rename(
            columns={"OutletID_left": "flows_from",
                     "OutletID_right": "flows_to"},
        ).sort_values("flows_from")

        mapping = dict(zip(close["flows_from"], close["flows_to"]))

        # Collapse chains: A -> B -> C becomes A -> C, B -> C
        cache: dict = {}
        def find_final(node):
            if node in cache:
                return cache[node]
            visited = set()
            current = node
            while current in mapping and current not in visited:
                visited.add(current)
                nxt = mapping[current]
                if pd.isna(nxt):
                    break
                current = nxt
            cache[node] = current
            return current

        close["flows_to"] = close["flows_from"].apply(find_final)
        mapping_final = dict(zip(close["flows_from"], close["flows_to"]))

        remaining_outlet_ids = set(gdf_outlets["OutletID"].unique())
        mapping_final = {
            src: dst for src, dst in mapping_final.items()
            if pd.notna(dst) and dst in remaining_outlet_ids
        }

        # Remap inlet OutletID to the chain destination
        gdf_inlets["OutletID"] = gdf_inlets["OutletID"].map(
            lambda x: mapping_final.get(x, x)
        )
        # Remove the source outlets of every chain
        from_ids = set(close["flows_from"].values)
        gdf_outlets = gdf_outlets.loc[
            ~gdf_outlets["OutletID"].isin(from_ids)
        ].reset_index(drop=True)

        # Orphan fix: any inlet still pointing to a removed outlet gets
        # remapped to its nearest surviving outlet.
        valid_outlet_ids = set(gdf_outlets["OutletID"].unique())
        orphan_mask = ~gdf_inlets["OutletID"].isin(valid_outlet_ids)
        if orphan_mask.any():
            orphans = gdf_inlets.loc[orphan_mask, ["INLET_ID", "geometry"]].copy()
            remap = gpd.sjoin_nearest(
                orphans,
                gdf_outlets[["OutletID", "geometry"]],
                how="left",
                distance_col="_orphan_dist",
            )
            remap_pairs = remap[["INLET_ID", "OutletID"]].dropna(
                subset=["OutletID"]
            )
            orphan_remap = dict(
                zip(remap_pairs["INLET_ID"], remap_pairs["OutletID"])
            )
            gdf_inlets["OutletID"] = gdf_inlets.apply(
                lambda r: orphan_remap.get(r["INLET_ID"], r["OutletID"]),
                axis=1,
            )

        gdf_inlets = gdf_inlets.drop_duplicates(subset=["geometry"]).reset_index(
            drop=True
        )
        self._gdf_priority_inlets = gdf_inlets
        self._gdf_priority_outlets = gdf_outlets

    def _finalize_priority_attributes(self):
        """
        Set all output columns on the priority inlets/outlets and write the
        intermediate USE00_TUNNELS file. ZIn / ZOut are re-sampled from DTM
        at the snapped geometries.
        """
        gdf_inlets = self._gdf_priority_inlets
        gdf_outlets = self._gdf_priority_outlets
        if gdf_inlets.empty and gdf_outlets.empty:
            self._gdf_priority_combined = self._empty_priority_gdf()
            return

        # Common columns expected downstream
        column_order = [
            "ID", "Type", "VP_Network_ID", "Inlet_Type", "VP_Sur_Index",
            "VP_QMax", "Width", "Conn_2D", "Conn_No", "pBlockage",
            "Number_of", "Lag_Approach", "Lag_Value",
            "ZIn", "ZOut", "if_type", "geometry",
        ]

        # ---- inlets ----
        if not gdf_inlets.empty:
            gdf_inlets["ID"] = (
                gdf_inlets.index + 1 + self.PRIORITY_INLET_ID_OFFSET
            ).astype(str)
            gdf_inlets["Type"] = "I"
            gdf_inlets["Inlet_Type"] = "StormOutletNo1"
            gdf_inlets["VP_Network_ID"] = (
                gdf_inlets["OutletID"].astype(int)
                + self.PRIORITY_NETWORK_OFFSET
            )
            gdf_inlets = gdf_inlets.sort_values(
                by=["VP_Network_ID", "ZIn"], ascending=[True, False],
            )
            gdf_inlets["VP_Sur_Index"] = gdf_inlets.groupby(
                "VP_Network_ID"
            ).cumcount()
            gdf_inlets["VP_QMax"] = 10.0
            gdf_inlets["Width"] = 2.0
            gdf_inlets["Conn_2D"] = "SX"
            gdf_inlets["Conn_No"] = 4
            gdf_inlets["pBlockage"] = 0.0
            gdf_inlets["Number_of"] = 1
            gdf_inlets["Lag_Approach"] = "None"
            gdf_inlets["Lag_Value"] = 0.0
            # if_type already set from source_type ('tunnel' / 'wb_drainage')
            # Preserve as-is.

        # ---- outlets ----
        if not gdf_outlets.empty:
            gdf_outlets["VP_Network_ID"] = (
                gdf_outlets["OutletID"].astype(int)
                + self.PRIORITY_NETWORK_OFFSET
            )
            gdf_outlets["ID"] = gdf_outlets["VP_Network_ID"].astype(str)
            gdf_outlets["Type"] = "O"
            gdf_outlets["Inlet_Type"] = "0"
            gdf_outlets["VP_Sur_Index"] = 0
            gdf_outlets["VP_QMax"] = 10.0
            gdf_outlets["Width"] = 2.0
            gdf_outlets["Conn_2D"] = "SX"
            gdf_outlets["Conn_No"] = 0
            gdf_outlets["pBlockage"] = 0.0
            gdf_outlets["Number_of"] = 0
            gdf_outlets["Lag_Approach"] = "None"
            gdf_outlets["Lag_Value"] = 0.0
            gdf_outlets["ZIn"] = None
            gdf_outlets["if_type"] = "tunnel"

            # Re-sample ZOut from DTM at the (snapped) outlet positions
            gdf_outlets = SampleRaster(
                points=gdf_outlets, raster=self._dtm,
                output=None, col_name="ZOut",
            ).process_whole()
            gdf_outlets = gdf_outlets[column_order]

        # ZIn for inlets: re-sample at the snapped geometry
        if not gdf_inlets.empty:
            gdf_inlets = SampleRaster(
                points=gdf_inlets, raster=self._dtm,
                output=None, col_name="ZIn",
            ).process_whole()
            gdf_inlets["ZIn"] = gdf_inlets["ZIn"].fillna(0).astype(int)

            # ZOut comes from the matching outlet via VP_Network_ID
            if not gdf_outlets.empty:
                outlet_z_map = gdf_outlets.set_index("VP_Network_ID")["ZOut"]
                gdf_inlets["ZOut"] = (
                    gdf_inlets["VP_Network_ID"].map(outlet_z_map)
                    .fillna(0).astype(int)
                )
            else:
                gdf_inlets["ZOut"] = 0
            gdf_inlets = gdf_inlets[column_order]

        self._gdf_priority_inlets = gdf_inlets
        self._gdf_priority_outlets = gdf_outlets


    # ------------------------------------------------------------------
    # 6. Snap priority points to the output grid
    # ------------------------------------------------------------------
    def _snap_priority(self):
        """
        Combine priority inlets + outlets, snap them all to the output grid
        (outlets win cell contention), re-sample DTM, drop orphans whose
        outlet (or inlet) was lost to the snap. Held in memory as
        `self._gdf_priority_combined`.
        """
        gdf_inlets = self._gdf_priority_inlets
        gdf_outlets = self._gdf_priority_outlets
        if gdf_inlets.empty and gdf_outlets.empty:
            self._gdf_priority_combined = self._empty_priority_gdf()
            return

        gdf_all = pd.concat(
            [gdf_inlets, gdf_outlets], ignore_index=True
        ) if not gdf_inlets.empty else gdf_outlets.copy()
        gdf_all = gpd.GeoDataFrame(
            gdf_all, geometry="geometry", crs=f"EPSG:{self.projection}"
        )

        n_before = len(gdf_all)
        gdf_all["_snap_priority"] = (
            gdf_all["Type"].map({"O": 0, "I": 1}).fillna(2).astype(int)
        )
        gdf_all = self._snapper.snap_and_deduplicate(
            gdf_all, priority_col="_snap_priority", relocate=True,
        )
        n_dropped = gdf_all.attrs.get("dropped", 0)
        gdf_all = gdf_all.drop(columns=["_snap_priority"])
        print(
            f"\tPriority points snapped to grid: "
            f"{n_before} -> {len(gdf_all)} (dropped {n_dropped})"
        )

        gdf_all = _resample_z_after_snap(
            gdf_all, dtm=self._dtm,
            type_col="Type", inlet_value="I", outlet_value="O",
            network_col="VP_Network_ID",
        )
        gdf_all["ZIn"] = gdf_all["ZIn"].fillna(0).astype(int)
        gdf_all["ZOut"] = gdf_all["ZOut"].fillna(0).astype(int)

        outlet_nets = set(
            gdf_all.loc[gdf_all["Type"] == "O", "VP_Network_ID"].unique()
        )
        inlet_nets = set(
            gdf_all.loc[gdf_all["Type"] == "I", "VP_Network_ID"].unique()
        )
        valid = outlet_nets & inlet_nets
        before = len(gdf_all)
        gdf_all = gdf_all[gdf_all["VP_Network_ID"].isin(valid)].reset_index(
            drop=True
        )
        if len(gdf_all) < before:
            print(
                f"\tDropped {before - len(gdf_all)} orphaned priority points "
                f"(no matching outlet/inlet in network)."
            )

        self._gdf_priority_combined = gdf_all

    # ------------------------------------------------------------------
    # 7. Road inlets via tiled (block-based) processing
    # ------------------------------------------------------------------
    def _generate_road_inlets_tiled(self):
        """
        Generate inlet points along roads using tile-by-tile processing.
        Per-tile work runs entirely in memory; only globally-shared helpers
        (create_inlets_selection, move_inlets) which require file paths
        write to scratch.
        """
        print("\tGenerating road inlets (tiled)")
        p = self._params

        all_filters = list(set(p["filter_a"] + p["filter_b"] + p["filter_c"]))
        gdf_roads = Vector.load_vector(self._roads_path)
        gdf_roads = filter_roads(gdf_roads, p["FIELD_FOR_FILTERING"], all_filters)
        gdf_intersections = get_line_intersections(gdf_roads)
        gdf_roads_split = split_lines_points(
            gdf_lines=gdf_roads, gdf_points=gdf_intersections,
        )
        # Persist the split roads to scratch -- they replace the original
        # clipped layer because some downstream helpers re-read this path.
        gdf_roads_split.to_file(self._roads_path)
        self._gdf_roads_split = gdf_roads_split

        bounds_b = self._roads_bounds_for_filter(
            gdf_roads_split, p["FIELD_FOR_FILTERING"], p["filter_b"]
        )
        if bounds_b is not None:
            extent = [bounds_b[0], bounds_b[2], bounds_b[1], bounds_b[3]]
            self._builtup_mask = create_builtup_area_mask(
                materials_file=p["MANNING"],
                extent=extent,
                threshold=0.25, window_size=11, overlap=50,
                epsg=self.projection,
            )
        else:
            self._builtup_mask = None

        bounds_c = self._roads_bounds_for_filter(
            gdf_roads_split, p["FIELD_FOR_FILTERING"], p["filter_c"]
        )
        if bounds_c is not None:
            gdf_addr = Vector.load_vector(
                p["ADDRESS_POINTS"], bbox=tuple(bounds_c)
            )
            gdf_addr_buffer = gdf_addr.copy()
            gdf_addr_buffer["geometry"] = gdf_addr_buffer.geometry.buffer(100)
            gdf_addr_buffer = gdf_addr_buffer.dissolve().explode(
                ignore_index=True
            )
            self._gdf_address_buffer = gdf_addr_buffer
            self._gdf_parking = Vector.load_vector(
                p["PARKING"], bbox=tuple(bounds_c)
            )
        else:
            self._gdf_address_buffer = None
            self._gdf_parking = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
                crs=f"EPSG:{self.projection}",
            )

        if not gdf_roads.empty:
            gdf_industrial = Vector.load_vector(
                p["INDUSTRIAL_ESTATES"], bbox=tuple(gdf_roads.total_bounds)
            )
            gdf_industrial = Vector.fix_geometry(gdf_industrial)
        else:
            gdf_industrial = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
                crs=f"EPSG:{self.projection}",
            )

        tile_size = self.virtual_pipes_parameters.get("tile_size_m", 5000)
        tile_overlap = self.virtual_pipes_parameters.get(
            "tile_overlap_m", 200
        )
        tiler = DomainTiler(
            bounds=tuple(self._gdf_mask.total_bounds),
            tile_size=tile_size,
            overlap=tile_overlap,
            crs=f"EPSG:{self.projection}",
        )
        print(
            f"\tTiling: {len(tiler)} tiles "
            f"({tile_size} m + {tile_overlap} m overlap)"
        )

        tile_results = []
        roads_filter_pieces = {"a": [], "b": [], "c": []}

        for tile_id, tile_bounds, tile_poly in tiler.iter_tiles():
            tile_pts, tile_road_pieces = self._process_road_tile(
                tile_id=tile_id, tile_poly=tile_poly,
                roads_split=gdf_roads_split,
                gdf_industrial=gdf_industrial,
            )
            if tile_pts is not None and not tile_pts.empty:
                tile_results.append(tile_pts)
            for k, piece in tile_road_pieces.items():
                if piece is not None and not piece.empty:
                    roads_filter_pieces[k].append(piece)

        if tile_results:
            all_pts = pd.concat(tile_results, ignore_index=True)
            all_pts = gpd.GeoDataFrame(
                all_pts, geometry="geometry", crs=f"EPSG:{self.projection}"
            )
            all_pts = all_pts.drop_duplicates(subset=["geometry"]).reset_index(
                drop=True
            )
        else:
            all_pts = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
                crs=f"EPSG:{self.projection}",
            )

        def _concat_or_empty(parts):
            if not parts:
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
                    crs=f"EPSG:{self.projection}",
                )
            return gpd.GeoDataFrame(
                pd.concat(parts, ignore_index=True),
                geometry="geometry",
                crs=f"EPSG:{self.projection}",
            ).drop_duplicates(subset=["geometry"]).reset_index(drop=True)

        gdf_roads_filtered = _concat_or_empty(
            roads_filter_pieces["a"]
            + roads_filter_pieces["b"]
            + roads_filter_pieces["c"]
        )
        self._gdf_roads_filtered = gdf_roads_filtered

        # ---- water exclusion + DTM snap + filter (global, single pass) ----
        # create_inlets_selection and move_inlets accept GDF input but
        # require a file path for output. Both outputs go to scratch and
        # are read back into memory; nothing persists.
        print(
            f"\tCreating basic inlets at interval: {p['interval_inlets']} m"
        )
        sel_path = os.path.join(
            self._session_tmp, f"inlets_water_excluded_{self.domain_name}.gpkg"
        )
        sel_path = create_inlets_selection(
            points=all_pts,
            water_network=[self._wb_path, self._rn_path],
            cell_size=self._cell_size,
            out_file=sel_path,
        )
        print("\tMoving inlets to lowest DTM in 3x3 window")
        moved_path = os.path.join(
            self._session_tmp, f"inlets_dtm_snapped_{self.domain_name}.gpkg"
        )
        moved_path = move_inlets(
            inlets=sel_path,
            out_file=moved_path,
            cell_size=self._cell_size,
            epsg=self.projection,
            dtm=self._dtm,
        )
        inlets = Vector.load_vector(moved_path)
        inlets = extract_inlets(
            dem=self._dtm, inlets=inlets, temp=self._session_tmp,
        )
        inlets = inlets[["pointid", "dtm_value", "geometry"]]
        inlets = filter_inlets(
            inlets, self._cell_size, increase_capacity=False
        )

        self._gdf_inlets_road = inlets

    @staticmethod
    def _roads_bounds_for_filter(gdf_roads_split, filter_field, filter_vals):
        """Bounding box of road segments matching the filter, or None."""
        subset = gdf_roads_split[
            gdf_roads_split[filter_field].isin(filter_vals)
        ]
        if subset.empty:
            return None
        return subset.total_bounds

    def _process_road_tile(self, tile_id, tile_poly, roads_split,
                            gdf_industrial):
        """
        Generate road inlet points for a single tile, entirely in memory.

        Returns (tile_points_gdf, dict of per-filter road pieces).
        Each returned GeoDataFrame carries a `source` column (a/b/c/d) so
        downstream consumers can trace where each point came from.
        """
        p = self._params
        ix, iy = tile_id
        crs_str = f"EPSG:{self.projection}"

        tile_roads = roads_split.clip(tile_poly)
        if tile_roads.empty:
            return None, {"a": None, "b": None, "c": None}

        pts_a, roads_a = self._gen_points_subset(
            gdf_roads=tile_roads, filter_vals=p["filter_a"],
        )
        pts_b, roads_b = self._gen_points_subset(
            gdf_roads=tile_roads, filter_vals=p["filter_b"],
        )
        pts_c, roads_c = self._gen_points_subset(
            gdf_roads=tile_roads, filter_vals=p["filter_c"],
        )
        all_filters = list(set(
            p["filter_a"] + p["filter_b"] + p["filter_c"]
        ))
        pts_d, _ = self._gen_points_subset(
            gdf_roads=tile_roads, filter_vals=all_filters,
        )

        # ---- per-filter clipping (in-tile, using global masks) ----
        if not gdf_industrial.empty and not pts_d.empty:
            pts_d = pts_d.clip(gdf_industrial)

        if self._builtup_mask is not None and not pts_b.empty:
            pts_b = pts_b.clip(self._builtup_mask)
        if self._builtup_mask is not None and not roads_b.empty:
            roads_b = roads_b.clip(self._builtup_mask)
            roads_b = roads_b[roads_b.geometry.length > 15]

        if (self._gdf_address_buffer is not None
                and not pts_c.empty):
            pts_c = pts_c.clip(self._gdf_address_buffer)
            roads_c = roads_c.clip(self._gdf_address_buffer)
            if not self._gdf_parking.empty:
                pts_c = pts_c.overlay(self._gdf_parking, how="difference")
                roads_c = roads_c.overlay(
                    self._gdf_parking, how="difference",
                )
            roads_c = roads_c[roads_c.geometry.length > 15]

        def _tagged(df, src):
            if df is None or df.empty:
                return None
            df = df.copy()
            df["source"] = src
            return df

        pieces = [
            _tagged(pts_a, "a"),
            _tagged(pts_b, "b"),
            _tagged(pts_c, "c"),
            _tagged(pts_d, "d"),
        ]
        pieces = [p for p in pieces if p is not None]
        if pieces:
            tile_pts = pd.concat(pieces, ignore_index=True)
            tile_pts = gpd.GeoDataFrame(
                tile_pts, geometry="geometry", crs=crs_str
            )
            tile_pts = tile_pts.drop_duplicates(subset=["geometry"])
        else:
            tile_pts = None

        n_total = 0 if tile_pts is None else len(tile_pts)
        print(
            f"\t  tile ({ix:02d},{iy:02d}): "
            f"a={len(pts_a) if pts_a is not None else 0}  "
            f"b={len(pts_b) if pts_b is not None else 0}  "
            f"c={len(pts_c) if pts_c is not None else 0}  "
            f"d={len(pts_d) if pts_d is not None else 0}  "
            f"-> {n_total} unique"
        )

        return tile_pts, {"a": roads_a, "b": roads_b, "c": roads_c}

    def _gen_points_subset(self, gdf_roads, filter_vals):
        """
        In-memory equivalent of `get_points_along_geometry` from lib_00:
        filter -> remove tunnels/bridges -> interpolate along each line
        at `interval_inlets` -> buffer/dissolve/centroid for dedup.

        Returns (points_gdf, filtered_roads_gdf).
        """
        crs_str = f"EPSG:{self.projection}"
        empty_pts = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=crs_str), crs=crs_str,
        )
        empty_roads = empty_pts.copy()
        if gdf_roads.empty:
            return empty_pts, empty_roads

        filtered = filter_roads(
            gdf_roads, self._params["FIELD_FOR_FILTERING"], filter_vals,
        )
        if filtered.empty:
            return empty_pts, filtered

        interval = self._params["interval_inlets"]
        inlets = []
        for line in filtered.geometry.values:
            if line is None or line.is_empty:
                continue
            length = line.length
            if length <= 0:
                continue
            distances = np.arange(interval, length, interval)
            distances = np.append(distances, length)
            inlets.extend(line.interpolate(d) for d in distances)
        if not inlets:
            return empty_pts, filtered

        pts = gpd.GeoDataFrame(geometry=inlets, crs=crs_str)
        pts.geometry = pts.geometry.buffer(self._cell_size)
        pts = pts.dissolve().explode(ignore_index=True)
        pts.geometry = pts.geometry.centroid
        pts = pts.reset_index(drop=True)
        pts["pointid"] = pts.index
        return pts, filtered


    # ------------------------------------------------------------------
    # 8. Densify inlets in depressions
    # ------------------------------------------------------------------
    def _densify_in_depressions(self):
        """
        Generate extra inlets where roads cross depressions.

        `create_new_inlets`, `create_inlets_selection`, and `move_inlets`
        require file paths. All three write into scratch; only the final
        filtered GDF is kept in memory as `self._gdf_inlets_densified`.
        """
        print("\tCreating densified inlets within depressions")
        densified = create_new_inlets(
            dem=self._dtm,
            depression_vector_path=self._filtered_depressions_path,
            gdf_roads=self._gdf_roads_filtered,
            interval=self._params["interval_inlets"] / 3,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_inlets_{self.domain_name}.gpkg",
            ),
            epsg=self.projection,
            cell_size=self._cell_size,
        )
        densified = Vector.load_vector(densified)
        sel_path = create_inlets_selection(
            points=densified,
            water_network=[self._wb_path, self._rn_path],
            cell_size=self._cell_size,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_water_excluded_{self.domain_name}.gpkg",
            ),
        )
        moved_path = move_inlets(
            inlets=sel_path,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_dtm_snapped_{self.domain_name}.gpkg",
            ),
            cell_size=self._cell_size,
            epsg=self.projection,
            dtm=self._dtm,
        )
        densified = Vector.load_vector(moved_path)
        densified = extract_inlets(
            dem=self._dtm, inlets=densified, temp=self._session_tmp,
        )
        densified = filter_inlets(
            densified, self._cell_size, increase_capacity=False
        )
        self._gdf_inlets_densified = densified

    def _densify_in_parking(self):
        """
        Generate extra inlets on parking-area road segments. All
        intermediate outputs in scratch; final result in memory.
        """
        print("\tAdding inlets to drain water from parking lots")
        roads_parking = self._gdf_roads_split.clip(self._gdf_parking)
        roads_parking = roads_parking[roads_parking.geometry.length > 10]

        if roads_parking.empty:
            self._gdf_inlets_parking = None
            return

        densified = create_new_inlets(
            dem=self._dtm,
            depression_vector_path=self._filtered_depressions_path,
            gdf_roads=roads_parking,
            interval=self._params["interval_inlets"] / 5,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_parking_{self.domain_name}.gpkg",
            ),
            epsg=self.projection,
            cell_size=self._cell_size,
        )
        densified = Vector.load_vector(densified)
        sel_path = create_inlets_selection(
            points=densified,
            water_network=[self._wb_path, self._rn_path],
            cell_size=self._cell_size,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_parking_water_excl_{self.domain_name}.gpkg",
            ),
        )
        moved_path = move_inlets(
            inlets=sel_path,
            out_file=os.path.join(
                self._session_tmp,
                f"densified_parking_dtm_snapped_{self.domain_name}.gpkg",
            ),
            cell_size=self._cell_size,
            epsg=self.projection,
            dtm=self._dtm,
        )
        densified = Vector.load_vector(moved_path)
        densified = extract_inlets(
            dem=self._dtm, inlets=densified, temp=self._session_tmp,
        )
        densified = filter_inlets(
            densified, self._cell_size, increase_capacity=False
        )
        self._gdf_inlets_parking = densified

    # ------------------------------------------------------------------
    # 10. Generate outlets on RN and waterbodies
    # ------------------------------------------------------------------
    def _generate_outlets(self):
        """
        Generate outlets along RN and waterbodies. Helpers from lib_00
        write to disk; we route them to scratch and read back into memory.
        """
        print("\tCreating outlets on RN and waterbodies")
        outlets_path = create_outlets(
            rn=self._rn_path,
            waterbody=self._wb_path,
            interval=self._params["interval_outlets"],
            out_file=os.path.join(
                self._session_tmp, f"outlets_raw_{self.domain_name}.gpkg",
            ),
            epsg=self.projection,
            cell_size=self._cell_size,
        )
        outlets_path = move_outlets(
            outlets=outlets_path,
            out_file=os.path.join(
                self._session_tmp,
                f"outlets_dtm_snapped_{self.domain_name}.gpkg",
            ),
            epsg=self.projection,
            cell_size=self._cell_size,
            dtm=self._dtm,
        )
        if self._params["FILTER_BUILDINGS"]:
            outlets_path = remove_outlets_buildings(
                outlets_path,
                self._params["BUILDINGS"],
                out_file=os.path.join(
                    self._session_tmp,
                    f"outlets_building_filtered_{self.domain_name}.gpkg",
                ),
                bounds=self._gdf_mask.total_bounds,
                bounds_epsg=self.projection,
            )
        self._gdf_outlets = Vector.load_vector(outlets_path)


    # ------------------------------------------------------------------
    # 11. Combine all inlets + outlets and snap to grid (pre-connect)
    # ------------------------------------------------------------------
    def _snap_general(self):
        """
        Combine all in-memory inlet sources (basic road, depression,
        parking, depression-lowest) with the outlets, then snap to grid.
        VP_Network_ID is assigned later by `connect()`.
        """
        inlets_parts = [
            g for g in (
                self._gdf_inlets_road,
                self._gdf_inlets_densified,
                self._gdf_inlets_parking,
                self._gdf_inlets_depressions,
            )
            if g is not None and not g.empty
        ]

        if inlets_parts:
            gdf_inlets = pd.concat(inlets_parts, ignore_index=True)
            gdf_inlets = gpd.GeoDataFrame(
                gdf_inlets, geometry="geometry",
                crs=f"EPSG:{self.projection}",
            )
            gdf_inlets = gdf_inlets.drop_duplicates(
                subset=["geometry"]
            ).reset_index(drop=True)
        else:
            gdf_inlets = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=f"EPSG:{self.projection}"),
                crs=f"EPSG:{self.projection}",
            )

        print(
            f"\tCombined inlets before cleaning: {len(gdf_inlets)}"
        )
        gdf_inlets = filter_inlets(
            gdf_inlets, self._cell_size, increase_capacity=True
        )
        gdf_inlets["type"] = "inlet"

        gdf_outlets = self._gdf_outlets.copy()
        gdf_outlets["type"] = "outlet"

        gdf_all = pd.concat([gdf_inlets, gdf_outlets], ignore_index=True)
        gdf_all = gpd.GeoDataFrame(
            gdf_all, geometry="geometry", crs=f"EPSG:{self.projection}",
        )
        gdf_all = gdf_all.reset_index(drop=True)
        gdf_all["pointid"] = gdf_all.index + 1

        n_before = len(gdf_all)
        gdf_all["_snap_priority"] = (
            gdf_all["type"]
            .map({"outlet": 0, "inlet": 1})
            .fillna(2)
            .astype(int)
        )
        gdf_all = self._snapper.snap_and_deduplicate(
            gdf_all, priority_col="_snap_priority", relocate=True,
        )
        n_dropped = gdf_all.attrs.get("dropped", 0)
        gdf_all = gdf_all.drop(columns=["_snap_priority"]).reset_index(drop=True)
        gdf_all["pointid"] = gdf_all.index + 1
        print(
            f"\tGeneral points snapped to grid: "
            f"{n_before} -> {len(gdf_all)} (dropped {n_dropped})"
        )

        self._gdf_inlets_general = gdf_all[gdf_all["type"] == "inlet"].copy(
        ).reset_index(drop=True)
        self._gdf_outlets_general = gdf_all[gdf_all["type"] == "outlet"].copy(
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 12. Post-clean general inlets/outlets against priority points
    # ------------------------------------------------------------------
    def _post_clean_against_priority(self):
        """
        Three proximity filters; final cleaned GDFs written to scratch
        only because `connect()` requires file paths.
          (a) Drop general inlets within 2.5 cells of any general outlet.
          (b) Drop general inlets within 2.5 cells of a priority point.
          (c) Drop general outlets within 2.5 cells of a priority point.
        """
        gdf_inlets = self._gdf_inlets_general
        gdf_outlets = self._gdf_outlets_general
        cs = self._cell_size

        if not gdf_inlets.empty and not gdf_outlets.empty:
            joined = gpd.sjoin_nearest(
                gdf_inlets, gdf_outlets, how="left",
                distance_col="dist_to_outlet",
                max_distance=5 * cs,
            )
            keep_idx = joined.index[
                joined["dist_to_outlet"].isna()
                | (joined["dist_to_outlet"] > 2.5 * cs)
            ]
            gdf_inlets = gdf_inlets.loc[keep_idx].reset_index(drop=True)

        gdf_prio = self._gdf_priority_combined
        if gdf_prio is not None and not gdf_prio.empty:
            if not gdf_inlets.empty:
                joined = gpd.sjoin_nearest(
                    gdf_inlets, gdf_prio, how="left",
                    distance_col="dist_to_priority",
                    max_distance=3 * cs,
                )
                keep_idx = joined.index[
                    joined["dist_to_priority"].isna()
                    | (joined["dist_to_priority"] > 2.5 * cs)
                ]
                gdf_inlets = gdf_inlets.loc[keep_idx].reset_index(drop=True)
            if not gdf_outlets.empty:
                joined = gpd.sjoin_nearest(
                    gdf_outlets, gdf_prio, how="left",
                    distance_col="dist_to_priority",
                    max_distance=3 * cs,
                )
                keep_idx = joined.index[
                    joined["dist_to_priority"].isna()
                    | (joined["dist_to_priority"] > 2.5 * cs)
                ]
                gdf_outlets = gdf_outlets.loc[keep_idx].reset_index(drop=True)

        self._cleaned_inlets_path = os.path.join(
            self._session_tmp, f"inlets_cleaned_{self.domain_name}.gpkg",
        )
        self._cleaned_outlets_path = os.path.join(
            self._session_tmp, f"outlets_cleaned_{self.domain_name}.gpkg",
        )
        gdf_inlets.to_file(
            self._cleaned_inlets_path, driver="GPKG", index=False
        )
        gdf_outlets.to_file(
            self._cleaned_outlets_path, driver="GPKG", index=False
        )

    # ------------------------------------------------------------------
    # 13. Connect inlets to outlets (chunked distance matrix)
    # ------------------------------------------------------------------
    def _run_connect(self):
        """
        `connect()` from lib_00 requires file paths and writes to disk.
        Output goes to scratch; `_snap_final()` reads it back and merges
        with the in-memory priority points before final save.
        """
        print("\tConnecting inlets to outlets")
        self._connect_out_path = os.path.join(
            self._session_tmp, f"connected_{self.domain_name}.gpkg",
        )
        connect(
            self._cleaned_inlets_path,
            self._cleaned_outlets_path,
            self._connect_out_path,
            self._dtm,
            self._session_tmp,
            epsg=self.projection,
        )


    # ------------------------------------------------------------------
    # 14. Final snap: merge connected result with priority points
    # ------------------------------------------------------------------
    def _snap_final(self):
        """
        Read the connected output from scratch, concatenate with the
        in-memory priority points, do a final grid snap (outlets win cell
        contention), recompute VP_Sur_Index and dist (vectorised). The
        final GDF is held in memory and only saved by `_save_output()`.
        """
        gdf_outfile = gpd.read_file(self._connect_out_path)
        gdf_outfile["if_type"] = "normal"

        gdf_priority = getattr(self, "_gdf_priority_combined", None)
        if gdf_priority is not None and not gdf_priority.empty:
            gdf_priority = gdf_priority.sort_values(
                by=["VP_Network_ID", "ZIn"], ascending=[True, False],
            )
            gdf = pd.concat([gdf_priority, gdf_outfile], ignore_index=True)
        else:
            gdf = gdf_outfile

        gdf = gpd.GeoDataFrame(
            gdf, geometry="geometry", crs=f"EPSG:{self.projection}",
        )

        # Final snap: outlets win
        gdf["_snap_priority"] = (
            gdf["Type"].map({"O": 0, "I": 1}).fillna(2).astype(int)
        )
        n_before = len(gdf)
        gdf = self._snapper.snap_and_deduplicate(
            gdf, priority_col="_snap_priority", relocate=True,
        )
        n_dropped = gdf.attrs.get("dropped", 0)
        gdf = gdf.drop(columns=["_snap_priority"])
        print(
            f"\tFinal snap: {n_before} -> {len(gdf)} (dropped {n_dropped})"
        )

        # Re-sample Z at new positions, propagate to inlets
        gdf = _resample_z_after_snap(
            gdf, dtm=self._dtm,
            type_col="Type", inlet_value="I", outlet_value="O",
            network_col="VP_Network_ID",
        )

        # Outlet ZIn = ZOut (preserved from original convention)
        mask_o = gdf["Type"] == "O"
        gdf.loc[mask_o, "ZIn"] = gdf.loc[mask_o, "ZOut"]

        # Vectorised distance from each inlet to its outlet
        # (previously this used apply(axis=1) with per-row dict lookups)
        outlets = (
            gdf[mask_o][["VP_Network_ID", "geometry"]]
            .drop_duplicates(subset=["VP_Network_ID"])
            .set_index("VP_Network_ID")
        )
        paired = gpd.GeoSeries(
            gdf["VP_Network_ID"].map(outlets["geometry"]).values,
            index=gdf.index,
            crs=gdf.crs,
        )
        dist = gdf.geometry.distance(paired, align=True)
        dist.loc[mask_o] = 0.0
        gdf["dist"] = dist.round(2)

        # Sort inlets within each network by ZIn descending, set VP_Sur_Index
        gdf = gdf.sort_values(
            by=["VP_Network_ID", "ZIn"], ascending=[True, False],
        ).reset_index(drop=True)
        gdf["VP_Sur_Index"] = gdf.groupby("VP_Network_ID").cumcount()
        gdf.loc[gdf["Type"] == "O", "VP_Sur_Index"] = 0

        # Drop networks with no inlets at all
        nets_with_inlets = set(
            gdf.loc[gdf["Type"] == "I", "VP_Network_ID"].unique()
        )
        nets_with_outlets = set(
            gdf.loc[gdf["Type"] == "O", "VP_Network_ID"].unique()
        )
        keep_nets = nets_with_inlets & nets_with_outlets
        before = len(gdf)
        gdf = gdf[gdf["VP_Network_ID"].isin(keep_nets)].reset_index(drop=True)
        if len(gdf) < before:
            print(
                f"\tDropped {before - len(gdf)} points from networks with "
                "missing endpoints."
            )

        # Number_of: clamp to [1, 12]
        if "Number_of" in gdf.columns:
            gdf["Number_of"] = (
                gdf["Number_of"].fillna(1).astype(int).clip(lower=1, upper=12)
            )
        else:
            gdf["Number_of"] = 1

        # Lag fields default
        if "Lag_Approach" not in gdf.columns:
            gdf["Lag_Approach"] = "None"
        else:
            gdf["Lag_Approach"] = gdf["Lag_Approach"].fillna("None")
        if "Lag_Value" not in gdf.columns:
            gdf["Lag_Value"] = 0.0
        else:
            gdf["Lag_Value"] = gdf["Lag_Value"].fillna(0.0)

        self._gdf_final = gdf

    # ------------------------------------------------------------------
    # 15. Save final output with the correct dtypes
    # ------------------------------------------------------------------
    def _save_output(self):
        """Cast columns to their final dtypes and persist the GPKG."""
        gdf = self._gdf_final
        if gdf.empty:
            print(f"\tNo final pit points generated for {self.domain_name}")
            return

        int_cols   = ["VP_Network_ID", "Conn_No", "Number_of"]
        float_cols = [
            "VP_Sur_Index", "VP_QMax", "Width", "pBlockage",
            "Lag_Value", "ZIn", "ZOut", "dist",
        ]
        str_cols   = [
            "ID", "Type", "Inlet_Type", "Conn_2D",
            "Lag_Approach", "if_type",
        ]

        for col in int_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(
                    gdf[col], errors="coerce"
                ).fillna(0).astype(int)
        for col in float_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(
                    gdf[col], errors="coerce"
                ).fillna(0.0).astype(float)
        for col in str_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna("").astype(str)

        gdf.to_file(self._out_path, driver="GPKG", index=False)
        print(f"\tOutput saved to: {self._out_path}")
        print(f"\tTotal points: {len(gdf)} "
              f"(inlets={len(gdf[gdf['Type']=='I'])}, "
              f"outlets={len(gdf[gdf['Type']=='O'])})")
