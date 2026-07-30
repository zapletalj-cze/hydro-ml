"""Compares three ATL08 quality-filter sets on the near-levee export:
A cloud_flag_atm only (current), B + layer_flag==0, C + terrain_flag==0.
For each: point counts, candidates near axis, crest points (longitudinal
prominence proxy = 2D ring prominence >= 0.7), median dz over DSM, and
agreement of crest points with the 1 m DTM crest at the detected axis.
Writes filter_comparison.json + csv. Read-only on all rasters."""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from osgeo import gdal

gdal.UseExceptions()

# ---- CONFIG -----------------------------------------------------------------
POINTS_GPKG   = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\at08_terrain_near_levees.gpkg")
DTM_TIF       = Path(r"B:\01_Projects\154_Poland_Flood_v3\01_MD\01_HAZARD\01_DTM\1m\Poland_dem_1m.tif")
DSM_TIF       = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c_10m.tif")  # GLO-30 mosaic used by script 15 - adjust if named differently
SEGMENTS_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\processing\levee_segments_z.gpkg")

CAND_DIST_M, RING_MIN_M, RING_MAX_M = 30.0, 50.0, 200.0
LEVEE_EXCL_M, MIN_REF_PTS = 30.0, 10
PROM_THR, PROM_CAP_M = 0.7, 8.0
CREST_WIN_M, GROUND_WIN_M = 5.0, 1.0
DATUM_OFFSET_M = 0.01     # from the corridor estimate (script 29)

OUT_DIR = Path(__file__).parent / "diagnostics_ch4"


class Sampler:
    def __init__(self, path):
        self.ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        self.gt = self.ds.GetGeoTransform()
        self.band = self.ds.GetRasterBand(1)
        self.nod = self.band.GetNoDataValue()
        self.nx, self.ny = self.ds.RasterXSize, self.ds.RasterYSize

    def win(self, x, y, radius_m):
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

    def vmax(self, x, y, r):
        a = self.win(x, y, r)
        return float(np.nanmax(a)) if a is not None and np.isfinite(a).any() else np.nan

    def vmed(self, x, y, r):
        a = self.win(x, y, r)
        return float(np.nanmedian(a)) if a is not None and np.isfinite(a).any() else np.nan


def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def stats(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "median_m": round(float(np.median(d)), 3),
            "mae_m": round(float(np.mean(np.abs(d))), 3),
            "p20_m": round(float(np.percentile(d, 20)), 3),
            "p80_m": round(float(np.percentile(d, 80)), 3)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts = gpd.read_file(POINTS_GPKG, engine="pyogrio")
    if pts.crs is None or pts.crs.to_epsg() != 2180:
        pts = pts.set_crs(epsg=2180) if pts.crs is None else pts.to_crs(epsg=2180)
    seg = gpd.read_file(SEGMENTS_GPKG, engine="pyogrio")
    if seg.crs is None or seg.crs.to_epsg() != 2180:
        seg = seg.set_crs(epsg=2180) if seg.crs is None else seg.to_crs(epsg=2180)

    c_h = pick(pts.columns, "h_te_ortho", "h_ortho", "h", "height_ortho")
    c_cloud = pick(pts.columns, "cloud_flag_atm", "cloud_flag", "cloud")
    c_layer = pick(pts.columns, "layer_flag")
    c_terr = pick(pts.columns, "terrain_flag", "terrain_flg")
    print("columns picked:", c_h, c_cloud, c_layer, c_terr)
    if None in (c_h, c_cloud, c_layer, c_terr):
        raise KeyError(f"missing a needed column; available: {list(pts.columns)}")

    h = pd.to_numeric(pts[c_h], errors="coerce").values
    cloud = pd.to_numeric(pts[c_cloud], errors="coerce").values
    layer = pd.to_numeric(pts[c_layer], errors="coerce").values
    terr = pd.to_numeric(pts[c_terr], errors="coerce").values
    xy = np.c_[pts.geometry.x.values, pts.geometry.y.values]

    report = {"n_export": int(len(pts)),
              "value_counts": {
                  "cloud_flag_atm": {str(k): int(v) for k, v in
                                     pd.Series(cloud).value_counts().items()},
                  "layer_flag": {str(k): int(v) for k, v in
                                 pd.Series(layer).value_counts().items()},
                  "terrain_flag": {str(k): int(v) for k, v in
                                   pd.Series(terr).value_counts().items()}}}

    # distances to the detected axis
    j = gpd.sjoin_nearest(pts[["geometry"]], seg[["geometry"]], how="left",
                          max_distance=RING_MAX_M + 50, distance_col="d")
    d_axis = np.full(len(pts), np.inf)
    dmin = j.groupby(j.index)["d"].min()
    d_axis[dmin.index.values] = dmin.values

    dtm = Sampler(DTM_TIF)
    dsm = Sampler(DSM_TIF)

    from shapely.ops import unary_union, nearest_points
    from shapely.geometry import Point as _P
    axis = unary_union(list(seg.geometry.values))

    base_ok = np.isfinite(h) & (cloud <= 1)
    filters = {
        "A_cloud_only": base_ok,
        "B_plus_layer0": base_ok & (layer == 0),
        "C_plus_layer0_terrain0": base_ok & (layer == 0) & (terr == 0),
    }

    rows = []
    dtm_cache, dsm_cache = {}, {}
    for name, ok in filters.items():
        is_ref = ok & (d_axis > LEVEE_EXCL_M)
        is_cand = ok & (d_axis <= CAND_DIST_M)
        tree = cKDTree(xy[is_ref])
        href = h[is_ref]
        crest_idx = []
        for i in np.where(is_cand)[0]:
            idx = tree.query_ball_point(xy[i], RING_MAX_M)
            dd = np.linalg.norm(xy[is_ref][idx] - xy[i], axis=1)
            ring = [jj for jj, dj in zip(idx, dd) if dj >= RING_MIN_M]
            if len(ring) < MIN_REF_PTS:
                continue
            p = h[i] - float(np.median(href[ring]))
            if PROM_THR <= p <= PROM_CAP_M:
                crest_idx.append(i)

        dz, dtm_diff = [], []
        for i in crest_idx:
            if i not in dsm_cache:
                dsm_cache[i] = dsm.vmed(xy[i, 0], xy[i, 1], 5.0)
            if np.isfinite(dsm_cache[i]):
                dz.append(h[i] - dsm_cache[i])
            if i not in dtm_cache:
                on_ax = nearest_points(_P(xy[i, 0], xy[i, 1]), axis)[1]
                dtm_cache[i] = dtm.vmax(on_ax.x, on_ax.y, CREST_WIN_M)
            if np.isfinite(dtm_cache[i]):
                dtm_diff.append(h[i] - (dtm_cache[i] - DATUM_OFFSET_M))

        row = {"filter": name,
               "n_pass": int(ok.sum()),
               "n_candidates": int(is_cand.sum()),
               "n_crest": len(crest_idx),
               "dz_over_dsm": stats(dz),
               "crest_vs_dtm": stats(dtm_diff)}
        rows.append(row)
        report[name] = row
        print(row)

    (OUT_DIR / "filter_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    flat = []
    for r in rows:
        f = {"filter": r["filter"], "n_pass": r["n_pass"],
             "n_candidates": r["n_candidates"], "n_crest": r["n_crest"]}
        for k, v in r["dz_over_dsm"].items():
            f[f"dz_{k}"] = v
        for k, v in r["crest_vs_dtm"].items():
            f[f"dtm_{k}"] = v
        flat.append(f)
    pd.DataFrame(flat).to_csv(OUT_DIR / "filter_comparison.csv", index=False)
    print("written: filter_comparison.json / .csv")


if __name__ == "__main__":
    main()