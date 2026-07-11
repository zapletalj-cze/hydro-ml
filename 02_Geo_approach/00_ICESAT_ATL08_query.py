"""
ATL08 track query and terrain height extraction for a polygon AOI.
Uses earthaccess (stable NASA library) instead of deprecated icepyx v1.x.

Extracts median terrain height (h_te_median) from all ATL08 tracks intersecting
a user-defined polygon and converts it to EGM2008 orthometric height so the
output matches the Copernicus DSM datum and is a true elevation (nadmorska
vyska), not an ellipsoidal height.

Why the datum step: ATL08 heights are ellipsoidal (WGS84 / ITRF2014). Copernicus
GLO-30 is orthometric (EGM2008 geoid). The two differ by the geoid undulation N
(tens of metres). This script converts ATL08 to EGM2008 via PROJ, records N, and
outputs the orthometric height as the primary elevation field.

Requirements:
    pip install earthaccess h5py geopandas shapely pyogrio pandas pyproj

Earthdata credentials - store in ~/.netrc:
    machine urs.earthdata.nasa.gov
        login YOUR_USERNAME
        password YOUR_PASSWORD

The datum conversion needs the EGM2008 grid available to PROJ. If it is not
installed locally, set USE_PROJ_NETWORK = True, or run once:
    pyproj sync --file us_nga_egm2008_1
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*DataGranule.size.*",
    category=FutureWarning,
    module="earthaccess",
)

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import earthaccess
import pyproj
from pyproj import Transformer
from pathlib import Path
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# AOI polygon - Lower Vistula (Torun -> Gdansk)
AOI_POLYGON = [
    [14.1, 49.0],  # Southwest
    [24.2, 49.0],  # Southeast
    [24.2, 55.0],  # Northeast
    [14.1, 55.0],  # Northwest
    [14.1, 49.0],  # Closing
]

# Alternatively load from file:
# AOI_POLYGON = load_polygon_from_file("aoi.gpkg")

DATE_RANGE = ("2019-01-01", "2025-12-31")  # full mission
OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08")
OUTPUT_GPKG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_terrain_heights_new.gpkg")
OUTPUT_CSV = OUTPUT_GPKG.with_suffix(".csv")

TERRAIN_VAR = "h_te_median"  # robust terrain height per 100m segment
BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

# ------- 20 m terrain points (crest detection needs finer sampling) ---------
# ATL08 v005+ provides terrain heights at 20 m subsegments
# (land_segments/latitude_20m, longitude_20m, terrain/h_te_best_fit_20m).
# A levee crest a few metres wide is diluted in the 100 m median, so the 20 m
# points are used as the primary source when present; 100 m is the fallback.
USE_20M_TERRAIN = True

# ------- Crest selection against detected levees ----------------------------
# Points are kept as crest candidates when (a) they lie within LEVEE_BUFFER_M
# of a detected levee line and (b) their along-track profile shows a clear
# raised feature: height minus the median of neighbours in the window exceeds
# MIN_PROMINENCE_M. The immediate vicinity (PROM_EXCLUDE_M) is excluded from
# the baseline so the bump does not raise its own reference.
DETECTED_LEVEES_GPKG = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\predictions_eval\levees_predicted_Odra.gpkg"
)
CRS_METRIC = 2180            # metric CRS for buffering and along-track distances
LEVEE_BUFFER_M = 30.0        # max distance point-to-levee to be a candidate
PROM_WINDOW_M = 300.0        # along-track neighbourhood for the baseline
PROM_EXCLUDE_M = 40.0        # exclude this vicinity from the baseline
MIN_NEIGHBORS = 4            # minimum neighbours to compute a baseline
MIN_PROMINENCE_M = 1.0       # required height above the baseline
OUTPUT_CREST_GPKG = OUTPUT_GPKG.with_name(OUTPUT_GPKG.stem + "_crest.gpkg")

# ------- Quality filters (consistent with 10_atl08_crest_heights.py) --------
MAX_TE_UNCERTAINTY_M = 1.5   # drop segments with terrain uncertainty above this
MIN_TE_PHOTONS = 50          # minimum terrain photons per segment
FILTER_CLOUDS = True         # drop cloud / blowing-snow flagged segments
FILTER_SATURATION = True     # drop saturated segments
FILTER_TERRAIN_FLAG = True   # drop segments flagged as deviating from reference DEM
REQUIRE_NIGHT = False        # keep only night segments (off: keep day+night, record flag)

# ------- Datum: ellipsoidal WGS84 -> orthometric EGM2008 --------------------
USE_PROJ_NETWORK = False           # True if the EGM2008 grid is not installed locally
EGM2008_COMPOUND = "EPSG:4326+3855"  # WGS84 horizontal + EGM2008 height
WGS84_3D = "EPSG:4979"               # WGS84 ellipsoidal 3D

FILL_ABS = 1e30  # ATL08 float fill (~3.4e38); |value| above this is invalid


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def load_polygon_from_file(path: str) -> list:
    gdf = gpd.read_file(path, engine="pyogrio").to_crs(epsg=4326)
    coords = list(gdf.geometry.iloc[0].exterior.coords)
    return [[lon, lat] for lon, lat in coords]


def bbox_from_polygon(polygon: list) -> tuple:
    """Returns (lon_min, lat_min, lon_max, lat_max) from polygon coords."""
    lons = [c[0] for c in polygon]
    lats = [c[1] for c in polygon]
    return (min(lons), min(lats), max(lons), max(lats))


def _col(group, name, n):
    """Read a dataset by name, or an all-NaN array of length n if absent."""
    if group is not None and name in group:
        return group[name][:]
    return np.full(n, np.nan)


def _mask_fill(arr):
    a = np.asarray(arr, dtype=np.float64)
    a[np.abs(a) >= FILL_ABS] = np.nan
    return a


# ---------------------------------------------------------------------------
# STEP 1: Search granules via earthaccess
# ---------------------------------------------------------------------------


def search_granules(polygon: list, date_range: tuple) -> list:
    print("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="netrc")

    bbox = bbox_from_polygon(polygon)
    print(f"Searching ATL08 granules in bbox {bbox}...")

    results = earthaccess.search_data(
        short_name="ATL08",
        bounding_box=bbox,
        temporal=date_range,
        count=-1,  # full run
    )
    print(f"  Found: {len(results)} granule(s)")
    return results


# ---------------------------------------------------------------------------
# STEP 2: Download granules
# ---------------------------------------------------------------------------


def download_granules(granules: list, output_dir: Path) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already downloaded
    existing = {f.name for f in output_dir.glob("ATL08*.h5")}
    to_download = [
        g
        for g in granules
        if not any(link.split("/")[-1] in existing for link in g.data_links())
    ]

    if not to_download:
        print("All granules already downloaded.")
    else:
        print(f"Downloading {len(to_download)} granule(s)...")
        earthaccess.download(to_download, local_path=str(output_dir))

    return sorted(output_dir.glob("ATL08*.h5"))


# ---------------------------------------------------------------------------
# STEP 3: Extract terrain heights (with quality fields), still ellipsoidal
# ---------------------------------------------------------------------------


def extract_terrain_heights(h5_path: Path, aoi_polygon: Polygon) -> list:
    records = []
    granule_name = h5_path.stem
    minx, miny, maxx, maxy = aoi_polygon.bounds

    with h5py.File(h5_path, "r") as f:
        for beam in BEAMS:
            if beam not in f:
                continue

            land_seg = f[beam].get("land_segments")
            if land_seg is None:
                continue

            # ATL08 v006+: terrain heights live in land_segments/terrain/ subgroup
            terrain = land_seg.get("terrain")
            if terrain is None:
                print(f"  Warning: no terrain group in {beam}")
                continue

            lat = land_seg["latitude"][:]
            lon = land_seg["longitude"][:]
            n = len(lat)

            h_te = _mask_fill(_col(terrain, TERRAIN_VAR, n))
            te_unc = _mask_fill(_col(terrain, "h_te_uncertainty", n))
            n_te = np.asarray(_col(terrain, "n_te_photons", n), dtype=np.float64)

            night = np.asarray(_col(land_seg, "night_flag", n), dtype=np.float64)
            layer = np.asarray(_col(land_seg, "layer_flag", n), dtype=np.float64)
            cloud = np.asarray(_col(land_seg, "cloud_flag_atm", n), dtype=np.float64)
            msw = np.asarray(_col(land_seg, "msw_flag", n), dtype=np.float64)
            sat = np.asarray(_col(land_seg, "sat_flag", n), dtype=np.float64)
            terr_flg = np.asarray(_col(land_seg, "terrain_flg", n), dtype=np.float64)
            snr = np.asarray(_col(land_seg, "snr", n), dtype=np.float64)
            dem_h = _mask_fill(_col(land_seg, "dem_h", n))
            delta_time = np.asarray(_col(land_seg, "delta_time", n), dtype=np.float64)

            # ---- Optional 20 m subsegment terrain (finer sampling for crests) ----
            # Quality flags exist at the 100 m segment level and are inherited
            # by the five 20 m subpoints of each segment.
            use20 = (USE_20M_TERRAIN
                     and "latitude_20m" in land_seg
                     and "longitude_20m" in land_seg
                     and "h_te_best_fit_20m" in terrain)
            if use20:
                lat20 = np.asarray(land_seg["latitude_20m"][:], dtype=np.float64)
                lon20 = np.asarray(land_seg["longitude_20m"][:], dtype=np.float64)
                h20 = _mask_fill(terrain["h_te_best_fit_20m"][:])

            # ---- Quality mask ----------------------------------------------
            # Height validity: on the 20 m route a segment counts as valid when
            # ANY of its 20 m heights is valid; the 100 m median may be fill
            # while the subsegments are fine (gating on h_te killed those).
            if use20:
                valid = np.isfinite(h20).any(axis=1)
            else:
                valid = np.isfinite(h_te)
            # QA fields are pass-through when fill/absent: thresholds apply
            # only where the field carries a real value. h_te_uncertainty is
            # frequently fill in ATL08, requiring it finite silently dropped
            # most segments.
            valid &= np.where(np.isfinite(te_unc), te_unc <= MAX_TE_UNCERTAINTY_M, True)
            valid &= np.where(np.isfinite(n_te), n_te >= MIN_TE_PHOTONS, True)
            if FILTER_CLOUDS:
                clear = np.where(np.isfinite(layer), layer == 0, True)
                clear &= np.where(np.isfinite(cloud), cloud <= 1, True)
                valid &= clear
            if FILTER_SATURATION:
                valid &= np.where(np.isfinite(sat), sat == 0, True)
            if FILTER_TERRAIN_FLAG:
                valid &= np.where(np.isfinite(terr_flg), terr_flg == 0, True)
            if REQUIRE_NIGHT:
                valid &= np.where(np.isfinite(night), night == 1, False)

            # ---- Spatial pre-filter by bbox (cheap), then exact polygon test ----
            valid &= (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
            idx = np.where(valid)[0]
            if idx.size == 0:
                continue

            pts = gpd.GeoSeries([Point(lon[i], lat[i]) for i in idx], crs="EPSG:4326")
            inside = pts.within(aoi_polygon).values

            for k, i in enumerate(idx):
                if not inside[k]:
                    continue
                if use20:
                    # Emit the five 20 m subpoints of this quality-passing segment.
                    for j in range(lat20.shape[1]):
                        la, lo, hh = lat20[i, j], lon20[i, j], h20[i, j]
                        if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(hh)):
                            continue
                        if np.abs(la) >= FILL_ABS or np.abs(lo) >= FILL_ABS:
                            continue
                        records.append(
                            {
                                "lat": float(la),
                                "lon": float(lo),
                                "res": "20m",
                                "h_te_ellipsoid": float(hh),
                                "h_te_uncertainty": float(te_unc[i]),
                                "n_te_photons": int(n_te[i]) if np.isfinite(n_te[i]) else -1,
                                "snr": float(snr[i]) if np.isfinite(snr[i]) else np.nan,
                                "night_flag": float(night[i]) if np.isfinite(night[i]) else np.nan,
                                "layer_flag": float(layer[i]) if np.isfinite(layer[i]) else np.nan,
                                "cloud_flag_atm": float(cloud[i]) if np.isfinite(cloud[i]) else np.nan,
                                "msw_flag": float(msw[i]) if np.isfinite(msw[i]) else np.nan,
                                "sat_flag": float(sat[i]) if np.isfinite(sat[i]) else np.nan,
                                "terrain_flg": float(terr_flg[i]) if np.isfinite(terr_flg[i]) else np.nan,
                                "dem_h": float(dem_h[i]) if np.isfinite(dem_h[i]) else np.nan,
                                # small offset keeps the five subpoints ordered along track
                                "delta_time": float(delta_time[i]) + 0.0028 * j
                                if np.isfinite(delta_time[i]) else np.nan,
                                "beam": beam,
                                "granule": granule_name,
                            }
                        )
                    continue
                records.append(
                    {
                        "lat": float(lat[i]),
                        "lon": float(lon[i]),
                        "res": "100m",
                        "h_te_ellipsoid": float(h_te[i]),
                        "h_te_uncertainty": float(te_unc[i]),
                        "n_te_photons": int(n_te[i]) if np.isfinite(n_te[i]) else -1,
                        "snr": float(snr[i]) if np.isfinite(snr[i]) else np.nan,
                        "night_flag": float(night[i]) if np.isfinite(night[i]) else np.nan,
                        "layer_flag": float(layer[i]) if np.isfinite(layer[i]) else np.nan,
                        "cloud_flag_atm": float(cloud[i]) if np.isfinite(cloud[i]) else np.nan,
                        "msw_flag": float(msw[i]) if np.isfinite(msw[i]) else np.nan,
                        "sat_flag": float(sat[i]) if np.isfinite(sat[i]) else np.nan,
                        "terrain_flg": float(terr_flg[i]) if np.isfinite(terr_flg[i]) else np.nan,
                        "dem_h": float(dem_h[i]) if np.isfinite(dem_h[i]) else np.nan,
                        "delta_time": float(delta_time[i]) if np.isfinite(delta_time[i]) else np.nan,
                        "beam": beam,
                        "granule": granule_name,
                    }
                )

    return records


# ---------------------------------------------------------------------------
# STEP 4: Datum conversion (WGS84 ellipsoid -> EGM2008 orthometric)
# ---------------------------------------------------------------------------


def add_orthometric_height(df: pd.DataFrame) -> pd.DataFrame:
    """Add geoid_N and h_te_ortho (EGM2008), matching the Copernicus DSM datum."""
    if USE_PROJ_NETWORK:
        pyproj.network.set_network_enabled(active=True)
    tf = Transformer.from_crs(WGS84_3D, EGM2008_COMPOUND, always_xy=True)
    _, _, h_ortho = tf.transform(df["lon"].values, df["lat"].values,
                                 df["h_te_ellipsoid"].values)
    h_ortho = np.asarray(h_ortho, dtype=np.float64)
    df["geoid_N"] = df["h_te_ellipsoid"].values - h_ortho
    df["h_te_ortho"] = h_ortho
    return df


# ---------------------------------------------------------------------------
# STEP 4b: Crest selection against detected levees
# ---------------------------------------------------------------------------


def flag_near_levees(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flag points lying within LEVEE_BUFFER_M of a detected levee line."""
    from shapely.ops import unary_union

    gdf["near_levee"] = False
    if not Path(DETECTED_LEVEES_GPKG).exists():
        print(f"  detected levees not found, skipping crest selection: {DETECTED_LEVEES_GPKG}")
        return gdf

    levees = gpd.read_file(DETECTED_LEVEES_GPKG, engine="pyogrio")
    if levees.crs is None:
        levees.set_crs(epsg=CRS_METRIC, inplace=True)
    elif levees.crs.to_epsg() != CRS_METRIC:
        levees = levees.to_crs(epsg=CRS_METRIC)

    zone = unary_union(levees.geometry.buffer(LEVEE_BUFFER_M).tolist())
    pts_m = gdf.to_crs(epsg=CRS_METRIC)
    gdf["near_levee"] = pts_m.geometry.within(zone).values
    print(f"  points within {LEVEE_BUFFER_M:.0f} m of a detected levee: "
          f"{int(gdf['near_levee'].sum()):,} / {len(gdf):,}")
    return gdf


def add_prominence(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Along-track prominence for near-levee candidates: point height minus the
    median height of neighbours within PROM_WINDOW_M along the same track,
    excluding the PROM_EXCLUDE_M vicinity so the bump does not raise its own
    baseline. Computed per (granule, beam, res) group, ordered by delta_time.
    """
    gdf["prominence_m"] = np.nan
    if not gdf["near_levee"].any():
        return gdf

    pts_m = gdf.to_crs(epsg=CRS_METRIC)
    xs, ys = pts_m.geometry.x.values, pts_m.geometry.y.values

    for _, idx in gdf.groupby(["granule", "beam", "res"]).groups.items():
        idx = list(idx)
        sub = gdf.loc[idx]
        order = np.argsort(sub["delta_time"].values)
        oidx = np.array(idx)[order]
        x = xs[gdf.index.get_indexer(oidx)]
        y = ys[gdf.index.get_indexer(oidx)]
        h = gdf.loc[oidx, "h_te_ortho"].values
        near = gdf.loc[oidx, "near_levee"].values
        # cumulative along-track distance
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

        prom = np.full(len(oidx), np.nan)
        for i in np.where(near)[0]:
            dd = np.abs(d - d[i])
            m = (dd <= PROM_WINDOW_M) & (dd >= PROM_EXCLUDE_M) & np.isfinite(h)
            if int(m.sum()) >= MIN_NEIGHBORS:
                prom[i] = h[i] - float(np.median(h[m]))
        gdf.loc[oidx, "prominence_m"] = prom

    return gdf


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = gpd.GeoSeries(
        [Point(lo, la) for lo, la in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    aoi_polygon = Polygon(AOI_POLYGON)

    granules = search_granules(AOI_POLYGON, DATE_RANGE)
    if not granules:
        print("No granules found. Check AOI coordinates and date range.")
        return

    h5_files = download_granules(granules, OUTPUT_DIR)
    print(f"\nExtracting terrain heights from {len(h5_files)} file(s)...")

    all_records = []
    for h5_path in h5_files:
        print(f"  {h5_path.name}")
        records = extract_terrain_heights(h5_path, aoi_polygon)
        print(f"    -> {len(records)} points")
        all_records.extend(records)

    print(f"\nTotal points: {len(all_records)}")
    if not all_records:
        print("No records extracted - check AOI, date range or quality thresholds.")
        return

    df = pd.DataFrame(all_records)

    print("Converting to EGM2008 orthometric height (Copernicus DSM datum)...")
    df = add_orthometric_height(df)
    print(f"  geoid undulation N: mean {df['geoid_N'].mean():+.2f} m "
          f"[{df['geoid_N'].min():+.2f}, {df['geoid_N'].max():+.2f}]")

    # Primary elevation field first
    ordered = ["lon", "lat", "res", "beam", "granule", "delta_time",
               "h_te_ortho", "h_te_ellipsoid", "geoid_N",
               "h_te_uncertainty", "n_te_photons", "snr",
               "night_flag", "layer_flag", "cloud_flag_atm", "msw_flag",
               "sat_flag", "terrain_flg", "dem_h"]
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]

    gdf = build_geodataframe(df)

    print("\nCrest selection against detected levees...")
    gdf = flag_near_levees(gdf)
    gdf = add_prominence(gdf)
    gdf["crest_flag"] = gdf["near_levee"] & (gdf["prominence_m"] >= MIN_PROMINENCE_M)
    n_crest = int(gdf["crest_flag"].sum())
    print(f"  crest candidates (prominence >= {MIN_PROMINENCE_M} m): {n_crest:,}")

    pd.DataFrame(gdf.drop(columns="geometry")).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV  -> {OUTPUT_CSV}")

    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_GPKG, driver="GPKG", engine="pyogrio")
    print(f"Saved GPKG -> {OUTPUT_GPKG}")

    if n_crest > 0:
        gdf[gdf["crest_flag"]].to_file(OUTPUT_CREST_GPKG, driver="GPKG", engine="pyogrio")
        print(f"Saved crest subset -> {OUTPUT_CREST_GPKG}")

    h = gdf["h_te_ortho"]
    print(f"\nOrthometric terrain height (EGM2008) summary:")
    print(f"  n:    {len(gdf)}")
    print(f"  min:  {h.min():.2f} m")
    print(f"  max:  {h.max():.2f} m")
    print(f"  mean: {h.mean():.2f} m")
    print(f"  std:  {h.std():.2f} m")
    if n_crest > 0:
        hc = gdf.loc[gdf["crest_flag"], "h_te_ortho"]
        pc = gdf.loc[gdf["crest_flag"], "prominence_m"]
        print(f"\nCrest candidates summary:")
        print(f"  n:            {n_crest}")
        print(f"  height mean:  {hc.mean():.2f} m")
        print(f"  prominence:   mean {pc.mean():.2f} m, max {pc.max():.2f} m")


if __name__ == "__main__":
    main()