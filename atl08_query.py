"""
ATL08 track query and terrain height extraction for a polygon AOI.
Uses earthaccess (stable NASA library) instead of deprecated icepyx v1.x.

Extracts h_te_median (median terrain height) from all ATL08 tracks
intersecting a user-defined polygon.

Requirements:
    pip install earthaccess h5py geopandas shapely pyogrio pandas

Earthdata credentials — store in ~/.netrc:
    machine urs.earthdata.nasa.gov
        login YOUR_USERNAME
        password YOUR_PASSWORD
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
from pathlib import Path
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# AOI polygon — Lower Vistula (Toruń → Gdańsk)
AOI_POLYGON = [
    [14.1, 49.0],  # Southwest
    [24.2, 49.0],  # Southeast
    [24.2, 55.0],  # Northeast
    [14.1, 55.0],  # Northwest
    [14.1, 49.0],  # Closing
]

# Alternatively load from file:
# AOI_POLYGON = load_polygon_from_file("aoi.gpkg")

DATE_RANGE = ("2019-01-01", "2025-12-31")   # full mission
OUTPUT_DIR  = Path(r"C:\Computation\data\atl08_PL")
OUTPUT_GPKG = Path(r"C:\Computation\data\atl08_terrain_heights.gpkg")

TERRAIN_VAR = "h_te_median"   # robust terrain height per 100m segment
BEAMS       = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
MIN_PHOTONS = 5               # minimum photons per segment


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
        g for g in granules
        if not any(
            link.split("/")[-1] in existing
            for link in g.data_links()
        )
    ]

    if not to_download:
        print("All granules already downloaded.")
    else:
        print(f"Downloading {len(to_download)} granule(s)...")
        earthaccess.download(to_download, local_path=str(output_dir))

    return sorted(output_dir.glob("ATL08*.h5"))


# ---------------------------------------------------------------------------
# STEP 3: Extract terrain heights
# ---------------------------------------------------------------------------

def extract_terrain_heights(h5_path: Path, aoi_polygon: Polygon) -> list:
    records = []
    granule_name = h5_path.stem

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

            try:
                lat  = land_seg["latitude"][:]
                lon  = land_seg["longitude"][:]
                h_te = terrain[TERRAIN_VAR][:]    # <-- terrain/ subgroup
                n_ph = land_seg["n_seg_ph"][:]
                snr  = land_seg["snr"][:]
            except KeyError as e:
                print(f"  Warning: missing variable in {beam}: {e}")
                continue

            # Quality filter
            valid = (
                (np.abs(h_te) < 1e10) &   # exclude fill values (~3.4e38)
                (n_ph >= MIN_PHOTONS) &    # minimum photon count
                (snr > 0)                  # positive SNR
            )

            lat, lon, h_te, n_ph = (
                lat[valid], lon[valid], h_te[valid], n_ph[valid]
            )

            if len(lat) == 0:
                continue

            # Vectorised spatial filter using numpy + shapely bulk check
            pts = gpd.GeoSeries(
                [Point(lo, la) for lo, la in zip(lon, lat)],
                crs="EPSG:4326",
            )
            inside = pts.within(aoi_polygon)

            for i in np.where(inside)[0]:
                records.append({
                    "lat":        float(lat[i]),
                    "lon":        float(lon[i]),
                    TERRAIN_VAR:  float(h_te[i]),
                    "n_seg_ph":   int(n_ph[i]),
                    "beam":       beam,
                    "granule":    granule_name,
                })

    return records


# ---------------------------------------------------------------------------
# STEP 4: Build GeoDataFrame and save
# ---------------------------------------------------------------------------

def build_geodataframe(records: list) -> gpd.GeoDataFrame:
    if not records:
        raise ValueError("No records extracted — check AOI or date range.")
    df = pd.DataFrame(records)
    geometry = gpd.GeoSeries(
        [Point(r["lon"], r["lat"]) for r in records],
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
        print(f"    → {len(records)} points")
        all_records.extend(records)

    print(f"\nTotal points: {len(all_records)}")

    gdf = build_geodataframe(all_records)
    gdf.to_file(OUTPUT_GPKG, driver="GPKG", engine="pyogrio")
    print(f"Saved → {OUTPUT_GPKG}")

    h = gdf[TERRAIN_VAR]
    print(f"\nTerrain height summary:")
    print(f"  n:    {len(gdf)}")
    print(f"  min:  {h.min():.2f} m")
    print(f"  max:  {h.max():.2f} m")
    print(f"  mean: {h.mean():.2f} m")
    print(f"  std:  {h.std():.2f} m")


if __name__ == "__main__":
    main()