"""
Gate Test: Detekce hrází z Copernicus DSM
=========================================
Cíl: Ověřit, zda jsou protipovodňové hráze viditelné v 30m Copernicus DSM
     a kolik ICESat-2 crossings přes hráze existuje.

Výstup: 3 figury + statistiky do konzole
  1. Příčné profily DSM přes známé hráze
  2. TPI vizualizace na různých radiusech
  3. ICESat-2 crossing statistiky a ukázkové profily

Prerekvizity:
  pip install geopandas rasterio numpy scipy matplotlib shapely pyproj
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from scipy.ndimage import uniform_filter, minimum_filter
from shapely.geometry import LineString, box
from shapely.ops import transform as shapely_transform
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pyproj

# =============================================================================
# KONFIGURACE — UPRAV CESTY K DATŮM
# =============================================================================

BDOT_GPKG = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\OT_BUZM_L_Poland_files_selected.gpkg"
ICESAT2_GPKG = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sat_lidar\01_data\ICE_SAT\ATL08\atl08_terrain_heights.gpkg"
COPDEM_TIFF = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\COP_DSM\COP_DSM_Poland_2180_c.tif"  # tile pokrývající testovací oblast
LIDAR_DTM = None  # volitelné: cesta k 1m DTM pro srovnání, None = přeskočit

# testovací oblast — dolní Visla (Toruń–Grudziądz)
BBOX_WGS84 = (18.4, 52.9, 19.0, 53.5)

# parametry
PROFILE_LENGTH_M = 800  # délka příčného profilu na každou stranu od hráze
PROFILE_STEP_M = 10  # krok vzorkování podél profilu
N_PROFILES = 20  # počet profilů k vizualizaci
TPI_RADII_PX = [3, 5, 7, 10]  # TPI radiusy v pixelech (× 30m = 90–300 m)
LEVEE_BUFFER_M = 50  # buffer kolem hráze pro ICESat-2 matching


def load_dsm_window(dsm_path, bbox_wgs84):
    with rasterio.open(dsm_path) as src:
        if src.crs.to_epsg() != 4326:
            bbox_native = transform_bounds(4326, src.crs, *bbox_wgs84)
        else:
            bbox_native = bbox_wgs84
        window = from_bounds(*bbox_native, src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, win_transform, src.crs


def compute_tpi(dem, radius_px):
    mean_elev = uniform_filter(dem, size=2 * radius_px + 1, mode="nearest")
    return dem - mean_elev


def compute_relative_elevation(dem, radius_px):
    min_elev = minimum_filter(dem, size=2 * radius_px + 1, mode="nearest")
    return dem - min_elev


def create_perpendicular_profile(line_geom, position_along, length, step):
    point = line_geom.interpolate(position_along)
    d = 0.5
    p1 = line_geom.interpolate(max(0, position_along - d))
    p2 = line_geom.interpolate(min(line_geom.length, position_along + d))
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0:
        return None, None
    perp_dx = -dy / norm
    perp_dy = dx / norm

    distances = np.arange(-length, length + step, step)
    coords = [(point.x + perp_dx * d, point.y + perp_dy * d) for d in distances]
    return distances, coords


def sample_raster_at_coords(data, transform, coords):
    values = []
    for x, y in coords:
        col, row = ~transform * (x, y)
        col, row = int(round(col)), int(round(row))
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            values.append(data[row, col])
        else:
            values.append(np.nan)
    return np.array(values)


def reproject_gdf(gdf, target_crs):
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


# =============================================================================
# TEST 1: PROFILY DSM PŘES ZNÁMÉ HRÁZE
# =============================================================================


def test_1_profiles(dsm_data, dsm_transform, dsm_crs, levees_gdf):
    print("\n" + "=" * 60)
    print("TEST 1: Příčné profily DSM přes známé hráze")
    print("=" * 60)

    levees = reproject_gdf(levees_gdf, dsm_crs)

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle(
        "Příčné profily Copernicus DSM přes známé hráze (BDOT10k)", fontsize=14
    )
    axes = axes.flatten()

    profile_count = 0
    prominences = []

    for _, row in levees.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            geom = list(geom.geoms)[0]
        if geom.length < PROFILE_LENGTH_M:
            continue

        positions = np.linspace(geom.length * 0.2, geom.length * 0.8, 5)

        for pos in positions:
            if profile_count >= N_PROFILES:
                break

            distances, coords = create_perpendicular_profile(
                geom, pos, PROFILE_LENGTH_M, PROFILE_STEP_M
            )
            if distances is None:
                continue

            elevations = sample_raster_at_coords(dsm_data, dsm_transform, coords)

            if np.all(np.isnan(elevations)):
                continue

            center_idx = len(elevations) // 2
            center_elev = elevations[center_idx]
            margin = 5
            left_min = (
                np.nanmin(elevations[: center_idx - margin])
                if center_idx > margin
                else np.nan
            )
            right_min = (
                np.nanmin(elevations[center_idx + margin :])
                if center_idx + margin < len(elevations)
                else np.nan
            )
            surrounding_min = np.nanmin([left_min, right_min])
            prominence = center_elev - surrounding_min

            if not np.isnan(prominence):
                prominences.append(prominence)

            ax = axes[profile_count]
            ax.plot(distances, elevations, "b-", linewidth=1)
            ax.axvline(0, color="r", linestyle="--", linewidth=1, label="hráz")
            ax.set_title(f"P{profile_count + 1}: Δh={prominence:.1f}m", fontsize=9)
            ax.set_xlabel("m")
            ax.set_ylabel("m n.m.")
            ax.grid(True, alpha=0.3)
            profile_count += 1

        if profile_count >= N_PROFILES:
            break

    plt.tight_layout()
    plt.savefig("gate_test_1_profiles.png", dpi=150)
    plt.close()

    if prominences:
        prominences = np.array(prominences)
        print(f"\nPočet profilů: {len(prominences)}")
        print(f"Prominence hráze v DSM:")
        print(f"  Medián:  {np.median(prominences):.2f} m")
        print(f"  Průměr:  {np.mean(prominences):.2f} m")
        print(f"  Min:     {np.min(prominences):.2f} m")
        print(f"  Max:     {np.max(prominences):.2f} m")
        print(f"  >1m:     {np.sum(prominences > 1) / len(prominences) * 100:.0f} %")
        print(f"  >2m:     {np.sum(prominences > 2) / len(prominences) * 100:.0f} %")
        print(f"  >3m:     {np.sum(prominences > 3) / len(prominences) * 100:.0f} %")

        verdict = "GO" if np.median(prominences) > 1.0 else "ZVÁŽIT"
        print(f"\n→ VERDIKT TESTU 1: {verdict}")
        if verdict == "GO":
            print("  Hráze jsou v DSM viditelné, mediánová prominence > 1 m")
        else:
            print("  Prominence je nízká, ML detekce bude velmi obtížná")
    else:
        print("Žádné profily nebyly extrahovány — zkontroluj data")

    return prominences


# =============================================================================
# TEST 2: TPI NA RŮZNÝCH RADIUSECH
# =============================================================================


def test_2_tpi(dsm_data, dsm_transform, dsm_crs, levees_gdf):
    print("\n" + "=" * 60)
    print("TEST 2: TPI vizualizace na různých radiusech")
    print("=" * 60)

    levees = reproject_gdf(levees_gdf, dsm_crs)
    bbox_geom = box(*levees.total_bounds)
    cx, cy = bbox_geom.centroid.x, bbox_geom.centroid.y

    view_size = 3000  # metrů
    view_bounds = (cx - view_size, cy - view_size, cx + view_size, cy + view_size)

    col_min, row_max = ~dsm_transform * (view_bounds[0], view_bounds[1])
    col_max, row_min = ~dsm_transform * (view_bounds[2], view_bounds[3])
    r_min, r_max = int(max(0, row_min)), int(min(dsm_data.shape[0], row_max))
    c_min, c_max = int(max(0, col_min)), int(min(dsm_data.shape[1], col_max))

    dsm_crop = dsm_data[r_min:r_max, c_min:c_max]

    n_panels = len(TPI_RADII_PX) + 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("TPI a relativní elevace na různých radiusech", fontsize=14)
    axes = axes.flatten()

    extent = [view_bounds[0], view_bounds[2], view_bounds[1], view_bounds[3]]

    axes[0].imshow(dsm_crop, extent=extent, cmap="terrain", origin="upper")
    axes[0].set_title("Copernicus DSM")

    for levee_geom in levees.geometry:
        if levee_geom is None:
            continue
        if levee_geom.geom_type == "MultiLineString":
            for part in levee_geom.geoms:
                xs, ys = part.xy
                axes[0].plot(xs, ys, "r-", linewidth=0.8)
        else:
            xs, ys = levee_geom.xy
            axes[0].plot(xs, ys, "r-", linewidth=0.8)

    best_radius = None
    best_contrast = 0

    for i, radius in enumerate(TPI_RADII_PX):
        tpi = compute_tpi(dsm_crop, radius)
        vmax = np.nanpercentile(np.abs(tpi), 98)

        ax = axes[i + 1]
        ax.imshow(
            tpi, extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper"
        )
        ax.set_title(f"TPI r={radius}px ({radius * 30}m)")

        for levee_geom in levees.geometry:
            if levee_geom is None:
                continue
            if levee_geom.geom_type == "MultiLineString":
                for part in levee_geom.geoms:
                    xs, ys = part.xy
                    ax.plot(xs, ys, "k-", linewidth=0.8)
            else:
                xs, ys = levee_geom.xy
                ax.plot(xs, ys, "k-", linewidth=0.8)

        print(
            f"  TPI r={radius}px ({radius * 30}m): "
            f"mean={np.nanmean(tpi):.2f}, std={np.nanstd(tpi):.2f}, "
            f"max={np.nanmax(tpi):.2f}"
        )

    rel_elev = compute_relative_elevation(dsm_crop, 5)
    ax = axes[len(TPI_RADII_PX) + 1]
    vmax = np.nanpercentile(rel_elev, 98)
    ax.imshow(rel_elev, extent=extent, cmap="hot_r", vmin=0, vmax=vmax, origin="upper")
    ax.set_title("Relative elevation (r=5px)")

    for levee_geom in levees.geometry:
        if levee_geom is None:
            continue
        if levee_geom.geom_type == "MultiLineString":
            for part in levee_geom.geoms:
                xs, ys = part.xy
                ax.plot(xs, ys, "cyan", linewidth=0.8)
        else:
            xs, ys = levee_geom.xy
            ax.plot(xs, ys, "cyan", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("gate_test_2_tpi.png", dpi=150)
    plt.close()

    print(f"\n→ Vizuální inspekce: gate_test_2_tpi.png")
    print(f"  Otázka: 'Svítí' hráze (černé linie) v TPI mapách?")


# =============================================================================
# TEST 3: ICESat-2 CROSSINGS
# =============================================================================


def test_3_icesat2(levees_gdf, icesat2_path, bbox_wgs84):
    print("\n" + "=" * 60)
    print("TEST 3: ICESat-2 crossings přes hráze")
    print("=" * 60)

    bbox_geom = box(*bbox_wgs84)
    levees_wgs = reproject_gdf(levees_gdf, "EPSG:4326")

    try:
        icesat2 = gpd.read_file(icesat2_path, bbox=bbox_wgs84)
    except Exception as e:
        print(f"Chyba při načítání ICESat-2: {e}")
        print("Zkus omezit bbox nebo zkontroluj formát GPKG")
        return

    print(f"ICESat-2 bodů v testovací oblasti: {len(icesat2)}")

    if len(icesat2) == 0:
        print("Žádné ICESat-2 body v testovací oblasti")
        return

    print(f"ICESat-2 sloupce: {list(icesat2.columns)}")

    utm_crs = levees_wgs.estimate_utm_crs()
    levees_utm = levees_wgs.to_crs(utm_crs)
    icesat2_utm = icesat2.to_crs(utm_crs)

    levee_buffer = levees_utm.buffer(LEVEE_BUFFER_M)
    levee_buffer_gdf = gpd.GeoDataFrame(geometry=levee_buffer, crs=utm_crs)

    crossings = gpd.sjoin(icesat2_utm, levee_buffer_gdf, predicate="within")

    print(f"\nICESat-2 body na hrázích (buffer {LEVEE_BUFFER_M}m): {len(crossings)}")
    print(f"Celkem ICESat-2 bodů v oblasti: {len(icesat2)}")
    print(f"Poměr: {len(crossings) / max(len(icesat2), 1) * 100:.2f} %")

    h_col = None
    for candidate in [
        "h_te_mean",
        "h_mean_canopy",
        "h_te_best_fit",
        "elevation",
        "h_te_median",
    ]:
        if candidate in crossings.columns:
            h_col = candidate
            break

    if h_col and len(crossings) > 0:
        print(f"\nElevační sloupec: {h_col}")
        print(
            f"  Rozsah elevace na hrázích: {crossings[h_col].min():.1f} – {crossings[h_col].max():.1f} m"
        )

    if len(crossings) > 100:
        verdict = "GO"
    elif len(crossings) > 20:
        verdict = "OMEZENÉ — proof of concept možný"
    else:
        verdict = "NEDOSTATEČNÉ — ICESat-2 odhad výšky bude nespolehlivý"

    print(f"\n→ VERDIKT TESTU 3: {verdict}")

    return crossings


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("GATE TEST: Detekce hrází z Copernicus DSM")
    print("=" * 60)

    for path, name in [(BDOT_GPKG, "BDOT10k"), (COPDEM_TIFF, "Copernicus DSM")]:
        if not Path(path).exists():
            print(f"CHYBA: {name} nenalezen: {path}")
            print("Uprav cesty v sekci KONFIGURACE")
            return

    print("Načítám BDOT10k hráze...")
    levees = gpd.read_file(BDOT_GPKG)
    print(f"  Počet features: {len(levees)}")
    print(f"  CRS: {levees.crs}")

    bbox_geom = box(*BBOX_WGS84)
    levees_wgs = reproject_gdf(levees, "EPSG:4326")
    levees_clip = levees_wgs.clip(bbox_geom)
    print(f"  Features v testovací oblasti: {len(levees_clip)}")

    if len(levees_clip) == 0:
        print("CHYBA: Žádné hráze v testovací oblasti — uprav BBOX_WGS84")
        return

    print("\nNačítám Copernicus DSM...")
    dsm_data, dsm_transform, dsm_crs = load_dsm_window(COPDEM_TIFF, BBOX_WGS84)
    print(f"  Rozměr: {dsm_data.shape}")
    print(f"  Elevace: {np.nanmin(dsm_data):.1f} – {np.nanmax(dsm_data):.1f} m")
    print(f"  CRS: {dsm_crs}")

    prominences = test_1_profiles(dsm_data, dsm_transform, dsm_crs, levees_clip)
    test_2_tpi(dsm_data, dsm_transform, dsm_crs, levees_clip)

    if Path(ICESAT2_GPKG).exists():
        test_3_icesat2(levees, ICESAT2_GPKG, BBOX_WGS84)
    else:
        print(f"\nICESat-2 GPKG nenalezen: {ICESAT2_GPKG} — test 3 přeskočen")

    print("\n" + "=" * 60)
    print("GATE TEST DOKONČEN")
    print("=" * 60)
    print("\nVýstupy:")
    print("  gate_test_1_profiles.png  — příčné profily DSM přes hráze")
    print("  gate_test_2_tpi.png       — TPI vizualizace")
    print("\nDalší kroky:")
    print("  1. Vizuálně zkontroluj profily — je prominence > 1m?")
    print("  2. Zkontroluj TPI — 'svítí' hráze na nějakém radiusu?")
    print("  3. Zkontroluj ICESat-2 — je dostatek crossings?")


if __name__ == "__main__":
    main()
