"""
make_detection_map.py
=====================
Render a presentation-quality detection map for the elevation-based levee
detection: the model's probability raster as background, the extracted detected
centerlines, and the reference levees on top, cropped to a chosen window.

Reads the outputs of the inference script (04b_interference.py / v0.6):
    detected_levees_prob.tif    stitched probability raster (EPSG:2180)
    detected_levees.gpkg        detected centerlines (EPSG:2180)
and a reference levee vector you point it at (e.g. BDOT10k waly/groble).

Needs GDAL + geopandas + matplotlib + shapely. Run locally (the sandbox that
prepared this does not have GDAL). Edit the CONFIG block, then:
    python make_detection_map.py

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal
gdal.UseExceptions()

# ============================================================
# CONFIG  -- edit these
# ============================================================

PROB_TIF        = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\interference_outputs\detected_levees_prob.tif")
DETECTED_GPKG   = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\interference_outputs\detected_levees.gpkg")
REFERENCE_GPKG  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg")   # e.g. BDOT10k waly/groble in EPSG:2180
REFERENCE_LAYER = None                            # set a layer name if the GPKG has several

OUTPUT_PNG = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\thesis_figures\fix_map01.png")

CRS_TARGET = 2180

# Crop window in EPSG:2180 (xmin, ymin, xmax, ymax). Pick a few-km window with a
# clear levee so the detail is visible; the full AOI is too large to read well.
# Set to None to use the full probability-raster extent (may be very large).
CROP_BBOX = None
# Alternative: if CROP_BBOX is None and AUTO_CROP_AROUND_VECTOR is True, the
# script centres a window of AUTO_CROP_SIZE_M on a vector centroid.
# AUTO_CROP_SOURCE: "detected" (default) or "reference".
AUTO_CROP_AROUND_VECTOR = True
AUTO_CROP_SOURCE = "detected"
AUTO_CROP_SIZE_M = 8000
MAX_WINDOW_SEARCH_CANDIDATES = 40
# Pick which found candidate window to use (0 = first, 1 = second, ...).
AUTO_CROP_CANDIDATE_INDEX = 1
# Number of different map images to export (recommended 3-5).
NUM_OUTPUT_IMAGES = 5

# Display
PROB_DISPLAY_FLOOR = 0.05   # probabilities below this are shown as transparent
SHOW_COLORBAR      = True
LINE_WIDTH_DET     = 1.8
LINE_WIDTH_REF     = 1.8

# Colours (deck palette)
TEAL = "#0E7C7B"   # detected
WARM = "#C2410C"   # reference
INK  = "#1E293B"
MUTED = "#64748B"

# Probability colourmap: white -> light teal -> deep teal
PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob_teal", ["#FFFFFF", "#CFE8E7", "#7FB7B6", "#0E7C7B", "#0A5A59"]
)

# ============================================================
# RASTER WINDOW READING
# ============================================================

def read_prob_window(prob_path, bbox):
    """
    Read the probability raster, optionally cropped to bbox (xmin,ymin,xmax,ymax)
    in CRS_TARGET. Returns (array, extent) where extent = [xmin,xmax,ymin,ymax]
    for imshow(origin='upper').
    """
    ds = gdal.Open(str(prob_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open {prob_path}")
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ox, px, _, oy, _, py = gt          # py is negative
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    if bbox is None:
        arr = band.ReadAsArray().astype(np.float32)
        xmin, xmax = ox, ox + nx * px
        ymax, ymin = oy, oy + ny * py
        ds = None
    else:
        xmin, ymin, xmax, ymax = bbox
        col0 = int(np.floor((xmin - ox) / px))
        col1 = int(np.ceil((xmax - ox) / px))
        row0 = int(np.floor((ymax - oy) / py))   # py negative -> top row
        row1 = int(np.ceil((ymin - oy) / py))
        col0, col1 = max(0, col0), min(nx, col1)
        row0, row1 = max(0, row0), min(ny, row1)
        if col1 <= col0 or row1 <= row0:
            ds = None
            raise ValueError("CROP_BBOX does not overlap the probability raster.")
        arr = band.ReadAsArray(col0, row0, col1 - col0, row1 - row0).astype(np.float32)
        xmin = ox + col0 * px
        xmax = ox + col1 * px
        ymax = oy + row0 * py
        ymin = oy + row1 * py
        ds = None

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, [xmin, xmax, ymin, ymax]


# ============================================================
# VECTOR LOADING
# ============================================================

def load_vector(path, layer, bbox_geom):
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=CRS_TARGET, inplace=True)
    elif gdf.crs.to_epsg() != CRS_TARGET:
        gdf = gdf.to_crs(epsg=CRS_TARGET)
    if bbox_geom is not None:
        gdf = gpd.clip(gdf, bbox_geom)
    return gdf


# ============================================================
# MAIN
# ============================================================

def resolve_bbox():
    if CROP_BBOX is not None:
        return CROP_BBOX
    if AUTO_CROP_AROUND_VECTOR:
        source = str(AUTO_CROP_SOURCE).strip().lower()
        if source == "detected" and DETECTED_GPKG.exists():
            vec = gpd.read_file(DETECTED_GPKG)
            label = "detected"
        elif REFERENCE_GPKG.exists():
            vec = gpd.read_file(REFERENCE_GPKG, layer=REFERENCE_LAYER) if REFERENCE_LAYER \
                  else gpd.read_file(REFERENCE_GPKG)
            label = "reference"
        elif DETECTED_GPKG.exists():
            vec = gpd.read_file(DETECTED_GPKG)
            label = "detected"
        else:
            return None

        if vec.crs and vec.crs.to_epsg() != CRS_TARGET:
            vec = vec.to_crs(epsg=CRS_TARGET)
        cx, cy = float(vec.geometry.union_all().centroid.x), float(vec.geometry.union_all().centroid.y)
        h = AUTO_CROP_SIZE_M / 2
        print(f"  auto-crop {AUTO_CROP_SIZE_M} m around {label} centroid ({cx:.0f}, {cy:.0f})")
        return (cx - h, cy - h, cx + h, cy + h)
    return None


def _raster_extent_box(prob_path):
    ds = gdal.Open(str(prob_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open {prob_path}")
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ox, px, _, oy, _, py = gt
    xmin, xmax = ox, ox + nx * px
    ymax, ymin = oy, oy + ny * py
    ds = None
    return box(xmin, ymin, xmax, ymax)


def _load_vector_full(path, layer):
    if not path.exists():
        return None
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=CRS_TARGET, inplace=True)
    elif gdf.crs.to_epsg() != CRS_TARGET:
        gdf = gdf.to_crs(epsg=CRS_TARGET)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    return gdf


def find_bboxes_with_levee(max_count=None):
    """
    Search crop windows containing levees and overlapping the raster.
    Returns a list of (bbox, label, px, py).
    """
    h = AUTO_CROP_SIZE_M / 2
    raster_geom = _raster_extent_box(PROB_TIF)

    source = str(AUTO_CROP_SOURCE).strip().lower()
    if source == "detected":
        source_specs = [("detected", DETECTED_GPKG, None), ("reference", REFERENCE_GPKG, REFERENCE_LAYER)]
    else:
        source_specs = [("reference", REFERENCE_GPKG, REFERENCE_LAYER), ("detected", DETECTED_GPKG, None)]

    found_windows = []
    seen = set()

    for label, path, layer in source_specs:
        vec = _load_vector_full(path, layer)
        if vec is None or vec.empty:
            continue

        vec = gpd.clip(vec, raster_geom)
        if vec.empty:
            continue

        candidates = [vec.geometry.union_all().centroid]
        n = min(len(vec), MAX_WINDOW_SEARCH_CANDIDATES)
        if n > 0:
            idxs = np.linspace(0, len(vec) - 1, n, dtype=int)
            candidates.extend(list(vec.iloc[idxs].geometry.representative_point().values))

        for p in candidates:
            bbox = (float(p.x - h), float(p.y - h), float(p.x + h), float(p.y + h))
            window_geom = box(*bbox)
            if int(vec.intersects(window_geom).sum()) > 0:
                key = tuple(round(v, 2) for v in bbox)
                if key not in seen:
                    seen.add(key)
                    found_windows.append((bbox, label, float(p.x), float(p.y)))
                    if max_count is not None and len(found_windows) >= max_count:
                        return found_windows

    return found_windows


def load_window_data(bbox):
    prob, extent = read_prob_window(PROB_TIF, bbox)
    bbox_geom = box(extent[0], extent[2], extent[1], extent[3])
    detected = load_vector(DETECTED_GPKG, None, bbox_geom) if DETECTED_GPKG.exists() else None
    reference = load_vector(REFERENCE_GPKG, REFERENCE_LAYER, bbox_geom) if REFERENCE_GPKG.exists() else None
    has_detected = detected is not None and len(detected) > 0
    has_reference = reference is not None and len(reference) > 0
    return prob, extent, detected, reference, has_detected, has_reference


def output_path_for_index(base_path, index):
    return base_path.with_name(f"{base_path.stem}_{index:02d}{base_path.suffix}")


def render_and_save(prob, extent, detected, reference, has_detected, has_reference, output_png):
    # mask weak probabilities for a cleaner background
    prob_disp = np.where(prob < PROB_DISPLAY_FLOOR, np.nan, prob)

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    im = ax.imshow(prob_disp, extent=extent, origin="upper",
                   cmap=PROB_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")

    if has_reference:
        reference.plot(ax=ax, color=WARM, linewidth=LINE_WIDTH_REF, zorder=3)
    if has_detected:
        detected.plot(ax=ax, color=TEAL, linewidth=LINE_WIDTH_DET, zorder=4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("X [m, EPSG:2180]", color=MUTED, fontsize=9)
    ax.set_ylabel("Y [m, EPSG:2180]", color=MUTED, fontsize=9)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("Detekce hrází z výškového modelu (povodí Odra)", color=INK, fontsize=13)

    handles = [
        Line2D([0], [0], color=TEAL, lw=2.4, label="detekované hráze"),
        Line2D([0], [0], color=WARM, lw=2.4, label="referenční hráze (BDOT10k)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=10)

    if SHOW_COLORBAR:
        cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cbar.set_label("pravděpodobnost hráze", color=MUTED, fontsize=9)
        cbar.ax.tick_params(labelsize=8, colors=MUTED)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")


def main():
    target_n = max(1, min(int(NUM_OUTPUT_IMAGES), 5))
    print(f"Preparing up to {target_n} output image(s)...")

    bboxes = []
    bbox0 = resolve_bbox()
    if bbox0 is not None:
        bboxes.append((bbox0, "initial", None, None))

    need_more = target_n - len(bboxes)
    if need_more > 0:
        found = find_bboxes_with_levee(max_count=max(target_n * 3, MAX_WINDOW_SEARCH_CANDIDATES))
        if found:
            start = max(0, int(AUTO_CROP_CANDIDATE_INDEX)) % len(found)
            for i in range(len(found)):
                if len(bboxes) >= target_n:
                    break
                cand = found[(start + i) % len(found)]
                bbox = cand[0]
                key = tuple(round(v, 2) for v in bbox)
                if any(tuple(round(v, 2) for v in existing[0]) == key for existing in bboxes):
                    continue
                bboxes.append(cand)

    if not bboxes:
        print("No candidate windows found. Skipping image export.")
        return

    saved = 0
    for i, (bbox, label, px, py) in enumerate(bboxes, start=1):
        print(f"Reading probability raster for window {i}/{len(bboxes)}...")
        prob, extent, detected, reference, has_detected, has_reference = load_window_data(bbox)
        print(f"  raster window: {prob.shape}, extent {[round(e) for e in extent]}")
        if detected is not None:
            print(f"  detected lines in window:  {len(detected)}")
        if reference is not None:
            print(f"  reference lines in window: {len(reference)}")

        if not (has_detected or has_reference):
            print("  no levees in this window, skipping")
            continue

        output_png = output_path_for_index(OUTPUT_PNG, i)
        if label != "initial" and px is not None and py is not None:
            print(f"  using {label} candidate near ({px:.0f}, {py:.0f})")
        render_and_save(prob, extent, detected, reference, has_detected, has_reference, output_png)
        saved += 1

    if saved == 0:
        print("No levees found inside any searched window. Skipping image export.")
    else:
        print(f"Exported {saved} image(s).")


if __name__ == "__main__":
    main()
