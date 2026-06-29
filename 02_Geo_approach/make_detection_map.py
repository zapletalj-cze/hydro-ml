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

PROB_TIF        = Path("detected_levees_prob.tif")
DETECTED_GPKG   = Path("detected_levees.gpkg")
REFERENCE_GPKG  = Path("reference_levees.gpkg")   # e.g. BDOT10k waly/groble in EPSG:2180
REFERENCE_LAYER = None                            # set a layer name if the GPKG has several

OUTPUT_PNG = Path("fig_detection_map.png")

CRS_TARGET = 2180

# Crop window in EPSG:2180 (xmin, ymin, xmax, ymax). Pick a few-km window with a
# clear levee so the detail is visible; the full AOI is too large to read well.
# Set to None to use the full probability-raster extent (may be very large).
CROP_BBOX = None
# Alternative: if CROP_BBOX is None and AUTO_CROP_AROUND_REFERENCE is True, the
# script centres a window of AUTO_CROP_SIZE_M on the reference layer centroid.
AUTO_CROP_AROUND_REFERENCE = True
AUTO_CROP_SIZE_M = 8000

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
    if AUTO_CROP_AROUND_REFERENCE and REFERENCE_GPKG.exists():
        ref = gpd.read_file(REFERENCE_GPKG, layer=REFERENCE_LAYER) if REFERENCE_LAYER \
              else gpd.read_file(REFERENCE_GPKG)
        if ref.crs and ref.crs.to_epsg() != CRS_TARGET:
            ref = ref.to_crs(epsg=CRS_TARGET)
        cx, cy = float(ref.geometry.union_all().centroid.x), float(ref.geometry.union_all().centroid.y)
        h = AUTO_CROP_SIZE_M / 2
        print(f"  auto-crop {AUTO_CROP_SIZE_M} m around reference centroid ({cx:.0f}, {cy:.0f})")
        return (cx - h, cy - h, cx + h, cy + h)
    return None


def main():
    print("Reading probability raster...")
    bbox = resolve_bbox()
    prob, extent = read_prob_window(PROB_TIF, bbox)
    print(f"  raster window: {prob.shape}, extent {[round(e) for e in extent]}")

    bbox_geom = box(extent[0], extent[2], extent[1], extent[3])

    print("Reading vectors...")
    detected = load_vector(DETECTED_GPKG, None, bbox_geom) if DETECTED_GPKG.exists() else None
    reference = load_vector(REFERENCE_GPKG, REFERENCE_LAYER, bbox_geom) if REFERENCE_GPKG.exists() else None
    if detected is not None:
        print(f"  detected lines in window:  {len(detected)}")
    if reference is not None:
        print(f"  reference lines in window: {len(reference)}")

    # mask weak probabilities for a cleaner background
    prob_disp = np.where(prob < PROB_DISPLAY_FLOOR, np.nan, prob)

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    im = ax.imshow(prob_disp, extent=extent, origin="upper",
                   cmap=PROB_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")

    if reference is not None and len(reference):
        reference.plot(ax=ax, color=WARM, linewidth=LINE_WIDTH_REF, zorder=3)
    if detected is not None and len(detected):
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
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
