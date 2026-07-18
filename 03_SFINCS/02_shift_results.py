"""
SFINCS results extraction: georeferenced GeoTIFFs from sfincs_map.nc
====================================================================

The SFINCS map output (sfincs_map.nc) carries no CF grid mapping, so GIS
software loads it without CRS and with a wrong origin. This script restores
the georeference from the model itself:

    - x0, y0, dx, dy, mmax, nmax (and epsg if present) are parsed from
      sfincs.inp - the authoritative source of the grid definition
    - zsmax (max water level) and zb (bed level) are read from sfincs_map.nc
      via the GDAL NetCDF driver (no rasterio, no netCDF4 dependency)
    - axis orientation is auto-detected from the x/y coordinate arrays in the
      file (transpose and/or vertical flip as needed), so the export does not
      depend on the storage order of the (m, n) dimensions
    - outputs per model: zsmax.tif (water level [m EGM2008]) and hmax.tif
      (water depth = zsmax - zb, masked below MIN_DEPTH_M)

Also prints a steadiness check if the hourly 'zs' variable is present:
maximum |zs(T) - zs(0.75 T)| over wet cells; values near zero confirm the
constant-Q run reached steady state.

Limitations: regular non-rotated grids only (rotation=0, as built by the
paired-model builder).

Dependencies: GDAL, numpy. No rasterio, no fiona.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

# ============================================================
# CONFIG
# ============================================================

MODEL_ROOTS = [
    Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_baseline"),
    Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_levees"),
]

FALLBACK_EPSG = 2180     # used when sfincs.inp carries no epsg entry
MIN_DEPTH_M   = 0.05     # depths below this are written as nodata
NODATA        = -9999.0
STEADY_FRAC   = 0.75     # steadiness check: compare zs at this fraction vs end

# ============================================================
# INP PARSING
# ============================================================

def parse_inp(inp_path):
    text = inp_path.read_text()
    def get(key, cast=float, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(\S+)", text, flags=re.M)
        if m is None:
            if default is not None:
                return default
            raise KeyError(f"'{key}' not found in {inp_path}")
        return cast(m.group(1))
    inp = {
        "x0": get("x0"), "y0": get("y0"),
        "dx": get("dx"), "dy": get("dy"),
        "mmax": get("mmax", int), "nmax": get("nmax", int),
        "rotation": get("rotation", float, 0.0),
        "epsg": get("epsg", int, FALLBACK_EPSG),
    }
    if abs(inp["rotation"]) > 1e-6:
        raise NotImplementedError("Rotated grids are not supported here.")
    return inp


# ============================================================
# NETCDF READING (GDAL subdatasets)
# ============================================================

def read_var(nc_path, var):
    """Read a NetCDF variable as (bands, rows, cols) float64 array."""
    ds = gdal.Open(f'NETCDF:"{nc_path}":{var}')
    arr = ds.ReadAsArray().astype(np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    # GDAL netcdf fill values arrive as large numbers; mask them
    band = ds.GetRasterBand(1)
    fill = band.GetNoDataValue()
    if fill is not None:
        arr[arr == fill] = np.nan
    arr[np.abs(arr) > 1e20] = np.nan
    ds = None
    return arr


def orient(arr2d, x2d, y2d):
    """Return the array in GeoTIFF orientation (row 0 = north, col 0 = west),
    deciding transpose/flip from the coordinate arrays themselves."""
    # transpose so that x varies along columns
    dx_along_cols = np.nanmax(np.abs(np.diff(x2d, axis=1)))
    dx_along_rows = np.nanmax(np.abs(np.diff(x2d, axis=0)))
    if dx_along_rows > dx_along_cols:
        arr2d, x2d, y2d = arr2d.T, x2d.T, y2d.T
    # flip columns so x increases eastward
    if x2d[0, 0] > x2d[0, -1]:
        arr2d, x2d, y2d = arr2d[:, ::-1], x2d[:, ::-1], y2d[:, ::-1]
    # flip rows so y decreases downward (north up)
    if y2d[0, 0] < y2d[-1, 0]:
        arr2d, x2d, y2d = arr2d[::-1, :], x2d[::-1, :], y2d[::-1, :]
    return arr2d, x2d, y2d


def write_gtiff(path, arr2d, inp, nodata=NODATA):
    rows, cols = arr2d.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), cols, rows, 1, gdal.GDT_Float32,
                    options=["COMPRESS=LZW", "TILED=YES"])
    top = inp["y0"] + inp["nmax"] * inp["dy"]
    ds.SetGeoTransform((inp["x0"], inp["dx"], 0.0, top, 0.0, -inp["dy"]))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(inp["epsg"])
    ds.SetProjection(srs.ExportToWkt())
    out = np.where(np.isfinite(arr2d), arr2d, nodata).astype(np.float32)
    band = ds.GetRasterBand(1)
    band.WriteArray(out)
    band.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    print(f"  saved {path.name}  ({rows}x{cols}, EPSG:{inp['epsg']})")


# ============================================================
# PER-MODEL EXTRACTION
# ============================================================

def process_model(root):
    print(f"\n=== {root.name}")
    nc = root / "sfincs_map.nc"
    inp = parse_inp(root / "sfincs.inp")
    if not nc.exists():
        print("  sfincs_map.nc not found - run the model first")
        return

    x2d = read_var(nc, "x")[0]
    y2d = read_var(nc, "y")[0]

    # ---- zsmax: max over all output slices ----
    zsmax = np.nanmax(read_var(nc, "zsmax"), axis=0)
    zsmax_o, _, _ = orient(zsmax, x2d.copy(), y2d.copy())
    write_gtiff(root / "zsmax.tif", zsmax_o, inp)

    # ---- depth: zsmax - zb ----
    try:
        zb = read_var(nc, "zb")[0]
        hmax = zsmax - zb
        hmax[hmax < MIN_DEPTH_M] = np.nan
        hmax_o, _, _ = orient(hmax, x2d.copy(), y2d.copy())
        write_gtiff(root / "hmax.tif", hmax_o, inp)
        wet = np.isfinite(hmax_o)
        if wet.any():
            print(f"  wet area: {wet.sum() * inp['dx'] * inp['dy'] / 1e6:.2f} km2, "
                  f"max depth {np.nanmax(hmax_o):.2f} m")
    except Exception as e:
        print(f"  zb not readable ({e}); hmax.tif skipped")

    # ---- steadiness check on hourly zs, if present ----
    try:
        zs = read_var(nc, "zs")
        n_t = zs.shape[0]
        if n_t >= 4:
            i75 = int(round(STEADY_FRAC * (n_t - 1)))
            diff = np.abs(zs[-1] - zs[i75])
            wet = np.isfinite(zs[-1]) & np.isfinite(zs[i75])
            d_max = float(np.nanmax(diff[wet])) if wet.any() else float("nan")
            d_med = float(np.nanmedian(diff[wet])) if wet.any() else float("nan")
            print(f"  steadiness: |zs(T) - zs({STEADY_FRAC:.2f} T)| "
                  f"median {d_med:.3f} m, max {d_max:.3f} m "
                  f"({'OK' if d_max < 0.05 else 'NOT steady - extend SIM_HOURS'})")
    except Exception:
        print("  hourly zs not present; steadiness check skipped")


def main():
    for root in MODEL_ROOTS:
        process_model(root)
    print("\nDone. Load zsmax.tif / hmax.tif in QGIS - georeference is set "
          "from sfincs.inp (x0, y0, dx, dy, EPSG).")


if __name__ == "__main__":
    main()