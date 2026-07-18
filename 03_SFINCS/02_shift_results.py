from os.path import join
from hydromt_sfincs import SfincsModel
import xarray as xr
import rioxarray  # noqa: F401  (enables .rio accessor)

sfincs_root = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_levees"

mod = SfincsModel(sfincs_root, mode="r")
mod.output.read()

# --- 1) Get hmax and collapse time dim if present ---
hmax = mod.output.data["hmax"]
if "timemax" in hmax.dims:
    hmax = hmax.max("timemax")
elif "time" in hmax.dims:
    hmax = hmax.max("time")

print("hmax dims :", hmax.dims, "| shape:", hmax.shape)

# --- 2) Real-world coords from the grid definition (sfincs.inp) ---
grid = mod.grid
coords = grid.coordinates          # dict
print("coord keys:", list(coords), "| rotation:", grid.rotation)

# --- 3) Build a georeferenced DataArray ---
if "x" in coords and "y" in coords:
    # Non-rotated grid — 1D x and y
    x_vals = coords["x"][1]        # ("x", array)  -> take array
    y_vals = coords["y"][1]

    hmax_geo = xr.DataArray(
        data=hmax.values,
        dims=("y", "x"),
        coords={"y": y_vals, "x": x_vals},
        name="hmax",
    )
else:
    # Rotated grid — 2D xc, yc (can't write directly to GeoTIFF without reproject)
    raise NotImplementedError(
        "Grid is rotated; use utils.downscale_floodmap or reproject to a regular raster."
    )

# --- 4) CRS from sfincs.inp epsg ---
epsg = mod.config.get("epsg")
print("EPSG from sfincs.inp:", epsg)
hmax_geo = hmax_geo.rio.write_crs(f"EPSG:{epsg}")
hmax_geo = hmax_geo.rio.set_spatial_dims("x", "y")

# --- 5) Mask dry cells and export ---
hmax_geo = hmax_geo.where(hmax_geo > 0.05)

out = join(sfincs_root, "hmax.tif")
hmax_geo.rio.to_raster(out, compress="LZW", dtype="float32")
print("Written:", out)