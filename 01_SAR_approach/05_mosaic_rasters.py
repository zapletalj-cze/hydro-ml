from gis import MosaicRasters
import geopandas as gpd
from gis import Vector, Raster
from osgeo import gdal, osr
import os

TILEINDEX_ROOT = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\sentinel1_data\processed_selected"
AOI = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\AOI_Poland.gpkg"
SNAP_RASTER = r"B:\01_Projects\154_Poland_Flood_v3\01_MD\01_HAZARD\00_GIS\01_SnapRaster\SnapRaster_PL_10m_2180.tif"
files = [
    "asc_VV_tile_index.gpkg",
    "desc_VV_tile_index.gpkg",
    "asc_VH_tile_index.gpkg",
    "desc_VH_tile_index.gpkg",
]
CELL_SIZE = 10
EPSG = 2180

gdf_aoi = gpd.read_file(AOI)
bounds = gdf_aoi.total_bounds
extent_aoi = bounds[0], bounds[2], bounds[1], bounds[3]
extent_aoi_snapped = Vector.align_extent_to_snap(extent_aoi, SNAP_RASTER)
nodata = -84544
if not os.path.exists(os.path.join(TILEINDEX_ROOT, "ascending")):
    os.makedirs(os.path.join(TILEINDEX_ROOT, "ascending"))
if not os.path.exists(os.path.join(TILEINDEX_ROOT, "descending")):
    os.makedirs(os.path.join(TILEINDEX_ROOT, "descending"))
for file in files:
    tileindex_path = f"{TILEINDEX_ROOT}/{file}"
    gdf_tileindex = gpd.read_file(tileindex_path)
    if nodata == -84544:
        nodata = Raster.get_raster_nodata(gdf_tileindex["path"].iloc[0])
    groups = gdf_tileindex["group"].unique().tolist()
    for group in groups:
        gdf_group = gdf_tileindex[gdf_tileindex["group"] == group]
        asc_desc = "asc" if "asc" in file else "desc"
        dir_sub = "ascending" if "asc" in file else "descending"

        specifics = "VH" if "VH" in file else "VV"
        outfilename = f"{group}_{asc_desc}_{specifics}_sigma0-elp.tif"
        outpath = f"{TILEINDEX_ROOT}/{dir_sub}/{outfilename}"
        lock_path = f"{outpath}.tmp"

        if os.path.exists(outpath):
            print(f"Skipping (already exists): {outpath}")
            continue

        if os.path.exists(lock_path):
            print(f"Skipping (being processed by another process): {outpath}")
            continue

        try:
            # Atomic lock file creation to avoid races between parallel processes.
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_fd)
        except FileExistsError:
            print(f"Skipping (lock appeared concurrently): {outpath}")
            continue

        rasters_group = gdf_group["path"].tolist()
        warped_rasters = []
        try:
            for idx, raster in enumerate(rasters_group):
                print(f"Processing raster: {raster}")
                # Open raster to get properties
                ds_in = gdal.Open(raster)
                if ds_in is None:
                    raise FileNotFoundError(f"Could not open raster: {raster}")

                # Get raster information
                raster_nodata = ds_in.GetRasterBand(1).GetNoDataValue()
                data_type = ds_in.GetRasterBand(1).DataType
                srs_in = osr.SpatialReference()
                srs_in.ImportFromWkt(ds_in.GetProjection())

                srs_out = osr.SpatialReference()
                srs_out.ImportFromEPSG(EPSG)

                # Create vsimem output path
                vsimem_path = f"/vsimem/warped_{group}_{asc_desc}_{specifics}_{idx}.tif"

                # Prepare bounds: convert (xmin, xmax, ymin, ymax) to (minx, miny, maxx, maxy)
                bounds = (
                    extent_aoi_snapped[0],
                    extent_aoi_snapped[2],
                    extent_aoi_snapped[1],
                    extent_aoi_snapped[3],
                )

                # Clip by AOI and snap to extent bounds
                warp_options = gdal.WarpOptions(
                    format="GTiff",
                    cutlineDSName=AOI,
                    xRes=CELL_SIZE,
                    yRes=CELL_SIZE,
                    dstSRS=srs_out,
                    srcSRS=srs_in,
                    outputBounds=bounds,
                    srcNodata=raster_nodata,
                    dstNodata=nodata,
                    outputType=data_type,
                )
                gdal.Warp(vsimem_path, ds_in, options=warp_options)
                ds_in = None

                warped_rasters.append(vsimem_path)

            MosaicRasters(
                warped_rasters,
                output=outpath,
                cell_size=CELL_SIZE,
                epsg=EPSG,
                data_type=gdal.GDT_Float32,
                no_data=nodata,
                extent=extent_aoi_snapped,
            ).mosaic_in_order()
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
