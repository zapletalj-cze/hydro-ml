import os
import glob
import yaml

from osgeo import gdal

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
IN_DIR  = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\nisar_data"
OUT_DIR = r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\nisar_data\GTIFF"

os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# YAML helpers
# --------------------------------------------------------------------
def load_runconfig_yaml(yaml_path):
    """Load runconfig YAML, return dict or None."""
    if not os.path.exists(yaml_path):
        print(f"[WARN] Runconfig YAML not found: {yaml_path}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        print(f"[WARN] Failed to parse YAML {yaml_path}: {e}")
        return None


def get_geocode_params_from_yaml(cfg):
    """
    Extract geocode parameters:
        output_epsg
        top_left (x_abs, y_abs)
        bottom_right (x_abs, y_abs)
    Return (epsg, min_x, min_y, max_x, max_y) or None.
    """
    try:
        geocode = cfg["runconfig"]["groups"]["processing"]["geocode"]
    except KeyError:
        print("[WARN] No processing.geocode group in YAML.")
        return None

    epsg = geocode.get("output_epsg", None)
    if epsg is None:
        print("[WARN] output_epsg missing in YAML.")
        return None

    tl = geocode.get("top_left", {})
    br = geocode.get("bottom_right", {})

    x_tl = tl.get("x_abs", None)
    y_tl = tl.get("y_abs", None)
    x_br = br.get("x_abs", None)
    y_br = br.get("y_abs", None)

    if None in (x_tl, y_tl, x_br, y_br):
        print("[WARN] Some of top_left/bottom_right x_abs/y_abs are missing.")
        return None

    # top_left = (x_min, y_max), bottom_right = (x_max, y_min)
    min_x = float(x_tl)
    max_y = float(y_tl)
    max_x = float(x_br)
    min_y = float(y_br)

    return int(epsg), min_x, min_y, max_x, max_y


# --------------------------------------------------------------------
# Subdataset search helpers
# --------------------------------------------------------------------
def list_subdataset_names(h5_path):
    ds = gdal.Open(h5_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"[ERROR] Cannot open HDF5 file: {h5_path}")
        return []
    return [s[0] for s in ds.GetSubDatasets()]


def find_subdataset_by_keyword(names, keyword):
    """
    Najde první subdataset, jehož název obsahuje daný řetězec (case-sensitive).
    Vrací název nebo None.
    """
    for name in names:
        if keyword in name:
            return name
    return None


# --------------------------------------------------------------------
# Export helper
# --------------------------------------------------------------------
def export_subdataset_to_tif(sds_name, out_tif, epsg, min_x, min_y, max_x, max_y):
    print(f"[INFO]  Translating {sds_name}")
    print(f"[INFO]  -> {out_tif}")
    gdal.Translate(
        out_tif,
        sds_name,
        format="GTiff",
        outputSRS=f"EPSG:{epsg}",
        outputBounds=[min_x, min_y, max_x, max_y],
        creationOptions=["TILED=YES", "COMPRESS=LZW"],
    )

    # quick extent check
    ds_out = gdal.Open(out_tif, gdal.GA_ReadOnly)
    if ds_out:
        gt = ds_out.GetGeoTransform()
        x_size = ds_out.RasterXSize
        y_size = ds_out.RasterYSize
        ulx = gt[0]
        uly = gt[3]
        lrx = gt[0] + x_size * gt[1] + y_size * gt[2]
        lry = gt[3] + x_size * gt[4] + y_size * gt[5]
        print(f"[INFO]  Written extent: {ulx},{lry} : {lrx},{uly}")
        ds_out = None


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    h5_files = glob.glob(os.path.join(IN_DIR, "*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {IN_DIR}")
        return

    for h5_path in h5_files:
        base = os.path.splitext(os.path.basename(h5_path))[0]
        rc_yaml_path = os.path.join(IN_DIR, base + ".rc.yaml")

        print(f"\n[INFO] Processing {base}")
        print(f"[INFO] HDF5: {h5_path}")
        print(f"[INFO] Runconfig YAML: {rc_yaml_path}")

        cfg = load_runconfig_yaml(rc_yaml_path)
        if cfg is None:
            print("[WARN] Skipping file because YAML could not be loaded.")
            continue

        geocode_params = get_geocode_params_from_yaml(cfg)
        if geocode_params is None:
            print("[WARN] Skipping file because geocode parameters are missing.")
            continue

        epsg, min_x, min_y, max_x, max_y = geocode_params
        print(f"[INFO] From YAML: EPSG={epsg}, "
              f"bbox={min_x},{min_y} : {max_x},{max_y}")

        sds_names = list_subdataset_names(h5_path)
        if not sds_names:
            print("[WARN] No subdatasets found; skipping file.")
            continue

        # HVHV
        hvhv_sds = find_subdataset_by_keyword(sds_names, "HVHV")
        if hvhv_sds:
            out_tif_hvhv = os.path.join(OUT_DIR, base + "_HVHV.tif")
            export_subdataset_to_tif(hvhv_sds, out_tif_hvhv, epsg, min_x, min_y, max_x, max_y)
        else:
            print("[WARN] No HVHV subdataset found.")

        # HHHH
        hhhh_sds = find_subdataset_by_keyword(sds_names, "HHHH")
        if hhhh_sds:
            out_tif_hhhh = os.path.join(OUT_DIR, base + "_HHHH.tif")
            export_subdataset_to_tif(hhhh_sds, out_tif_hhhh, epsg, min_x, min_y, max_x, max_y)
        else:
            print("[WARN] No HHHH subdataset found.")

        # rtcGammaToSigmaFactor
        rtc_sds = find_subdataset_by_keyword(sds_names, "rtcGammaToSigmaFactor")
        if rtc_sds:
            out_tif_rtc = os.path.join(OUT_DIR, base + "_rtcGammaToSigmaFactor.tif")
            export_subdataset_to_tif(rtc_sds, out_tif_rtc, epsg, min_x, min_y, max_x, max_y)
        else:
            print("[WARN] No rtcGammaToSigmaFactor subdataset found.")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()