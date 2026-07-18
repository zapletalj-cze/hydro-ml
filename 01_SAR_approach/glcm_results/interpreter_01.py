"""
SAR proof-of-concept diagnostics harvest (no recomputation)
===========================================================

Collects every number needed for thesis section 4.1 from artifacts already on
disk and writes them into ONE json + ONE readable txt. Nothing heavy is
computed: no GLCM, no training. The only work done is (a) reading rasters to
count pixels and (b) reconstructing the deterministic test sample (same
RANDOM_STATE) and scoring it with the SAVED model, because the classification
report was never written to disk by the training script.

Outputs (in OUT_DIR):
    diagnostics_sar.json    machine-readable, paste/upload into the chat
    diagnostics_sar.txt     the same, human-readable
    pr_points.csv           downsampled PR curve for the thesis figure

Set EVALUATE_TEST = False to skip the test-set scoring (then the report and
AP/ROC come only from model_metadata.json, without per-class detail).

Dependencies: GDAL, geopandas+pyogrio, numpy, pandas, xgboost, scikit-learn.
No rasterio, no fiona.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG (mirrors the v04 training script)
# ============================================================

ASC_DIR      = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\sentinel1_data\processed\ascending')
DESC_DIR     = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\sentinel1_data\processed\descending')
BDOT10K_PATH = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\01_data\levees_selection\WalyNaspy.gpkg')

SCRIPT_DIR   = Path(__file__).resolve().parent
LEGACY_OUT_DIR = Path(r'D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\sentinel\02_modeldevelopment\v04')
OUT_DIR      = SCRIPT_DIR


def resolve_artifact(name):
    for base_dir in (SCRIPT_DIR, LEGACY_OUT_DIR):
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return SCRIPT_DIR / name


FEATURE_TIF  = resolve_artifact('feature_stack.tif')
LABEL_TIF    = resolve_artifact('label_mask.tif')
MODEL_JSON   = resolve_artifact('xgb_levee_model.json')
META_JSON    = resolve_artifact('model_metadata.json')
PROBA_TIF    = resolve_artifact('levee_probability.tif')
PRED_TIF     = resolve_artifact('levee_prediction.tif')

BLOCK_SIZE    = 256
TEST_FRACTION = 0.3
N_POSITIVE    = 50_000
N_NEGATIVE    = 50_000
RANDOM_STATE  = 42
PIXEL_SIZE    = 10.0

EVALUATE_TEST = True     # deterministic test-sample scoring with the saved model

BAND_NAMES = [
    'asc_vv_mean', 'asc_vh_mean', 'asc_vv_std', 'asc_vh_std', 'asc_ratio',
    'asc_glcm_contrast', 'asc_glcm_homogeneity', 'asc_glcm_energy', 'asc_glcm_correlation',
    'desc_vv_mean', 'desc_vh_mean', 'desc_vv_std', 'desc_vh_std', 'desc_ratio',
    'desc_glcm_contrast', 'desc_glcm_homogeneity', 'desc_glcm_energy', 'desc_glcm_correlation',
]

report = {}


def open_raster(path):
    from osgeo import gdal
    if not path.exists():
        return None
    ds = gdal.Open(str(path))
    return ds

# ============================================================
# 1. ENVIRONMENT
# ============================================================

def harvest_environment():
    env = {"python": sys.version.split()[0]}
    for mod in ("numpy", "xgboost", "sklearn", "geopandas"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "n/a"
    return env

# ============================================================
# 2. SCENES (file listing only)
# ============================================================

def collect_scenes(directory, polarization):
    for pattern in [f'*_{polarization}_sigma0-elp.tif',
                    f'*_{polarization}_gamma0-elp.tif',
                    f'*_{polarization}_grd_elp.tif',
                    f'*{polarization}*.tif']:
        scenes = sorted(directory.glob(pattern))
        if scenes:
            return scenes
    return []


def harvest_scenes():
    out = {}
    date_re = re.compile(r'(\d{8})T\d{6}')
    for name, d in (("ascending", ASC_DIR), ("descending", DESC_DIR)):
        entry = {}
        for pol in ("VV", "VH"):
            scenes = collect_scenes(d, pol)
            dates = sorted({m.group(1) for s in scenes
                            for m in [date_re.search(s.name)] if m})
            entry[pol] = {
                "n_scenes": len(scenes),
                "date_min": dates[0] if dates else None,
                "date_max": dates[-1] if dates else None,
                "n_unique_dates": len(dates),
            }
        out[name] = entry
    return out

# ============================================================
# 3. REFERENCE (vector counts + rasterized label counts)
# ============================================================

def harvest_reference():
    import geopandas as gpd
    from osgeo import gdal
    gdal.UseExceptions()

    out = {}
    gdf = gpd.read_file(BDOT10K_PATH, engine='pyogrio')
    out["n_features"] = int(len(gdf))
    try:
        out["total_length_km"] = float(gdf.geometry.length.sum() / 1000.0)
    except Exception:
        out["total_length_km"] = None
    for col in ("rodzaj", "source"):
        if col in gdf.columns:
            out[f"by_{col}"] = {str(k): int(v)
                                for k, v in gdf[col].value_counts().items()}

    ds = open_raster(LABEL_TIF)
    if ds is not None:
        arr = ds.GetRasterBand(1).ReadAsArray()
        nod = ds.GetRasterBand(1).GetNoDataValue()
        valid = np.ones(arr.shape, bool) if nod is None else (arr != nod)
        pos = int(((arr == 1) & valid).sum())
        neg = int(((arr == 0) & valid).sum())
        out["label_raster"] = {
            "shape": list(arr.shape),
            "positive_pixels": pos,
            "negative_pixels": neg,
            "positive_share": pos / max(pos + neg, 1),
            "positive_area_km2": pos * PIXEL_SIZE**2 / 1e6,
        }
        ds = None
    else:
        out["missing_artifact"] = str(LABEL_TIF)
    return out

# ============================================================
# 4. SPLIT, SAMPLING AND (OPTIONAL) TEST-SET SCORING
# ============================================================

def assign_spatial_blocks(nrows, ncols, block_size, test_fraction, random_state):
    rng = np.random.default_rng(random_state)
    n_br = int(np.ceil(nrows / block_size))
    n_bc = int(np.ceil(ncols / block_size))
    n_blocks = n_br * n_bc
    n_test = max(1, int(round(n_blocks * test_fraction)))
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)
    test_set = set(block_ids[:n_test].tolist())
    split = np.zeros((nrows, ncols), dtype=np.uint8)
    for bi in range(n_br):
        for bj in range(n_bc):
            if bi * n_bc + bj in test_set:
                r0, r1 = bi * block_size, min((bi + 1) * block_size, nrows)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, ncols)
                split[r0:r1, c0:c1] = 1
    return split, {"blocks_rows": n_br, "blocks_cols": n_bc,
                   "n_blocks": n_blocks, "n_test_blocks": n_test,
                   "n_train_blocks": n_blocks - n_test}


def harvest_split_and_eval():
    from osgeo import gdal
    gdal.UseExceptions()

    out = {}
    missing = [str(path) for path in (FEATURE_TIF, LABEL_TIF) if not path.exists()]
    if missing:
        out["missing_artifacts"] = missing
        return out

    ds = gdal.Open(str(FEATURE_TIF))
    if ds is None:
        out["missing_artifacts"] = [str(FEATURE_TIF)]
        return out

    nrows, ncols, nbands = ds.RasterYSize, ds.RasterXSize, ds.RasterCount
    out["stack"] = {"nrows": nrows, "ncols": ncols, "nbands": nbands}

    split, blocks = assign_spatial_blocks(nrows, ncols, BLOCK_SIZE,
                                          TEST_FRACTION, RANDOM_STATE)
    out["blocks"] = blocks

    lab_ds = gdal.Open(str(LABEL_TIF))
    if lab_ds is None:
        out["missing_artifacts"] = [str(LABEL_TIF)]
        return out

    lab = lab_ds.GetRasterBand(1).ReadAsArray()
    y_flat = (lab == 1).astype(np.int8).ravel()
    s_flat = split.ravel()

    # full stack in RAM, exactly as the training script held it
    X_flat = np.stack([ds.GetRasterBand(b + 1).ReadAsArray()
                       for b in range(nbands)], axis=-1).reshape(-1, nbands)
    nod = ds.GetRasterBand(1).GetNoDataValue()
    if nod is not None:
        X_flat[X_flat == nod] = np.nan
    valid = ~np.any(np.isnan(X_flat), axis=1)
    out["valid_pixels"] = int(valid.sum())
    ds = None

    rng = np.random.default_rng(RANDOM_STATE)
    samples = {}

    def sample_set(set_id, frac):
        mask = (s_flat == set_id) & valid
        pos_idx = np.where(mask & (y_flat == 1))[0]
        neg_idx = np.where(mask & (y_flat == 0))[0]
        n_pos = min(int(N_POSITIVE * frac), len(pos_idx))
        n_neg = min(int(N_NEGATIVE * frac), len(neg_idx))
        idx = np.concatenate([rng.choice(pos_idx, size=n_pos, replace=False),
                              rng.choice(neg_idx, size=n_neg, replace=False)])
        rng.shuffle(idx)
        return idx, {"available_pos": int(len(pos_idx)),
                     "available_neg": int(len(neg_idx)),
                     "sampled_pos": n_pos, "sampled_neg": n_neg}

    idx_train, samples["train"] = sample_set(0, 0.7)
    idx_test, samples["test"] = sample_set(1, 0.3)
    out["sampling"] = samples

    if EVALUATE_TEST:
        import xgboost as xgb
        from sklearn.metrics import (classification_report,
                                     average_precision_score, roc_auc_score,
                                     precision_recall_curve)
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_JSON))

        X_test = X_flat[idx_test].astype(np.float32)
        y_test = y_flat[idx_test]
        proba = model.predict_proba(X_test)[:, 1]

        p, r, thr = precision_recall_curve(y_test, proba)
        f1 = 2 * p * r / (p + r + 1e-10)
        bi = int(np.argmax(f1[:-1]))
        best_thr = float(thr[bi])

        out["test_eval"] = {
            "n_test": int(len(y_test)),
            "ap": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "best_threshold": best_thr,
            "best_f1": float(f1[bi]),
            "report_at_best_thr": classification_report(
                y_test, (proba >= best_thr).astype(int),
                target_names=["non-levee", "levee"],
                digits=3, output_dict=True),
        }
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        out["feature_importance"] = [
            {"rank": k + 1, "feature": BAND_NAMES[i], "importance": float(imp[i])}
            for k, i in enumerate(order)]

        step = max(1, len(p) // 200)
        pd.DataFrame({"precision": p[::step], "recall": r[::step]}
                     ).to_csv(OUT_DIR / "pr_points.csv", index=False)
    return out

# ============================================================
# 5. INFERENCE RASTER STATS
# ============================================================

def harvest_inference():
    from osgeo import gdal
    gdal.UseExceptions()
    out = {}
    if PRED_TIF.exists():
        ds = gdal.Open(str(PRED_TIF))
        arr = ds.GetRasterBand(1).ReadAsArray()
        n_levee = int((arr == 1).sum())
        n_valid = int((arr != 255).sum())
        out["prediction"] = {
            "levee_pixels": n_levee,
            "levee_area_km2": n_levee * PIXEL_SIZE**2 / 1e6,
            "valid_pixels": n_valid,
            "levee_share_of_valid": n_levee / max(n_valid, 1),
        }
        ds = None
    if PROBA_TIF.exists():
        out["probability_map_exists"] = True
    if not PRED_TIF.exists() and not PROBA_TIF.exists():
        out["missing_artifacts"] = [str(PRED_TIF), str(PROBA_TIF)]
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report["environment"] = harvest_environment()
    report["scenes"] = harvest_scenes()
    report["reference"] = harvest_reference()
    if META_JSON.exists():
        report["model_metadata"] = json.loads(META_JSON.read_text())
    report["split_sampling_eval"] = harvest_split_and_eval()
    report["inference"] = harvest_inference()

    json_path = OUT_DIR / "diagnostics_sar.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = []
    def w(s=""):
        lines.append(s)
    w("SAR PROOF-OF-CONCEPT DIAGNOSTICS (section 4.1 inputs)")
    w("=" * 60)
    w(json.dumps(report, indent=2, ensure_ascii=False))
    (OUT_DIR / "diagnostics_sar.txt").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print(f"written: {json_path}")
    print(f"written: {OUT_DIR / 'diagnostics_sar.txt'}")


if __name__ == "__main__":
    main()