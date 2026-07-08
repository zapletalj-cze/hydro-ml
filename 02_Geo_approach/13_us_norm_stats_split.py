"""
US normalization statistics with a calibration/report split
===========================================================

Prepares the US domain for FAIR evaluation of the trained model:

1. Splits the US patch metadata by reach id (comid) into a CALIBRATION subset
   and a REPORT subset. Statistics and the decision threshold may only be
   derived on the calibration subset; final metrics are reported on the report
   subset. This guard prevents tuning and testing on the same data.
2. Computes per-channel normalization statistics on the CALIBRATION patches
   only, with the exact same accumulation as the training script (float64
   sums over full arrays, nan_to_num), and writes them in the norm_stats.json
   format the evaluation script reads. dsm and water get sentinel entries,
   mirroring training (dsm is per-patch median-normalized, water stays 0/1).

Outputs (OUTPUT_DIR):
    metadata_us_calib.csv    calibration subset (stats + threshold tuning)
    metadata_us_report.csv   report subset (final metrics)
    norm_stats_US.json       statistics computed on the calibration subset

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

US_PATCHES_DIR  = Path(r"D:\...\patches_US\patches")
US_METADATA_CSV = Path(r"D:\...\patches_US\patches_metadata.csv")
OUTPUT_DIR      = Path(r"D:\...\patches_US\us_eval_prep")

INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15",
                  "canopy_height", "canopy_height_sd", "water"]
WATER_CHANNEL = "water"

SPLIT_BY   = "comid"
CALIB_FRAC = 0.30      # share of reaches used for stats + threshold
SEED       = 42


# ============================================================
# SPLIT
# ============================================================

def split_metadata(df):
    """Group-split by reach id so segments of one levee stay on one side."""
    df = df.copy()
    df[SPLIT_BY] = pd.to_numeric(df[SPLIT_BY], errors="coerce")
    n_missing = int(df[SPLIT_BY].isna().sum())
    if n_missing:
        print(f"  dropping {n_missing} patches with no {SPLIT_BY}")
        df = df[df[SPLIT_BY].notna()].copy()
    df[SPLIT_BY] = df[SPLIT_BY].astype(int)

    sources = df[SPLIT_BY].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(sources)
    n_calib = max(1, int(len(sources) * CALIB_FRAC))
    calib_sources = set(sources[:n_calib])

    df["us_split"] = df[SPLIT_BY].apply(
        lambda s: "calib" if s in calib_sources else "report")
    df_calib = df[df["us_split"] == "calib"].reset_index(drop=True)
    df_report = df[df["us_split"] == "report"].reset_index(drop=True)

    print(f"  reaches: {len(sources)} total, {n_calib} calib")
    for name, d in (("calib", df_calib), ("report", df_report)):
        n_pos = int((d["patch_type"] == "positive").sum())
        print(f"  {name}: {len(d)} patches ({n_pos} positive)")
    return df_calib, df_report


# ============================================================
# NORM STATS (identical accumulation to the training script)
# ============================================================

def compute_norm_stats(df_calib):
    channels_to_normalize = [c for c in INPUT_CHANNELS
                             if c not in ("dsm", WATER_CHANNEL)]
    sums = {c: 0.0 for c in channels_to_normalize}
    sq_sums = {c: 0.0 for c in channels_to_normalize}
    counts = {c: 0 for c in channels_to_normalize}

    for _, row in tqdm(df_calib.iterrows(), total=len(df_calib),
                       desc="Computing US norm stats (calib)"):
        npz_path = US_PATCHES_DIR / f"{row['patch_id']}.npz"
        if not npz_path.exists():
            continue
        channels = dict(np.load(npz_path))
        for c in channels_to_normalize:
            arr = np.nan_to_num(channels[c].astype(np.float64))
            sums[c] += arr.sum()
            sq_sums[c] += (arr ** 2).sum()
            counts[c] += arr.size

    norm_stats = {}
    for c in channels_to_normalize:
        if counts[c] == 0:
            raise RuntimeError(f"No pixels accumulated for channel {c}")
        mean = sums[c] / counts[c]
        var = sq_sums[c] / counts[c] - mean ** 2
        std = float(np.sqrt(max(var, 1e-12)))
        norm_stats[c] = {"mean": float(mean), "std": std}

    norm_stats["dsm"] = {"mean": 0.0, "std": 1.0}            # per-patch median
    norm_stats[WATER_CHANNEL] = {"mean": 0.0, "std": 1.0}    # binary 0/1
    return norm_stats


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(US_METADATA_CSV)
    print(f"US metadata: {len(df)} patches")

    df_calib, df_report = split_metadata(df)
    calib_path = OUTPUT_DIR / "metadata_us_calib.csv"
    report_path = OUTPUT_DIR / "metadata_us_report.csv"
    df_calib.to_csv(calib_path, index=False)
    df_report.to_csv(report_path, index=False)
    print(f"  saved {calib_path.name}, {report_path.name}")

    norm_stats = compute_norm_stats(df_calib)
    out_json = OUTPUT_DIR / "norm_stats_US.json"
    with open(out_json, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\n  saved {out_json.name}")
    for c, s in norm_stats.items():
        print(f"  {c:20s} mean={s['mean']:10.3f}  std={s['std']:10.3f}")

    print("\nNext steps:")
    print("  1) evaluate the CALIB subset with norm_stats_US.json and")
    print("     SAVE_PREDICTIONS=True, then run 14_threshold_analysis.py")
    print("  2) evaluate the REPORT subset with both stats variants and both")
    print("     thresholds; report those numbers in the thesis")


if __name__ == "__main__":
    main()
