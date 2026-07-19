"""Channel-ablation harvest (section 4.2.3): reads the evaluation folders of
the ablation runs and writes one JSON + TXT. Only sums/means over existing
CSVs, nothing recomputed."""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---- CONFIG: fill the evaluation output folders -----------------------------
# Adjust names/paths to the ablation variants that exist.
EVAL_RUNS = {
    "v1_dsm":      Path(r"D:\PATH\TO\ablation_v1_dsm\eval_basin_B"),
    "v2_dsm_tpi":  Path(r"D:\PATH\TO\ablation_v2_dsm_tpi\eval_basin_B"),
    "v3_full":     Path(r"D:\PATH\TO\ablation_v3_full\eval_basin_B"),
}
DIAG_OUT = Path(__file__).parent / "diagnostics_ch4"

report = {"notes": []}


def harvest(name, d):
    d = Path(d)
    out = {"dir": str(d)}
    if not d.exists():
        report["notes"].append(f"'{name}': folder not found ({d})")
        return out

    cfg = d / "eval_config.json"
    if cfg.exists():
        out["eval_config"] = json.loads(cfg.read_text())

    per_patch = sorted(d.glob("*_results_per_patch.csv"))
    if not per_patch:
        report["notes"].append(f"'{name}': no *_results_per_patch.csv in {d}")
        return out
    df = pd.read_csv(per_patch[0])
    out["per_patch_file"] = per_patch[0].name
    out["n_patches"] = int(len(df))

    low = {c.lower(): c for c in df.columns}
    if all(k in low for k in ("tp", "fp", "fn")):
        tp = float(df[low["tp"]].sum())
        fp = float(df[low["fp"]].sum())
        fn = float(df[low["fn"]].sum())
        eps = 1e-9
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        out["micro"] = {"precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r + eps),
                        "iou": tp / (tp + fp + fn + eps),
                        "dice": 2 * tp / (2 * tp + fp + fn + eps)}

    metric_cols = [c for c in ("precision", "recall", "f1", "dice", "iou",
                               "cldice") if c in low]
    def block(sub):
        return {m: {"mean": float(pd.to_numeric(sub[low[m]], errors="coerce").mean()),
                    "std": float(pd.to_numeric(sub[low[m]], errors="coerce").std())}
                for m in metric_cols}
    out["per_patch_all"] = block(df)
    if "patch_type" in low:
        pos = df[df[low["patch_type"]] == "positive"]
        out["n_positive"] = int(len(pos))
        out["per_patch_positive"] = block(pos)

    by_cat = sorted(d.glob("*_results_by_category.csv"))
    if by_cat:
        out["by_category"] = pd.read_csv(by_cat[0]).to_dict(orient="records")
    return out


def main():
    DIAG_OUT.mkdir(parents=True, exist_ok=True)
    for name, d in EVAL_RUNS.items():
        report[name] = harvest(name, d)

    (DIAG_OUT / "ablation_channels.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (DIAG_OUT / "ablation_channels.txt").write_text(
        "CHANNEL ABLATION (section 4.2.3 inputs)\n" + "=" * 60 + "\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print("written:", DIAG_OUT / "ablation_channels.json")
    print("written:", DIAG_OUT / "ablation_channels.txt")
    if report["notes"]:
        print("notes:", *report["notes"], sep="\n  ")


if __name__ == "__main__":
    main()