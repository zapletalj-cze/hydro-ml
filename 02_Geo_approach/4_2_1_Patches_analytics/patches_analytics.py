"""
Patch inventory for thesis section 4.2.1 (training / validation / test data)
============================================================================

Reads the metadata CSVs of the patch datasets (PL + USA folders), counts
patches by every categorical dimension present (patch_type, basin, category,
split, ...), cross-checks the counts against the actual *.npz files on disk,
and looks inside the training run folder for any stored train/val split
artifact (the internal 85:15 split by river reach). Computes nothing beyond
value counts.

Outputs (DIAG_OUT):
    patch_inventory.json      machine-readable
    patch_inventory.txt       the same, human-readable (paste into chat)
    patch_inventory_table.csv tidy long table: dataset, column, value,
                              patch_type, n

If the metadata carries no split column and no split artifact is found, the
inventory will say so explicitly - then the 85:15 validation counts must be
taken from the training log instead.

Dependencies: pandas, numpy only.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL")

DATASETS = {
    "PL":  {"dir": Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL") / "patches" / "patches_PL_train",
            "metadata": r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_train\patches_metadata.csv"},   # None -> auto-discover *metadata*.csv in dir
    "USA": {"dir": Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_USA") / "patches",
            "metadata": r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_USA\patches_metadata.csv"},
}

# Training run folder - searched for stored split artifacts (val ids etc.)
TRAIN_RUN_DIR = BASE / "training_v06_segformer_PL_US"

DIAG_OUT = Path(__file__).parent / "diagnostics_ch4"

# columns that are candidates for grouping (matched case-insensitively)
GROUP_CANDIDATES = ["patch_type", "basin", "river", "region", "split",
                    "subset", "role", "category", "source", "povodi",
                    "dataset", "area"]
MAX_UNIQUE = 40   # a column with more unique values is not categorical

report = {"notes": []}
tidy_rows = []


def note(msg):
    report["notes"].append(msg)
    print(f"  NOTE: {msg}")


# ============================================================
# PER-DATASET INVENTORY
# ============================================================

def find_metadata(dirpath, explicit):
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    cands = sorted(Path(dirpath).rglob("*metadata*.csv"))
    return cands[0] if cands else None


def count_npz(dirpath):
    d = Path(dirpath)
    n = len(list(d.glob("*.npz")))
    if n == 0:
        for sub in ("patches",):
            n = len(list((d / sub).glob("*.npz")))
            if n:
                break
    return n


def categorical_columns(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(g in cl for g in GROUP_CANDIDATES) or (
                df[c].dtype == object and df[c].nunique() <= MAX_UNIQUE):
            if df[c].nunique() <= MAX_UNIQUE:
                cols.append(c)
    return cols


def inventory_dataset(name, dirpath, metadata):
    d = Path(dirpath)
    out = {"dir": str(d)}
    if not d.exists():
        note(f"dataset '{name}': folder not found ({d})")
        return out

    meta = find_metadata(d, metadata)
    if meta is None:
        note(f"dataset '{name}': no *metadata*.csv found under {d}")
        return out
    out["metadata_csv"] = str(meta)

    df = pd.read_csv(meta)
    out["n_rows"] = int(len(df))
    out["columns"] = list(df.columns)

    n_files = count_npz(d)
    out["n_npz_files"] = n_files
    if n_files and n_files != len(df):
        note(f"dataset '{name}': {len(df)} metadata rows vs {n_files} npz "
             f"files - check which is authoritative")

    cats = categorical_columns(df)
    out["counts"] = {}
    for c in cats:
        vc = df[c].astype(str).value_counts()
        out["counts"][c] = {str(k): int(v) for k, v in vc.items()}
        for k, v in vc.items():
            tidy_rows.append({"dataset": name, "column": c, "value": str(k),
                              "patch_type": "", "n": int(v)})

    # cross-tab of every group column vs patch_type, if present
    pt = next((c for c in df.columns if c.lower() == "patch_type"), None)
    if pt:
        out["crosstabs"] = {}
        for c in cats:
            if c == pt:
                continue
            ct = pd.crosstab(df[c].astype(str), df[pt].astype(str))
            out["crosstabs"][c] = {str(i): {str(j): int(ct.loc[i, j])
                                            for j in ct.columns}
                                   for i in ct.index}
            for i in ct.index:
                for j in ct.columns:
                    tidy_rows.append({"dataset": name, "column": c,
                                      "value": str(i), "patch_type": str(j),
                                      "n": int(ct.loc[i, j])})
    else:
        note(f"dataset '{name}': no patch_type column")
    return out


# ============================================================
# SPLIT ARTIFACTS IN THE TRAINING RUN
# ============================================================

def harvest_split_artifacts(train_dir, datasets_meta):
    d = Path(train_dir)
    out = {"dir": str(d), "artifacts": []}
    if not d.exists():
        note(f"training run folder not found ({d})")
        return out

    patterns = ("*val*", "*split*", "*train_ids*", "*fold*")
    seen = set()
    for pat in patterns:
        for f in sorted(d.rglob(pat)):
            if not f.is_file() or f.suffix.lower() not in (".json", ".csv", ".txt"):
                continue
            if f in seen or f.stat().st_size > 50_000_000:
                continue
            seen.add(f)
            entry = {"file": str(f.relative_to(d)),
                     "size_kb": round(f.stat().st_size / 1024, 1)}
            ids = None
            try:
                if f.suffix.lower() == ".json":
                    obj = json.loads(f.read_text())
                    if isinstance(obj, list):
                        ids = [str(x) for x in obj]
                    elif isinstance(obj, dict):
                        entry["keys"] = list(obj.keys())[:10]
                        for key in ("val", "val_ids", "validation"):
                            if key in obj and isinstance(obj[key], list):
                                ids = [str(x) for x in obj[key]]
                                entry["ids_from_key"] = key
                                break
                else:
                    dfa = pd.read_csv(f)
                    entry["columns"] = list(dfa.columns)
                    entry["n_rows"] = int(len(dfa))
            except Exception as e:
                entry["parse_error"] = str(e)

            if ids is not None:
                entry["n_ids"] = len(ids)
                # match ids against each dataset's metadata id-like columns
                for name, df in datasets_meta.items():
                    if df is None:
                        continue
                    for col in df.columns:
                        if col.lower() in ("patch_id", "comid", "source_idx",
                                           "reach_id", "id"):
                            hit = df[col].astype(str).isin(ids).sum()
                            if hit:
                                entry.setdefault("matches", {})[
                                    f"{name}.{col}"] = int(hit)
            out["artifacts"].append(entry)
    if not out["artifacts"]:
        note("no split artifact found in the training run - validation "
             "counts must come from the training log")
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    DIAG_OUT.mkdir(parents=True, exist_ok=True)

    datasets_meta = {}
    for name, spec in DATASETS.items():
        inv = inventory_dataset(name, spec["dir"], spec["metadata"])
        report[f"dataset_{name}"] = inv
        meta_path = inv.get("metadata_csv")
        datasets_meta[name] = pd.read_csv(meta_path) if meta_path else None

    report["train_split_artifacts"] = harvest_split_artifacts(
        TRAIN_RUN_DIR, datasets_meta)

    json_path = DIAG_OUT / "patch_inventory.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    default=str), encoding="utf-8")
    (DIAG_OUT / "patch_inventory.txt").write_text(
        "PATCH INVENTORY (section 4.2.1 inputs)\n" + "=" * 60 + "\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    pd.DataFrame(tidy_rows).to_csv(DIAG_OUT / "patch_inventory_table.csv",
                                   index=False)
    print(f"\nwritten: {json_path}")
    print(f"written: {DIAG_OUT / 'patch_inventory.txt'}")
    print(f"written: {DIAG_OUT / 'patch_inventory_table.csv'}")


if __name__ == "__main__":
    main()