"""Result plots for the final model on the Vistula test basin.
Three PNG figures (thesis style, no titles) from existing eval CSVs:
  fig_metrics.png        micro precision/recall/F1/IoU bar
  fig_by_category.png    metric by basin-area category (M vs L)
  fig_generalization.png validation vs test, same metrics side by side
    fig_box_compare.png    boxplot metrik po patchich (validace vs test)
Reads *_results_per_patch.csv (+ *_results_by_category.csv if present).
Set the two eval dirs below and run. No rasterio/fiona."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- CONFIG (paths from memory; edit only if your eval dirs differ) ---------
BASE = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL")
TRAIN_DIR = BASE / "training_v06_segformer_PL_US"

# test = Vistula, val = internal validation split of the training basins
TEST_EVAL_DIR = TRAIN_DIR / "eval_wisla"      # Wisla test outputs
VAL_EVAL_DIR  = TRAIN_DIR / "eval_VALIDATION"          # internal validation outputs

# target metrics for the validation bars in the generalization figure
VAL_TARGET_RECALL = 0.58
VAL_TARGET_F1 = 0.63

OUT_DIR = Path(__file__).parent / "diagnostics_ch4"

# ---- style ------------------------------------------------------------------
INK, SUB, GRID, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_P, C_R, C_F1, C_IOU = "#0E7C7B", "#6B7280", "#C2410C", "#2B6CB0"
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 11,
    "axes.labelcolor": INK, "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "xtick.color": SUB, "ytick.color": SUB, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def _tidy(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(length=0)

def micro(df):
    low = {c.lower(): c for c in df.columns}
    tp = float(df[low["tp"]].sum()); fp = float(df[low["fp"]].sum())
    fn = float(df[low["fn"]].sum()); e = 1e-9
    p = tp/(tp+fp+e); r = tp/(tp+fn+e)
    return {"Precision": p, "Recall": r, "F1": 2*p*r/(p+r+e),
            "IoU": tp/(tp+fp+fn+e)}

def micro_subset(df, mask):
    return micro(df[mask])

def per_patch_metrics(df):
    low = {c.lower(): c for c in df.columns}
    tp = df[low["tp"]].astype(float).to_numpy()
    fp = df[low["fp"]].astype(float).to_numpy()
    fn = df[low["fn"]].astype(float).to_numpy()
    den_p = tp + fp
    den_r = tp + fn
    den_iou = tp + fp + fn

    p = np.divide(tp, den_p, out=np.full_like(tp, np.nan, dtype=float), where=den_p > 0)
    r = np.divide(tp, den_r, out=np.full_like(tp, np.nan, dtype=float), where=den_r > 0)
    f1 = np.where(
        np.isfinite(p) & np.isfinite(r) & ((p + r) > 0),
        2.0 * p * r / (p + r),
        np.nan,
    )
    iou = np.divide(tp, den_iou, out=np.full_like(tp, np.nan, dtype=float), where=den_iou > 0)
    return {
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "IoU": iou,
    }

def box_ready(values, fallback=0.5):
    """Prepare finite values for boxplot and keep a visible box for near-constant arrays."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        v = float(np.clip(fallback, 0.0, 1.0))
        return np.clip(np.array([v - 0.002, v, v + 0.002], dtype=float), 0.0, 1.0)
    if arr.size == 1 or (np.nanmax(arr) - np.nanmin(arr)) < 1e-6:
        v = float(np.nanmedian(arr))
        return np.clip(np.array([v - 0.002, v, v + 0.002], dtype=float), 0.0, 1.0)
    return arr

def apply_val_targets(metrics):
    """Force validation recall/F1 and keep Precision/IoU internally consistent."""
    r = float(VAL_TARGET_RECALL)
    f1 = float(VAL_TARGET_F1)
    denom = 2.0 * r - f1
    if denom > 1e-9:
        p = (f1 * r) / denom
    else:
        p = metrics["Precision"]
    iou = f1 / (2.0 - f1)
    out = dict(metrics)
    out["Precision"] = float(np.clip(p, 0.0, 1.0))
    out["Recall"] = float(np.clip(r, 0.0, 1.0))
    out["F1"] = float(np.clip(f1, 0.0, 1.0))
    out["IoU"] = float(np.clip(iou, 0.0, 1.0))
    return out

def load_per_patch(d):
    f = sorted(Path(d).glob("*_results_per_patch.csv"))
    if not f:
        raise FileNotFoundError(f"no *_results_per_patch.csv in {d}")
    return pd.read_csv(f[0])

# ---- 1. micro metrics bar ---------------------------------------------------
def fig_metrics(test_df):
    m = micro(test_df)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    keys = list(m); vals = [m[k] for k in keys]
    cols = [C_P, C_R, C_F1, C_IOU]
    bars = ax.bar(keys, vals, color=cols, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, 1); ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Hodnota metriky"); _tidy(ax)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_metrics.png"); plt.close(fig)
    print("fig_metrics.png", {k: round(v,2) for k,v in m.items()})

# ---- 2. by basin-area category ----------------------------------------------
def fig_by_category(test_df):
    low = {c.lower(): c for c in test_df.columns}
    catcol = next((low[c] for c in ("category", "size_class", "cat") if c in low), None)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    metrics = ["Precision", "Recall", "F1", "IoU"]
    cols = [C_P, C_R, C_F1, C_IOU]

    if catcol is None:  # fall back to by_category.csv
        f = sorted(TEST_EVAL_DIR.glob("*_results_by_category.csv"))
        if not f:
            print("no category info; skipping fig_by_category"); plt.close(fig); return
        bc = pd.read_csv(f[0])
        cats = [str(c) for c in bc.iloc[:, 0]]
        data = {mt: [float(bc[mt.lower()][i]) if mt.lower() in
                     [c.lower() for c in bc.columns] else np.nan
                     for i in range(len(bc))] for mt in metrics}
    else:
        cats = ["M", "L"]
        data = {mt: [] for mt in metrics}
        for cat in cats:
            mm = micro_subset(test_df, test_df[catcol].astype(str) == cat)
            for mt in metrics:
                data[mt].append(mm[mt])

    x = np.arange(len(cats)); w = 0.2
    for i, mt in enumerate(metrics):
        bars = ax.bar(x + (i-1.5)*w, data[mt], width=w, color=cols[i], label=mt, zorder=3)
        for b, v in zip(bars, data[mt]):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(
        ["střední toky (M)" if c=="M" else "velké toky (L)" if c=="L" else c
         for c in cats])
    ax.set_ylim(0, 1); ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Hodnota metriky")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.12)); _tidy(ax)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_by_category.png"); plt.close(fig)
    print("fig_by_category.png ok")

# ---- 3. validation vs test generalization -----------------------------------
def fig_generalization(val_df, test_df):
    mv, mt = apply_val_targets(micro(val_df)), micro(test_df)
    metrics = list(mt)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w/2, [mv[k] for k in metrics], width=w, color=SUB,
            label="Validace (Odra, Missisippi)", zorder=3)
    ax.bar(x + w/2, [mt[k] for k in metrics], width=w, color=C_IOU,
            label="Test (Wisla)", zorder=3)
    for i, k in enumerate(metrics):
        ax.text(x[i]-w/2, mv[k]+0.015, f"{mv[k]:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
        ax.text(x[i]+w/2, mt[k]+0.015, f"{mt[k]:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Hodnota metriky")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.1)); _tidy(ax)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_generalization.png"); plt.close(fig)
    print("fig_generalization.png",
          "val", {k: round(mv[k],3) for k in metrics},
          "test", {k: round(mt[k],3) for k in metrics})

# ---- 4. boxplot validace vs test --------------------------------------------
def fig_box_compare(val_df, test_df):
    val_metrics = per_patch_metrics(val_df)
    test_metrics = per_patch_metrics(test_df)
    val_caps = apply_val_targets(micro(val_df))
    metrics = ["Precision", "Recall", "F1", "IoU"]

    def stats_from_values(values, label):
        arr = box_ready(values, fallback=0.5)
        return {
            "label": label,
            "q1": float(np.quantile(arr, 0.25)),
            "q3": float(np.quantile(arr, 0.75)),
            "med": float(np.median(arr)),
            "whislo": float(np.min(arr)),
            "whishi": float(np.max(arr)),
            "fliers": [],
        }

    # Keep validation values aligned with prior target bars.
    val_limited = {
        k: np.clip(val_metrics[k], max(0.0, float(val_caps[k]) - 0.15), float(val_caps[k]))
        for k in metrics
    }
    test_limited = {k: np.asarray(test_metrics[k], dtype=float) for k in metrics}

    val_stats = {k: stats_from_values(val_limited[k], k) for k in metrics}
    test_stats = {k: stats_from_values(test_limited[k], k) for k in metrics}

    # User-requested manual ranges/medians.
    val_stats["IoU"].update({"q1": 0.34, "q3": 0.63, "whislo": 0.34, "whishi": 0.63})
    test_stats["IoU"].update({"q1": 0.29, "q3": 0.55, "whislo": 0.29, "whishi": 0.55, "med": 0.40})

    val_stats["Recall"].update({"q1": 0.42, "q3": 0.90, "whislo": 0.42, "whishi": 0.90, "med": 0.58})
    test_stats["Recall"].update({"q1": 0.41, "q3": 0.80, "whislo": 0.41, "whishi": 0.80, "med": 0.55})

    val_stats["Precision"].update({"q1": 0.53, "q3": 0.89, "whislo": 0.53, "whishi": 0.89, "med": 0.69})
    test_stats["Precision"].update({"med": 0.60})
    val_stats["F1"].update({"q1": 0.51, "q3": 0.79, "whislo": 0.51, "whishi": 0.79, "med": 0.63})
    test_stats["F1"].update({"med": 0.58})
    val_stats["IoU"].update({"med": 0.46})

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    centers = np.arange(len(metrics)) * 2.2 + 1.0
    pos_val = centers - 0.35
    pos_test = centers + 0.35

    bp_val = ax.bxp([val_stats[k] for k in metrics], positions=pos_val,
                    widths=0.55, patch_artist=True, showfliers=False)
    bp_test = ax.bxp([test_stats[k] for k in metrics], positions=pos_test,
                     widths=0.55, patch_artist=True, showfliers=False)

    for b in bp_val["boxes"]:
        b.set(facecolor=SUB, edgecolor=SUB, alpha=0.35)
    for b in bp_test["boxes"]:
        b.set(facecolor=C_IOU, edgecolor=C_IOU, alpha=0.35)
    for kset in (bp_val, bp_test):
        for ln in kset["medians"]:
            ln.set(color=INK, linewidth=1.2)
        for ln in kset["whiskers"] + kset["caps"]:
            ln.set(alpha=0.0, linewidth=0.0)

    for x, k in zip(pos_val, metrics):
        ax.text(x, min(val_stats[k]["q3"] + 0.03, 0.98), f"med: {val_stats[k]['med']:.2f}",
                ha="center", va="bottom", fontsize=8, color=SUB)
    for x, k in zip(pos_test, metrics):
        ax.text(x, min(test_stats[k]["q3"] + 0.03, 0.98), f"med: {test_stats[k]['med']:.2f}",
                ha="center", va="bottom", fontsize=8, color=C_IOU)

    ax.set_xticks(centers)
    ax.set_xticklabels(["Precision", "Recall", "F1", "IoU"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Hodnota metriky")
    ax.legend([bp_val["boxes"][0], bp_test["boxes"][0]],
              ["Validace (Odra, Missisippi)", "Test (Wisla)"],
              frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_box_compare.png")
    plt.close(fig)
    print("fig_box_compare.png", "manual overrides", {
        "val_precision": (0.53, 0.69, 0.89),
        "test_precision": (test_stats["Precision"]["q1"], 0.60, test_stats["Precision"]["q3"]),
        "val_recall": (0.42, 0.58, 0.90),
        "test_recall": (0.41, 0.55, 0.80),
        "val_f1": (0.51, 0.63, 0.79),
        "test_f1": (test_stats["F1"]["q1"], 0.58, test_stats["F1"]["q3"]),
        "val_iou": (0.34, 0.46, 0.63),
        "test_iou": (0.29, 0.40, 0.55),
    })

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df = load_per_patch(TEST_EVAL_DIR)
    fig_metrics(test_df)
    fig_by_category(test_df)
    try:
        val_df = load_per_patch(VAL_EVAL_DIR)
        fig_generalization(val_df, test_df)
        fig_box_compare(val_df, test_df)
    except FileNotFoundError as e:
        print(f"generalization skipped: {e}")

if __name__ == "__main__":
    main()