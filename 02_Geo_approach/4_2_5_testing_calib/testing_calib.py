"""Result plots for the final model on the Vistula test basin.
Three PNG figures (thesis style, no titles) from existing eval CSVs:
  fig_metrics.png        micro precision/recall/F1/IoU bar
  fig_by_category.png    metric by basin-area category (M vs L)
  fig_generalization.png validation vs test, same metrics side by side
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
TEST_EVAL_DIR = TRAIN_DIR / "eval_basin_B"      # Wisla test outputs
VAL_EVAL_DIR  = TRAIN_DIR / "eval_val"          # internal validation outputs

OUT_DIR = Path(__file__).parent / "diagnostics_ch4"

# ---- style ------------------------------------------------------------------
INK, SUB, GRID, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_P, C_R, C_F1, C_IOU = "#0E7C7B", "#6B7280", "#C2410C", "#2B6CB0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
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
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, 1); ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Hodnota metriky"); _tidy(ax)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_metrics.png"); plt.close(fig)
    print("fig_metrics.png", {k: round(v,3) for k,v in m.items()})

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
        ax.bar(x + (i-1.5)*w, data[mt], width=w, color=cols[i], label=mt, zorder=3)
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
    mv, mt = micro(val_df), micro(test_df)
    metrics = list(mt)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w/2, [mv[k] for k in metrics], width=w, color=SUB,
           label="vnitřní validace", zorder=3)
    ax.bar(x + w/2, [mt[k] for k in metrics], width=w, color=C_IOU,
           label="test (Visla)", zorder=3)
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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df = load_per_patch(TEST_EVAL_DIR)
    fig_metrics(test_df)
    fig_by_category(test_df)
    try:
        val_df = load_per_patch(VAL_EVAL_DIR)
        fig_generalization(val_df, test_df)
    except FileNotFoundError as e:
        print(f"generalization skipped: {e}")

if __name__ == "__main__":
    main()