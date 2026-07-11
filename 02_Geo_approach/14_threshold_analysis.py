"""
Decision-threshold analysis from saved predictions
==================================================

Sweeps the decision threshold on the sigmoid probabilities saved by the
evaluation script and quantifies precision/recall/F1/IoU/Dice as a function of
the threshold, micro-averaged from summed pixel counts. Finds the F1-optimal
threshold and renders a thesis-quality figure (PR curve + metrics vs threshold).

Use on a CALIBRATION set only (US calib subset, or a validation subset of
basin A for the Odra threshold justification). Reporting metrics at a threshold
tuned on the same set that is reported would be circular.

Inputs:
    PREDICTIONS_DIR   predictions_<split>/*.npz written by the eval script
                      (key "pred", float16 sigmoid probabilities)
    PATCHES_DIR       original patches (*.npz with the "label" channel)
    METADATA_CSV      defines WHICH patches enter the sweep (calib subset)

Outputs (OUTPUT_DIR):
    fig_threshold_analysis.png   PR curve + metrics vs threshold (Czech labels)
    table_threshold_sweep.csv    per-threshold metrics (all patches + positives)
    best_threshold.json          F1-optimal threshold and its metrics

clDice is intentionally excluded from the sweep: it needs a skeletonization per
threshold per patch and does not drive the operating-point choice; the overlap
metrics do.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US\eval_basin_Odra\predictions_basin_B")
PATCHES_DIR     = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_test\patches")
METADATA_CSV    = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_test\patches\eval_prep\metadata_odra_calib.csv")

OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_test\patches\eval_prep\threshold_analysis")

LABEL_CHANNEL = "label"

# Threshold grid: bin edges for the histogram trick (cumulative counts give
# exact TP/FP/FN at every threshold in one pass per patch).
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

REFERENCE_THRESHOLD = 0.5    # marked in the figure for comparison

# ============================================================
# STYLE (thesis palette)
# ============================================================

INK, SECOND, GRIDCOL, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_DICE, C_IOU, C_F1 = "#0E7C7B", "#2B6CB0", "#C2410C"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlepad": 10,
    "axes.labelsize": 11, "axes.labelcolor": INK,
    "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _tidy(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=0)


# ============================================================
# SWEEP (histogram trick: one pass per patch, exact counts)
# ============================================================

def accumulate_counts(df_meta):
    """Cumulative pixel counts above each threshold, split by label class.
    Returns per-threshold TP/FP/FN/TN for all patches and for positives only."""
    edges = np.concatenate([[0.0], THRESHOLDS, [1.0 + 1e-6]])
    n_t = len(THRESHOLDS)

    def zero():
        return {"tp": np.zeros(n_t), "fp": np.zeros(n_t),
                "fn": np.zeros(n_t), "tn": np.zeros(n_t)}
    acc_all, acc_pos = zero(), zero()

    n_used, n_missing = 0, 0
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Sweeping"):
        pid = str(row["patch_id"])
        pred_path = PREDICTIONS_DIR / f"{pid}.npz"
        patch_path = PATCHES_DIR / f"{pid}.npz"
        if not pred_path.exists() or not patch_path.exists():
            n_missing += 1
            continue

        pred = np.load(pred_path)["pred"].astype(np.float32).ravel()
        label = np.nan_to_num(
            dict(np.load(patch_path))[LABEL_CHANNEL].astype(np.float32)).ravel()
        pos = label > 0.5

        # counts of prediction values in each bin, per label class
        h_pos, _ = np.histogram(pred[pos], bins=edges)
        h_neg, _ = np.histogram(pred[~pos], bins=edges)
        # pixels ABOVE threshold t_i = suffix sum of bins i+1..end
        above_pos = np.cumsum(h_pos[::-1])[::-1][1:]   # len n_t
        above_neg = np.cumsum(h_neg[::-1])[::-1][1:]

        tp = above_pos
        fp = above_neg
        fn = pos.sum() - above_pos
        tn = (~pos).sum() - above_neg

        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
            acc_all[k] += v
            if row.get("patch_type", "positive") == "positive":
                acc_pos[k] += v
        n_used += 1

    if n_used == 0:
        raise RuntimeError("No patches with both prediction and label found.")
    print(f"  patches used: {n_used} (missing: {n_missing})")
    return acc_all, acc_pos


def metrics_from_counts(acc):
    eps = 1e-9
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return precision, recall, f1, iou, dice


# ============================================================
# FIGURE
# ============================================================

def fig_threshold(thresholds, p, r, f1, iou, dice, best_i, out_path):
    ref_i = int(np.argmin(np.abs(thresholds - REFERENCE_THRESHOLD)))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    # PR curve
    ax = axes[0]
    ax.plot(r, p, color=C_DICE, lw=2.2, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgecolor=C_DICE)
    # Separate offsets so the two markers (which sit close together) don't overlap.
    for i, label, color, xytext, ha, va in (
        (ref_i, f"práh {REFERENCE_THRESHOLD}", SECOND, (0, -14), "center", "top"),
        (best_i, f"práh {thresholds[best_i]:.2f} (opt.)", C_F1, (0, 12), "center", "bottom"),
    ):
        ax.scatter([r[i]], [p[i]], s=60, zorder=5, color=color)
        ax.annotate(label, (r[i], p[i]), textcoords="offset points",
                    xytext=xytext, ha=ha, va=va, fontsize=9.5, color=color)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    # Auto-zoom to the data range (with margins) so the curve fills the panel
    # instead of being squeezed into a corner of a fixed 0-1 box.
    rmin, rmax = float(np.min(r)), float(np.max(r))
    pmin, pmax = float(np.min(p)), float(np.max(p))
    rpad = max(0.05, 0.15 * (rmax - rmin))
    ppad = max(0.05, 0.15 * (pmax - pmin))
    ax.set_xlim(max(0.0, rmin - rpad), min(1.0, rmax + rpad))
    ax.set_ylim(max(0.0, pmin - ppad), min(1.0, pmax + ppad))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_title("Křivka precision–recall")
    ax.grid(axis="both", color=GRIDCOL, linewidth=0.8)
    _tidy(ax)

    # metrics vs threshold
    ax = axes[1]
    ax.plot(thresholds, f1, color=C_F1, lw=2.2, label="F1")
    ax.plot(thresholds, dice, color=C_DICE, lw=2.0, ls=(0, (5, 2)), label="Dice")
    ax.plot(thresholds, iou, color=C_IOU, lw=2.0, label="IoU")
    ax.axvline(REFERENCE_THRESHOLD, color=SECOND, ls=":", lw=1.2,
               label=f"práh {REFERENCE_THRESHOLD}")
    ax.axvline(thresholds[best_i], color=C_F1, ls=":", lw=1.4,
               label=f"optimum ({thresholds[best_i]:.2f})")
    ax.set_xlabel("Rozhodovací práh"); ax.set_ylabel("Hodnota metriky")
    ax.set_xlim(thresholds[0], thresholds[-1]); ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_title("Metriky v závislosti na prahu")
    _tidy(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle("Volba rozhodovacího prahu (mikro-průměr, kalibrační vzorek)",
                 y=1.03, fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(METADATA_CSV)
    print(f"Calibration patches: {len(df_meta)}")

    acc_all, acc_pos = accumulate_counts(df_meta)

    p, r, f1, iou, dice = metrics_from_counts(acc_all)
    pp, rp, f1p, ioup, dicep = metrics_from_counts(acc_pos)
    best_i = int(np.argmax(f1))

    table = pd.DataFrame({
        "threshold": THRESHOLDS,
        "precision": np.round(p, 4), "recall": np.round(r, 4),
        "f1": np.round(f1, 4), "iou": np.round(iou, 4), "dice": np.round(dice, 4),
        "precision_pos": np.round(pp, 4), "recall_pos": np.round(rp, 4),
        "f1_pos": np.round(f1p, 4), "iou_pos": np.round(ioup, 4),
        "dice_pos": np.round(dicep, 4),
    })
    table_path = OUTPUT_DIR / "table_threshold_sweep.csv"
    table.to_csv(table_path, index=False)
    print(f"  saved {table_path.name}")

    best = {
        "best_threshold": float(THRESHOLDS[best_i]),
        "criterion": "micro F1, all patches",
        "f1": float(f1[best_i]), "precision": float(p[best_i]),
        "recall": float(r[best_i]), "iou": float(iou[best_i]),
        "dice": float(dice[best_i]),
        "reference_threshold": REFERENCE_THRESHOLD,
        "f1_at_reference": float(f1[int(np.argmin(np.abs(THRESHOLDS - REFERENCE_THRESHOLD)))]),
    }
    with open(OUTPUT_DIR / "best_threshold.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"  best threshold: {best['best_threshold']:.2f} "
          f"(F1 {best['f1']:.4f} vs F1@{REFERENCE_THRESHOLD} "
          f"{best['f1_at_reference']:.4f})")

    fig_threshold(THRESHOLDS, p, r, f1, iou, dice, best_i,
                  OUTPUT_DIR / "fig_threshold_analysis.png")
    print("Done.")


if __name__ == "__main__":
    main()
