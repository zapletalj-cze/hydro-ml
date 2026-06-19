"""
Levee Detection - Thesis Figures and Metrics
=============================================

Reads the CSVs written by train_segformer.py and evaluate_segformer.py and
produces publication-quality (PNG, 300 dpi) figures and summary tables for the
DSM branch of the thesis. Nothing here needs the raw patches, only the CSVs.

Inputs:
    training_history.csv          (train_segformer.py)
        epoch, train_loss, val_loss, val_dice, val_cldice, val_score, lr
    val_results_per_patch.csv     (train_segformer.py, validation on basin A)
    basin_B_results_per_patch.csv (evaluate_segformer.py, held-out basin B)
        patch_id, category, tp, fp, fn, tn, precision, recall, f1,
        dice, iou, cldice, n_label_px, patch_type

Outputs (into OUTPUT_DIR):
    fig_training_curves.png            train/val loss + val Dice/clDice vs epoch
    fig_metrics_by_category.png        per-patch Dice/IoU/clDice by M vs L (basin B)
    fig_generalization_A_vs_B.png      Dice/clDice, validation A vs test B
    table_per_patch_means_basin_B.csv  mean/median per category
    table_micro_basin_B.csv            micro-averaged from summed counts
    table_training_summary.csv         best epoch and its metrics

Any input that is missing is skipped with a message, so the script runs whether
or not basin B has been evaluated yet.

Author:   Jakub Zapletal
Date:     2026-06-18
Version:  0.1
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

TRAINING_HISTORY_CSV = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\training_history.csv")
VAL_RESULTS_CSV      = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\val_results_per_patch.csv")
TEST_RESULTS_CSV     = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\eval_PL_v01\basin_B_results_per_patch.csv")

OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\thesis_figures")


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_TRAIN  = "#444444"
C_VAL    = "#1f77b4"
C_DICE   = "#2ca02c"
C_CLDICE = "#9467bd"
C_A      = "#7f7f7f"
C_B      = "#1f77b4"
EDGE     = "#333333"
MEDIAN   = "#d62728"

# Czech axis labels; metric names kept as-is (Dice, IoU, clDice, ...)
LBL_EPOCH = "Epocha"
LBL_LOSS  = "Ztráta"
LBL_CAT   = "Kategorie (plocha povodí)"

CATEGORY_ORDER  = ["M", "L"]
CATEGORY_LABELS = {"M": "M\n(2 000–10 000 km²)", "L": "L\n(> 10 000 km²)"}

METRICS_DIST = ["dice", "iou", "cldice"]
METRIC_TITLE = {"dice": "Dice", "iou": "IoU", "cldice": "clDice",
                "precision": "Precision", "recall": "Recall", "f1": "F1"}


# ============================================================
# HELPERS
# ============================================================

def load_csv(path, what):
    if not path.exists():
        print(f"  [skip] {what}: not found ({path})")
        return None
    df = pd.read_csv(path)
    print(f"  [ok]   {what}: {len(df)} rows")
    return df


def positives(df):
    """Rows that carry a real levee label (detection-quality metrics only make
    sense here; negatives have an empty label so their Dice is trivially 0)."""
    return df[df["patch_type"] == "positive"].copy()


def styled_boxplot(ax, data, labels, ylabel, color, title=None, ylim=(0, 1)):
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.55,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#999999",
                        markeredgecolor="none", alpha=0.4),
    )
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.45, edgecolor=EDGE, linewidth=1.0)
    for med in bp["medians"]:
        med.set(color=MEDIAN, linewidth=1.6)
    for whisk in bp["whiskers"]:
        whisk.set(color=EDGE, linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color=EDGE, linewidth=1.0)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    if title:
        ax.set_title(title)


def save_fig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")


# ============================================================
# FIGURE 1: TRAINING CURVES
# ============================================================

def figure_training_curves(hist, out_dir):
    best_epoch = int(hist.loc[hist["val_score"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_loss"], color=C_TRAIN, linewidth=1.8,
            linestyle="-", label="trénink")
    ax.plot(hist["epoch"], hist["val_loss"], color=C_VAL, linewidth=1.8,
            linestyle="--", label="validace")
    ax.axvline(best_epoch, color="#888888", linestyle=":", linewidth=1.2,
               label=f"nejlepší epocha ({best_epoch})")
    ax.set_xlabel(LBL_EPOCH)
    ax.set_ylabel(LBL_LOSS)
    ax.set_title("Trénovací a validační ztráta")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(hist["epoch"], hist["val_dice"], color=C_DICE, linewidth=1.8, label="Dice")
    ax.plot(hist["epoch"], hist["val_cldice"], color=C_CLDICE, linewidth=1.8, label="clDice")
    ax.axvline(best_epoch, color="#888888", linestyle=":", linewidth=1.2,
               label=f"nejlepší epocha ({best_epoch})")
    ax.set_xlabel(LBL_EPOCH)
    ax.set_ylabel("Hodnota metriky")
    ax.set_ylim(0, 1)
    ax.set_title("Validační Dice a clDice")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_fig(fig, out_dir / "fig_training_curves.png")


# ============================================================
# FIGURE 2: PER-PATCH METRICS BY CATEGORY (basin B)
# ============================================================

def figure_metrics_by_category(df_b, out_dir):
    pos = positives(df_b)
    cats = [c for c in CATEGORY_ORDER if (pos["category"] == c).any()]
    if not cats:
        print("  [skip] metrics-by-category: no positive patches with M/L category")
        return

    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    colors = [C_DICE, C_VAL, C_CLDICE]

    fig, axes = plt.subplots(1, len(METRICS_DIST), figsize=(10.5, 3.6))
    for ax, metric, color in zip(axes, METRICS_DIST, colors):
        data = [pos.loc[pos["category"] == c, metric].values for c in cats]
        styled_boxplot(ax, data, labels, METRIC_TITLE[metric], color,
                       title=METRIC_TITLE[metric])
        ax.set_xlabel(LBL_CAT)

    fig.suptitle("Rozdělení metrik po patchích podle kategorie (povodí B, pozitivní patche)",
                 y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_metrics_by_category.png")


# ============================================================
# FIGURE 3: GENERALIZATION  validation A vs test B
# ============================================================

def figure_generalization(df_a, df_b, out_dir):
    pos_a = positives(df_a)
    pos_b = positives(df_b)

    metrics = ["dice", "cldice"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.4, 3.6))
    for ax, metric in zip(axes, metrics):
        data = [pos_a[metric].values, pos_b[metric].values]
        styled_boxplot(ax, data, ["validace (A)", "test (B)"], METRIC_TITLE[metric],
                       C_B, title=METRIC_TITLE[metric])

    fig.suptitle("Generalizace mezi povodími (pozitivní patche)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_generalization_A_vs_B.png")


# ============================================================
# TABLES
# ============================================================

def micro_from_counts(df):
    tp = float(df["tp"].sum())
    fp = float(df["fp"].sum())
    fn = float(df["fn"].sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    return {"Precision": precision, "Recall": recall, "F1": f1, "IoU": iou, "Dice": dice}


def tables_basin_B(df_b, out_dir):
    pos = positives(df_b)

    # Per-patch means/medians by category (positives only)
    rows = []
    groups = [(c, pos[pos["category"] == c]) for c in CATEGORY_ORDER
              if (pos["category"] == c).any()]
    groups.append(("vše (pozitivní)", pos))
    for name, g in groups:
        if len(g) == 0:
            continue
        row = {"skupina": name, "n_patchu": len(g)}
        for m in ["dice", "iou", "cldice"]:
            row[f"{METRIC_TITLE[m]}_mean"]   = round(float(g[m].mean()), 4)
            row[f"{METRIC_TITLE[m]}_median"] = round(float(g[m].median()), 4)
        for m in ["precision", "recall", "f1"]:
            row[f"{METRIC_TITLE[m]}_mean"] = round(float(g[m].mean()), 4)
        rows.append(row)
    means_df = pd.DataFrame(rows)
    means_path = out_dir / "table_per_patch_means_basin_B.csv"
    means_df.to_csv(means_path, index=False)
    print(f"  saved {means_path.name}")

    # Micro-averaged from summed counts: positives only, and overall (incl. negatives)
    micro_rows = []
    mp = micro_from_counts(pos); mp["skupina"] = "pozitivní (mikro)"; micro_rows.append(mp)
    mo = micro_from_counts(df_b); mo["skupina"] = "celkově vč. negativ (mikro)"; micro_rows.append(mo)
    micro_df = pd.DataFrame(micro_rows)[
        ["skupina", "Precision", "Recall", "F1", "IoU", "Dice"]
    ].round(4)
    micro_path = out_dir / "table_micro_basin_B.csv"
    micro_df.to_csv(micro_path, index=False)
    print(f"  saved {micro_path.name}")

    print("\n  Per-patch means by category (basin B):")
    print(means_df.to_string(index=False))
    print("\n  Micro-averaged (basin B):")
    print(micro_df.to_string(index=False))


def table_training_summary(hist, out_dir):
    best_i = int(hist["val_score"].idxmax())
    best = hist.iloc[best_i]
    summary = pd.DataFrame([{
        "best_epoch":       int(best["epoch"]),
        "val_dice":         round(float(best["val_dice"]), 4),
        "val_cldice":       round(float(best["val_cldice"]), 4),
        "val_score":        round(float(best["val_score"]), 4),
        "val_loss":         round(float(best["val_loss"]), 4),
        "train_loss":       round(float(best["train_loss"]), 4),
        "final_train_loss": round(float(hist["train_loss"].iloc[-1]), 4),
        "final_val_loss":   round(float(hist["val_loss"].iloc[-1]), 4),
        "n_epochs":         int(hist["epoch"].max()),
    }])
    path = out_dir / "table_training_summary.csv"
    summary.to_csv(path, index=False)
    print(f"  saved {path.name}")
    print("\n  Training summary:")
    print(summary.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}\n")

    print("Loading inputs...")
    hist = load_csv(TRAINING_HISTORY_CSV, "training history")
    df_a = load_csv(VAL_RESULTS_CSV, "validation (basin A)")
    df_b = load_csv(TEST_RESULTS_CSV, "test (basin B)")

    print("\nFigures and tables...")
    if hist is not None:
        figure_training_curves(hist, OUTPUT_DIR)
        table_training_summary(hist, OUTPUT_DIR)

    if df_b is not None:
        figure_metrics_by_category(df_b, OUTPUT_DIR)
        tables_basin_B(df_b, OUTPUT_DIR)

    if df_a is not None and df_b is not None:
        figure_generalization(df_a, df_b, OUTPUT_DIR)
    elif df_b is not None:
        print("  [skip] generalization figure: needs both validation (A) and test (B)")

    print("\nDone.")


if __name__ == "__main__":
    main()
