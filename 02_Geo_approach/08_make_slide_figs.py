"""
make_slide_figs.py
==================
Generate presentation figures for the elevation-based levee detection, styled in
the deck palette (teal / warm / navy), from the CSVs produced by the training
and evaluation scripts. No raw patches needed, only CSVs.

INPUT CSV SCHEMAS (must match what the pipeline writes):

  training_history.csv            (03_train_segformer_v02.py)
      epoch, train_loss, val_loss, val_dice, val_cldice, val_score, lr

  basin_B_results_per_patch.csv   (03b_evaluate_train_segformer_v02.py)
      patch_id, category, tp, fp, fn, tn,
      precision, recall, f1, dice, iou, cldice, n_label_px, patch_type

  val_results_per_patch.csv       (optional, validation on basin A, same schema)

  comparison_overall.csv          (optional, 03b_compare_variants.py)
      variant, ..., test_mean_dice, test_mean_iou, test_mean_cldice,
      test_mean_f1, test_mean_precision, test_mean_recall, ...

Every input is optional: a missing file is skipped with a message, so the
script runs whether or not basin B / the ablation has been produced yet.

Outputs (PNG, 200 dpi, transparent where useful) into OUTPUT_DIR.

Author:  prepared for Jakub Zapletal
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIG  -- edit these paths to your run
# ============================================================

TRAINING_HISTORY_CSV = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\training_history.csv")
BASIN_B_CSV          = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\eval_basin_B\basin_B_results_per_patch.csv")
VAL_A_CSV            = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v04_segformer\training_v01\val_results_per_patch.csv")    
ABLATION_OVERALL_CSV = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_comparison\comparison_overall.csv")        # optional

OUTPUT_DIR = Path("slide_figures")

# Categories present in the data, in display order. S is included so the script
# also works on the v03 channel-ablation data (S/M/L); on M/L-only data the
# missing category is simply skipped.
CATEGORY_ORDER  = ["S", "M", "L"]
CATEGORY_LABELS = {
    "S": "S",
    "M": "M\n(2 000–10 000 km²)",
    "L": "L\n(> 10 000 km²)",
}

# ============================================================
# DECK PALETTE + STYLE
# ============================================================

TEAL   = "#0E7C7B"
TEAL_L = "#7FB7B6"
WARM   = "#C2410C"
NAVY   = "#13293D"
INK    = "#1E293B"
MUTED  = "#64748B"
GRID   = "#E2E8F0"
DICE_C = "#0E7C7B"
IOU_C  = "#2B6CB0"
CLD_C  = "#7C3AED"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 1.0,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#94A3B8",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
})

LBL_EPOCH = "Epocha"
LBL_LOSS  = "Ztráta"
LBL_CAT   = "Kategorie (plocha povodí)"
METRIC_TITLE = {"dice": "Dice", "iou": "IoU", "cldice": "clDice",
                "precision": "Precision", "recall": "Recall", "f1": "F1"}


def load_csv(path, what):
    if not Path(path).exists():
        print(f"  [skip] {what}: not found ({path})")
        return None
    df = pd.read_csv(path)
    print(f"  [ok]   {what}: {len(df)} rows")
    return df


def positives(df):
    if "patch_type" not in df.columns:
        return df.copy()
    return df[df["patch_type"] == "positive"].copy()


def save_fig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {Path(path).name}")


def style_box(ax, data, labels, color, title, ylim=(0, 1)):
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="#B6C2CF",
                                    markeredgecolor="none", alpha=0.5),
                    medianprops=dict(color=WARM, linewidth=1.8),
                    whiskerprops=dict(color="#94A3B8", linewidth=1.0),
                    capprops=dict(color="#94A3B8", linewidth=1.0))
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.4)
    ax.set_title(title, color=INK)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


# ============================================================
# FIGURE 1: TRAINING CURVES
# ============================================================

def figure_training_curves(hist, out_dir):
    need = {"epoch", "train_loss", "val_loss", "val_dice", "val_cldice", "val_score"}
    if not need.issubset(hist.columns):
        print(f"  [skip] training curves: missing columns {need - set(hist.columns)}")
        return
    best_epoch = int(hist.loc[hist["val_score"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_loss"], color=NAVY, lw=2.0, label="trénink")
    ax.plot(hist["epoch"], hist["val_loss"], color=TEAL, lw=2.0, ls="--", label="validace")
    ax.axvline(best_epoch, color=MUTED, ls=":", lw=1.2, label=f"nejlepší epocha ({best_epoch})")
    ax.set_xlabel(LBL_EPOCH); ax.set_ylabel(LBL_LOSS)
    ax.set_title("Trénovací a validační ztráta")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(hist["epoch"], hist["val_dice"], color=DICE_C, lw=2.0, label="Dice")
    ax.plot(hist["epoch"], hist["val_cldice"], color=CLD_C, lw=2.0, label="clDice")
    ax.axvline(best_epoch, color=MUTED, ls=":", lw=1.2, label=f"nejlepší epocha ({best_epoch})")
    ax.set_xlabel(LBL_EPOCH); ax.set_ylabel("Hodnota metriky"); ax.set_ylim(0, 1)
    ax.set_title("Validační Dice a clDice")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_fig(fig, out_dir / "fig_training_curves.png")


# ============================================================
# FIGURE 2: BASIN B HEADLINE METRICS (micro-averaged + mean clDice)
# ============================================================

def figure_basin_b_headline(df_b, out_dir):
    pos = positives(df_b)
    if len(pos) == 0:
        print("  [skip] basin B headline: no positive patches")
        return
    tp, fp, fn = float(pos["tp"].sum()), float(pos["fp"].sum()), float(pos["fn"].sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    cldice    = float(pos["cldice"].mean())  # clDice has no additive counts; mean per patch

    names  = ["Precision", "Recall", "IoU", "Dice", "clDice"]
    values = [precision, recall, iou, dice, cldice]
    colors = [MUTED, MUTED, IOU_C, DICE_C, CLD_C]

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    bars = ax.bar(names, values, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hodnota")
    ax.set_title("Metriky na nezávislém testovacím povodí (Odra)")
    ax.grid(axis="x", visible=False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    fig.text(0.012, 0.015, "Precision, Recall, IoU a Dice průměrované ze sečtených TP/FP/FN. clDice je průměr po patchích.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, out_dir / "fig_basin_B_headline.png")


# ============================================================
# FIGURE 3: BASIN B METRICS BY CATEGORY (M vs L)
# ============================================================

def figure_metrics_by_category(df_b, out_dir):
    pos = positives(df_b)
    if "category" not in pos.columns:
        print("  [skip] metrics by category: no 'category' column")
        return
    cats = [c for c in CATEGORY_ORDER if (pos["category"] == c).any()]
    if not cats:
        print("  [skip] metrics by category: no recognised categories present")
        return
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    metrics = ["dice", "iou", "cldice"]
    colors  = [DICE_C, IOU_C, CLD_C]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    for ax, m, c in zip(axes, metrics, colors):
        data = [pos.loc[pos["category"] == cat, m].values for cat in cats]
        style_box(ax, data, labels, c, METRIC_TITLE[m])
        ax.set_xlabel(LBL_CAT)
    fig.suptitle("Rozdělení metrik po patchích podle kategorie povodí (Odra, pozitivní patche)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_metrics_by_category.png")


# ============================================================
# FIGURE 4: GENERALIZATION  validation A vs test B
# ============================================================

def figure_generalization(df_a, df_b, out_dir):
    pa, pb = positives(df_a), positives(df_b)
    if len(pa) == 0 or len(pb) == 0:
        print("  [skip] generalization: need positives in both A and B")
        return
    metrics = ["dice", "cldice"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, m in zip(axes, metrics):
        style_box(ax, [pa[m].values, pb[m].values],
                  ["validace (A)", "test (B)"], TEAL, METRIC_TITLE[m])
    fig.suptitle("Generalizace mezi povodími (pozitivní patche)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_generalization_A_vs_B.png")


# ============================================================
# FIGURE 5: ABLATION (grouped bars per variant)
# ============================================================
# Generic over whatever variants the CSV contains: it reads the 'variant'
# column and the test_mean_* columns. Works for channel ablation
# (v1 DSM / v2 DSM+TPI / v3 DSM+TPI+aux) OR an architecture ablation, depending
# only on which comparison_overall.csv you point it at. The figure title is
# deliberately neutral; set ABLATION_TITLE to match the data you feed it.

ABLATION_TITLE = "Ablace, metriky na testovacím povodí"

def figure_ablation(df_overall, out_dir):
    if "variant" not in df_overall.columns:
        print("  [skip] ablation: no 'variant' column")
        return
    metric_cols = [("test_mean_dice", "Dice", DICE_C),
                   ("test_mean_iou", "IoU", IOU_C),
                   ("test_mean_cldice", "clDice", CLD_C)]

    # F1 and Dice are equivalent for binary segmentation; only include F1 if Dice is unavailable.
    if "test_mean_dice" not in df_overall.columns and "test_mean_f1" in df_overall.columns:
        metric_cols.insert(0, ("test_mean_f1", "Dice/F1", DICE_C))
    present = [(col, lbl, c) for col, lbl, c in metric_cols if col in df_overall.columns]
    if not present:
        print("  [skip] ablation: no test_mean_* columns found")
        return
    variants = df_overall["variant"].tolist()
    x = np.arange(len(present))
    n = len(variants)
    bw = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    variant_colors = [NAVY, TEAL, WARM, IOU_C, CLD_C, MUTED]
    for i, (_, row) in enumerate(df_overall.iterrows()):
        vals = [float(row[col]) for col, _, _ in present]
        off = (i - n / 2 + 0.5) * bw
        bars = ax.bar(x + off, vals, bw, label=str(row["variant"]),
                      color=variant_colors[i % len(variant_colors)],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in present])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hodnota (mean per-patch)")
    ax.set_title(ABLATION_TITLE)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=min(n, 3), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_ablation.png")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}\n")

    print("Loading inputs...")
    hist       = load_csv(TRAINING_HISTORY_CSV, "training history")
    df_b       = load_csv(BASIN_B_CSV, "test (basin B)")
    df_a       = load_csv(VAL_A_CSV, "validation (basin A)")
    df_overall = load_csv(ABLATION_OVERALL_CSV, "ablation overall")

    print("\nFigures...")
    if hist is not None:
        figure_training_curves(hist, OUTPUT_DIR)
    if df_b is not None:
        figure_basin_b_headline(df_b, OUTPUT_DIR)
        figure_metrics_by_category(df_b, OUTPUT_DIR)
    if df_a is not None and df_b is not None:
        figure_generalization(df_a, df_b, OUTPUT_DIR)
    if df_overall is not None:
        figure_ablation(df_overall, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
