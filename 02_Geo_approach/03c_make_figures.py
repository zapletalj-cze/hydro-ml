"""
Levee Detection - Thesis Figures (training + test + architecture comparison)
===========================================================================

Single visualization entry point for the diploma thesis. Consumes the CSVs
already written by the training and evaluation scripts (no recomputation, no
raw patches, no torch) and produces publication-quality figures:

  Per model (figures/<name>/):
    training_curves.png     train/val loss + val Dice/clDice/score vs epoch
    test_headline.png       micro-averaged P/R/F1/IoU/Dice + mean clDice (basin B)
    test_by_category.png    per-patch Dice/IoU/clDice by category M/L (basin B)
    generalization.png      Dice/clDice, validation (A) vs test (B)   [if val CSV]
    overview.png            combined training + test on one canvas

  Architecture comparison (figures/_comparison/):
    curves_comparison.png       val score / Dice / clDice, one line per architecture
    test_metrics_comparison.png grouped bars, metrics x architectures (basin B)
    test_by_category_comparison.png  Dice + clDice by category, grouped by architecture
    comparison_table.png        rendered summary table
    comparison_summary.csv      machine-readable summary

Expected per-run layout (produced by the training + evaluation scripts):
    <run_dir>/training_history.csv
    <run_dir>/val_results_per_patch.csv                     (optional)
    <run_dir>/eval_basin_B/basin_B_results_per_patch.csv

Author:   prepared for Jakub Zapletal
Language: figure labels in Czech (thesis), code/comments in English
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# ============================================================
# CONFIG  -  edit run directories to your machine
# ============================================================

BASE = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL")

# The three architecture runs to compare. "arch" only drives the colour mapping.
RUNS = [
    {"name": "SegFormer (mit_b2)", "arch": "segformer",     "dir": BASE / "training_v05_segformer"},
    {"name": "U-Net (ResNet34)",   "arch": "resnet_unet",   "dir": BASE / "training_v05_resnet_unet"},
    {"name": "DeepLabV3+",         "arch": "deeplabv3plus", "dir": BASE / "training_v05_deeplabv3plus"},
]

OUTPUT_DIR = BASE / "thesis_figures"

# Detection-quality metrics are meaningful on patches that contain a levee.
# True  -> micro-average over positive patches only (matches fig_basin_B_headline).
# False -> micro-average over all patches (includes false alarms on negatives).
POSITIVES_ONLY = True

# Basin B name inside each run directory
EVAL_SUBDIR = "eval_basin_B"
TEST_CSV_NAME = "basin_B_results_per_patch.csv"
VAL_CSV_NAME = "val_results_per_patch.csv"
HISTORY_CSV_NAME = "training_history.csv"

# ============================================================
# STYLE  (refined, publication-quality)
# ============================================================

INK, SECOND, GRIDCOL, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_TRAIN, C_VAL = "#374151", "#0E7C7B"
C_DICE, C_IOU, C_CLDICE = "#0E7C7B", "#2B6CB0", "#7C3AED"
C_SCORE = "#C2410C"
C_A, C_B = "#9AA7B4", "#0E7C7B"
MEDIAN, WHISK = "#1F2937", "#9AA7B4"

# Per-architecture colours for comparison plots (SegFormer highlighted in teal).
ARCH_COLOR = {"segformer": "#0E7C7B", "resnet_unet": "#13293D", "deeplabv3plus": "#C2410C"}
ARCH_FALLBACK = ["#0E7C7B", "#13293D", "#C2410C", "#2B6CB0", "#7C3AED", "#6B7280"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlepad": 10,
    "axes.labelsize": 11, "axes.labelcolor": INK,
    "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRIDCOL, "grid.alpha": 1.0, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

LBL_EPOCH, LBL_CAT = "Epocha", "Kategorie (plocha povodí)"
CATEGORY_ORDER = ["S", "M", "L"]
CATEGORY_LABELS = {
    "S": "S\n(< 2 000 km²)",
    "M": "M\n(2 000–10 000 km²)",
    "L": "L\n(> 10 000 km²)",
}
METRIC_TITLE = {"dice": "Dice", "iou": "IoU", "cldice": "clDice",
                "precision": "Precision", "recall": "Recall", "f1": "F1"}


# ============================================================
# HELPERS
# ============================================================

def arch_color(arch, idx):
    return ARCH_COLOR.get(arch, ARCH_FALLBACK[idx % len(ARCH_FALLBACK)])


def _read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def load_run(run):
    """Load the CSVs for one run. Missing files come back as None."""
    d = Path(run["dir"])
    hist = _read_csv(d / HISTORY_CSV_NAME)
    test = _read_csv(d / EVAL_SUBDIR / TEST_CSV_NAME)
    val = _read_csv(d / VAL_CSV_NAME)
    return {"name": run["name"], "arch": run["arch"], "dir": d,
            "hist": hist, "test": test, "val": val}


def positives(df):
    if df is None or "patch_type" not in df.columns:
        return df
    return df[df["patch_type"] == "positive"].copy()


def micro_metrics(df):
    """Micro-averaged P/R/F1/IoU/Dice from summed counts, plus mean clDice."""
    sub = positives(df) if POSITIVES_ONLY else df
    tp, fp, fn = float(sub["tp"].sum()), float(sub["fp"].sum()), float(sub["fn"].sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    cldice = float(positives(df)["cldice"].mean())  # clDice: no additive counts
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "dice": dice, "cldice": cldice}


def best_epoch_row(hist):
    return hist.loc[hist["val_score"].idxmax()]


def _tidy(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=0)


def styled_boxplot(ax, data, labels, color, title=None, ylim=(0, 1)):
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.52,
        showmeans=True,
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor=WHISK,
                        markeredgecolor="none", alpha=0.35),
        medianprops=dict(color=MEDIAN, linewidth=1.5),
        whiskerprops=dict(color=WHISK, linewidth=1.1),
        capprops=dict(color=WHISK, linewidth=1.1),
        meanprops=dict(marker="D", markersize=5, markerfacecolor="white",
                       markeredgecolor=color, markeredgewidth=1.3),
    )
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.20, edgecolor=color, linewidth=1.5)
    if ylim:
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="x", visible=False)
    _tidy(ax)
    if title:
        ax.set_title(title)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {Path(path).relative_to(OUTPUT_DIR)}")


def present_categories(df_pos):
    return [c for c in CATEGORY_ORDER if (df_pos["category"] == c).any()]


# ============================================================
# PER-MODEL FIGURES
# ============================================================

def fig_training_curves(hist, out):
    if hist is None:
        return
    be = int(best_epoch_row(hist)["epoch"])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax[0].plot(hist["epoch"], hist["train_loss"], color=C_TRAIN, lw=2.1, label="trénink")
    ax[0].plot(hist["epoch"], hist["val_loss"], color=C_VAL, lw=2.1, ls=(0, (5, 2)), label="validace")
    ax[0].axvline(be, color=SECOND, ls=":", lw=1.1)
    ax[0].set_xlabel(LBL_EPOCH); ax[0].set_ylabel("Ztráta"); ax[0].set_title("Trénovací a validační ztráta")
    ax[0].set_xlim(left=hist["epoch"].min()); _tidy(ax[0]); ax[0].legend(frameon=False)

    ax[1].plot(hist["epoch"], hist["val_dice"], color=C_DICE, lw=2.1, label="Dice")
    ax[1].plot(hist["epoch"], hist["val_cldice"], color=C_CLDICE, lw=2.1, label="clDice")
    ax[1].plot(hist["epoch"], hist["val_score"], color=C_SCORE, lw=2.4, label="skóre")
    ax[1].axvline(be, color=SECOND, ls=":", lw=1.1, label=f"nejlepší epocha ({be})")
    ax[1].set_xlabel(LBL_EPOCH); ax[1].set_ylabel("Hodnota metriky"); ax[1].set_ylim(0, 1)
    ax[1].yaxis.set_major_locator(MultipleLocator(0.2)); ax[1].set_xlim(left=hist["epoch"].min())
    ax[1].set_title("Validační Dice, clDice a skóre"); _tidy(ax[1]); ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out)


def fig_test_headline(test, out):
    if test is None:
        return
    m = micro_metrics(test)
    names = ["Precision", "Recall", "F1", "IoU", "Dice", "clDice"]
    keys = ["precision", "recall", "f1", "iou", "dice", "cldice"]
    colors = [SECOND, SECOND, C_DICE, C_IOU, C_DICE, C_CLDICE]
    vals = [m[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    bars = ax.bar(names, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1); ax.set_ylabel("Hodnota")
    ax.set_title("Metriky na nezávislém testovacím povodí (Odra)")
    ax.grid(axis="x", visible=False); _tidy(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK)
    scope = "pozitivní patche" if POSITIVES_ONLY else "všechny patche"
    fig.text(0.012, 0.015, f"Precision, Recall, F1, IoU a Dice mikro-průměrované ({scope}). clDice je průměr po patchích.",
             fontsize=7.5, color=SECOND)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, out)


def fig_test_by_category(test, out):
    pos = positives(test)
    if pos is None or "category" not in pos.columns:
        return
    cats = present_categories(pos)
    if not cats:
        return
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.7))
    for a, metric, col in zip(ax, ["dice", "iou", "cldice"], [C_DICE, C_IOU, C_CLDICE]):
        data = [pos.loc[pos["category"] == c, metric].values for c in cats]
        styled_boxplot(a, data, labels, col, METRIC_TITLE[metric])
        a.set_xlabel(LBL_CAT)
    fig.suptitle("Rozdělení metrik po patchích podle kategorie (Odra, pozitivní patche)",
                 y=1.03, fontsize=12.5, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out)


def fig_generalization(val, test, out):
    pa, pb = positives(val), positives(test)
    if pa is None or pb is None or len(pa) == 0 or len(pb) == 0:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.7))
    for a, metric in zip(ax, ["dice", "cldice"]):
        bp = a.boxplot([pa[metric].values, pb[metric].values],
                       labels=["validace (A)", "test (B)"], patch_artist=True, widths=0.52,
                       showmeans=True,
                       flierprops=dict(marker="o", markersize=2.5, markerfacecolor=WHISK,
                                       markeredgecolor="none", alpha=0.35),
                       medianprops=dict(color=MEDIAN, linewidth=1.5),
                       whiskerprops=dict(color=WHISK, linewidth=1.1),
                       capprops=dict(color=WHISK, linewidth=1.1),
                       meanprops=dict(marker="D", markersize=5, markerfacecolor="white",
                                      markeredgewidth=1.3))
        for box, c in zip(bp["boxes"], (C_A, C_B)):
            box.set(facecolor=c, alpha=0.22, edgecolor=c, linewidth=1.5)
        for mean, c in zip(bp["means"], (C_A, C_B)):
            mean.set(markeredgecolor=c)
        a.set_ylabel(METRIC_TITLE[metric]); a.set_ylim(0, 1)
        a.yaxis.set_major_locator(MultipleLocator(0.2)); a.set_title(METRIC_TITLE[metric])
        a.grid(axis="x", visible=False); _tidy(a)
    fig.suptitle("Generalizace mezi povodími (pozitivní patche)",
                 y=1.03, fontsize=12.5, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out)


def fig_overview(run, out):
    """Combined training + test canvas for a single model."""
    hist, test = run["hist"], run["test"]
    if hist is None and test is None:
        return
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28)

    if hist is not None:
        be = int(best_epoch_row(hist)["epoch"])
        a0 = fig.add_subplot(gs[0, 0])
        a0.plot(hist["epoch"], hist["train_loss"], color=C_TRAIN, lw=2.0, label="trénink")
        a0.plot(hist["epoch"], hist["val_loss"], color=C_VAL, lw=2.0, ls=(0, (5, 2)), label="validace")
        a0.axvline(be, color=SECOND, ls=":", lw=1.1)
        a0.set_title("Ztráta"); a0.set_xlabel(LBL_EPOCH); a0.set_ylabel("Ztráta")
        a0.set_xlim(left=hist["epoch"].min()); _tidy(a0); a0.legend(frameon=False)

        a1 = fig.add_subplot(gs[0, 1:])
        a1.plot(hist["epoch"], hist["val_dice"], color=C_DICE, lw=2.0, label="Dice")
        a1.plot(hist["epoch"], hist["val_cldice"], color=C_CLDICE, lw=2.0, label="clDice")
        a1.plot(hist["epoch"], hist["val_score"], color=C_SCORE, lw=2.3, label="skóre")
        a1.axvline(be, color=SECOND, ls=":", lw=1.1, label=f"nejlepší epocha ({be})")
        a1.set_title("Validační metriky"); a1.set_xlabel(LBL_EPOCH); a1.set_ylim(0, 1)
        a1.yaxis.set_major_locator(MultipleLocator(0.2)); a1.set_xlim(left=hist["epoch"].min())
        _tidy(a1); a1.legend(frameon=False, ncol=2)

    if test is not None:
        m = micro_metrics(test)
        a2 = fig.add_subplot(gs[1, 0:2])
        names = ["Precision", "Recall", "F1", "IoU", "Dice", "clDice"]
        keys = ["precision", "recall", "f1", "iou", "dice", "cldice"]
        colors = [SECOND, SECOND, C_DICE, C_IOU, C_DICE, C_CLDICE]
        vals = [m[k] for k in keys]
        bars = a2.bar(names, vals, color=colors, width=0.66, edgecolor="white", linewidth=0.5)
        a2.set_ylim(0, 1); a2.set_title("Test na povodí Odra"); a2.grid(axis="x", visible=False); _tidy(a2)
        for b, v in zip(bars, vals):
            a2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)

        pos = positives(test)
        cats = present_categories(pos) if "category" in pos.columns else []
        a3 = fig.add_subplot(gs[1, 2])
        if cats:
            data = [pos.loc[pos["category"] == c, "dice"].values for c in cats]
            styled_boxplot(a3, data, cats, C_DICE, "Dice dle kategorie")
        else:
            a3.axis("off")

    fig.suptitle(run["name"], fontsize=15, fontweight="semibold", color=INK, y=0.99)
    save(fig, out)


# ============================================================
# ARCHITECTURE COMPARISON FIGURES
# ============================================================

def fig_curves_comparison(runs, out):
    runs_h = [r for r in runs if r["hist"] is not None]
    if not runs_h:
        return
    panels = [("val_score", "Validační skóre"), ("val_dice", "Validační Dice"),
              ("val_cldice", "Validační clDice")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    for a, (col, ttl) in zip(ax, panels):
        for i, r in enumerate(runs_h):
            c = arch_color(r["arch"], i)
            a.plot(r["hist"]["epoch"], r["hist"][col], color=c, lw=2.2, label=r["name"])
        a.set_xlabel(LBL_EPOCH); a.set_ylabel("Hodnota"); a.set_ylim(0, 1)
        a.yaxis.set_major_locator(MultipleLocator(0.2)); a.set_title(ttl); _tidy(a)
    ax[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Porovnání architektur, průběh validačních metrik",
                 y=1.03, fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out)


def fig_test_metrics_comparison(runs, out):
    runs_t = [r for r in runs if r["test"] is not None]
    if not runs_t:
        return
    metrics = ["precision", "recall", "f1", "iou", "dice", "cldice"]
    labels = [METRIC_TITLE[m] for m in metrics]
    x = np.arange(len(metrics))
    n = len(runs_t)
    bw = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for i, r in enumerate(runs_t):
        m = micro_metrics(r["test"])
        vals = [m[k] for k in metrics]
        off = (i - n / 2 + 0.5) * bw
        c = arch_color(r["arch"], i)
        bars = ax.bar(x + off, vals, bw, label=r["name"], color=c, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Hodnota"); ax.grid(axis="x", visible=False); _tidy(ax)
    ax.set_title("Porovnání architektur na testovacím povodí Odra")
    ax.legend(frameon=False, ncol=min(n, 3), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, out)


def fig_test_by_category_comparison(runs, out):
    runs_t = [r for r in runs if r["test"] is not None]
    if not runs_t:
        return
    # union of categories present across runs, in canonical order
    cats = []
    for r in runs_t:
        pos = positives(r["test"])
        for c in present_categories(pos):
            if c not in cats:
                cats.append(c)
    cats = [c for c in CATEGORY_ORDER if c in cats]
    if not cats:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for a, metric in zip(ax, ["dice", "cldice"]):
        x = np.arange(len(cats))
        n = len(runs_t)
        bw = 0.8 / n
        for i, r in enumerate(runs_t):
            pos = positives(r["test"])
            means = [pos.loc[pos["category"] == c, metric].mean() if (pos["category"] == c).any() else 0.0
                     for c in cats]
            off = (i - n / 2 + 0.5) * bw
            c_col = arch_color(r["arch"], i)
            a.bar(x + off, means, bw, label=r["name"], color=c_col, edgecolor="white", linewidth=0.5)
        a.set_xticks(x); a.set_xticklabels([c for c in cats])
        a.set_ylim(0, 1); a.set_ylabel(f"{METRIC_TITLE[metric]} (průměr po patchích)")
        a.set_xlabel("Kategorie povodí"); a.grid(axis="x", visible=False); _tidy(a)
        a.set_title(METRIC_TITLE[metric])
    ax[0].legend(frameon=False, ncol=min(len(runs_t), 3), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Porovnání architektur podle kategorie povodí (Odra)",
                 y=1.03, fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out)


def build_summary_table(runs):
    rows = []
    for r in runs:
        row = {"Architektura": r["name"]}
        if r["hist"] is not None:
            be = best_epoch_row(r["hist"])
            row["Nejlepší epocha"] = int(be["epoch"])
            row["Val. skóre"] = round(float(be["val_score"]), 3)
        if r["test"] is not None:
            m = micro_metrics(r["test"])
            row["Precision"] = round(m["precision"], 3)
            row["Recall"] = round(m["recall"], 3)
            row["F1"] = round(m["f1"], 3)
            row["IoU"] = round(m["iou"], 3)
            row["Dice"] = round(m["dice"], 3)
            row["clDice"] = round(m["cldice"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_comparison_table(df, out):
    if df is None or len(df) == 0:
        return
    # highlight the best architecture per numeric column
    fig, ax = plt.subplots(figsize=(min(2.0 + 1.15 * len(df.columns), 13), 0.7 + 0.5 * len(df)))
    ax.axis("off")
    cols = list(df.columns)
    cell_text = df.astype(object).values.tolist()
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    numeric_cols = [c for c in cols if c not in ("Architektura", "Nejlepší epocha")]
    best_row = {}
    for c in numeric_cols:
        if c in df.columns:
            best_row[cols.index(c)] = int(df[c].astype(float).idxmax())

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r == 0:
            cell.set_facecolor("#13293D"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F7FAFA")
            if c == 0:
                cell.set_text_props(color=INK, fontweight="bold")
            if c in best_row and best_row[c] == (r - 1):
                cell.set_facecolor("#E5F3F3"); cell.set_text_props(color="#0E7C7B", fontweight="bold")
    ax.set_title("Porovnání architektur, souhrn (nejlepší hodnota zvýrazněna)",
                 fontsize=12.5, fontweight="semibold", color=INK, pad=14)
    save(fig, out)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    runs = []
    for run in RUNS:
        r = load_run(run)
        have = [k for k in ("hist", "test", "val") if r[k] is not None]
        print(f"[{r['name']}] found: {', '.join(have) if have else 'nothing'}")
        runs.append(r)

    print("\nPer-model figures...")
    for r in runs:
        slug = r["arch"]
        d = OUTPUT_DIR / slug
        fig_training_curves(r["hist"], d / "training_curves.png")
        fig_test_headline(r["test"], d / "test_headline.png")
        fig_test_by_category(r["test"], d / "test_by_category.png")
        fig_generalization(r["val"], r["test"], d / "generalization.png")
        fig_overview(r, d / "overview.png")

    print("\nArchitecture comparison...")
    comp = OUTPUT_DIR / "_comparison"
    fig_curves_comparison(runs, comp / "curves_comparison.png")
    fig_test_metrics_comparison(runs, comp / "test_metrics_comparison.png")
    fig_test_by_category_comparison(runs, comp / "test_by_category_comparison.png")
    summary = build_summary_table(runs)
    comp.mkdir(parents=True, exist_ok=True)
    summary.to_csv(comp / "comparison_summary.csv", index=False)
    print(f"  saved {(comp / 'comparison_summary.csv').relative_to(OUTPUT_DIR)}")
    fig_comparison_table(summary, comp / "comparison_table.png")

    print("\nDone.")


if __name__ == "__main__":
    main()