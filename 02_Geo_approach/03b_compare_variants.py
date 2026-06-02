"""
Compare Variants — Ablation Summary  (CSV-based, no metrics_summary.json)
==========================================================================

Reads training artifacts directly from each variant's OUTPUT_DIR. Works with
the current train_segformer.py output (which does not produce
metrics_summary.json).

Required files per variant directory:
    - training_history.csv
    - test_results_per_patch.csv
    - val_results_per_patch.csv

Produces in OUTPUT_DIR:
    - comparison_overall.csv          (one row per variant, key test metrics)
    - comparison_by_category.csv      (per S/M/L × patch_type breakdown)
    - comparison_by_region.csv        (per PL/NL × patch_type breakdown)
    - comparison_training_curves.png  (val_score / val_dice / val_cldice overlay)
    - comparison_final_metrics.png    (bar chart of final test metrics)
    - comparison_by_category.png      (per-category Dice/clDice bars)
    - Console table + relative improvements

To use:
    1. Edit VARIANTS dict below: variant name -> path to OUTPUT_DIR
    2. Run: python compare_variants.py

Author:   Jakub Zapletal
Date:     2026-05-21
Version:  0.2 (derives data from CSVs)
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
# CONFIG — edit these
# ============================================================

VARIANTS = {
    "v1 (DSM)":          Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v1_dsm_only"),
    "v2 (DSM+TPI)":      Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v2_dsm_tpi"),
    "v3 (DSM+TPI+aux)":  Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight"),
}

OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_comparison")


# ============================================================
# LOAD RUN ARTIFACTS
# ============================================================

def load_run(variant_name, run_dir):
    """
    Load training history + per-patch test/val metrics for one finished variant.
    Reconstructs best-epoch info from training_history.csv (val_score column).
    """
    history_path = run_dir / "training_history.csv"
    test_path    = run_dir / "test_results_per_patch.csv"
    val_path     = run_dir / "val_results_per_patch.csv"

    for p, desc in [(history_path, "training_history.csv"),
                    (test_path,    "test_results_per_patch.csv"),
                    (val_path,     "val_results_per_patch.csv")]:
        if not p.exists():
            raise FileNotFoundError(
                f"[{variant_name}] {desc} not found at {p}. "
                "Has this run finished?"
            )

    history = pd.read_csv(history_path)
    test    = pd.read_csv(test_path)
    val     = pd.read_csv(val_path)

    # Find best epoch from the history (criterion used during training: val_score = (val_dice+val_cldice)/2)
    if "val_score" in history.columns:
        best_idx = history["val_score"].idxmax()
        best_val_score = float(history.loc[best_idx, "val_score"])
    else:
        # Fallback if older history without val_score
        if "val_cldice" in history.columns:
            history["val_score"] = (history["val_dice"] + history["val_cldice"]) / 2
            best_idx = history["val_score"].idxmax()
            best_val_score = float(history.loc[best_idx, "val_score"])
        else:
            best_idx = history["val_dice"].idxmax()
            best_val_score = float("nan")

    return {
        "name":            variant_name,
        "run_dir":         run_dir,
        "history":         history,
        "test_results":    test,
        "val_results":     val,
        "n_epochs_run":    len(history),
        "best_epoch":      int(history.loc[best_idx, "epoch"]),
        "best_val_score":  best_val_score,
        "best_val_dice":   float(history.loc[best_idx, "val_dice"]),
        "best_val_cldice": float(history.loc[best_idx, "val_cldice"]) if "val_cldice" in history.columns else float("nan"),
        "best_val_loss":   float(history.loc[best_idx, "val_loss"])   if "val_loss"   in history.columns else float("nan"),
    }


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def aggregate_results(df):
    """
    Aggregate per-patch metrics dataframe. Returns dict with mean/std (per-patch),
    micro-averaged (from summed TP/FP/FN/TN), and the raw counts.
    """
    eps = 1e-6
    total_tp = int(df["tp"].sum())
    total_fp = int(df["fp"].sum())
    total_fn = int(df["fn"].sum())
    total_tn = int(df["tn"].sum())

    micro_precision = total_tp / (total_tp + total_fp + eps)
    micro_recall    = total_tp / (total_tp + total_fn + eps)
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    micro_iou       = total_tp / (total_tp + total_fp + total_fn + eps)
    micro_dice      = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)

    return {
        "n_patches": len(df),
        "mean": {
            "dice":      float(df["dice"].mean()),
            "iou":       float(df["iou"].mean()),
            "cldice":    float(df["cldice"].mean()),
            "f1":        float(df["f1"].mean()),
            "precision": float(df["precision"].mean()),
            "recall":    float(df["recall"].mean()),
        },
        "std": {
            "dice":      float(df["dice"].std()),
            "iou":       float(df["iou"].std()),
            "cldice":    float(df["cldice"].std()),
            "f1":        float(df["f1"].std()),
        },
        "micro": {
            "dice":      float(micro_dice),
            "iou":       float(micro_iou),
            "f1":        float(micro_f1),
            "precision": float(micro_precision),
            "recall":    float(micro_recall),
        },
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn},
    }


# ============================================================
# COMPARISON TABLES
# ============================================================

def build_overall_table(runs):
    """One row per variant, columns = key test metrics."""
    rows = []
    for name, run in runs.items():
        agg = aggregate_results(run["test_results"])
        rows.append({
            "variant":              name,
            "run_dir":              str(run["run_dir"]),
            "n_epochs_run":         run["n_epochs_run"],
            "best_epoch":           run["best_epoch"],
            "best_val_score":       run["best_val_score"],
            "best_val_dice":        run["best_val_dice"],
            "best_val_cldice":      run["best_val_cldice"],
            "best_val_loss":        run["best_val_loss"],
            # Test set, mean per patch
            "test_mean_dice":       agg["mean"]["dice"],
            "test_mean_iou":        agg["mean"]["iou"],
            "test_mean_cldice":     agg["mean"]["cldice"],
            "test_mean_f1":         agg["mean"]["f1"],
            "test_mean_precision":  agg["mean"]["precision"],
            "test_mean_recall":     agg["mean"]["recall"],
            # Per-patch std (selected)
            "test_std_dice":        agg["std"]["dice"],
            "test_std_cldice":      agg["std"]["cldice"],
            # Test set, micro-averaged (from summed counts)
            "test_micro_dice":      agg["micro"]["dice"],
            "test_micro_iou":       agg["micro"]["iou"],
            "test_micro_f1":        agg["micro"]["f1"],
            "test_micro_precision": agg["micro"]["precision"],
            "test_micro_recall":    agg["micro"]["recall"],
            # Confusion matrix totals
            "tp": agg["counts"]["tp"], "fp": agg["counts"]["fp"],
            "fn": agg["counts"]["fn"], "tn": agg["counts"]["tn"],
            "n_patches": agg["n_patches"],
        })
    return pd.DataFrame(rows)


def build_breakdown_table(runs, group_cols):
    """
    Per (category × patch_type) or (region × patch_type) breakdown across variants.
    Aggregates the per-patch metrics dataframe within each group.
    """
    rows = []
    for variant_name, run in runs.items():
        df = run["test_results"]
        for keys, sub in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            agg = aggregate_results(sub)
            rows.append({
                "variant":         variant_name,
                "group":           " | ".join(str(k) for k in keys),
                "n_patches":       agg["n_patches"],
                "mean_dice":       agg["mean"]["dice"],
                "mean_iou":        agg["mean"]["iou"],
                "mean_cldice":     agg["mean"]["cldice"],
                "mean_f1":         agg["mean"]["f1"],
                "mean_precision":  agg["mean"]["precision"],
                "mean_recall":     agg["mean"]["recall"],
                "micro_dice":      agg["micro"]["dice"],
                "micro_iou":       agg["micro"]["iou"],
                "micro_f1":        agg["micro"]["f1"],
                "micro_precision": agg["micro"]["precision"],
                "micro_recall":    agg["micro"]["recall"],
            })
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves_overlay(runs, output_path):
    """Overlay val_score / val_dice / val_cldice curves across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for variant_name, run in runs.items():
        h = run["history"]
        if "val_score" in h.columns:
            axes[0].plot(h["epoch"], h["val_score"], linewidth=2, label=variant_name)
        axes[1].plot(h["epoch"], h["val_dice"], linewidth=2, label=variant_name)
        if "val_cldice" in h.columns:
            axes[2].plot(h["epoch"], h["val_cldice"], linewidth=2, label=variant_name)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Score")
    axes[0].set_title("Val Score (Dice + clDice)/2")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Dice")
    axes[1].set_title("Val Dice")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Val clDice")
    axes[2].set_title("Val clDice")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_final_metrics_bar(df_overall, output_path):
    """Bar chart of final test metrics per variant (mean per-patch)."""
    metric_cols   = ["test_mean_dice", "test_mean_iou", "test_mean_cldice",
                     "test_mean_f1", "test_mean_precision", "test_mean_recall"]
    metric_labels = ["Dice", "IoU", "clDice", "F1", "Precision", "Recall"]

    n_variants = len(df_overall)
    n_metrics  = len(metric_cols)
    x          = np.arange(n_metrics)
    bar_width  = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(df_overall.iterrows()):
        values = [row[c] for c in metric_cols]
        offset = (i - n_variants / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=row["variant"])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Test set, final metrics per variant (mean per-patch)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_breakdown_by_category(df_breakdown, output_path):
    """Per-category Dice/clDice bars, positive patches only."""
    df_pos = df_breakdown[df_breakdown["group"].str.endswith("positive")].copy()
    if len(df_pos) == 0:
        print(f"No 'positive' patch_type rows in breakdown, skipping {output_path}")
        return

    df_pos["category"] = df_pos["group"].str.split(" | ").str[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
        (axes[0], "mean_dice",   "Mean per-patch Dice"),
        (axes[1], "mean_cldice", "Mean per-patch clDice"),
    ]:
        pivot = df_pos.pivot(index="category", columns="variant", values=metric)
        existing_cats = [c for c in ["S", "M", "L"] if c in pivot.index]
        if existing_cats:
            pivot = pivot.reindex(existing_cats)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_xlabel("Category (upstream area)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}, by category (positive patches)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(VARIANTS)} variant runs:")
    runs = {}
    for name, run_dir in VARIANTS.items():
        print(f"  {name}: {run_dir}")
        runs[name] = load_run(name, run_dir)

    # Tables
    print("\nBuilding comparison tables...")

    df_overall = build_overall_table(runs)
    overall_csv = OUTPUT_DIR / "comparison_overall.csv"
    df_overall.to_csv(overall_csv, index=False)
    print(f"Saved {overall_csv}")

    df_by_cat = build_breakdown_table(runs, ["category", "patch_type"])
    by_cat_csv = OUTPUT_DIR / "comparison_by_category.csv"
    df_by_cat.to_csv(by_cat_csv, index=False)
    print(f"Saved {by_cat_csv}")

    df_by_region = build_breakdown_table(runs, ["region", "patch_type"])
    by_region_csv = OUTPUT_DIR / "comparison_by_region.csv"
    df_by_region.to_csv(by_region_csv, index=False)
    print(f"Saved {by_region_csv}")

    # Plots
    print("\nGenerating plots...")
    plot_training_curves_overlay(runs,       OUTPUT_DIR / "comparison_training_curves.png")
    plot_final_metrics_bar(df_overall,       OUTPUT_DIR / "comparison_final_metrics.png")
    plot_breakdown_by_category(df_by_cat,    OUTPUT_DIR / "comparison_by_category.png")

    # Console summary
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON (test set, mean per-patch)")
    print("=" * 70)
    cols = ["variant", "n_epochs_run", "best_epoch",
            "test_mean_dice", "test_mean_iou", "test_mean_cldice", "test_mean_f1"]
    print(df_overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("MICRO-AVERAGED METRICS (test set)")
    print("=" * 70)
    cols_micro = ["variant", "test_micro_dice", "test_micro_iou",
                  "test_micro_f1", "test_micro_precision", "test_micro_recall"]
    print(df_overall[cols_micro].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Improvements relative to the first variant (assumed baseline)
    if len(df_overall) >= 2:
        baseline = df_overall.iloc[0]
        print("\n" + "=" * 70)
        print(f"IMPROVEMENT vs baseline '{baseline['variant']}'  (relative %, mean per-patch)")
        print("=" * 70)
        for i in range(1, len(df_overall)):
            row = df_overall.iloc[i]
            d_dice   = (row["test_mean_dice"]   - baseline["test_mean_dice"])   / max(baseline["test_mean_dice"],   1e-6) * 100
            d_cldice = (row["test_mean_cldice"] - baseline["test_mean_cldice"]) / max(baseline["test_mean_cldice"], 1e-6) * 100
            d_iou    = (row["test_mean_iou"]    - baseline["test_mean_iou"])    / max(baseline["test_mean_iou"],    1e-6) * 100
            d_f1     = (row["test_mean_f1"]     - baseline["test_mean_f1"])     / max(baseline["test_mean_f1"],     1e-6) * 100
            print(f"  {row['variant']:30s}  Dice {d_dice:+6.1f}%   clDice {d_cldice:+6.1f}%   IoU {d_iou:+6.1f}%   F1 {d_f1:+6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
