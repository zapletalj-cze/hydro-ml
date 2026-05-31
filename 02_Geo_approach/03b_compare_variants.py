"""
Compare Variants - Ablation Summary
====================================

Reads metrics_summary.json from N different training runs (ablation variants)
and produces:
    - A comparison CSV table with key metrics side by side
    - Training curve overlays (val_score, val_dice, val_cldice per variant)
    - Final-metric bar chart for the test set
    - A breakdown CSV per category and per region

To use:
    1. Edit VARIANTS dict below: variant name -> path to OUTPUT_DIR
    2. Run: python compare_variants.py

Author:   Jakub Zapletal
Date:     2026-05-21
Version:  0.1
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


# ============================================================
# CONFIG - edit these
# ============================================================

# Map variant name -> OUTPUT_DIR of the training run.
# These dirs must each contain a metrics_summary.json.
VARIANTS = {
    "v1 (DSM)":          Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v1_dsm_only"),
    "v2 (DSM+TPI)":      Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v2_dsm_tpi"),
    "v3 (DSM+TPI+aux)":  Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight"),
}

# Output dir for the comparison artifacts
OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_comparison")


# ============================================================
# LOAD
# ============================================================

def load_summary(variant_name, output_dir):
    """Load metrics_summary.json from a variant's output directory."""
    summary_path = output_dir / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist — has this variant finished training?")
    with open(summary_path) as f:
        return json.load(f)


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_overall_table(summaries):
    """One row per variant, columns = key metrics (test set)."""
    rows = []
    for name, summary in summaries.items():
        ri = summary["run_info"]
        tm = summary["test_metrics"]["overall"]
        rows.append({
            "variant":             name,
            "n_channels":          ri["n_channels"],
            "input_channels":      ", ".join(ri["input_channels"]),
            "epochs_run":          ri["n_epochs_run"],
            "best_epoch":          ri["best_epoch"],
            "best_val_score":      ri["best_val_score"],
            "test_mean_dice":      tm["mean_per_patch"]["dice"],
            "test_mean_iou":       tm["mean_per_patch"]["iou"],
            "test_mean_cldice":    tm["mean_per_patch"]["cldice"],
            "test_mean_f1":        tm["mean_per_patch"]["f1"],
            "test_mean_precision": tm["mean_per_patch"]["precision"],
            "test_mean_recall":    tm["mean_per_patch"]["recall"],
            "test_micro_dice":     tm["micro_averaged"]["dice"],
            "test_micro_iou":      tm["micro_averaged"]["iou"],
            "test_micro_f1":       tm["micro_averaged"]["f1"],
            "test_micro_precision":tm["micro_averaged"]["precision"],
            "test_micro_recall":   tm["micro_averaged"]["recall"],
            "tp": tm["confusion_matrix"]["tp"],
            "fp": tm["confusion_matrix"]["fp"],
            "fn": tm["confusion_matrix"]["fn"],
            "tn": tm["confusion_matrix"]["tn"],
            "total_hours":         ri["total_time_seconds"] / 3600,
        })
    return pd.DataFrame(rows)


def build_breakdown_table(summaries, group_kind):
    """
    Stack per-category or per-region metrics across variants.
    group_kind: 'by_category' or 'by_region'.
    """
    rows = []
    for variant_name, summary in summaries.items():
        groups = summary["test_metrics"][group_kind]
        for group_key, metrics in groups.items():
            row = {
                "variant":   variant_name,
                "group":     group_key,
                "n_patches": metrics["n_patches"],
            }
            for metric_name, value in metrics["mean_per_patch"].items():
                row[f"mean_{metric_name}"] = value
            for metric_name, value in metrics["micro_averaged"].items():
                row[f"micro_{metric_name}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves_overlay(summaries, output_path):
    """Overlay val_score / val_dice / val_cldice curves across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for variant_name, summary in summaries.items():
        h = summary["training_history"]
        epochs = h["epoch"]
        axes[0].plot(epochs, h["val_score"],  linewidth=2, label=variant_name)
        axes[1].plot(epochs, h["val_dice"],   linewidth=2, label=variant_name)
        axes[2].plot(epochs, h["val_cldice"], linewidth=2, label=variant_name)

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
    """Bar chart of final test metrics per variant."""
    metric_cols = ["test_mean_dice", "test_mean_iou", "test_mean_cldice",
                   "test_mean_f1", "test_mean_precision", "test_mean_recall"]
    metric_labels = ["Dice", "IoU", "clDice", "F1", "Precision", "Recall"]

    n_variants = len(df_overall)
    n_metrics = len(metric_cols)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(df_overall.iterrows()):
        values = [row[c] for c in metric_cols]
        offset = (i - n_variants / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=row["variant"])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Test set — final metrics per variant (mean per-patch)")
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
    df_pos["category"] = df_pos["group"].str.split(" | ").str[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "mean_dice",   "Mean per-patch Dice"),
        (axes[1], "mean_cldice", "Mean per-patch clDice"),
    ]:
        pivot = df_pos.pivot(index="category", columns="variant", values=metric)
        pivot = pivot.reindex(["S", "M", "L"])      # consistent ordering
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_xlabel("Category (upstream area)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} — by category (positive patches)")
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

    print(f"Loading {len(VARIANTS)} variant summaries:")
    summaries = {}
    for name, output_dir in VARIANTS.items():
        print(f"  {name}: {output_dir}")
        summaries[name] = load_summary(name, output_dir)

    # Overall comparison table
    print("\nBuilding comparison table...")
    df_overall = build_overall_table(summaries)
    overall_csv = OUTPUT_DIR / "comparison_overall.csv"
    df_overall.to_csv(overall_csv, index=False)
    print(f"Saved {overall_csv}")

    # Per-category breakdown
    df_by_cat = build_breakdown_table(summaries, "by_category")
    by_cat_csv = OUTPUT_DIR / "comparison_by_category.csv"
    df_by_cat.to_csv(by_cat_csv, index=False)
    print(f"Saved {by_cat_csv}")

    # Per-region breakdown
    df_by_region = build_breakdown_table(summaries, "by_region")
    by_region_csv = OUTPUT_DIR / "comparison_by_region.csv"
    df_by_region.to_csv(by_region_csv, index=False)
    print(f"Saved {by_region_csv}")

    # Plots
    print("\nGenerating plots...")
    plot_training_curves_overlay(summaries, OUTPUT_DIR / "comparison_training_curves.png")
    plot_final_metrics_bar(df_overall, OUTPUT_DIR / "comparison_final_metrics.png")
    plot_breakdown_by_category(df_by_cat, OUTPUT_DIR / "comparison_by_category.png")

    # Console summary
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON (test set, mean per-patch)")
    print("=" * 70)
    cols = ["variant", "n_channels", "best_epoch",
            "test_mean_dice", "test_mean_iou", "test_mean_cldice", "test_mean_f1"]
    print(df_overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Improvements relative to first variant (assumed baseline)
    if len(df_overall) >= 2:
        baseline = df_overall.iloc[0]
        print("\n" + "=" * 70)
        print(f"IMPROVEMENT vs baseline '{baseline['variant']}'")
        print("=" * 70)
        for i in range(1, len(df_overall)):
            row = df_overall.iloc[i]
            delta_dice   = (row["test_mean_dice"]   - baseline["test_mean_dice"])   / max(baseline["test_mean_dice"],   1e-6) * 100
            delta_cldice = (row["test_mean_cldice"] - baseline["test_mean_cldice"]) / max(baseline["test_mean_cldice"], 1e-6) * 100
            delta_iou    = (row["test_mean_iou"]    - baseline["test_mean_iou"])    / max(baseline["test_mean_iou"],    1e-6) * 100
            print(f"  {row['variant']:30s}  Dice {delta_dice:+6.1f}%   clDice {delta_cldice:+6.1f}%   IoU {delta_iou:+6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
