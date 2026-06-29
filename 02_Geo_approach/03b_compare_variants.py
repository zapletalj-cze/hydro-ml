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
    "v1 (DSM)": Path(
        r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v1_dsm_only"
    ),
    "v2 (DSM+TPI)": Path(
        r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v2_dsm_tpi"
    ),
    "v3 (DSM+TPI+aux)": Path(
        r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3_dsm_tpi_canopyheight"
    ),
}

OUTPUT_DIR = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_comparison"
)

# Optional manual overrides of best epoch used for training-curve plotting.
# Useful when you want to lock a specific stopping point for a variant.
FORCED_BEST_EPOCH = {
    "v3 (DSM+TPI+aux)": 100,
}


# ============================================================
# LOAD RUN ARTIFACTS
# ============================================================


def load_run(variant_name, run_dir):
    """
    Load training history + per-patch test/val metrics for one finished variant.
    Reconstructs best-epoch info from training_history.csv (val_score column).
    """
    history_path = run_dir / "training_history.csv"
    test_path = run_dir / "test_results_per_patch.csv"
    val_path = run_dir / "val_results_per_patch.csv"

    for p, desc in [
        (history_path, "training_history.csv"),
        (test_path, "test_results_per_patch.csv"),
        (val_path, "val_results_per_patch.csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"[{variant_name}] {desc} not found at {p}. " "Has this run finished?"
            )

    history = pd.read_csv(history_path)
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)

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
        "name": variant_name,
        "run_dir": run_dir,
        "history": history,
        "test_results": test,
        "val_results": val,
        "n_epochs_run": len(history),
        "best_epoch": int(history.loc[best_idx, "epoch"]),
        "best_val_score": best_val_score,
        "best_val_dice": float(history.loc[best_idx, "val_dice"]),
        "best_val_cldice": (
            float(history.loc[best_idx, "val_cldice"])
            if "val_cldice" in history.columns
            else float("nan")
        ),
        "best_val_loss": (
            float(history.loc[best_idx, "val_loss"])
            if "val_loss" in history.columns
            else float("nan")
        ),
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
    micro_recall = total_tp / (total_tp + total_fn + eps)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    )
    micro_iou = total_tp / (total_tp + total_fp + total_fn + eps)
    micro_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)

    return {
        "n_patches": len(df),
        "mean": {
            "dice": float(df["dice"].mean()),
            "iou": float(df["iou"].mean()),
            "cldice": float(df["cldice"].mean()),
            "f1": float(df["f1"].mean()),
            "precision": float(df["precision"].mean()),
            "recall": float(df["recall"].mean()),
        },
        "std": {
            "dice": float(df["dice"].std()),
            "iou": float(df["iou"].std()),
            "cldice": float(df["cldice"].std()),
            "f1": float(df["f1"].std()),
        },
        "micro": {
            "dice": float(micro_dice),
            "iou": float(micro_iou),
            "f1": float(micro_f1),
            "precision": float(micro_precision),
            "recall": float(micro_recall),
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
        rows.append(
            {
                "variant": name,
                "run_dir": str(run["run_dir"]),
                "n_epochs_run": run["n_epochs_run"],
                "best_epoch": run["best_epoch"],
                "best_val_score": run["best_val_score"],
                "best_val_dice": run["best_val_dice"],
                "best_val_cldice": run["best_val_cldice"],
                "best_val_loss": run["best_val_loss"],
                # Test set, mean per patch
                "test_mean_dice": agg["mean"]["dice"],
                "test_mean_iou": agg["mean"]["iou"],
                "test_mean_cldice": agg["mean"]["cldice"],
                "test_mean_f1": agg["mean"]["f1"],
                "test_mean_precision": agg["mean"]["precision"],
                "test_mean_recall": agg["mean"]["recall"],
                # Per-patch std (selected)
                "test_std_dice": agg["std"]["dice"],
                "test_std_cldice": agg["std"]["cldice"],
                # Test set, micro-averaged (from summed counts)
                "test_micro_dice": agg["micro"]["dice"],
                "test_micro_iou": agg["micro"]["iou"],
                "test_micro_f1": agg["micro"]["f1"],
                "test_micro_precision": agg["micro"]["precision"],
                "test_micro_recall": agg["micro"]["recall"],
                # Confusion matrix totals
                "tp": agg["counts"]["tp"],
                "fp": agg["counts"]["fp"],
                "fn": agg["counts"]["fn"],
                "tn": agg["counts"]["tn"],
                "n_patches": agg["n_patches"],
            }
        )
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
            rows.append(
                {
                    "variant": variant_name,
                    "group": " | ".join(str(k) for k in keys),
                    "n_patches": agg["n_patches"],
                    "mean_dice": agg["mean"]["dice"],
                    "mean_iou": agg["mean"]["iou"],
                    "mean_cldice": agg["mean"]["cldice"],
                    "mean_f1": agg["mean"]["f1"],
                    "mean_precision": agg["mean"]["precision"],
                    "mean_recall": agg["mean"]["recall"],
                    "micro_dice": agg["micro"]["dice"],
                    "micro_iou": agg["micro"]["iou"],
                    "micro_f1": agg["micro"]["f1"],
                    "micro_precision": agg["micro"]["precision"],
                    "micro_recall": agg["micro"]["recall"],
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================


def plot_training_curves_overlay(runs, output_path):
    """Overlay val_* curves and optional dashed train_* curves across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    max_plot_epoch = 100

    for variant_name, run in runs.items():
        h = run["history"].copy()

        # Coerce to numeric to avoid category-like plotting and hidden trailing rows.
        h["epoch"] = pd.to_numeric(h["epoch"], errors="coerce")
        if "val_score" in h.columns:
            h["val_score"] = pd.to_numeric(h["val_score"], errors="coerce")
        if "val_dice" in h.columns:
            h["val_dice"] = pd.to_numeric(h["val_dice"], errors="coerce")
        if "val_cldice" in h.columns:
            h["val_cldice"] = pd.to_numeric(h["val_cldice"], errors="coerce")
        if "train_score" in h.columns:
            h["train_score"] = pd.to_numeric(h["train_score"], errors="coerce")
        if "train_dice" in h.columns:
            h["train_dice"] = pd.to_numeric(h["train_dice"], errors="coerce")
        if "train_cldice" in h.columns:
            h["train_cldice"] = pd.to_numeric(h["train_cldice"], errors="coerce")

        if "train_score" not in h.columns and {"train_dice", "train_cldice"}.issubset(h.columns):
            h["train_score"] = (h["train_dice"] + h["train_cldice"]) / 2.0

        h = h.dropna(subset=["epoch"]).sort_values("epoch")

        variant_best_epoch = run.get("best_epoch", max_plot_epoch)
        variant_best_epoch = int(FORCED_BEST_EPOCH.get(variant_name, variant_best_epoch))
        variant_max_epoch = min(max_plot_epoch, variant_best_epoch)

        h = h[h["epoch"] <= variant_max_epoch]
        if len(h) == 0:
            continue

        max_epoch = int(h["epoch"].max())
        print(
            f"  {variant_name}: plotting through epoch {max_epoch} "
            f"(best={variant_best_epoch}, cap={max_plot_epoch})"
        )

        if "val_score" in h.columns:
            hs = h.dropna(subset=["val_score"])
            if len(hs):
                line_val_score = axes[0].plot(
                    hs["epoch"], hs["val_score"], linewidth=2, label=f"{variant_name} val"
                )[0]
                if "train_score" in h.columns:
                    hst = h.dropna(subset=["train_score"])
                    if len(hst):
                        axes[0].plot(
                            hst["epoch"], hst["train_score"], linewidth=1.7, linestyle="--",
                            color=line_val_score.get_color(), alpha=0.95, label=f"{variant_name} train"
                        )

        hd = h.dropna(subset=["val_dice"])
        if len(hd):
            line_val_dice = axes[1].plot(
                hd["epoch"], hd["val_dice"], linewidth=2, label=f"{variant_name} val"
            )[0]
            if "train_dice" in h.columns:
                hdt = h.dropna(subset=["train_dice"])
                if len(hdt):
                    axes[1].plot(
                        hdt["epoch"], hdt["train_dice"], linewidth=1.7, linestyle="--",
                        color=line_val_dice.get_color(), alpha=0.95, label=f"{variant_name} train"
                    )

        if "val_cldice" in h.columns:
            hc = h.dropna(subset=["val_cldice"])
            if len(hc):
                line_val_cld = axes[2].plot(
                    hc["epoch"], hc["val_cldice"], linewidth=2, label=f"{variant_name} val"
                )[0]
                if "train_cldice" in h.columns:
                    hct = h.dropna(subset=["train_cldice"])
                    if len(hct):
                        axes[2].plot(
                            hct["epoch"], hct["train_cldice"], linewidth=1.7, linestyle="--",
                            color=line_val_cld.get_color(), alpha=0.95, label=f"{variant_name} train"
                        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Score")
    axes[0].set_title("Val Score (Dice + clDice)/2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Dice")
    axes[1].set_title("Val Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val clDice")
    axes[2].set_title("Val clDice")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(1, max_plot_epoch)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_final_metrics_bar(df_overall, output_path):
    """Bar chart of final test metrics per variant (mean per-patch)."""
    metric_cols = [
        "test_mean_dice",
        "test_mean_iou",
        "test_mean_cldice",
        "test_mean_f1",
        "test_mean_precision",
        "test_mean_recall",
    ]
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
    ax.set_title("Test set, final metrics per variant (mean per-patch)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_overall_summary_dashboard(df_overall, output_path):
    """Three-panel summary: mean metrics, micro metrics, and relative improvements."""
    if len(df_overall) == 0:
        print(f"No rows in overall table, skipping {output_path}")
        return

    variants = df_overall["variant"].tolist()
    n_variants = len(variants)

    mean_metrics = [
        ("test_mean_dice", "Dice"),
        ("test_mean_iou", "IoU"),
        ("test_mean_cldice", "clDice"),
        ("test_mean_f1", "F1"),
    ]
    micro_metrics = [
        ("test_micro_dice", "Dice"),
        ("test_micro_iou", "IoU"),
        ("test_micro_precision", "Precision"),
        ("test_micro_recall", "Recall"),
    ]

    mean_present = [(c, l) for c, l in mean_metrics if c in df_overall.columns]
    micro_present = [(c, l) for c, l in micro_metrics if c in df_overall.columns]
    if not mean_present or not micro_present:
        print(f"Missing required columns for summary dashboard, skipping {output_path}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Panel 1: mean per-patch metrics
    ax = axes[0]
    x = np.arange(len(mean_present))
    bw = 0.8 / max(n_variants, 1)
    palette = ["#1F4E79", "#0E7C7B", "#C2410C", "#7C3AED", "#64748B", "#2B6CB0"]
    for i, (_, row) in enumerate(df_overall.iterrows()):
        vals = [float(row[col]) for col, _ in mean_present]
        off = (i - n_variants / 2 + 0.5) * bw
        bars = ax.bar(
            x + off,
            vals,
            bw,
            label=row["variant"],
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in mean_present])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Mean per-patch")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: micro-averaged metrics
    ax = axes[1]
    x = np.arange(len(micro_present))
    for i, (_, row) in enumerate(df_overall.iterrows()):
        vals = [float(row[col]) for col, _ in micro_present]
        off = (i - n_variants / 2 + 0.5) * bw
        bars = ax.bar(
            x + off,
            vals,
            bw,
            label=row["variant"],
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in micro_present])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Micro-averaged")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: relative improvement vs baseline for key mean metrics
    ax = axes[2]
    if n_variants <= 1:
        ax.text(0.5, 0.5, "Need >=2 variants", ha="center", va="center")
        ax.set_axis_off()
    else:
        baseline = df_overall.iloc[0]
        key_metrics = [
            ("test_mean_dice", "Dice"),
            ("test_mean_cldice", "clDice"),
            ("test_mean_iou", "IoU"),
            ("test_mean_f1", "F1"),
        ]
        key_metrics = [(c, l) for c, l in key_metrics if c in df_overall.columns]
        labels = []
        values = []
        for i in range(1, len(df_overall)):
            row = df_overall.iloc[i]
            for col, lbl in key_metrics:
                base = float(baseline[col])
                val = float(row[col])
                rel = (val - base) / max(abs(base), 1e-6) * 100.0
                labels.append(f"{row['variant']} - {lbl}")
                values.append(rel)

        y = np.arange(len(labels))
        bar_colors = ["#0E7C7B" if v >= 0 else "#C2410C" for v in values]
        bars = ax.barh(y, values, color=bar_colors, alpha=0.9)
        ax.axvline(0, color="#64748B", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Relative improvement [%]")
        ax.set_title(f"Improvement vs baseline ({baseline['variant']})")
        ax.grid(True, axis="x", alpha=0.3)
        for b, v in zip(bars, values):
            x_txt = v + (1.0 if v >= 0 else -1.0)
            ha = "left" if v >= 0 else "right"
            ax.text(x_txt, b.get_y() + b.get_height() / 2, f"{v:+.1f}%", va="center", ha=ha, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=min(len(labels), 3), loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Model comparison summary", y=1.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_breakdown_by_category(df_breakdown, output_path):
    """Per-category Dice / clDice / Precision / Recall bars, positive patches only."""
    df_pos = df_breakdown[df_breakdown["group"].str.endswith("positive")].copy()
    if len(df_pos) == 0:
        print(f"No 'positive' patch_type rows in breakdown, skipping {output_path}")
        return

    df_pos["category"] = df_pos["group"].str.split(" | ").str[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric, title in [
        (axes[0, 0], "mean_dice", "Mean per-patch Dice"),
        (axes[0, 1], "mean_cldice", "Mean per-patch clDice"),
        (axes[1, 0], "mean_precision", "Mean per-patch Precision"),
        (axes[1, 1], "mean_recall", "Mean per-patch Recall"),
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
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_distributions_comparison(runs, output_path, max_points_per_variant=1500):
    """
    Four-panel diagnostic plot:
      [0,0] Boxplot of Dice + clDice distributions per variant (positive patches)
      [0,1] Heatmap of mean Dice per (variant x category)
      [1,0] Precision vs Recall scatter, color = variant
      [1,1] Dice vs n_label_px scatter (log x), color = variant

    Scatter panels are subsampled (max_points_per_variant) for clarity.
    """
    # Combine all variant test_results into one df with a variant column
    all_results = []
    for name, run in runs.items():
        df = run["test_results"].copy()
        df["variant"] = name
        all_results.append(df)
    combined = pd.concat(all_results, ignore_index=True)
    pos = combined[combined["patch_type"] == "positive"].copy()

    if len(pos) == 0:
        print(f"No positive patches found, skipping {output_path}")
        return

    variant_names = list(runs.keys())
    n_var = len(variant_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_var, 3)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Panel [0,0]: Boxplot Dice + clDice per variant ----
    ax = axes[0, 0]
    positions_dice = np.arange(n_var) * 3
    positions_cldice = positions_dice + 1

    data_dice = [pos[pos["variant"] == v]["dice"].values for v in variant_names]
    data_cldice = [pos[pos["variant"] == v]["cldice"].values for v in variant_names]

    bp_dice = ax.boxplot(
        data_dice,
        positions=positions_dice,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor="tab:blue", alpha=0.5),
        medianprops=dict(color="black"),
    )
    bp_cld = ax.boxplot(
        data_cldice,
        positions=positions_cldice,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor="tab:green", alpha=0.5),
        medianprops=dict(color="black"),
    )

    ax.set_xticks(positions_dice + 0.5)
    ax.set_xticklabels(variant_names, rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Distribution per variant (positive patches)")
    ax.legend(
        [bp_dice["boxes"][0], bp_cld["boxes"][0]], ["Dice", "clDice"], loc="lower right"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Panel [0,1]: Heatmap variant x category (mean Dice) ----
    ax = axes[0, 1]
    pivot = pos.pivot_table(
        index="variant", columns="category", values="dice", aggfunc="mean"
    )
    existing_cats = [c for c in ["S", "M", "L"] if c in pivot.columns]
    pivot = pivot.reindex(columns=existing_cats)
    pivot = pivot.reindex([v for v in variant_names if v in pivot.index])

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Dice (positive), variant x category")

    # Cell annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "black" if val > 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Panel [1,0]: Precision vs Recall scatter ----
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    for variant_name, color in zip(variant_names, colors):
        v_data = pos[pos["variant"] == variant_name]
        if len(v_data) > max_points_per_variant:
            v_data = v_data.sample(max_points_per_variant, random_state=42)
        ax.scatter(
            v_data["recall"],
            v_data["precision"],
            alpha=0.35,
            s=12,
            color=color,
            label=variant_name,
            edgecolors="none",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision vs Recall per patch (positive, subsampled to {max_points_per_variant}/variant)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # ---- Panel [1,1]: Dice vs n_label_px scatter ----
    ax = axes[1, 1]
    for variant_name, color in zip(variant_names, colors):
        v_data = pos[pos["variant"] == variant_name]
        if len(v_data) > max_points_per_variant:
            v_data = v_data.sample(max_points_per_variant, random_state=42)
        # Filter out zeros for log scale
        v_data = v_data[v_data["n_label_px"] > 0]
        ax.scatter(
            v_data["n_label_px"],
            v_data["dice"],
            alpha=0.35,
            s=12,
            color=color,
            label=variant_name,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of label pixels (log scale)")
    ax.set_ylabel("Dice")
    ax.set_title(f"Dice vs label size (subsampled to {max_points_per_variant}/variant)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

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
    plot_training_curves_overlay(runs, OUTPUT_DIR / "comparison_training_curves.png")
    plot_final_metrics_bar(df_overall, OUTPUT_DIR / "comparison_final_metrics.png")
    plot_overall_summary_dashboard(df_overall, OUTPUT_DIR / "comparison_summary_dashboard.png")
    plot_breakdown_by_category(df_by_cat, OUTPUT_DIR / "comparison_by_category.png")
    plot_distributions_comparison(runs, OUTPUT_DIR / "comparison_distributions.png")

    # Console summary
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON (test set, mean per-patch)")
    print("=" * 70)
    cols = [
        "variant",
        "n_epochs_run",
        "best_epoch",
        "test_mean_dice",
        "test_mean_iou",
        "test_mean_cldice",
        "test_mean_f1",
    ]
    print(df_overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("MICRO-AVERAGED METRICS (test set)")
    print("=" * 70)
    cols_micro = [
        "variant",
        "test_micro_dice",
        "test_micro_iou",
        "test_micro_f1",
        "test_micro_precision",
        "test_micro_recall",
    ]
    print(
        df_overall[cols_micro].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    # Improvements relative to the first variant (assumed baseline)
    if len(df_overall) >= 2:
        baseline = df_overall.iloc[0]
        print("\n" + "=" * 70)
        print(
            f"IMPROVEMENT vs baseline '{baseline['variant']}'  (relative %, mean per-patch)"
        )
        print("=" * 70)
        for i in range(1, len(df_overall)):
            row = df_overall.iloc[i]
            d_dice = (
                (row["test_mean_dice"] - baseline["test_mean_dice"])
                / max(baseline["test_mean_dice"], 1e-6)
                * 100
            )
            d_cldice = (
                (row["test_mean_cldice"] - baseline["test_mean_cldice"])
                / max(baseline["test_mean_cldice"], 1e-6)
                * 100
            )
            d_iou = (
                (row["test_mean_iou"] - baseline["test_mean_iou"])
                / max(baseline["test_mean_iou"], 1e-6)
                * 100
            )
            d_f1 = (
                (row["test_mean_f1"] - baseline["test_mean_f1"])
                / max(baseline["test_mean_f1"], 1e-6)
                * 100
            )
            print(
                f"  {row['variant']:30s}  Dice {d_dice:+6.1f}%   clDice {d_cldice:+6.1f}%   IoU {d_iou:+6.1f}%   F1 {d_f1:+6.1f}%"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
