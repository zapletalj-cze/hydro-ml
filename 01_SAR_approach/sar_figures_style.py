"""
Thesis-style figures for the Sentinel-1 / XGBoost script
========================================================

Drop-in replacements for the two matplotlib blocks (sections 6.1 and 6.3).
Same palette as the thesis figures (09_make_figures.py), no titles.

Usage inside the SAR script:

    from sar_figures_style import plot_pr_curve, plot_feature_importance

    # section 6.1 (replace the fig/ax block):
    plot_pr_curve(recall, precision, best_idx, best_thr, best_f1,
                  OUT_DIR / 'pr_curve.png')

    # section 6.3 (replace the fig/ax block):
    plot_feature_importance(BAND_NAMES, importance,
                            OUT_DIR / 'feature_importance.png')

Or paste the functions directly into the script.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- thesis palette ----
INK, SECOND, GRIDCOL, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_MAIN, C_MARK = "#0E7C7B", "#C2410C"     # teal curve/bars, warm marker

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11, "axes.labelcolor": INK,
    "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
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


def plot_pr_curve(recall, precision, best_idx, best_thr, best_f1, out_path):
    """Precision-recall curve with the F1-optimal threshold marked. No title."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(recall, precision, color=C_MAIN, lw=2.2)
    ax.scatter([recall[best_idx]], [precision[best_idx]], s=65, zorder=5,
               color=C_MARK)
    ax.annotate(f"práh {best_thr:.2f} (F1 {best_f1:.2f})",
                (recall[best_idx], precision[best_idx]),
                textcoords="offset points", xytext=(10, -14),
                fontsize=9.5, color=C_MARK)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="x", color=GRIDCOL, linewidth=0.8)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(band_names, importance, out_path, top_n=None):
    """Horizontal feature-importance bars, most important on top. No title."""
    importance = np.asarray(importance, dtype=float)
    order = np.argsort(importance)          # ascending -> top ends up on top
    if top_n is not None:
        order = order[-top_n:]
    names = [band_names[i] for i in order]
    vals = importance[order]

    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(names) + 1.2))
    bars = ax.barh(names, vals, color=C_MAIN, alpha=0.9, edgecolor="white")
    vmax = vals.max() if len(vals) else 1.0
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008 * vmax, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8.5, color=SECOND)
    ax.set_xlabel("Významnost příznaku (gain)")
    ax.set_xlim(0, vmax * 1.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRIDCOL, linewidth=0.8)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
