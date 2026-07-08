"""
Channel domain diagnostics: training domain (PL+NL) vs US (Mississippi)
=======================================================================

Quantifies how far the US input distributions lie from the training domain,
channel by channel. This is the evidence layer for the transfer discussion in
the thesis: it shows WHICH channels are out of distribution before any
re-normalization, and by how much.

Method
------
- Random sample of patches per domain (MAX_PATCHES, seeded), random pixel
  subsample per patch (PIXELS_PER_PATCH) to bound memory.
- dsm is compared AFTER per-patch median subtraction, mirroring the training
  normalization (absolute elevation never reaches the network).
- water is a binary mask; compared via the water-pixel fraction.
- Other channels are compared on RAW values, because the z-scoring applied at
  inference uses TRAINING statistics; the raw-distribution gap is exactly what
  the network experiences as input shift.

Shift measures per channel
--------------------------
- standardized mean difference d = (mean_US - mean_PL) / std_PL
- Wasserstein-1 distance (from quantiles) in raw units and in PL sigma units

Outputs (OUTPUT_DIR):
    fig_channel_distributions.png   overlaid densities per channel (thesis style)
    fig_channel_shift.png           W1 / sigma_PL bar chart per channel
    table_channel_stats.csv         per-domain stats + shift measures

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
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIG
# ============================================================

DOMAIN_A = {
    "name": "trénink (PL + NL)",
    "patches_dir": Path(r"D:\...\_FINAL_EVAL\patches\patches_PL_train\patches"),
    "metadata_csv": Path(r"D:\...\_FINAL_EVAL\patches\patches_PL_train\patches_metadata.csv"),
}
DOMAIN_B = {
    "name": "US (Mississippi)",
    "patches_dir": Path(r"D:\...\patches_US\patches"),
    "metadata_csv": Path(r"D:\...\patches_US\patches_metadata.csv"),
}

OUTPUT_DIR = Path(r"D:\...\domain_diagnostics")

INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15",
                  "canopy_height", "canopy_height_sd", "water"]
WATER_CHANNEL = "water"

MAX_PATCHES = 300          # per domain
PIXELS_PER_PATCH = 4096    # random pixel subsample per patch
SEED = 42

# ============================================================
# STYLE (thesis palette, consistent with 09_make_figures.py)
# ============================================================

INK, SECOND, GRIDCOL, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_A, C_B = "#0E7C7B", "#C2410C"   # domain A teal, domain B warm

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlepad": 8,
    "axes.labelsize": 10.5, "axes.labelcolor": INK,
    "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "legend.fontsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
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
# SAMPLING
# ============================================================

def sample_domain(domain, rng):
    """Collect a pixel sample per channel for one domain."""
    meta = pd.read_csv(domain["metadata_csv"])
    ids = meta["patch_id"].astype(str).tolist()
    rng.shuffle(ids)
    ids = ids[:MAX_PATCHES]

    values = {c: [] for c in INPUT_CHANNELS}
    n_used = 0
    for pid in ids:
        npz_path = Path(domain["patches_dir"]) / f"{pid}.npz"
        if not npz_path.exists():
            continue
        channels = dict(np.load(npz_path))
        if not all(c in channels for c in INPUT_CHANNELS):
            continue
        for c in INPUT_CHANNELS:
            arr = np.nan_to_num(channels[c].astype(np.float64)).ravel()
            if c == "dsm":
                arr = arr - np.median(arr)   # mirror training normalization
            k = min(PIXELS_PER_PATCH, arr.size)
            take = rng.choice(arr.size, size=k, replace=False)
            values[c].append(arr[take])
        n_used += 1

    if n_used == 0:
        raise RuntimeError(f"No usable patches in {domain['patches_dir']}")
    print(f"  {domain['name']}: {n_used} patches, "
          f"~{n_used * PIXELS_PER_PATCH:,} px per channel")
    return {c: np.concatenate(v) for c, v in values.items()}


# ============================================================
# STATS
# ============================================================

def wasserstein1(a, b, n_q=200):
    """W1 distance approximated from matched quantiles."""
    qs = np.linspace(0.005, 0.995, n_q)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def channel_stats(a, b):
    """Per-domain stats and shift measures for one channel."""
    def base(x):
        return {"mean": float(x.mean()), "std": float(x.std()),
                "median": float(np.median(x)),
                "p2": float(np.percentile(x, 2)),
                "p98": float(np.percentile(x, 98))}
    sa, sb = base(a), base(b)
    std_a = sa["std"] if sa["std"] > 1e-9 else 1e-9
    d = (sb["mean"] - sa["mean"]) / std_a
    w1 = wasserstein1(a, b)
    return sa, sb, d, w1, w1 / std_a


# ============================================================
# FIGURES
# ============================================================

def fig_distributions(data_a, data_b, name_a, name_b, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2))
    axes = axes.ravel()

    for ax, ch in zip(axes, INPUT_CHANNELS):
        a, b = data_a[ch], data_b[ch]
        if ch == WATER_CHANNEL:
            fa, fb = float(a.mean()), float(b.mean())
            bars = ax.bar([0, 1], [fa, fb], width=0.55,
                          color=[C_A, C_B], edgecolor="white")
            for bar, v in zip(bars, (fa, fb)):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["trénink", "US"])
            ax.set_ylabel("Podíl vodních pixelů")
            ax.set_title(ch)
            ax.grid(axis="x", visible=False)
            _tidy(ax)
            continue

        lo = min(np.percentile(a, 0.5), np.percentile(b, 0.5))
        hi = max(np.percentile(a, 99.5), np.percentile(b, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins=bins, density=True, color=C_A, alpha=0.45,
                label=name_a, edgecolor="none")
        ax.hist(b, bins=bins, density=True, color=C_B, alpha=0.45,
                label=name_b, edgecolor="none")
        ax.set_title(ch)
        ax.set_ylabel("Hustota")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", visible=False)
        _tidy(ax)

    # last panel: legend only
    axes[-1].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_A, alpha=0.6),
               plt.Rectangle((0, 0), 1, 1, color=C_B, alpha=0.6)]
    axes[-1].legend(handles, [name_a, name_b], loc="center", frameon=False,
                    fontsize=11)

    fig.suptitle("Rozdělení vstupních kanálů, trénovací doména vs US",
                 y=1.0, fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_shift(table, out_path):
    sub = table[table["kanál"] != WATER_CHANNEL]
    order = sub.sort_values("W1_v_sigma_trénink")
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.barh(order["kanál"], order["W1_v_sigma_trénink"],
                   color=C_B, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, order["W1_v_sigma_trénink"]):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9.5, color=INK)
    ax.set_xlabel("Posun rozdělení W1 (v jednotkách σ trénovací domény)")
    ax.set_title("Míra posunu vstupních kanálů mezi doménami")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRIDCOL, linewidth=0.8)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Sampling domains...")
    data_a = sample_domain(DOMAIN_A, rng)
    data_b = sample_domain(DOMAIN_B, rng)

    rows = []
    for ch in INPUT_CHANNELS:
        sa, sb, d, w1, w1s = channel_stats(data_a[ch], data_b[ch])
        rows.append({
            "kanál": ch,
            "trénink_mean": round(sa["mean"], 4), "trénink_std": round(sa["std"], 4),
            "trénink_median": round(sa["median"], 4),
            "trénink_p2": round(sa["p2"], 4), "trénink_p98": round(sa["p98"], 4),
            "US_mean": round(sb["mean"], 4), "US_std": round(sb["std"], 4),
            "US_median": round(sb["median"], 4),
            "US_p2": round(sb["p2"], 4), "US_p98": round(sb["p98"], 4),
            "std_mean_diff_d": round(d, 3),
            "W1": round(w1, 4),
            "W1_v_sigma_trénink": round(w1s, 3),
        })
    table = pd.DataFrame(rows)
    table_path = OUTPUT_DIR / "table_channel_stats.csv"
    table.to_csv(table_path, index=False)
    print(f"\n  saved {table_path.name}")
    print(table[["kanál", "std_mean_diff_d", "W1_v_sigma_trénink"]]
          .to_string(index=False))

    print("\nFigures...")
    fig_distributions(data_a, data_b, DOMAIN_A["name"], DOMAIN_B["name"],
                      OUTPUT_DIR / "fig_channel_distributions.png")
    fig_shift(table, OUTPUT_DIR / "fig_channel_shift.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
