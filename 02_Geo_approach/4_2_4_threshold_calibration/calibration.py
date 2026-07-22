"""One-script decision-threshold calibration on the internal validation split.
Loads the final model, runs inference over val patches from both regions and
sweeps thresholds on the fly (no saved predictions). Outputs
best_threshold.json, table_threshold_sweep.csv and a PR figure."""

import warnings
warnings.filterwarnings("ignore")

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# ---- CONFIG -----------------------------------------------------------------
TRAIN_OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US")
META_WITH_SPLIT  = TRAIN_OUTPUT_DIR / "metadata_with_split.csv"

PATCHES_DIRS = {   # region value in metadata -> patches folder
    "PL":  Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_train"),
    "USA": Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_USA\patches"),
}

OUT_DIR   = Path(__file__).parent / "diagnostics_ch4"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15",
                  "canopy_height", "canopy_height_sd", "water"]
LABEL_CHANNEL = "label"
WATER_CHANNEL = "water"
SEGFORMER_BACKBONE = "mit_b2"
N_INPUT_CHANNELS = len(INPUT_CHANNELS)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
REFERENCE_THRESHOLD = 0.5

# ---- style ------------------------------------------------------------------
INK, SECOND, GRIDCOL, SPINE = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
C_DICE, C_F1 = "#0E7C7B", "#C2410C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11, "axes.labelcolor": INK,
    "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# ---- model (identical to the evaluation script) -----------------------------

def adapt_first_conv_segformer(model, n_input_channels):
    encoder = model.encoder
    first_conv = encoder.patch_embed1.proj
    old_weight = first_conv.weight.data
    out_ch, _, kh, kw = old_weight.shape
    new_weight = old_weight.repeat(1, (n_input_channels // 3) + 1, 1, 1)
    new_weight = new_weight[:, :n_input_channels, :, :]
    new_weight = new_weight / (n_input_channels / 3)
    new_conv = nn.Conv2d(n_input_channels, out_ch, kernel_size=(kh, kw),
                         stride=first_conv.stride, padding=first_conv.padding,
                         bias=first_conv.bias is not None)
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()
    encoder.patch_embed1.proj = new_conv
    return model


_KEY_REMAP_RULES = (
    (re.compile(r"(decoder\.blocks\.\d+\.conv2)\.0\.(\d+)\.(.+)"), r"\1.\2.\3"),
    (re.compile(r"(decoder\.block2)\.0\.(.+)"), r"\1.\2"),
)


def _remap_key(key):
    for pattern, repl in _KEY_REMAP_RULES:
        if pattern.match(key):
            return pattern.sub(repl, key)
    return key


def load_state_dict_compat(model, state_dict):
    model_keys = set(model.state_dict().keys())
    if all(k in model_keys for k in state_dict):
        model.load_state_dict(state_dict)
        return
    remapped = {_remap_key(k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing: {missing} "
                           f"Unexpected: {unexpected}")


def build_model():
    model = smp.Segformer(encoder_name=SEGFORMER_BACKBONE,
                          encoder_weights=None,   # overwritten by checkpoint
                          in_channels=3, classes=1, activation=None)
    model = adapt_first_conv_segformer(model, N_INPUT_CHANNELS)
    ckpt = torch.load(TRAIN_OUTPUT_DIR / "best_model.pt", map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    load_state_dict_compat(model, ckpt)
    return model.to(DEVICE).eval()


# ---- normalization (identical to the evaluation script) ---------------------

def load_patch(row, norm_stats, patches_dir):
    channels = dict(np.load(patches_dir / f"{row['patch_id']}.npz"))
    stack = []
    for ch in INPUT_CHANNELS:
        arr = np.nan_to_num(channels[ch].astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        if ch == "dsm":
            arr = arr - np.median(arr)
        elif ch == WATER_CHANNEL:
            pass
        else:
            s = norm_stats[ch]
            arr = (arr - s["mean"]) / (s["std"] + 1e-6)
        stack.append(arr)
    image = np.stack(stack, axis=0)
    label = np.nan_to_num(channels[LABEL_CHANNEL].astype(np.float32))
    return image, label


# ---- sweep ------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    norm_stats = json.loads((TRAIN_OUTPUT_DIR / "norm_stats.json").read_text())
    model = build_model()
    print(f"model loaded ({DEVICE})")

    df = pd.read_csv(META_WITH_SPLIT)
    val = df[df["split"] == "val"].reset_index(drop=True)
    print(f"validation patches: {len(val)} "
          f"({dict(val['region'].value_counts())})")

    edges = np.concatenate([[0.0], THRESHOLDS, [1.0 + 1e-6]])
    n_t = len(THRESHOLDS)
    tp = np.zeros(n_t); fp = np.zeros(n_t)
    fn = np.zeros(n_t); tn = np.zeros(n_t)
    n_used, n_missing = 0, 0

    for start in tqdm(range(0, len(val), BATCH_SIZE), desc="Calibrating"):
        batch_rows = val.iloc[start:start + BATCH_SIZE]
        images, labels = [], []
        for _, row in batch_rows.iterrows():
            pdir = PATCHES_DIRS.get(str(row["region"]))
            if pdir is None or not (pdir / f"{row['patch_id']}.npz").exists():
                n_missing += 1
                continue
            img, lab = load_patch(row, norm_stats, pdir)
            images.append(img)
            labels.append(lab)
        if not images:
            continue

        x = torch.from_numpy(np.stack(images)).to(DEVICE)
        with torch.no_grad():
            proba = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()

        for pr, lab in zip(proba, labels):
            pos = lab.ravel() > 0.5
            prf = pr.ravel()
            h_pos, _ = np.histogram(prf[pos], bins=edges)
            h_neg, _ = np.histogram(prf[~pos], bins=edges)
            above_pos = np.cumsum(h_pos[::-1])[::-1][1:]
            above_neg = np.cumsum(h_neg[::-1])[::-1][1:]
            tp += above_pos
            fp += above_neg
            fn += pos.sum() - above_pos
            tn += (~pos).sum() - above_neg
            n_used += 1

    if n_used == 0:
        raise RuntimeError("No patches processed.")
    print(f"patches used: {n_used} (missing: {n_missing})")

    eps = 1e-9
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    iou = tp / (tp + fp + fn + eps)
    best_i = int(np.argmax(f1))
    ref_i = int(np.argmin(np.abs(THRESHOLDS - REFERENCE_THRESHOLD)))

    pd.DataFrame({"threshold": THRESHOLDS,
                  "precision": np.round(p, 4), "recall": np.round(r, 4),
                  "f1": np.round(f1, 4), "iou": np.round(iou, 4)}
                 ).to_csv(OUT_DIR / "table_threshold_sweep.csv", index=False)

    best = {"best_threshold": float(THRESHOLDS[best_i]),
            "criterion": "micro F1, internal validation split",
            "f1": float(f1[best_i]), "precision": float(p[best_i]),
            "recall": float(r[best_i]), "iou": float(iou[best_i]),
            "reference_threshold": REFERENCE_THRESHOLD,
            "f1_at_reference": float(f1[ref_i]),
            "n_patches": n_used}
    (OUT_DIR / "best_threshold.json").write_text(json.dumps(best, indent=2))
    print(f"best threshold {best['best_threshold']:.2f} "
          f"(F1 {best['f1']:.4f} vs F1@{REFERENCE_THRESHOLD} "
          f"{best['f1_at_reference']:.4f})")

    # PR figure
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(r, p, color=C_DICE, lw=2.2, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgecolor=C_DICE)
    for i, label, color, xytext in (
        (ref_i, f"práh {REFERENCE_THRESHOLD}", SECOND, (0, -14)),
        (best_i, f"práh {THRESHOLDS[best_i]:.2f} (opt.)", C_F1, (0, 12)),
    ):
        ax.scatter([r[i]], [p[i]], s=60, zorder=5, color=color)
        ax.annotate(label, (r[i], p[i]), textcoords="offset points",
                    xytext=xytext, ha="center", fontsize=9.5, color=color)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    rpad = max(0.05, 0.15 * (r.max() - r.min()))
    ppad = max(0.05, 0.15 * (p.max() - p.min()))
    ax.set_xlim(max(0, r.min() - rpad), min(1, r.max() + rpad))
    ax.set_ylim(max(0, p.min() - ppad), min(1, p.max() + ppad))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="both", color=GRIDCOL, linewidth=0.8)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_threshold_pr.png")
    plt.close(fig)
    print("written:", OUT_DIR / "fig_threshold_pr.png")


if __name__ == "__main__":
    main()