"""
Levee Detection - Evaluation Script
===================================

Evaluates a trained model on a held-out patch set (basin B, or the US report
subset). Loads best_model.pt from the training run and scores every patch in
the given metadata CSV as a single set - no split, no training.

v0.3 changes (transfer-fair evaluation):
    - NORM_STATS_JSON: optional override of the normalization statistics.
      None  -> TRAIN_OUTPUT_DIR/norm_stats.json (training-domain stats, as before)
      Path  -> e.g. norm_stats_US.json computed on the US calibration subset;
               weights stay identical, only the input normalization adapts.
    - THRESHOLD: configurable decision threshold (was hardcoded 0.5).
    - VARIANT_TAG: appended to the output dir so variants never overwrite
      each other (e.g. "plstats_t050", "usstats_t050", "usstats_topt").
    - eval_config.json: provenance record of checkpoint, stats, threshold.

CONFIG must match the values used at training time (channels, architecture,
backbone), otherwise the saved weights will not load or the inputs will not
line up with what the model expects.

Author:   Jakub Zapletal
Version:  0.3
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ============================================================
# CONFIG
# ============================================================

# ------- Paths ----------------------------------------------
# Training run that produced best_model.pt (+ norm_stats.json)
TRAIN_OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v05_segformer")

# Patch set to evaluate (basin B, or US calib / US report subset)
TEST_PATCHES_DIR  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_test\patches")
TEST_METADATA_CSV = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_test\patches_metadata.csv")

# ------- Transfer-fair evaluation ----------------------------
# Normalization statistics. None -> TRAIN_OUTPUT_DIR/norm_stats.json.
# For the adapted-input US evaluation set this to the norm_stats_US.json
# written by 13_us_norm_stats_split.py (computed on the US CALIB subset).
NORM_STATS_JSON = None            # e.g. Path(r"D:\...\us_eval_prep\norm_stats_US.json")

# Decision threshold on the sigmoid probability (was hardcoded 0.5).
THRESHOLD = 0.5

# Name of the evaluated set and variant tag; together they define the output
# dir, so no two variants can overwrite each other.
SPLIT_NAME  = "basin_B"           # e.g. "us_calib", "us_report"
VARIANT_TAG = "plstats_t050"      # e.g. "usstats_t050", "usstats_topt"

EVAL_OUTPUT_DIR = TRAIN_OUTPUT_DIR / f"eval_{SPLIT_NAME}_{VARIANT_TAG}"


# ------- Reproducibility ------------------------------------
SEED = 42

# ------- Hardware -------------------------------------------
NUM_WORKERS = 16

# ------- Channels (must match training) ----------------------
INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd", "water"]
LABEL_CHANNEL  = "label"
WATER_CHANNEL  = "water"

# ------- Inference / metrics --------------------------------
BATCH_SIZE       = 16
MIXED_PRECISION  = True
CLDICE_ITER      = 5
SAVE_PREDICTIONS = True

# ------- Architecture (must match training) ------------------
ARCHITECTURE = "segformer"  # "resnet_unet", "segformer", "deeplabv3plus"

RESNET_BACKBONE     = "resnet34"
SEGFORMER_BACKBONE  = "mit_b2"
DEEPLAB_BACKBONE    = "resnet50"

# weights are loaded from the checkpoint, so encoder pretraining is irrelevant here
RESNET_ENCODER_WEIGHTS    = None
DEEPLAB_ENCODER_WEIGHTS   = None
SEGFORMER_ENCODER_WEIGHTS = None

# Derived (do not edit)
N_INPUT_CHANNELS = len(INPUT_CHANNELS)
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval_log.txt"

    log = logging.getLogger()
    for h in list(log.handlers):
        log.removeHandler(h)

    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    log.info("=" * 70)
    log.info("Evaluation run started")
    log.info(f"Train output dir: {TRAIN_OUTPUT_DIR}")
    log.info(f"Test set:         {TEST_PATCHES_DIR}")
    log.info(f"Variant:          {SPLIT_NAME} / {VARIANT_TAG}")
    return log


# ============================================================
# DATASET
# ============================================================

class LeveeDataset(Dataset):
    """Loads .npz patches and applies the same normalization used at training."""

    def __init__(self, metadata_df, patches_root_dir, input_channels, norm_stats):
        self.metadata = metadata_df.reset_index(drop=True)
        self.patches_root_dir = patches_root_dir
        self.input_channels = input_channels
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.metadata)

    def _load_npz(self, row):
        npz_path = self.patches_root_dir / f"{row['patch_id']}.npz"
        return dict(np.load(npz_path))

    def _normalize(self, channels):
        out = {}
        for ch_name in self.input_channels:
            arr = channels[ch_name].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if ch_name == "dsm":
                arr = arr - np.median(arr)
            elif ch_name == WATER_CHANNEL:
                pass                            # binary mask, leave as 0/1
            else:
                stats = self.norm_stats[ch_name]
                arr = (arr - stats["mean"]) / (stats["std"] + 1e-6)
            out[ch_name] = arr
        return out

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        channels = self._load_npz(row)
        channels_norm = self._normalize(channels)

        image = np.stack([channels_norm[c] for c in self.input_channels], axis=0)
        label = channels[LABEL_CHANNEL].astype(np.float32)[np.newaxis, ...]

        cat = row["category"]
        if pd.isna(cat):                        # negatives have no category
            cat = row["patch_type"]

        return {
            "image":    torch.from_numpy(image),
            "label":    torch.from_numpy(label),
            "patch_id": row["patch_id"],
            "category": str(cat),
        }


# ============================================================
# METRIC HELPER (soft skeleton for clDice)
# ============================================================

def soft_skeleton(x, n_iter):
    def soft_erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    def soft_dilate(img):
        return F.max_pool2d(img, (3, 3), stride=1, padding=1)

    def soft_open(img):
        return soft_dilate(soft_erode(img))

    skel = F.relu(x - soft_open(x))
    img = x
    for _ in range(n_iter):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img)) * (1.0 - skel)
    return skel


# ============================================================
# MODEL
# ============================================================

def adapt_first_conv_for_extra_channels(model, n_input_channels):
    encoder = model.encoder
    if not hasattr(encoder, "conv1"):
        raise RuntimeError("Encoder has no conv1 - unknown structure")

    first_conv = encoder.conv1
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

    encoder.conv1 = new_conv
    return model


def adapt_first_conv_segformer(model, n_input_channels):
    encoder = model.encoder
    if not hasattr(encoder, "patch_embed1"):
        raise RuntimeError("SegFormer encoder has no patch_embed1 - check smp version")

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


def build_model():
    """Rebuild the architecture (channel-adapted) so the checkpoint weights load into it."""
    if ARCHITECTURE == "resnet_unet":
        model = smp.Unet(encoder_name=RESNET_BACKBONE, encoder_weights=RESNET_ENCODER_WEIGHTS,
                         in_channels=3, classes=1, activation=None)
        model = adapt_first_conv_for_extra_channels(model, N_INPUT_CHANNELS)

    elif ARCHITECTURE == "segformer":
        model = smp.Segformer(encoder_name=SEGFORMER_BACKBONE, encoder_weights=SEGFORMER_ENCODER_WEIGHTS,
                              in_channels=3, classes=1, activation=None)
        model = adapt_first_conv_segformer(model, N_INPUT_CHANNELS)

    elif ARCHITECTURE == "deeplabv3plus":
        model = smp.DeepLabV3Plus(encoder_name=DEEPLAB_BACKBONE, encoder_weights=DEEPLAB_ENCODER_WEIGHTS,
                                  in_channels=3, classes=1, activation=None)
        model = adapt_first_conv_for_extra_channels(model, N_INPUT_CHANNELS)

    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

    model = model.to(DEVICE)
    return model


# smp introduced an extra Sequential nesting level in some decoder blocks
# between versions. Each rule strips that redundant ".0" so old checkpoints load.
_KEY_REMAP_RULES = (
    # Unet decoder Conv2dReLU: blocks.N.conv2.0.<i>.<rest> -> blocks.N.conv2.<i>.<rest>
    (re.compile(r"(decoder\.blocks\.\d+\.conv2)\.0\.(\d+)\.(.+)"), r"\1.\2.\3"),
    # DeepLabV3+ SeparableConv block: block2.0.<rest> -> block2.<rest>
    (re.compile(r"(decoder\.block2)\.0\.(.+)"), r"\1.\2"),
)


def _remap_state_dict_key(key):
    for pattern, repl in _KEY_REMAP_RULES:
        if pattern.match(key):
            return pattern.sub(repl, key)
    return key


def load_state_dict_compat(model, state_dict):
    """Load weights, bridging smp version differences in the decoder structure."""
    model_keys = set(model.state_dict().keys())
    if all(k in model_keys for k in state_dict):
        model.load_state_dict(state_dict)
        return

    remapped = {_remap_state_dict_key(k): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch after remap. Missing: {missing} Unexpected: {unexpected}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, df_test, norm_stats):
    """Score every patch in the test set; save per-patch predictions and metrics."""
    pred_dir = EVAL_OUTPUT_DIR / f"predictions_{SPLIT_NAME}"
    if SAVE_PREDICTIONS:
        pred_dir.mkdir(parents=True, exist_ok=True)

    ds = LeveeDataset(df_test, TEST_PATCHES_DIR, INPUT_CHANNELS, norm_stats)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {SPLIT_NAME} ({VARIANT_TAG})"):
            image = batch["image"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            with autocast(enabled=MIXED_PRECISION):
                pred = model(image)
                pred_prob = torch.sigmoid(pred)

            pred_bin = (pred_prob > THRESHOLD).float()

            for i in range(image.size(0)):
                p = pred_bin[i:i+1]
                l = label[i:i+1]

                tp = int((p * l).sum().item())
                fp = int((p * (1.0 - l)).sum().item())
                fn = int(((1.0 - p) * l).sum().item())
                tn = int(((1.0 - p) * (1.0 - l)).sum().item())

                precision = tp / (tp + fp + 1e-6)
                recall    = tp / (tp + fn + 1e-6)
                f1        = 2 * precision * recall / (precision + recall + 1e-6)
                iou       = tp / (tp + fp + fn + 1e-6)
                dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)

                skel_p = soft_skeleton(p, CLDICE_ITER)
                skel_l = soft_skeleton(l, CLDICE_ITER)
                tprec  = ((skel_p * l).sum() + 1e-6) / (skel_p.sum() + 1e-6)
                trec   = ((p * skel_l).sum() + 1e-6) / (skel_l.sum() + 1e-6)
                cldice = (2 * tprec * trec / (tprec + trec)).item()

                records.append({
                    "patch_id":   batch["patch_id"][i],
                    "category":   batch["category"][i],
                    "tp":         tp,
                    "fp":         fp,
                    "fn":         fn,
                    "tn":         tn,
                    "precision":  precision,
                    "recall":     recall,
                    "f1":         f1,
                    "dice":       dice,
                    "iou":        iou,
                    "cldice":     cldice,
                    "n_label_px": int(l.sum().item()),
                })

                if SAVE_PREDICTIONS:
                    pred_to_save = pred_prob[i, 0].cpu().numpy().astype(np.float16)
                    np.savez_compressed(pred_dir / f"{batch['patch_id'][i]}.npz", pred=pred_to_save)

    df_results = pd.DataFrame(records)
    df_results = df_results.merge(
        df_test[["patch_id", "patch_type"]],
        on="patch_id", how="left",
    )
    return df_results


def report_metrics(df_results, split_name):
    logging.info("=" * 70)
    logging.info(f"{split_name.upper()} SET METRICS  (threshold={THRESHOLD})")
    logging.info("=" * 70)

    df_results = df_results.copy()
    df_results["category"] = df_results["category"].fillna(df_results["patch_type"])

    logging.info(f"\n--- {split_name}: By category x patch_type (mean per-patch) ---")
    agg_cat = df_results.groupby(["category", "patch_type"])[
        ["dice", "iou", "cldice", "precision", "recall", "f1"]
    ].mean()
    for line in agg_cat.to_string(float_format=lambda x: f"{x:.4f}").splitlines():
        logging.info(line)

    logging.info(f"\n--- {split_name}: Overall ---")
    logging.info(f"  Mean per-patch Dice:   {df_results['dice'].mean():.4f} +/- {df_results['dice'].std():.4f}")
    logging.info(f"  Mean per-patch IoU:    {df_results['iou'].mean():.4f} +/- {df_results['iou'].std():.4f}")
    logging.info(f"  Mean per-patch clDice: {df_results['cldice'].mean():.4f} +/- {df_results['cldice'].std():.4f}")
    logging.info(f"  Mean per-patch F1:     {df_results['f1'].mean():.4f} +/- {df_results['f1'].std():.4f}")

    total_tp = int(df_results["tp"].sum())
    total_fp = int(df_results["fp"].sum())
    total_fn = int(df_results["fn"].sum())
    total_tn = int(df_results["tn"].sum())

    micro_precision = total_tp / (total_tp + total_fp + 1e-6)
    micro_recall    = total_tp / (total_tp + total_fn + 1e-6)
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)
    micro_iou       = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    micro_dice      = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)

    logging.info(f"  Confusion matrix (summed over patches):")
    logging.info(f"    TP = {total_tp:>12,}    FP = {total_fp:>12,}")
    logging.info(f"    FN = {total_fn:>12,}    TN = {total_tn:>12,}")
    logging.info(f"  Micro-averaged Precision: {micro_precision:.4f}")
    logging.info(f"  Micro-averaged Recall:    {micro_recall:.4f}")
    logging.info(f"  Micro-averaged F1:        {micro_f1:.4f}")
    logging.info(f"  Micro-averaged IoU:       {micro_iou:.4f}")
    logging.info(f"  Micro-averaged Dice:      {micro_dice:.4f}")

    return agg_cat


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EVAL_OUTPUT_DIR)

    log.info(f"Device: {DEVICE}")
    log.info(f"Architecture: {ARCHITECTURE}")
    log.info(f"Input channels ({N_INPUT_CHANNELS}): {INPUT_CHANNELS}")
    log.info(f"Threshold: {THRESHOLD}")

    # Norm stats: training-domain by default, or an explicit override
    norm_stats_path = Path(NORM_STATS_JSON) if NORM_STATS_JSON else (TRAIN_OUTPUT_DIR / "norm_stats.json")
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    log.info(f"Loaded norm_stats from {norm_stats_path}")

    # Test metadata (single evaluation set, no split)
    df_test = pd.read_csv(TEST_METADATA_CSV)
    log.info(f"Test patches: {len(df_test)} "
             f"({(df_test['patch_type'] == 'positive').sum()} pos, "
             f"{(df_test['patch_type'] == 'negative').sum()} neg)")

    # Build model and load the trained weights
    checkpoint_path = TRAIN_OUTPUT_DIR / "best_model.pt"
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    load_state_dict_compat(model, checkpoint["model_state"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}: "
             f"val_score={checkpoint.get('val_score', float('nan')):.4f}, "
             f"val_dice={checkpoint['val_dice']:.4f}, "
             f"val_cldice={checkpoint.get('val_cldice', float('nan')):.4f}")

    # Provenance record so every variant is reconstructible
    with open(EVAL_OUTPUT_DIR / "eval_config.json", "w") as f:
        json.dump({
            "split_name": SPLIT_NAME,
            "variant_tag": VARIANT_TAG,
            "threshold": THRESHOLD,
            "norm_stats": str(norm_stats_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "architecture": ARCHITECTURE,
            "test_metadata": str(TEST_METADATA_CSV),
        }, f, indent=2)

    # Evaluate
    results = evaluate(model, df_test, norm_stats)
    results.to_csv(EVAL_OUTPUT_DIR / f"{SPLIT_NAME}_results_per_patch.csv", index=False)

    agg_cat = report_metrics(results, SPLIT_NAME)
    agg_cat.to_csv(EVAL_OUTPUT_DIR / f"{SPLIT_NAME}_results_by_category.csv")

    if SAVE_PREDICTIONS:
        n_pred = len(list((EVAL_OUTPUT_DIR / f"predictions_{SPLIT_NAME}").glob("*.npz")))
        log.info(f"Saved predictions: {n_pred} files")

    log.info("Done.")


if __name__ == "__main__":
    main()
