"""
Levee Detection - Training Script (converted from notebook for performance)
============================================================================

Converted to a standalone Python script so DataLoader workers can use
multiprocessing without Windows/Jupyter deadlocks. With num_workers > 0 the
GPU utilisation goes from ~10-30% to ~80%+ on this workload.

Configuration knobs are constants at the top of the file (no CLI).
To run a different ablation variant, edit the CONFIG block:
    - ARCHITECTURE
    - INPUT_CHANNELS  (defines variant v1 / v2 / v3 / ...)
    - OUTPUT_DIR       (must be unique per variant - otherwise overwrites)

Pipeline (called from main()):
    1. load + split metadata
    2. compute or load per-channel normalization stats
    3. build dataset + dataloaders
    4. build model (architecture-aware)
    5. train()           - epoch loop with best-checkpoint logging
    6. evaluate_split()  - val + test with full metrics + saved predictions

Author:   Jakub Zapletal
Date:     2026-05-21
Version:  0.1
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")               # no display required when running headless
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


# ============================================================
# CONFIG  -  edit these for each ablation run
# ============================================================

# ------- Paths ----------------------------------------------
PATCHES_DIR_PL  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_PL\patches")
PATCHES_DIR_NL  = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_NL\patches")
METADATA_CSV_PL = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_PL\patches_metadata.csv")
METADATA_CSV_NL = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v01_NL\patches_metadata.csv")

# Output dir - MUST be unique per variant
OUTPUT_DIR = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\training_v03_segformer_v3")


# ------- Reproducibility ------------------------------------
SEED = 42


# ------- Hardware -------------------------------------------
# num_workers > 0 now works because we run as a script with __main__ guard.
# 8 is a safe default for most workstations; raise to 12-16 if CPU has cores.
NUM_WORKERS = 8


# ------- Channels (ablation variant!) ------------------------
# v1: INPUT_CHANNELS = ["dsm"]
# v2: INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15"]
# v3: INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd"]
INPUT_CHANNELS = ["dsm", "tpi_r5", "tpi_r10", "tpi_r15", "canopy_height", "canopy_height_sd"]
LABEL_CHANNEL  = "label"


# ------- Train/val/test split -------------------------------
SPLIT_BY   = "source_idx"
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15


# ------- Training hyperparameters ---------------------------
BATCH_SIZE      = 16
N_EPOCHS        = 100
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
GRAD_CLIP       = 1.0
MIXED_PRECISION = True


# ------- Loss function --------------------------------------
LOSS_DICE_WEIGHT = 0.5
CLDICE_ITER      = 5
BCE_POS_WEIGHT   = 20.0


# ------- Architecture ---------------------------------------
# One of: "resnet_unet" | "segformer" | "deeplabv3plus"
ARCHITECTURE = "segformer"

RESNET_BACKBONE     = "resnet34"
SEGFORMER_BACKBONE  = "mit_b2"
DEEPLAB_BACKBONE    = "resnet50"

RESNET_ENCODER_WEIGHTS    = "imagenet"
DEEPLAB_ENCODER_WEIGHTS   = "imagenet"
SEGFORMER_ENCODER_WEIGHTS = "imagenet"

DECODER_DROPOUT_P = 0.1


# ------- Augmentation ---------------------------------------
USE_FLIP_H = True
USE_FLIP_V = True
USE_ROT_90 = True


# ------- Evaluation ------------------------------------------
SAVE_PREDICTIONS = True   # save per-patch predictions for stacking


# Derived (do not edit)
N_INPUT_CHANNELS = len(INPUT_CHANNELS)
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir):
    """Configure logging to both console and a file inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.txt"

    log = logging.getLogger()
    # Clear any default handlers (e.g. from previous runs in same Python session)
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
    log.info("Training run started")
    log.info(f"Output dir: {output_dir}")
    return log


# ============================================================
# DATASET
# ============================================================

class LeveeDataset(Dataset):
    """
    Loads .npz patches and applies normalization + augmentation.
    Each sample is a dict with: image (C,H,W) tensor, label (1,H,W) tensor,
    patch_id (str), category (str).
    """

    def __init__(self, metadata_df, patches_root_dirs, input_channels, norm_stats, augment=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.patches_root_dirs = patches_root_dirs
        self.input_channels = input_channels
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def _load_npz(self, row):
        region = row["region"]
        npz_path = self.patches_root_dirs[region] / f"{row['patch_id']}.npz"
        return dict(np.load(npz_path))

    def _normalize(self, channels):
        out = {}
        for ch_name in self.input_channels:
            arr = channels[ch_name].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if ch_name == "dsm":
                arr = arr - np.median(arr)
            else:
                stats = self.norm_stats[ch_name]
                arr = (arr - stats["mean"]) / (stats["std"] + 1e-6)
            out[ch_name] = arr
        return out

    def _augment(self, image, label):
        if USE_ROT_90:
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                label = np.rot90(label, k=k, axes=(1, 2)).copy()
        if USE_FLIP_H and np.random.rand() < 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        if USE_FLIP_V and np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return image, label

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        channels = self._load_npz(row)
        channels_norm = self._normalize(channels)

        image = np.stack([channels_norm[c] for c in self.input_channels], axis=0)
        label = channels[LABEL_CHANNEL].astype(np.float32)[np.newaxis, ...]

        if self.augment:
            image, label = self._augment(image, label)

        return {
            "image":    torch.from_numpy(image),
            "label":    torch.from_numpy(label),
            "patch_id": row["patch_id"],
            "category": row["category"],
        }


# ============================================================
# LOSS
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


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def cldice_loss(pred, target, n_iter, eps=1e-6):
    pred = torch.sigmoid(pred)
    skel_pred = soft_skeleton(pred, n_iter)
    skel_target = soft_skeleton(target, n_iter)

    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) + eps
    tprec = tprec / (skel_pred.sum(dim=(1, 2, 3)) + eps)

    trec = (pred * skel_target).sum(dim=(1, 2, 3)) + eps
    trec = trec / (skel_target.sum(dim=(1, 2, 3)) + eps)

    cldice = 2 * tprec * trec / (tprec + trec)
    return 1 - cldice.mean()


def combined_loss(pred, target):
    pw = torch.tensor(BCE_POS_WEIGHT, device=pred.device, dtype=pred.dtype)
    l_bce    = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    l_dice   = dice_loss(pred, target)
    l_cldice = cldice_loss(pred, target, n_iter=CLDICE_ITER)
    return l_bce + LOSS_DICE_WEIGHT * l_dice + (1 - LOSS_DICE_WEIGHT) * l_cldice


# ============================================================
# MODEL
# ============================================================

def adapt_first_conv_for_extra_channels(model, n_input_channels):
    """Replicate 3-channel pretrained first-conv weights for N-channel input (ResNet family)."""
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
    """Replicate 3-channel pretrained patch_embed1.proj weights for N-channel input."""
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
    """Build model based on ARCHITECTURE, with first-conv channel adaptation and decoder dropout."""
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

    # Decoder dropout (architecture-specific)
    if ARCHITECTURE == "resnet_unet":
        for block in model.decoder.blocks:
            block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=DECODER_DROPOUT_P))
        logging.info(f"Decoder dropout: {DECODER_DROPOUT_P} (applied to {len(model.decoder.blocks)} U-Net blocks)")
    elif ARCHITECTURE == "deeplabv3plus":
        if hasattr(model.decoder, "block2"):
            model.decoder.block2 = nn.Sequential(model.decoder.block2, nn.Dropout2d(p=DECODER_DROPOUT_P))
            logging.info(f"Decoder dropout: {DECODER_DROPOUT_P} (applied to DeepLabV3+ block2)")
        else:
            logging.warning("DeepLabV3+ decoder.block2 not found - skipping dropout")
    elif ARCHITECTURE == "segformer":
        logging.info("No decoder dropout for SegFormer (lightweight MLP head)")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Architecture: {ARCHITECTURE}")
    logging.info(f"Trainable parameters: {n_params:,}")
    return model


# ============================================================
# DATA LOADING + SPLIT
# ============================================================

def load_and_split_metadata():
    """Load PL+NL metadata, build source-grouped train/val/test split."""
    df_pl = pd.read_csv(METADATA_CSV_PL)
    df_pl["region"] = "PL"
    df_nl = pd.read_csv(METADATA_CSV_NL)
    df_nl["region"] = "NL"
    df = pd.concat([df_pl, df_nl], ignore_index=True)

    df["comid"] = df["comid"].astype(int)
    df["source_idx_global"] = df["region"] + "_" + df["comid"].astype(str)

    unique_sources = df["source_idx_global"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_sources)

    n_total = len(unique_sources)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)

    train_sources = set(unique_sources[:n_train])
    val_sources   = set(unique_sources[n_train:n_train + n_val])
    test_sources  = set(unique_sources[n_train + n_val:])

    df["split"] = df["source_idx_global"].apply(
        lambda s: "train" if s in train_sources
        else "val" if s in val_sources
        else "test"
    )

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)

    logging.info(f"Total patches: {len(df)}")
    logging.info(f"  Train: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    logging.info(f"  Val:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
    logging.info(f"  Test:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

    df.to_csv(OUTPUT_DIR / "metadata_with_split.csv", index=False)
    logging.info(f"First test patch_ids (sanity check): {df_test['patch_id'].head().tolist()}")

    return df_train, df_val, df_test


def compute_or_load_norm_stats(df_train):
    """Compute per-channel normalization statistics on the train set (Welford-style accumulation)."""
    norm_stats_path = OUTPUT_DIR / "norm_stats.json"

    if norm_stats_path.exists():
        logging.info(f"Loading existing norm_stats from {norm_stats_path}")
        with open(norm_stats_path) as f:
            return json.load(f)

    channels_to_normalize = [c for c in INPUT_CHANNELS if c != "dsm"]
    sums    = {c: 0.0 for c in channels_to_normalize}
    sq_sums = {c: 0.0 for c in channels_to_normalize}
    counts  = {c: 0   for c in channels_to_normalize}

    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Computing norm stats"):
        npz_path = {"PL": PATCHES_DIR_PL, "NL": PATCHES_DIR_NL}[row["region"]] / f"{row['patch_id']}.npz"
        channels = dict(np.load(npz_path))
        for c in channels_to_normalize:
            arr = np.nan_to_num(channels[c].astype(np.float64))
            sums[c]    += arr.sum()
            sq_sums[c] += (arr ** 2).sum()
            counts[c]  += arr.size

    norm_stats = {}
    for c in channels_to_normalize:
        mean = sums[c] / counts[c]
        var  = sq_sums[c] / counts[c] - mean ** 2
        std  = np.sqrt(max(var, 1e-12))
        norm_stats[c] = {"mean": float(mean), "std": float(std)}

    norm_stats["dsm"] = {"mean": 0.0, "std": 1.0}   # sentinel; DSM is per-patch normalized

    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    for c, s in norm_stats.items():
        logging.info(f"  norm_stats {c:20s}  mean={s['mean']:8.3f}  std={s['std']:8.3f}")

    return norm_stats


def build_dataloaders(df_train, df_val, norm_stats):
    patches_root_dirs = {"PL": PATCHES_DIR_PL, "NL": PATCHES_DIR_NL}

    train_ds = LeveeDataset(df_train, patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=True)
    val_ds   = LeveeDataset(df_val,   patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    return train_loader, val_loader, patches_root_dirs


# ============================================================
# TRAIN
# ============================================================

def train(model, train_loader, val_loader):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    scaler = torch.amp.GradScaler(device="cuda", enabled=MIXED_PRECISION)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_dice": [], "train_cldice": [],
        "val_dice": [], "val_cldice": [], "val_score": [], "lr": [],
    }
    best_val_score = -float("inf")
    checkpoint_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(N_EPOCHS):
        # --- Train epoch ---
        model.train()
        train_loss_sum, train_dice_sum, train_cldice_sum, n_train_batches = 0.0, 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [train]"):
            image = batch["image"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=MIXED_PRECISION):
                pred = model(image)
                loss = combined_loss(pred, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            # Track train Dice / clDice (binarized, no_grad) - for overfitting analysis
            with torch.no_grad():
                pred_bin_t = (torch.sigmoid(pred.detach()) > 0.5).float()
                inter_t = (pred_bin_t * label).sum(dim=(1, 2, 3))
                union_t = pred_bin_t.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3))
                train_dice_sum += ((2 * inter_t + 1e-6) / (union_t + 1e-6)).mean().item()

                skel_p_t = soft_skeleton(pred_bin_t, CLDICE_ITER)
                skel_l_t = soft_skeleton(label,      CLDICE_ITER)
                tprec_t = ((skel_p_t * label).sum(dim=(1, 2, 3)) + 1e-6) / (skel_p_t.sum(dim=(1, 2, 3)) + 1e-6)
                trec_t  = ((pred_bin_t * skel_l_t).sum(dim=(1, 2, 3)) + 1e-6) / (skel_l_t.sum(dim=(1, 2, 3)) + 1e-6)
                train_cldice_sum += (2 * tprec_t * trec_t / (tprec_t + trec_t)).mean().item()

            n_train_batches += 1

        train_loss   = train_loss_sum   / n_train_batches
        train_dice   = train_dice_sum   / n_train_batches
        train_cldice = train_cldice_sum / n_train_batches

        # --- Validation ---
        model.eval()
        val_loss_sum, val_dice_sum, val_cldice_sum, n_val_batches = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [val]"):
                image = batch["image"].to(DEVICE, non_blocking=True)
                label = batch["label"].to(DEVICE, non_blocking=True)

                with autocast(enabled=MIXED_PRECISION):
                    pred = model(image)
                    loss = combined_loss(pred, label)

                val_loss_sum += loss.item()
                pred_bin = (torch.sigmoid(pred) > 0.5).float()

                inter = (pred_bin * label).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3))
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                val_dice_sum += dice.mean().item()

                skel_p = soft_skeleton(pred_bin, CLDICE_ITER)
                skel_l = soft_skeleton(label,    CLDICE_ITER)
                tprec = ((skel_p * label).sum(dim=(1, 2, 3)) + 1e-6) / (skel_p.sum(dim=(1, 2, 3)) + 1e-6)
                trec  = ((pred_bin * skel_l).sum(dim=(1, 2, 3)) + 1e-6) / (skel_l.sum(dim=(1, 2, 3)) + 1e-6)
                cldice = 2 * tprec * trec / (tprec + trec)
                val_cldice_sum += cldice.mean().item()

                n_val_batches += 1

        val_loss   = val_loss_sum   / n_val_batches
        val_dice   = val_dice_sum   / n_val_batches
        val_cldice = val_cldice_sum / n_val_batches
        val_score  = (val_dice + val_cldice) / 2

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["train_cldice"].append(train_cldice)
        history["val_dice"].append(val_dice)
        history["val_cldice"].append(val_cldice)
        history["val_score"].append(val_score)
        history["lr"].append(current_lr)

        scheduler.step()

        logging.info(
            f"Epoch {epoch+1:3d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_dice={train_dice:.4f} | val_dice={val_dice:.4f} | "
            f"train_cldice={train_cldice:.4f} | val_cldice={val_cldice:.4f} | "
            f"val_score={val_score:.4f} | lr={current_lr:.2e}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_dice":   val_dice,
                "val_cldice": val_cldice,
                "val_score":  val_score,
            }, checkpoint_path)
            logging.info(f"  -> new best (val_score={val_score:.4f}); saved checkpoint")

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    return history, checkpoint_path


def plot_training_curves(history):
    """Multi-panel evolution plot: loss, Dice (train+val), clDice (train+val), val_score, LR."""
    hist_df = pd.DataFrame(history)
    best_epoch = int(hist_df.loc[hist_df["val_score"].idxmax(), "epoch"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # [0,0] Loss
    axes[0, 0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss", linewidth=2)
    axes[0, 0].plot(hist_df["epoch"], hist_df["val_loss"],   label="Val loss",   linewidth=2)
    axes[0, 0].axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # [0,1] Dice
    axes[0, 1].plot(hist_df["epoch"], hist_df["train_dice"], color="tab:blue",  linewidth=2, label="Train Dice")
    axes[0, 1].plot(hist_df["epoch"], hist_df["val_dice"],   color="tab:green", linewidth=2, label="Val Dice")
    axes[0, 1].axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Dice")
    axes[0, 1].set_title("Dice (train vs val)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # [0,2] clDice
    axes[0, 2].plot(hist_df["epoch"], hist_df["train_cldice"], color="tab:cyan",   linewidth=2, label="Train clDice")
    axes[0, 2].plot(hist_df["epoch"], hist_df["val_cldice"],   color="tab:purple", linewidth=2, label="Val clDice")
    axes[0, 2].axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("clDice")
    axes[0, 2].set_title("clDice (train vs val)")
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    # [1,0] val_score (selection criterion)
    axes[1, 0].plot(hist_df["epoch"], hist_df["val_score"], color="tab:orange", linewidth=2, label="Val Score (Dice+clDice)/2")
    axes[1, 0].axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Val Score (combined selection criterion)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Train/val gap (Dice + clDice) - overfitting indicator
    axes[1, 1].plot(hist_df["epoch"], hist_df["train_dice"]   - hist_df["val_dice"],   color="tab:green",  linewidth=2, label="Dice gap")
    axes[1, 1].plot(hist_df["epoch"], hist_df["train_cldice"] - hist_df["val_cldice"], color="tab:purple", linewidth=2, label="clDice gap")
    axes[1, 1].axhline(0, color="black", linewidth=0.5)
    axes[1, 1].axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Train - Val")
    axes[1, 1].set_title("Train/Val gap (overfitting indicator)")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # [1,2] LR schedule
    axes[1, 2].plot(hist_df["epoch"], hist_df["lr"], color="tab:red", linewidth=2)
    axes[1, 2].axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("Learning rate")
    axes[1, 2].set_title("LR schedule")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved training curves to {out_path}")


def build_metrics_summary(history, checkpoint, val_results, test_results, run_start_time):
    """Build a structured summary dict (for JSON export and downstream comparison)."""
    import time

    # Helper: aggregate a results df into a dict of metrics
    def aggregate(df):
        total_tp = int(df["tp"].sum())
        total_fp = int(df["fp"].sum())
        total_fn = int(df["fn"].sum())
        total_tn = int(df["tn"].sum())

        micro_precision = total_tp / (total_tp + total_fp + 1e-6)
        micro_recall    = total_tp / (total_tp + total_fn + 1e-6)
        micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)
        micro_iou       = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        micro_dice      = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)

        return {
            "n_patches": len(df),
            "mean_per_patch": {
                "dice":      float(df["dice"].mean()),
                "iou":       float(df["iou"].mean()),
                "cldice":    float(df["cldice"].mean()),
                "f1":        float(df["f1"].mean()),
                "precision": float(df["precision"].mean()),
                "recall":    float(df["recall"].mean()),
            },
            "std_per_patch": {
                "dice":      float(df["dice"].std()),
                "iou":       float(df["iou"].std()),
                "cldice":    float(df["cldice"].std()),
                "f1":        float(df["f1"].std()),
                "precision": float(df["precision"].std()),
                "recall":    float(df["recall"].std()),
            },
            "micro_averaged": {
                "dice":      float(micro_dice),
                "iou":       float(micro_iou),
                "f1":        float(micro_f1),
                "precision": float(micro_precision),
                "recall":    float(micro_recall),
            },
            "confusion_matrix": {
                "tp": total_tp, "fp": total_fp,
                "fn": total_fn, "tn": total_tn,
            },
        }

    def aggregate_grouped(df, group_cols):
        """Aggregate per group (e.g. ['category', 'patch_type'])."""
        out = {}
        for keys, sub in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key = " | ".join(str(k) for k in keys)
            out[key] = aggregate(sub)
        return out

    summary = {
        "run_info": {
            "architecture":         ARCHITECTURE,
            "segformer_backbone":   SEGFORMER_BACKBONE if ARCHITECTURE == "segformer" else None,
            "input_channels":       INPUT_CHANNELS,
            "n_channels":           N_INPUT_CHANNELS,
            "n_epochs_planned":     N_EPOCHS,
            "n_epochs_run":         len(history["epoch"]),
            "batch_size":           BATCH_SIZE,
            "learning_rate":        LR,
            "weight_decay":         WEIGHT_DECAY,
            "loss_dice_weight":     LOSS_DICE_WEIGHT,
            "bce_pos_weight":       BCE_POS_WEIGHT,
            "seed":                 SEED,
            "output_dir":           str(OUTPUT_DIR),
            "best_epoch":           int(checkpoint["epoch"]),
            "best_val_score":       float(checkpoint.get("val_score", float("nan"))),
            "best_val_dice":        float(checkpoint["val_dice"]),
            "best_val_cldice":      float(checkpoint.get("val_cldice", float("nan"))),
            "best_val_loss":        float(checkpoint["val_loss"]),
            "total_time_seconds":   float(time.time() - run_start_time),
        },
        "training_history": {
            k: [float(v) for v in vals] if k != "epoch" else [int(v) for v in vals]
            for k, vals in history.items()
        },
        "val_metrics": {
            "overall":     aggregate(val_results),
            "by_category": aggregate_grouped(val_results,  ["category", "patch_type"]),
            "by_region":   aggregate_grouped(val_results,  ["region",   "patch_type"]),
        },
        "test_metrics": {
            "overall":     aggregate(test_results),
            "by_category": aggregate_grouped(test_results, ["category", "patch_type"]),
            "by_region":   aggregate_grouped(test_results, ["region",   "patch_type"]),
        },
    }
    return summary


def save_metrics_summary_json(summary, output_path):
    """Save structured summary as JSON (compare_variants.py reads this)."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved metrics summary to {output_path}")


# ============================================================
# EVALUATION  (with extended metrics: TP/FP/FN/TN, precision, recall, F1)
# ============================================================

def evaluate_split(model, df_split, split_name, norm_stats, patches_root_dirs):
    """
    Evaluate model on a split, save per-patch predictions, return metrics df.
    Extended per-patch metrics:
      - dice, iou, cldice                  (existing)
      - tp, fp, fn, tn                     (confusion matrix counts)
      - precision, recall, f1              (derived from counts)
    Plus micro-averaged Dice/IoU/F1 in the aggregation step.
    """
    pred_dir = OUTPUT_DIR / f"predictions_{split_name}"
    if SAVE_PREDICTIONS:
        pred_dir.mkdir(exist_ok=True)

    ds = LeveeDataset(df_split, patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=False)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name} set"):
            image = batch["image"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            with autocast(enabled=MIXED_PRECISION):
                pred = model(image)
                pred_prob = torch.sigmoid(pred)

            pred_bin = (pred_prob > 0.5).float()

            for i in range(image.size(0)):
                p = pred_bin[i:i+1]
                l = label[i:i+1]

                # Counts (confusion matrix)
                tp = int((p * l).sum().item())
                fp = int((p * (1.0 - l)).sum().item())
                fn = int(((1.0 - p) * l).sum().item())
                tn = int(((1.0 - p) * (1.0 - l)).sum().item())

                # Derived from counts
                precision = tp / (tp + fp + 1e-6)
                recall    = tp / (tp + fn + 1e-6)
                f1        = 2 * precision * recall / (precision + recall + 1e-6)
                iou       = tp / (tp + fp + fn + 1e-6)
                dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)

                # clDice on binarized prediction
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
        df_split[["patch_id", "region", "patch_type"]],
        on="patch_id", how="left",
    )
    return df_results


def report_metrics(df_results, split_name):
    """Log per-category/region/patch_type breakdown plus overall mean and micro-averaged metrics."""
    logging.info("=" * 70)
    logging.info(f"{split_name.upper()} SET METRICS")
    logging.info("=" * 70)

    # By category x patch_type
    logging.info(f"\n--- {split_name}: By category x patch_type (mean per-patch) ---")
    agg_cat = df_results.groupby(["category", "patch_type"])[
        ["dice", "iou", "cldice", "precision", "recall", "f1"]
    ].mean()
    for line in agg_cat.to_string(float_format=lambda x: f"{x:.4f}").splitlines():
        logging.info(line)

    # By region x patch_type
    logging.info(f"\n--- {split_name}: By region x patch_type (mean per-patch) ---")
    agg_region = df_results.groupby(["region", "patch_type"])[
        ["dice", "iou", "cldice", "precision", "recall", "f1"]
    ].mean()
    for line in agg_region.to_string(float_format=lambda x: f"{x:.4f}").splitlines():
        logging.info(line)

    # Overall: both mean per-patch AND micro-averaged (from summed counts)
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

    return agg_cat, agg_region


# ============================================================
# MAIN
# ============================================================

def main():
    import time
    run_start_time = time.time()

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUTPUT_DIR)

    log.info(f"Device: {DEVICE}")
    log.info(f"Architecture: {ARCHITECTURE}")
    log.info(f"Input channels ({N_INPUT_CHANNELS}): {INPUT_CHANNELS}")
    log.info(f"Hyperparams: batch={BATCH_SIZE}, epochs={N_EPOCHS}, lr={LR}, wd={WEIGHT_DECAY}")
    log.info(f"num_workers={NUM_WORKERS}")

    # 1. Load data and split
    df_train, df_val, df_test = load_and_split_metadata()

    # 2. Norm stats
    norm_stats = compute_or_load_norm_stats(df_train)

    # 3. Build dataloaders
    train_loader, val_loader, patches_root_dirs = build_dataloaders(df_train, df_val, norm_stats)

    # 4. Build model
    model = build_model()

    # 5. Train
    log.info("Starting training...")
    history, checkpoint_path = train(model, train_loader, val_loader)

    # 6. Plot curves
    plot_training_curves(history)

    # 7. Load best checkpoint and evaluate
    log.info("Loading best checkpoint for evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info(f"Best checkpoint from epoch {checkpoint['epoch']}: "
             f"val_score={checkpoint.get('val_score', float('nan')):.4f}, "
             f"val_dice={checkpoint['val_dice']:.4f}, "
             f"val_cldice={checkpoint.get('val_cldice', float('nan')):.4f}")

    val_results  = evaluate_split(model, df_val,  "val",  norm_stats, patches_root_dirs)
    test_results = evaluate_split(model, df_test, "test", norm_stats, patches_root_dirs)

    val_results.to_csv (OUTPUT_DIR / "val_results_per_patch.csv",  index=False)
    test_results.to_csv(OUTPUT_DIR / "test_results_per_patch.csv", index=False)

    report_metrics(val_results,  "val")
    agg_cat, agg_region = report_metrics(test_results, "test")
    agg_cat.to_csv(OUTPUT_DIR / "test_results_by_category.csv")
    agg_region.to_csv(OUTPUT_DIR / "test_results_by_region.csv")

    if SAVE_PREDICTIONS:
        n_val  = len(list((OUTPUT_DIR / "predictions_val").glob("*.npz")))
        n_test = len(list((OUTPUT_DIR / "predictions_test").glob("*.npz")))
        log.info(f"Saved predictions: val={n_val}, test={n_test} files")

    # Build + save metrics summary (machine-readable, used by compare_variants.py)
    summary = build_metrics_summary(history, checkpoint, val_results, test_results, run_start_time)
    save_metrics_summary_json(summary, OUTPUT_DIR / "metrics_summary.json")

    elapsed_h = (summary["run_info"]["total_time_seconds"]) / 3600
    log.info(f"Total run time: {elapsed_h:.2f} hours")
    log.info("Done.")


if __name__ == "__main__":
    main()
