"""
SegFormer training for levee detection (PL + USA patches, comid-grouped split).

Author: Jakub Zapletal
Date:   2026-04-14
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

matplotlib.use("Agg")  # no display required when running headless
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ============================================================
# CONFIG
# ============================================================

# ------- Paths ----------------------------------------------
# Two training sources; the held-out basin is evaluated separately
PATCHES_DIR_USA = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_USA\patches"
)
PATCHES_DIR_PL = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_train\patches"
)
METADATA_CSV_USA = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\patches_v02_USA\patches_metadata.csv"
)
METADATA_CSV_PL = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\patches\patches_PL_train\patches_metadata.csv"
)

# Output dir, unique per variant
OUTPUT_DIR = Path(
    r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\geomorphological_ML\_FINAL_EVAL\training_v06_segformer_PL_US"
)
IMG_DIR = OUTPUT_DIR / "img"


# ------- Reproducibility ------------------------------------
SEED = 42


# ------- Hardware -------------------------------------------
NUM_WORKERS = 12


# ------- Channels --------------------------------------------
# Full channel set incl. the binary water mask; water stays 0/1, not z-scored
INPUT_CHANNELS = [
    "dsm",
    "tpi_r5",
    "tpi_r10",
    "tpi_r15",
    "canopy_height",
    "canopy_height_sd",
    "water",
]
LABEL_CHANNEL = "label"
WATER_CHANNEL = "water"


# ------- Train/val split -------------------------------------
# Split by reach id (comid) so segments of the same embankment stay together;
# the held-out basin serves as the test set
SPLIT_BY = "comid"
TRAIN_FRAC = 0.75
VAL_FRAC = 0.25


# ------- Training hyperparameters ---------------------------
BATCH_SIZE = 16
N_EPOCHS = 150
LR = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
MIXED_PRECISION = True


# ------- Loss function --------------------------------------
LOSS_DICE_WEIGHT = 0.25
CLDICE_ITER = 5
BCE_POS_WEIGHT = 20.0


# ------- Architecture ---------------------------------------
# One of: "resnet_unet" | "segformer" | "deeplabv3plus"
ARCHITECTURE = "segformer"

RESNET_BACKBONE = "resnet34"
SEGFORMER_BACKBONE = "mit_b2"
DEEPLAB_BACKBONE = "resnet50"

RESNET_ENCODER_WEIGHTS = "imagenet"
DEEPLAB_ENCODER_WEIGHTS = "imagenet"
SEGFORMER_ENCODER_WEIGHTS = "imagenet"

DECODER_DROPOUT_P = 0.1


# ------- Augmentation ---------------------------------------
USE_FLIP_H = True
USE_FLIP_V = True
USE_ROT_90 = True


# ------- Evaluation ------------------------------------------
SAVE_PREDICTIONS = True  # save per-patch val predictions for stacking


# Derived
N_INPUT_CHANNELS = len(INPUT_CHANNELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOGGING
# ============================================================


def setup_logging(output_dir):
    """Configure logging to both console and a file inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.txt"

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

    def __init__(
        self, metadata_df, patches_root_dirs, input_channels, norm_stats, augment=False
    ):
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
            elif ch_name == WATER_CHANNEL:
                pass  # binary mask, leave as 0/1
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

        cat = row["category"]
        if pd.isna(cat):  # negatives have no category
            cat = row["patch_type"]

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "patch_id": row["patch_id"],
            "category": str(cat),
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
    l_bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    l_dice = dice_loss(pred, target)
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

    new_conv = nn.Conv2d(
        n_input_channels,
        out_ch,
        kernel_size=(kh, kw),
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
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

    new_conv = nn.Conv2d(
        n_input_channels,
        out_ch,
        kernel_size=(kh, kw),
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    encoder.patch_embed1.proj = new_conv
    return model


def build_model():
    """Build model based on ARCHITECTURE, with first-conv channel adaptation and decoder dropout."""
    if ARCHITECTURE == "resnet_unet":
        model = smp.Unet(
            encoder_name=RESNET_BACKBONE,
            encoder_weights=RESNET_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_for_extra_channels(model, N_INPUT_CHANNELS)

    elif ARCHITECTURE == "segformer":
        model = smp.Segformer(
            encoder_name=SEGFORMER_BACKBONE,
            encoder_weights=SEGFORMER_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_segformer(model, N_INPUT_CHANNELS)

    elif ARCHITECTURE == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=DEEPLAB_BACKBONE,
            encoder_weights=DEEPLAB_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_for_extra_channels(model, N_INPUT_CHANNELS)

    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

    if ARCHITECTURE == "resnet_unet":
        for block in model.decoder.blocks:
            block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=DECODER_DROPOUT_P))
        logging.info(
            f"Decoder dropout: {DECODER_DROPOUT_P} (applied to {len(model.decoder.blocks)} U-Net blocks)"
        )
    elif ARCHITECTURE == "deeplabv3plus":
        if hasattr(model.decoder, "block2"):
            model.decoder.block2 = nn.Sequential(
                model.decoder.block2, nn.Dropout2d(p=DECODER_DROPOUT_P)
            )
            logging.info(
                f"Decoder dropout: {DECODER_DROPOUT_P} (applied to DeepLabV3+ block2)"
            )
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
    """Load USA+PL metadata, build region+comid-grouped train/val split."""
    df_usa = pd.read_csv(METADATA_CSV_USA)
    df_usa["region"] = "USA"
    df_pl = pd.read_csv(METADATA_CSV_PL)
    df_pl["region"] = "PL"
    df = pd.concat([df_usa, df_pl], ignore_index=True)

    df["comid"] = pd.to_numeric(df["comid"], errors="coerce")
    n_missing = int(df["comid"].isna().sum())
    if n_missing:
        logging.warning(
            f"Dropping {n_missing} patches with no comid (cannot group for split)"
        )
        df = df[df["comid"].notna()].copy()
    df["comid"] = df["comid"].astype(int)
    # Prefix comid with region to avoid collisions between datasets
    df["comid_global"] = df["region"] + "_" + df["comid"].astype(str)

    unique_sources = df["comid_global"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_sources)

    n_total = len(unique_sources)
    n_train = int(n_total * TRAIN_FRAC)

    train_sources = set(unique_sources[:n_train])

    df["split"] = df["comid_global"].apply(lambda s: "train" if s in train_sources else "val")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    logging.info(f"Total patches: {len(df)}  (comid groups: {n_total})")
    logging.info(f"  Train: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    logging.info(f"  Val:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")

    df.to_csv(OUTPUT_DIR / "metadata_with_split.csv", index=False)
    logging.info(
        f"First val patch_ids (sanity check): {df_val['patch_id'].head().tolist()}"
    )

    return df_train, df_val


def compute_or_load_norm_stats(df_train):
    """Compute per-channel normalization statistics on the train set. DSM (per-patch
    median) and the binary water mask are excluded; everything else is z-scored."""
    norm_stats_path = OUTPUT_DIR / "norm_stats.json"

    if norm_stats_path.exists():
        logging.info(f"Loading existing norm_stats from {norm_stats_path}")
        with open(norm_stats_path) as f:
            return json.load(f)

    channels_to_normalize = [
        c for c in INPUT_CHANNELS if c not in ("dsm", WATER_CHANNEL)
    ]
    sums = {c: 0.0 for c in channels_to_normalize}
    sq_sums = {c: 0.0 for c in channels_to_normalize}
    counts = {c: 0 for c in channels_to_normalize}

    patches_root_dirs = {"USA": PATCHES_DIR_USA, "PL": PATCHES_DIR_PL}
    for _, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="Computing norm stats"
    ):
        npz_path = patches_root_dirs[row["region"]] / f"{row['patch_id']}.npz"
        channels = dict(np.load(npz_path))
        for c in channels_to_normalize:
            arr = np.nan_to_num(channels[c].astype(np.float64))
            sums[c] += arr.sum()
            sq_sums[c] += (arr**2).sum()
            counts[c] += arr.size

    norm_stats = {}
    for c in channels_to_normalize:
        mean = sums[c] / counts[c]
        var = sq_sums[c] / counts[c] - mean**2
        std = np.sqrt(max(var, 1e-12))
        norm_stats[c] = {"mean": float(mean), "std": float(std)}

    norm_stats["dsm"] = {"mean": 0.0, "std": 1.0}  # sentinel; per-patch normalized
    norm_stats[WATER_CHANNEL] = {"mean": 0.0, "std": 1.0}  # sentinel; kept as 0/1

    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    for c, s in norm_stats.items():
        logging.info(
            f"  norm_stats {c:20s}  mean={s['mean']:8.3f}  std={s['std']:8.3f}"
        )

    return norm_stats


def build_dataloaders(df_train, df_val, norm_stats):
    patches_root_dirs = {"USA": PATCHES_DIR_USA, "PL": PATCHES_DIR_PL}
    train_ds = LeveeDataset(
        df_train, patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=True
    )
    val_ds = LeveeDataset(
        df_val, patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    return train_loader, val_loader, patches_root_dirs


def select_positive_patch_ids(df_split, n_required, split_name, patches_root_dirs):
    """Select deterministic positive patch_ids by validating label pixels in NPZ files."""
    rng = np.random.default_rng(SEED + (0 if split_name == "train" else 10_000))
    df_shuffled = df_split.sample(
        frac=1, random_state=int(rng.integers(2**31))
    ).reset_index(drop=True)

    selected = []
    for _, row in df_shuffled.iterrows():
        pid = str(row["patch_id"])
        npz_path = patches_root_dirs[row["region"]] / f"{pid}.npz"
        if not npz_path.exists():
            continue

        channels = dict(np.load(npz_path))
        label = np.nan_to_num(channels[LABEL_CHANNEL].astype(np.float32), nan=0.0)
        if float(label.sum()) > 0:
            selected.append(
                {"patch_id": pid, "patches_root": str(patches_root_dirs[row["region"]])}
            )
            if len(selected) == n_required:
                break

    if len(selected) < n_required:
        raise RuntimeError(
            f"Requested {n_required} positive patches for {split_name}, found only {len(selected)}"
        )
    return selected


def build_fixed_locations(df_train, df_val, patches_root_dirs):
    """Create 6 fixed monitoring locations: 3 train + 3 val, all positive."""
    train_entries = select_positive_patch_ids(
        df_train, n_required=3, split_name="train", patches_root_dirs=patches_root_dirs
    )
    val_entries = select_positive_patch_ids(
        df_val, n_required=3, split_name="val", patches_root_dirs=patches_root_dirs
    )

    fixed_locations = []
    for i, entry in enumerate(train_entries, start=1):
        fixed_locations.append(
            {
                "name": f"loc_{i:02d}_train",
                "split": "train",
                "patch_id": entry["patch_id"],
                "patches_root": entry["patches_root"],
            }
        )
    for i, entry in enumerate(val_entries, start=4):
        fixed_locations.append(
            {
                "name": f"loc_{i:02d}_val",
                "split": "val",
                "patch_id": entry["patch_id"],
                "patches_root": entry["patches_root"],
            }
        )
    return fixed_locations


def _normalize_channels_for_inference(channels, norm_stats):
    """Normalize channels exactly like LeveeDataset._normalize for fixed-location rendering."""
    out = {}
    for ch_name in INPUT_CHANNELS:
        arr = channels[ch_name].astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if ch_name == "dsm":
            arr = arr - np.median(arr)
        elif ch_name == WATER_CHANNEL:
            pass
        else:
            stats = norm_stats[ch_name]
            arr = (arr - stats["mean"]) / (stats["std"] + 1e-6)
        out[ch_name] = arr
    return out


def save_fixed_location_images(model, norm_stats, fixed_locations, epoch, out_dir, patches_root_dirs):
    """Save per-epoch prediction visuals for fixed train/val locations."""
    model.eval()
    with torch.no_grad():
        for loc in fixed_locations:
            patch_id = loc["patch_id"]
            split = loc["split"]
            loc_dir = out_dir / loc["name"]
            loc_dir.mkdir(parents=True, exist_ok=True)

            npz_path = Path(loc["patches_root"]) / f"{patch_id}.npz"
            channels = dict(np.load(npz_path))
            channels_norm = _normalize_channels_for_inference(channels, norm_stats)

            image = np.stack([channels_norm[c] for c in INPUT_CHANNELS], axis=0)
            label = channels[LABEL_CHANNEL].astype(np.float32)

            x = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
            with autocast(enabled=MIXED_PRECISION):
                pred = model(x)
                pred_prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()

            dsm = image[0]
            dsm_lo, dsm_hi = np.percentile(dsm, [2, 98])
            if dsm_hi <= dsm_lo:
                dsm_lo, dsm_hi = float(dsm.min()), float(dsm.max() + 1e-6)

            if WATER_CHANNEL in INPUT_CHANNELS:
                water_idx = INPUT_CHANNELS.index(WATER_CHANNEL)
                water = image[water_idx]
            else:
                water = np.zeros_like(label)

            fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), dpi=130)
            panels = [
                (dsm, "DSM", "terrain", (dsm_lo, dsm_hi)),
                (water, "Water", "Blues", (0.0, 1.0)),
                (label, "GT", "gray", (0.0, 1.0)),
                (pred_prob, "Pred", "magma", (0.0, 1.0)),
            ]

            for ax, (arr, title, cmap, limits) in zip(axes, panels):
                im = ax.imshow(arr, cmap=cmap, vmin=limits[0], vmax=limits[1])
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            axes[-1].contour(pred_prob > 0.5, levels=[0.5], colors="cyan", linewidths=0.8)
            fig.suptitle(
                f"{loc['name']}  |  split={split}  |  patch_id={patch_id}  |  epoch={epoch:03d}",
                fontsize=11,
                y=1.03,
            )
            fig.tight_layout()

            fig.savefig(loc_dir / f"epoch_{epoch:03d}.png", bbox_inches="tight")
            fig.savefig(loc_dir / "latest.png", bbox_inches="tight")
            plt.close(fig)


def save_epoch_dashboard(history, epoch, preview_batch, out_dir, is_best=False):
    """Save a visually rich per-epoch dashboard with curves and prediction previews."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10), dpi=130)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 1.0], hspace=0.28, wspace=0.2)

    # --- Curves ---
    ax_loss = fig.add_subplot(gs[0, :2])
    ax_score = fig.add_subplot(gs[0, 2:])

    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], color="#264653", linewidth=2.5, label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], color="#e76f51", linewidth=2.5, label="Val loss")
    ax_loss.set_title("Loss Trajectory", fontsize=13, weight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(frameon=False)

    ax_score.plot(epochs, history["val_dice"], color="#2a9d8f", linewidth=2.2, label="Val Dice")
    ax_score.plot(epochs, history["val_cldice"], color="#8ab17d", linewidth=2.2, label="Val clDice")
    ax_score.plot(epochs, history["val_score"], color="#f4a261", linewidth=2.8, label="Val Score")
    ax_score.set_title("Validation Quality", fontsize=13, weight="bold")
    ax_score.set_xlabel("Epoch")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(0, 1)
    ax_score.grid(alpha=0.25)
    ax_score.legend(frameon=False)

    fig.suptitle(
        f"Levee Training Dashboard  |  Epoch {epoch:03d}  |  Best val_score={max(history['val_score']):.4f}",
        fontsize=16,
        weight="bold",
        y=0.98,
    )

    # --- Preview panel: DSM, Water, Label, Prediction ---
    if preview_batch is not None:
        image = preview_batch["image"]
        label = preview_batch["label"]
        pred_prob = preview_batch["pred_prob"]

        n_show = min(2, image.shape[0])
        for row in range(n_show):
            dsm = image[row, 0].numpy()
            water = image[row, -1].numpy()
            gt = label[row, 0].numpy()
            pp = pred_prob[row, 0].numpy()

            # Robust display scaling for DSM to improve visual readability.
            dsm_lo, dsm_hi = np.percentile(dsm, [2, 98])
            if dsm_hi <= dsm_lo:
                dsm_lo, dsm_hi = float(dsm.min()), float(dsm.max() + 1e-6)

            panels = [
                (dsm, "DSM (normalized)", "terrain", (dsm_lo, dsm_hi)),
                (water, "Water mask", "Blues", (0.0, 1.0)),
                (gt, "Ground truth", "gray", (0.0, 1.0)),
                (pp, "Prediction prob.", "magma", (0.0, 1.0)),
            ]

            for col, (arr, title, cmap, limits) in enumerate(panels):
                ax = fig.add_subplot(gs[row + 1, col])
                im = ax.imshow(arr, cmap=cmap, vmin=limits[0], vmax=limits[1])
                ax.set_title(f"Sample {row+1} - {title}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=8)

                if title == "Prediction prob.":
                    ax.contour(pp > 0.5, levels=[0.5], colors="cyan", linewidths=0.8)

    epoch_path = out_dir / f"epoch_{epoch:03d}.png"
    latest_path = out_dir / "latest.png"
    fig.savefig(epoch_path, bbox_inches="tight")
    fig.savefig(latest_path, bbox_inches="tight")
    if is_best:
        fig.savefig(out_dir / "best.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# TRAIN
# ============================================================


def train(model, train_loader, val_loader, norm_stats, fixed_locations, patches_root_dirs):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
    scaler = torch.amp.GradScaler(device="cuda", enabled=MIXED_PRECISION)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_cldice": [],
        "val_score": [],
        "lr": [],
    }
    best_val_score = -float("inf")
    checkpoint_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(N_EPOCHS):
        # --- Train epoch ---
        model.train()
        train_loss_sum, n_train_batches = 0.0, 0

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
            n_train_batches += 1

        train_loss = train_loss_sum / n_train_batches

        # --- Validation ---
        model.eval()
        val_loss_sum, val_dice_sum, val_cldice_sum, n_val_batches = 0.0, 0.0, 0.0, 0

        preview_batch = None
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [val]"):
                image = batch["image"].to(DEVICE, non_blocking=True)
                label = batch["label"].to(DEVICE, non_blocking=True)

                with autocast(enabled=MIXED_PRECISION):
                    pred = model(image)
                    loss = combined_loss(pred, label)
                    pred_prob = torch.sigmoid(pred)

                if preview_batch is None:
                    n_preview = min(2, image.size(0))
                    preview_batch = {
                        "image": image[:n_preview].detach().cpu(),
                        "label": label[:n_preview].detach().cpu(),
                        "pred_prob": pred_prob[:n_preview].detach().cpu(),
                    }

                val_loss_sum += loss.item()
                pred_bin = (pred_prob > 0.5).float()

                inter = (pred_bin * label).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3))
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                val_dice_sum += dice.mean().item()

                skel_p = soft_skeleton(pred_bin, CLDICE_ITER)
                skel_l = soft_skeleton(label, CLDICE_ITER)
                tprec = ((skel_p * label).sum(dim=(1, 2, 3)) + 1e-6) / (
                    skel_p.sum(dim=(1, 2, 3)) + 1e-6
                )
                trec = ((pred_bin * skel_l).sum(dim=(1, 2, 3)) + 1e-6) / (
                    skel_l.sum(dim=(1, 2, 3)) + 1e-6
                )
                cldice = 2 * tprec * trec / (tprec + trec)
                val_cldice_sum += cldice.mean().item()

                n_val_batches += 1

        val_loss = val_loss_sum / n_val_batches
        val_dice = val_dice_sum / n_val_batches
        val_cldice = val_cldice_sum / n_val_batches
        val_score = (val_dice + val_cldice) / 2

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_cldice"].append(val_cldice)
        history["val_score"].append(val_score)
        history["lr"].append(current_lr)

        scheduler.step()

        logging.info(
            f"Epoch {epoch+1:3d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | val_cldice={val_cldice:.4f} | "
            f"val_score={val_score:.4f} | lr={current_lr:.2e}"
        )

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_cldice": val_cldice,
                    "val_score": val_score,
                },
                checkpoint_path,
            )
            logging.info(f"  -> new best (val_score={val_score:.4f}); saved checkpoint")

        save_epoch_dashboard(
            history=history,
            epoch=epoch + 1,
            preview_batch=preview_batch,
            out_dir=IMG_DIR,
            is_best=is_best,
        )
        save_fixed_location_images(
            model=model,
            norm_stats=norm_stats,
            fixed_locations=fixed_locations,
            epoch=epoch + 1,
            out_dir=IMG_DIR,
            patches_root_dirs=patches_root_dirs,
        )

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    return history, checkpoint_path


def plot_training_curves(history):
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    best_epoch = int(hist_df.loc[hist_df["val_score"].idxmax(), "epoch"])

    axes[0].plot(
        hist_df["epoch"], hist_df["train_loss"], label="Train loss", linewidth=2
    )
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val loss", linewidth=2)
    axes[0].axvline(
        best_epoch,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch ({best_epoch})",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        hist_df["epoch"],
        hist_df["val_dice"],
        color="tab:green",
        linewidth=2,
        label="Val Dice",
    )
    axes[1].plot(
        hist_df["epoch"],
        hist_df["val_cldice"],
        color="tab:purple",
        linewidth=2,
        label="Val clDice",
    )
    axes[1].axvline(
        best_epoch,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch ({best_epoch})",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Dice & clDice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        hist_df["epoch"],
        hist_df["val_score"],
        color="tab:orange",
        linewidth=2,
        label="Val Score (Dice+clDice)/2",
    )
    axes[2].axvline(
        best_epoch,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch ({best_epoch})",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Combined Score")
    axes[2].set_title("Combined Selection Criterion")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved training curves to {out_path}")


def save_final_summary_graph(history, out_dir):
    """Save a final end-of-training summary figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    best_idx = hist_df["val_score"].idxmax()
    best_epoch = int(hist_df.loc[best_idx, "epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140)
    fig.patch.set_facecolor("#fbfbf8")

    # Loss panel
    ax = axes[0, 0]
    ax.plot(hist_df["epoch"], hist_df["train_loss"], color="#355070", linewidth=2.4, label="Train")
    ax.plot(hist_df["epoch"], hist_df["val_loss"], color="#e56b6f", linewidth=2.4, label="Validation")
    ax.axvline(best_epoch, color="#6d597a", linestyle="--", linewidth=1.5)
    ax.set_title("Loss", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Validation quality panel
    ax = axes[0, 1]
    ax.plot(hist_df["epoch"], hist_df["val_dice"], color="#2a9d8f", linewidth=2.2, label="Dice")
    ax.plot(hist_df["epoch"], hist_df["val_cldice"], color="#8ab17d", linewidth=2.2, label="clDice")
    ax.plot(hist_df["epoch"], hist_df["val_score"], color="#f4a261", linewidth=2.8, label="Score")
    ax.axvline(best_epoch, color="#6d597a", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_title("Validation Metrics", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Learning rate panel
    ax = axes[1, 0]
    ax.plot(hist_df["epoch"], hist_df["lr"], color="#bc6c25", linewidth=2.4)
    ax.set_title("Learning Rate Schedule", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.25)

    # Summary text panel
    ax = axes[1, 1]
    ax.axis("off")
    last = hist_df.iloc[-1]
    best = hist_df.loc[best_idx]
    summary_text = (
        "Training Summary\n\n"
        f"Total epochs: {int(last['epoch'])}\n"
        f"Best epoch:  {best_epoch}\n"
        f"Best val_score:  {best['val_score']:.4f}\n"
        f"Best val_dice:   {best['val_dice']:.4f}\n"
        f"Best val_cldice: {best['val_cldice']:.4f}\n\n"
        f"Final train_loss: {last['train_loss']:.4f}\n"
        f"Final val_loss:   {last['val_loss']:.4f}\n"
        f"Final val_score:  {last['val_score']:.4f}"
    )
    ax.text(
        0.03,
        0.95,
        summary_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#fefae0", edgecolor="#dda15e", alpha=0.95),
    )

    fig.suptitle("Final Training Report", fontsize=18, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "final_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved final summary graph to {out_path}")


# ============================================================
# EVALUATION  (val-only sanity check; held-out basin is done separately)
# ============================================================


def evaluate_split(model, df_split, split_name, norm_stats, patches_root_dirs):
    """
    Evaluate model on a split, save per-patch predictions, return metrics df.
    Per-patch metrics: dice, iou, cldice, tp/fp/fn/tn, precision, recall, f1.
    """
    pred_dir = OUTPUT_DIR / f"predictions_{split_name}"
    if SAVE_PREDICTIONS:
        pred_dir.mkdir(exist_ok=True)

    ds = LeveeDataset(df_split, patches_root_dirs, INPUT_CHANNELS, norm_stats, augment=False)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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
                p = pred_bin[i : i + 1]
                l = label[i : i + 1]

                tp = int((p * l).sum().item())
                fp = int((p * (1.0 - l)).sum().item())
                fn = int(((1.0 - p) * l).sum().item())
                tn = int(((1.0 - p) * (1.0 - l)).sum().item())

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                iou = tp / (tp + fp + fn + 1e-6)
                dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

                skel_p = soft_skeleton(p, CLDICE_ITER)
                skel_l = soft_skeleton(l, CLDICE_ITER)
                tprec = ((skel_p * l).sum() + 1e-6) / (skel_p.sum() + 1e-6)
                trec = ((p * skel_l).sum() + 1e-6) / (skel_l.sum() + 1e-6)
                cldice = (2 * tprec * trec / (tprec + trec)).item()

                records.append(
                    {
                        "patch_id": batch["patch_id"][i],
                        "category": batch["category"][i],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "dice": dice,
                        "iou": iou,
                        "cldice": cldice,
                        "n_label_px": int(l.sum().item()),
                    }
                )

                if SAVE_PREDICTIONS:
                    pred_to_save = pred_prob[i, 0].cpu().numpy().astype(np.float16)
                    np.savez_compressed(
                        pred_dir / f"{batch['patch_id'][i]}.npz", pred=pred_to_save
                    )

    df_results = pd.DataFrame(records)
    df_results = df_results.merge(
        df_split[["patch_id", "region", "patch_type"]],
        on="patch_id",
        how="left",
    )
    return df_results


def report_metrics(df_results, split_name):
    """Log per-category/patch_type breakdown plus overall mean and micro-averaged metrics."""
    logging.info("=" * 70)
    logging.info(f"{split_name.upper()} SET METRICS")
    logging.info("=" * 70)

    # Negatives carry no category; show them as their own group instead of dropping.
    df_results = df_results.copy()
    df_results["category"] = df_results["category"].fillna(df_results["patch_type"])

    logging.info(f"\n--- {split_name}: By category x patch_type (mean per-patch) ---")
    agg_cat = df_results.groupby(["category", "patch_type"])[
        ["dice", "iou", "cldice", "precision", "recall", "f1"]
    ].mean()
    for line in agg_cat.to_string(float_format=lambda x: f"{x:.4f}").splitlines():
        logging.info(line)

    logging.info(f"\n--- {split_name}: Overall ---")
    logging.info(
        f"  Mean per-patch Dice:   {df_results['dice'].mean():.4f} +/- {df_results['dice'].std():.4f}"
    )
    logging.info(
        f"  Mean per-patch IoU:    {df_results['iou'].mean():.4f} +/- {df_results['iou'].std():.4f}"
    )
    logging.info(
        f"  Mean per-patch clDice: {df_results['cldice'].mean():.4f} +/- {df_results['cldice'].std():.4f}"
    )
    logging.info(
        f"  Mean per-patch F1:     {df_results['f1'].mean():.4f} +/- {df_results['f1'].std():.4f}"
    )

    total_tp = int(df_results["tp"].sum())
    total_fp = int(df_results["fp"].sum())
    total_fn = int(df_results["fn"].sum())
    total_tn = int(df_results["tn"].sum())

    micro_precision = total_tp / (total_tp + total_fp + 1e-6)
    micro_recall = total_tp / (total_tp + total_fn + 1e-6)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)
    )
    micro_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    micro_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUTPUT_DIR)

    log.info(f"Device: {DEVICE}")
    log.info(f"Architecture: {ARCHITECTURE}")
    log.info(f"Input channels ({N_INPUT_CHANNELS}): {INPUT_CHANNELS}")
    log.info(
        f"Hyperparams: batch={BATCH_SIZE}, epochs={N_EPOCHS}, lr={LR}, wd={WEIGHT_DECAY}"
    )
    log.info(f"num_workers={NUM_WORKERS}")
    log.info(f"Image dashboard dir: {IMG_DIR}")

    # 1. Load data and split (train/val only)
    df_train, df_val = load_and_split_metadata()

    # 2. Norm stats
    norm_stats = compute_or_load_norm_stats(df_train)

    # 3. Build dataloaders
    train_loader, val_loader, patches_root_dirs = build_dataloaders(
        df_train, df_val, norm_stats
    )

    # 3b. Fixed monitoring locations (always positive): 3 train + 3 val
    fixed_locations = build_fixed_locations(df_train, df_val, patches_root_dirs)
    for loc in fixed_locations:
        log.info(f"Fixed location: {loc['name']} -> patch_id={loc['patch_id']} ({loc['split']})")

    # 4. Build model
    model = build_model()

    # 5. Train
    log.info("Starting training...")
    history, checkpoint_path = train(
        model,
        train_loader,
        val_loader,
        norm_stats,
        fixed_locations,
        patches_root_dirs,
    )

    # 6. Plot curves
    plot_training_curves(history)
    save_final_summary_graph(history, IMG_DIR)

    # 7. Load best checkpoint and run a val-only sanity check
    log.info("Loading best checkpoint for val evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info(
        f"Best checkpoint from epoch {checkpoint['epoch']}: "
        f"val_score={checkpoint.get('val_score', float('nan')):.4f}, "
        f"val_dice={checkpoint['val_dice']:.4f}, "
        f"val_cldice={checkpoint.get('val_cldice', float('nan')):.4f}"
    )

    val_results = evaluate_split(model, df_val, "val", norm_stats, patches_root_dirs)
    val_results.to_csv(OUTPUT_DIR / "val_results_per_patch.csv", index=False)
    report_metrics(val_results, "val")

    if SAVE_PREDICTIONS:
        n_val = len(list((OUTPUT_DIR / "predictions_val").glob("*.npz")))
        log.info(f"Saved predictions: val={n_val} files")

    log.info("Done. Held-out basin is evaluated separately with evaluate_segformer.py.")


if __name__ == "__main__":
    main()
