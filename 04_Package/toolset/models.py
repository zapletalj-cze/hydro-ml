"""
Model tools: dataset, normalization, losses, architecture builders and
first-conv channel adaptation.

Author: Jakub Zapletal
Date:   2026-04-04
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import segmentation_models_pytorch as smp


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_channel_dict(channels, input_channels, norm_stats, water_channel="water"):
    """Normalize a dict of channel arrays: DSM per-patch median subtraction,
    binary water kept 0/1, everything else z-scored with the training stats."""
    out = {}
    for ch_name in input_channels:
        arr = channels[ch_name].astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if ch_name == "dsm":
            arr = arr - np.median(arr)
        elif ch_name == water_channel:
            pass  # binary mask, leave as 0/1
        else:
            stats = norm_stats[ch_name]
            arr = (arr - stats["mean"]) / (stats["std"] + 1e-6)
        out[ch_name] = arr
    return out


def normalize_patch_array(patch, norm_stats, zscore_channel_names):
    """Normalize a stacked (C, H, W) patch: channel 0 is DSM (per-patch median),
    channels 1..n are z-scored in the order of zscore_channel_names, the rest
    (e.g. the binary water mask) stay untouched."""
    out = patch.copy()
    out[0] = out[0] - np.median(out[0])
    for i, name in enumerate(zscore_channel_names, start=1):
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        out[i] = (out[i] - mean) / (std + 1e-8)
    return out


def compute_norm_stats(df, patches_root_dirs, input_channels, exclude=("dsm", "water")):
    """Streaming per-channel mean/std over the given patches. Channels in
    'exclude' get sentinel stats (mean 0, std 1)."""
    channels_to_normalize = [c for c in input_channels if c not in exclude]
    sums = {c: 0.0 for c in channels_to_normalize}
    sq_sums = {c: 0.0 for c in channels_to_normalize}
    counts = {c: 0 for c in channels_to_normalize}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing norm stats"):
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

    for c in exclude:
        if c in input_channels:
            norm_stats[c] = {"mean": 0.0, "std": 1.0}  # sentinel

    return norm_stats


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
        self,
        metadata_df,
        patches_root_dirs,
        input_channels,
        norm_stats,
        label_channel="label",
        water_channel="water",
        augment=False,
        flip_h=True,
        flip_v=True,
        rot90=True,
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.patches_root_dirs = patches_root_dirs
        self.input_channels = input_channels
        self.norm_stats = norm_stats
        self.label_channel = label_channel
        self.water_channel = water_channel
        self.augment = augment
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rot90 = rot90

    def __len__(self):
        return len(self.metadata)

    def _load_npz(self, row):
        region = row["region"]
        npz_path = self.patches_root_dirs[region] / f"{row['patch_id']}.npz"
        return dict(np.load(npz_path))

    def _augment(self, image, label):
        if self.rot90:
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                label = np.rot90(label, k=k, axes=(1, 2)).copy()
        if self.flip_h and np.random.rand() < 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        if self.flip_v and np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return image, label

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        channels = self._load_npz(row)
        channels_norm = normalize_channel_dict(
            channels, self.input_channels, self.norm_stats, self.water_channel
        )

        image = np.stack([channels_norm[c] for c in self.input_channels], axis=0)
        label = channels[self.label_channel].astype(np.float32)[np.newaxis, ...]

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
# LOSSES
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


def combined_loss(pred, target, bce_pos_weight, dice_weight, cldice_iter):
    """BCE + dice_weight * Dice + (1 - dice_weight) * clDice."""
    pw = torch.tensor(bce_pos_weight, device=pred.device, dtype=pred.dtype)
    l_bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    l_dice = dice_loss(pred, target)
    l_cldice = cldice_loss(pred, target, n_iter=cldice_iter)
    return l_bce + dice_weight * l_dice + (1 - dice_weight) * l_cldice


# ============================================================
# ARCHITECTURES
# ============================================================


def adapt_first_conv_resnet(model, n_input_channels):
    """Replicate 3-channel pretrained first-conv weights for N-channel input
    (ResNet-family encoders)."""
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
    """Replicate 3-channel pretrained patch_embed1.proj weights for N channels."""
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


def build_model(
    architecture,
    backbone,
    encoder_weights,
    n_input_channels,
    decoder_dropout_p=0.1,
    device="cpu",
):
    """Build a segmentation model with first-conv channel adaptation and
    decoder dropout. architecture: resnet_unet | segformer | deeplabv3plus."""
    if architecture == "resnet_unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_resnet(model, n_input_channels)

    elif architecture == "segformer":
        model = smp.Segformer(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_segformer(model, n_input_channels)

    elif architecture == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model = adapt_first_conv_resnet(model, n_input_channels)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if architecture == "resnet_unet":
        for block in model.decoder.blocks:
            block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=decoder_dropout_p))
        logging.info(
            f"Decoder dropout: {decoder_dropout_p} (applied to {len(model.decoder.blocks)} U-Net blocks)"
        )
    elif architecture == "deeplabv3plus":
        if hasattr(model.decoder, "block2"):
            model.decoder.block2 = nn.Sequential(
                model.decoder.block2, nn.Dropout2d(p=decoder_dropout_p)
            )
            logging.info(
                f"Decoder dropout: {decoder_dropout_p} (applied to DeepLabV3+ block2)"
            )
        else:
            logging.warning("DeepLabV3+ decoder.block2 not found - skipping dropout")
    elif architecture == "segformer":
        logging.info("No decoder dropout for SegFormer (lightweight MLP head)")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Architecture: {architecture}")
    logging.info(f"Trainable parameters: {n_params:,}")
    return model


def load_segformer_checkpoint(checkpoint_path, backbone, n_input_channels, device):
    """Build a SegFormer with adapted input channels and load a checkpoint."""
    model = smp.Segformer(
        encoder_name=backbone,
        encoder_weights=None,  # weights come from the checkpoint
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = adapt_first_conv_segformer(model, n_input_channels)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint
