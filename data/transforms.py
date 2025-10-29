#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image transforms and HOG+color feature extraction utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torchvision import transforms as T

from skimage.color import rgb2gray
from skimage.feature import hog


def get_normalize_stats(name: Optional[str] = "imagenet"):
    """Return (mean, std) for a known normalization preset.

    Parameters
    ----------
    name : Optional[str]
        Preset name. Supported: "imagenet". If None or unknown, returns None.

    Returns
    -------
    Optional[tuple[list[float], list[float]]]
        (mean, std) in RGB order, each in [0, 1], or None if not applicable.
    """
    if name is None:
        return None
    name = name.lower()
    if name == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return mean, std
    else:
        return None


def get_train_transforms(aug: bool = True, size: int = 96, normalize: str = "imagenet"):
    """Build torchvision transforms for training.

    Parameters
    ----------
    aug : bool
        Whether to include data augmentation.
    size : int
        Output crop size (square).
    normalize : str
        Normalization preset passed to `get_normalize_stats`.

    Returns
    -------
    torchvision.transforms.Compose
        Composed training transform pipeline.
    """
    ops = [T.Resize(110)]
    if aug:
        ops.extend([
            T.RandomResizedCrop(size, scale=(0.9, 1.0)),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            T.RandomHorizontalFlip(p=0.5),
        ])
    else:
        ops.extend([T.CenterCrop(size)])
    ops.append(T.ToTensor())
    stats = get_normalize_stats(normalize)
    if stats is not None:
        ops.append(T.Normalize(*stats))
    return T.Compose(ops)


def get_eval_transforms(size: int = 96, normalize: str = "imagenet"):
    """Build torchvision transforms for evaluation/inference.

    Parameters
    ----------
    size : int
        Output crop size (square).
    normalize : str
        Normalization preset passed to `get_normalize_stats`.

    Returns
    -------
    torchvision.transforms.Compose
        Composed evaluation transform pipeline.
    """
    ops = [T.Resize(110), T.CenterCrop(size), T.ToTensor()]
    stats = get_normalize_stats(normalize)
    if stats is not None:
        ops.append(T.Normalize(*stats))
    return T.Compose(ops)


def extract_hog_color(pil_img: Image.Image) -> np.ndarray:
    """Extract HOG (on grayscale) + RGB color histogram features.

    HOG parameters
    --------------
    orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
    block_norm="L2-Hys", feature_vector=True

    Color histogram
    ---------------
    16 bins per RGB channel in [0, 1], concatenated (48 dims).

    Parameters
    ----------
    pil_img : PIL.Image.Image
        Input PIL image. Converted to RGB if needed.

    Returns
    -------
    numpy.ndarray
        1D feature vector: `[hog_features, color_hist]`.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).astype(np.float32) / 255.0  # (H, W, 3)
    gray = rgb2gray(arr)  # (H, W)
    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    hist_list = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 1.0), density=True)
        hist_list.append(hist.astype(np.float32))
    color_hist = np.concatenate(hist_list, axis=0)
    feat = np.concatenate([hog_vec.astype(np.float32), color_hist], axis=0)
    return feat
