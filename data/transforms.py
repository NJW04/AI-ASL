#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torchvision import transforms as T

from skimage.color import rgb2gray
from skimage.feature import hog


def get_normalize_stats(name: Optional[str] = "imagenet"):
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
    ops = [T.Resize(110), T.CenterCrop(size), T.ToTensor()]
    stats = get_normalize_stats(normalize)
    if stats is not None:
        ops.append(T.Normalize(*stats))
    return T.Compose(ops)


def extract_hog_color(pil_img: Image.Image) -> np.ndarray:
    """
    Extract simple HOG (grayscale) + color histogram (RGB) features.
    - HOG: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), L2-Hys
    - Color histogram: 16 bins per channel, range [0,1], concatenated (48 dims)

    Returns:
      np.ndarray of shape (D,)
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).astype(np.float32) / 255.0  # (H,W,3)
    gray = rgb2gray(arr)  # (H,W)
    # skimage>=0.19 uses channel_axis instead of multichannel. We'll try to be robust:
    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    # Color hist per channel
    hist_list = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 1.0), density=True)
        hist_list.append(hist.astype(np.float32))
    color_hist = np.concatenate(hist_list, axis=0)
    feat = np.concatenate([hog_vec.astype(np.float32), color_hist], axis=0)
    return feat
