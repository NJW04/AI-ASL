#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASL Alphabet dataset utilities:
- ensure_data: optional Kaggle download and structure validation
- make_split_lists: stratified train/val/test split using index files (no moves)
- build_dataloaders: DataLoaders for train/val/test with torchvision transforms
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image  # noqa: F401 (used indirectly by default_loader)
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

from .transforms import get_train_transforms, get_eval_transforms

# =========================
# Inlined utility helpers
# (No changes in this section)
# =========================

def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_json(data: Dict, path: Path, indent: int = 2):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    """
    Logger writing to console and <run_dir>/run.log (if run_dir is writable).
    Prevents duplicate handlers for the same run_dir/name.
    """
    run_dir = Path(run_dir)
    try:
        ensure_dir(run_dir)
        logfile = run_dir / "run.log"
    except Exception:
        logfile = None  # fallback to console-only if path invalid

    logger_name = f"{run_dir.resolve()}/{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)

        if logfile is not None:
            try:
                fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
                logger.addHandler(fh)
            except Exception:
                # If file handler fails (e.g., permissions), we still have console logging
                pass

    return logger

def log_banner(logger: logging.Logger, title: str):
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_data(root_dir: Path,
                use_kaggle: bool = False,
                kaggle_dataset: str = "grassknoted/asl-alphabet",
                logger=None) -> Tuple[Path, Path]:
    """
    Ensure dataset is present under ./data/:
      ./data/asl_alphabet_train/ (29 subfolders)
      ./data/asl_alphabet_test/
    If use_kaggle=True, attempt Kaggle CLI download & unzip into root_dir.

    Returns:
      (train_dir, test_dir) as Paths

    Also writes ./data/class_indices.json with sorted class->index mapping
    based on the train directory structure (subfolder names).
    """
    root_dir = Path(root_dir)
    ensure_dir(root_dir)
    logger = logger or get_logger(root_dir, name="ensure_data")
    log_banner(logger, "ENSURE DATA")

    train_dir = root_dir / "asl_alphabet_train"
    test_dir = root_dir / "asl_alphabet_test"

    if use_kaggle and (not train_dir.exists() or not test_dir.exists()):
        logger.info("Attempting Kaggle download (dataset: %s)...", kaggle_dataset)
        # Users must provide Kaggle token. We just invoke CLI.
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", kaggle_dataset,
                "-p", str(root_dir)
            ]
            subprocess.run(cmd, check=True)
            # Unzip the largest zip in root_dir (dataset)
            zips = sorted(root_dir.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
            if not zips:
                raise RuntimeError("No zip file found after kaggle download.")
            zip_path = zips[0]
            logger.info("Unzipping %s ...", zip_path.name)
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(root_dir)], check=True)
            # Try to identify train/test directories if nested
            candidates = list(root_dir.rglob("asl_alphabet_train"))
            if candidates:
                src_train = candidates[0]
                if src_train != train_dir:
                    logger.info("Placing %s -> %s", src_train, train_dir)
                    if not train_dir.exists():
                        src_train.rename(train_dir)
            candidates = list(root_dir.rglob("asl_alphabet_test"))
            if candidates:
                src_test = candidates[0]
                if src_test != test_dir:
                    logger.info("Placing %s -> %s", src_test, test_dir)
                    if not test_dir.exists():
                        src_test.rename(test_dir)
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Install with `pip install kaggle`.")
        except subprocess.CalledProcessError as e:
            logger.error("Kaggle download/unzip failed: %s", e)

    # Validate structure
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found at {train_dir}. "
                                "Place dataset under ./data/ or use --use-kaggle.")
    # --- MODIFICATION: test_dir is no longer required to exist ---
    if not test_dir.exists():
        logger.warning("Test folder not found at %s.", test_dir)
        logger.warning("This is OK, as the test set will be built from the split file.")
    # --- END MODIFICATION ---

    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(classes) != 29:
        logger.warning("Expected 29 classes; found %d. Classes: %s", len(classes), classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    write_json(class_to_idx, root_dir / "class_indices.json")
    logger.info("Wrote class mapping to %s", root_dir / "class_indices.json")
    return train_dir, test_dir


def _list_images_per_class(train_dir: Path) -> Dict[str, List[Path]]:
    per_class: Dict[str, List[Path]] = {}
    for cdir in sorted([p for p in Path(train_dir).iterdir() if p.is_dir()]):
        imgs = [p for p in cdir.rglob("*") if _is_image(p)]
        if imgs:
            per_class[cdir.name] = sorted(imgs)
    return per_class


# --- MODIFICATION: Function updated to create train/val/test splits ---
def make_split_lists(train_dir: Path,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1, # <-- NEW argument
                     seed: int = 42,
                     subset_per_class: Optional[int] = None) -> Path:
    """
    Build stratified split lists (train/val/test) with per-class stratification.
    Does not move files.
    Saves to ./splits/asl_split_s{seed}_v{val*100}_t{test*100}.json

    If subset_per_class is set, sample up to N examples per class **after**
    stratification (applied independently on train and val).
    The test set is *never* subsetted.
    """
    train_dir = Path(train_dir)
    assert train_dir.exists(), f"{train_dir} does not exist."
    splits_dir = Path("splits")
    ensure_dir(splits_dir)

    per_class = _list_images_per_class(train_dir)
    classes = sorted(per_class.keys())
    random.seed(seed)

    train_list: List[Dict[str, str]] = []
    val_list: List[Dict[str, str]] = []
    test_list: List[Dict[str, str]] = [] # <-- NEW

    for cls in classes:
        imgs = per_class[cls]
        random.shuffle(imgs)
        n = len(imgs)
        
        # Calculate split indices
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))

        # Carve out the lists
        test_imgs = imgs[:n_test]
        val_imgs = imgs[n_test : n_test + n_val]
        train_imgs = imgs[n_test + n_val :]

        # Handle subsetting (Test set is NEVER subsetted)
        if subset_per_class is not None:
            train_imgs = train_imgs[:subset_per_class]
            val_imgs = val_imgs[:min(len(val_imgs), subset_per_class)]

        for p in train_imgs:
            train_list.append({"path": str(p), "label": cls})
        for p in val_imgs:
            val_list.append({"path": str(p), "label": cls})
        for p in test_imgs: # <-- NEW
            test_list.append({"path": str(p), "label": cls})

    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio, # <-- NEW
        "subset_per_class": subset_per_class,
        "class_names": classes,
        "train": train_list,
        "val": val_list,
        "test": test_list, # <-- NEW
    }
    # --- NEW file name ---
    split_path = splits_dir / f"asl_split_s{seed}_v{int(val_ratio*100)}_t{int(test_ratio*100)}.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    return split_path
# --- END MODIFICATION ---


@dataclass
class Item:
    path: str
    label: int


class ASLPathsDataset(Dataset):
    def __init__(self,
                 items: Sequence[Item],
                 transform=None):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = default_loader(it.path)  # PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, it.label, it.path


# --- MODIFICATION: Refactored to load from data, not file ---
def _load_split_items_from_data(split_data: Dict, phase: str, class_to_idx: Dict[str, int]) -> List[Item]:
    """
    Loads a list of Items (path, label_id) from a pre-loaded
    split dictionary for a given phase ('train', 'val', or 'test').
    """
    items: List[Item] = []
    if phase not in split_data:
        return [] # Return empty list if phase (e.g., "test") doesn't exist
        
    for rec in split_data[phase]:
        p = Path(rec["path"])
        if not p.exists():
            # skip missing
            continue
        label_idx = class_to_idx[rec["label"]]
        items.append(Item(path=str(p), label=label_idx))
    return items
# --- END MODIFICATION ---


def _infer_label_from_name(path: Path, classes: Sequence[str]) -> Optional[str]:
    """
    Best-effort: if test images are single files with class name in the filename,
    try to map. We match by substring against known class names.
    (Used by _build_test_items fallback)
    """
    name = path.stem.lower()
    for cls in classes:
        if cls.lower() in name:
            return cls
    # Also consider first token before non-letters
    m = re.split(r"[^A-Za-z]+", path.stem)
    if m:
        token = m[0].upper()
        if token in classes:
            return token
    return None


def _build_test_items(test_dir: Path, classes: Sequence[str]) -> List[Item]:
    """
    Fallback function to load test items from the physical `test_dir`.
    This is kept for backward compatibility or for cases where
    no 'test' split is defined in the JSON.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Item] = []
    if not test_dir.exists():
        return []
        
    if any(p.is_dir() for p in test_dir.iterdir()):
        # Foldered test set
        for cdir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
            label = cdir.name
            if label not in class_to_idx:
                continue
            for p in cdir.rglob("*"):
                if _is_image(p):
                    items.append(Item(path=str(p), label=class_to_idx[label]))
    else:
        # Flat images, try to infer labels from names
        for p in sorted([p for p in test_dir.iterdir() if _is_image(p)]):
            maybe = _infer_label_from_name(p, classes)
            if maybe is None:
                # skip unknowns
                continue
            items.append(Item(path=str(p), label=class_to_idx[maybe]))
    return items


def build_dataloaders(train_dir: Path,
                      test_dir: Path,
                      split_json: Path,
                      batch_size: int = 64,
                      num_workers: int = 2,
                      size: int = 96,
                      aug: bool = True):
    """
    Returns:
      train_loader, val_loader, test_loader, class_names
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    with open(Path("data") / "class_indices.json", "r") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    logger = get_logger(Path("."), name="build_dataloaders")

    # --- MODIFICATION: Load all splits from the JSON file ---
    with open(split_json, "r") as f:
        split_data = json.load(f)

    train_items = _load_split_items_from_data(split_data, "train", class_to_idx)
    val_items = _load_split_items_from_data(split_data, "val", class_to_idx)
    
    # Try to load 'test' items from the split file first
    test_items = _load_split_items_from_data(split_data, "test", class_to_idx)

    if not test_items:
        # Fallback if "test" list was missing or empty
        logger.warning("No 'test' items found in split_json. Falling back to loading from test_dir: %s", test_dir)
        test_items = _build_test_items(test_dir, class_names)
    else:
        logger.info("Loaded %d 'test' items from split_json.", len(test_items))
    # --- END MODIFICATION ---

    logger.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_items), len(val_items), len(test_items))

    tr_tfms = get_train_transforms(aug=aug, size=size, normalize="imagenet")
    ev_tfms = get_eval_transforms(size=size, normalize="imagenet")

    train_ds = ASLPathsDataset(train_items, transform=tr_tfms)
    val_ds = ASLPathsDataset(val_items, transform=ev_tfms)
    test_ds = ASLPathsDataset(test_items, transform=ev_tfms) if test_items else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = device == "cuda"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, class_names


def summarize_class_distribution(split_json: Path) -> Dict[str, Dict[str, int]]:
    with open(split_json, "r") as f:
        data = json.load(f)
    
    # --- MODIFICATION: Add 'test' key if it exists ---
    counts = {
        "train": Counter([rec["label"] for rec in data["train"]]),
        "val": Counter([rec["label"] for rec in data["val"]])
    }
    summary = {
        "train": dict(counts["train"]),
        "val": dict(counts["val"])
    }
    
    if "test" in data and data["test"]:
        counts["test"] = Counter([rec["label"] for rec in data["test"]])
        summary["test"] = dict(counts["test"])
    
    return summary
    # --- END MODIFICATION ---