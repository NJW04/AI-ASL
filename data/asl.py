from __future__ import annotations
"""ASL dataset utilities: data ensuring, splits, datasets, and dataloaders."""

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

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

from .transforms import get_train_transforms, get_eval_transforms


def ensure_dir(path: Path):
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(data: Dict, path: Path, indent: int = 2):
    """Write a Python object as JSON to ``path`` with the given indentation."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    """Create a logger that logs to console and ``<run_dir>/run.log`` if writable.

    Prevents duplicate handlers for the same ``run_dir``/``name`` combination.

    Parameters
    ----------
    run_dir : Path
        Directory to write the log file into.
    name : str
        Logger name suffix.
    level : int
        Logging level (e.g., ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    run_dir = Path(run_dir)
    try:
        ensure_dir(run_dir)
        logfile = run_dir / "run.log"
    except Exception:
        logfile = None

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
                pass

    return logger


def log_banner(logger: logging.Logger, title: str):
    """Log a formatted banner with the provided title."""
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(p: Path) -> bool:
    """Return True if ``p`` is a file with a known image extension."""
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_data(
    root_dir: Path,
    use_kaggle: bool = False,
    kaggle_dataset: str = "grassknoted/asl-alphabet",
    logger=None,
) -> Tuple[Path, Path]:
    """Ensure the ASL dataset exists under ``root_dir``; optionally download via Kaggle.

    Expects the following structure:
      ``./data/asl_alphabet_train/`` (29 subfolders)
      ``./data/asl_alphabet_test/``

    Also writes ``./data/class_indices.json`` mapping class name -> index based on
    train subfolder names.

    Parameters
    ----------
    root_dir : Path
        Dataset root directory (e.g., ``./data``).
    use_kaggle : bool
        Whether to attempt Kaggle CLI download/unzip.
    kaggle_dataset : str
        Kaggle dataset identifier.
    logger : logging.Logger | None
        Logger for progress messages. If ``None``, a new one is created.

    Returns
    -------
    (Path, Path)
        Tuple of (``train_dir``, ``test_dir``).
    """
    root_dir = Path(root_dir)
    ensure_dir(root_dir)
    logger = logger or get_logger(root_dir, name="ensure_data")
    log_banner(logger, "ENSURE DATA")

    train_dir = root_dir / "asl_alphabet_train"
    test_dir = root_dir / "asl_alphabet_test"

    if use_kaggle and (not train_dir.exists() or not test_dir.exists()):
        logger.info("Attempting Kaggle download (dataset: %s)...", kaggle_dataset)
        try:
            cmd = ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(root_dir)]
            subprocess.run(cmd, check=True)
            zips = sorted(root_dir.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
            if not zips:
                raise RuntimeError("No zip file found after kaggle download.")
            zip_path = zips[0]
            logger.info("Unzipping %s ...", zip_path.name)
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(root_dir)], check=True)
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

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found at {train_dir}. "
            "Place dataset under ./data/ or use --use-kaggle."
        )
    if not test_dir.exists():
        logger.warning("Test folder not found at %s.", test_dir)
        logger.warning("This is OK, as the test set will be built from the split file.")

    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(classes) != 29:
        logger.warning("Expected 29 classes; found %d. Classes: %s", len(classes), classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    write_json(class_to_idx, root_dir / "class_indices.json")
    logger.info("Wrote class mapping to %s", root_dir / "class_indices.json")
    return train_dir, test_dir


def _list_images_per_class(train_dir: Path) -> Dict[str, List[Path]]:
    """Return a mapping from class name to a sorted list of image Paths."""
    per_class: Dict[str, List[Path]] = {}
    for cdir in sorted([p for p in Path(train_dir).iterdir() if p.is_dir()]):
        imgs = [p for p in cdir.rglob("*") if _is_image(p)]
        if imgs:
            per_class[cdir.name] = sorted(imgs)
    return per_class


def make_split_lists(
    train_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    subset_per_class: Optional[int] = None,
) -> Path:
    """Create stratified train/val/test JSON split lists without moving files.

    Saves to:
    ``./splits/asl_split_s{seed}_v{val*100}_t{test*100}.json``

    If ``subset_per_class`` is provided, subsets train and val per-class after
    stratification; test is never subsetted.

    Parameters
    ----------
    train_dir : Path
        Path to the training directory with class subfolders.
    val_ratio : float
        Fraction of each class to use for validation.
    test_ratio : float
        Fraction of each class to use for test.
    seed : int
        Random seed for shuffling.
    subset_per_class : Optional[int]
        Max number of samples per class for train/val after split.

    Returns
    -------
    Path
        Path to the written split JSON file.
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
    test_list: List[Dict[str, str]] = []

    for cls in classes:
        imgs = per_class[cls]
        random.shuffle(imgs)
        n = len(imgs)

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))

        test_imgs = imgs[:n_test]
        val_imgs = imgs[n_test: n_test + n_val]
        train_imgs = imgs[n_test + n_val:]

        if subset_per_class is not None:
            train_imgs = train_imgs[:subset_per_class]
            val_imgs = val_imgs[:min(len(val_imgs), subset_per_class)]

        for p in train_imgs:
            train_list.append({"path": str(p), "label": cls})
        for p in val_imgs:
            val_list.append({"path": str(p), "label": cls})
        for p in test_imgs:
            test_list.append({"path": str(p), "label": cls})

    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "subset_per_class": subset_per_class,
        "class_names": classes,
        "train": train_list,
        "val": val_list,
        "test": test_list,
    }
    split_path = splits_dir / f"asl_split_s{seed}_v{int(val_ratio*100)}_t{int(test_ratio*100)}.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    return split_path


@dataclass
class Item:
    """Single dataset record with path and integer label."""
    path: str
    label: int


class ASLPathsDataset(Dataset):
    """Dataset that loads images from stored file paths with optional transforms."""

    def __init__(self, items: Sequence[Item], transform=None):
        """Initialize with a sequence of ``Item`` and an optional transform."""
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        """Return dataset size."""
        return len(self.items)

    def __getitem__(self, idx):
        """Load and return ``(image, label, path)`` for index ``idx``."""
        it = self.items[idx]
        img = default_loader(it.path)  # PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, it.label, it.path


def _load_split_items_from_data(split_data: Dict, phase: str, class_to_idx: Dict[str, int]) -> List[Item]:
    """Load items for ``phase`` (train/val/test) from an in-memory split dict."""
    items: List[Item] = []
    if phase not in split_data:
        return []
    for rec in split_data[phase]:
        p = Path(rec["path"])
        if not p.exists():
            continue
        label_idx = class_to_idx[rec["label"]]
        items.append(Item(path=str(p), label=label_idx))
    return items


def _infer_label_from_name(path: Path, classes: Sequence[str]) -> Optional[str]:
    """Infer a class label from a filename by substring/token match, if possible."""
    name = path.stem.lower()
    for cls in classes:
        if cls.lower() in name:
            return cls
    m = re.split(r"[^A-Za-z]+", path.stem)
    if m:
        token = m[0].upper()
        if token in classes:
            return token
    return None


def _build_test_items(test_dir: Path, classes: Sequence[str]) -> List[Item]:
    """Fallback loader for test items from ``test_dir`` if split JSON lacks 'test'."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Item] = []
    if not test_dir.exists():
        return []

    if any(p.is_dir() for p in test_dir.iterdir()):
        for cdir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
            label = cdir.name
            if label not in class_to_idx:
                continue
            for p in cdir.rglob("*"):
                if _is_image(p):
                    items.append(Item(path=str(p), label=class_to_idx[label]))
    else:
        for p in sorted([p for p in test_dir.iterdir() if _is_image(p)]):
            maybe = _infer_label_from_name(p, classes)
            if maybe is None:
                continue
            items.append(Item(path=str(p), label=class_to_idx[maybe]))
    return items


def build_dataloaders(
    train_dir: Path,
    test_dir: Path,
    split_json: Path,
    batch_size: int = 64,
    num_workers: int = 2,
    size: int = 96,
    aug: bool = True,
):
    """Build PyTorch dataloaders for train/val/test based on a split JSON.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, test_loader, class_names)``, where
        ``test_loader`` may be ``None`` if no test items are available.
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    with open(Path("data") / "class_indices.json", "r") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    logger = get_logger(Path("."), name="build_dataloaders")

    with open(split_json, "r") as f:
        split_data = json.load(f)

    train_items = _load_split_items_from_data(split_data, "train", class_to_idx)
    val_items = _load_split_items_from_data(split_data, "val", class_to_idx)
    test_items = _load_split_items_from_data(split_data, "test", class_to_idx)

    if not test_items:
        logger.warning("No 'test' items found in split_json. Falling back to loading from test_dir: %s", test_dir)
        test_items = _build_test_items(test_dir, class_names)
    else:
        logger.info("Loaded %d 'test' items from split_json.", len(test_items))

    logger.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_items), len(val_items), len(test_items))

    tr_tfms = get_train_transforms(aug=aug, size=size, normalize="imagenet")
    ev_tfms = get_eval_transforms(size=size, normalize="imagenet")

    train_ds = ASLPathsDataset(train_items, transform=tr_tfms)
    val_ds = ASLPathsDataset(val_items, transform=ev_tfms)
    test_ds = ASLPathsDataset(test_items, transform=ev_tfms) if test_items else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = device == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
        )
    return train_loader, val_loader, test_loader, class_names


def summarize_class_distribution(split_json: Path) -> Dict[str, Dict[str, int]]:
    """Summarize per-class counts for train/val/test (if present) from a split JSON.

    Parameters
    ----------
    split_json : Path
        Path to the split JSON file.

    Returns
    -------
    dict
        Mapping of split name to a dict of label -> count.
    """
    with open(split_json, "r") as f:
        data = json.load(f)

    counts = {
        "train": Counter([rec["label"] for rec in data["train"]]),
        "val": Counter([rec["label"] for rec in data["val"]]),
    }
    summary = {
        "train": dict(counts["train"]),
        "val": dict(counts["val"]),
    }

    if "test" in data and data["test"]:
        counts["test"] = Counter([rec["label"] for rec in data["test"]])
        summary["test"] = dict(counts["test"])

    return summary
