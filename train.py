#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train script for ASL Alphabet:
- From-scratch CNN (configurable blocks/activation)
- Early stopping on val macro-F1
- Saves run folder with readable name and all artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import datetime as _dt
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import tqdm globally, handle missing case later
try:
    import tqdm
except ImportError:
    tqdm = None # Define tqdm as None if not installed

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data.asl import (
    ensure_data,
    make_split_lists,
    build_dataloaders,
    summarize_class_distribution,
)
from models.cnn_small import CNNSmall # Make sure this import is correct


# =========================
# Inlined utility helpers
# =========================

# ---- io helpers ----
def _ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _slugify(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text.strip().lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")

def _timestamp_slug(slug: str) -> str:
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}__{_slugify(slug)}" if slug else ts

def create_run_dir(artifacts_root: Path, slug: str) -> Path:
    """
    Create a timestamped run directory under artifacts_root.
    Example: artifacts/asl_runs/2025-10-27_14-05-33__train-cnn_small-lr0.001-b64-ep15
    """
    run_dir = Path(artifacts_root) / _timestamp_slug(slug)
    _ensure_dir(run_dir)
    return run_dir

def write_json(data: Dict[str, Any], path: Path, indent: int = 2):
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def copy_to(src: Path, dst: Path):
    dst = Path(dst)
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)

# ---- logging helpers ----
def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    """
    Logger writing to console and <run_dir>/run.log.
    Prevents duplicate handlers for the same run_dir/name.
    """
    run_dir = Path(run_dir)
    _ensure_dir(run_dir)
    logger_name = f"{run_dir.resolve()}/{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)

        # file
        fh = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    return logger

def log_banner(logger: logging.Logger, title: str):
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)

# ---- metrics & plotting ----
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_rec = recall_score(y_true, y_pred, average=None).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_rec,
        "labels": labels,
    }

def _save_figure(out_path: Path, dpi: int = 160):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(out_path)

# ---- seeding ----
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =========================
# Original training logic
# =========================

# --- MODIFIED to accept new args ---
def build_model(model_name: str, num_classes: int, dropout: float,
                num_blocks: int = 3, activation: str = "relu"):
    if model_name != "cnn_small":
        raise ValueError("Only cnn_small is supported.")
    # Pass the arguments to the CNNSmall constructor
    return CNNSmall(num_classes=num_classes, base_channels=32, dropout=dropout,
                    num_blocks=num_blocks, activation=activation)
# --- END MODIFICATION ---

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    preds, gts, paths = [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            preds.append(pred.cpu().numpy())
            gts.append(y.cpu().numpy())
            paths.extend(p)
    y_true = np.concatenate(gts, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, 0.0, y_true, y_pred, paths, []


def run_training(args) -> Dict:
    set_seed(args.seed, deterministic=True)

    os.environ.setdefault("TORCH_HOME", str(Path(args.artifacts_root) / "torch_home"))

    train_dir, test_dir = ensure_data(Path("data"), use_kaggle=args.use_kaggle)

    split_json = make_split_lists(args.train_dir, val_ratio=args.val_ratio,
                                  seed=args.seed, subset_per_class=args.subset_per_class)

    # --- UPDATED SLUG to potentially include new params ---
    slug_parts = [
        f"train-{args.model}",
        f"lr{args.lr:g}",
        f"b{args.batch_size}",
        f"bl{args.num_blocks}",
        f"act-{args.activation}",
        f"dr{args.dropout:g}",
        f"ep{args.epochs}"
    ]
    if args.aug: slug_parts.append("aug")
    if args.size != 96: slug_parts.append(f"sz{args.size}")
    if args.subset_per_class: slug_parts.append(f"sub{args.subset_per_class}")
    slug = "_".join(slug_parts)
    # --- END UPDATE ---

    run_dir = create_run_dir(Path(args.artifacts_root), slug)
    _ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "TRAIN")

    dist = summarize_class_distribution(split_json)
    logger.info("Args: %s", vars(args))
    logger.info("Class distribution (train): %s", dist["train"])
    logger.info("Class distribution (val):   %s", dist["val"])

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.train_dir, args.test_dir, split_json, batch_size=args.batch_size,
        num_workers=args.num_workers, size=args.size, aug=args.aug
    )
    write_json({c: i for i, c in enumerate(class_names)}, run_dir / "class_indices.json")

    # --- MODIFIED to pass new args ---
    model = build_model(args.model, num_classes=len(class_names), dropout=args.dropout,
                        num_blocks=args.num_blocks, activation=args.activation)
    # --- END MODIFICATION ---

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    logger.info("Model: %s (blocks=%d, act=%s)",
                model.__class__.__name__, args.num_blocks, args.activation) # Updated log
    logger.info("Device: %s", device)

    class_weights = None
    if args.class_weights:
        labels = []
        for _, y, _ in train_loader:
            labels.extend(y.tolist())
        counts = np.bincount(np.array(labels), minlength=len(class_names))
        inv = counts.sum() / np.maximum(counts, 1)
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        logger.info("Using class weights: %s", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=args.lr, weight_decay=args.weight_decay)

    train_csv = (run_dir / "train_log.csv").open("w")
    val_csv = (run_dir / "val_log.csv").open("w")
    print("epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1", file=train_csv)
    print("epoch,val_macro_f1", file=val_csv)

    best = {"macro_f1": -1.0, "epoch": -1}
    patience_counter = 0

    # Define a dummy tqdm class if tqdm is not installed
    tqdm_iterator = tqdm.tqdm if tqdm else lambda x, **kwargs: x

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        # Simple progress bar for training
        pbar = tqdm_iterator(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
            # Update progress bar description (only if tqdm exists)
            if tqdm:
                pbar.set_postfix({"loss": f"{total_loss / total:.4f}", "acc": f"{correct / total:.4f}"})
        if tqdm: pbar.close() # Close progress bar if tqdm exists

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, _, y_true, y_pred, _, _ = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, class_names)
        val_acc, val_macro_f1 = metrics["accuracy"], metrics["macro_f1"]
        logger.info("Train: loss=%.4f acc=%.4f | Val: loss=%.4f acc=%.4f macroF1=%.4f",
                    train_loss, train_acc, val_loss, val_acc, val_macro_f1)

        print(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_macro_f1:.6f}", file=train_csv)
        print(f"{epoch},{val_macro_f1:.6f}", file=val_csv)
        train_csv.flush()
        val_csv.flush()

        if val_macro_f1 > best["macro_f1"]:
            best.update({"macro_f1": val_macro_f1, "epoch": epoch})
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_json({"epoch": epoch, "macro_f1": val_macro_f1}, run_dir / "best.json")
            plot_confusion(y_true, y_pred, class_names, out_path=run_dir / "confmat_val.png",
                           title=f"Val Confusion (epoch {epoch})")
            patience_counter = 0
            logger.info("New best macro-F1 %.4f at epoch %d. Checkpoint saved.", val_macro_f1, epoch)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d patience).", patience_counter, args.patience)
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    train_csv.close()
    val_csv.close()

    # --- UPDATED CONFIG SAVING ---
    cfg = vars(args).copy()
    # Convert Path objects to strings for JSON
    for k, v in cfg.items():
        if isinstance(v, Path):
            cfg[k] = str(v)
    cfg["split_json"] = str(split_json) # Ensure split_json path is saved
    write_json(cfg, run_dir / "config.json")
    # --- END UPDATE ---

    logger.info("Training completed. Artifacts at: %s", run_dir)
    return {"run_dir": str(run_dir), "best_macro_f1": best["macro_f1"], "best_epoch": best["epoch"]}


# --- MODIFIED to accept args_list ---
def parse_args(args_list=None): # Add args_list=None
    ap = argparse.ArgumentParser(description="Train a CNN for ASL Alphabet (29 classes)")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--model", type=str, choices=["cnn_small"], default="cnn_small")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.3)

    # --- ADDED NEW ARGUMENTS ---
    ap.add_argument("--num-blocks", type=int, default=3, help="Number of conv blocks in CNNSmall")
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation function")
    # --- END OF NEW ARGUMENTS ---

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))

    # Pass the list to the internal parse_args method
    return ap.parse_args(args_list)
# --- END MODIFICATION ---

if __name__ == "__main__":
    # Check if tqdm is available before using it
    if tqdm is None:
        print("tqdm not found, install it for progress bars: pip install tqdm")
        # Define a dummy tqdm class if not installed
        class tqdm:
            def __init__(self, iterable=None, *args, **kwargs): self.iterable = iterable
            def __iter__(self): return iter(self.iterable)
            def set_postfix(self, *args, **kwargs): pass
            def close(self): pass

    args = parse_args() # Parses command-line args when run directly
    run_training(args)