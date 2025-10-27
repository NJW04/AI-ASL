#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train script for ASL Alphabet:
- From-scratch CNN (default) or MobileNetV3-Small head (--model mobilenet [--transfer])
- Early stopping on val macro-F1
- Saves run folder with readable name and all artifacts
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.asl import ensure_data, make_split_lists, build_dataloaders, summarize_class_distribution
from models.cnn_small import CNNSmall
from models.mobilenet_head import MobileNetV3SmallHead
from utils.io import create_run_dir, write_json, ensure_dir, copy_to
from utils.log import get_logger, log_banner
from utils.metrics import compute_metrics, plot_confusion
from utils.seed import set_seed


def build_model(model_name: str, num_classes: int, dropout: float, transfer: bool):
    if model_name == "cnn_small":
        return CNNSmall(num_classes=num_classes, base_channels=32, dropout=dropout)
    elif model_name == "mobilenet":
        return MobileNetV3SmallHead(num_classes=num_classes, transfer=transfer)
    else:
        raise ValueError(f"Unknown model: {model_name}")


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

    # Route any torchvision cache away from ~/.cache to artifacts (to satisfy "no .cache" usage)
    os.environ.setdefault("TORCH_HOME", str(Path(args.artifacts_root) / "torch_home"))

    # Ensure data present and class mapping
    train_dir, test_dir = ensure_data(Path("data"), use_kaggle=args.use_kaggle)

    # Create split
    split_json = make_split_lists(args.train_dir, val_ratio=args.val_ratio,
                                  seed=args.seed, subset_per_class=args.subset_per_class)

    # Create run dir
    slug = f"train-{args.model}_small-lr{args.lr:g}-b{args.batch_size}-ep{args.epochs}"
    run_dir = create_run_dir(Path(args.artifacts_root), slug)
    ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "TRAIN")

    # Summaries
    dist = summarize_class_distribution(split_json)
    logger.info("Args: %s", vars(args))
    logger.info("Class distribution (train): %s", dist["train"])
    logger.info("Class distribution (val):   %s", dist["val"])

    # Dataloaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.train_dir, args.test_dir, split_json, batch_size=args.batch_size,
        num_workers=args.num_workers, size=args.size, aug=args.aug
    )
    write_json({c: i for i, c in enumerate(class_names)}, run_dir / "class_indices.json")

    # Model
    model = build_model(args.model, num_classes=len(class_names), dropout=args.dropout, transfer=args.transfer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model: %s", model.__class__.__name__)
    logger.info("Device: %s", device)

    if args.model == "mobilenet" and args.transfer and hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True)
        logger.info("Backbone frozen for first %d epoch(s).", args.freeze_backbone_epochs)

    # Class weights (optional)
    class_weights = None
    if args.class_weights:
        # Count from training data
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

    # Logs
    train_csv = (run_dir / "train_log.csv").open("w")
    val_csv = (run_dir / "val_log.csv").open("w")
    print("epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1", file=train_csv)
    print("epoch,val_macro_f1", file=val_csv)

    best = {"macro_f1": -1.0, "epoch": -1}
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for x, y, _ in train_loader:
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
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Unfreeze after freeze_backbone_epochs (if applicable)
        if args.model == "mobilenet" and args.transfer and epoch == args.freeze_backbone_epochs and hasattr(model, "freeze_backbone"):
            model.freeze_backbone(False)
            # Recreate optimizer to include new params if any
            optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                             lr=args.lr, weight_decay=args.weight_decay)
            logger.info("Backbone unfrozen.")

        # Validation
        val_loss, _, y_true, y_pred, _, _ = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, class_names)
        val_acc, val_macro_f1 = metrics["accuracy"], metrics["macro_f1"]
        logger.info("Train: loss=%.4f acc=%.4f | Val: loss=%.4f acc=%.4f macroF1=%.4f",
                    train_loss, train_acc, val_loss, val_acc, val_macro_f1)

        print(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_macro_f1:.6f}", file=train_csv)
        print(f"{epoch},{val_macro_f1:.6f}", file=val_csv)
        train_csv.flush()
        val_csv.flush()

        # Early stopping on macro-F1
        if val_macro_f1 > best["macro_f1"]:
            best.update({"macro_f1": val_macro_f1, "epoch": epoch})
            # Save best
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_json({"epoch": epoch, "macro_f1": val_macro_f1}, run_dir / "best.json")
            # Confusion matrix for val
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

    # Save final metadata
    cfg = vars(args).copy()
    cfg["split_json"] = str(Path("splits") / f"asl_val_split_seed{args.seed}_r{int(args.val_ratio*100)}.json")
    write_json(cfg, run_dir / "config.json")

    logger.info("Training completed. Artifacts at: %s", run_dir)
    return {"run_dir": str(run_dir), "best_macro_f1": best["macro_f1"], "best_epoch": best["epoch"]}


def parse_args():
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
    ap.add_argument("--model", type=str, choices=["cnn_small", "mobilenet"], default="cnn_small")
    ap.add_argument("--transfer", action="store_true", help="Use pretrained weights (mobilenet only)")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3, help="Epochs to freeze backbone when --transfer")
    ap.add_argument("--aug", action="store_true", help="Enable data augmentation")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
