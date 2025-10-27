#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a saved checkpoint on val/test splits.
Saves:
- metrics_val.json / metrics_test.json
- predictions_val.csv / predictions_test.csv (path,true,pred,top3)
- confmat_val.png / confmat_test.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from data.asl import build_dataloaders
from models.cnn_small import CNNSmall
from models.mobilenet_head import MobileNetV3SmallHead
from utils.io import create_run_dir, write_json
from utils.log import get_logger, log_banner
from utils.metrics import compute_metrics, plot_confusion


def _infer_model_type(checkpoint_path: Path) -> str:
    # Try to read sibling best.json/config.json to identify model
    cfg_candidates = [checkpoint_path.parent / "best.json", checkpoint_path.parent / "config.json"]
    for cp in cfg_candidates:
        if cp.exists():
            try:
                with open(cp, "r") as f:
                    d = json.load(f)
                if "model" in d:
                    return d["model"]
            except Exception:
                pass
    return "cnn_small"


def _build_model(model_name: str, num_classes: int):
    if model_name == "mobilenet":
        return MobileNetV3SmallHead(num_classes=num_classes, transfer=False)
    return CNNSmall(num_classes=num_classes, base_channels=32, dropout=0.3)


def _softmax_topk(logits: torch.Tensor, k=3):
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
    return vals.cpu().numpy(), idxs.cpu().numpy()


def evaluate_split(model: nn.Module, loader, class_names: List[str], device: torch.device, out_csv: Path):
    model.eval()
    y_true_all, y_pred_all = [], []
    topk_lines = []
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += float(loss.item())

            pred = logits.argmax(dim=1)
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

            vals, idxs = _softmax_topk(logits, k=3)
            for i in range(x.size(0)):
                top3 = [class_names[j] for j in idxs[i].tolist()]
                line = f"{paths[i]},{class_names[y[i].item()]},{class_names[pred[i].item()]},{'|'.join(top3)}"
                topk_lines.append(line)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("path,true,pred,top3\n")
        for line in topk_lines:
            f.write(line + "\n")
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint on val/test splits")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    args = ap.parse_args()

    run_dir = create_run_dir(args.artifacts_root, f"eval-checkpoint-{args.checkpoint.stem[:8]}")
    logger = get_logger(run_dir)
    log_banner(logger, "EVAL")

    # Load class mapping written in data/
    with open(Path("data") / "class_indices.json", "r") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    # Use split json from sibling config if available
    cfg_candidates = [args.checkpoint.parent / "config.json"]
    split_json = None
    for cp in cfg_candidates:
        if cp.exists():
            with open(cp, "r") as f:
                cfg = json.load(f)
            if "split_json" in cfg:
                split_json = Path(cfg["split_json"])
                break
    if split_json is None or not split_json.exists():
        # Fallback to default name
        split_json = Path("splits") / "asl_val_split_seed42_r10.json"

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.train_dir, args.test_dir, split_json, batch_size=64, num_workers=args.num_workers,
        size=args.size, aug=False
    )

    model_name = _infer_model_type(args.checkpoint)
    model = _build_model(model_name, num_classes=len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    logger.info("Loaded checkpoint: %s", args.checkpoint)
    logger.info("Model: %s  Device: %s", model.__class__.__name__, device)

    # Val
    y_true, y_pred = evaluate_split(model, val_loader, class_names, device, out_csv=run_dir / "predictions_val.csv")
    metrics = compute_metrics(y_true, y_pred, class_names)
    write_json(metrics, run_dir / "metrics_val.json")
    plot_confusion(y_true, y_pred, class_names, out_path=run_dir / "confmat_val.png", title="Confusion (val)")
    logger.info("VAL: acc=%.4f macroF1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    # Test (if present)
    if test_loader is not None:
        y_true_t, y_pred_t = evaluate_split(model, test_loader, class_names, device, out_csv=run_dir / "predictions_test.csv")
        metrics_t = compute_metrics(y_true_t, y_pred_t, class_names)
        write_json(metrics_t, run_dir / "metrics_test.json")
        plot_confusion(y_true_t, y_pred_t, class_names, out_path=run_dir / "confmat_test.png", title="Confusion (test)")
        logger.info("TEST: acc=%.4f macroF1=%.4f", metrics_t["accuracy"], metrics_t["macro_f1"])

    logger.info("Eval artifacts at: %s", run_dir)


if __name__ == "__main__":
    main()
