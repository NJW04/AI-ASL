#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOG + Color histogram features -> Logistic Regression baseline
- Builds/uses split JSON (no file moves)
- Extracts features with optional per-run on-disk cache (features.pkl) inside run folder ONLY
- Evaluates on val + test, saves metrics + confusion matrices
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from data.asl import ensure_data, make_split_lists, summarize_class_distribution
from data.transforms import extract_hog_color
from utils.io import write_json, ensure_dir
from utils.log import get_logger, log_banner
from utils.metrics import plot_confusion


def _extract_feature_vec(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return extract_hog_color(img)


def _load_split(split_json: Path):
    import json
    with open(split_json, "r") as f:
        d = json.load(f)
    class_names = d["class_names"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["train"]]
    val = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["val"]]
    return train, val, class_names


def _load_test_items(test_dir: Path, class_names: List[str]):
    # Accept either foldered or flat test dir (best effort)
    from data.asl import _build_test_items
    items = _build_test_items(test_dir, class_names)
    return [(it.path, it.label) for it in items]


def _extract_features_cached(pairs: List[Tuple[str, int]], cache_path: Path, logger):
    if cache_path.exists():
        logger.info("Loading cached features: %s", cache_path)
        with open(cache_path, "rb") as f:
            feats = pickle.load(f)
        return feats["X"], feats["y"]

    X_list, y_list = [], []
    for i, (p, y) in enumerate(pairs):
        vec = _extract_feature_vec(p)
        X_list.append(vec)
        y_list.append(y)
        if (i + 1) % 500 == 0:
            logger.info("Extracted %d/%d features...", i + 1, len(pairs))
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    logger.info("Saved features cache: %s", cache_path)
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Baseline: Logistic Regression on HOG + Color hist features")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true", help="Download via Kaggle API if dataset not present")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    args = ap.parse_args()

    run_dir = Path(args.artifacts_root) / __import__("time").strftime("%Y-%m-%d_%H%M") / "baseline-logreg"
    # Human-friendly single folder (per requirements)
    run_dir = Path(args.artifacts_root) / (__import__("time").strftime("%Y-%m-%d_%H%M") + "-baseline-logreg")
    ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "BASELINE: LOGISTIC REGRESSION (HOG + COLOR)")

    # Ensure data & splits
    train_dir, test_dir = ensure_data(Path("data"), use_kaggle=args.use_kaggle, logger=logger)
    split_json = make_split_lists(args.train_dir, val_ratio=args.val_ratio,
                                  seed=args.seed, subset_per_class=args.subset_per_class)
    dist = summarize_class_distribution(split_json)
    logger.info("Class distribution (train): %s", dist["train"])
    logger.info("Class distribution (val):   %s", dist["val"])

    # Load split and test
    train_pairs, val_pairs, class_names = _load_split(split_json)
    test_pairs = _load_test_items(args.test_dir, class_names)
    write_json({c: i for i, c in enumerate(class_names)}, run_dir / "class_indices.json")

    # Features (with per-run cache)
    Xtr, ytr = _extract_features_cached(train_pairs, run_dir / "features_train.pkl", logger)
    Xva, yva = _extract_features_cached(val_pairs, run_dir / "features_val.pkl", logger)
    Xte, yte = (None, None)
    if test_pairs:
        Xte, yte = _extract_features_cached(test_pairs, run_dir / "features_test.pkl", logger)

    logger.info("Feature dims: train=%s, val=%s", Xtr.shape, Xva.shape)
    # Pipeline: Standardize -> LogisticRegression
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000, multi_class="auto", n_jobs=None, verbose=0)
    )
    clf.fit(Xtr, ytr)
    logger.info("Model fit complete.")

    def _eval(X, y, split_name: str):
        pred = clf.predict(X)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro"))
        write_json({"accuracy": acc, "macro_f1": f1}, run_dir / f"metrics_{split_name}.json")
        plot_confusion(y, pred, labels=class_names, out_path=run_dir / f"confmat_{split_name}.png",
                       title=f"LogReg Confusion ({split_name})")
        logger.info("%s: acc=%.4f macroF1=%.4f", split_name.upper(), acc, f1)

    _eval(Xva, yva, "val")
    if Xte is not None:
        _eval(Xte, yte, "test")

    # Save config
    cfg = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "val_ratio": args.val_ratio,
        "subset_per_class": args.subset_per_class,
        "seed": args.seed,
        "split_json": str(split_json),
        "class_names": class_names,
    }
    write_json(cfg, run_dir / "config.json")
    logger.info("Run artifacts in: %s", run_dir)


if __name__ == "__main__":
    main()
