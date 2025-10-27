#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOG + Color histogram features -> Linear classifier baseline
- Builds/uses split JSON (no file moves)
- Extracts features with a per-run on-disk cache (features.pkl) inside the run folder only
- Trains with SGD (logistic loss) in epochs so we can log per-epoch macro F1 on train and val
- Evaluates on val and test, saves metrics and confusion matrices
- Writes epoch-wise CSV: epoch_log.csv with train/val F1 and accuracy
"""
from __future__ import annotations

import argparse
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier

from data.asl import ensure_data, make_split_lists, summarize_class_distribution
from data.transforms import extract_hog_color


# -----------------
# Small IO helpers
# -----------------
def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_json(data: Dict[str, Any], path: Path, indent: int = 2):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    logger_name = f"{run_dir.resolve()}/{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
        fh = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger

def log_banner(logger: logging.Logger, title: str):
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)

def _save_figure(out_path: Path, dpi: int = 160):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
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


# =========================
# Baseline feature logic
# =========================
def _extract_feature_vec(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return extract_hog_color(img)

def _load_split(split_json: Path):
    with open(split_json, "r") as f:
        d = json.load(f)
    class_names = d["class_names"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["train"]]
    val = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["val"]]
    return train, val, class_names

def _load_test_items(test_dir: Path, class_names: List[str]):
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


# =========================
# Training and evaluation
# =========================
def main():
    ap = argparse.ArgumentParser(description="Baseline: SGD(Logistic) on HOG + Color features with per-epoch metrics")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true", help="Download via Kaggle API if dataset not present")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    # new knobs for epoch logging
    ap.add_argument("--epochs", type=int, default=10, help="Number of SGD epochs for the linear baseline")
    ap.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    ap.add_argument("--alpha", type=float, default=0.0001, help="L2 regularization strength for SGD")
    args = ap.parse_args()

    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(args.artifacts_root) / f"{ts}-baseline-logreg"
    ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "BASELINE: LOGISTIC (SGD) WITH PER-EPOCH METRICS")

    # Ensure data and split
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

    # Feature extraction with cache
    Xtr, ytr = _extract_features_cached(train_pairs, run_dir / "features_train.pkl", logger)
    Xva, yva = _extract_features_cached(val_pairs, run_dir / "features_val.pkl", logger)
    Xte, yte = (None, None)
    if test_pairs:
        Xte, yte = _extract_features_cached(test_pairs, run_dir / "features_test.pkl", logger)
    logger.info("Feature dims: train=%s, val=%s", Xtr.shape, Xva.shape)

    # Standardize using train statistics only
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte) if Xte is not None else None

    # SGD classifier with logistic loss, trained in epochs so we can record metrics
    classes = np.arange(len(class_names), dtype=np.int64)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="optimal",
        eta0=args.lr,
        max_iter=1,          # we call partial_fit per epoch
        tol=None,
        shuffle=True,
        random_state=args.seed
    )

    # Epoch logging CSV
    epoch_csv = (run_dir / "epoch_log.csv").open("w", encoding="utf-8")
    print("epoch,acc_train,macro_f1_train,acc_val,macro_f1_val", file=epoch_csv)

    # Train for args.epochs epochs with partial_fit
    n = Xtr_s.shape[0]
    idx = np.arange(n)
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        clf.partial_fit(Xtr_s[idx], ytr[idx], classes=classes)

        # Metrics each epoch
        pred_tr = clf.predict(Xtr_s)
        pred_va = clf.predict(Xva_s)
        acc_tr = float(accuracy_score(ytr, pred_tr))
        f1_tr = float(f1_score(ytr, pred_tr, average="macro"))
        acc_va = float(accuracy_score(yva, pred_va))
        f1_va = float(f1_score(yva, pred_va, average="macro"))
        print(f"{epoch},{acc_tr:.6f},{f1_tr:.6f},{acc_va:.6f},{f1_va:.6f}", file=epoch_csv)
        epoch_csv.flush()
        logger.info("Epoch %d/%d  train: acc=%.4f f1=%.4f  val: acc=%.4f f1=%.4f",
                    epoch, args.epochs, acc_tr, f1_tr, acc_va, f1_va)

    epoch_csv.close()
    logger.info("Epoch logging saved to %s", run_dir / "epoch_log.csv")

    # Final evaluation helper
    def _eval(X, y, split_name: str):
        pred = clf.predict(X)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro"))
        write_json({"accuracy": acc, "macro_f1": f1}, run_dir / f"metrics_{split_name}.json")
        plot_confusion(y, pred, labels=class_names, out_path=run_dir / f"confmat_{split_name}.png",
                       title=f"Logistic(SGD) Confusion ({split_name})")
        logger.info("%s: acc=%.4f macroF1=%.4f", split_name.upper(), acc, f1)

    # Save final metrics and plots
    _eval(Xva_s, yva, "val")
    if Xte_s is not None:
        _eval(Xte_s, yte, "test")

    # Save config for reproducibility
    cfg = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "val_ratio": args.val_ratio,
        "subset_per_class": args.subset_per_class,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "split_json": str(split_json),
        "class_names": class_names,
    }
    write_json(cfg, run_dir / "config.json")
    logger.info("Run artifacts in: %s", run_dir)


if __name__ == "__main__":
    main()
