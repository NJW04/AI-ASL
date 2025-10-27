#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP + Color histogram features -> Linear classifier baseline
- USES MULTIPROCESSING for faster (parallel) feature extraction cache building
- USES PERSISTENT CACHE in data/cache/ to run extraction only ONCE
- Trains with SGD (logistic loss) in epochs so we can log per-epoch macro F1 on train and val
- Evaluates on val and test, saves metrics and confusion matrices
- SILENCES specific sklearn UserWarnings for cleaner logs
- INCLUDES EARLY STOPPING to save time and find the best model
"""
from __future__ import annotations

import argparse
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count
import warnings
import copy  # <--- NEW IMPORT (replaces clone)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
# from sklearn.base import clone  <--- REMOVED
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import tqdm

from data.asl import ensure_data, make_split_lists, summarize_class_distribution

# --- THIS BLOCK SUPPRESSES THE REPETITIVE SKLEARN WARNING ---
warnings.filterwarnings(
    "ignore", 
    message="The number of unique classes is greater than 50% of the number of samples."
)
# --- END OF BLOCK ---


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
    """
    Extracts a combined LBP (texture) and Color (RGB) histogram feature vector.
    """
    # LBP parameters
    P = 24  # number of points
    R = 3   # radius
    METHOD = 'uniform'
    N_BINS_COLOR = 32 # Bins per color channel
    EPS = 1e-7 # for normalization stability

    img_rgb = Image.open(img_path).convert("RGB")
    img_np = np.array(img_rgb)

    # --- 1. LBP (Texture) Features ---
    img_gray_float = rgb2gray(img_np)
    # Convert float (0.0-1.0) to integer (0-255) to suppress warning
    img_gray = (img_gray_float * 255).astype(np.uint8) 
    
    lbp = local_binary_pattern(img_gray, P, R, METHOD)
    
    n_bins_lbp = P + 2
    hist_lbp, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_bins_lbp + 1),
                               range=(0, n_bins_lbp))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + EPS)

    # --- 2. Color Features ---
    hist_r, _ = np.histogram(img_np[:, :, 0].ravel(), bins=N_BINS_COLOR, range=(0, 256))
    hist_g, _ = np.histogram(img_np[:, :, 1].ravel(), bins=N_BINS_COLOR, range=(0, 256))
    hist_b, _ = np.histogram(img_np[:, :, 2].ravel(), bins=N_BINS_COLOR, range=(0, 256))
    
    hist_r = hist_r.astype("float") / (hist_r.sum() + EPS)
    hist_g = hist_g.astype("float") / (hist_g.sum() + EPS)
    hist_b = hist_b.astype("float") / (hist_b.sum() + EPS)

    # --- 3. Concatenate all features ---
    vec = np.concatenate([hist_lbp, hist_r, hist_g, hist_b])
    
    return vec

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

    # --- Start parallel extraction ---
    n_workers = cpu_count()
    logger.info(f"Cache not found. Extracting {len(pairs)} features using {n_workers} cores...")

    image_paths = [p for p, y in pairs]
    y_list = [y for p, y in pairs]
    
    X_list = []
    with Pool(processes=n_workers) as pool:
        for vec in tqdm.tqdm(pool.imap(_extract_feature_vec, image_paths), total=len(image_paths)):
            X_list.append(vec)

    logger.info("Feature extraction complete. Stacking...")

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
    ap = argparse.ArgumentParser(description="Baseline: SGD(Logistic) on LBP + Color features with per-epoch metrics")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true", help="Download via Kaggle API if dataset not present")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    # new knobs for epoch logging
    ap.add_argument("--epochs", type=int, default=20, help="Number of SGD epochs for the linear baseline")
    ap.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    ap.add_argument("--alpha", type=float, default=0.0001, help="L2 regularization strength for SGD")
    ap.add_argument("--patience", type=int, default=10, help="Epochs to wait for improvement before early stopping")
    
    args = ap.parse_args()

    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(args.artifacts_root) / f"{ts}-baseline-lbp" 
    ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "BASELINE: LBP (SGD) WITH PER-EPOCH METRICS") 

    # ---
    # 1. DATA SETUP & FEATURE EXTRACTION
    # ---
    log_banner(logger, "1. DATA SETUP & FEATURE EXTRACTION")
    
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

    # Define a persistent cache directory
    cache_dir = Path("data/cache")
    ensure_dir(cache_dir)
    split_name = Path(split_json).stem 
    
    # Define feature params (must match _extract_feature_vec)
    LBP_P = 24
    LBP_R = 3
    N_BINS_COLOR = 32
    feature_sig = f"lbp-p{LBP_P}-r{LBP_R}_color-b{N_BINS_COLOR}"
    
    # Define stable cache file paths
    cache_train_path = cache_dir / f"{split_name}_{feature_sig}_train.pkl"
    cache_val_path = cache_dir / f"{split_name}_{feature_sig}_val.pkl"
    cache_test_path = cache_dir / f"{split_name}_{feature_sig}_test.pkl"

    logger.info("Using persistent cache for train: %s", cache_train_path)
    logger.info("Using persistent cache for val:   %s", cache_val_path)
    
    # Feature extraction with cache
    Xtr, ytr = _extract_features_cached(train_pairs, cache_train_path, logger)
    Xva, yva = _extract_features_cached(val_pairs, cache_val_path, logger)
    
    Xte, yte = (None, None)
    if test_pairs:
        logger.info("Using persistent cache for test:  %s", cache_test_path)
        Xte, yte = _extract_features_cached(test_pairs, cache_test_path, logger)
    
    logger.info("Feature dims: train=%s, val=%s", Xtr.shape, Xva.shape)

    # ---
    # 2. MODEL TRAINING (SGD)
    # ---
    log_banner(logger, "2. MODEL TRAINING (SGD)")

    # Standardize using train statistics only
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte) if Xte is not None else None

    # SGD classifier with logistic loss
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

    # --- START OF EARLY STOPPING LOGIC ---
    n = Xtr_s.shape[0]
    idx = np.arange(n)
    
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_clf = None # This will store the best model
    
    logger.info("Starting training for %d epochs with patience=%d...", args.epochs, args.patience)
    
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
        
        # Check for improvement
        if f1_va > best_val_f1:
            best_val_f1 = f1_va
            best_epoch = epoch
            epochs_no_improve = 0
            best_clf = copy.deepcopy(clf) # <--- THIS IS THE FIX (was clone(clf))
            logger.info("Epoch %d/%d  train: acc=%.4f f1=%.4f  val: acc=%.4f f1=%.4f  (NEW BEST)",
                         epoch, args.epochs, acc_tr, f1_tr, acc_va, f1_va)
        else:
            epochs_no_improve += 1
            logger.info("Epoch %d/%d  train: acc=%.4f f1=%.4f  val: acc=%.4f f1=%.4f  (Patience %d/%d)",
                         epoch, args.epochs, acc_tr, f1_tr, acc_va, f1_va, epochs_no_improve, args.patience)
        
        # Trigger early stopping
        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered at epoch %d. Best score: %.4f at epoch %d.",
                         epoch, best_val_f1, best_epoch)
            break
            
    # --- END OF EARLY STOPPING LOGIC ---

    epoch_csv.close()
    logger.info("Epoch logging saved to %s", run_dir / "epoch_log.csv")


    # ---
    # 3. FINAL EVALUATION
    # ---
    log_banner(logger, "3. FINAL EVALUATION")
    
    # Restore the best model for final evaluation
    if best_clf is not None:
        logger.info("Restoring best model from epoch %d (val_f1=%.4f)", best_epoch, best_val_f1)
        clf = best_clf
    else:
        logger.warning("No best model was saved. Using model from final epoch.")


    # Final evaluation helper
    def _eval(X, y, split_name: str):
        pred = clf.predict(X)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro"))
        write_json({"accuracy": acc, "macro_f1": f1}, run_dir / f"metrics_{split_name}.json")
        plot_confusion(y, pred, labels=class_names, out_path=run_dir / f"confmat_{split_name}.png",
                       title=f"LBP(SGD) Confusion ({split_name})")
        logger.info("%s: acc=%.4f macroF1=%.4f", split_name.upper(), acc, f1)

    # Save final metrics and plots
    _eval(Xva_s, yva, "val")
    if Xte_s is not None:
        _eval(Xte_s, yte, "test")

    # ---
    # 4. RUN COMPLETE
    # ---
    log_banner(logger, "RUN COMPLETE")
    
    # Save config for reproducibility
    cfg = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "val_ratio": args.val_ratio,
        "subset_per_class": args.subset_per_class,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience, # <-- Save new param
        "best_epoch": best_epoch,  # <-- Save best epoch
        "best_val_f1": best_val_f1, # <-- Save best score
        "lr": args.lr,
        "alpha": args.alpha,
        "split_json": str(split_json),
        "class_names": class_names,
    }
    write_json(cfg, run_dir / "config.json")
    logger.info("All artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()