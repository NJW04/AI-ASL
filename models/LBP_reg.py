"""LBP + RGB color histogram baseline with cached features and SGD training.

- Parallel (multiprocessing) feature extraction with a persistent cache.
- Per-epoch logging (accuracy, macro-F1) and early stopping on val macro-F1.
- Final evaluation on val (and test if available) with confusion matrices.
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
import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import tqdm

from data.asl import ensure_data, make_split_lists, summarize_class_distribution

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.",
)


def ensure_dir(path: Path):
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, Any], path: Path, indent: int = 2):
    """Write JSON ``data`` to ``path`` with indentation."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    """Create a logger that logs to console and ``<run_dir>/run.log``.

    Parameters
    ----------
    run_dir : Path
        Directory for the log file.
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
    """Log a formatted banner with the provided title."""
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)


def _save_figure(out_path: Path, dpi: int = 160):
    """Save the current matplotlib figure to ``out_path``."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path, title: str):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(out_path)


def _extract_feature_vec(img_path: str) -> np.ndarray:
    """Extract LBP texture + RGB color histogram features for one image."""
    P = 24
    R = 3
    METHOD = "uniform"
    N_BINS_COLOR = 32
    EPS = 1e-7

    img_rgb = Image.open(img_path).convert("RGB")
    img_np = np.array(img_rgb)

    img_gray_float = rgb2gray(img_np)
    img_gray = (img_gray_float * 255).astype(np.uint8)

    lbp = local_binary_pattern(img_gray, P, R, METHOD)

    n_bins_lbp = P + 2
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins_lbp + 1), range=(0, n_bins_lbp))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + EPS)

    hist_r, _ = np.histogram(img_np[:, :, 0].ravel(), bins=N_BINS_COLOR, range=(0, 256))
    hist_g, _ = np.histogram(img_np[:, :, 1].ravel(), bins=N_BINS_COLOR, range=(0, 256))
    hist_b, _ = np.histogram(img_np[:, :, 2].ravel(), bins=N_BINS_COLOR, range=(0, 256))

    hist_r = hist_r.astype("float") / (hist_r.sum() + EPS)
    hist_g = hist_g.astype("float") / (hist_g.sum() + EPS)
    hist_b = hist_b.astype("float") / (hist_b.sum() + EPS)

    vec = np.concatenate([hist_lbp, hist_r, hist_g, hist_b])
    return vec


def _load_split(split_json: Path):
    """Load train/val/test tuples and class names from a split JSON."""
    with open(split_json, "r") as f:
        d = json.load(f)
    class_names = d["class_names"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["train"]]
    val = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["val"]]

    test = []
    if "test" in d and d["test"]:
        test = [(rec["path"], class_to_idx[rec["label"]]) for rec in d["test"]]

    return train, val, test, class_names


def _load_test_items(test_dir: Path, class_names: List[str]):
    """Fallback for loading test items from a physical directory."""
    from data.asl import _build_test_items

    items = _build_test_items(test_dir, class_names)
    return [(it.path, it.label) for it in items]


def _extract_features_cached(pairs: List[Tuple[str, int]], cache_path: Path, logger):
    """Extract features for (path, label) pairs, using a persistent cache if available."""
    if cache_path.exists():
        logger.info("Loading cached features: %s", cache_path)
        with open(cache_path, "rb") as f:
            feats = pickle.load(f)
        return feats["X"], feats["y"]

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


def main():
    """Train and evaluate the LBP+Color baseline with per-epoch logging and early stopping."""
    ap = argparse.ArgumentParser(
        description="Baseline: SGD(Logistic) on LBP + Color features with per-epoch metrics"
    )
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-kaggle", action="store_true", help="Download via Kaggle API if dataset not present")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    ap.add_argument("--epochs", type=int, default=50, help="Number of SGD epochs for the linear baseline")
    ap.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    ap.add_argument("--alpha", type=float, default=0.0001, help="L2 regularization strength for SGD")
    ap.add_argument("--patience", type=int, default=5, help="Epochs to wait for improvement before early stopping")

    args = ap.parse_args()
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(args.artifacts_root) / f"{ts}-baseline-lbp"
    ensure_dir(run_dir)
    logger = get_logger(run_dir)
    log_banner(logger, "BASELINE: LBP (SGD) WITH PER-EPOCH METRICS")

    log_banner(logger, "1. DATA SETUP & FEATURE EXTRACTION")

    train_dir, test_dir = ensure_data(Path("data"), use_kaggle=args.use_kaggle, logger=logger)
    split_json = make_split_lists(
        args.train_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        subset_per_class=args.subset_per_class,
    )

    dist = summarize_class_distribution(split_json)
    logger.info("Class distribution (train): %s", dist["train"])
    logger.info("Class distribution (val):   %s", dist["val"])
    if "test" in dist:
        logger.info("Class distribution (test):  %s", dist["test"])

    train_pairs, val_pairs, test_pairs, class_names = _load_split(split_json)

    if not test_pairs:
        logger.warning(
            "No 'test' split found in JSON, falling back to loading from test_dir: %s", args.test_dir
        )
        test_pairs = _load_test_items(args.test_dir, class_names)
    else:
        logger.info("Loaded %d 'test' items from split_json.", len(test_pairs))

    write_json({c: i for i, c in enumerate(class_names)}, run_dir / "class_indices.json")

    cache_dir = Path("data/cache")
    ensure_dir(cache_dir)
    split_name = Path(split_json).stem

    LBP_P = 24
    LBP_R = 3
    N_BINS_COLOR = 32
    feature_sig = f"lbp-p{LBP_P}-r{LBP_R}_color-b{N_BINS_COLOR}"

    cache_train_path = cache_dir / f"{split_name}_{feature_sig}_train.pkl"
    cache_val_path = cache_dir / f"{split_name}_{feature_sig}_val.pkl"
    cache_test_path = cache_dir / f"{split_name}_{feature_sig}_test.pkl"

    logger.info("Using persistent cache for train: %s", cache_train_path)
    logger.info("Using persistent cache for val:   %s", cache_val_path)

    Xtr, ytr = _extract_features_cached(train_pairs, cache_train_path, logger)
    Xva, yva = _extract_features_cached(val_pairs, cache_val_path, logger)

    Xte, yte = (None, None)
    if test_pairs:
        logger.info("Using persistent cache for test:  %s", cache_test_path)
        Xte, yte = _extract_features_cached(test_pairs, cache_test_path, logger)

    logger.info("Feature dims: train=%s, val=%s", Xtr.shape, Xva.shape)

    log_banner(logger, "2. MODEL TRAINING (SGD)")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte) if Xte is not None else None

    classes = np.arange(len(class_names), dtype=np.int64)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="optimal",
        eta0=args.lr,
        max_iter=1,
        tol=None,
        shuffle=True,
        random_state=args.seed,
    )

    epoch_csv = (run_dir / "epoch_log.csv").open("w", encoding="utf-8")
    print("epoch,acc_train,macro_f1_train,acc_val,macro_f1_val", file=epoch_csv)

    n = Xtr_s.shape[0]
    idx = np.arange(n)

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_clf = None

    logger.info("Starting training for %d epochs with patience=%d...", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        clf.partial_fit(Xtr_s[idx], ytr[idx], classes=classes)

        pred_tr = clf.predict(Xtr_s)
        pred_va = clf.predict(Xva_s)
        acc_tr = float(accuracy_score(ytr, pred_tr))
        f1_tr = float(f1_score(ytr, pred_tr, average="macro"))
        acc_va = float(accuracy_score(yva, pred_va))
        f1_va = float(f1_score(yva, pred_va, average="macro"))
        print(f"{epoch},{acc_tr:.6f},{f1_tr:.6f},{acc_va:.6f},{f1_va:.6f}", file=epoch_csv)
        epoch_csv.flush()

        if f1_va > best_val_f1:
            best_val_f1 = f1_va
            best_epoch = epoch
            epochs_no_improve = 0
            best_clf = copy.deepcopy(clf)
            logger.info(
                "Epoch %d/%d  train: acc=%.4f f1=%.4f  val: acc=%.4f f1=%.4f  (NEW BEST)",
                epoch,
                args.epochs,
                acc_tr,
                f1_tr,
                acc_va,
                f1_va,
            )
        else:
            epochs_no_improve += 1
            logger.info(
                "Epoch %d/%d  train: acc=%.4f f1=%.4f  val: acc=%.4f f1=%.4f  (Patience %d/%d)",
                epoch,
                args.epochs,
                acc_tr,
                f1_tr,
                acc_va,
                f1_va,
                epochs_no_improve,
                args.patience,
            )

        if epochs_no_improve >= args.patience:
            logger.info(
                "Early stopping triggered at epoch %d. Best score: %.4f at epoch %d.",
                epoch,
                best_val_f1,
                best_epoch,
            )
            break

    epoch_csv.close()
    logger.info("Epoch logging saved to %s", run_dir / "epoch_log.csv")

    log_banner(logger, "3. FINAL EVALUATION")

    if best_clf is not None:
        logger.info("Restoring best model from epoch %d (val_f1=%.4f)", best_epoch, best_val_f1)
        clf = best_clf
    else:
        logger.warning("No best model was saved. Using model from final epoch.")

    def _eval(X, y, split_name: str):
        """Evaluate on a split, save metrics and confusion matrix."""
        pred = clf.predict(X)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro"))
        write_json({"accuracy": acc, "macro_f1": f1}, run_dir / f"metrics_{split_name}.json")
        plot_confusion(
            y,
            pred,
            labels=class_names,
            out_path=run_dir / f"confmat_{split_name}.png",
            title=f"LBP(SGD) Confusion ({split_name})",
        )
        logger.info("%s: acc=%.4f macroF1=%.4f", split_name.upper(), acc, f1)

    _eval(Xva_s, yva, "val")
    if Xte_s is not None:
        _eval(Xte_s, yte, "test")

    log_banner(logger, "RUN COMPLETE")

    cfg = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "subset_per_class": args.subset_per_class,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "lr": args.lr,
        "alpha": args.alpha,
        "split_json": str(split_json),
        "class_names": class_names,
    }
    write_json(cfg, run_dir / "config.json")
    logger.info("All artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()
