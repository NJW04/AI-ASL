"""Small Optuna hyperparameter search for the ASL CNN.

Search space
------------
- lr: [1e-4 .. 3e-3] (log-uniform)
- weight_decay: [1e-6 .. 1e-3] (log-uniform)
- dropout: [0.0 .. 0.4]
- base channels: {32} (fixed)
- num_blocks: {2, 3, 4}
- activation: {"relu", "gelu"}
- batch_size: {32, 64, 128}
- size: {96, 128}
- subset_per_class: {300}
- aug: {False} (fixed during tuning)

Optimizes validation macro-F1 by calling the training loop with reduced epochs.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import datetime as _dt

import optuna

# Import training utilities (rename parse_args to avoid conflicts)
from train import run_training, parse_args as parse_train_args


def _ensure_dir(path: Path):
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    """Return a filesystem-safe slug derived from ``text``."""
    s = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text.strip().lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def _timestamp_slug(slug: str) -> str:
    """Return a timestamp-prefixed slug (``YYYY-MM-DD_HH-MM-SS__slug``)."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}__{_slugify(slug)}" if slug else ts


def create_run_dir(artifacts_root: Path, slug: str) -> Path:
    """Create and return a new run directory under ``artifacts_root``."""
    run_dir = Path(artifacts_root) / _timestamp_slug(slug)
    _ensure_dir(run_dir)
    return run_dir


def write_json(data: dict[str, Any], path: Path, indent: int = 2):
    """Write ``data`` to ``path`` as JSON with ``indent`` spaces."""
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
    """Return a logger that writes to console and ``<run_dir>/run.log``."""
    run_dir = Path(run_dir)
    _ensure_dir(run_dir)
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
    """Log a banner-style section heading."""
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)


# =========================
# Tuning script
# =========================
def main():
    """Run Optuna hyperparameter tuning and save results/artifacts."""
    ap = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    ap.add_argument("--n-trials", type=int, default=8, help="Number of Optuna trials to run")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs per trial (keep low for tuning)")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    args = ap.parse_args()

    run_dir = create_run_dir(args.artifacts_root, f"tune-optuna-{args.n_trials}trials")
    logger = get_logger(run_dir)
    log_banner(logger, "TUNE: OPTUNA (Fixed aug=False)")

    trials_csv = (run_dir / "trials.csv").open("w")
    print(
        "trial,lr,weight_decay,dropout,size,subset_per_class,num_blocks,activation,batch_size,macro_f1,run_dir",
        file=trials_csv,
    )

    def objective(trial: optuna.Trial):
        """Single Optuna trial: sample hyperparameters, train, and return macro-F1."""
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        size = trial.suggest_categorical("size", [96, 128])
        subset_per_class = trial.suggest_categorical("subset_per_class", [300])
        num_blocks = trial.suggest_categorical("num_blocks", [2, 3, 4])
        activation = trial.suggest_categorical("activation", ["relu", "gelu"])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_args = parse_train_args([])
        train_args.train_dir = args.train_dir
        train_args.test_dir = args.test_dir
        train_args.epochs = args.epochs
        train_args.lr = lr
        train_args.weight_decay = weight_decay
        train_args.dropout = dropout
        train_args.aug = False
        train_args.size = size
        train_args.subset_per_class = subset_per_class
        train_args.num_blocks = num_blocks
        train_args.activation = activation
        train_args.batch_size = batch_size
        train_args.model = "cnn_small"
        train_args.seed = args.seed
        train_args.artifacts_root = Path(args.artifacts_root) / "hp-tuning"
        _ensure_dir(train_args.artifacts_root)
        train_args.patience = 3

        res = run_training(train_args)
        macro_f1 = float(res["best_macro_f1"])

        print(
            f"{trial.number},{lr},{weight_decay},{dropout},{size},{subset_per_class},"
            f"{num_blocks},{activation},{batch_size},{macro_f1},{res['run_dir']}",
            file=trials_csv,
            flush=True,
        )
        return macro_f1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "trial": study.best_trial.number,
    }
    write_json(best, run_dir / "best.json")
    logger.info("Best macro-F1: %.4f | Params: %s", best["value"], best["params"])
    trials_csv.close()
    logger.info("Tuning complete. Artifacts saved to: %s", run_dir)
    logger.info(
        "Next steps: Take the 'best.json' parameters and run train.py manually with --aug and without --aug."
    )


if __name__ == "__main__":
    main()
