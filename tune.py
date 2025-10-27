#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small hyperparameter search with Optuna.
Search space:
- lr [1e-4..3e-3]
- weight_decay [1e-6..1e-3]
- dropout [0..0.4]
- base channels {32} (fixed in CNNSmall, keep simple)
- aug {on,off}
- size {96,128}
- subset_per_class {300}
Optimizes val macro-F1 using the training loop (with fewer epochs by default).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import datetime as _dt

import optuna

from train import run_training, parse_args as parse_train_args


# =========================
# Inlined utility helpers
# =========================

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
    run_dir = Path(artifacts_root) / _timestamp_slug(slug)
    _ensure_dir(run_dir)
    return run_dir

def write_json(data: dict[str, Any], path: Path, indent: int = 2):
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def get_logger(run_dir: Path, name: str = "run", level=logging.INFO) -> logging.Logger:
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
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)


# =========================
# Tuning script
# =========================

def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    ap.add_argument("--train-dir", type=Path, default=Path("data/asl_alphabet_train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/asl_alphabet_test"))
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts/asl_runs"))
    args = ap.parse_args()

    run_dir = create_run_dir(args.artifacts_root, f"tune-optuna-{args.n_trials}trials")
    logger = get_logger(run_dir)
    log_banner(logger, "TUNE: OPTUNA")

    trials_csv = (run_dir / "trials.csv").open("w")
    print("trial,lr,weight_decay,dropout,aug,size,subset_per_class,macro_f1,run_dir", file=trials_csv)

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        aug = trial.suggest_categorical("aug", [True, False])
        size = trial.suggest_categorical("size", [96, 128])
        subset_per_class = trial.suggest_categorical("subset_per_class", [300])

        # Build train args
        train_args = parse_train_args()
        train_args.train_dir = args.train_dir
        train_args.test_dir = args.test_dir
        train_args.epochs = args.epochs
        train_args.lr = lr
        train_args.weight_decay = weight_decay
        train_args.dropout = dropout
        train_args.aug = aug
        train_args.size = size
        train_args.subset_per_class = subset_per_class
        train_args.model = "cnn_small"
        train_args.transfer = False
        train_args.seed = args.seed
        train_args.artifacts_root = Path(args.artifacts_root) / "hp-tuning"
        _ensure_dir(train_args.artifacts_root)

        res = run_training(train_args)
        macro_f1 = float(res["best_macro_f1"])
        print(f"{trial.number},{lr},{weight_decay},{dropout},{aug},{size},{subset_per_class},{macro_f1},{res['run_dir']}", file=trials_csv, flush=True)
        return macro_f1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "trial": study.best_trial.number
    }
    write_json(best, run_dir / "best.json")
    logger.info("Best macro-F1: %.4f | Params: %s", best["value"], best["params"])
    trials_csv.close()


if __name__ == "__main__":
    main()
