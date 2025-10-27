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
- subset_per_class {None, 500}
Optimizes val macro-F1 using the training loop (with fewer epochs by default).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import optuna

from train import run_training, parse_args as parse_train_args
from utils.io import create_run_dir, write_json, ensure_dir
from utils.log import get_logger, log_banner


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
        from utils.io import ensure_dir
        train_args.artifacts_root = Path(args.artifacts_root) / "hp-tuning"
        ensure_dir(train_args.artifacts_root)

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
