#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(run_dir: Path, name: str = "run", level=logging.INFO):
    logger = logging.getLogger(str(run_dir.resolve()) + "/" + name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate handlers in notebooks

    # Clear existing handlers if any
    if logger.handlers:
        return logger

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    # File
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "run.log", mode="a")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def log_banner(logger: logging.Logger, title: str):
    line = "=" * (len(title) + 4)
    logger.info("\n%s\n| %s |\n%s", line, title, line)
