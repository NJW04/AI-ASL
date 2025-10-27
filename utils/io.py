#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import shutil
import hashlib


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path):
    from pathlib import Path as _P
    def _default(o):
        # cleanly serialize Path and any other odd types
        if isinstance(o, _P):
            return str(o)
        return str(o)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)



def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_figure(out_path: Path, tight: bool = True, dpi: int = 160):
    ensure_dir(out_path.parent)
    if tight:
        plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def timestamp_slug(prefix: str) -> str:
    from time import strftime
    return f"{strftime('%Y-%m-%d_%H%M')}-{prefix}"


def create_run_dir(artifacts_root: Path, slug: str) -> Path:
    run_dir = Path(artifacts_root) / timestamp_slug(slug)
    ensure_dir(run_dir)
    return run_dir


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def copy_to(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
