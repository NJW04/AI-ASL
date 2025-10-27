#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, recall_score
from utils.io import save_figure


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_rec = recall_score(y_true, y_pred, average=None).tolist()
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_rec,
        "labels": labels
    }
    return metrics


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(out_path)
