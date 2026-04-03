"""Shared binary classification metrics (sklearn) for train / evaluate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])


def binary_classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }
