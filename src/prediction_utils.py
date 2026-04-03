"""Shared helpers for bankruptcy / distress scoring (consistent train + app)."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_distress_proba(model: Any, x) -> np.ndarray:
    """Return P(label=1) where label=1 means distress / bankruptcy in training data."""
    return model.predict_proba(x)[:, 1].astype(np.float64)


def bankruptcy_risk_band(prob: float, threshold: float) -> str:
    """
    Map probability to a simple band using the F1-tuned threshold.

    - Below threshold: lower predicted distress risk
    - Near threshold: borderline / watchlist
    - Above threshold: elevated distress risk
    """
    p = float(prob)
    t = float(threshold)
    margin = 0.08
    if p >= t + margin:
        return "Elevated distress risk"
    if p >= t - margin:
        return "Borderline / watchlist"
    return "Lower distress risk"


def bankruptcy_summary(prob: float, threshold: float, pred_positive: bool) -> str:
    """One short paragraph for non-technical users."""
    band = bankruptcy_risk_band(prob, threshold)
    pct = 100.0 * prob
    if pred_positive:
        return (
            f"**Prediction:** model flags this case as **distressed** (bankruptcy-related label in the dataset). "
            f"Estimated probability of distress: **{pct:.1f}%**. "
            f"Risk band: **{band}**. "
            "Treat this as a screening signal — validate with fundamentals and domain judgment."
        )
    return (
        f"**Prediction:** model does **not** flag distress at the current decision threshold. "
        f"Estimated probability of distress: **{pct:.1f}%**. "
        f"Risk band: **{band}**. "
        "Rare events are hard; still monitor leverage and profitability trends."
    )
