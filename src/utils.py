"""
Utility helpers for local-only (Mac-friendly) training and inference.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def ensure_dir(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict[str, Any], path: str) -> None:
    """Persist dictionary to JSON with stable formatting."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str) -> dict[str, Any]:
    """Load JSON as dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    """Current UTC timestamp string."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
