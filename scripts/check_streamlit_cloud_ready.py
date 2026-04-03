#!/usr/bin/env python3
"""Verify files and imports before Streamlit Community Cloud (or local) run."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(120)
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def main() -> int:
    os.chdir(ROOT)
    failed = False

    required = [
        "app/streamlit_app.py",
        "requirements-app.txt",
        "runtime.txt",
        ".streamlit/config.toml",
        ".gitattributes",
        "data/processed/tabular.csv",
        "checkpoints/local_lite/model.joblib",
        "checkpoints/local_lite/meta.json",
    ]

    print("=== Files (repo root) ===")
    for rel in required:
        p = os.path.join(ROOT, rel)
        ok = os.path.isfile(p) or os.path.isdir(p)
        print(f"{'OK' if ok else 'MISS'} {rel}")
        if not ok:
            failed = True
        elif os.path.isfile(p) and _is_lfs_pointer(p):
            print(f"    WARN: {rel} is a Git LFS pointer — run: git lfs pull")
            failed = True

    print("\n=== Python imports (Cloud install target) ===")
    try:
        import joblib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import plotly  # noqa: F401
        import requests  # noqa: F401
        import sklearn  # noqa: F401
        import streamlit  # noqa: F401
        import torch  # noqa: F401
        import yaml  # noqa: F401

        print("OK  streamlit, pandas, numpy, sklearn, torch, plotly, joblib, requests, pyyaml")
    except ImportError as e:
        print(f"MISS {e}")
        print("    Hint: python3 -m pip install -r requirements-app.txt")
        failed = True

    print("\n=== Summary ===")
    if failed:
        print("[ACTION] Fix items above before deploying or running the app.")
        return 1
    print("[READY]  OK to deploy on Streamlit Cloud or run scripts/run_streamlit_local.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
