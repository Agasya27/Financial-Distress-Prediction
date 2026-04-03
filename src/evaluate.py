"""Primary evaluation entrypoint.

Default mode evaluates multimodal fusion model.
Use `--mode local_lite` for sklearn baseline evaluation.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.dataset import time_split
from src.feature_table import load_feature_table
from src.evaluate_multimodal import main as evaluate_multimodal_main
from src.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local lightweight model.")
    parser.add_argument("--mode", choices=["multimodal", "local_lite"], default="multimodal")
    parser.add_argument("--model-path", default="checkpoints/local_lite/model.joblib")
    parser.add_argument("--tabular-path", default="data/processed/tabular.csv")
    parser.add_argument("--mda-dir", default="data/raw/mda_texts")
    parser.add_argument("--graph-path", default="data/processed/graph.pt")
    parser.add_argument("--checkpoint", default="checkpoints/multimodal/best_model.pt")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predictions-out", default="checkpoints/local_lite/predictions.csv")
    parser.add_argument("--metrics-out", default="checkpoints/local_lite/metrics_eval.json")
    parser.add_argument("--no-text-stats", action="store_true")
    parser.add_argument("--no-graph-degree", action="store_true")
    return parser.parse_args()


def calc_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    if args.mode == "multimodal":
        import sys

        sys.argv = [
            "evaluate_multimodal",
            "--tabular-path",
            args.tabular_path,
            "--checkpoint",
            args.checkpoint,
            "--out-metrics",
            "checkpoints/multimodal/metrics_eval.json",
            "--test-ratio",
            str(args.test_ratio),
            "--max-rows",
            str(args.max_rows),
            "--batch-size",
            str(args.batch_size),
        ]
        evaluate_multimodal_main()
        return

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    x_df, y, _feature_cols, base_df = load_feature_table(
        tabular_path=args.tabular_path,
        mda_dir=args.mda_dir,
        graph_path=args.graph_path,
        use_text_stats=not args.no_text_stats,
        use_graph_degree=not args.no_graph_degree,
        max_rows=args.max_rows,
    )
    test_mask = ~time_split(base_df, test_ratio=args.test_ratio)

    model = joblib.load(args.model_path)
    meta_path = os.path.join(os.path.dirname(args.model_path), "meta.json")
    threshold = 0.5
    if os.path.exists(meta_path):
        try:
            threshold = float(load_json(meta_path).get("recommended_threshold", 0.5))
        except Exception:
            threshold = 0.5

    x_test = x_df.loc[test_mask]
    y_test = y[test_mask]
    probs = model.predict_proba(x_test)[:, 1]
    metrics = calc_metrics(y_test, probs, threshold=threshold)

    pred_df = base_df.loc[test_mask].copy()
    pred_df["score"] = probs
    pred_df["pred_label"] = (probs >= threshold).astype(int)
    pred_df.to_csv(args.predictions_out, index=False)
    save_json(metrics, args.metrics_out)

    print("[DONE] Evaluation complete")
    print(f"[DONE] predictions: {args.predictions_out}")
    print(f"[DONE] metrics:     {args.metrics_out}")
    print(metrics)


if __name__ == "__main__":
    main()
