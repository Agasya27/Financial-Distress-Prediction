"""
Blend local_lite (sklearn) and multimodal (torch) probabilities on the time holdout.

Searches a scalar mix: alpha * p_lite + (1 - alpha) * p_mm to improve PR-AUC / ROC-AUC.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.feature_table import load_feature_table
from src.metrics_classification import best_f1_threshold, binary_classification_metrics
from src.models import MultiModalFusionModel
from src.train_multimodal import (
    MMSet,
    _load_base_table,
    _load_text_embeddings,
    build_arrays,
    pick_device,
    time_split_idx,
)
from src.utils import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend local lite + multimodal on time holdout")
    p.add_argument("--tabular-path", default="data/processed/tabular.csv")
    p.add_argument("--mda-dir", default="data/raw/mda_texts")
    p.add_argument("--graph-path", default="data/processed/graph.pt")
    p.add_argument("--text-emb-path", default="data/processed/text_embeddings.pt")
    p.add_argument("--lite-model", default="checkpoints/local_lite/model.joblib")
    p.add_argument("--mm-checkpoint", default="checkpoints/multimodal/best_model.pt")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-metrics", default="checkpoints/blend_metrics_eval.json")
    p.add_argument("--no-text-stats", action="store_true")
    p.add_argument("--no-graph-degree", action="store_true")
    p.add_argument("--grid", type=int, default=41, help="Number of alpha grid points in [0,1].")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.lite_model):
        raise FileNotFoundError(args.lite_model)
    if not os.path.isfile(args.mm_checkpoint):
        raise FileNotFoundError(args.mm_checkpoint)

    base = _load_base_table(args.tabular_path, args.max_rows)
    text_emb = _load_text_embeddings(args.text_emb_path)
    packed = build_arrays(base, text_emb)
    _tr, va_idx = time_split_idx(packed.dates, args.test_ratio)
    y_val = packed.y[va_idx].astype(np.float32)

    x_df, _y_tab, _, _base_df = load_feature_table(
        tabular_path=args.tabular_path,
        mda_dir=args.mda_dir,
        graph_path=args.graph_path,
        use_text_stats=not args.no_text_stats,
        use_graph_degree=not args.no_graph_degree,
        max_rows=args.max_rows,
    )
    if len(x_df) != len(base):
        raise ValueError(
            f"Row mismatch: tabular base {len(base)} vs load_feature_table {len(x_df)}. "
            "Use same --tabular-path / --max-rows / feature flags as training."
        )

    sk_model = joblib.load(args.lite_model)
    x_va = x_df.iloc[va_idx]
    p_lite = sk_model.predict_proba(x_va)[:, 1].astype(np.float64)

    device = pick_device()
    ckpt = torch.load(args.mm_checkpoint, map_location=device, weights_only=False)
    mm = MultiModalFusionModel(tabular_in_dim=packed.tab_x.shape[1], modality_dropout_p=0.0).to(device)
    mm.load_state_dict(ckpt["model_state"])
    mm.eval()

    val_set = MMSet(packed.tab_x[va_idx], packed.txt_x[va_idx], packed.gph_x[va_idx], packed.y[va_idx])
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for tab_x, txt_x, gph_x, _y in loader:
            logits = mm(tab_x.to(device), txt_x.to(device), gph_x.to(device))["logits"].float()
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    p_mm = np.concatenate(chunks).astype(np.float64)

    grid = max(2, int(args.grid))
    alphas = np.linspace(0.0, 1.0, grid)
    best_pr = -1.0
    best_roc = -1.0
    best_a = 0.5
    rows: list[dict[str, float]] = []
    for a in alphas:
        pb = a * p_lite + (1.0 - a) * p_mm
        pr = float(average_precision_score(y_val, pb))
        roc = float(roc_auc_score(y_val, pb))
        rows.append({"alpha": float(a), "pr_auc": pr, "roc_auc": roc})
        if pr > best_pr:
            best_pr = pr
            best_roc = roc
            best_a = float(a)

    p_best = best_a * p_lite + (1.0 - best_a) * p_mm
    thr = best_f1_threshold(y_val, p_best)
    cls = binary_classification_metrics(y_val.astype(np.int32), p_best, threshold=thr)

    solo_lite_pr = float(average_precision_score(y_val, p_lite))
    solo_mm_pr = float(average_precision_score(y_val, p_mm))
    solo_lite_roc = float(roc_auc_score(y_val, p_lite))
    solo_mm_roc = float(roc_auc_score(y_val, p_mm))

    result = {
        "best_alpha": best_a,
        "blend_pr_auc": best_pr,
        "blend_roc_auc": best_roc,
        "lite_only_pr_auc": solo_lite_pr,
        "lite_only_roc_auc": solo_lite_roc,
        "mm_only_pr_auc": solo_mm_pr,
        "mm_only_roc_auc": solo_mm_roc,
        "grid_search": rows,
        **cls,
    }
    save_json(result, args.out_metrics)
    print(result)


if __name__ == "__main__":
    main()
