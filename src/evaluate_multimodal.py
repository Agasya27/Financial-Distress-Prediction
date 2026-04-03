from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics_classification import best_f1_threshold, binary_classification_metrics
from src.models import MultiModalFusionModel
from src.train_multimodal import MMSet, _load_base_table, _load_text_embeddings, build_arrays, pick_device, time_split_idx
from src.utils import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multimodal checkpoint")
    p.add_argument("--tabular-path", default="data/processed/tabular.csv")
    p.add_argument("--text-emb-path", default="data/processed/text_embeddings.pt")
    p.add_argument("--checkpoint", default="checkpoints/multimodal/best_model.pt")
    p.add_argument("--out-metrics", default="checkpoints/multimodal/metrics_eval.json")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = pick_device()
    base = _load_base_table(args.tabular_path, args.max_rows)
    text_emb = _load_text_embeddings(args.text_emb_path)
    packed = build_arrays(base, text_emb)
    _tr, va_idx = time_split_idx(packed.dates, args.test_ratio)
    val_set = MMSet(packed.tab_x[va_idx], packed.txt_x[va_idx], packed.gph_x[va_idx], packed.y[va_idx])
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultiModalFusionModel(tabular_in_dim=packed.tab_x.shape[1]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ys, ps = [], []
    atts = []
    with torch.no_grad():
        for tab_x, txt_x, gph_x, y in loader:
            out = model(tab_x.to(device), txt_x.to(device), gph_x.to(device))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(out["logits"]).cpu().numpy())
            atts.append(out["attention"].cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    att = np.concatenate(atts, axis=0)
    thr = best_f1_threshold(y_true, y_prob)
    cls = binary_classification_metrics(y_true, y_prob, threshold=thr)
    result = {
        "attention_mean_tabular": float(att[:, 0].mean()),
        "attention_mean_text": float(att[:, 1].mean()),
        "attention_mean_graph": float(att[:, 2].mean()),
        **cls,
    }
    save_json(result, args.out_metrics)
    print(result)


if __name__ == "__main__":
    main()
