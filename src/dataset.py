"""
Dataset utilities for multimodal financial distress modeling.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from src.feature_table import load_feature_table


class FinancialDistressDataset(Dataset):
    """
    Multimodal dataset with modality masking.

    Returns per sample:
    - tabular: (25,) float tensor or None
    - text: (max_chunks, 768) float tensor or None
    - graph_node_idx: int
    - label: float
    - mask: dict[str, bool] with True=available
    """

    def __init__(
        self,
        tabular_path: str = "data/processed/tabular.csv",
        text_emb_path: str = "data/processed/text_embeddings.pt",
        graph_path: str = "data/processed/graph.pt",
        training: bool = True,
        mask_prob: float = 0.3,
        max_rows: int = 0,
    ) -> None:
        super().__init__()
        if not os.path.exists(tabular_path):
            raise FileNotFoundError(f"Missing tabular file: {tabular_path}")

        df = pd.read_csv(tabular_path, low_memory=False)
        if max_rows > 0:
            df = df.iloc[:max_rows].copy()
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
        df = df.dropna(subset=["datadate"]).reset_index(drop=True)

        self.training = training
        self.mask_prob = float(mask_prob)
        self.max_chunks = 10
        self.emb_dim = 768

        feature_cols = [c for c in df.columns if c not in {"cik", "datadate", "label"}]
        tab = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        tab = tab.replace([np.inf, -np.inf], np.nan)
        tab = tab.fillna(tab.median(numeric_only=True)).fillna(0.0)
        scaler = RobustScaler()
        self.tabular = scaler.fit_transform(tab.to_numpy(dtype=np.float32)).astype(np.float32)
        self.feature_cols = feature_cols

        self.labels = df["label"].astype(float).to_numpy(dtype=np.float32)
        self.keys = [f"{int(c)}_{pd.Timestamp(d).strftime('%Y-%m-%d')}" for c, d in zip(df["cik"], df["datadate"])]
        self.ciks = df["cik"].astype(int).to_numpy()
        self.graph_node_idx = np.arange(len(df), dtype=np.int64)

        self.text_embeddings: dict[str, torch.Tensor] = {}
        self.text_missing: dict[str, bool] = {}
        if os.path.exists(text_emb_path):
            obj = torch.load(text_emb_path, map_location="cpu", weights_only=False)
            self.text_embeddings = obj.get("embeddings", {})
            self.text_missing = obj.get("missing", {})

        # Optional graph mapping from graph.pt's node_to_cik.
        self.cik_to_node: dict[int, int] = {}
        if os.path.exists(graph_path):
            try:
                graph = torch.load(graph_path, map_location="cpu", weights_only=False)
                node_to_cik = getattr(graph, "node_to_cik", {})
                self.cik_to_node = {int(v): int(k) for k, v in node_to_cik.items()}
                self.graph_node_idx = np.array([self.cik_to_node.get(int(c), -1) for c in self.ciks], dtype=np.int64)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.labels)

    def _modality_mask(self, mask: dict[str, bool]) -> dict[str, bool]:
        if not self.training:
            return mask
        out = dict(mask)
        for k in ("tabular", "text", "graph"):
            if out[k] and random.random() < self.mask_prob:
                out[k] = False
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tab = torch.tensor(self.tabular[idx], dtype=torch.float32)
        key = self.keys[idx]

        txt = self.text_embeddings.get(key)
        txt_missing = bool(self.text_missing.get(key, txt is None))
        if txt is None:
            txt = torch.zeros((self.max_chunks, self.emb_dim), dtype=torch.float32)
        elif not isinstance(txt, torch.Tensor):
            txt = torch.tensor(txt, dtype=torch.float32)
        txt = txt[: self.max_chunks].float()
        if txt.shape[0] < self.max_chunks:
            pad = torch.zeros((self.max_chunks - txt.shape[0], self.emb_dim), dtype=torch.float32)
            txt = torch.cat([txt, pad], dim=0)

        node_idx = int(self.graph_node_idx[idx])
        mask = {"tabular": True, "text": not txt_missing, "graph": node_idx >= 0}
        mask = self._modality_mask(mask)

        sample = {
            "tabular": tab if mask["tabular"] else None,
            "text": txt if mask["text"] else None,
            "graph_node_idx": node_idx,
            "label": float(self.labels[idx]),
            "mask": mask,
        }
        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples that may contain None modalities."""
    tab_dim = 25
    text_shape = (10, 768)
    for s in batch:
        if s["tabular"] is not None:
            tab_dim = int(s["tabular"].shape[0])
            break
    tabs, texts, nodes, labels = [], [], [], []
    masks = {"tabular": [], "text": [], "graph": []}
    for s in batch:
        tabs.append(s["tabular"] if s["tabular"] is not None else torch.zeros(tab_dim))
        texts.append(s["text"] if s["text"] is not None else torch.zeros(text_shape))
        nodes.append(int(s["graph_node_idx"]))
        labels.append(float(s["label"]))
        for k in masks:
            masks[k].append(bool(s["mask"][k]))

    return {
        "tabular": torch.stack(tabs, dim=0),
        "text": torch.stack(texts, dim=0),
        "graph_node_idx": torch.tensor(nodes, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "mask": {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()},
    }


def time_split(
    base_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> np.ndarray:
    """Return boolean mask for train rows using chronological split."""
    ordered = base_df.sort_values("datadate").reset_index(drop=False)
    split_idx = int(len(ordered) * (1.0 - test_ratio))
    train_idx = set(ordered.iloc[:split_idx]["index"].tolist())
    mask = base_df.index.to_series().isin(train_idx).to_numpy()
    return mask
