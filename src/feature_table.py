"""
Tabular + optional MD&A stats + graph degree features for sklearn / Streamlit.

Kept separate from `dataset.py` so the Streamlit app does not import PyTorch `Dataset` / multimodal stack at startup.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

BASE_TABULAR_DROP = {"cik", "datadate", "label"}


def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datadate"] = pd.to_datetime(out["datadate"], errors="coerce")
    out = out.dropna(subset=["datadate"])
    return out


def _add_text_stats(df: pd.DataFrame, mda_dir: str) -> pd.DataFrame:
    """Add lightweight text features without loading transformer models."""
    out = df.copy()
    chars: list[int] = []
    words: list[int] = []

    for _, row in out.iterrows():
        file_name = f"{int(row['cik'])}_{row['datadate'].strftime('%Y-%m-%d')}.txt"
        path = os.path.join(mda_dir, file_name)
        if not os.path.exists(path):
            chars.append(0)
            words.append(0)
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chars.append(len(text))
            words.append(len(text.split()))
        except OSError:
            chars.append(0)
            words.append(0)

    out["mda_char_count"] = chars
    out["mda_word_count"] = words
    out["mda_log_char_count"] = np.log1p(out["mda_char_count"].astype(float))
    out["mda_log_word_count"] = np.log1p(out["mda_word_count"].astype(float))
    return out


def _add_graph_degree(df: pd.DataFrame, graph_path: str) -> pd.DataFrame:
    """Add per-CIK degree from PyG graph if available (lazy torch import)."""
    out = df.copy()
    if not os.path.exists(graph_path):
        out["graph_degree"] = 0.0
        return out

    try:
        import torch
    except Exception:
        out["graph_degree"] = 0.0
        return out

    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        edge_index = graph.edge_index.numpy()
        node_to_cik = getattr(graph, "node_to_cik", {})
        num_nodes = int(getattr(graph, "num_nodes", graph.x.shape[0]))
        degrees = np.zeros(num_nodes, dtype=np.float32)
        if edge_index.size > 0:
            src_nodes = edge_index[0]
            for n in src_nodes:
                degrees[int(n)] += 1.0
        cik_degree = {int(node_to_cik[i]): float(degrees[i]) for i in range(num_nodes) if i in node_to_cik}
        out["graph_degree"] = out["cik"].astype(int).map(cik_degree).fillna(0.0)
    except Exception:
        out["graph_degree"] = 0.0
    return out


def load_feature_table(
    tabular_path: str = "data/processed/tabular.csv",
    mda_dir: str = "data/raw/mda_texts",
    graph_path: str = "data/processed/graph.pt",
    use_text_stats: bool = True,
    use_graph_degree: bool = True,
    max_rows: int = 0,
    required_features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    """
    Build model-ready matrix X and label vector y.

    Returns:
        X_df: numeric feature dataframe
        y: binary labels
        feature_columns: ordered feature names
        base_df: metadata dataframe (`cik`, `datadate`, `label`) for downstream eval
    """
    if not os.path.exists(tabular_path):
        raise FileNotFoundError(f"Missing tabular file: {tabular_path}")

    df = pd.read_csv(tabular_path, low_memory=False)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    df = _coerce_datetime(df)
    if "label" not in df.columns:
        raise ValueError("Tabular data missing 'label' column.")

    # Optional feature sources. For deployment, callers can disable these to avoid
    # loading large artifacts; if `required_features` asks for these columns we
    # will add zero-filled placeholders later.
    if use_text_stats:
        df = _add_text_stats(df, mda_dir)
    if use_graph_degree:
        df = _add_graph_degree(df, graph_path)

    numeric_cols: list[str] = [c for c in df.columns if c not in BASE_TABULAR_DROP]

    # Downcast to reduce memory footprint for deployment environments.
    x_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    if required_features:
        # Ensure a stable schema for already-trained models without forcing the
        # app to load MD&A files or graph artifacts.
        for c in required_features:
            if c not in x_df.columns:
                x_df[c] = np.float32(0.0)
        # Drop extras and preserve ordering
        x_df = x_df[[c for c in required_features]]
        numeric_cols = list(x_df.columns)
    y = df["label"].astype(int).to_numpy()
    base_df = df[["cik", "datadate", "label"]].copy()
    return x_df, y, numeric_cols, base_df
