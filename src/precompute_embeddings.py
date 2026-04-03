"""
Step 3: Precompute FinBERT [CLS] embeddings for MD&A text.

Loads MD&A plain-text files from ``data/raw/mda_texts/``, runs
``ProsusAI/finbert`` once, and saves ``data/processed/text_embeddings.pt``.

Supports **periodic checkpoints** and **resume** after Colab disconnects
(see ``data.embedding_checkpoint_every`` in ``config.yaml``).

Run once before training; training code must not import FinBERT again.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

FINBERT_MODEL = "ProsusAI/finbert"


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    """Select best available inference device (CUDA > MPS > CPU).

    Returns:
        ``torch.device`` instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sample_key(cik: int, datadate: str) -> str:
    """Build canonical key matching MD&A filenames and tabular rows.

    Args:
        cik: SEC CIK (integer).
        datadate: ISO date string ``YYYY-MM-DD``.

    Returns:
        Key string ``"{cik}_{datadate}"``.
    """
    return f"{int(cik)}_{datadate}"


def chunk_token_ids_to_tensors(
    token_ids: list[int],
    tokenizer: Any,
    max_length: int,
    overlap: int,
) -> list[torch.Tensor]:
    """Split token ids into overlapping windows for FinBERT.

    Each window is at most ``max_length`` tokens total (including
    ``[CLS]`` and ``[SEP]``).

    Args:
        token_ids: Token ids without special tokens, shape conceptually ``(T,)``.
        tokenizer: Hugging Face tokenizer (provides ``cls_token_id``, ``sep_token_id``, ``pad_token_id``).
        max_length: Maximum sequence length for the model (512 for FinBERT).
        overlap: Overlap in **content** tokens between consecutive chunks.

    Returns:
        List of ``input_ids`` tensors, each shape ``(max_length,)`` int64.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0

    max_content = max_length - 2  # room for CLS and SEP
    if max_content <= 0:
        return []
    stride = max(1, max_content - overlap)

    chunks: list[torch.Tensor] = []
    if not token_ids:
        return chunks

    start = 0
    while start < len(token_ids):
        piece = token_ids[start : start + max_content]
        ids = [cls_id] + piece + [sep_id]  # (<= max_length,)
        if len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        chunks.append(torch.tensor(ids, dtype=torch.long))  # (max_length,)
        if start + max_content >= len(token_ids):
            break
        start += stride
    return chunks


def embed_text_finbert(
    text: str,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_chunks: int,
    max_length: int,
    overlap: int,
) -> tuple[torch.Tensor, bool]:
    """Compute padded FinBERT [CLS] embeddings for one MD&A document.

    Args:
        text: Raw MD&A plain text (may be empty).
        tokenizer: FinBERT tokenizer.
        model: FinBERT model (768-dim hidden states).
        device: Torch device for inference.
        max_chunks: Maximum number of chunks to keep (from config).
        max_length: Tokens per chunk (512).
        overlap: Token overlap between chunks (50).

    Returns:
        Tuple ``(emb, missing)`` where ``emb`` has shape ``(max_chunks, 768)``
        and ``missing`` is True if text was empty or produced no chunks.
    """
    hidden = getattr(model.config, "hidden_size", 768)
    out = torch.zeros(max_chunks, hidden, dtype=torch.float32)  # (max_chunks, 768)

    if not text or not str(text).strip():
        return out, True

    token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    chunk_tensors = chunk_token_ids_to_tensors(
        token_ids, tokenizer, max_length=max_length, overlap=overlap
    )
    if not chunk_tensors:
        return out, True

    chunk_tensors = chunk_tensors[:max_chunks]

    model.eval()
    with torch.no_grad():
        batch = torch.stack(chunk_tensors, dim=0).to(device)  # (num_chunks, max_length)
        attention_mask = (batch != tokenizer.pad_token_id).long()  # (num_chunks, max_length)
        outputs = model(input_ids=batch, attention_mask=attention_mask)
        cls_vecs = outputs.last_hidden_state[:, 0, :]  # (num_chunks, 768)
        out[: cls_vecs.shape[0]] = cls_vecs.cpu().float()

    return out, False


def _load_partial_checkpoint(
    partial_path: str,
) -> tuple[dict[str, torch.Tensor], dict[str, bool], int]:
    """Load partial embedding checkpoint if present.

    Args:
        partial_path: Path to ``text_embeddings_partial.pt``.

    Returns:
        Tuple ``(embeddings, missing, next_row_index)`` for resume.
    """
    if not os.path.isfile(partial_path):
        return {}, {}, 0
    ckpt = torch.load(partial_path, map_location="cpu", weights_only=False)
    emb = ckpt.get("embeddings", {})
    miss = ckpt.get("missing", {})
    nxt = int(ckpt.get("next_row_index", 0))
    print(f"[INFO] Resuming from partial checkpoint: {partial_path} (next_row_index={nxt})")
    return emb, miss, nxt


def _save_partial_checkpoint(
    partial_path: str,
    embeddings: dict[str, torch.Tensor],
    missing: dict[str, bool],
    next_row_index: int,
) -> None:
    """Save partial state for resume after disconnect.

    Args:
        partial_path: Output path for partial file.
        embeddings: Embedding dict (keys → ``(max_chunks, 768)``).
        missing: Missing-text flags per key.
        next_row_index: Next row index in ``tabular.csv`` to process.
    """
    os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "missing": missing,
            "next_row_index": next_row_index,
        },
        partial_path,
    )


def run_precompute(config_path: str = "config.yaml") -> None:
    """Load tabular rows, embed MD&A texts, save ``text_embeddings.pt``.

    Checkpoints every ``data.embedding_checkpoint_every`` rows to
    ``data.embedding_partial_path`` and resumes if that file exists.

    Args:
        config_path: Path to ``config.yaml``.
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    max_chunks = int(data_cfg["max_text_chunks"])
    tabular_path = os.path.join("data", "processed", "tabular.csv")
    mda_dir = os.path.join("data", "raw", "mda_texts")
    out_path = os.path.join("data", "processed", "text_embeddings.pt")
    partial_path = data_cfg.get(
        "embedding_partial_path", "data/processed/text_embeddings_partial.pt"
    )
    save_every = int(data_cfg.get("embedding_checkpoint_every", 1000))

    if not os.path.isfile(tabular_path):
        raise FileNotFoundError(
            f"Missing {tabular_path}. Run Step 2 (src.data_pipeline) first."
        )

    import pandas as pd

    tab = pd.read_csv(tabular_path, low_memory=False)
    if "cik" not in tab.columns or "datadate" not in tab.columns:
        raise ValueError("tabular.csv must contain columns: cik, datadate")

    tab["datadate"] = pd.to_datetime(tab["datadate"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    device = pick_device()
    print(f"[INFO] Loading FinBERT ({FINBERT_MODEL}) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModel.from_pretrained(FINBERT_MODEL).to(device)
    model.eval()

    embeddings, missing, start_idx = _load_partial_checkpoint(partial_path)
    n_rows = len(tab)
    if start_idx > n_rows:
        start_idx = 0
        print("[WARN] Partial next_row_index past end of tabular; starting from 0.")

    tail = tab.iloc[start_idx:]
    for offset, row in enumerate(
        tqdm(tail.itertuples(index=False), total=len(tail), desc="FinBERT MD&A")
    ):
        row_idx = start_idx + offset

        cik = int(row.cik)
        datadate = str(row.datadate)
        key = build_sample_key(cik, datadate)
        path = os.path.join(mda_dir, f"{key}.txt")

        text = ""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                text = ""

        emb, is_missing = embed_text_finbert(
            text,
            tokenizer,
            model,
            device,
            max_chunks=max_chunks,
            max_length=512,
            overlap=50,
        )
        embeddings[key] = emb  # (max_chunks, 768)
        missing[key] = is_missing

        processed_count = row_idx + 1
        if save_every > 0 and processed_count % save_every == 0:
            _save_partial_checkpoint(
                partial_path, embeddings, missing, processed_count
            )
            tqdm.write(
                f"[CHECKPOINT] rows={processed_count}/{n_rows} -> {partial_path}"
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"embeddings": embeddings, "missing": missing}, out_path)
    n_miss = sum(1 for v in missing.values() if v)
    print(f"[DONE] Saved {out_path}")
    print(f"[INFO] Keys: {len(embeddings)}  missing_text: {n_miss}")
    if os.path.isfile(partial_path):
        try:
            os.remove(partial_path)
            print(f"[INFO] Removed partial checkpoint: {partial_path}")
        except OSError:
            print(f"[WARN] Could not remove partial file: {partial_path}")


if __name__ == "__main__":
    run_precompute()
