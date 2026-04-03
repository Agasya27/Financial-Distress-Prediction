"""
Train true multimodal model locally (Mac-safe):
- Tabular branch
- Text branch (FinBERT chunk embeddings from precompute file)
- Graph branch (degree proxy)
- Fusion + CVAE regularization
"""

from __future__ import annotations

import argparse
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models import MultiModalFusionModel
from src.utils import ensure_dir, now_iso, save_json


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train local multimodal fusion model")
    p.add_argument("--tabular-path", default="data/processed/tabular.csv")
    p.add_argument("--text-emb-path", default="data/processed/text_embeddings.pt")
    p.add_argument("--out-dir", default="checkpoints/multimodal")
    p.add_argument("--epochs", type=int, default=8, help="Max epochs (early stopping may stop sooner).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val PR-AUC does not improve for this many epochs (0 = disabled).",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=3,
        help="Minimum epochs before early stopping can trigger.",
    )
    p.add_argument(
        "--balance-train",
        action="store_true",
        help="Oversample minority class each epoch (recommended for rare distress label).",
    )
    p.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help="focal: harder-example weighting; often helps PR-AUC on imbalance.",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision even on CUDA.",
    )
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="plateau: reduce LR when val PR-AUC stalls (good for long GPU runs). none: fixed LR.",
    )
    p.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="Train-only: randomly zero each modality embedding with this probability (forces use of text/graph).",
    )
    p.add_argument(
        "--gate-entropy-coef",
        type=float,
        default=0.0,
        help="Subtract coef * gate_entropy from loss (higher entropy → more balanced tab/text/graph gates).",
    )
    return p.parse_args()


def _build_graph_degree(df: pd.DataFrame, graph_path: str = "data/processed/graph.pt") -> pd.Series:
    if not os.path.exists(graph_path):
        return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        edge_index = graph.edge_index.numpy()
        node_to_cik = getattr(graph, "node_to_cik", {})
        num_nodes = int(getattr(graph, "num_nodes", graph.x.shape[0]))
        deg = np.zeros(num_nodes, dtype=np.float32)
        if edge_index.size > 0:
            for n in edge_index[0]:
                deg[int(n)] += 1.0
        cik_deg = {int(node_to_cik[i]): float(deg[i]) for i in range(num_nodes) if i in node_to_cik}
        return df["cik"].astype(int).map(cik_deg).fillna(0.0)
    except Exception:
        return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def _load_base_table(tabular_path: str, max_rows: int) -> pd.DataFrame:
    if not os.path.exists(tabular_path):
        raise FileNotFoundError(f"Missing tabular file: {tabular_path}")
    df = pd.read_csv(tabular_path, low_memory=False)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["datadate"]).reset_index(drop=True)
    return df


def _load_text_embeddings(path: str, max_chunks: int = 10, emb_dim: int = 768) -> dict[str, torch.Tensor]:
    if not os.path.exists(path):
        return {}
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj.get("embeddings", {})
    out: dict[str, torch.Tensor] = {}
    for k, v in emb.items():
        t = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        if t.ndim == 2 and t.shape[1] == emb_dim:
            out[k] = t[:max_chunks].float()
    return out


@dataclass
class PackedData:
    tab_x: np.ndarray
    txt_x: np.ndarray
    gph_x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    feature_cols: list[str]


def build_arrays(df: pd.DataFrame, text_emb: dict[str, torch.Tensor]) -> PackedData:
    df = df.copy()
    df["graph_degree"] = _build_graph_degree(df)

    feature_cols = [c for c in df.columns if c not in {"cik", "datadate", "label"}]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    x_np = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(x_np, axis=0, keepdims=True)
    std = np.nanstd(x_np, axis=0, keepdims=True)
    std[std == 0] = 1.0
    tab_x = ((x_np - mean) / std).astype(np.float32)

    txt = []
    for _, r in df.iterrows():
        key = f"{int(r['cik'])}_{pd.Timestamp(r['datadate']).strftime('%Y-%m-%d')}"
        t = text_emb.get(key)
        if t is None:
            t = torch.zeros((10, 768), dtype=torch.float32)
        txt.append(t.numpy())
    txt_x = np.stack(txt, axis=0).astype(np.float32)  # (N,10,768)
    t_mean = np.nanmean(txt_x, axis=(0, 1), keepdims=True)
    t_std = np.nanstd(txt_x, axis=(0, 1), keepdims=True)
    t_std[t_std == 0] = 1.0
    txt_x = ((txt_x - t_mean) / t_std).astype(np.float32)
    txt_x = np.nan_to_num(txt_x, nan=0.0, posinf=0.0, neginf=0.0)

    gph_x = df[["graph_degree"]].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy(dtype=np.float32)
    dates = df["datadate"].to_numpy()
    return PackedData(tab_x=tab_x, txt_x=txt_x, gph_x=gph_x, y=y, dates=dates, feature_cols=feature_cols)


class MMSet(Dataset):
    def __init__(self, tab_x: np.ndarray, txt_x: np.ndarray, gph_x: np.ndarray, y: np.ndarray):
        self.tab_x = torch.tensor(tab_x, dtype=torch.float32)
        self.txt_x = torch.tensor(txt_x, dtype=torch.float32)
        self.gph_x = torch.tensor(gph_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tab_x[idx], self.txt_x[idx], self.gph_x[idx], self.y[idx]


def time_split_idx(dates: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(dates)
    n_train = int(len(dates) * (1.0 - ratio))
    tr = order[:n_train]
    va = order[n_train:]
    return tr, va


def evaluate(model: MultiModalFusionModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for tab_x, txt_x, gph_x, y in loader:
            tab_x = tab_x.to(device)
            txt_x = txt_x.to(device)
            gph_x = gph_x.to(device)
            logits = model(tab_x, txt_x, gph_x)["logits"].float()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p.append(probs)
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    if not np.all(np.isfinite(y_prob)):
        raise ValueError(
            "Model produced non-finite probabilities (NaN/Inf). Training likely diverged; "
            "try --no-amp, smaller --lr, or --loss bce."
        )
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cls_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    loss_name: str,
    pos_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    logits = logits.clamp(min=-40.0, max=40.0)
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    pt = probs * y + (1.0 - probs) * (1.0 - y)
    pt = pt.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.mean(((1.0 - pt) ** focal_gamma) * bce)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    device = pick_device()
    _set_seed(args.seed)

    base = _load_base_table(args.tabular_path, args.max_rows)
    text_emb = _load_text_embeddings(args.text_emb_path)
    packed = build_arrays(base, text_emb)
    tr_idx, va_idx = time_split_idx(packed.dates, args.test_ratio)

    train_set = MMSet(packed.tab_x[tr_idx], packed.txt_x[tr_idx], packed.gph_x[tr_idx], packed.y[tr_idx])
    val_set = MMSet(packed.tab_x[va_idx], packed.txt_x[va_idx], packed.gph_x[va_idx], packed.y[va_idx])

    y_tr = packed.y[tr_idx].astype(int)
    if args.balance_train:
        counts = np.bincount(y_tr, minlength=2)
        inv = 1.0 / np.maximum(counts.astype(np.float64), 1.0)
        w_per_class = inv / inv.sum() * 2.0
        sample_w = np.array([w_per_class[yi] for yi in y_tr], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    else:
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
        )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = MultiModalFusionModel(
        tabular_in_dim=packed.tab_x.shape[1],
        modality_dropout_p=max(0.0, min(0.9, args.modality_dropout)),
    ).to(device)
    pos_rate = max(float(np.mean(packed.y[tr_idx])), 1e-6)
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", factor=0.5, patience=4, min_lr=1e-6
        )

    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
            amp_autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            amp_autocast = lambda: torch.cuda.amp.autocast(enabled=True)
    else:
        scaler = None
        amp_autocast = nullcontext

    best_pr = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    hist: list[dict] = []
    epochs_no_gain = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_opt_steps = 0
        n_warn_nonfinite = 0
        for tab_x, txt_x, gph_x, y in train_loader:
            tab_x = tab_x.to(device)
            txt_x = txt_x.to(device)
            gph_x = gph_x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            # Epoch 1 + focal: full FP32 forward avoids CVAE/KL blowups in fp16.
            skip_amp = use_amp and epoch == 1 and args.loss == "focal"
            fwd_ctx = nullcontext() if skip_amp else amp_autocast()
            with fwd_ctx:
                out = model(tab_x, txt_x, gph_x)
            logits_f = out["logits"].float()
            y_f = y.float()
            pw_f = pos_weight.float()
            cls_loss = _cls_loss(logits_f, y_f, args.loss, pw_f, args.focal_gamma)
            cvae_raw = out["cvae_loss"].float()
            cvae_safe = torch.nan_to_num(cvae_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 500.0)
            attn_f = out["attention"].float()
            gate_ent = -(attn_f * (attn_f + 1e-8).log()).sum(dim=1).mean()
            loss = cls_loss + 0.01 * cvae_safe - float(args.gate_entropy_coef) * gate_ent

            if not torch.isfinite(loss):
                n_warn_nonfinite += 1
                if n_warn_nonfinite <= 3:
                    print("[WARN] Skipping step: non-finite loss")
                continue

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optim.step()
            running += float(loss.item())
            n_opt_steps += 1

        if n_warn_nonfinite > 3:
            print(f"[WARN] Skipped {n_warn_nonfinite} non-finite batches this epoch")

        roc, pr = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_opt_steps, 1),
            "val_roc_auc": roc,
            "val_pr_auc": pr,
            "lr": float(optim.param_groups[0]["lr"]),
        }
        hist.append(row)
        print(row)
        if scheduler is not None:
            scheduler.step(pr)

        improved = pr > best_pr + 1e-7
        if improved:
            best_pr = pr
            epochs_no_gain = 0
            torch.save({"model_state": model.state_dict(), "feature_cols": packed.feature_cols}, best_path)
        else:
            epochs_no_gain += 1

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_epochs
            and epochs_no_gain >= args.early_stopping_patience
        ):
            stopped_early = True
            print(f"[EARLY_STOP] no val PR-AUC gain for {epochs_no_gain} epochs (best={best_pr:.6f})")
            break

    save_json(
        {
            "created_at": now_iso(),
            "device": str(device),
            "epochs_run": len(hist),
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "loss": args.loss,
            "balance_train": args.balance_train,
            "amp": use_amp,
            "early_stopping_patience": args.early_stopping_patience,
            "stopped_early": stopped_early,
            "lr_scheduler": args.lr_scheduler,
            "modality_dropout": args.modality_dropout,
            "gate_entropy_coef": args.gate_entropy_coef,
            "seed": args.seed,
            "best_val_pr_auc": best_pr,
            "history": hist,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "feature_cols": packed.feature_cols,
        },
        os.path.join(args.out_dir, "metrics.json"),
    )
    print(f"[DONE] Saved multimodal checkpoint: {best_path}")


if __name__ == "__main__":
    main()
