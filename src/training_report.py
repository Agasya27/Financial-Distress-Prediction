"""
Integrated training / evaluation visuals: multimodal fusion + local_lite baseline.

Reads JSON artifacts under checkpoints/ and builds a Plotly figure (or HTML file).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import load_json


def _safe_load(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def build_integrated_training_figure(
    multimodal_train_path: str = "checkpoints/multimodal/metrics.json",
    multimodal_eval_path: str = "checkpoints/multimodal/metrics_eval.json",
    local_meta_path: str = "checkpoints/local_lite/meta.json",
    local_train_metrics_path: str = "checkpoints/local_lite/metrics.json",
    local_eval_path: str = "checkpoints/local_lite/metrics_eval.json",
    blend_metrics_path: str = "checkpoints/blend_metrics_eval.json",
    root: str = ".",
) -> go.Figure:
    """Assemble a multi-panel figure from on-disk checkpoint JSON."""
    def p(rel: str) -> str:
        return rel if root == "." else os.path.join(root, rel)

    mm_train = _safe_load(p(multimodal_train_path))
    mm_eval = _safe_load(p(multimodal_eval_path))
    loc_meta = _safe_load(p(local_meta_path))
    loc_train = _safe_load(p(local_train_metrics_path))
    loc_eval = _safe_load(p(local_eval_path))
    blend_eval = _safe_load(p(blend_metrics_path))

    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]],
        subplot_titles=(
            "Multimodal fusion — epochs (time-based validation)",
            "Local lite — candidate models (validation PR-AUC)",
            "Holdout metrics — multimodal vs local lite vs blend (if available)",
            "Local lite — training holdout vs evaluation run",
        ),
        vertical_spacing=0.07,
        row_heights=[0.28, 0.22, 0.28, 0.22],
    )

    # Row 1: multimodal history
    if mm_train and mm_train.get("history"):
        hist = mm_train["history"]
        epochs = [int(h["epoch"]) for h in hist]
        loss = [float(h["train_loss"]) for h in hist]
        vroc = [float(h["val_roc_auc"]) for h in hist]
        vpr = [float(h["val_pr_auc"]) for h in hist]
        fig.add_trace(
            go.Scatter(x=epochs, y=loss, name="train loss", mode="lines+markers", line=dict(color="#636EFA")),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=vroc, name="val ROC-AUC", mode="lines+markers", line=dict(color="#00CC96")),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=vpr, name="val PR-AUC", mode="lines+markers", line=dict(color="#EF553B")),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="train loss", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="AUC", row=1, col=1, secondary_y=True, rangemode="tozero")
    else:
        fig.add_annotation(
            text="No multimodal training history (run: python3 -m src.train --mode multimodal)",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    # Row 2: local lite model selection
    selection = (loc_meta or {}).get("model_selection")
    if isinstance(selection, list) and selection:
        names = [str(x.get("name", "")) for x in selection]
        prs = [float(x.get("val_pr_auc", 0.0)) for x in selection]
        fig.add_trace(
            go.Bar(x=names, y=prs, name="val PR-AUC", marker_color="#AB63FA", showlegend=False),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="val PR-AUC", row=2, col=1, rangemode="tozero")
    else:
        fig.add_annotation(
            text="No model_selection in meta (re-run local_lite train for candidate bars)",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=2,
            col=1,
        )

    # Row 3: compare eval metrics multimodal vs local lite
    metric_keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    labels = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    mm_vals: list[float | None] = []
    loc_vals: list[float | None] = []
    blend_vals: list[float | None] = []
    for k in metric_keys:
        mm_vals.append(float(mm_eval[k]) if mm_eval and k in mm_eval else None)
        loc_vals.append(float(loc_eval[k]) if loc_eval and k in loc_eval else None)
        blend_vals.append(float(blend_eval[k]) if blend_eval and k in blend_eval else None)

    if any(v is not None for v in mm_vals) or any(v is not None for v in loc_vals) or any(
        v is not None for v in blend_vals
    ):
        show_labels: list[str] = []
        mm_plot: list[float] = []
        loc_plot: list[float] = []
        blend_plot: list[float] = []
        for lab, mv, lv, bv in zip(labels, mm_vals, loc_vals, blend_vals):
            if mv is None and lv is None and bv is None:
                continue
            show_labels.append(lab)
            mm_plot.append(float(mv) if mv is not None else float("nan"))
            loc_plot.append(float(lv) if lv is not None else float("nan"))
            blend_plot.append(float(bv) if bv is not None else float("nan"))

        fig.add_trace(
            go.Bar(name="Multimodal (eval)", x=show_labels, y=mm_plot, marker_color="#636EFA"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(name="Local lite (eval)", x=show_labels, y=loc_plot, marker_color="#FF9F43"),
            row=3,
            col=1,
        )
        if any(v is not None and not np.isnan(v) for v in blend_plot):
            fig.add_trace(
                go.Bar(name="Blend (eval)", x=show_labels, y=blend_plot, marker_color="#2CA02C"),
                row=3,
                col=1,
            )
        fig.update_yaxes(title_text="score", row=3, col=1, rangemode="tozero")
        fig.update_layout(barmode="group")
    else:
        fig.add_annotation(
            text="Run evaluate for both modes to populate holdout bars",
            xref="x3 domain",
            yref="y3 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=3,
            col=1,
        )

    # Row 4: local train holdout vs eval (same metrics file may differ if data changed)
    if loc_train or loc_eval:
        keys4 = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
        labs4 = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
        t_vals = [float(loc_train[k]) if loc_train and k in loc_train else None for k in keys4]
        e_vals = [float(loc_eval[k]) if loc_eval and k in loc_eval else None for k in keys4]
        sl, tv, ev = [], [], []
        for lab, a, b in zip(labs4, t_vals, e_vals):
            if a is None and b is None:
                continue
            sl.append(lab)
            tv.append(a if a is not None else 0.0)
            ev.append(b if b is not None else 0.0)
        if sl:
            fig.add_trace(
                go.Bar(name="Local train holdout", x=sl, y=tv, marker_color="#2ECC71", showlegend=True),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Bar(name="Local evaluate run", x=sl, y=ev, marker_color="#E74C3C", showlegend=True),
                row=4,
                col=1,
            )
        fig.update_yaxes(title_text="score", row=4, col=1, rangemode="tozero")
    else:
        fig.add_annotation(
            text="No local lite metrics JSON",
            xref="x4 domain",
            yref="y4 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=4,
            col=1,
        )

    fig.update_layout(
        height=1100,
        title_text="Financial distress — integrated training & evaluation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=40),
    )
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Write integrated training report HTML (Plotly).")
    parser.add_argument("--out", default="checkpoints/training_report.html", help="Output HTML path")
    parser.add_argument("--root", default=".", help="Project root (prepended to default paths)")
    args = parser.parse_args()
    fig = build_integrated_training_figure(root=args.root)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.root, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
