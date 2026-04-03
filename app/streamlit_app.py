"""
Lightweight Streamlit dashboard for local MacBook workflow.
"""

from __future__ import annotations

import json
import os
from typing import Any


_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DOTENV_HINT: str | None = None


def _load_app_env() -> None:
    """Load `.env` / `.env.local` / fallback `.env.example` before other imports use os.environ."""
    global _DOTENV_HINT
    _DOTENV_HINT = None
    try:
        from src.env_bootstrap import load_financial_distress_dotenv

        _DOTENV_HINT = load_financial_distress_dotenv(_APP_ROOT)
    except ImportError:
        pass


_load_app_env()

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_table import load_feature_table
from src.openrouter_client import (
    DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL,
    get_openrouter_api_key,
    summarize_prediction_bundle,
    summarize_training_artifacts,
)
from src.prediction_utils import bankruptcy_risk_band, bankruptcy_summary
from src.training_report import build_integrated_training_figure


DEFAULT_MODEL = "checkpoints/local_lite/model.joblib"
DEFAULT_META = "checkpoints/local_lite/meta.json"
DEFAULT_METRICS = "checkpoints/local_lite/metrics.json"
DEFAULT_MM_TRAIN_METRICS = "checkpoints/multimodal/metrics.json"
DEFAULT_MM_EVAL_METRICS = "checkpoints/multimodal/metrics_eval.json"
DEFAULT_LOCAL_EVAL_METRICS = "checkpoints/local_lite/metrics_eval.json"
DEFAULT_BLEND_METRICS = "checkpoints/blend_metrics_eval.json"
DEFAULT_TRAINING_REPORT_HTML = "checkpoints/training_report.html"
DEFAULT_TABULAR = "data/processed/tabular.csv"
DEFAULT_MDA_DIR = "data/raw/mda_texts"
DEFAULT_GRAPH = "data/processed/graph.pt"


@st.cache_data(show_spinner="Loading model, features, and portfolio scores (cached until paths change)…")
def _cached_scoring_bundle(
    model_path: str,
    tabular_path: str,
    mda_dir: str,
    graph_path: str,
    use_text_stats: bool,
    use_graph_degree: bool,
) -> tuple[Any, pd.DataFrame, Any, pd.DataFrame, np.ndarray]:
    """Load sklearn model + full feature table + portfolio probabilities. Heavy; keyed by paths/flags."""
    model = joblib.load(model_path)
    x_df, y, _, base_df = load_feature_table(
        tabular_path=tabular_path,
        mda_dir=mda_dir,
        graph_path=graph_path,
        use_text_stats=use_text_stats,
        use_graph_degree=use_graph_degree,
    )
    probs = model.predict_proba(x_df)[:, 1]
    return model, x_df, y, base_df, probs


@st.cache_data(show_spinner=False)
def _cached_json_file(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        out = json.load(f)
    return out if isinstance(out, dict) else {}


def _status_from_risk(risk: float, threshold: float) -> str:
    if risk >= threshold + 0.08:
        return "High Risk"
    if risk >= threshold - 0.08:
        return "Moderate Risk"
    return "Healthy"


def _safe_float(v: object, fallback: float = 0.0) -> float:
    try:
        if v is None:
            return fallback
        return float(v)
    except Exception:
        return fallback


def _coerce_json_context(obj: Any, depth: int = 0) -> Any:
    """Make nested structures safe for json.dumps (numpy/pandas/NaN)."""
    if depth > 28:
        return "…"
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return [_coerce_json_context(x, depth + 1) for x in obj.ravel().tolist()[:500]]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 400:
                out["_truncated_"] = True
                break
            out[str(k)] = _coerce_json_context(v, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_coerce_json_context(x, depth + 1) for x in obj[:400]]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    try:
        if hasattr(obj, "item"):
            return _coerce_json_context(obj.item(), depth + 1)
    except Exception:
        pass
    return str(obj)[:2000]


def _build_manual_feature_row(
    feature_cols: list[str],
    dte: float | None,
    roa: float | None,
    roe: float | None,
    current_ratio: float | None,
    net_profit_margin: float | None,
    revenue_growth_pct: float | None,
    mda_text: str,
    include_graph: bool,
    graph_degree_input: float | None,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    row = {c: np.nan for c in feature_cols}

    # Core ratios directly map where available.
    mapping = {
        "debt_to_equity": dte,
        "return_on_assets": roa,
        "return_on_equity": roe,
        "current_ratio": current_ratio,
        # Dataset feature closest to net profit margin.
        "operating_margin": net_profit_margin,
    }
    for k, v in mapping.items():
        if k in row and v is not None:
            row[k] = float(v)

    # Optional helper mapping for revenue growth to existing turnover proxy.
    if "asset_turnover" in row and revenue_growth_pct is not None:
        row["asset_turnover"] = float(revenue_growth_pct) / 100.0

    if "mda_word_count" in row:
        wc = len(mda_text.split()) if mda_text.strip() else 0
        row["mda_word_count"] = float(wc)
    if "mda_char_count" in row:
        cc = len(mda_text) if mda_text.strip() else 0
        row["mda_char_count"] = float(cc)
    if "mda_log_word_count" in row:
        row["mda_log_word_count"] = float(np.log1p(max(_safe_float(row.get("mda_word_count"), 0.0), 0.0)))
    if "mda_log_char_count" in row:
        row["mda_log_char_count"] = float(np.log1p(max(_safe_float(row.get("mda_char_count"), 0.0), 0.0)))

    if "graph_degree" in row:
        if include_graph:
            if graph_degree_input is not None:
                row["graph_degree"] = float(graph_degree_input)
            else:
                row["graph_degree"] = float(reference_df["graph_degree"].median()) if "graph_degree" in reference_df else 0.0
        else:
            row["graph_degree"] = 0.0

    return pd.DataFrame([row], columns=feature_cols)


def _closest_companies(dataset_x: pd.DataFrame, dataset_meta: pd.DataFrame, user_x: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    # Distance in standardized numeric space.
    med = dataset_x.median()
    std = dataset_x.std().replace(0, 1.0).fillna(1.0)
    z_data = (dataset_x.fillna(med) - med) / std
    z_user = (user_x.fillna(med) - med) / std

    diffs = z_data.to_numpy() - z_user.to_numpy()
    dists = np.sqrt(np.sum(np.square(diffs), axis=1))
    nearest_idx = np.argsort(dists)[:n]

    out = dataset_meta.iloc[nearest_idx].copy()
    out["distance"] = dists[nearest_idx]
    return out


def _top_factor_explanations(
    user_map: dict[str, float | None],
    ref_df: pd.DataFrame,
) -> list[str]:
    # Positive sign means "higher increases risk"; negative means "lower increases risk".
    risk_dirs = {
        "debt_to_equity": +1.0,
        "return_on_assets": -1.0,
        "return_on_equity": -1.0,
        "current_ratio": -1.0,
        "operating_margin": -1.0,
        "asset_turnover": -1.0,
    }
    labels = {
        "debt_to_equity": "Debt-to-Equity",
        "return_on_assets": "ROA",
        "return_on_equity": "ROE",
        "current_ratio": "Current Ratio",
        "operating_margin": "Net Profit Margin (proxy: Operating Margin)",
        "asset_turnover": "Revenue Growth proxy (Asset Turnover)",
    }

    scored: list[tuple[float, str]] = []
    for feat, value in user_map.items():
        if value is None or feat not in ref_df.columns:
            continue
        avg = _safe_float(ref_df[feat].mean(), np.nan)
        std = _safe_float(ref_df[feat].std(), np.nan)
        if not np.isfinite(avg) or not np.isfinite(std) or std == 0:
            continue
        z = (float(value) - avg) / std
        directed = z * risk_dirs.get(feat, 0.0)
        msg = f"{labels.get(feat, feat)} is {abs(z):.2f} std {'above' if z >= 0 else 'below'} dataset average."
        scored.append((directed, msg))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for s, m in scored if s > 0][:3]


def _prepare_model_input_from_user_df(
    user_df: pd.DataFrame,
    trained_features: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build robust model input from uploaded CSV.

    Supports simple aliases like:
    - net_profit_margin -> operating_margin
    - revenue_growth or revenue_growth_pct -> asset_turnover proxy
    """
    df = user_df.copy()
    alias_map = {
        "net_profit_margin": "operating_margin",
        "revenue_growth": "asset_turnover",
        "revenue_growth_pct": "asset_turnover",
        "roa": "return_on_assets",
        "roe": "return_on_equity",
        "de_ratio": "debt_to_equity",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Convert percentage-style growth if obvious.
    if "asset_turnover" in df.columns:
        vals = pd.to_numeric(df["asset_turnover"], errors="coerce")
        if np.nanmedian(np.abs(vals.to_numpy(dtype=float))) > 3:
            df["asset_turnover"] = vals / 100.0

    # Fill missing training features from reference medians.
    for col in trained_features:
        if col not in df.columns:
            df[col] = np.nan
    model_input = df[trained_features].apply(pd.to_numeric, errors="coerce")
    ref_med = reference_df.median(numeric_only=True)
    model_input = model_input.fillna(ref_med).fillna(0.0)
    return model_input


def _confidence_from_margin(risk: float, threshold: float) -> float:
    margin = abs(float(risk) - float(threshold))
    return float(np.clip(margin / 0.25, 0.0, 1.0))


def _metric_row_ui(prefix: str, data: dict) -> None:
    keys = ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold")
    if not data or not any(k in data for k in keys):
        st.caption(f"No classification metrics in `{prefix}`. Run `python3 -m src.evaluate --mode …`.")
        return
    a, b, c = st.columns(3)
    a.metric("ROC-AUC", f"{data.get('roc_auc', float('nan')):.4f}")
    b.metric("PR-AUC", f"{data.get('pr_auc', float('nan')):.4f}")
    c.metric("F1", f"{data.get('f1', float('nan')):.4f}")
    d, e, f = st.columns(3)
    d.metric("Precision", f"{data.get('precision', float('nan')):.4f}")
    e.metric("Recall", f"{data.get('recall', float('nan')):.4f}")
    f.metric("Threshold", f"{data.get('threshold', float('nan')):.4f}")
    att = []
    for k in ("attention_mean_tabular", "attention_mean_text", "attention_mean_graph"):
        if k in data:
            short = k.replace("attention_mean_", "")
            att.append(f"{short}: {float(data[k]):.3f}")
    if att:
        st.caption("Fusion attention (mean): " + " · ".join(att))


def _openrouter_model_resolved() -> str:
    """Same rule as `get_openrouter_model()` in openrouter_client (kept local to avoid stale-import issues)."""
    return (os.getenv("OPENROUTER_MODEL") or OPENROUTER_DEFAULT_MODEL).strip()


def _openrouter_key_resolved() -> str | None:
    """API key from `.env` next to this app (always `_APP_ROOT`), then Streamlit secrets."""
    env = get_openrouter_api_key(_APP_ROOT)
    if env:
        return env
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip() or None
    except Exception:
        pass
    return None


def _openrouter_missing_key_help() -> str:
    """Safe diagnostic (no secret values)."""
    env_path = os.path.join(_APP_ROOT, ".env")
    exists = os.path.isfile(env_path)
    var_set = bool((os.getenv("OPENROUTER_API_KEY") or "").strip())
    try:
        from src.env_bootstrap import read_openrouter_key_from_dotenv_files

        in_any_dotenv = bool(read_openrouter_key_from_dotenv_files(_APP_ROOT))
    except Exception:
        in_any_dotenv = False
    bits = [
        f"Expected **`.env`** next to `app/` and `src/`: `{env_path}` — exists: **{exists}**.",
        f"`OPENROUTER_API_KEY` in process env: **{'yes' if var_set else 'no'}**.",
        f"Found in `.env` / `.env.local` / `.env.example` via direct file parse: **{'yes' if in_any_dotenv else 'no'}**.",
    ]
    if exists and in_any_dotenv and not var_set:
        bits.append(
            "A key exists on disk but not in `os.environ` (Streamlit/host quirk). "
            "This app version reads the key from the file anyway — **hard-refresh** the browser; if it still fails, restart Streamlit."
        )
    bits.append(
        "On **SSH / Lightning**, `.env` must be on that machine. "
        "Avoid a space after `=` in `.env` (`OPENROUTER_API_KEY=sk-...`). "
        "`unset OPENROUTER_API_KEY` if the shell exported an empty value."
    )
    return " ".join(bits)


def _blend_metrics_ui(path: str, data: dict) -> None:
    if not data:
        st.caption(f"No blend metrics at `{path}`. Run `python3 -m src.evaluate_blend` after training both models.")
        return
    st.caption("Blended holdout: `α·P(local_lite) + (1−α)·P(multimodal)` with α grid-searched on PR-AUC.")
    a, b, c, d = st.columns(4)
    a.metric("Best α", f"{_safe_float(data.get('best_alpha'), float('nan')):.4f}")
    b.metric("Blend PR-AUC", f"{_safe_float(data.get('blend_pr_auc'), float('nan')):.4f}")
    c.metric("Blend ROC-AUC", f"{_safe_float(data.get('blend_roc_auc'), float('nan')):.4f}")
    d.metric("Blend F1 @ thr", f"{_safe_float(data.get('f1'), float('nan')):.4f}")
    e, f, g, h = st.columns(4)
    e.metric("Lite-only PR-AUC", f"{_safe_float(data.get('lite_only_pr_auc'), float('nan')):.4f}")
    f.metric("MM-only PR-AUC", f"{_safe_float(data.get('mm_only_pr_auc'), float('nan')):.4f}")
    g.metric("Lite-only ROC", f"{_safe_float(data.get('lite_only_roc_auc'), float('nan')):.4f}")
    h.metric("MM-only ROC", f"{_safe_float(data.get('mm_only_roc_auc'), float('nan')):.4f}")


def _ai_prediction_panel(
    *,
    context: dict[str, Any],
    state_prefix: str,
) -> None:
    st.markdown("#### AI narrative (OpenRouter)")
    api_key = _openrouter_key_resolved()
    if not api_key:
        st.caption("Configure **`OPENROUTER_API_KEY`** in project `.env` (or Streamlit secrets) to enable.")
        st.caption(_openrouter_missing_key_help())
        return
    model = _openrouter_model_resolved()
    txt_key = f"{state_prefix}_ai_text"
    err_key = f"{state_prefix}_ai_err"
    safe_context = _coerce_json_context(context) if isinstance(context, dict) else context

    # `st.form` + submit is more reliable than `st.button` for long-running calls (avoids rerun edge cases).
    with st.form(key=f"ai_form_{state_prefix}"):
        submitted = st.form_submit_button("Generate AI summary")

    if submitted:
        with st.spinner("Calling OpenRouter…"):
            try:
                text = summarize_prediction_bundle(api_key, context=safe_context, model=model or None)
                st.session_state[txt_key] = text
                st.session_state.pop(err_key, None)
                st.success("Received response from OpenRouter.")
                if text and str(text).strip():
                    st.markdown(text)
                else:
                    st.warning("Model returned an empty string.")
            except Exception as exc:
                st.session_state[err_key] = str(exc)
                st.session_state.pop(txt_key, None)
                st.error(f"OpenRouter request failed: {exc}")

    if not submitted:
        if st.session_state.get(err_key):
            st.error(st.session_state[err_key])
        elif txt_key in st.session_state:
            val = st.session_state[txt_key]
            if val and str(val).strip():
                st.markdown("**Last summary:**")
                st.markdown(val)
            else:
                st.warning(
                    "Last run returned an empty reply. Try another `OPENROUTER_MODEL` in `.env` and submit again."
                )


def _training_snapshot_for_llm(
    mm_train: dict,
    mm_eval: dict,
    loc_eval: dict,
    blend: dict,
    loc_meta: dict,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "local_lite_eval_metrics": (
            {k: loc_eval.get(k) for k in ("roc_auc", "pr_auc", "f1", "threshold") if k in loc_eval} if loc_eval else {}
        ),
        "multimodal_eval_metrics": (
            {k: mm_eval.get(k) for k in ("roc_auc", "pr_auc", "f1", "threshold") if k in mm_eval} if mm_eval else {}
        ),
    }
    if blend:
        snap["blend"] = {
            "best_alpha": blend.get("best_alpha"),
            "blend_pr_auc": blend.get("blend_pr_auc"),
            "blend_roc_auc": blend.get("blend_roc_auc"),
            "lite_only_pr_auc": blend.get("lite_only_pr_auc"),
            "mm_only_pr_auc": blend.get("mm_only_pr_auc"),
        }
    if mm_train:
        hist = mm_train.get("history") or []
        last = hist[-1] if hist else {}
        snap["multimodal_training_tail"] = {
            "epochs_logged": len(hist),
            "last_epoch": last.get("epoch"),
            "last_val_pr_auc": last.get("val_pr_auc"),
            "last_val_roc_auc": last.get("val_roc_auc"),
        }
        cfg = {}
        for k in ("modality_dropout", "gate_entropy_coef", "loss", "best_val_pr_auc"):
            if k in mm_train:
                cfg[k] = mm_train[k]
        if cfg:
            snap["multimodal_train_config"] = cfg
    sel = (loc_meta or {}).get("model_selection")
    if isinstance(sel, list) and sel:
        snap["local_lite_model_pick"] = sorted(sel, key=lambda x: float(x.get("val_pr_auc", 0)), reverse=True)[:3]
    return snap


st.set_page_config(
    page_title="Financial Distress Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Financial Distress & Bankruptcy Risk")
st.caption(
    "Model output: **probability of financial distress** (same as bankruptcy-related label in the ECL dataset). "
    "Higher score → higher predicted risk. Not legal or investment advice."
)

with st.sidebar:
    st.header("Paths")
    model_path = st.text_input("Model path", DEFAULT_MODEL)
    meta_path = st.text_input("Meta path", DEFAULT_META)
    metrics_path = st.text_input("Training metrics path", DEFAULT_METRICS)
    st.markdown("**Training Insights**")
    mm_train_path = st.text_input("Multimodal train metrics JSON", DEFAULT_MM_TRAIN_METRICS)
    mm_eval_path = st.text_input("Multimodal eval metrics JSON", DEFAULT_MM_EVAL_METRICS)
    local_eval_path = st.text_input("Local lite eval metrics JSON", DEFAULT_LOCAL_EVAL_METRICS)
    blend_metrics_path = st.text_input("Blend metrics JSON", DEFAULT_BLEND_METRICS)
    training_report_html_path = st.text_input("Training report HTML (optional)", DEFAULT_TRAINING_REPORT_HTML)
    tabular_path = st.text_input("Tabular CSV", DEFAULT_TABULAR)
    mda_dir = st.text_input("MD&A folder", DEFAULT_MDA_DIR)
    graph_path = st.text_input("Graph path", DEFAULT_GRAPH)
    use_text = st.checkbox("Use text stats", value=True)
    use_graph = st.checkbox("Use graph degree", value=True)
    top_k = st.number_input("Top K high-risk rows", min_value=10, max_value=5000, value=100, step=10)
    st.divider()
    if _DOTENV_HINT:
        st.warning(_DOTENV_HINT)
    elif not _openrouter_key_resolved():
        st.caption(
            "OpenRouter: add **`OPENROUTER_API_KEY`** to `financial_distress/.env` "
            "(copy `.env.example` → `.env`). If `.env` exists on another computer, copy it here or use Streamlit Cloud secrets."
        )
        st.caption(_openrouter_missing_key_help())

if not os.path.exists(model_path):
    st.warning(f"Model not found: `{model_path}`. Train first via `python3 -m src.train`.")
    st.stop()

if not os.path.exists(tabular_path):
    st.warning(f"Tabular CSV not found: `{tabular_path}`.")
    st.stop()

try:
    model, x_df, y, base_df, probs = _cached_scoring_bundle(
        model_path=model_path,
        tabular_path=tabular_path,
        mda_dir=mda_dir,
        graph_path=graph_path,
        use_text_stats=use_text,
        use_graph_degree=use_graph,
    )
except Exception as exc:
    st.error(f"Failed to load model or features: {exc}")
    st.stop()

model_meta: dict[str, Any] = _cached_json_file(meta_path)
train_metrics: dict[str, Any] = _cached_json_file(metrics_path)
blend_metrics: dict[str, Any] = _cached_json_file(blend_metrics_path)
training_llm_context = _training_snapshot_for_llm(
    _cached_json_file(mm_train_path),
    _cached_json_file(mm_eval_path),
    _cached_json_file(local_eval_path),
    blend_metrics,
    model_meta,
)

cal_note = model_meta.get("probability_calibration", "none")
if cal_note and cal_note != "none":
    st.sidebar.success(f"Calibrated scores: {cal_note}")
else:
    st.sidebar.caption("Scores: raw model probability (uncalibrated)")

result_df = base_df.copy()
result_df["risk_score"] = probs
result_df["label"] = y
result_df = result_df.sort_values("risk_score", ascending=False)
result_df["year"] = pd.to_datetime(result_df["datadate"], errors="coerce").dt.year

thr_default = float(model_meta.get("recommended_threshold", 0.5))
col1, col2, col3 = st.columns(3)
col1.metric("Filings scored", f"{len(result_df):,}")
col2.metric("Avg distress probability", f"{result_df['risk_score'].mean():.2%}")
col3.metric("Historical distress rate (data)", f"{result_df['label'].mean():.2%}")
st.caption(
    f"Decision threshold (F1-tuned on holdout): **{thr_default:.3f}** — scores above this are flagged as distressed."
)

tab_scoring, tab_upload, tab_context, tab_eda, tab_insights = st.tabs(
    ["Portfolio Scoring", "Upload Your Records", "Contextual Analysis", "EDA", "Training Insights"]
)

with tab_scoring:
    st.subheader("Highest predicted bankruptcy / distress risk")
    st.dataframe(result_df.head(int(top_k)), width="stretch")
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored CSV",
        data=csv,
        file_name="local_lite_scored.csv",
        mime="text/csv",
    )

with tab_upload:
    st.subheader("Upload records and score")
    st.write("Simple CSV works. You can upload only key ratios; missing fields are auto-completed using dataset medians.")
    st.caption("Recommended simple columns: debt_to_equity, return_on_assets, return_on_equity, current_ratio, net_profit_margin, revenue_growth_pct")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        user_df = pd.read_csv(uploaded, low_memory=False)
        trained_features = model_meta.get("features", list(x_df.columns))
        model_input = _prepare_model_input_from_user_df(user_df, trained_features, x_df)
        user_probs = model.predict_proba(model_input)[:, 1]
        out = user_df.copy()
        out["bankruptcy_probability"] = user_probs
        out["risk_score"] = user_probs
        threshold = float(model_meta.get("recommended_threshold", 0.5))
        out["pred_label"] = (out["risk_score"] >= threshold).astype(int)
        out["status"] = out["pred_label"].map({1: "Distress flagged", 0: "Not flagged"})
        out = out.sort_values("risk_score", ascending=False)
        st.success(f"Scored {len(out):,} rows with **bankruptcy / distress probability**.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(out):,}")
        m2.metric("Avg distress probability", f"{out['risk_score'].mean():.2%}")
        m3.metric("Distress-flagged rows", f"{int(out['pred_label'].sum()):,}")
        m4.metric("Flag rate", f"{(100.0 * out['pred_label'].mean()):.2f}%")

        st.markdown("### Individual Company View")
        out = out.reset_index(drop=True)
        choices = [f"#{i+1} | P(distress)={out.loc[i, 'risk_score']:.2%}" for i in out.index]
        selected = st.selectbox("Select a company row to analyze", options=list(range(len(choices))), format_func=lambda i: choices[i])
        row = out.loc[selected]

        c1, c2, c3 = st.columns(3)
        c1.metric("P(bankruptcy / distress)", f"{row['risk_score']:.2%}")
        c2.metric("Prediction", str(row["status"]))
        percentile = float((out["risk_score"] <= row["risk_score"]).mean() * 100.0)
        c3.metric("vs uploaded set (percentile)", f"{percentile:.1f}%")
        conf = _confidence_from_margin(float(row["risk_score"]), threshold)
        st.progress(conf, text=f"Decision confidence: {conf:.0%}")

        st.markdown(
            bankruptcy_summary(
                float(row["risk_score"]),
                threshold,
                bool(row["pred_label"] == 1),
            )
        )
        st.caption(f"Risk band: **{bankruptcy_risk_band(float(row['risk_score']), threshold)}**")

        pred_ctx: dict[str, Any] = {
            "scenario": "upload_csv_row",
            "model": "local_lite",
            "risk_probability": float(row["risk_score"]),
            "threshold": float(threshold),
            "pred_distress_flag": bool(int(row["pred_label"]) == 1),
            "risk_band": bankruptcy_risk_band(float(row["risk_score"]), threshold),
            "percentile_among_upload": float(percentile),
            "row_feature_subset": {
                k: _safe_float(row.get(k), float("nan"))
                for k in (
                    "debt_to_equity",
                    "return_on_assets",
                    "return_on_equity",
                    "current_ratio",
                    "operating_margin",
                    "asset_turnover",
                    "graph_degree",
                    "mda_word_count",
                )
                if k in row.index
            },
            "checkpoint_training_snapshot": training_llm_context,
        }
        _ai_prediction_panel(
            context=pred_ctx,
            state_prefix=f"upload_{int(selected)}",
        )

        st.markdown("**Selected record details**")
        st.dataframe(row.to_frame("value"), width="stretch")

        explain_features = [
            "debt_to_equity",
            "return_on_assets",
            "return_on_equity",
            "current_ratio",
            "operating_margin",
            "asset_turnover",
            "graph_degree",
            "mda_word_count",
        ]
        compare_rows = []
        for f in explain_features:
            if f in out.columns:
                compare_rows.append(
                    {
                        "feature": f,
                        "selected_value": _safe_float(row.get(f), np.nan),
                        "uploaded_avg": _safe_float(out[f].mean(), np.nan),
                    }
                )
        if compare_rows:
            comp_df = pd.DataFrame(compare_rows)
            st.markdown("**Selected vs uploaded average (key features)**")
            st.dataframe(comp_df, width="stretch")

        st.markdown("**Predicted status split**")
        status_counts = out["status"].value_counts().rename_axis("status").to_frame("count")
        st.bar_chart(status_counts)

        st.markdown("**Risk score distribution (uploaded file)**")
        up_bins = pd.cut(out["risk_score"], bins=10)
        up_hist = up_bins.value_counts().sort_index()
        up_hist_df = up_hist.rename_axis("bin").reset_index(name="count")
        up_hist_df["bin"] = up_hist_df["bin"].astype(str)
        st.bar_chart(up_hist_df.set_index("bin"))

with tab_context:
    st.subheader("Meaningful Input -> Contextual Output")
    st.write("Enter 6 simple ratios. The app auto-maps them to model features and explains the result in business language.")
    if "ctx_panel_visible" not in st.session_state:
        st.session_state["ctx_panel_visible"] = False
    c1, c2, c3 = st.columns(3)
    debt_to_equity = c1.number_input("Debt-to-Equity Ratio", value=1.0, step=0.1, format="%.4f")
    roa = c2.number_input("Return on Assets (ROA)", value=0.03, step=0.01, format="%.4f")
    roe = c3.number_input("Return on Equity (ROE)", value=0.08, step=0.01, format="%.4f")

    c4, c5, c6 = st.columns(3)
    current_ratio = c4.number_input("Current Ratio", value=1.2, step=0.1, format="%.4f")
    net_profit_margin = c5.number_input("Net Profit Margin", value=0.05, step=0.01, format="%.4f")
    revenue_growth_pct = c6.number_input("Revenue Growth (%)", value=5.0, step=0.5, format="%.3f")

    mda_text = st.text_area("Optional MD&A / company description text", value="", height=120)
    include_graph_context = st.checkbox("Include graph/network context", value=True)
    graph_degree_input = None
    if include_graph_context and "graph_degree" in x_df.columns:
        graph_degree_input = st.slider("Network connectivity proxy (graph degree)", min_value=0.0, max_value=1000.0, value=25.0, step=1.0)

    gcol, ccol = st.columns([4, 1])
    with gcol:
        if st.button("Generate Contextual Analysis", type="primary", key="ctx_generate_main"):
            st.session_state["ctx_panel_visible"] = True
    with ccol:
        if st.button("Clear", key="ctx_clear_panel"):
            st.session_state["ctx_panel_visible"] = False

    if st.session_state.get("ctx_panel_visible"):
        threshold = float(model_meta.get("recommended_threshold", 0.5))
        features = model_meta.get("features", list(x_df.columns))
        user_x = _build_manual_feature_row(
            feature_cols=features,
            dte=debt_to_equity,
            roa=roa,
            roe=roe,
            current_ratio=current_ratio,
            net_profit_margin=net_profit_margin,
            revenue_growth_pct=revenue_growth_pct,
            mda_text=mda_text,
            include_graph=include_graph_context,
            graph_degree_input=graph_degree_input,
            reference_df=x_df,
        )
        risk = float(model.predict_proba(user_x)[:, 1][0])
        health = float(1.0 - risk)
        status = _status_from_risk(risk, threshold)

        percentile = float((result_df["risk_score"] <= risk).mean() * 100.0)
        anomaly = float(np.clip(np.abs((risk - result_df["risk_score"].mean()) / max(result_df["risk_score"].std(), 1e-6)), 0.0, 5.0) / 5.0)

        similar = _closest_companies(x_df, result_df[["cik", "datadate", "label", "risk_score"]], user_x, n=3)

        user_map = {
            "debt_to_equity": debt_to_equity,
            "return_on_assets": roa,
            "return_on_equity": roe,
            "current_ratio": current_ratio,
            "operating_margin": net_profit_margin,
            "asset_turnover": revenue_growth_pct / 100.0,
        }
        reasons = _top_factor_explanations(user_map, x_df)

        # Heuristic modality contributions for interpretability fallback.
        tab_w = 0.75
        text_w = 0.20 if mda_text.strip() else 0.05
        graph_w = 0.15 if include_graph_context else 0.0
        s = tab_w + text_w + graph_w
        modality = {
            "Tabular": tab_w / s,
            "Text": text_w / s,
            "Graph": graph_w / s if s > 0 else 0.0,
        }

        st.markdown("### 1) Bankruptcy / distress prediction")
        pred_flag = risk >= threshold
        m1, m2, m3 = st.columns(3)
        m1.metric("P(healthy)", f"{health:.2%}")
        m2.metric("P(distress / bankruptcy)", f"{risk:.2%}")
        m3.metric("Flag at threshold?", "Yes" if pred_flag else "No")
        conf = _confidence_from_margin(risk, threshold)
        st.progress(conf, text=f"Decision confidence: {conf:.0%}")
        st.markdown(
            bankruptcy_summary(risk, threshold, pred_flag),
        )
        st.caption(f"Risk band: **{bankruptcy_risk_band(risk, threshold)}** · Status label: **{status}**")

        st.markdown("### 2) Dataset Context")
        st.write(f"This company has higher risk than approximately **{percentile:.1f}%** of filings in the dataset.")
        st.write(f"Anomaly score: **{anomaly:.3f}** (higher means more unusual vs dataset risk distribution).")

        st.markdown("### 3) Risk Explanation")
        if reasons:
            for r in reasons[:3]:
                st.write(f"- {r}")
        else:
            st.write("- Inputs are close to dataset averages; no strong single risk driver found.")

        avg_cmp = pd.DataFrame(
            {
                "metric": ["Debt-to-Equity", "ROA", "ROE", "Current Ratio", "Net Profit Margin (proxy)", "Revenue Growth (proxy)"],
                "user_value": [
                    debt_to_equity,
                    roa,
                    roe,
                    current_ratio,
                    net_profit_margin,
                    revenue_growth_pct / 100.0,
                ],
                "dataset_avg": [
                    _safe_float(x_df["debt_to_equity"].mean()) if "debt_to_equity" in x_df else np.nan,
                    _safe_float(x_df["return_on_assets"].mean()) if "return_on_assets" in x_df else np.nan,
                    _safe_float(x_df["return_on_equity"].mean()) if "return_on_equity" in x_df else np.nan,
                    _safe_float(x_df["current_ratio"].mean()) if "current_ratio" in x_df else np.nan,
                    _safe_float(x_df["operating_margin"].mean()) if "operating_margin" in x_df else np.nan,
                    _safe_float(x_df["asset_turnover"].mean()) if "asset_turnover" in x_df else np.nan,
                ],
            }
        )
        st.dataframe(avg_cmp, width="stretch")

        st.markdown("### 4) Similar Cases")
        similar = similar.copy()
        similar["outcome"] = similar["label"].map({1: "bankrupt", 0: "survived"})
        st.dataframe(similar[["cik", "datadate", "risk_score", "outcome", "distance"]], width="stretch")

        st.markdown("### 5) Model Insight (Modality Contribution)")
        mod_df = pd.DataFrame({"modality": list(modality.keys()), "weight": list(modality.values())})
        st.bar_chart(mod_df.set_index("modality"))
        st.caption("For local-lite mode, modality contribution is an interpretable fallback based on provided inputs and enabled context.")

        ctx_ctx: dict[str, Any] = {
            "scenario": "manual_ratio_form",
            "model": "local_lite",
            "risk_probability": risk,
            "threshold": float(threshold),
            "pred_distress_flag": pred_flag,
            "risk_band": bankruptcy_risk_band(risk, threshold),
            "status_label": status,
            "dataset_percentile": percentile,
            "anomaly_score": anomaly,
            "heuristic_modality_weights": modality,
            "top_reasons": reasons[:5],
            "inputs": {
                "debt_to_equity": debt_to_equity,
                "return_on_assets": roa,
                "return_on_equity": roe,
                "current_ratio": current_ratio,
                "operating_margin_proxy": net_profit_margin,
                "revenue_growth_proxy": revenue_growth_pct / 100.0,
                "mda_word_count": len(mda_text.split()) if mda_text.strip() else 0,
                "graph_degree": float(graph_degree_input) if graph_degree_input is not None else None,
            },
            "checkpoint_training_snapshot": training_llm_context,
        }
        _ai_prediction_panel(
            context=ctx_ctx,
            state_prefix="context_manual",
        )

with tab_eda:
    st.subheader("Dataset EDA")
    col_a, col_b = st.columns(2)
    col_a.metric("Total rows", f"{len(result_df):,}")
    col_b.metric("Positive labels", f"{int(result_df['label'].sum()):,}")

    st.markdown("**Label distribution**")
    label_dist = result_df["label"].value_counts().rename_axis("label").reset_index(name="count")
    st.bar_chart(label_dist.set_index("label"))

    st.markdown("**Risk score distribution (10 bins)**")
    bins = pd.cut(result_df["risk_score"], bins=10)
    hist = bins.value_counts().sort_index()
    hist_df = hist.rename_axis("bin").reset_index(name="count")
    hist_df["bin"] = hist_df["bin"].astype(str)
    st.bar_chart(hist_df.set_index("bin"))

    if result_df["year"].notna().any():
        st.markdown("**Rows by year**")
        by_year = result_df.groupby("year", as_index=True).size().sort_index()
        st.line_chart(by_year)

    st.markdown("**Top missing feature columns**")
    miss = x_df.isna().mean().sort_values(ascending=False).head(15)
    st.dataframe((miss * 100).rename("missing_pct").to_frame(), width="stretch")

with tab_insights:
    st.header("Training Insights")
    st.markdown(
        "Interactive charts compare **multimodal fusion** (neural) vs **local lite** (sklearn) using the "
        "checkpoint JSON files under `checkpoints/`. Regenerate the static HTML with "
        "`python3 -m src.training_report --out checkpoints/training_report.html` from the project root."
    )

    try:
        report_fig = build_integrated_training_figure(
            multimodal_train_path=mm_train_path,
            multimodal_eval_path=mm_eval_path,
            local_meta_path=meta_path,
            local_train_metrics_path=metrics_path,
            local_eval_path=local_eval_path,
            blend_metrics_path=blend_metrics_path,
        )
        st.plotly_chart(report_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not build Plotly report: {exc}")

    if os.path.isfile(training_report_html_path):
        with open(training_report_html_path, "rb") as hf:
            st.download_button(
                label="Download training_report.html",
                data=hf.read(),
                file_name="training_report.html",
                mime="text/html",
                key="dl_training_report_html",
            )
    else:
        st.caption(f"No file at `{training_report_html_path}` — run `python3 -m src.training_report` to create it.")

    st.divider()
    st.subheader("Holdout metrics snapshot (from evaluate)")
    mm_eval_data = _cached_json_file(mm_eval_path)
    loc_eval_data = _cached_json_file(local_eval_path)
    left, right = st.columns(2)
    with left:
        st.markdown("**Multimodal** — `metrics_eval.json`")
        _metric_row_ui(mm_eval_path, mm_eval_data)
        with st.expander("Raw multimodal eval JSON"):
            st.json(mm_eval_data if mm_eval_data else {})
    with right:
        st.markdown("**Local lite** — `metrics_eval.json`")
        _metric_row_ui(local_eval_path, loc_eval_data)
        with st.expander("Raw local lite eval JSON"):
            st.json(loc_eval_data if loc_eval_data else {})

    st.divider()
    st.subheader("Blend — `blend_metrics_eval.json`")
    _blend_metrics_ui(blend_metrics_path, blend_metrics)
    with st.expander("Raw blend metrics JSON"):
        st.json(blend_metrics if blend_metrics else {})

    st.divider()
    st.subheader("AI summary — full training picture (OpenRouter)")
    st.caption("Uses checkpoint JSON summaries only (no raw row-level data). Key and model come from `.env` / environment / Streamlit secrets.")
    train_key = _openrouter_key_resolved()
    train_txt_key = "train_ai_text"
    train_err_key = "train_ai_err"
    train_safe_ctx = _coerce_json_context(training_llm_context)
    if train_key:
        with st.form(key="train_ai_form"):
            train_submitted = st.form_submit_button("Generate training comparison summary")

        if train_submitted:
            with st.spinner("Calling OpenRouter…"):
                try:
                    ttext = summarize_training_artifacts(
                        train_key,
                        context=train_safe_ctx,
                        model=_openrouter_model_resolved() or None,
                    )
                    st.session_state[train_txt_key] = ttext
                    st.session_state.pop(train_err_key, None)
                    st.success("Received response from OpenRouter.")
                    if ttext and str(ttext).strip():
                        st.markdown(ttext)
                    else:
                        st.warning("Model returned an empty string.")
                except Exception as exc:
                    st.session_state[train_err_key] = str(exc)
                    st.session_state.pop(train_txt_key, None)
                    st.error(f"OpenRouter request failed: {exc}")

        if not train_submitted:
            if st.session_state.get(train_err_key):
                st.error(st.session_state[train_err_key])
            elif train_txt_key in st.session_state:
                tval = st.session_state[train_txt_key]
                if tval and str(tval).strip():
                    st.markdown("**Last training summary:**")
                    st.markdown(tval)
                else:
                    st.warning("Last run returned an empty training summary. Try another `OPENROUTER_MODEL` in `.env`.")
    else:
        st.caption("Set **`OPENROUTER_API_KEY`** in `financial_distress/.env` (or deployment secrets) to use this.")
        st.caption(_openrouter_missing_key_help())

    st.divider()
    st.subheader("Local lite — training holdout (`metrics.json`)")
    if train_metrics:
        _metric_row_ui(metrics_path, train_metrics)
        with st.expander("Raw local lite training metrics JSON"):
            st.json(train_metrics)
    else:
        st.info("No metrics file found yet. Train first: `python3 -m src.train --mode local_lite`.")
