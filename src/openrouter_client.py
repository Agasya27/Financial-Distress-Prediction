"""
OpenRouter chat completions for optional narrative summaries (prediction + training context).

Configure with environment variables (recommended):
  OPENROUTER_API_KEY   — required for API calls
  OPENROUTER_MODEL     — default: openai/gpt-4o-mini
  OPENROUTER_HTTP_REFERER — optional site URL for OpenRouter rankings
  OPENROUTER_APP_TITLE    — optional app name header

Deployment: set `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` if you do not use `.env`.

Project root `.env` is loaded automatically (see `.env.example`) when this module is imported. The Streamlit UI does not collect API keys.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TIMEOUT_S = 75

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_dotenv() -> None:
    from src.env_bootstrap import load_financial_distress_dotenv

    load_financial_distress_dotenv(_PROJECT_ROOT)


_load_dotenv()


def get_openrouter_api_key(project_root: str | None = None) -> str | None:
    """Resolve key from env + disk. Pass `project_root` from Streamlit as the app’s `financial_distress/` folder."""
    from src.env_bootstrap import load_financial_distress_dotenv, read_openrouter_key_from_dotenv_files

    root = _PROJECT_ROOT if project_root is None else project_root
    load_financial_distress_dotenv(root)
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    key = read_openrouter_key_from_dotenv_files(root)
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
    return key


def get_openrouter_model() -> str:
    return (os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()


def openrouter_chat(
    api_key: str,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 900,
    temperature: float = 0.35,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Return assistant message text or raise requests.HTTPError / KeyError on bad response."""
    m = (model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Financial Distress Prediction"),
    }
    body: dict[str, Any] = {
        "model": m,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout_s)
    if not resp.ok:
        detail = (resp.text or "")[:800]
        raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {detail or resp.reason}")
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON: {(resp.text or '')[:400]}") from exc
    return _extract_assistant_text(data)


def _extract_assistant_text(data: dict[str, Any]) -> str:
    """OpenRouter / OpenAI-style: `content` may be a string or a list of {type, text} parts."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response missing choices: {json.dumps(data)[:500]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise RuntimeError(f"OpenRouter message has no content: {json.dumps(choices[0])[:500]}")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    parts.append(str(t))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def summarize_prediction_bundle(
    api_key: str,
    *,
    context: dict[str, Any],
    model: str | None = None,
) -> str:
    """
    Ask the model for a concise executive summary of one prediction + optional training snapshot.

    `context` should be JSON-serializable (no secrets).
    """
    system = (
        "You are a financial risk analyst assistant. Write a clear, accurate summary for a non-lawyer "
        "business reader. The model predicts probability of financial distress (dataset-specific label), "
        "not a legal bankruptcy determination. Be cautious with causality; describe associations. "
        "Use short sections or bullets. No investment advice. Under 350 words unless the user data is very rich."
    )
    try:
        ctx_json = json.dumps(context, indent=2, default=str)[:12000]
    except TypeError as exc:
        raise TypeError(f"Prediction context is not JSON-serializable: {exc}") from exc
    user = (
        "Summarize the following structured prediction context. "
        "Highlight risk level, how it compares to the reference portfolio if given, "
        "and 2–4 concrete drivers or caveats. If training metrics are present, "
        "mention how trustworthy the score might be at a high level (calibration, class imbalance).\n\n"
        "```json\n"
        + ctx_json
        + "\n```"
    )
    return openrouter_chat(
        api_key,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )


def summarize_training_artifacts(
    api_key: str,
    *,
    context: dict[str, Any],
    model: str | None = None,
) -> str:
    """Narrative comparing local lite vs multimodal vs blend from checkpoint JSON summaries."""
    system = (
        "You are a machine learning engineer explaining model comparison results to a product owner. "
        "Compare metrics honestly; note PR-AUC vs ROC-AUC tradeoffs for imbalanced distress data. "
        "No hype. Under 400 words. Bullet lists welcome."
    )
    try:
        ctx_json = json.dumps(context, indent=2, default=str)[:14000]
    except TypeError as exc:
        raise TypeError(f"Training context is not JSON-serializable: {exc}") from exc
    user = (
        "Explain what these training/evaluation artifacts suggest about model quality and next steps.\n\n"
        "```json\n"
        + ctx_json
        + "\n```"
    )
    return openrouter_chat(
        api_key,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
    )
