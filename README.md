# Financial Distress & Bankruptcy Risk Prediction

End-to-end project for predicting **financial distress** (bankruptcy-related labels from the ECL-style dataset) using tabular financial features, optional MD&A text signals, and optional graph context. Designed to run **locally on macOS** (including Apple Silicon).

## Quick start

```bash
cd Financial-Distress-Prediction   # use your clone path if different
python3 -m pip install -r requirements.txt
python3 -m src.project_health_check
python3 -m src.train --mode local_lite
python3 -m src.evaluate --mode local_lite
python3 -m streamlit run app/streamlit_app.py
```

For deployment or GPU workflows, see [Further reading](#further-reading).

---

## Prerequisites

- **Python** 3.10+ recommended (PyTorch, scikit-learn, Streamlit).
- **Disk**: processed data, checkpoints, and optional embeddings can grow large; allow several GB if you run the full SEC pipeline and FinBERT precompute.
- **Optional**: Apple Silicon uses MPS where supported by PyTorch; CPU-only runs are possible for `local_lite` and the app if artifacts already exist.

---

## Environment variables

| Variable | Where | Purpose |
|----------|--------|---------|
| `OPENROUTER_API_KEY` | `.env`, shell, or Streamlit secrets | Enables optional AI narratives in the app (`src/openrouter_client.py`). |
| `OPENROUTER_MODEL` | Same (optional) | Overrides default OpenRouter model id. |

Copy `.env.example` to **`.env`** and set secrets there; do not commit real keys. See [OpenRouter](#optional--openrouter-ai-summaries-in-the-app) below.

---

## What is built (summary)

| Area | What exists | Location |
|------|-------------|----------|
| **Data pipeline** | Loads ECL CSV, pulls SEC XBRL + submissions, downloads 10-K HTML, extracts MD&A Item 7, builds SIC-based `graph.pt` | `src/data_pipeline.py` |
| **Text embeddings** | FinBERT `[CLS]` per chunk, checkpoints + resume | `src/precompute_embeddings.py` |
| **Features for app / sklearn** | Tabular + optional MD&A length stats + graph degree | `src/feature_table.py` (`load_feature_table`) |
| **PyTorch dataset** | `FinancialDistressDataset` + masking + `collate_fn` for multimodal experiments | `src/dataset.py` |
| **Primary model (recommended for accuracy)** | Sklearn pipeline: model selection (GBDT / RF / ExtraTrees / LogReg / optional ensemble) + **sigmoid probability calibration** + F1-tuned threshold | `src/train.py` (`--mode local_lite`) |
| **Multimodal model** | Tabular + FinBERT chunks + graph proxy, fusion + attention | `src/train_multimodal.py`, `src/models/*`, `src/evaluate_multimodal.py` |
| **Evaluation** | Unified CLI: local multimodal or local_lite | `src/evaluate.py` |
| **Streamlit app** | Portfolio scoring, CSV upload, contextual analysis, EDA, training metrics (local / multimodal / **blend**), optional **OpenRouter** AI narratives | `app/streamlit_app.py` |
| **Blend eval** | Grid-search mix of local_lite + multimodal probs on time holdout | `src/evaluate_blend.py` |
| **OpenRouter** | Optional LLM summaries (API key via env or Streamlit secrets) | `src/openrouter_client.py` |
| **Utilities** | JSON helpers, shared prediction copy (`bankruptcy_summary`, risk bands) | `src/utils.py`, `src/prediction_utils.py` |
| **Health check** | Verifies paths and artifacts before training | `src/project_health_check.py` |
| **Config** | Hyperparameters and paths | `config.yaml` |
| **Streamlit Cloud** | Lighter `requirements-app.txt`, secrets, layout | `docs/STREAMLIT_CLOUD.md` |
| **GPU / Lightning** | Remote training workflow | `docs/LIGHTNING_GPU_TRAINING.md` |
| **Sample data** | Example upload CSV | `sample_upload_records.csv` |

---

## Project layout

```
Financial-Distress-Prediction/
├── app/streamlit_app.py
├── config.yaml
├── requirements.txt              # Full stack (training + SEC + multimodal)
├── requirements-app.txt          # Streamlit demo only (see docs/STREAMLIT_CLOUD.md)
├── docs/
│   ├── STREAMLIT_CLOUD.md
│   └── LIGHTNING_GPU_TRAINING.md
├── data/
│   ├── ECL (1).csv               # source CSV (path also in config.yaml → data.ecl_csv_path)
│   ├── raw/mda_texts/
│   ├── processed/
│   └── README.md
├── src/
│   ├── feature_table.py          # load_feature_table (used by app without full torch Dataset)
│   ├── data_pipeline.py
│   ├── precompute_embeddings.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_blend.py
│   ├── train_multimodal.py
│   ├── evaluate_multimodal.py
│   ├── project_health_check.py
│   ├── openrouter_client.py
│   ├── env_bootstrap.py
│   ├── training_report.py
│   ├── metrics_classification.py
│   └── models/
├── checkpoints/
│   ├── local_lite/
│   └── multimodal/
└── scripts/
    ├── check_streamlit_cloud_ready.py
    ├── lightning_train_multimodal.sh
    └── run_streamlit_local.sh
```

---

## How to run (typical workflow)

### 1. Install

```bash
cd Financial-Distress-Prediction
python3 -m pip install -r requirements.txt
```

### 2. Sanity check

```bash
python3 -m src.project_health_check
```

### 3. Train the **recommended** predictor (local lite — calibrated probabilities)

```bash
python3 -m src.train --mode local_lite
python3 -m src.evaluate --mode local_lite
```

This writes:

- `checkpoints/local_lite/model.joblib` — fitted model (may include `CalibratedClassifierCV`)
- `checkpoints/local_lite/meta.json` — features list, threshold, calibration note, label meaning
- `checkpoints/local_lite/metrics.json` — holdout metrics

### 4. Launch the app

```bash
python3 -m streamlit run app/streamlit_app.py
```

You can also use `scripts/run_streamlit_local.sh` if it matches your setup. Default model path in the app points to `checkpoints/local_lite/model.joblib`.

**Streamlit Community Cloud:** use `requirements-app.txt` and follow [`docs/STREAMLIT_CLOUD.md`](docs/STREAMLIT_CLOUD.md). Add `OPENROUTER_API_KEY` (and optional `OPENROUTER_MODEL`) under app **Secrets**.

### Optional — OpenRouter AI summaries in the app

1. From the project root, copy `.env.example` to **`.env`** and set `OPENROUTER_API_KEY` there. **Do not put real keys only in `.env.example`** (that file is for templates). If `.env` is missing, the app can fall back to `.env.example` and will show a sidebar warning.
2. Run `python3 -m pip install -r requirements.txt` (includes `python-dotenv`). The app loads `.env` (then `.env.local`, then `.env.example` only if the key is still unset). File values **override** empty shell variables so a blank `export OPENROUTER_API_KEY=` cannot block `.env`.
3. Run Streamlit from the **project root** so paths resolve. On **Lightning.ai / SSH**, create `.env` on the **remote** studio (or use secrets); your Mac `.env` is not visible there.
4. Alternatives: export `OPENROUTER_API_KEY` in the shell, or use `.streamlit/secrets.toml` for deployment (see `.streamlit/secrets.toml.example`).
5. There is **no API key field in the UI**; use **Generate AI summary** buttons once `.env` / secrets define the key.

### 5. Optional: multimodal training (needs `text_embeddings.pt`)

```bash
python3 -m src.precompute_embeddings    # one-time, GPU/CPU heavy
python3 -m src.train                    # default = multimodal
python3 -m src.evaluate
```

Or explicitly:

```bash
python3 -m src.train_multimodal --epochs 8 --batch-size 64
python3 -m src.evaluate_multimodal
```

### 6. Optional: blend local_lite + multimodal on the time holdout

Requires both `checkpoints/local_lite/model.joblib` and `checkpoints/multimodal/best_model.pt` (and the usual tabular / embedding paths). Writes `checkpoints/blend_metrics_eval.json` for the Streamlit metrics view.

```bash
python3 -m src.evaluate_blend
```

---

## What the app does

- **Portfolio scoring**: scores all rows in `tabular.csv` (with optional text/graph features), sortable export.
- **Upload CSV**: user can upload **simple columns** (e.g. debt_to_equity, ROA, ROE, …); missing model fields are filled from dataset medians; shows **P(distress)** and plain-language summary.
- **Contextual analysis**: manual ratios + optional MD&A text + graph toggle; compares to dataset and shows similar historical rows.
- **EDA / training metrics**: integrated Plotly chart (multimodal history, local model pick, holdout bars including **blend** when `checkpoints/blend_metrics_eval.json` exists).
- **OpenRouter**: per-prediction narratives (upload + manual form) and a **training comparison** summary; credentials only via `.env` / environment / Streamlit secrets (not entered in the app).

Copy used in the UI explains that **label = 1** corresponds to **distress / bankruptcy-related outcome in the training data**, not legal advice.

---

## Data generation (if starting from ECL only)

1. Place the ECL CSV at the path in `config.yaml` (`data.ecl_csv_path`; the repo includes `data/ECL (1).csv` — on case-sensitive filesystems, match the config filename exactly).
2. Run the pipeline (network + SEC rate limits apply):

   ```bash
   python3 -m src.data_pipeline
   ```

3. Optional FinBERT embeddings:

   ```bash
   python3 -m src.precompute_embeddings
   ```

See [`data/README.md`](data/README.md) for artifact expectations.

---

## Metrics & limitations

- The positive class (distress/bankruptcy) is **rare**; **PR-AUC** and **F1 at a tuned threshold** matter more than raw accuracy.
- Outputs are **screening-style probabilities**; always combine with judgment and other data.
- Multimodal training is a **lighter Mac-friendly** variant compared to a full FT-Transformer + GAT + full CVAE spec; the **local_lite** path is tuned for **stable deployment** in the Streamlit app.

---

## Further reading

- [`docs/STREAMLIT_CLOUD.md`](docs/STREAMLIT_CLOUD.md) — deploy the UI to Streamlit Cloud  
- [`docs/LIGHTNING_GPU_TRAINING.md`](docs/LIGHTNING_GPU_TRAINING.md) — GPU training on Lightning.ai  
- [`data/README.md`](data/README.md) — data artifacts  

---

*Layout trimmed for active training + app paths; Streamlit loads `feature_table` without importing the full multimodal `Dataset` stack.*
