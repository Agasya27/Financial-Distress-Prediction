# Deploy on Streamlit Community Cloud

## 1. Push this folder to GitHub

Use the `financial_distress/` directory as the **repository root** (recommended), or note the subpath if the repo contains a parent folder.

Do **not** commit `.env` or `.streamlit/secrets.toml` (they are gitignored). Large data (`data/ECL (1).csv`) and `checkpoints/` are usually excluded—see below.

**Git LFS:** `graph.pt`, `model.joblib`, and large processed tabular CSVs are tracked with [Git LFS](https://git-lfs.github.com/) (see `.gitattributes`). For a local clone, install Git LFS and run `git lfs install` once, then `git clone` as usual. Streamlit Cloud pulls LFS objects from GitHub when it builds the app.

## 2. App settings on [share.streamlit.io](https://share.streamlit.io)

| Setting | Value |
|--------|--------|
| **Main file path** | `app/streamlit_app.py` |
| **Python version** | 3.11 (recommended) |
| **Requirements file** | `requirements-app.txt` |
| **Python** | `runtime.txt` pins **3.11** (Streamlit reads it from the repo root). |

If your repo root is **above** `financial_distress/`, set the main file to `financial_distress/app/streamlit_app.py` and, if the UI allows, set the working directory to `financial_distress`.

## 3. Secrets (OpenRouter)

In the app dashboard → **Secrets**, add:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

The app reads these via `st.secrets` when `.env` is not present.

## 4. Data and model artifacts

The UI expects, by default:

- `data/processed/tabular.csv`
- `checkpoints/local_lite/model.joblib` and `meta.json`
- Optional: `data/processed/graph.pt`, `data/raw/mda_texts/`, multimodal checkpoints for Training Insights

**Options:**

1. **Commit small samples** (not the full ECL CSV) for a public demo, **or**
2. **Attach files** in Streamlit Cloud (if your plan supports persistent storage), **or**
3. **Build with Git LFS** for `tabular.csv` + `model.joblib` (watch size limits).

Without `tabular.csv` and `model.joblib`, the app will stop at the “Model not found” or “Tabular CSV not found” warning.

## 5. Resource limits

`requirements-app.txt` omits `torch-geometric` and `transformers` to keep installs lighter. PyTorch is still required if you use **graph degree** from `graph.pt`. If the app OOMs on the free tier, turn off **Use graph degree** in the sidebar or use a paid workspace.

## 6. Local run (same as Cloud)

```bash
cd financial_distress
python3 -m pip install -r requirements-app.txt
python3 -m streamlit run app/streamlit_app.py
```

Use `requirements.txt` for full multimodal + SEC training locally.
