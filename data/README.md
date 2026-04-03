# Data Setup

This project expects the ECL dataset CSV at:

- `data/ECL (1).csv`

Generated artifacts:

- `data/processed/tabular.csv` (from `src.data_pipeline`)
- `data/raw/mda_texts/*.txt` (from `src.data_pipeline`)
- `data/processed/graph.pt` (from `src.data_pipeline`)
- `data/processed/text_embeddings.pt` (from `src.precompute_embeddings`)

## Build order

1. Run Step 2 pipeline:
   - `python3 -m src.data_pipeline`
2. Precompute text embeddings:
   - `python3 -m src.precompute_embeddings`
3. Train model:
   - `python3 -m src.train`
