# Train on Lightning.ai GPU (higher throughput)

## 1. SSH into your Studio

After running Lightning’s setup script locally, connect:

```bash
ssh s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai
```

Keep the Studio **running** in the Lightning UI while you SSH.

## 2. Get the project onto the Studio

Run these on your **Mac** (not inside `ssh`). Do **not** paste lines that start with `#` alone — zsh will error on `#`.

### Option A — `scp` folder (simple)

```bash
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -r \
  "/Users/agasya/Financial Distress Prediction/financial_distress" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress
```

If the connection **stalls** (common on large `data/` or `ECL (1).csv`), use **Option B**.

### Option B — one `.tgz` (more reliable on flaky SSH)

```bash
cd "/Users/agasya/Financial Distress Prediction"
tar czf financial_distress.tgz financial_distress
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=120 financial_distress.tgz \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/
```

On Lightning:

```bash
cd ~ && tar xzf financial_distress.tgz
```

### Option C — upload only what’s missing after a failed copy

If small files copied but **`data/ECL (1).csv`** or **`data/processed/`** failed, retry **only** those paths:

```bash
scp -o ServerAliveInterval=30 -r \
  "/Users/agasya/Financial Distress Prediction/financial_distress/data/ECL (1).csv" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress/data/

scp -o ServerAliveInterval=30 -r \
  "/Users/agasya/Financial Distress Prediction/financial_distress/data/processed" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress/data/
```

Or use Lightning’s **file upload** in the browser for large CSVs.

### Option D — `rsync` on Mac (if installed)

```bash
rsync -avz --partial --progress -e "ssh -o ServerAliveInterval=30" \
  "/Users/agasya/Financial Distress Prediction/financial_distress/" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress/
```

## 3. On the Studio (after SSH)

```bash
cd ~/financial_distress
python3 -m pip install -r requirements.txt

# Optional: verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### Sync latest **Python trainers** (if you see `unrecognized arguments`)

Your Studio may still have an **old** copy of the repo. New flags (`--balance-train`, `--loss focal`, `--early-stopping-patience`, etc.) only exist after you copy **updated** `src/` from your Mac.

**Check on Lightning** (should print a line; if empty, the file is old):

```bash
grep -n "balance-train" "$HOME/financial_distress/src/train_multimodal.py"
```

**Fast fix from your Mac** (overwrite trainers; replace SSH user if yours differs).  
`~` on the **remote** side is expanded by SSH to your Studio home (often `/teamspace/studios/this_studio`).

Include **`metrics_classification.py`** — without it, `python3 -m src.train --mode local_lite` fails with `No module named 'src.metrics_classification'` (multimodal-only partial copies are a common cause).

```bash
scp -o ServerAliveInterval=30 \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/train.py" \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/train_multimodal.py" \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/metrics_classification.py" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress/src/
```

If your project is **not** under remote `~/financial_distress`, use the full path you `cd` into, e.g.:

```bash
scp -o ServerAliveInterval=30 \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/train.py" \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/train_multimodal.py" \
  "/Users/agasya/Financial Distress Prediction/financial_distress/src/metrics_classification.py" \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:/teamspace/studios/this_studio/financial_distress/src/
```

Or re-upload a fresh tarball (Mac):

```bash
cd "/Users/agasya/Financial Distress Prediction"
COPYFILE_DISABLE=1 tar czf financial_distress.tgz financial_distress
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
  financial_distress.tgz \
  s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:/teamspace/studios/this_studio/
```

Then on Lightning: `cd "$HOME" && tar xzf financial_distress.tgz` (or extract next to your existing tree and use the new `financial_distress/` folder).

## 4. Train for better accuracy (recommended order)

**A. Strong tabular model (fast, often best ROC-AUC on this dataset)**

```bash
cd ~/financial_distress
python3 -m src.train --mode local_lite
python3 -m src.evaluate --mode local_lite
```

**B. Multimodal on GPU (target: better PR-AUC / ranking than short 8-epoch runs)**

Use the **studio project root** (often `$HOME` = `/teamspace/studios/this_studio` in Cursor):

```bash
cd "$HOME/financial_distress"
python3 -m pip install -r requirements.txt

python3 -m src.precompute_embeddings
```

**Recommended Lightning command** (class-balanced batches, focal loss, early stopping, AMP on CUDA, LR schedule):

```bash
cd "$HOME/financial_distress"
python3 -m src.train --mode multimodal \
  --epochs 60 \
  --batch-size 128 \
  --lr 2.5e-4 \
  --early-stopping-patience 12 \
  --min-epochs 8 \
  --balance-train \
  --loss focal \
  --lr-scheduler plateau \
  --modality-dropout 0.15 \
  --gate-entropy-coef 0.03 \
  --seed 42
python3 -m src.evaluate --mode multimodal
python3 -m src.evaluate_blend
```

Or run the bundled script:

```bash
chmod +x scripts/lightning_train_multimodal.sh
./scripts/lightning_train_multimodal.sh
```

- **`--balance-train`**: oversamples distressed rows each epoch (helps rare positive class).
- **`--loss focal`**: down-weights easy negatives; often improves PR-AUC vs plain BCE.
- **`--modality-dropout`**: randomly drops a modality’s representation during training so fusion does not collapse to tabular-only attention.
- **`--gate-entropy-coef`**: small bonus on fusion gate entropy to encourage using text/graph when helpful.
- **`--early-stopping-patience`**: stops when validation PR-AUC stops improving (saves time, reduces overfit).
- **`python3 -m src.evaluate_blend`**: grid-searches `α·p_lite + (1−α)·p_multimodal` on the same time holdout (needs `checkpoints/local_lite/model.joblib` and `checkpoints/multimodal/best_model.pt`).
- **AMP** turns on automatically on **CUDA** only.

If GPU OOM, lower `--batch-size` (e.g. 64). For a quicker smoke test, use `--epochs 15 --early-stopping-patience 4`.

## 5. Copy checkpoints back to your Mac

```bash
rsync -avz s_01kna0ptkhafqt08tdq8vf6tx3@ssh.lightning.ai:~/financial_distress/checkpoints/ \
  "/Users/agasya/Financial Distress Prediction/financial_distress/checkpoints/"
```

Then point Streamlit at `checkpoints/local_lite/model.joblib` or the multimodal `.pt` if you wire the app to it.

## Security note

Anyone with your old setup URL can request keys tied to that session. If this doc or a chat was shared publicly, **rotate** by generating a new SSH setup link in Lightning and avoid reusing old tokens.
