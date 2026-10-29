# Kaggle Training Notebooks

These notebooks are **self-contained** — no repo code needed inside Kaggle. Each reads the same
input dataset and writes exported model artifacts you download back into `models/`.

## Step 1 — generate the dataset locally

```bash
uv run python -m datagen.generate --out data/processed/synthetic \
    --kaggle-out kaggle_dataset --months 24 --seed 42
```

This creates `kaggle_dataset/` containing `transactions.parquet` + `dataset-metadata.json`.

## Step 2 — publish it as a Kaggle Dataset (one-time)

```bash
# after placing kaggle.json at ~/.kaggle/kaggle.json:
sed -i '' 's/YOUR_KAGGLE_USERNAME/<your-username>/' kaggle_dataset/dataset-metadata.json
kaggle datasets create -p kaggle_dataset --dir-mode zip
```

(Or on the Kaggle website: **New Dataset → Upload → transactions.parquet → Create**.)

## Step 3 — run the notebooks

| Notebook | What it trains | GPU | Output you download |
|---|---|---|---|
| `01_categoriser_distilbert.ipynb` | DistilBERT fine-tune vs TF-IDF baseline; exports **int8 ONNX** | optional (T4 = ~10× faster) | `categoriser/` → `models/categoriser/` |
| `02_anomaly_detection.ipynb` | Isolation Forest + periodicity/subscription detector | no | `anomaly/model.pkl` → `models/anomaly/` |
| `03_forecast_lightgbm.ipynb` | LightGBM quantile (q10/50/90) vs ETS/naive backtest | no | `forecast/lgbm_*.txt` → `models/forecast/` |
| `04_text2sql_qlora.ipynb` | LoRA fine-tune of Qwen2.5-1.5B for text-to-SQL | **yes — T4×2 or P100** | `text2sql/adapter/` → `models/assistant/` |

For each notebook on Kaggle:

1. **New Notebook** → either upload the `.ipynb` **or** open the matching file in `py/`,
   copy the whole thing, and paste it into a single code cell — it's self-contained.
2. **Input → + Add Data** → search `fincopilot-synthetic-transactions` → add.
3. **Settings → Accelerator**: GPU only for notebook 04; leave CPU for the rest.
4. **Save & Run All** → wait → in the right-hand **Output** panel download `/kaggle/working/<name>/`.
5. Copy the downloaded folder into the repo at `models/<name>/`.

## After training

```bash
# verify the ONNX model loads in the app's inference path
uv run python -c "from fincopilot.ml.categoriser.infer import load_categoriser; \
    c = load_categoriser(); print(c.version, c.predict(['walmart chicago'],[84.2],[4]))"
```

Paste real metrics from each `metrics.json` into the results table in `README.md` — never round up.
