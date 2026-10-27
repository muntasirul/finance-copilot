# Finance Copilot

> Privacy-first personal finance intelligence. Import bank statements (PDF/CSV), get them
> auto-categorised, catch anomalies and subscription creep, forecast your cash-flow, and ask
> questions in plain language — with every model running **on your device**. Nothing leaves it.

![CI](https://github.com/USER/finance-copilot/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

<!-- demo GIF placeholder -->

## Why

Bank apps categorise badly, open-banking APIs don't exist in many countries, and cloud finance
apps require handing your entire financial history to a third party. Finance Copilot runs
entirely offline on top of an encrypted local database.

## What's inside

| Component | Model | Status |
|---|---|---|
| Merchant-string categoriser (60 classes) | DistilBERT fine-tune → int8 ONNX (baselines: TF-IDF+LR, fastText) | in progress |
| Anomaly & subscription detection | Isolation Forest + autoencoder + periodicity detector | planned |
| 30-day cash-flow forecast | LightGBM quantile + N-BEATS (baselines: ETS, Prophet) | planned |
| Local LLM assistant (text-to-SQL) | LoRA-tuned Qwen2.5-1.5B via llama.cpp, sqlglot guard | planned |

## Architecture

```
Tauri (React/TS UI)  ──localhost──►  Python sidecar (FastAPI)  ──►  SQLCipher (encrypted SQLite)
                                        ├── ingestion (parser plugins, normalise, dedupe)
                                        ├── ml (categoriser, anomaly, forecast, assistant)
                                        └── analytics (SQL views, KPIs)
```

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design, data strategy, and build phases.

## Quick start (dev)

```bash
make setup            # creates .venv with uv and installs dev deps
make download-data    # pulls public datasets via Kaggle API (needs ~/.kaggle/kaggle.json)
make datagen          # generates synthetic labelled transactions + statements
make test
make run-api          # http://127.0.0.1:8765/docs
```

## Results

_Will be populated with held-out-persona metrics as each model lands._

| Component | Metric | Baseline | Ours |
|---|---|---|---|
| Categoriser | Macro-F1 | – | – |
| Anomaly | Recall @ ≤2 FA/100 | – | – |
| Forecast | sMAPE | – | – |
| Text-to-SQL | Execution acc. | – | – |

## Training

Training runs on Kaggle (free GPU). Notebooks live in [`notebooks/kaggle/`](notebooks/kaggle/);
each one is self-contained and writes exported artifacts (ONNX / LoRA adapters) to `/kaggle/working`.

## License

MIT
