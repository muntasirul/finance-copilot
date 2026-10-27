# Finance Copilot — Privacy-First Personal Finance Intelligence

> A local-first desktop app that turns raw bank statements (PDF/CSV) into categorised
> transactions, spending insights, anomaly alerts, cash-flow forecasts, and a natural-language
> assistant — with every model running on-device and zero data leaving the machine.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [What This Project Proves on a CV](#3-what-this-project-proves-on-a-cv)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack](#5-tech-stack)
6. [Data Strategy](#6-data-strategy)
7. [ML Components](#7-ml-components)
8. [Repository Structure](#8-repository-structure)
9. [Step-by-Step Build Plan (8 Phases)](#9-step-by-step-build-plan)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Testing Strategy](#11-testing-strategy)
12. [CI/CD & Release](#12-cicd--release)
13. [Security & Privacy Design](#13-security--privacy-design)
14. [Documentation Deliverables](#14-documentation-deliverables)
15. [Timeline](#15-timeline)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Stretch Goals](#17-stretch-goals)
18. [CV Bullets (Target)](#18-cv-bullets-target)
19. [Definition of Done](#19-definition-of-done)

---

## 1. Problem Statement

Most people cannot answer "where did my money go last month?" without manual spreadsheet work.

- Bank apps categorise transactions poorly (or not at all), especially outside the US/EU where
  merchant strings look like `POS 4521 DHK*UBERBD 2210` or `BKASH P2P 017XXXXXXX`.
- Open-banking APIs (Plaid etc.) do not exist in many countries; users only have PDF/CSV statements.
- Cloud finance apps (Mint, YNAB, Monarch) require uploading full financial history to a third
  party — a real privacy and security risk.
- Subscription creep, duplicate charges, and fraudulent transactions go unnoticed for months.
- Nobody knows whether they will run out of money before the next payday.

**Finance Copilot** solves this with an offline desktop app that ingests statements, applies
locally-trained ML models, and lets the user ask questions in plain language.

---

## 2. Goals & Non-Goals

### Goals
- Parse PDF and CSV statements from multiple banks via a pluggable parser system.
- Auto-categorise transactions with a trained text classifier (not rules only).
- Detect anomalies: unusual spend, duplicate charges, new/changed subscriptions.
- Forecast daily balance / cash-flow for the next 30 days with uncertainty bands.
- Natural-language Q&A over the user's own data using a local LLM + text-to-SQL.
- Interactive dashboards: spending by category, trends, budgets, subscriptions.
- 100% offline. Encrypted local storage. No telemetry.
- Production-grade engineering: tests, CI, signed releases, docs, model cards.

### Non-Goals (v1)
- No live bank connections / open-banking.
- No multi-user or cloud sync.
- No investment / portfolio tracking.
- No mobile app.

---

## 3. What This Project Proves on a CV

| Role claimed | Evidence produced by this project |
|---|---|
| **AI/ML Engineer** | Trained merchant-string classifier (DistilBERT + fastText baseline), anomaly detection (Isolation Forest + Autoencoder), time-series forecasting (LightGBM + N-BEATS), LoRA fine-tuned local LLM for text-to-SQL, experiment tracking, evaluation harness, quantised ONNX/GGUF export, model cards |
| **Data Analyst** | EDA notebooks, spending/budget/cohort dashboards, SQL-heavy analytics layer, insight reports, metric design (budget variance, subscription burn, savings rate) |
| **Software Engineer** | Desktop app (Tauri + React), Python backend (FastAPI sidecar), encrypted SQLite, plugin architecture for parsers, async job pipeline, >85% test coverage, GitHub Actions CI, cross-platform signed releases, semantic versioning, ADRs |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Desktop App (Tauri)                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  React + TypeScript UI                                        │  │
│  │  Dashboard │ Transactions │ Budgets │ Alerts │ Forecast │ Chat │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │ HTTP / WebSocket (localhost only)    │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │  Python Sidecar (FastAPI, bundled with PyInstaller)           │  │
│  │                                                               │  │
│  │  ┌────────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────────┐  │  │
│  │  │ Ingestion  │ │ ML Services │ │ Analytics│ │  Assistant  │  │  │
│  │  │ - PDF/CSV  │ │ - Categoriser│ │ - SQL    │ │ - Local LLM │  │  │
│  │  │ - Parser   │ │ - Anomaly   │ │   views  │ │   (Ollama / │  │  │
│  │  │   plugins  │ │ - Forecast  │ │ - KPIs   │ │   llama.cpp)│  │  │
│  │  │ - Dedupe   │ │             │ │          │ │ - Text2SQL  │  │  │
│  │  └─────┬──────┘ └──────┬──────┘ └────┬─────┘ └──────┬──────┘  │  │
│  │        └───────────────┴─────────────┴──────────────┘         │  │
│  │                              │                                │  │
│  │                   ┌──────────▼──────────┐                     │  │
│  │                   │  SQLCipher (SQLite) │  ← encrypted at rest│  │
│  │                   │  + SQLAlchemy ORM   │                     │  │
│  │                   └─────────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Offline training pipeline (separate, not shipped in the app):
  data/raw ─► synthetic generator ─► labelled dataset ─► train scripts ─► MLflow
          ─► export ONNX / GGUF ─► models/ (bundled with app release)
```

### Key design decisions (write these up as ADRs)
1. **Tauri over Electron** — smaller binary, Rust security model, native webview.
2. **Python sidecar** — keeps the ML stack in Python; Tauri spawns it as a child process bound to
   `127.0.0.1` with a random port and a per-session auth token.
3. **SQLCipher** — transparent encryption; key derived from user passphrase via Argon2id.
4. **ONNX Runtime for inference** — no PyTorch in the shipped app; smaller, faster, CPU-only.
5. **Ollama / llama.cpp for the LLM** — optional component; app is fully functional without it.
6. **Parser plugins** — each bank format is a Python class registered via entry points; users can
   add their own without touching core code.

---

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Desktop shell | Tauri 2 (Rust) | Small, secure, cross-platform |
| Frontend | React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui | Modern, fast to build |
| Charts | Recharts / Apache ECharts | Rich financial charts |
| Backend | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 | Async, typed, testable |
| Database | SQLite + SQLCipher (via `sqlcipher3`) | Encrypted, zero-config |
| Migrations | Alembic | Schema versioning |
| PDF parsing | `pdfplumber`, `camelot` (tables), `pytesseract` (scanned fallback) | Covers text and image PDFs |
| ML – classic | scikit-learn, LightGBM, fastText | Baselines and tabular |
| ML – deep | PyTorch, HuggingFace Transformers, PEFT (LoRA) | Classifier + LLM fine-tune |
| Forecasting | LightGBM, `neuralforecast` (N-BEATS), statsmodels | Baseline → deep |
| Inference | ONNX Runtime, llama.cpp / Ollama (GGUF) | CPU-only, offline |
| Experiment tracking | MLflow (local) | Reproducible experiments |
| Data versioning | DVC (local remote) | Dataset lineage |
| Testing | pytest, pytest-cov, hypothesis, Playwright (e2e), Vitest | Full pyramid |
| Lint/format | ruff, mypy, eslint, prettier, pre-commit | Code quality |
| CI/CD | GitHub Actions | Test matrix, build, release |
| Packaging | PyInstaller (sidecar), Tauri bundler | Signed installers |
| Docs | MkDocs Material | Project site |

---

## 6. Data Strategy

Real bank data is private, so the project uses a **three-tier data strategy** — this is itself
a strong talking point.

### Tier 1 — Synthetic transaction generator (`src/datagen/`)
Build a configurable generator that produces realistic statements:
- Personas (student, salaried professional, freelancer, family) with income schedules.
- Merchant catalogue of ~800 merchants across 60 categories, with realistic raw-string
  templates per bank, e.g.
  - `POS {terminal_id} {city}*{merchant_code} {date}`
  - `CARD PURCHASE {merchant_name} {city} {country}`
  - `BKASH PAYMENT {masked_phone} REF {ref}`
  - `NEFT/{ref}/{payer_name}/{bank}`
- Recurring subscriptions, salary, rent, utilities with jitter.
- Injected anomalies with ground-truth labels: duplicate charge, amount spike, new merchant
  burst, subscription price increase, out-of-pattern geography.
- Output in multiple bank CSV/PDF layouts (generate PDFs with ReportLab to test parsers).

### Tier 2 — Public datasets (for realism and transfer)
- Kaggle "Bank Transaction Categorization" / "Personal Finance Transactions" datasets.
- Kaggle "Credit Card Transactions" (simulated, with merchant category codes).
- MCC (Merchant Category Code) list from ISO 18245 / Visa for a category taxonomy.
- Use these to pre-train the classifier before fine-tuning on synthetic + personal labels.

### Tier 3 — Your own anonymised statements (optional, never committed)
- Hash account numbers, strip names, keep merchant strings and amounts.
- Label ~500 transactions via a built-in labelling screen in the app (active learning:
  the model asks for labels on its least-confident predictions).

### Category taxonomy (v1 — 2 levels, ~60 leaves)
```
Income        : salary, freelance, refund, interest, transfer_in, other_income
Housing       : rent, mortgage, utilities_electric, utilities_gas, utilities_water, internet, phone
Food          : groceries, restaurants, cafes, food_delivery
Transport     : fuel, ride_hailing, public_transit, parking, vehicle_maintenance
Shopping      : clothing, electronics, home_goods, online_marketplace
Health        : pharmacy, doctor, hospital, gym, insurance_health
Entertainment : streaming, gaming, events, books
Education     : tuition, courses, books_edu
Finance       : loan_payment, credit_card_payment, bank_fee, investment, insurance_other
Family        : childcare, pets, gifts, charity
Travel        : flights, hotels, travel_other
Transfers     : transfer_out, cash_withdrawal, p2p
Other         : uncategorised
```

---

## 7. ML Components

### 7.1 Merchant-String Categoriser
| Item | Detail |
|---|---|
| Task | Multi-class text classification (60 classes) on raw transaction descriptor + amount + weekday |
| Baselines | (a) Rules/regex, (b) TF-IDF char n-grams + Logistic Regression, (c) fastText |
| Main model | DistilBERT (or `bert-base-multilingual-cased` if Bangla/other scripts) fine-tuned; input = `"{descriptor} [SEP] amount_bucket={b} weekday={w}"` |
| Tricks | Text normalisation (strip ref numbers, masked digits → `<NUM>`), label smoothing, class weights, confidence threshold → `uncategorised` + active-learning queue |
| Metric | Macro-F1, top-3 accuracy, calibration (ECE), latency per 1k rows |
| Target | Macro-F1 ≥ 0.90 on held-out synthetic+public; ≥ 0.80 on personal data before personal fine-tune |
| Export | ONNX, int8 dynamic quantisation; < 70 MB; < 5 ms/row on CPU |
| On-device learning | User corrections stored; nightly incremental fine-tune of the classifier head only |

### 7.2 Anomaly & Subscription Detection
| Item | Detail |
|---|---|
| Subscription detection | Group by normalised merchant; detect periodicity (7/14/30/365 d ± tolerance) using autocorrelation + amount stability; flag price changes |
| Duplicate detection | Same merchant, same amount, within 72 h → rule + learned threshold |
| Point anomalies | Isolation Forest on features: amount z-score per category, hour, weekday, merchant novelty, days since last txn with merchant |
| Sequence anomalies | LSTM/dense autoencoder on 30-day spending vectors per category; reconstruction error threshold |
| Metric | Precision@k, recall on injected anomalies, false-alert rate per 100 txns |
| Target | Recall ≥ 0.85 on injected anomalies at ≤ 2 false alerts / 100 txns |

### 7.3 Cash-Flow Forecasting
| Item | Detail |
|---|---|
| Task | Forecast daily net cash-flow and end-of-day balance for next 30 days with 10/50/90 quantiles |
| Baselines | Seasonal naive, ETS, Prophet |
| Main models | LightGBM with lag/calendar/known-recurring features (quantile objective); N-BEATS for comparison |
| Known future inputs | Detected subscriptions, salary dates, scheduled bills |
| Metric | MAPE / sMAPE on balance, pinball loss, coverage of 80% interval, "days-to-zero" accuracy |
| Target | sMAPE ≤ 12%, interval coverage within ±5% of nominal |

### 7.4 Local LLM Assistant (Text-to-SQL + Insights)
| Item | Detail |
|---|---|
| Task | Answer questions like "How much did I spend on food delivery in Q3 vs Q2?" by generating SQL over a curated analytics schema, executing it, and summarising |
| Base model | Qwen2.5-1.5B-Instruct or Llama-3.2-3B-Instruct (GGUF via Ollama / llama.cpp) |
| Fine-tuning | LoRA on ~2k synthetic (question, schema, SQL, answer) pairs generated from templates + paraphrases; evaluate execution accuracy |
| Guardrails | Read-only DB connection, SQL AST validation (sqlglot) — SELECT only, whitelisted tables, row limit, timeout |
| Fallback | If no LLM installed → canned question templates with parameter pickers |
| Metric | Execution accuracy on held-out questions, answer faithfulness (compare numbers to SQL result), latency |
| Target | ≥ 85% execution accuracy on 200-question test set |

### 7.5 Experiment Management
- Every training run logged to local MLflow: params, metrics, artifacts, dataset hash (DVC).
- `make train-<component>` / `make eval-<component>` reproducible with fixed seeds.
- Model registry folder `models/` with `MODEL_CARD.md` per model.

---

## 8. Repository Structure

```
finance-copilot/
├── README.md
├── PROJECT_PLAN.md                ← this file
├── LICENSE                        (MIT)
├── Makefile
├── pyproject.toml                 (uv / poetry; ruff, mypy, pytest config)
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 (lint, type-check, tests, coverage — matrix: ubuntu/mac/win)
│   │   ├── ml-eval.yml            (runs eval harness on PRs touching models/ or src/ml/)
│   │   └── release.yml            (build sidecar + Tauri bundles, sign, attach to GitHub Release)
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/                          (MkDocs)
│   ├── index.md
│   ├── architecture.md
│   ├── adr/                       (ADR-001-tauri.md, ADR-002-sqlcipher.md, ...)
│   ├── parsers.md                 (how to write a bank parser plugin)
│   ├── models/                    (model cards)
│   └── privacy.md
├── apps/
│   └── desktop/                   (Tauri + React)
│       ├── src-tauri/
│       │   ├── src/main.rs        (spawns sidecar, manages lifecycle, secure token)
│       │   └── tauri.conf.json
│       ├── src/
│       │   ├── pages/             (Dashboard, Transactions, Budgets, Alerts, Forecast, Chat, Settings)
│       │   ├── components/
│       │   ├── api/               (typed client generated from OpenAPI)
│       │   └── store/
│       ├── e2e/                   (Playwright)
│       └── package.json
├── src/
│   └── fincopilot/                (Python package — the sidecar)
│       ├── __init__.py
│       ├── main.py                (FastAPI app factory, startup, auth middleware)
│       ├── config.py
│       ├── db/
│       │   ├── models.py          (SQLAlchemy ORM)
│       │   ├── session.py         (SQLCipher connection, key derivation)
│       │   └── migrations/        (Alembic)
│       ├── ingestion/
│       │   ├── base.py            (StatementParser ABC, ParsedTransaction schema)
│       │   ├── registry.py        (plugin discovery)
│       │   ├── parsers/
│       │   │   ├── generic_csv.py
│       │   │   ├── generic_pdf.py
│       │   │   ├── bank_a.py ...
│       │   ├── normalise.py       (descriptor cleaning)
│       │   └── dedupe.py
│       ├── ml/
│       │   ├── categoriser/
│       │   │   ├── features.py
│       │   │   ├── train.py
│       │   │   ├── evaluate.py
│       │   │   ├── export_onnx.py
│       │   │   └── infer.py       (ONNX runtime wrapper)
│       │   ├── anomaly/
│       │   │   ├── subscriptions.py
│       │   │   ├── isolation_forest.py
│       │   │   ├── autoencoder.py
│       │   │   └── evaluate.py
│       │   ├── forecast/
│       │   │   ├── features.py
│       │   │   ├── lgbm_quantile.py
│       │   │   ├── nbeats.py
│       │   │   └── evaluate.py
│       │   └── assistant/
│       │       ├── schema_prompt.py
│       │       ├── text2sql.py
│       │       ├── guard.py       (sqlglot validation)
│       │       ├── finetune_lora.py
│       │       └── evaluate.py
│       ├── analytics/
│       │   ├── views.sql          (materialised analytics views)
│       │   ├── kpis.py            (savings rate, burn, budget variance, ...)
│       │   └── reports.py
│       ├── api/
│       │   ├── routes_ingest.py
│       │   ├── routes_transactions.py
│       │   ├── routes_analytics.py
│       │   ├── routes_alerts.py
│       │   ├── routes_forecast.py
│       │   └── routes_chat.py     (WebSocket streaming)
│       └── jobs/
│           └── scheduler.py       (APScheduler: nightly retrain, anomaly scan)
├── datagen/
│   ├── personas.yaml
│   ├── merchants.yaml
│   ├── templates.yaml             (bank descriptor templates)
│   ├── generate.py
│   └── render_pdf.py
├── data/                          (DVC-tracked; raw/ interim/ processed/)
├── notebooks/
│   ├── 01_eda_transactions.ipynb
│   ├── 02_categoriser_baselines.ipynb
│   ├── 03_anomaly_analysis.ipynb
│   ├── 04_forecast_backtests.ipynb
│   └── 05_text2sql_eval.ipynb
├── models/                        (exported ONNX/GGUF + MODEL_CARD.md; large files via GitHub Releases)
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── ml/                        (metric regression tests with thresholds)
│   └── fixtures/                  (sample statements, golden outputs)
└── scripts/
    ├── bootstrap_dev.sh
    └── build_sidecar.sh
```

---

## 9. Step-by-Step Build Plan

Each phase ends with a **tagged release** (`v0.1.0`, `v0.2.0`, …) and a short changelog entry.
Work in feature branches, open PRs against `main` (even solo — reviewers notice this).

### Phase 0 — Project Setup (Days 1–2)
1. `git init`, MIT licence, `.gitignore` (Python, Node, Rust, data/, models/*.onnx).
2. Python env with `uv` (or poetry); pin Python 3.11. Add `pyproject.toml` with ruff, mypy, pytest.
3. `pre-commit` hooks: ruff, ruff-format, mypy, prettier, eslint, trailing whitespace, no large files.
4. `Makefile` targets: `setup`, `lint`, `test`, `datagen`, `train-*`, `eval-*`, `run-api`, `run-app`, `build`.
5. GitHub repo: branch protection on `main`, PR template, issue templates, labels.
6. `ci.yml`: lint + mypy + pytest with coverage upload (Codecov badge).
7. Write README skeleton: problem, demo GIF placeholder, architecture image placeholder, badges.
8. Write ADR-001 (Tauri), ADR-002 (Python sidecar), ADR-003 (SQLCipher).

**Deliverable:** green CI on an empty-but-structured repo. Tag `v0.0.1`.

---

### Phase 1 — Synthetic Data Generator (Days 3–6)
1. Define `personas.yaml` (income, pay schedule, rent, typical categories & spend distributions).
2. Define `merchants.yaml` (~800 merchants → category, typical amount range, recurrence).
3. Define `templates.yaml` — raw descriptor templates per "bank profile" (at least 4 profiles with
   different quirks: truncation to 24 chars, uppercase, embedded ref numbers, local-language words).
4. Implement `generate.py` → emits `transactions.parquet` with columns:
   `txn_id, account_id, date, posted_date, raw_descriptor, amount, direction, balance, merchant_id,
   category, subcategory, is_recurring, anomaly_type (nullable), persona`.
5. Inject anomalies with labels (≥ 5 types, ~1.5% of rows).
6. Implement `render_pdf.py` producing bank-style PDF statements with ReportLab (tables, headers,
   page breaks, running balance) for ≥ 2 layouts; plus CSV in ≥ 3 column conventions.
7. Unit tests: schema, determinism with seed, anomaly rate, balance continuity.
8. Notebook `01_eda_transactions.ipynb`: distributions, seasonality, class balance — export key
   charts to `docs/`.
9. Track `data/processed/` with DVC (local remote on an external drive is fine).

**Deliverable:** `make datagen` produces 12 personas × 24 months (~150k txns) + PDFs/CSVs. Tag `v0.1.0`.

---

### Phase 2 — Backend Core: DB, Ingestion, Parsers (Days 7–12)
1. SQLAlchemy models: `Account, Transaction, Merchant, Category, Label, Alert, Forecast,
   Budget, ImportBatch, AuditLog`.
2. SQLCipher session: derive key with Argon2id from passphrase; store salt in app dir; unit test
   that DB file is unreadable without key.
3. Alembic initial migration.
4. `StatementParser` ABC: `detect(file) -> confidence`, `parse(file) -> list[ParsedTransaction]`.
5. Registry with entry-point discovery; `generic_csv` (header heuristics + column mapper),
   `generic_pdf` (pdfplumber tables; OCR fallback flag), and 2 bank-specific parsers matching
   the synthetic PDF layouts.
6. Normalisation pipeline: unicode NFKC, strip ref/number tokens, collapse whitespace,
   merchant canonicalisation (`UBER*TRIP`, `UBER BV` → `uber`).
7. Dedupe: hash of (account, date, amount, normalised descriptor) + fuzzy window for re-imports.
8. FastAPI routes: `POST /imports` (upload → background job), `GET /imports/{id}`,
   `GET /transactions` (filter, paginate, sort), `PATCH /transactions/{id}` (manual category).
9. Localhost-only binding, per-session bearer token passed from Tauri at spawn time.
10. Tests: parser golden files (fixtures), round-trip import of every synthetic layout,
    dedupe on double import, API contract tests with `httpx`.

**Deliverable:** import any synthetic statement via API; 90%+ coverage on ingestion. Tag `v0.2.0`.

---

### Phase 3 — ML #1: Categoriser (Days 13–19)
1. Build dataset splits by **persona** (not random) to test generalisation to unseen users.
2. Baselines in `02_categoriser_baselines.ipynb`: rules → TF-IDF+LR → fastText. Log to MLflow.
3. Fine-tune DistilBERT (`train.py`): HF Trainer, class weights, label smoothing, early stopping.
4. Evaluate: macro-F1, per-class F1, confusion matrix, calibration curve, latency.
5. Export to ONNX; dynamic int8 quantisation; verify parity (Δ macro-F1 < 0.005).
6. `infer.py` wrapper with batching + confidence threshold → `uncategorised` + `needs_review`.
7. Wire into ingestion job: auto-categorise on import; store `confidence` and `model_version`.
8. Active-learning endpoint: `GET /review-queue` (lowest-confidence), `POST /labels`.
9. Incremental retrain job: retrain classifier head on user labels (freeze encoder); keep
   versioned models; rollback if eval regresses.
10. `tests/ml/test_categoriser_regression.py`: fail CI if macro-F1 < threshold on fixture set.
11. Write `docs/models/categoriser.md` model card.

**Deliverable:** auto-categorised imports, review queue, model card, metrics table in README. Tag `v0.3.0`.

---

### Phase 4 — Desktop App MVP (Days 20–27)
1. Tauri 2 scaffold; Rust `main.rs` spawns PyInstaller sidecar, waits for health check, passes token.
2. First-run flow: create passphrase → DB created; unlock screen on later launches.
3. Generate typed API client from FastAPI OpenAPI (`openapi-typescript`).
4. Pages:
   - **Import**: drag-drop, parser auto-detect, progress, summary of new/duplicate rows.
   - **Transactions**: virtualised table, filters, inline category edit (writes labels).
   - **Review queue**: swipe-style quick labelling.
   - **Dashboard**: monthly income vs spend, category donut, top merchants, trend lines.
   - **Settings**: parsers installed, model versions, export/backup, delete all data.
5. State management (Zustand/TanStack Query), error boundaries, loading skeletons, dark mode.
6. Playwright e2e: first-run → import fixture → see categorised transactions → edit a category.
7. Vitest unit tests for utilities and components.
8. Record demo GIF for README.

**Deliverable:** installable dev build on your OS; e2e green. Tag `v0.4.0`.

---

### Phase 5 — ML #2: Anomalies & Subscriptions + Alerts UI (Days 28–33)
1. `subscriptions.py`: periodicity detection; output `Subscription(merchant, cadence, amount,
   next_expected, price_change_pct)`.
2. Duplicate-charge rule + evaluation on injected duplicates.
3. Isolation Forest features + training; threshold chosen on validation to hit false-alert budget.
4. Autoencoder on category-day matrices; compare against IF; keep the better or ensemble.
5. Alerts table + nightly scan job + `GET /alerts`, `POST /alerts/{id}/dismiss` (feedback stored
   for future threshold tuning).
6. UI: Alerts page with explanation per alert ("3.1× your usual Groceries spend; new merchant").
7. Subscriptions page: list, monthly burn, upcoming renewals calendar, price-increase badges.
8. Notebook `03_anomaly_analysis.ipynb` + model card + regression test.

**Deliverable:** alerts and subscriptions working end-to-end; metrics in README. Tag `v0.5.0`.

---

### Phase 6 — ML #3: Cash-Flow Forecast + Budgets (Days 34–39)
1. Feature pipeline: daily aggregates, lags (1,7,14,28), rolling stats, calendar, payday flags,
   known-recurring future amounts from subscription detector.
2. Baselines: seasonal naive, ETS, Prophet — rolling-origin backtest (12 folds).
3. LightGBM quantile models (q10/q50/q90); N-BEATS via `neuralforecast` for comparison.
4. Metrics: sMAPE, pinball loss, interval coverage, days-to-zero error. Log to MLflow.
5. `GET /forecast` returns daily balance path + bands + "projected low point" + confidence.
6. Budgets: CRUD per category, variance vs actual, pace indicator ("on track to exceed by 14%").
7. UI: Forecast page (fan chart), Budgets page (progress bars, pace).
8. Notebook `04_forecast_backtests.ipynb`, model card, regression test.

**Deliverable:** forecast + budgets live. Tag `v0.6.0`.

---

### Phase 7 — ML #4: Local LLM Assistant (Days 40–47)
1. Curated analytics schema (views): `v_transactions`, `v_monthly_category`, `v_merchants`,
   `v_subscriptions`, `v_budgets` — small, well-named, documented in the prompt.
2. Synthetic Q→SQL dataset: ~40 question templates × paraphrases × parameter fills → ~2k pairs;
   validate every SQL executes on a synthetic DB; hold out 200 for test.
3. Zero-shot baseline with base Qwen2.5-1.5B via Ollama; measure execution accuracy.
4. LoRA fine-tune (PEFT, 4-bit if GPU available; otherwise CPU small run or Colab); merge and
   convert to GGUF (`llama.cpp` `convert_hf_to_gguf.py`), quantise Q4_K_M.
5. `guard.py`: parse with sqlglot → allow only SELECT, whitelist views, inject `LIMIT`, timeout,
   read-only connection. Tests with adversarial prompts (DROP, UNION, file reads).
6. Answer synthesis: run SQL → table → LLM summarises in 1–3 sentences with the numbers; UI shows
   the SQL and the table for transparency.
7. WebSocket streaming endpoint; Chat page with suggested questions; graceful "LLM not
   installed" state with template-based fallback.
8. Notebook `05_text2sql_eval.ipynb`, model card, regression test on execution accuracy.

**Deliverable:** natural-language Q&A over local data. Tag `v0.7.0`.

---

### Phase 8 — Hardening, Release, Portfolio Polish (Days 48–56)
1. Security pass: dependency audit (`pip-audit`, `npm audit`), threat model doc, secret scanning.
2. Performance: import 100k rows < 60 s; dashboard queries < 200 ms (add indexes, materialised views).
3. Backup/restore (encrypted export), "delete all data" with confirmation, data-retention settings.
4. `release.yml`: build sidecar per OS, Tauri bundles (dmg/msi/AppImage), code-sign where possible,
   attach to GitHub Release with checksums; models fetched from Release assets on first run.
5. MkDocs site: architecture, ADRs, model cards, parser plugin guide, privacy page.
6. README final: hero GIF, architecture diagram, **results table** (all four models vs baselines),
   quick start, "how it works", roadmap.
7. Blog post / write-up (1,000–1,500 words): "Building a fully offline finance copilot: lessons
   from messy merchant strings and text-to-SQL on a 1.5B model."
8. HuggingFace Hub: publish the categoriser (ONNX) and text-to-SQL LoRA adapter with model cards;
   publish the synthetic dataset.
9. Tag `v1.0.0`.

---

## 10. Evaluation & Metrics

| Component | Primary metric | Secondary | Baseline to beat | Target |
|---|---|---|---|---|
| Categoriser | Macro-F1 | Top-3 acc, ECE, ms/row | TF-IDF+LR | ≥ 0.90 F1, < 5 ms/row |
| Anomaly | Recall@false-alert budget | Precision@10 | Rules only | ≥ 0.85 recall @ ≤ 2 FA/100 |
| Subscriptions | Detection F1 | Price-change detection acc | Rules | ≥ 0.95 F1 |
| Forecast | sMAPE (balance) | Pinball, coverage, days-to-zero | Seasonal naive, Prophet | ≤ 12% sMAPE |
| Text-to-SQL | Execution accuracy | Faithfulness, latency | Zero-shot base model | ≥ 85% |

All numbers reported on held-out **personas** (unseen users) and, where available, on personal data.

---

## 11. Testing Strategy

- **Unit** (fast, pure): normalisation, dedupe hashing, feature functions, SQL guard, KPI maths.
  Property-based tests with Hypothesis for parsers and normaliser.
- **Integration**: end-to-end import of each fixture statement into a temp encrypted DB; API
  contract tests; scheduler jobs.
- **ML regression**: `tests/ml/` loads exported models and asserts metrics on fixed fixture sets
  don't drop below thresholds (prevents silent regressions in PRs).
- **E2E**: Playwright against the built app (first run, import, label, alert, forecast, chat).
- **Security**: adversarial text-to-SQL prompts; DB unreadable without key; sidecar refuses
  requests without token.
- Coverage target: ≥ 85% Python, ≥ 70% TypeScript.

---

## 12. CI/CD & Release

- `ci.yml` on every PR: ruff, mypy, pytest (ubuntu/macos/windows × py3.11), vitest, eslint,
  coverage upload, build check of Tauri (no signing).
- `ml-eval.yml` on PRs touching `src/fincopilot/ml/**` or `models/**`: runs eval harness, posts a
  metrics-diff comment on the PR.
- `release.yml` on tag `v*`: PyInstaller sidecar per OS → Tauri bundle → checksums → GitHub Release
  with auto-generated changelog (conventional commits + `git-cliff`).
- Semantic versioning; `CHANGELOG.md`.

---

## 13. Security & Privacy Design

- All data in a single SQLCipher DB in the OS app-data dir; key from Argon2id(passphrase, salt).
- Sidecar binds `127.0.0.1:<random>`; requires `Authorization: Bearer <session-token>` generated by
  Tauri per launch; CORS disabled; no network egress (document and test with a firewall rule).
- LLM runs locally; prompts never leave device. SQL guard enforces read-only.
- No telemetry, no crash reporting by default (opt-in local log file only).
- Threat model document: attacker with disk access, malicious statement file, prompt injection via
  merchant string (e.g. a descriptor containing "ignore previous instructions") — test this.
- Dependency pinning + `pip-audit`/`npm audit` in CI; Dependabot enabled.

---

## 14. Documentation Deliverables

- `README.md` — problem, demo, architecture, results, quick start, roadmap.
- `docs/architecture.md` + diagram (Excalidraw/Mermaid).
- `docs/adr/` — at least 6 ADRs.
- `docs/models/*.md` — model cards (data, metrics, limitations, intended use).
- `docs/parsers.md` — plugin authoring guide with a worked example.
- `docs/privacy.md` + threat model.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`.
- Blog post / Medium / dev.to article; link in README.

---

## 15. Timeline

| Phase | Scope | Days | Tag |
|---|---|---|---|
| 0 | Setup, CI, ADRs | 1–2 | v0.0.1 |
| 1 | Synthetic data generator + EDA | 3–6 | v0.1.0 |
| 2 | DB, ingestion, parsers, API | 7–12 | v0.2.0 |
| 3 | Categoriser model + active learning | 13–19 | v0.3.0 |
| 4 | Desktop app MVP | 20–27 | v0.4.0 |
| 5 | Anomalies, subscriptions, alerts | 28–33 | v0.5.0 |
| 6 | Forecasting + budgets | 34–39 | v0.6.0 |
| 7 | Local LLM assistant | 40–47 | v0.7.0 |
| 8 | Hardening, release, polish, write-up | 48–56 | v1.0.0 |

~8 weeks at ~3–4 focused hours/day. If time is short, ship **v0.5.0** as a complete portfolio
piece and continue with 6–8 as visible roadmap items.

---

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data too clean → inflated metrics | Add noise profiles, truncation, typos; validate on public datasets and personal data; report both |
| PDF parsing brittle across banks | Keep generic parser + OCR fallback; make parsers plugins; publish a "parser request" issue template |
| LLM fine-tuning needs GPU | Use Colab/Kaggle free GPU for the LoRA run; keep base-model zero-shot as fallback; document both results |
| Tauri + Python sidecar packaging pain | Prototype packaging in Phase 0 on your main OS; add other OSes in Phase 8 |
| Scope creep | Each phase ships a tag; stretch items live in roadmap, not in the current milestone |
| Prompt injection via transaction text | Treat descriptors as data, never as instructions; SQL guard; test suite with adversarial strings |

---

## 17. Stretch Goals (post v1.0)

- Multi-currency with offline FX tables.
- Receipt photo → line items (reuse document-extraction model).
- Household mode (multiple accounts, shared budgets) — still local.
- Goal planning ("save 200k by June") with scenario simulation.
- Bangla / multilingual UI and merchant model.
- Export anonymised model-improvement labels (opt-in, differential privacy) — great ML-ethics talking point.

---

## 18. CV Bullets (Target)

- Built **Finance Copilot**, a fully offline desktop personal-finance app (Tauri, React, FastAPI,
  SQLCipher) with four on-device ML models; 85%+ test coverage, cross-platform CI/CD releases.
- Fine-tuned a DistilBERT merchant-string classifier (60 classes) reaching **0.9x macro-F1**,
  exported to int8 ONNX at **<5 ms/row**, with an active-learning loop that retrains on user
  corrections.
- Designed anomaly and subscription detection (Isolation Forest + autoencoder) achieving
  **0.8x recall at ≤2 false alerts per 100 transactions**.
- Delivered 30-day cash-flow forecasting with LightGBM quantile models (**sMAPE ≤ 12%**, calibrated
  80% intervals), outperforming Prophet by xx%.
- LoRA-fine-tuned a 1.5B local LLM for text-to-SQL over personal finance data (**8x% execution
  accuracy**, +xx pts vs zero-shot) with an AST-based SQL guard against injection.
- Authored synthetic transaction dataset generator (150k labelled rows, 4 bank formats, injected
  anomalies) and published models/dataset on HuggingFace with model cards.

*(Replace x with your real numbers when done — never inflate.)*

---

## 19. Definition of Done

- [ ] All 8 phases tagged; `v1.0.0` released with installers for at least one OS.
- [ ] README has demo GIF, architecture diagram, results table with baselines.
- [ ] CI green; coverage badges; ML regression tests in place.
- [ ] Four model cards + synthetic dataset card published.
- [ ] MkDocs site deployed to GitHub Pages.
- [ ] Blog post published and linked.
- [ ] Models + adapter on HuggingFace Hub.
- [ ] CV bullets filled with real measured numbers.
