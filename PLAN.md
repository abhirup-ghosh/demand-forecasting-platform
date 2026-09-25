# PLAN.md — Demand Forecasting Platform

**Owner:** Abhirup Ghosh
**Repo:** `abhirup-ghosh/demand-forecasting-platform` (public)
**Local path:** `/Users/abhirupghosh/Documents/Work/career/projects/demand-forecasting-platform`
**Status as of this writing:** repo scaffolded, plan written, zero implementation done. Everything
from P0.1 onward is unstarted.

This file is the **source of truth** for this project. If the GitHub Project board and this file
ever disagree, this file wins — update the board to match, not the other way round. It is written to
be followed by a lower-effort model/session with zero prior context: every task names exact files,
exact steps, and a mechanically checkable Definition of Done (DoD). Do not skip ahead or improvise
scope beyond what a task specifies without re-reading the relevant section of this file.

---

## 1. Origin and vision

This started as a take-home challenge for a Senior Data Scientist role at freshflow.ai (application
later withdrawn for unrelated reasons — the challenge itself was never submitted). The brief was:
explore a public retail dataset, forecast demand 4 weeks out, and assess model quality in business
terms, in under 3 hours. That brief is kept only as inspiration for scope — this project is not
being built to satisfy freshflow.ai or to be submitted anywhere. It is a standalone portfolio piece.

**What this becomes:** a complete, production-shaped demonstration of modern time-series forecasting
practice — not just "a model," but the full loop: rigorous EDA, a deliberately broad spectrum of
forecasting approaches evaluated head-to-head (not just one algorithm), honest backtesting with
uncertainty quantification, business-framed evaluation, a served API, an interactive dashboard,
containerization, CI, lightweight drift monitoring, and a public live demo link. The concrete dataset
is retail sales, but every design and writing choice targets the *general* problem class — multi-series
demand/capacity forecasting under uncertainty — the same shape of problem as staffing levels, energy
load, web traffic, or supply-chain planning. This is deliberate: the audience is recruiters and
hiring managers across domains, not retail specifically, so retail is the substrate and the
techniques are the point (see the "Domain framing" architecture decision below).

**On "what's the best forecasting approach to show off":** there isn't one — and picking a single
"best" model would itself be a weaker signal than what this project actually does, which is build
and honestly compare a full spectrum: naive baseline → classical statistical → gradient-boosted
trees → global deep learning → zero-shot foundation model. Seniority in this space reads as *knowing
which tier fits which situation and why*, backed by a real backtest, not as picking one trendy model.
See the Architecture Decisions table, "Forecasting breadth" row.

---

## 2. Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Language/runtime | Python 3.11+, managed via `uv` | `uv` is the current (2024+) fast, lockfile-based Python package/env manager. Using it over plain pip/poetry is a small, low-risk signal of current tooling fluency. |
| Core dataframe library | `pandas` (+ `pyarrow` backend) | Universal compatibility with the whole forecasting ecosystem below (statsforecast/mlforecast/neuralforecast are all pandas-frame-native). `polars` is deferred to P1 to avoid interop friction on the first build — see P1.1. |
| Forecasting breadth | 5-tier spectrum: naive baseline → `statsforecast` (AutoARIMA/AutoETS) → `mlforecast` (LightGBM, global model) → `neuralforecast` (NHITS, global deep model) → `Chronos-Bolt` (zero-shot foundation model) | Demonstrates the full modern toolkit and an honest, data-driven comparison rather than a single algorithm choice. Recruiters from different backgrounds will each recognize at least one tier; the comparison itself is the impressive part. |
| Deep learning model | NHITS via `neuralforecast` | Strong published benchmark performance, materially faster/simpler to configure than Transformer-based alternatives (PatchTST/TFT) — good "modern architecture" credibility per unit of build time for a scoped portfolio piece. PatchTST is the documented fallback if NHITS clearly underperforms (open decision #1). |
| Foundation model | `Chronos-Bolt-Small` (Amazon, via the `chronos-forecasting` package / Hugging Face) | Open-weights, genuinely zero-shot, runs on CPU in reasonable time, no API key or hosted-service cost (unlike Nixtla's hosted TimeGPT). Directly demonstrates awareness of the 2024–2026 "foundation models for time series" trend, which is a strong differentiator in interviews. |
| Uncertainty quantification | Conformal prediction intervals (`statsforecast`/`mlforecast` built-in `ConformalIntervals`) | Distribution-free, well-defended under interview questioning, avoids the "assumed-Gaussian-residuals" pitfall of naive quantile regression. |
| Primary point-forecast metric | WAPE (weighted absolute percentage error) | Robust to this dataset's heavy zero-inflation, where plain MAPE is undefined/explodes on zero-actual days. Directly interpretable to a business audience as "% of total demand volume missed." |
| Business framing | Simple asymmetric over-forecast vs under-forecast cost model | Directly answers the original brief's "explain in business terms" ask. Explicitly labeled as illustrative, since Favorita's real margin/holding-cost figures aren't available — see open decision #5. |
| Experiment tracking | MLflow, local/self-hosted (SQLite backend, Docker service in compose) | Free, open-source, industry-standard; shows tracking/registry maturity without requiring a paid W&B account. |
| Serving | FastAPI + Pydantic schemas | Modern async-capable API standard; free auto-generated OpenAPI docs at `/docs` are a nice recruiter-facing touch for zero extra work. |
| Dashboard/UI | Streamlit + Plotly | Fastest path to a genuinely interactive, good-looking public demo without building a separate JS frontend. |
| Containerization | Docker + `docker-compose` (api, dashboard, mlflow services) | Standard, portable, directly runnable by anyone who clones the repo. |
| CI | GitHub Actions (lint via `ruff`, test via `pytest`, Docker build) | Free on public repos; a visible green-checkmark badge is a cheap, real credibility signal. |
| Drift monitoring | Evidently — lightweight batch HTML report in P0, continuous/scheduled in P1 | Shows train/serving-skew awareness (real MLOps maturity) without building a full always-on monitoring stack inside a scoped portfolio build. |
| Deployment target | Hugging Face Spaces (Docker SDK), hosting the dashboard | Free, zero maintenance, no credit card, supports Docker directly — the best fit for a public, always-available recruiter-facing link. The FastAPI service is still fully built/tested/dockerized (see P0.9, P0.12) even though it isn't the thing kept always-on publicly. |
| Data handling | Kaggle CSVs are **never committed**; `data/raw/` and `data/processed/` are gitignored; data is fetched via the `kaggle` CLI or a documented manual fallback | The repo is **public**. Redistributing the raw competition dataset would violate Kaggle's competition rules. This is a hard constraint, not a style preference — see `data/README.md` (created in P0.2). |
| Repo naming / narrative framing | "Demand Forecasting Platform"; README/docs written around the general demand-forecasting problem class; real retail data kept as-is (not artificially renamed) | Matches the "generic narrative, real data" decision: real data keeps the project credible and lets you speak to real data-quality issues in interviews, while the writing frames it as a transferable technique set, not a retail project. |

---

## 3. Data model / design

### 3.1 Source data (Kaggle "Store Sales - Time Series Forecasting")

Six files, once downloaded into `data/raw/` (see P0.2 — **never commit these**):

| File | Grain | Key columns |
|---|---|---|
| `train.csv` | date × store_nbr × family | `id` (int), `date` (YYYY-MM-DD), `store_nbr` (int, 1–54), `family` (str, 33 categories, e.g. `GROCERY I`, `BEVERAGES`, `DAIRY`), `sales` (float — the forecast target; fractional because some items are sold by weight), `onpromotion` (int — count of items in that family on promotion at that store that day) |
| `stores.csv` | store_nbr | `store_nbr`, `city`, `state`, `type` (A–E), `cluster` (1–17) |
| `oil.csv` | date | `date`, `dcoilwtico` (float, daily WTI oil price; gaps on weekends/holidays — Ecuador's economy is oil-dependent, an interesting macro feature) |
| `holidays_events.csv` | date (multiple rows per date possible) | `date`, `type` (`Holiday`/`Transfer`/`Additional`/`Bridge`/`Work Day`/`Event`), `locale` (`National`/`Regional`/`Local`), `locale_name`, `description`, `transferred` (bool — **if true, the actual holiday observance moved to the paired `Transfer` row's date; the `Holiday` row itself is a normal working day.** Get this right in feature engineering, P0.4.) |
| `transactions.csv` | date × store_nbr | `date`, `store_nbr`, `transactions` (int, total store transactions that day) |
| `test.csv` / `sample_submission.csv` | — | Kaggle's official scoring files. **Not used** — we have no way to grade against Kaggle's hidden ground truth outside their platform, and we don't need to: see 3.2, we define our own honest holdout instead. |

Train data spans **2013-01-01 to 2017-08-15** (~4.5 years, daily). Unique `(store_nbr, family)`
combinations: up to 54 × 33 = 1782 series (validate the exact count in P0.2 — some combinations may
not exist or may be structurally all-zero, e.g. a family never stocked at a given store).

### 3.2 Forecast task definition (our own, not Kaggle's)

- **Target:** daily `sales`, per `(store_nbr, family)` series.
- **Horizon:** 28 days (4 weeks) — matches the original brief.
- **Granularity:** daily.
- **Series scope:** all ~1782 `(store_nbr, family)` series for the naive/statistical/ML/deep tiers
  (these libraries are built for exactly this scale). The Chronos zero-shot tier is run on a
  **stratified sample of 60 series** (top 20 by total volume, middle 20, bottom 20 excluding
  structurally-all-zero series) — an explicit, documented scoping decision to keep zero-shot
  inference runtime reasonable on a laptop, not a limitation of the method. State this plainly in
  `docs/model-evaluation.md` so it doesn't read as a shortcut.
- **Final holdout (matches "next 4 weeks"):** train on everything before **2017-07-19**, evaluate on
  **2017-07-19 .. 2017-08-15** (the last 28 days of available data).
- **Backtesting beyond the single holdout:** additionally run **5-fold rolling-origin
  cross-validation**, folds stepping backward in 28-day increments from the final holdout, each with
  an expanding training window. This avoids drawing conclusions from one possibly-lucky/unlucky
  4-week window. See P0.5 for the exact fold-generation logic.

### 3.3 Repository layout

Created incrementally as each P0 task lands — do not pre-create empty directories ahead of the task
that needs them, except where a task explicitly says to scaffold a package `__init__.py`.

```
demand-forecasting-platform/
├── README.md
├── PLAN.md
├── LICENSE
├── .gitignore
├── pyproject.toml            # P0.1
├── Makefile                  # P0.1
├── .env.example              # P0.1
├── docker-compose.yml        # P0.12
├── Dockerfile.api             # P0.9 / P0.12
├── Dockerfile.dashboard       # P0.10 / P0.12
├── .github/workflows/ci.yml  # P0.13
├── data/
│   ├── raw/                  # gitignored — P0.2
│   ├── processed/            # gitignored — P0.4
│   └── README.md             # P0.2, committed
├── src/forecasting_platform/
│   ├── __init__.py           # P0.1
│   ├── config.py             # P0.1
│   ├── data/
│   │   ├── download.py       # P0.2
│   │   └── validate.py       # P0.2
│   ├── features/
│   │   └── engineering.py    # P0.4
│   ├── models/
│   │   ├── baseline.py       # P0.6a
│   │   ├── statistical.py    # P0.6b
│   │   ├── ml.py              # P0.6c
│   │   ├── deep.py            # P0.6d
│   │   └── foundation.py     # P0.6e
│   ├── evaluation/
│   │   ├── backtest.py       # P0.5
│   │   ├── metrics.py        # P0.7
│   │   └── business_cost.py  # P0.7
│   ├── monitoring/
│   │   └── drift.py          # P0.11
│   └── serving/
│       ├── api.py             # P0.9
│       └── schemas.py         # P0.9
├── dashboard/
│   └── app.py                 # P0.10 (kept top-level, not under src/, since Streamlit's own
│                               #   convention favors a standalone entry script)
├── scripts/
│   ├── generate_test_fixture.py  # P0.2
│   ├── run_backtest.py           # P0.7
│   ├── train_final_model.py      # P0.8
│   └── generate_drift_report.py  # P0.11
├── notebooks/
│   └── 01_eda.ipynb          # P0.3
├── tests/
│   ├── fixtures/sample_train.csv  # P0.2, synthetic — see P0.2 step 4
│   ├── test_data.py           # P0.2
│   ├── test_features.py       # P0.4
│   ├── test_backtest.py       # P0.5
│   └── test_api.py            # P0.9
└── docs/
    ├── eda-findings.md         # P0.3
    ├── model-evaluation.md    # P0.15
    └── architecture.md         # P0.15
```

---

## 4. P0 — build the full pipeline end-to-end

P0 is the whole deliverable for a "tight portfolio piece": by the end of P0, the platform runs
end-to-end locally (`docker compose up`) and has one live public demo link. Do the tasks in order —
later tasks depend on earlier ones' files existing. Each task lists exact files, steps, and a DoD
you can check by running a command or observing a concrete output.

### P0.1 — Environment & tooling

**Goal:** a reproducible Python environment and the project skeleton every later task builds on.

**Files:** `pyproject.toml`, `Makefile`, `.env.example`, `src/forecasting_platform/__init__.py`,
`src/forecasting_platform/config.py`

**Steps:**
1. From the repo root, run `uv init --no-readme --name forecasting-platform --package .` (this
   creates/updates `pyproject.toml` and `src/forecasting_platform/`). If `uv init` complains that
   files already exist (README.md, PLAN.md, etc. are already there from kickoff), that's fine —
   it only touches `pyproject.toml` and creates the `src/` package layout.
2. Edit `pyproject.toml`: set `requires-python = ">=3.11"`.
3. Add runtime dependencies:
   `uv add pandas pyarrow numpy scikit-learn statsforecast mlforecast neuralforecast chronos-forecasting lightgbm fastapi "uvicorn[standard]" pydantic pydantic-settings streamlit plotly mlflow evidently kaggle python-dotenv`
4. Add dev dependencies:
   `uv add --dev pytest pytest-cov ruff mypy httpx jupyter`
5. Create `Makefile` with at least these targets (each a one-line wrapper over the `uv run ...`
   command the relevant task below specifies): `setup`, `data`, `validate-data`, `eda`, `features`,
   `backtest`, `train-final`, `api`, `dashboard`, `drift-report`, `test`, `lint`, `docker-up`,
   `docker-down`.
6. Create `src/forecasting_platform/config.py` with a `pydantic-settings` `Settings` class
   (loaded from `.env`, see step 7) exposing at least: `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`,
   `FORECAST_HORIZON: int = 28`, `QUANTILES: list[float] = [0.1, 0.5, 0.9]`,
   `BACKTEST_N_FOLDS: int = 5`, `BACKTEST_STEP_DAYS: int = 28`, `RANDOM_SEED: int = 42`,
   `MLFLOW_TRACKING_URI: str = "sqlite:///mlruns.db"`, `CHRONOS_MODEL_ID: str = "amazon/chronos-bolt-small"`,
   `CHRONOS_SERIES_SAMPLE_SIZE: int = 60`, `COST_UNDER_PER_UNIT: float = 3.0`,
   `COST_OVER_PER_UNIT: float = 1.0`. A single module-level `settings = Settings()` instance,
   imported everywhere else instead of re-reading env vars ad hoc.
7. Create `.env.example` with commented placeholders for `KAGGLE_USERNAME` and `KAGGLE_KEY` (get
   these from https://www.kaggle.com/settings → API → "Create New Token"; never commit the real
   `.env` — it's already gitignored).

**Definition of Done:**
- `uv run python -c "from forecasting_platform.config import settings; print(settings.FORECAST_HORIZON)"` prints `28`.
- `uv run pytest --collect-only` runs without import errors (0 tests collected is fine at this point).
- `uv run ruff check .` passes with no errors on the skeleton.

---

### P0.2 — Data acquisition & validation

**Goal:** get the real dataset locally (never committed) and a synthetic fixture (safe to commit)
for tests/CI, plus a validator that confirms the raw data matches what the rest of the pipeline
assumes.

**Files:** `src/forecasting_platform/data/download.py`, `src/forecasting_platform/data/validate.py`,
`data/README.md`, `scripts/generate_test_fixture.py`, `tests/fixtures/sample_train.csv`,
`tests/test_data.py`

**Steps:**
1. Write `data/README.md` documenting: the exact file list and schema from section 3.1 of this
   plan; the date range; **the redistribution constraint** (raw CSVs must never be committed — this
   repo is public); two ways to populate `data/raw/`:
   - **Kaggle CLI (preferred):** put `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`, then
     `uv run python -m forecasting_platform.data.download` (implemented in step 2).
   - **Manual fallback:** download the zip from the competition's Kaggle page by hand, unzip into
     `data/raw/`.
2. In `download.py`, implement `download_kaggle_dataset(dest_dir: Path) -> None`: authenticate via
   the `kaggle` package using env vars from `.env` (loaded via `python-dotenv`), call the
   competition download for `store-sales-time-series-forecasting`, unzip into `dest_dir`. Wrap the
   Kaggle API call so a clear error message points at `data/README.md`'s manual fallback if
   credentials are missing, rather than a raw stack trace.
3. In `validate.py`, implement `validate_raw_data(raw_dir: Path) -> None` that asserts and prints a
   summary for: all 6 expected files present; `train.csv` has exactly the columns from section 3.1
   with expected dtypes; `train.csv`'s date range matches 2013-01-01..2017-08-15; number of unique
   `(store_nbr, family)` pairs (print it — expect close to 1782, some may be missing); every
   `store_nbr` in `train.csv` exists in `stores.csv`; report the count of series that are
   structurally all-zero (sales == 0 for every row) — these need special handling later (they should
   just forecast zero, not be fed to every model tier). Raise with a clear message on any hard
   schema mismatch; only warn (print) on the informational counts.
4. In `scripts/generate_test_fixture.py`, generate a **synthetic** fixture — do **not** sample rows
   from the real downloaded data, since even a small extract is still redistribution of the Kaggle
   dataset. Programmatically create ~3 fake stores × ~3 fake families × 120 days with: a weekly
   seasonal pattern, a mild upward trend, additive noise, occasional zero-sales days, and a plausible
   `onpromotion` column — same column names/dtypes as real `train.csv`. Write to
   `tests/fixtures/sample_train.csv` (committed to the repo; this file requires no Kaggle credentials
   to exist and unblocks CI, see P0.13).
5. In `tests/test_data.py`, test `validate_raw_data` against `tests/fixtures/sample_train.csv`
   (schema checks only — the synthetic fixture doesn't need to match the real date range or series
   count, so parametrize those checks or skip them for the fixture path).

**Definition of Done:**
- `uv run python -m forecasting_platform.data.download` (with real Kaggle credentials in `.env`)
  populates `data/raw/` with all 6 files, OR the manual-download fallback in `data/README.md`
  produces the same result.
- `uv run python -m forecasting_platform.data.validate` against the real `data/raw/` prints "all
  checks passed" plus the series count and zero-inflation summary.
- `uv run pytest tests/test_data.py` passes using only the synthetic fixture (no real data or
  Kaggle credentials required — this must work in CI).

---

### P0.3 — Exploratory data analysis

**Goal:** the "short presentation of findings" the original brief asked for, productized as a
notebook + a written summary that later feeds the dashboard's Overview tab (P0.10) and the "where
this model breaks" section (P0.15).

**Files:** `notebooks/01_eda.ipynb`, `docs/eda-findings.md`

**Steps:** cover at least these 8 angles, each producing one chart and one or two sentences of
interpretation in `docs/eda-findings.md`:
1. Aggregate daily sales trend + weekly/yearly seasonality (plot rolling mean over the full 2013–2017 span).
2. Zero-inflation / intermittent demand: % of `(store, family, day)` rows with `sales == 0`, broken
   down by `family` — expect this to vary hugely (e.g. `BOOKS` near-always zero at many stores vs
   `GROCERY I` almost never zero) — this directly motivates the model-tier discussion in
   `docs/model-evaluation.md` (P0.15).
3. Holiday effects: average sales on `type == "Holiday"`, `locale == "National"` dates vs regular
   days. **Remember the `transferred` nuance from section 3.1** — a `transferred == True` holiday row
   is actually a normal working day; the paired `Transfer` row is when the holiday is really observed.
4. Promotion effect: correlation between `onpromotion` and `sales`, by family — report a few families
   where it's strong and a few where it's weak/absent.
5. Store heterogeneity: sales distribution by store `type` (A–E) and `cluster`.
6. Oil price macro relationship: correlate national aggregate daily sales against `dcoilwtico`.
   Report the honest result even if weak/null — Ecuador is oil-export dependent at a macro level, but
   the relationship to day-to-day grocery retail sales may well be indirect or negligible; a
   negative finding, stated plainly, is a better rigor signal than forcing a story.
7. **Regime-shift case study — the 2016-04-16 Ecuador earthquake:** show the sales pattern in the
   days/weeks after this date for affected regions/stores. This becomes the concrete worked example
   of "where a naive/statistical model breaks" in `docs/model-evaluation.md`.
8. Missing data in `oil.csv` (gaps on non-trading days) — document that the feature pipeline (P0.4)
   forward-fills these.

**Definition of Done:**
- `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`
  completes without error.
- `docs/eda-findings.md` contains all 8 numbered findings, each with a concrete number or chart
  reference.

---

### P0.4 — Feature engineering

**Goal:** a single, leakage-free feature-building function shared by the ML tier (P0.6c) and used
for reference by the others.

**Files:** `src/forecasting_platform/features/engineering.py`, `tests/test_features.py`

**Steps:**
1. Implement `build_feature_frame(sales_df: pd.DataFrame, stores_df: pd.DataFrame, oil_df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame`
   returning a long-format frame with columns: `unique_id` (str, `f"{store_nbr}_{family}"`), `ds`
   (date), `y` (sales), `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28`,
   `rolling_std_7`, `dow` (0–6), `month` (1–12), `is_national_holiday` (bool, correctly handling the
   `transferred` nuance from P0.3 step 3), `is_regional_holiday`, `is_local_holiday`, `onpromotion`,
   `oil_price` (forward-filled from `oil.csv`), `store_type`, `store_cluster`, `family`.
2. Note the **target transform is an open decision** (see Open Decisions #4): implement the function
   to return raw `y`; whether to `log1p`-transform before modeling is decided empirically in P0.7 by
   comparing backtest WAPE with/without, not fixed here.
3. Write an explicit **no-lookahead-leakage test**: for a synthetic series, assert that
   `lag_7` at date `D` equals `y` at `D - 7 days` for the same `unique_id`, and that no feature value
   at row `D` was computed using any row with `ds >= D`.

**Definition of Done:**
- `uv run pytest tests/test_features.py` passes, including the explicit leakage test.

---

### P0.5 — Backtesting framework

**Goal:** one shared, tested definition of the CV folds every model tier is evaluated on, so
comparisons across tiers are apples-to-apples.

**Files:** `src/forecasting_platform/evaluation/backtest.py`, `tests/test_backtest.py`

**Steps:**
1. Implement `generate_cv_folds(dates: pd.Series, horizon: int = 28, n_folds: int = 5) -> list[Fold]`
   where `Fold` is a small dataclass with `train_end_date`, `test_start_date`, `test_end_date`.
   Folds step backward from the end of the data in `horizon`-day increments; each fold's training
   window is **expanding** (everything before `train_end_date`), not a fixed sliding window. Fold 1
   (most recent) must be exactly the 2017-07-19..2017-08-15 holdout from section 3.2.
2. Implement `split_train_test(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]`
   applying a fold to a features/sales frame.

**Definition of Done:**
- `uv run pytest tests/test_backtest.py` passes, including: fold test windows do not overlap; fold 1's
  test window matches the exact dates from section 3.2; each fold's train window contains no dates
  `>= test_start_date`.

---

### P0.6 — Model tiers

Implement each tier as its own module. Every tier's public function takes a fold's train/test split
and returns a **long-format forecast frame**: columns `unique_id`, `ds`, `model_name`, `yhat`,
`yhat_lo80`, `yhat_hi80` (80% conformal interval). Consult each library's current docs for exact
constructor/method signatures if they differ from below (library APIs shift slightly across
versions) — the important, non-negotiable part is the **input/output contract** just described, so
every tier plugs into the same evaluation code (P0.7) unmodified.

#### P0.6a — Baseline
**File:** `src/forecasting_platform/models/baseline.py`
Implement `seasonal_naive(train_df, horizon, season_length=7)` (forecast = value from 7 days prior,
repeated) and `naive(train_df, horizon)` (forecast = last observed value, flat). No interval
estimate needed for these — use a constant-width band derived from in-sample residual std as a crude
fallback, documented as such.
**DoD:** returns a correctly-shaped forecast frame for a synthetic series; `uv run pytest
tests/test_models_baseline.py` (write this alongside) passes.

#### P0.6b — Classical statistical
**File:** `src/forecasting_platform/models/statistical.py`
Wrap `statsforecast.StatsForecast` with `models=[AutoARIMA(), AutoETS()]`, `freq="D"`, `n_jobs=-1`.
Use built-in conformal intervals (`level=[80]`) for the interval columns.
**DoD:** produces forecasts for all series in a fold within a few minutes on a laptop; forecast frame
matches the shared contract.

#### P0.6c — Gradient-boosted trees (global ML model)
**File:** `src/forecasting_platform/models/ml.py`
Wrap `mlforecast.MLForecast` with an `LGBMRegressor`, using lag/rolling-window transforms matching
P0.4's windows (7/14/28) for comparability, plus the calendar/holiday/promo/oil features from P0.4
as static/exogenous regressors. Use `mlforecast`'s `PredictionIntervals` for the conformal interval
columns.
**DoD:** same contract; log feature importances (from the fitted LightGBM model) to a plain text/CSV
artifact for later inclusion in `docs/model-evaluation.md`.

#### P0.6d — Global deep learning
**File:** `src/forecasting_platform/models/deep.py`
Wrap `neuralforecast.NeuralForecast` with `NHITS(h=28, input_size=56, ...)`, trained globally across
all series at once (this is the point of a "global" model — one model learns cross-series patterns).
**DoD:** trains without error on the full series set within a reasonable time budget (document actual
wall-clock time observed — if it's impractically slow on available hardware, that's a real, documented
finding, not a blocker: fall back to training on a stratified subset and say so explicitly, same as
the Chronos scoping decision).

#### P0.6e — Zero-shot foundation model
**File:** `src/forecasting_platform/models/foundation.py`
Load `amazon/chronos-bolt-small` via the `chronos-forecasting` package's `BaseChronosPipeline.from_pretrained(...)`.
Run **zero-shot** (no fine-tuning) `.predict(context, prediction_length=28)` per series, only on the
stratified 60-series sample from section 3.2 (`settings.CHRONOS_SERIES_SAMPLE_SIZE`). Convert the
model's quantile outputs to the shared contract (median → `yhat`, 10th/90th percentile → interval
columns).
**DoD:** produces forecasts for the 60-series sample; runtime logged; explicitly note in code comments
and later in `docs/model-evaluation.md` that this tier's series scope is intentionally narrower than
the others.

---

### P0.7 — Evaluation & business cost

**Goal:** run every tier through every backtest fold, compute a shared metric set, and translate
error into a business-framed cost.

**Files:** `src/forecasting_platform/evaluation/metrics.py`,
`src/forecasting_platform/evaluation/business_cost.py`, `scripts/run_backtest.py`

**Steps:**
1. In `metrics.py`, implement `wape(y_true, y_pred)`, `mape(y_true, y_pred)` (documented as unstable
   on zero-actual rows — report but don't lead with it), `rmse(y_true, y_pred)`,
   `pinball_loss(y_true, y_pred_quantiles, quantiles)`, `interval_coverage(y_true, lo, hi)` (fraction
   of actuals falling inside the predicted interval — compare against the nominal 80% to check
   calibration).
2. In `business_cost.py`, implement `business_cost(y_true, y_pred, cost_under=settings.COST_UNDER_PER_UNIT, cost_over=settings.COST_OVER_PER_UNIT)`
   computing `sum(cost_under * max(0, actual - forecast) + cost_over * max(0, forecast - actual))`.
   Document in a module docstring that the 3:1 under:over ratio is an **illustrative default**
   (stockout/lost-sale cost assumed higher than overstock/holding cost for perishable grocery), not
   fitted from Favorita's real financials — and that it's exposed as an adjustable dashboard control
   (P0.10) precisely because the "right" ratio is a business input, not a data-science one.
3. In `scripts/run_backtest.py`, orchestrate: for each of the 5 folds (P0.5) × each model tier
   (P0.6a–e, Chronos only on its subset), fit/predict, compute all metrics + business cost, write one
   row per `(model, fold)` to `results/leaderboard.csv`, and log the same as an MLflow run (params:
   `model`, `fold`; metrics: `wape`, `rmse`, `pinball_loss`, `interval_coverage`, `business_cost`;
   artifact: a forecast-vs-actual plot for a couple of representative series).

**Definition of Done:**
- `uv run python scripts/run_backtest.py` completes and produces `results/leaderboard.csv` with one
  row per `(model, fold)` pair.
- `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db` shows all the logged runs with their
  metrics.

---

### P0.8 — Model selection & final artifact

**Goal:** pick a champion from the backtest results (not decided in advance — see Open Decisions),
train it on the full history, and register it for serving.

**Files:** `scripts/train_final_model.py`

**Steps:**
1. From `results/leaderboard.csv`, compute mean WAPE across folds per model (excluding the naive
   baselines, which are a floor for comparison, not real candidates). The **lowest mean WAPE wins** —
   whichever model that turns out to be, document the actual result and the reasoning in
   `docs/model-evaluation.md` (P0.15). Do not hardcode an assumed winner in this plan.
2. Retrain the champion on the **full history** (through 2017-08-15, no holdout held back — this is
   the model that will actually serve forecasts for dates after the dataset ends).
3. Register it in the MLflow Model Registry as `demand-forecast-champion`, stage `Production`.

**Definition of Done:**
- MLflow's registry UI shows a `demand-forecast-champion` model with a version in `Production` stage.
- The registered model is loadable via `mlflow.pyfunc.load_model("models:/demand-forecast-champion/Production")`
  without error.

---

### P0.9 — Serving API

**Goal:** a FastAPI service serving forecasts from the registered champion model.

**Files:** `src/forecasting_platform/serving/api.py`, `src/forecasting_platform/serving/schemas.py`,
`tests/test_api.py`, `Dockerfile.api`

**Steps:**
1. In `schemas.py`, define Pydantic models: `ForecastRequest` (`store_nbr: int`, `family: str`,
   `horizon: int = 28`, validated `<= 28`), `ForecastPoint` (`ds: date`, `yhat: float`,
   `yhat_lo80: float`, `yhat_hi80: float`), `ForecastResponse` (`unique_id: str`, `model_name: str`,
   `forecast: list[ForecastPoint]`).
2. In `api.py`, build a FastAPI app with: `GET /health` (returns `{"status": "ok"}`); `GET /models`
   (returns the champion model's name + its latest backtest metrics from `results/leaderboard.csv`);
   `POST /forecast` (loads the champion model at app startup via MLflow, returns a `ForecastResponse`
   for the requested `store_nbr`/`family`, erroring clearly — HTTP 404 — for an unknown combination).
3. Write `Dockerfile.api`: multi-stage build using `uv`, `EXPOSE 8000`, `CMD ["uvicorn", "forecasting_platform.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]`.

**Definition of Done:**
- `uv run uvicorn forecasting_platform.serving.api:app --reload` starts; `curl localhost:8000/health`
  returns `{"status": "ok"}`; `localhost:8000/docs` renders the OpenAPI UI.
- `uv run pytest tests/test_api.py` (using FastAPI's `TestClient`) passes, covering `/health` and at
  least one successful and one 404 `/forecast` case.

---

### P0.10 — Dashboard

**Goal:** the recruiter-facing interactive surface — this is what the live demo link (P0.14) points
at.

**Files:** `dashboard/app.py`, `Dockerfile.dashboard`

**Steps:** build a Streamlit app with 4 tabs:
1. **Overview:** the 8 EDA findings from `docs/eda-findings.md` (P0.3), rendered with their charts
   (recompute lightweight ones inline; embed static images for expensive ones) — this is literally
   the original brief's "short presentation of findings," now interactive.
2. **Forecast Explorer:** dropdowns for `store_nbr` and `family`; plot historical `sales` + the
   champion model's forecast + its 80% interval; optionally overlay other model tiers' forecasts for
   the same series for comparison.
3. **Model Leaderboard:** render `results/leaderboard.csv` as a sortable table plus a bar chart of
   mean WAPE per model tier across folds.
4. **Business Impact:** sliders for `cost_under`/`cost_over` (defaulting to the illustrative 3:1 from
   P0.7); recompute and display total business cost for the champion model live as the sliders move.

For the **standalone/public deployment mode** (used on Hugging Face Spaces, P0.14), the dashboard
imports the forecasting/serving logic directly (in-process) rather than calling the FastAPI service
over the network — HF Spaces' free tier runs one container/process, so this avoids needing two
always-on services there. For the **local docker-compose mode** (P0.12), wire the dashboard to call
the FastAPI service over the compose network instead, as the more realistic microservice demo. Use a
config flag (`settings.DASHBOARD_STANDALONE_MODE`, add to `config.py`) to switch between the two
without duplicating the dashboard code.

**Definition of Done:**
- `uv run streamlit run dashboard/app.py` launches locally; all 4 tabs render without error against
  the real (downloaded) data.

---

### P0.11 — Drift monitoring (lightweight)

**Goal:** a real, if minimal, MLOps monitoring artifact — not a full production system, but genuine
evidence of train/serving-skew awareness.

**Files:** `src/forecasting_platform/monitoring/drift.py`, `scripts/generate_drift_report.py`

**Steps:**
1. In `drift.py`, wrap Evidently's `DataDriftPreset` comparing a reference window (the earliest
   backtest fold's training period) against the most recent 28-day window, over the feature set from
   P0.4.
2. In `generate_drift_report.py`, run this and save an HTML report to
   `reports/generated/drift_report.html` (gitignored — regenerable).
3. Link this report (or surface its top-line drift score) from the dashboard's Overview tab.

**Definition of Done:**
- `uv run python scripts/generate_drift_report.py` produces `reports/generated/drift_report.html`
  and it opens/renders correctly in a browser.

---

### P0.12 — Containerization

**Goal:** the whole stack runs with one command.

**Files:** `Dockerfile.api` (from P0.9), `Dockerfile.dashboard` (from P0.10), `docker-compose.yml`

**Steps:**
1. Write `Dockerfile.dashboard` mirroring `Dockerfile.api`'s multi-stage `uv` pattern, `EXPOSE 8501`,
   `CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]`.
2. Write `docker-compose.yml` with three services: `api` (port 8000), `dashboard` (port 8501,
   `DASHBOARD_STANDALONE_MODE=false`, depends on `api`), `mlflow` (port 5000, SQLite backend, a named
   volume for `mlruns.db` and artifacts).

**Definition of Done:**
- `docker compose up --build` starts all three services; `curl localhost:8000/health` succeeds;
  `localhost:8501` serves the dashboard and its Forecast Explorer tab successfully calls the `api`
  service (not standalone mode) for a live forecast.

---

### P0.13 — CI

**Goal:** a real, green CI badge — a cheap, honest credibility signal on a public repo.

**Files:** `.github/workflows/ci.yml`

**Steps:** on push/PR to `main`: check out, set up `uv` (the official `astral-sh/setup-uv` action),
`uv sync`, `uv run ruff check .`, `uv run pytest --cov=forecasting_platform` (this must pass using
only `tests/fixtures/sample_train.csv` — no Kaggle credentials available in CI, per the synthetic
fixture design from P0.2), then `docker build` (build-only, no push) for both `Dockerfile.api` and
`Dockerfile.dashboard` to catch Docker breakage early.

**Definition of Done:**
- Push a branch, open a PR (or push directly to `main` if working solo), and confirm via
  `gh run list` / the GitHub Actions tab that the workflow runs and passes.

---

### P0.14 — Deployment (public live demo)

**Goal:** a clickable, always-available public URL — the single most important recruiter-facing
artifact in this whole project.

**Files:** a Space-specific Dockerfile (can reuse `Dockerfile.dashboard` with
`DASHBOARD_STANDALONE_MODE=true`), README badge/link update

**Steps:**
1. Create a Hugging Face Space (Docker SDK) named `demand-forecasting-platform` under the
   `abhirup-ghosh` HF account.
2. The Space needs the trained champion model artifact available at build time. Since
   `artifacts/*.pkl`/MLflow's local store are gitignored in the *GitHub* repo (see the Data handling
   architecture decision — that gitignore rule exists for the raw Kaggle CSVs, not model weights, but
   the same directories are convenient to exclude broadly), commit a small serialized copy of the
   champion model **into the HF Space's own git repo** (HF Spaces are separate git repos from
   GitHub — this does not touch the public GitHub repo's `.gitignore` at all). Before doing this,
   sanity-check Kaggle's competition rules page: the standard reading is that redistribution
   restrictions apply to the *raw dataset*, not to derived model weights trained on it, but confirm
   this against the current rules text for this specific competition before publishing (see Open
   Decision #3).
3. Push the dashboard container (standalone mode) to the Space.
4. Once live, update `README.md`'s "Live demo" line with the real URL, and set the GitHub repo's
   homepage: `gh repo edit --homepage "<hf-space-url>"`.

**Definition of Done:**
- The HF Space builds successfully and serves the dashboard at a public URL.
- `README.md` no longer says "not yet deployed."

---

### P0.15 — Documentation & polish

**Goal:** the artifact a recruiter actually reads/skims — make it count.

**Files:** `README.md` (finalize), `docs/architecture.md`, `docs/model-evaluation.md`

**Steps:**
1. Finalize `README.md`: remove every "to be filled in" placeholder; add the final backtest
   leaderboard as a small table; add the live demo link; add a "Quick start" section with the exact
   `uv`/`docker compose` commands that actually work at this point.
2. Write `docs/architecture.md` with a Mermaid diagram of the system (data → features → 5 model tiers
   → evaluation/MLflow → serving API + dashboard → Docker/CI/HF Spaces) and a short paragraph per
   component.
3. Write `docs/model-evaluation.md` — this is the direct answer to the original brief's "assess model
   quality, explain in business terms, explain where it reaches its limits" ask:
   - Explain WAPE, pinball loss, and interval coverage in plain business language (what a WAPE of
     e.g. 22% actually means for someone planning inventory).
   - Report the final backtest leaderboard and which model was selected as champion, and why (per the
     data-driven rule from P0.8 — report the real result, whatever it turned out to be).
   - **"Where this model breaks down"** — a dedicated section covering, concretely: cold-start
     store/family combinations with little history; the 2016-04-16 earthquake as a real regime-shift
     example (from P0.3) where any model trained only on "normal" patterns would have badly
     underestimated demand; holiday-adjacent volatility; deep-discount promotion spikes; and
     structurally intermittent/near-always-zero series, where point-forecast metrics are almost
     meaningless and a different (e.g. classification-style "will this sell at all") framing would
     actually be more appropriate — naming this limitation explicitly is itself a demonstration of
     seniority.
4. `git tag v0.1.0` once the above is done, `git push --tags`.

**Definition of Done:**
- `README.md` has no remaining placeholder text.
- `docs/model-evaluation.md` contains all the elements listed above.
- `v0.1.0` tag exists on the remote (`git ls-remote --tags origin` shows it).

---

## 5. P1 — after P0 ships (optional, not required to call this project "done")

Not date-boxed the way a longer-running project would be — pick these up only if you want to keep
building after the P0 demo is live. Slightly less atomized than P0 since they're further out; refine
each into P0-style tasks (exact files/steps/DoD) when you actually pick it up.

- **P1.1 — Polars migration for the feature pipeline.** Swap `build_feature_frame`'s hot path
  (P0.4) to `polars`, benchmark old vs new on the full series set, report the speedup (or honest lack
  thereof) in `docs/architecture.md`.
- **P1.2 — Continuous drift monitoring.** Turn P0.11's one-shot script into a scheduled GitHub
  Action (weekly cron) that regenerates the drift report and opens/updates a tracking issue if drift
  exceeds a threshold.
- **P1.3 — Data/artifact versioning.** Add DVC (or lakeFS) for `data/processed/` and trained model
  artifacts, so a given MLflow run can be tied back to an exact data snapshot.
- **P1.4 — Hyperparameter tuning.** Add an Optuna tuning pass for the LightGBM (P0.6c) and NHITS
  (P0.6d) tiers, logged to MLflow as nested runs; compare tuned vs default in the leaderboard.
- **P1.5 — A second deep-learning contender.** Add PatchTST (or TFT) alongside NHITS, compare
  head-to-head — resolves Open Decision #1 with real evidence instead of the documented default.
- **P1.6 — Deeper test coverage.** Property-based tests (Hypothesis) for the feature engineering
  and metrics modules; expand `test_api.py` coverage.
- **P1.7 — Champion/challenger serving.** Extend the API (P0.9) to route a configurable % of
  `/forecast` requests to a challenger model, logging both predictions for later comparison — a real
  online-evaluation pattern, not just offline backtesting.
- **P1.8 — Write-up / video walkthrough.** A short (5–8 min) recorded walkthrough of the dashboard
  plus a written post. Once this exists, **flag it to Abhirup** — per `career/CLAUDE.md` convention
  9, a finished portfolio project like this should be considered for `career/site/` (a portfolio
  card, a CV line, a project page) and `career/profile/master-cv.md`, but only once there's a real,
  live result to point to, not before.

---

## 6. P2 — longer horizon (directional, not fully specified)

These are intentionally not broken into atomic tasks yet — refine them into P0-style tasks only when
their turn actually comes, informed by what P0/P1 revealed in practice. Don't over-specify a
speculative future phase today.

- **P2.1 — Prove real domain-agnosticism.** The strongest possible evidence for the "not domain
  specific" goal: re-run the *same* pipeline (only a new data-loading adapter should be needed) on a
  second, structurally different public dataset from a non-retail domain — e.g. an energy-load or
  web-traffic dataset. If the architecture from P0 genuinely doesn't need to change, that's the proof.
- **P2.2 — Streaming/incremental ingestion demo.** A simple simulated streaming ingestion (or real
  Kafka, if justified) feeding incremental retraining, to demonstrate awareness of the online-learning
  end of this problem space.
- **P2.3 — LLM forecast-explainer agent.** Given a forecast and its drivers (holiday proximity,
  promotion, recent trend), generate a natural-language explanation for a non-technical stakeholder —
  ties the forecasting work to the "AI engineering" edge of the skill set, not just classical DS.
- **P2.4 — Hosting cost-optimization writeup.** Compare always-on API hosting cost vs.
  serverless/cold-start tradeoffs at a hypothetical production scale, as a short cost-engineering note.

---

## 7. Task tracking

**Authoritative day-to-day:** GitHub Issues + a Projects (v2) board on this repo, one issue per P0/P1
task above (titled to match, e.g. `P0.6c — Gradient-boosted trees (global ML model)`), labeled `P0`
or `P1`, body linking back to the matching section of this file rather than duplicating its content.
P2 items get lightweight tracking issues too (labeled `P2`) but are not expected to be "ready" yet.

**If the board and this file ever disagree, this file wins** — update the board to match.

- **Definition of ready:** an issue is ready to start when everything needed is already in this
  file's corresponding section — no open question blocks starting it. If a task genuinely can't
  start without a decision only Abhirup can make, that decision belongs in section 8 (Open Decisions)
  below, not silently guessed at.
- **Definition of done:** the task's Definition of Done in this file is satisfied — a command
  succeeds, an artifact exists, a test passes. Close the GitHub issue when its DoD is met.

Routine commits/pushes during implementation don't need separate confirmation each time — creating
this repo as public already authorizes normal push activity on it. Destructive git operations
(force-push, history rewrite, deleting branches) still always need explicit confirmation, same as
anywhere else.

---

## 8. Open decisions log

Things intentionally left unresolved — don't guess at these, resolve them the way each entry says.

1. **NHITS vs PatchTST for the deep-learning tier.** Default is NHITS (P0.6d); switch only if P1.5's
   head-to-head shows PatchTST is clearly better, not on a hunch.
2. **Chronos-Bolt-Small vs -Base.** Start with Small (CPU-friendly, faster). Revisit Base only if
   P0.6e's observed runtime leaves headroom.
3. **Committing a trained model artifact into the HF Space repo for P0.14.** Recommended approach as
   written in P0.14 step 2 — but double-check Kaggle's current competition rules text for this
   specific competition before publishing anything derived-model-related publicly, since rules text
   can vary by competition and can change over time.
4. **`log1p` vs raw target transform for feature engineering (P0.4).** Decide empirically during
   P0.7 by comparing backtest WAPE with/without on the ML tier; do not fix this in advance.
5. **Business cost ratio (3:1 under:over, P0.7).** Illustrative default only. Since it's exposed as a
   live dashboard slider (P0.10), there's no need to "get it right" upfront — Abhirup or a viewer can
   adjust it interactively.
6. **Whether/when to add this project to `career/site/`.** Per `career/CLAUDE.md` convention 9 and
   P1.8 above: hold off until P0 actually ships with a real live demo and real backtest results, then
   ask Abhirup rather than doing it silently.
