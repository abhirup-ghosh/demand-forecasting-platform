# Demand Forecasting Platform

An end-to-end, production-shaped **multi-horizon demand forecasting platform**: data → features →
a full spectrum of forecasting approaches (naive baseline → classical statistical → gradient-boosted
trees → global deep learning → zero-shot foundation model) → rigorous backtesting with uncertainty
quantification → a served API and an interactive dashboard → containerized, CI-tested, and deployed
to a public live demo.

The concrete demo dataset is daily store-level retail sales (Kaggle's public
["Store Sales - Time Series Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
dataset), but every design choice here targets the general problem class — **multi-series demand /
capacity forecasting under uncertainty** — the same shape of problem as staffing, energy load, web
traffic, or supply-chain planning. Retail is the substrate; the techniques are the point.

**Status:** planning complete, implementation in progress. See [`PLAN.md`](./PLAN.md) for the full
scope, architecture decisions, and task-by-task build plan (source of truth — if this README or the
GitHub Project board ever disagrees with `PLAN.md`, `PLAN.md` wins).

**Live demo:** _not yet deployed — link goes here once P0.14 ships (see PLAN.md)._

## Why this exists

Originally scoped as a take-home challenge for a Senior Data Scientist role (application later
withdrawn for unrelated reasons). Kept going as a standalone portfolio project because time-series
forecasting is a skill area worth demonstrating properly: the interesting part isn't fitting one
model, it's the judgment calls — which model family for which regime, how to backtest honestly, how
to quantify and communicate uncertainty, and how to translate a forecast error metric into a
business cost.

## Architecture at a glance

See [`docs/architecture.md`](./docs/architecture.md) for the full diagram and component breakdown
once it exists. Summary (full rationale in `PLAN.md`'s Architecture Decisions table):

- **Models:** seasonal-naive baseline → `statsforecast` (AutoARIMA/AutoETS) → `mlforecast`
  (LightGBM, global model with lag/calendar features) → `neuralforecast` (global deep learning) →
  `Chronos` (zero-shot pretrained foundation model), compared honestly rather than picking a winner
  in advance.
- **Evaluation:** rolling-origin backtesting, WAPE/pinball-loss, conformal prediction intervals, and
  an asymmetric business-cost translation (over-forecast vs. under-forecast cost).
- **Serving:** FastAPI inference service + MLflow experiment tracking/model registry.
- **Interface:** Streamlit dashboard (EDA insights, forecast explorer, model leaderboard, business
  impact view).
- **Ops:** Docker/docker-compose, GitHub Actions CI, a lightweight Evidently drift report, deployed
  to Hugging Face Spaces.

## Quick start

_To be filled in once P0.1 (environment/tooling) ships — see `PLAN.md`._

## Repository layout

See `PLAN.md`'s architecture section for the full intended layout; directories are created as the
corresponding P0 task lands, not all up front.

## License

Code: MIT, see [`LICENSE`](./LICENSE). Dataset is Kaggle-governed and is not redistributed in this
repository — see `data/README.md`.
