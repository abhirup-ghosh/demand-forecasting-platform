"""Central configuration for the forecasting platform.

Every module imports the single ``settings`` instance below instead of reading environment
variables ad hoc. Values can be overridden via environment variables or a ``.env`` file at the
repository root (see ``.env.example``).
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# In a normal editable/dev checkout this file lives at <repo_root>/src/forecasting_platform/
# config.py, so parents[2] is the repo root. A *non-editable* install — which is what every
# Dockerfile here uses (`uv sync --no-editable`, PLAN.md P0.12) — puts this file under
# .venv/lib/python3.X/site-packages/forecasting_platform/ instead, where that heuristic silently
# resolves to .venv/lib/python3.X: every PROJECT_ROOT-derived default below would then point at a
# nonexistent path inside the venv. The Dockerfiles set FORECASTING_PLATFORM_ROOT=/app to sidestep
# this; local/dev runs (and anything else that doesn't set it) keep using the parents[2] heuristic.
PROJECT_ROOT = Path(
    os.environ.get("FORECASTING_PLATFORM_ROOT") or Path(__file__).resolve().parents[2]
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds non-settings secrets (e.g. KAGGLE_*)
    )

    # --- Data locations ---
    # Raw Kaggle download lives under eda/data/ (PLAN.md 3.1) — gitignored, never committed.
    DATA_RAW_DIR: Path = PROJECT_ROOT / "eda" / "data" / "store-sales-time-series-forecasting"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

    # --- Forecast task definition (PLAN.md section 3.2) ---
    FORECAST_HORIZON: int = 28
    QUANTILES: list[float] = [0.1, 0.5, 0.9]

    # --- Backtesting (PLAN.md P0.5) ---
    BACKTEST_N_FOLDS: int = 5
    BACKTEST_STEP_DAYS: int = 28

    RANDOM_SEED: int = 42

    # --- Experiment tracking ---
    MLFLOW_TRACKING_URI: str = "sqlite:///mlruns.db"

    # --- Zero-shot foundation model tier (PLAN.md P0.6e) ---
    # Base, not Small: Open Decision #2 resolved 2026-09-28 (ample runtime headroom).
    CHRONOS_MODEL_ID: str = "amazon/chronos-bolt-base"
    # Only used by select_chronos_sample(); the tier itself now runs on all series.
    CHRONOS_SERIES_SAMPLE_SIZE: int = 60

    # --- Business cost model (PLAN.md P0.7) — illustrative 3:1 under:over default ---
    COST_UNDER_PER_UNIT: float = 3.0
    COST_OVER_PER_UNIT: float = 1.0

    # --- Serving (PLAN.md P0.9) ---
    CHAMPION_MODEL_URI: str = "models:/demand-forecast-champion/Production"
    LEADERBOARD_PATH: Path = PROJECT_ROOT / "results" / "leaderboard.csv"
    BACKTEST_FORECAST_DIR: Path = PROJECT_ROOT / "results" / "forecasts"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"

    # --- Drift monitoring (PLAN.md P0.11) ---
    DRIFT_SUMMARY_PATH: Path = PROJECT_ROOT / "reports" / "generated" / "drift_summary.json"
    DRIFT_REPORT_PATH: Path = PROJECT_ROOT / "reports" / "generated" / "drift_report.html"

    # --- Dashboard (PLAN.md P0.10) ---
    # True: load the champion in-process (Hugging Face Spaces, one container). False: call the
    # FastAPI service at API_URL (docker-compose microservice mode, P0.12).
    DASHBOARD_STANDALONE_MODE: bool = True
    API_URL: str = "http://localhost:8000"


settings = Settings()
