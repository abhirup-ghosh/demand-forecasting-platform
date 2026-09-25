"""Central configuration for the forecasting platform.

Every module imports the single ``settings`` instance below instead of reading environment
variables ad hoc. Values can be overridden via environment variables or a ``.env`` file at the
repository root (see ``.env.example``).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds non-settings secrets (e.g. KAGGLE_*)
    )

    # --- Data locations ---
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
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
    CHRONOS_MODEL_ID: str = "amazon/chronos-bolt-small"
    CHRONOS_SERIES_SAMPLE_SIZE: int = 60

    # --- Business cost model (PLAN.md P0.7) — illustrative 3:1 under:over default ---
    COST_UNDER_PER_UNIT: float = 3.0
    COST_OVER_PER_UNIT: float = 1.0


settings = Settings()
