"""Tier 1 — classical statistical models (PLAN.md P0.6b; theory in docs/methodology.md §1.4).

Local AutoETS and AutoARIMA (one model per series) via ``statsforecast``, weekly seasonality
(``season_length=7``), with conformal 80% intervals.

Scoping choices (measured on a 60-series sample of fold 1, see PLAN.md P0.6b Outcome):

- **AutoARIMA is fitted on each series' last 180 days** (``ARIMA_HISTORY_DAYS``). On full history it
  took ~2 h per fold; on 180 days ~7 min, for +0.3 WAPE points on the sample. AutoETS is cheap
  and keeps the full history.
- **Structurally all-zero series** (no sales in the training window) are forecast as zero instead
  of being fitted, and **leading zeros** before a series' first sale (not yet recorded, not real
  demand — docs/eda-findings.md §2) are trimmed before fitting.
- **Series too short** for conformal calibration (fewer than ``min_length`` observations after
  trimming) fall back to seasonal naive (tier 0) under this tier's model names, so every series
  gets a forecast; the count is logged.
- Forecasts and bounds are clipped at 0 (sales can't be negative).
"""

import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from statsforecast.utils import ConformalIntervals

from forecasting_platform.config import settings
from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.baseline import seasonal_naive

logger = logging.getLogger(__name__)

SEASON_LENGTH = 7
ARIMA_HISTORY_DAYS = 180
CONFORMAL_WINDOWS = 2


def _prepare(train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Trim leading zeros per series; return (trimmed frame, ids of all-zero series)."""
    df = train_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    started = df.groupby("unique_id", sort=False)["y"].transform(lambda s: s.ne(0).cummax())
    all_zero = sorted(set(df["unique_id"]) - set(df.loc[started, "unique_id"]))
    return df[started].reset_index(drop=True), all_zero


def _run(df: pd.DataFrame, model, name: str, horizon: int, level: int, n_jobs: int) -> pd.DataFrame:
    sf = StatsForecast(
        models=[model],
        freq="D",
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=SEASON_LENGTH),
    )
    fc = sf.forecast(
        h=horizon,
        df=df,
        level=[level],
        prediction_intervals=ConformalIntervals(n_windows=CONFORMAL_WINDOWS, h=horizon),
    )
    return pd.DataFrame(
        {
            "unique_id": fc["unique_id"],
            "ds": fc["ds"],
            "model_name": name,
            "yhat": fc[name],
            "yhat_lo80": fc[f"{name}-lo-{level}"],
            "yhat_hi80": fc[f"{name}-hi-{level}"],
        }
    )


def _zero_forecast(ids: list[str], last_date: pd.Series, horizon: int, name: str) -> pd.DataFrame:
    steps = pd.to_timedelta(np.arange(1, horizon + 1), unit="D")
    return pd.DataFrame(
        {
            "unique_id": np.repeat(ids, horizon),
            "ds": np.concatenate([last_date[i] + steps for i in ids]) if ids else [],
            "model_name": name,
            "yhat": 0.0,
            "yhat_lo80": 0.0,
            "yhat_hi80": 0.0,
        }
    )


def forecast_statistical(
    train_df: pd.DataFrame,
    horizon: int = settings.FORECAST_HORIZON,
    arima_history_days: int = ARIMA_HISTORY_DAYS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """AutoETS + AutoARIMA forecasts for every series in ``train_df`` (long format, contract cols).

    ``train_df`` needs ``unique_id``, ``ds``, ``y`` on a gap-free daily calendar. Returns
    ``2 * n_series * horizon`` rows (``model_name`` in {"AutoETS", "AutoARIMA"}).
    """
    level = 80
    last_date = train_df.groupby("unique_id")["ds"].max()
    trimmed, all_zero = _prepare(train_df)

    # Conformal calibration needs n_windows * horizon points plus enough to fit a seasonal model.
    min_length = CONFORMAL_WINDOWS * horizon + 2 * SEASON_LENGTH
    lengths = trimmed.groupby("unique_id").size()
    short_ids = lengths[lengths < min_length].index
    fit_df = trimmed[~trimmed["unique_id"].isin(short_ids)]
    logger.info(
        "statistical tier: %d series fitted, %d all-zero -> 0, %d too short -> seasonal naive",
        fit_df["unique_id"].nunique(),
        len(all_zero),
        len(short_ids),
    )

    frames = []
    specs = [
        ("AutoETS", AutoETS(season_length=SEASON_LENGTH), fit_df),
        (
            "AutoARIMA",
            AutoARIMA(season_length=SEASON_LENGTH),
            fit_df.groupby("unique_id", sort=False).tail(arima_history_days),
        ),
    ]
    for name, model, df in specs:
        if not df.empty:
            frames.append(_run(df, model, name, horizon, level, n_jobs))
        if len(short_ids):
            # Untrimmed history: a series that just started may have < 9 points after trimming.
            short = seasonal_naive(train_df[train_df["unique_id"].isin(short_ids)], horizon)
            frames.append(short.assign(model_name=name))
        frames.append(_zero_forecast(all_zero, last_date, horizon, name))

    out = pd.concat(frames, ignore_index=True)
    for col in ("yhat", "yhat_lo80", "yhat_hi80"):
        out[col] = out[col].clip(lower=0).astype(float)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["model_name", "unique_id", "ds"], ignore_index=True)[FORECAST_COLUMNS]
