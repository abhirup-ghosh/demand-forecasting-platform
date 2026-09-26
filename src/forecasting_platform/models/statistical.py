"""Tier 1 — classical statistical models (PLAN.md P0.6b; theory in docs/methodology.md §1.4).

Local AutoETS and AutoARIMA (one model per series) via ``statsforecast``, weekly seasonality
(``season_length=7``), with conformal 80% intervals.

Scoping choices (measured on a 60-series sample of fold 1, see PLAN.md P0.6b Outcome):

- **AutoARIMA is fitted on each series' last 180 days** (``ARIMA_HISTORY_DAYS``). On full history it
  took ~2 h per fold; on 180 days ~7 min, for +0.3 WAPE points on the sample. AutoETS is cheap
  and keeps the full history.
- All-zero series, leading zeros, too-short series and clipping are handled by
  ``models.common`` (shared with the other fitted tiers); "too short" here means fewer points than
  conformal calibration needs.
"""

import logging

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from statsforecast.utils import ConformalIntervals

from forecasting_platform.config import settings
from forecasting_platform.models.common import fallback_forecasts, finalize, split_series

logger = logging.getLogger(__name__)

SEASON_LENGTH = 7
ARIMA_HISTORY_DAYS = 180
CONFORMAL_WINDOWS = 2


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
    # Conformal calibration needs n_windows * horizon points plus enough to fit a seasonal model.
    split = split_series(train_df[["unique_id", "ds", "y"]], CONFORMAL_WINDOWS * horizon + 14)
    logger.info("statistical tier: %s", split.summary())

    frames = []
    specs = [
        ("AutoETS", AutoETS(season_length=SEASON_LENGTH), split.fit_df),
        (
            "AutoARIMA",
            AutoARIMA(season_length=SEASON_LENGTH),
            split.fit_df.groupby("unique_id", sort=False).tail(arima_history_days),
        ),
    ]
    for name, model, df in specs:
        if not df.empty:
            frames.append(_run(df, model, name, horizon, level, n_jobs))
        frames.extend(fallback_forecasts(train_df, split, horizon, name))
    return finalize(frames)
