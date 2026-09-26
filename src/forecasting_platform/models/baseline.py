"""Tier 0 — naive baselines (PLAN.md P0.6a; theory in docs/methodology.md §1.3).

These are the floor every other tier must beat, not candidates for champion.

Interval caveat (documented simplification): the 80% band is a **constant** width,
``z_0.9 * std(in-sample residuals)``, where residuals are the baseline's own in-sample errors
(``y_t - y_{t-1}`` for naive, ``y_t - y_{t-m}`` for seasonal naive). A correct random-walk interval
widens with the horizon (~sqrt(h)), so these bands are too narrow at longer horizons; P0.7's
coverage metric shows by how much. The lower bound is clipped at 0 since sales can't be negative.

Input: a long-format ``train_df`` with ``unique_id``, ``ds``, ``y`` on a gap-free daily calendar
(as produced by ``features.engineering.build_feature_frame``).
"""

from statistics import NormalDist

import numpy as np
import pandas as pd

from forecasting_platform.models import FORECAST_COLUMNS

Z_80 = NormalDist().inv_cdf(0.9)  # two-sided 80% interval


def _forecast_frame(
    train_df: pd.DataFrame, horizon: int, lag: int, model_name: str
) -> pd.DataFrame:
    """Repeat each series' last ``lag`` values cyclically over ``horizon`` days."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    df = train_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    lengths = df.groupby("unique_id", sort=False).size()
    too_short = lengths[lengths < lag + 2]
    if not too_short.empty:
        raise ValueError(
            f"{len(too_short)} series have fewer than {lag + 2} observations "
            f"(needed for {model_name}), e.g. {too_short.index[0]!r}"
        )

    grouped = df.groupby("unique_id", sort=False)
    residual_std = (df["y"] - grouped["y"].shift(lag)).groupby(df["unique_id"], sort=False).std()
    last_block = grouped.tail(lag)  # the last `lag` observations of each series, in date order
    last_date = grouped["ds"].max()

    steps = np.arange(1, horizon + 1)
    uids = lengths.index.to_numpy()
    block = last_block["y"].to_numpy().reshape(len(uids), lag)
    yhat = block[:, (steps - 1) % lag]  # y_{T+h} = y_{T+h-lag(k+1)}, k = floor((h-1)/lag)

    out = pd.DataFrame(
        {
            "unique_id": np.repeat(uids, horizon),
            "ds": (
                np.repeat(last_date.loc[uids].to_numpy(), horizon)
                + np.tile(pd.to_timedelta(steps, unit="D").to_numpy(), len(uids))
            ),
            "model_name": model_name,
            "yhat": yhat.ravel(),
        }
    )
    half_width = np.repeat(Z_80 * residual_std.loc[uids].to_numpy(), horizon)
    out["yhat_lo80"] = np.clip(out["yhat"] - half_width, 0, None)
    out["yhat_hi80"] = out["yhat"] + half_width
    return out[FORECAST_COLUMNS]


def naive(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast = last observed value, flat over the horizon."""
    return _forecast_frame(train_df, horizon, lag=1, model_name="Naive")


def seasonal_naive(train_df: pd.DataFrame, horizon: int, season_length: int = 7) -> pd.DataFrame:
    """Forecast = the last observed season (default: last week), repeated over the horizon."""
    return _forecast_frame(train_df, horizon, lag=season_length, model_name="SeasonalNaive")
