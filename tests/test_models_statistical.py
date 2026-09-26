"""Tests for the statistical tier (synthetic data only; small horizon, single process for speed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.statistical import forecast_statistical

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
H = 14


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, parse_dates=["date"])
    df = raw.assign(unique_id=raw["store_nbr"].astype(str) + "_" + raw["family"])
    df = df.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]
    ds = df["ds"].drop_duplicates().sort_values()
    extra = [
        # Structurally all-zero series.
        pd.DataFrame({"unique_id": "zero", "ds": ds, "y": 0.0}),
        # Starts selling late: 80 leading zeros, then 40 days of sales -> too short -> fallback.
        pd.DataFrame({"unique_id": "late", "ds": ds, "y": np.r_[np.zeros(80), np.full(40, 5.0)]}),
    ]
    return pd.concat([df, *extra], ignore_index=True)


@pytest.fixture(scope="module")
def forecast(train: pd.DataFrame) -> pd.DataFrame:
    return forecast_statistical(train, horizon=H, n_jobs=1)


def test_output_contract(forecast: pd.DataFrame, train: pd.DataFrame) -> None:
    assert list(forecast.columns) == FORECAST_COLUMNS
    assert set(forecast["model_name"]) == {"AutoETS", "AutoARIMA"}
    n_series = train["unique_id"].nunique()
    assert len(forecast) == 2 * n_series * H
    assert not forecast.isna().any().any()
    assert not forecast.duplicated(["model_name", "unique_id", "ds"]).any()


def test_forecast_dates_follow_training_end(forecast: pd.DataFrame) -> None:
    expected = pd.date_range("2017-05-01", periods=H, freq="D")
    for _, g in forecast.groupby(["model_name", "unique_id"]):
        assert list(g["ds"]) == list(expected)


def test_intervals_ordered_and_non_negative(forecast: pd.DataFrame) -> None:
    assert (forecast[["yhat", "yhat_lo80", "yhat_hi80"]] >= 0).all().all()
    assert (forecast["yhat_lo80"] <= forecast["yhat"] + 1e-9).all()
    assert (forecast["yhat"] <= forecast["yhat_hi80"] + 1e-9).all()


def test_all_zero_series_forecast_as_zero(forecast: pd.DataFrame) -> None:
    zero = forecast[forecast["unique_id"] == "zero"]
    assert len(zero) == 2 * H
    assert (zero[["yhat", "yhat_lo80", "yhat_hi80"]] == 0).all().all()


def test_short_series_falls_back_to_seasonal_naive(forecast: pd.DataFrame) -> None:
    late = forecast[forecast["unique_id"] == "late"]
    assert len(late) == 2 * H
    assert (late["yhat"] == 5.0).all()  # last week of a constant series, repeated


def test_forecasts_track_series_level(forecast: pd.DataFrame, train: pd.DataFrame) -> None:
    """Sanity: fitted models forecast near each series' recent level, not arbitrary values."""
    recent = train[train["ds"] > "2017-04-02"].groupby("unique_id")["y"].mean()
    fitted = forecast[~forecast["unique_id"].isin(["zero", "late"])]
    mean_fc = fitted.groupby(["model_name", "unique_id"])["yhat"].mean()
    for (_, uid), value in mean_fc.items():
        assert value == pytest.approx(recent[uid], rel=0.35)
