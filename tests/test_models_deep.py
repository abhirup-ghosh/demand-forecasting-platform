"""Tests for the NHITS tier (synthetic data only; tiny training budget for speed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.deep import forecast_deep

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
H = 14


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, parse_dates=["date"])
    df = raw.assign(unique_id=raw["store_nbr"].astype(str) + "_" + raw["family"])
    df = df.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]
    ds = df["ds"].drop_duplicates().sort_values()
    zero = pd.DataFrame({"unique_id": "zero", "ds": ds, "y": 0.0})
    return pd.concat([df, zero], ignore_index=True)


@pytest.fixture(scope="module")
def forecast(train: pd.DataFrame) -> pd.DataFrame:
    return forecast_deep(train, horizon=H, max_steps=20)


def test_output_contract(forecast: pd.DataFrame) -> None:
    assert list(forecast.columns) == FORECAST_COLUMNS
    assert set(forecast["model_name"]) == {"NHITS"}
    assert len(forecast) == 10 * H
    assert not forecast.isna().any().any()
    expected = pd.date_range("2017-05-01", periods=H, freq="D")
    for _, g in forecast.groupby("unique_id"):
        assert list(g["ds"]) == list(expected)


def test_intervals_ordered_and_non_negative(forecast: pd.DataFrame) -> None:
    assert (forecast[["yhat", "yhat_lo80", "yhat_hi80"]] >= 0).all().all()
    assert (forecast["yhat_lo80"] <= forecast["yhat"] + 1e-6).all()
    assert (forecast["yhat"] <= forecast["yhat_hi80"] + 1e-6).all()


def test_all_zero_series_forecast_as_zero(forecast: pd.DataFrame) -> None:
    zero = forecast[forecast["unique_id"] == "zero"]
    assert (zero[["yhat", "yhat_lo80", "yhat_hi80"]] == 0).all().all()


def test_forecasts_on_series_scale(forecast: pd.DataFrame, train: pd.DataFrame) -> None:
    """Robust scaling: a globally trained net still forecasts each series near its own level."""
    recent = train[train["ds"] > "2017-04-02"].groupby("unique_id")["y"].mean()
    fitted = forecast[forecast["unique_id"] != "zero"].groupby("unique_id")["yhat"].mean()
    ratio = fitted / recent.loc[fitted.index]
    assert np.all((ratio > 0.5) & (ratio < 1.5))


def test_torch_never_loaded_in_calling_process(forecast: pd.DataFrame) -> None:
    """Regression: PyTorch and LightGBM OpenMP runtimes crash/deadlock in one process on macOS,
    so NHITS must train in a spawned child and never import torch here."""
    import sys

    assert "torch" not in sys.modules
