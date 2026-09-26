"""Tests for the global LightGBM tier (synthetic data only)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.evaluation.backtest import generate_cv_folds, split_train_test
from forecasting_platform.features.engineering import build_feature_frame
from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.ml import forecast_ml

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
H = 14


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(FIXTURE)
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "city": ["Quito", "Quito", "Manta"],
            "state": ["Pichincha", "Pichincha", "Manabi"],
            "type": ["A", "D", "C"],
            "cluster": [13, 8, 3],
        }
    )
    dates = pd.date_range("2017-01-01", "2017-04-30", freq="D")
    oil = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "dcoilwtico": np.linspace(50, 60, 120)})
    holidays = pd.DataFrame(
        [("2017-04-24", "Holiday", "National", "Ecuador", False, "x")],
        columns=["date", "type", "locale", "locale_name", "transferred", "description"],
    )
    features = build_feature_frame(sales, stores, oil, holidays)
    fold = generate_cv_folds(features["ds"], horizon=H, n_folds=1)[0]
    return split_train_test(features, fold)


@pytest.fixture(scope="module")
def run(split, tmp_path_factory) -> tuple[pd.DataFrame, Path]:
    train, test = split
    path = tmp_path_factory.mktemp("imp") / "importance.csv"
    return forecast_ml(train, test, horizon=H, importance_path=path), path


@pytest.fixture(scope="module")
def forecast(run) -> pd.DataFrame:
    return run[0]


def test_output_contract(forecast: pd.DataFrame, split) -> None:
    train, test = split
    assert list(forecast.columns) == FORECAST_COLUMNS
    assert set(forecast["model_name"]) == {"LightGBM"}
    assert len(forecast) == 9 * H
    assert not forecast.isna().any().any()
    merged = test.merge(forecast, on=["unique_id", "ds"], how="inner")
    assert len(merged) == 9 * H  # forecast dates are exactly the test window


def test_intervals_ordered_and_non_negative(forecast: pd.DataFrame) -> None:
    assert (forecast[["yhat", "yhat_lo80", "yhat_hi80"]] >= 0).all().all()
    assert (forecast["yhat_lo80"] <= forecast["yhat"] + 1e-9).all()
    assert (forecast["yhat"] <= forecast["yhat_hi80"] + 1e-9).all()


def test_test_y_is_never_used(split, forecast: pd.DataFrame) -> None:
    train, test = split
    blinded = forecast_ml(train, test.assign(y=np.nan), horizon=H)
    pd.testing.assert_frame_equal(blinded, forecast)


def test_future_oil_is_not_used(split, forecast: pd.DataFrame) -> None:
    """Oil isn't known in advance: changing the realised future price must not change forecasts."""
    train, test = split
    shocked = forecast_ml(train, test.assign(oil_price=test["oil_price"] * 3), horizon=H)
    pd.testing.assert_frame_equal(shocked, forecast)


def test_feature_importance_written(run) -> None:
    imp = pd.read_csv(run[1])
    assert list(imp.columns) == ["feature", "gain", "split", "gain_share"]
    for feature in ["lag7", "dayofweek", "onpromotion", "family"]:
        assert feature in set(imp["feature"])
    assert imp["gain_share"].sum() == pytest.approx(1.0)


def test_log_target_variant(split) -> None:
    train, test = split
    fc = forecast_ml(train, test, horizon=H, log_target=True)
    assert set(fc["model_name"]) == {"LightGBM_log1p"}
    assert len(fc) == 9 * H and not fc.isna().any().any()
