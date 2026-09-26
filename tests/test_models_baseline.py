"""Tests for the naive baselines (synthetic data only)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.baseline import Z_80, naive, seasonal_naive

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"


@pytest.fixture
def train() -> pd.DataFrame:
    """Two series: a pure weekly pattern (zero seasonal residuals) and a linear ramp."""
    ds = pd.date_range("2017-01-01", periods=35, freq="D")
    weekly = np.tile([10.0, 20, 30, 40, 50, 60, 70], 5)
    ramp = np.arange(35, dtype=float)
    return pd.concat(
        [
            pd.DataFrame({"unique_id": "weekly", "ds": ds, "y": weekly}),
            pd.DataFrame({"unique_id": "ramp", "ds": ds[:30], "y": ramp[:30]}),  # ends earlier
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=0)  # shuffled: functions must not rely on input order


@pytest.mark.parametrize("model", [naive, seasonal_naive])
def test_output_contract(model, train: pd.DataFrame) -> None:
    fc = model(train, horizon=28)
    assert list(fc.columns) == FORECAST_COLUMNS
    assert len(fc) == 2 * 28
    assert not fc.isna().any().any()
    assert (fc["yhat_lo80"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["yhat_hi80"]).all()
    for uid, last in [("weekly", "2017-02-04"), ("ramp", "2017-01-30")]:
        dates = fc.loc[fc["unique_id"] == uid, "ds"]
        expected = pd.date_range(pd.Timestamp(last) + pd.Timedelta(days=1), periods=28)
        assert list(dates) == list(expected)


def test_seasonal_naive_repeats_last_week(train: pd.DataFrame) -> None:
    fc = seasonal_naive(train, horizon=28)
    weekly = fc[fc["unique_id"] == "weekly"]
    # Last training day 2017-02-04 has y=70, so the next week starts at 10 again.
    np.testing.assert_array_equal(weekly["yhat"], np.tile([10.0, 20, 30, 40, 50, 60, 70], 4))
    # A perfect weekly pattern has zero seasonal residuals -> zero-width band.
    np.testing.assert_array_equal(weekly["yhat_hi80"], weekly["yhat"])
    ramp = fc[fc["unique_id"] == "ramp"]
    np.testing.assert_array_equal(ramp["yhat"].to_numpy()[:7], np.arange(23, 30, dtype=float))


def test_naive_is_flat_with_residual_std_band(train: pd.DataFrame) -> None:
    fc = naive(train, horizon=10)
    ramp = fc[fc["unique_id"] == "ramp"]
    assert (ramp["yhat"] == 29.0).all()
    # Ramp one-step residuals are all 1 -> std 0 -> zero-width band.
    assert (ramp["yhat_hi80"] == ramp["yhat"]).all()
    weekly = fc[fc["unique_id"] == "weekly"]
    y = train.loc[train["unique_id"] == "weekly"].sort_values("ds")["y"]
    expected_half_width = Z_80 * y.diff().std()
    assert weekly["yhat_hi80"].iloc[0] - 70.0 == pytest.approx(expected_half_width)


def test_lower_bound_clipped_at_zero() -> None:
    ds = pd.date_range("2017-01-01", periods=21, freq="D")
    spiky = pd.DataFrame({"unique_id": "s", "ds": ds, "y": np.tile([0.0, 0, 0, 0, 0, 0, 100], 3)})
    fc = seasonal_naive(spiky, horizon=7)
    assert (fc["yhat_lo80"] >= 0).all()


def test_too_short_series_raises() -> None:
    short = pd.DataFrame({"unique_id": "s", "ds": pd.date_range("2017-01-01", periods=5), "y": 1.0})
    with pytest.raises(ValueError, match="fewer than 9"):
        seasonal_naive(short, horizon=7)


def test_runs_on_fixture() -> None:
    raw = pd.read_csv(FIXTURE, parse_dates=["date"])
    df = raw.assign(unique_id=raw["store_nbr"].astype(str) + "_" + raw["family"])
    df = df.rename(columns={"date": "ds", "sales": "y"})
    for model in (naive, seasonal_naive):
        fc = model(df, horizon=28)
        assert fc["unique_id"].nunique() == 9 and len(fc) == 9 * 28
