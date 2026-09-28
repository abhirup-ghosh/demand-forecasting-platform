"""Tests for evaluation metrics and the business-cost model."""

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.evaluation.business_cost import business_cost
from forecasting_platform.evaluation.metrics import (
    interval_coverage,
    mape,
    pinball_loss,
    rmse,
    score_forecast,
    wape,
)


def test_wape_is_total_abs_error_over_total_volume() -> None:
    assert wape([10, 0, 30], [12, 1, 27]) == pytest.approx((2 + 1 + 3) / 40)
    assert np.isnan(wape([0, 0], [1, 1]))


def test_mape_skips_zero_actuals() -> None:
    assert mape([10, 0, 20], [11, 5, 18]) == pytest.approx((0.1 + 0.1) / 2)
    assert np.isnan(mape([0, 0], [1, 2]))


def test_rmse() -> None:
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_pinball_loss_known_values() -> None:
    # y=10; quantiles 0.1/0.5/0.9 predicted at 8/10/12:
    # q=0.1, diff=2 -> 0.2; q=0.5, diff=0 -> 0; q=0.9, diff=-2 -> (0.9-1)*-2 = 0.2
    assert pinball_loss([10.0], np.array([[8.0, 10.0, 12.0]])) == pytest.approx(0.4 / 3)
    # A perfect, zero-width forecast has zero loss.
    assert pinball_loss([5.0], np.array([[5.0, 5.0, 5.0]])) == 0.0


def test_interval_coverage_inclusive_bounds() -> None:
    assert interval_coverage([1, 5, 10], [1, 6, 0], [2, 7, 10]) == pytest.approx(2 / 3)


def test_business_cost_is_asymmetric() -> None:
    assert business_cost([10], [8], cost_under=3, cost_over=1) == 6.0  # under by 2
    assert business_cost([10], [12], cost_under=3, cost_over=1) == 2.0  # over by 2
    assert business_cost([10, 10], [8, 12]) == 8.0  # settings default 3:1


def test_score_forecast_excludes_all_zero_series_from_coverage() -> None:
    ds = pd.date_range("2017-01-01", periods=2)
    actuals = pd.DataFrame(
        {"unique_id": ["a", "a", "z", "z"], "ds": [*ds, *ds], "y": [10, 20, 0, 0]}
    )
    fc = actuals.drop(columns="y").assign(
        model_name="m",
        yhat=[10.0, 10.0, 0.0, 0.0],
        yhat_lo80=[9.0, 9.0, 0, 0],
        yhat_hi80=[11.0, 11.0, 0, 0],
    )
    s = score_forecast(actuals, fc, active_ids={"a"}, volume_band=pd.Series({"a": "top"}))
    assert s["interval_coverage"] == 0.5  # only series "a": 1 of 2 inside
    assert s["wape"] == pytest.approx(10 / 30)
    assert s["wape_top"] == pytest.approx(10 / 30)
    assert s["business_cost"] == pytest.approx(30.0)  # under by 10 at 3/unit


def test_score_forecast_rejects_missing_forecasts() -> None:
    actuals = pd.DataFrame({"unique_id": ["a"], "ds": [pd.Timestamp("2017-01-01")], "y": [1.0]})
    empty = pd.DataFrame(
        columns=["unique_id", "ds", "model_name", "yhat", "yhat_lo80", "yhat_hi80"]
    )
    empty = empty.astype({"ds": "datetime64[ns]", "yhat": float})
    with pytest.raises(ValueError, match="missing"):
        score_forecast(actuals, empty, active_ids={"a"})
