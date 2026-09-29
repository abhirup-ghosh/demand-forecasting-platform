"""Tests for champion selection and the registered pyfunc's serving logic (no torch needed)."""

import pandas as pd
import pytest

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.champion import ChampionForecaster, select_champion


def _board(rows):
    return pd.DataFrame(
        [
            {
                "model": m,
                "fold": f,
                "wape": w,
                "interval_coverage": 0.8,
                "pinball_loss": 1.0,
                "business_cost": 1.0,
            }
            for m, f, w in rows
        ]
    )


def test_lowest_mean_wape_wins_and_baselines_are_excluded() -> None:
    board = _board(
        [
            ("SeasonalNaive", 1, 0.01),
            ("SeasonalNaive", 2, 0.01),  # best, but a baseline
            ("A", 1, 0.10),
            ("A", 2, 0.20),  # mean 0.15
            ("B", 1, 0.14),
            ("B", 2, 0.14),
        ]  # mean 0.14 -> champion
    )
    champion, table = select_champion(board)
    assert champion == "B"
    assert list(table.index) == ["B", "A"]


def test_models_missing_a_fold_do_not_qualify() -> None:
    board = _board([("A", 1, 0.20), ("A", 2, 0.20), ("B", 1, 0.01)])
    assert select_champion(board)[0] == "A"


def test_no_candidates_raises() -> None:
    with pytest.raises(ValueError, match="no non-baseline"):
        select_champion(_board([("Naive", 1, 0.3)]))


@pytest.fixture
def model() -> ChampionForecaster:
    m = ChampionForecaster()
    ds = pd.date_range("2017-08-16", periods=28)
    m.forecasts = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "model_name": "NHITS",
                    "yhat": 1.0,
                    "yhat_lo80": 0.5,
                    "yhat_hi80": 1.5,
                }
            )
            for uid in ["1_A", "2_B"]
        ],
        ignore_index=True,
    )
    return m


def test_predict_filters_series_and_horizon(model: ChampionForecaster) -> None:
    out = model.predict(None, pd.DataFrame({"unique_id": ["2_B"], "horizon": [7]}))
    assert list(out.columns) == FORECAST_COLUMNS
    assert set(out["unique_id"]) == {"2_B"} and len(out) == 7
    assert out["ds"].min() == pd.Timestamp("2017-08-16")
    full = model.predict(None, pd.DataFrame({"unique_id": ["1_A", "2_B"]}))
    assert len(full) == 2 * 28  # default horizon


def test_predict_unknown_series_returns_empty(model: ChampionForecaster) -> None:
    assert model.predict(None, pd.DataFrame({"unique_id": ["99_X"]})).empty


def test_predict_rejects_horizon_above_28(model: ChampionForecaster) -> None:
    with pytest.raises(ValueError, match="horizon"):
        model.predict(None, pd.DataFrame({"unique_id": ["1_A"], "horizon": [29]}))
