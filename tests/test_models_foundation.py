"""Tests for the Chronos-Bolt tier.

The pure-numpy parts are always tested. The real-model test loads ``amazon/chronos-bolt-small``
(the smallest variant is enough to exercise the pipeline) in an isolated process and is skipped
when that model isn't in the local Hugging Face cache (e.g. in CI).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.foundation import (
    MODEL_NAME,
    forecast_foundation,
    model_name_for,
    prepare_contexts,
    quantiles_to_contract,
    select_chronos_sample,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
H = 14
MODEL_CACHED = any(
    (Path.home() / ".cache" / "huggingface" / "hub").glob("models--amazon--chronos-bolt-small")
)


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, parse_dates=["date"])
    df = raw.assign(unique_id=raw["store_nbr"].astype(str) + "_" + raw["family"])
    return df.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]


def test_sample_is_stratified_by_volume() -> None:
    ds = pd.date_range("2017-01-01", periods=10)
    frames = [pd.DataFrame({"unique_id": f"s{i:03d}", "ds": ds, "y": float(i)}) for i in range(100)]
    df = pd.concat(frames, ignore_index=True)  # s000 is all-zero, volume grows with i
    sample = select_chronos_sample(df, n=60)
    assert len(sample) == 60 and "s000" not in sample
    ranks = sorted(int(s[1:]) for s in sample)
    assert ranks[:20] == list(range(1, 21))  # bottom 20 (all-zero excluded)
    assert ranks[-20:] == list(range(80, 100))  # top 20
    assert all(40 <= r <= 60 for r in ranks[20:40])  # middle 20, around the median


def test_contexts_trim_leading_zeros_and_flag_all_zero() -> None:
    ds = pd.date_range("2017-01-01", periods=6)
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": "a", "ds": ds, "y": [0, 0, 1, 0, 2, 3.0]}),
            pd.DataFrame({"unique_id": "z", "ds": ds, "y": 0.0}),
        ]
    )
    ids, contexts, all_zero, last_date = prepare_contexts(df)
    assert ids == ["a"] and all_zero == ["z"]
    np.testing.assert_array_equal(contexts[0], [1, 0, 2, 3])  # interior zero kept
    assert last_date["z"] == ds[-1]


def test_quantiles_map_to_contract() -> None:
    last = pd.Series({"a": pd.Timestamp("2017-04-30"), "z": pd.Timestamp("2017-04-30")})
    q = np.stack([np.full((H,), 1.0), np.full((H,), 2.0), np.full((H,), 4.0)], axis=-1)[None]
    fc = quantiles_to_contract(["a"], q, ["z"], last, H)
    assert list(fc.columns) == FORECAST_COLUMNS and len(fc) == 2 * H
    a = fc[fc["unique_id"] == "a"]
    assert (a["yhat_lo80"] == 1).all() and (a["yhat"] == 2).all() and (a["yhat_hi80"] == 4).all()
    assert (fc.loc[fc["unique_id"] == "z", ["yhat", "yhat_lo80", "yhat_hi80"]] == 0).all().all()
    assert set(fc["model_name"]) == {MODEL_NAME}


def test_model_name_records_size() -> None:
    assert model_name_for("amazon/chronos-bolt-base") == "ChronosBolt-base"
    assert model_name_for("amazon/chronos-bolt-small") == "ChronosBolt-small"


@pytest.mark.skipif(not MODEL_CACHED, reason="chronos-bolt-small not in the local HF cache")
def test_real_model_zero_shot(train: pd.DataFrame) -> None:
    # Default scope: every series in train_df (no series_ids).
    fc = forecast_foundation(train, horizon=H, model_id="amazon/chronos-bolt-small")
    assert list(fc.columns) == FORECAST_COLUMNS and len(fc) == 9 * H
    assert set(fc["model_name"]) == {"ChronosBolt-small"}
    assert not fc.isna().any().any()
    assert (fc["yhat_lo80"] <= fc["yhat"] + 1e-6).all() and (
        fc["yhat"] <= fc["yhat_hi80"] + 1e-6
    ).all()
    assert fc.attrs["inference_seconds"] > 0
    import sys

    assert "torch" not in sys.modules  # ran in a spawned child
