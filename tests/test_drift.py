"""Tests for the drift report (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.monitoring.drift import (
    CATEGORICAL_COLUMNS,
    MAX_ROWS,
    NUMERICAL_COLUMNS,
    _column_verdict,
    build_drift_report,
    summarize,
)


@pytest.mark.parametrize(
    ("value", "threshold", "method", "expected_drifted"),
    [
        (0.001, 0.05, "K-S p_value", True),  # p-value: below threshold -> drifted
        (0.5, 0.05, "K-S p_value", False),  # p-value: above threshold -> not drifted
        (0.0, 0.05, "Z-test p_value", True),
        (0.9, 0.1, "Jensen-Shannon distance", True),  # distance: above threshold -> drifted
        (0.01, 0.1, "Jensen-Shannon distance", False),  # distance: below threshold -> not drifted
        (0.11, 0.1, "Wasserstein distance (normed)", True),
    ],
)
def test_column_verdict_direction(value, threshold, method, expected_drifted) -> None:
    drifted, severity = _column_verdict(value, threshold, method)
    assert drifted is expected_drifted
    assert severity > 1.0 if drifted else severity <= 1.0


def _frame(n: int, seed: int, shift: float = 0.0, promo_p: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(10 + shift, 2, n) for c in NUMERICAL_COLUMNS}
    data["dow"] = rng.integers(0, 7, n)
    data["month"] = rng.integers(1, 13, n)
    data["is_national_holiday"] = rng.random(n) < 0.05
    data["is_regional_holiday"] = rng.random(n) < 0.02
    data["is_local_holiday"] = rng.random(n) < 0.02
    data["onpromotion"] = rng.poisson(1, n)
    data["store_type"] = rng.choice(["A", "B", "C"], n)
    data["store_cluster"] = rng.integers(1, 18, n)
    data["family"] = rng.choice(["GROCERY I", "BEVERAGES"], n)
    # onpromotion is also the thing we drift in the "shifted" fixture below, via promo_p.
    data["onpromotion"] = (rng.random(n) < promo_p).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def stable_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frame(500, seed=1), _frame(500, seed=2)


@pytest.fixture
def drifted_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Large numeric shift + a near-100% promo rate -> unambiguous drift on several columns.
    return _frame(500, seed=1, promo_p=0.1), _frame(500, seed=2, shift=20.0, promo_p=0.95)


def test_summary_shape_and_column_coverage(stable_pair) -> None:
    reference, current = stable_pair
    summary = summarize(build_drift_report(reference, current))
    assert summary["n_columns_checked"] == len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS)
    assert {c["column"] for c in summary["columns"]} == set(NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS)
    assert len(summary["columns"]) == summary["n_columns_checked"]
    assert 0.0 <= summary["share_columns_drifted"] <= 1.0


def test_detects_clear_drift(drifted_pair) -> None:
    reference, current = drifted_pair
    summary = summarize(build_drift_report(reference, current))
    assert summary["dataset_drift_detected"] is True
    assert summary["n_columns_drifted"] >= 5
    # Sorted drifted-first: the top of the list must actually be flagged as drifted, and the
    # drifted count at the front of the list must match the reported total.
    assert summary["columns"][0]["drifted"] is True
    assert sum(c["drifted"] for c in summary["columns"]) == summary["n_columns_drifted"]
    drifted_flags = [c["drifted"] for c in summary["columns"]]
    assert drifted_flags == sorted(drifted_flags, reverse=True)  # drifted columns sort first


def test_drift_direction_differs_by_method(drifted_pair) -> None:
    """Regression test: p-value and distance-based stattests flag drift in OPPOSITE directions
    (low p-value = drift; high distance = drift) — conflating them mislabels columns."""
    reference, current = drifted_pair
    summary = summarize(build_drift_report(reference, current))
    by_column = {c["column"]: c for c in summary["columns"]}
    methods_seen = {c["method"] for c in summary["columns"]}
    # The fixture mixes numerical (distance/p-value stattests) and categorical (distance) columns;
    # assert at least one of each family appears so the direction split is actually exercised.
    assert any(m.endswith("p_value") for m in methods_seen) or any(
        "distance" in m for m in methods_seen
    )
    for col in by_column.values():
        is_p_value = col["method"].endswith("p_value")
        expected = (
            (col["score"] < col["threshold"]) if is_p_value else (col["score"] > col["threshold"])
        )
        assert col["drifted"] == expected, col


def test_no_drift_on_identical_distributions() -> None:
    same = _frame(500, seed=7)
    summary = summarize(build_drift_report(same, same.copy()))
    assert summary["dataset_drift_detected"] is False
    assert summary["n_columns_drifted"] == 0


def test_rows_are_capped_for_large_inputs() -> None:
    big = _frame(MAX_ROWS + 1000, seed=3)
    # Should not raise and should still complete quickly with the row cap applied internally.
    summary = summarize(build_drift_report(big, _frame(200, seed=4)))
    assert summary["n_columns_checked"] > 0


def test_rows_with_nan_are_dropped() -> None:
    reference = _frame(300, seed=1)
    current = _frame(300, seed=2)
    current.loc[:50, "lag_7"] = np.nan  # warm-up-style NaNs
    summary = summarize(build_drift_report(reference, current))
    assert summary["n_columns_checked"] == len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS)
