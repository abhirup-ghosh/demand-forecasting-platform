"""Tests for the shared backtest folds (synthetic data only)."""

from pathlib import Path

import pandas as pd
import pytest

from forecasting_platform.evaluation.backtest import Fold, generate_cv_folds, split_train_test

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
REAL_CALENDAR = pd.Series(pd.date_range("2013-01-01", "2017-08-15", freq="D"))


@pytest.fixture
def folds() -> list[Fold]:
    return generate_cv_folds(REAL_CALENDAR, horizon=28, n_folds=5)


def test_fold_1_is_the_final_holdout(folds: list[Fold]) -> None:
    f1 = folds[0]
    assert f1.fold_id == 1
    assert f1.test_start_date == pd.Timestamp("2017-07-19")
    assert f1.test_end_date == pd.Timestamp("2017-08-15")
    assert f1.train_end_date == pd.Timestamp("2017-07-18")


def test_test_windows_do_not_overlap_and_are_contiguous(folds: list[Fold]) -> None:
    assert len(folds) == 5
    for newer, older in zip(folds, folds[1:], strict=False):
        assert older.test_end_date < newer.test_start_date
        assert newer.test_start_date - older.test_end_date == pd.Timedelta(days=1)
    for f in folds:
        assert (f.test_end_date - f.test_start_date).days + 1 == 28


def test_oldest_fold_dates(folds: list[Fold]) -> None:
    f5 = folds[-1]
    assert f5.test_start_date == pd.Timestamp("2017-03-29")
    assert f5.test_end_date == pd.Timestamp("2017-04-25")


def test_train_windows_precede_test_and_expand(folds: list[Fold]) -> None:
    df = pd.DataFrame({"ds": REAL_CALENDAR, "y": 1.0})
    prev_train_len = None
    for f in folds:
        train, test = split_train_test(df, f)
        assert train["ds"].max() < f.test_start_date  # no dates >= test_start_date
        assert train["ds"].min() == REAL_CALENDAR.min()  # expanding, not sliding
        assert test["ds"].min() == f.test_start_date and test["ds"].max() == f.test_end_date
        assert len(test) == 28
        if prev_train_len is not None:  # older folds train on strictly less data
            assert len(train) == prev_train_len - 28
        prev_train_len = len(train)


def test_split_works_on_the_fixture_with_raw_date_column() -> None:
    raw = pd.read_csv(FIXTURE)  # 2017-01-01..2017-04-30, 'date' column as strings
    folds = generate_cv_folds(raw["date"], horizon=14, n_folds=3)
    assert folds[0].test_end_date == pd.Timestamp("2017-04-30")
    for f in folds:
        train, test = split_train_test(raw, f, date_col="date")
        assert pd.to_datetime(train["date"]).max() < f.test_start_date
        assert test["date"].nunique() == 14
        assert len(test) == 14 * 9  # every series appears in every test window


def test_too_little_history_raises() -> None:
    short = pd.Series(pd.date_range("2017-01-01", periods=100, freq="D"))
    with pytest.raises(ValueError, match="Not enough history"):
        generate_cv_folds(short, horizon=28, n_folds=5)
