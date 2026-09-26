"""Shared rolling-origin cross-validation folds (PLAN.md P0.5).

Every model tier is evaluated on exactly these folds, so comparisons are apples-to-apples.
Folds step backward from the last date in ``horizon``-day increments; each fold's training window
is **expanding** (everything up to and including ``train_end_date``). Fold 1 is the most recent
and, on the real data, is the final holdout 2017-07-19..2017-08-15 (PLAN.md section 3.2).
"""

from dataclasses import dataclass

import pandas as pd

from forecasting_platform.config import settings


@dataclass(frozen=True)
class Fold:
    """One backtest fold. All dates are inclusive; ``fold_id`` 1 is the most recent."""

    fold_id: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def generate_cv_folds(
    dates: pd.Series,
    horizon: int = settings.FORECAST_HORIZON,
    n_folds: int = settings.BACKTEST_N_FOLDS,
) -> list[Fold]:
    """Build ``n_folds`` non-overlapping ``horizon``-day test windows ending at ``dates.max()``.

    Raises ``ValueError`` if the data can't fit ``n_folds`` test windows plus at least
    ``horizon`` days of training history before the oldest fold.
    """
    if horizon < 1 or n_folds < 1:
        raise ValueError(f"horizon and n_folds must be >= 1, got {horizon=}, {n_folds=}")
    ds = pd.to_datetime(pd.Series(dates))
    first, last = ds.min().normalize(), ds.max().normalize()
    day = pd.Timedelta(days=1)

    folds = []
    for i in range(n_folds):
        test_end = last - i * horizon * day
        test_start = test_end - (horizon - 1) * day
        folds.append(Fold(i + 1, test_start - day, test_start, test_end))

    oldest_train_days = (folds[-1].train_end_date - first).days + 1
    if oldest_train_days < horizon:
        raise ValueError(
            f"Not enough history for {n_folds} folds of {horizon} days: the oldest fold would "
            f"train on only {max(oldest_train_days, 0)} day(s) (need >= {horizon})."
        )
    return folds


def split_train_test(
    df: pd.DataFrame, fold: Fold, date_col: str = "ds"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into (train, test) for ``fold``: train = ``date_col <= train_end_date``,
    test = ``test_start_date <= date_col <= test_end_date``. Later rows are dropped."""
    ds = pd.to_datetime(df[date_col])
    train = df[ds <= fold.train_end_date]
    test = df[(ds >= fold.test_start_date) & (ds <= fold.test_end_date)]
    return train, test
