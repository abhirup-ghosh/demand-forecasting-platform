"""Tests for the feature frame, using only synthetic data (no Kaggle data needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting_platform.features.engineering import FEATURE_COLUMNS, build_feature_frame

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"
TARGET_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
]
MISSING_DATE = "2017-02-10"  # dropped from the fixture, like 25 December in the real data


@pytest.fixture
def sales() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    return df[df["date"] != MISSING_DATE].reset_index(drop=True)


@pytest.fixture
def stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "city": ["Quito", "Quito", "Manta"],
            "state": ["Pichincha", "Pichincha", "Manabi"],
            "type": ["A", "D", "C"],
            "cluster": [13, 8, 3],
        }
    )


@pytest.fixture
def oil() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", "2017-04-30", freq="D")
    prices = np.linspace(50, 60, len(dates))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "dcoilwtico": prices})
    df.loc[0, "dcoilwtico"] = np.nan  # leading gap -> must be back-filled
    df.loc[10, "dcoilwtico"] = np.nan  # NaN price on a present row -> forward-filled
    return df[~dates.dayofweek.isin([5, 6]) | (df.index == 0)]  # no weekend rows, like oil.csv


@pytest.fixture
def holidays() -> pd.DataFrame:
    rows = [
        # A national holiday moved from 2017-01-20 (normal day) to 2017-01-23 (observed).
        ("2017-01-20", "Holiday", "National", "Ecuador", True),
        ("2017-01-23", "Transfer", "National", "Ecuador", False),
        ("2017-02-27", "Holiday", "National", "Ecuador", False),
        ("2017-02-28", "Bridge", "National", "Ecuador", False),
        ("2017-03-01", "Work Day", "National", "Ecuador", False),
        ("2017-03-02", "Event", "National", "Ecuador", False),
        ("2017-03-10", "Holiday", "Regional", "Manabi", False),
        ("2017-03-15", "Holiday", "Local", "Quito", False),
        ("2017-03-15", "Holiday", "Local", "Quito", False),  # duplicate row, as in the real file
    ]
    return pd.DataFrame(
        rows, columns=["date", "type", "locale", "locale_name", "transferred"]
    ).assign(description="x")


@pytest.fixture
def features(sales, stores, oil, holidays) -> pd.DataFrame:
    return build_feature_frame(sales, stores, oil, holidays)


def _row(features: pd.DataFrame, uid: str, date: str) -> pd.Series:
    return features[(features["unique_id"] == uid) & (features["ds"] == date)].iloc[0]


def test_columns_and_shape(features: pd.DataFrame) -> None:
    assert list(features.columns) == FEATURE_COLUMNS
    assert features["unique_id"].nunique() == 9
    assert len(features) == 9 * 120  # complete daily grid, missing date restored
    assert not features.duplicated(["unique_id", "ds"]).any()
    assert features["unique_id"].str.fullmatch(r"\d+_FAMILY_[ABC]").all()


def test_missing_calendar_date_filled_with_zero(features: pd.DataFrame) -> None:
    filled = features[features["ds"] == MISSING_DATE]
    assert len(filled) == 9
    assert (filled["y"] == 0).all() and (filled["onpromotion"] == 0).all()


def test_lag_7_equals_y_seven_days_earlier(features: pd.DataFrame) -> None:
    """Explicit leakage check: lag_7 at D == y at D - 7 days for the same series, by date."""
    y_by_date = features.set_index(["unique_id", "ds"])["y"]
    for k in (7, 14, 28):
        earlier = pd.MultiIndex.from_arrays(
            [features["unique_id"], features["ds"] - pd.Timedelta(days=k)]
        )
        expected = y_by_date.reindex(earlier).to_numpy()
        np.testing.assert_array_equal(features[f"lag_{k}"].to_numpy(), expected)


@pytest.mark.parametrize("cutoff", ["2017-01-15", "2017-02-11", "2017-03-20", "2017-04-30"])
def test_no_feature_uses_rows_at_or_after_d(sales, stores, oil, holidays, cutoff) -> None:
    """Explicit leakage check: perturbing y on all rows with ds >= D leaves features at D intact."""
    base = build_feature_frame(sales, stores, oil, holidays)
    future = pd.to_datetime(sales["date"]) >= cutoff
    perturbed_sales = sales.assign(sales=sales["sales"].where(~future, sales["sales"] * 100 + 999))
    perturbed = build_feature_frame(perturbed_sales, stores, oil, holidays)

    at_d = base["ds"] == cutoff
    pd.testing.assert_frame_equal(
        base.loc[at_d, TARGET_FEATURES], perturbed.loc[at_d, TARGET_FEATURES]
    )
    # Sanity: the perturbation really changed y from D onward.
    assert (base.loc[at_d, "y"] != perturbed.loc[at_d, "y"]).any()


def test_rolling_windows_use_only_past_values(features: pd.DataFrame) -> None:
    s = features[features["unique_id"] == "1_FAMILY_A"].set_index("ds")["y"]
    d = pd.Timestamp("2017-03-01")
    window7 = s.loc[d - pd.Timedelta(days=7) : d - pd.Timedelta(days=1)]
    row = _row(features, "1_FAMILY_A", "2017-03-01")
    assert row["rolling_mean_7"] == pytest.approx(window7.mean())
    assert row["rolling_std_7"] == pytest.approx(window7.std())
    window28 = s.loc[d - pd.Timedelta(days=28) : d - pd.Timedelta(days=1)]
    assert row["rolling_mean_28"] == pytest.approx(window28.mean())


def test_transferred_holiday_is_observed_on_transfer_date(features: pd.DataFrame) -> None:
    national = features.groupby("ds")["is_national_holiday"].first()
    assert not national[pd.Timestamp("2017-01-20")]  # transferred=True -> normal working day
    assert national[pd.Timestamp("2017-01-23")]  # the Transfer row is the observed date
    assert national[pd.Timestamp("2017-02-27")] and national[pd.Timestamp("2017-02-28")]
    assert not national[pd.Timestamp("2017-03-01")]  # Work Day
    assert not national[pd.Timestamp("2017-03-02")]  # Event
    assert national.sum() == 3


def test_regional_and_local_holidays_match_store_location(features: pd.DataFrame) -> None:
    regional = features[features["is_regional_holiday"]]
    assert set(regional["ds"]) == {pd.Timestamp("2017-03-10")}
    assert set(regional["unique_id"].str.split("_").str[0]) == {"3"}  # Manabi store only
    local = features[features["is_local_holiday"]]
    assert set(local["ds"]) == {pd.Timestamp("2017-03-15")}
    assert set(local["unique_id"].str.split("_").str[0]) == {"1", "2"}  # Quito stores only
    assert len(local) == 6  # duplicate holiday row doesn't duplicate feature rows


def test_oil_forward_filled_with_leading_backfill(features: pd.DataFrame, oil) -> None:
    oil_by_date = features.groupby("ds")["oil_price"].first()
    assert oil_by_date.notna().all()
    # Leading NaN on 2017-01-01 takes the first observed price.
    first_price = oil["dcoilwtico"].dropna().iloc[0]
    assert oil_by_date[pd.Timestamp("2017-01-01")] == pytest.approx(first_price)
    # A Saturday carries Friday's price forward.
    assert oil_by_date[pd.Timestamp("2017-02-04")] == oil_by_date[pd.Timestamp("2017-02-03")]


def test_static_and_calendar_columns(features: pd.DataFrame) -> None:
    row = _row(features, "3_FAMILY_C", "2017-03-05")  # a Sunday
    assert row["store_type"] == "C" and row["store_cluster"] == 3 and row["family"] == "FAMILY_C"
    assert row["dow"] == 6 and row["month"] == 3
