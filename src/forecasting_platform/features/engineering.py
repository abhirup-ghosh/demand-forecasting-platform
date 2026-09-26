"""Leakage-free feature frame shared by the model tiers (PLAN.md P0.4).

Usage: ``uv run python -m forecasting_platform.features.engineering`` builds the frame from
``settings.DATA_RAW_DIR`` and writes ``settings.DATA_PROCESSED_DIR / "features.parquet"``.

Design notes (see ``docs/eda-findings.md`` for the evidence behind each):

- **Complete daily calendar.** ``train.csv`` has no rows for 25 December (stores closed). Every
  series is reindexed to a gap-free daily calendar with ``y = 0`` / ``onpromotion = 0`` on
  missing dates, so a lag of *k* rows is exactly a lag of *k* days.
- **No lookahead.** Every ``y``-derived feature at date ``D`` uses only rows with ``ds < D``:
  lags shift by >= 7 days, and rolling windows are computed on ``y`` shifted by one day.
  Note that ``lag_7``/``lag_14`` are only *known* at forecast time for the first 7/14 days of a
  28-day horizon; the ML tier (P0.6c) handles this through ``mlforecast``'s recursive lags.
- **Holidays.** A holiday counts on the date it is *observed*: ``Holiday``, ``Transfer``,
  ``Additional`` and ``Bridge`` rows with ``transferred == False``. A ``Holiday`` row with
  ``transferred == True`` is a normal working day. ``Work Day`` and ``Event`` rows are not
  holidays. Regional holidays match ``stores.state``; local holidays match ``stores.city``.
- **Oil.** Forward-filled over non-trading days, then back-filled only for the very first
  calendar day (2013-01-01 has no earlier price to carry forward). ``oil_price`` at ``D`` is
  treated as a same-day exogenous value; forecasting beyond the data would require its future
  values, so whether the ML tier uses it is decided in P0.6c.
- **Target.** ``y`` is returned raw; whether to ``log1p``-transform it is Open Decision #4,
  settled empirically in P0.7.
"""

import pandas as pd

from forecasting_platform.config import settings

FEATURE_COLUMNS = [
    "unique_id",
    "ds",
    "y",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "dow",
    "month",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "onpromotion",
    "oil_price",
    "store_type",
    "store_cluster",
    "family",
]
OBSERVED_HOLIDAY_TYPES = ("Holiday", "Transfer", "Additional", "Bridge")
LAGS = (7, 14, 28)


def _observed_holidays(holidays_df: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Unique ``(date, locale_name)`` pairs on which a ``locale`` holiday is actually observed."""
    h = holidays_df
    mask = (
        (h["locale"] == locale)
        & h["type"].isin(OBSERVED_HOLIDAY_TYPES)
        & ~h["transferred"].astype(bool)
    )
    out = h.loc[mask, ["date", "locale_name"]].drop_duplicates()
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_feature_frame(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the long-format feature frame (one row per ``unique_id`` x day).

    ``sales_df`` has ``train.csv``'s columns; the other frames have their raw-file columns.
    Returns exactly ``FEATURE_COLUMNS``, sorted by ``unique_id`` then ``ds``.
    """
    sales = sales_df[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    calendar = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D", name="date")

    # Complete (series x day) grid; dates absent from train.csv (25 Dec) become zero-sales rows.
    series = sales[["store_nbr", "family"]].drop_duplicates()
    grid = series.merge(pd.DataFrame({"date": calendar}), how="cross")
    df = grid.merge(sales, on=["store_nbr", "family", "date"], how="left")
    df[["sales", "onpromotion"]] = df[["sales", "onpromotion"]].fillna(0)
    df["onpromotion"] = df["onpromotion"].astype("int64")

    df["unique_id"] = df["store_nbr"].astype(str) + "_" + df["family"]
    df = df.rename(columns={"date": "ds", "sales": "y"})
    df = df.sort_values(["unique_id", "ds"], ignore_index=True)

    # Target-derived features: strictly past values only.
    by_series = df.groupby("unique_id", sort=False)["y"]
    for k in LAGS:
        df[f"lag_{k}"] = by_series.shift(k)
    past = by_series.shift(1).groupby(df["unique_id"], sort=False)
    df["rolling_mean_7"] = past.transform(lambda s: s.rolling(7).mean())
    df["rolling_mean_28"] = past.transform(lambda s: s.rolling(28).mean())
    df["rolling_std_7"] = past.transform(lambda s: s.rolling(7).std())

    # Calendar.
    df["dow"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    # Store attributes.
    stores = stores_df.rename(columns={"type": "store_type", "cluster": "store_cluster"})
    df = df.merge(
        stores[["store_nbr", "city", "state", "store_type", "store_cluster"]],
        on="store_nbr",
        how="left",
    )

    # Holidays, by observed date and locale.
    national = set(_observed_holidays(holidays_df, "National")["date"])
    df["is_national_holiday"] = df["ds"].isin(national)
    for locale, store_col, out_col in [
        ("Regional", "state", "is_regional_holiday"),
        ("Local", "city", "is_local_holiday"),
    ]:
        observed = _observed_holidays(holidays_df, locale)
        keys = pd.MultiIndex.from_frame(observed[["date", "locale_name"]])
        df[out_col] = pd.MultiIndex.from_frame(df[["ds", store_col]]).isin(keys)

    # Oil: forward-fill non-trading days, back-fill only the leading gap.
    oil = oil_df.assign(date=pd.to_datetime(oil_df["date"])).set_index("date")["dcoilwtico"]
    oil_daily = oil.reindex(calendar).ffill().bfill()
    df["oil_price"] = df["ds"].map(oil_daily)

    df = df.sort_values(["unique_id", "ds"], ignore_index=True)
    return df[FEATURE_COLUMNS]


def main() -> None:
    raw = settings.DATA_RAW_DIR
    features = build_feature_frame(
        pd.read_csv(raw / "train.csv"),
        pd.read_csv(raw / "stores.csv"),
        pd.read_csv(raw / "oil.csv"),
        pd.read_csv(raw / "holidays_events.csv"),
    )
    out = settings.DATA_PROCESSED_DIR / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    print(
        f"Wrote {len(features):,} rows x {features.shape[1]} columns "
        f"({features['unique_id'].nunique()} series) to {out}"
    )


if __name__ == "__main__":
    main()
