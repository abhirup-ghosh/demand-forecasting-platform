"""Validate the raw Kaggle files against the schema the rest of the pipeline assumes.

Usage: ``uv run python -m forecasting_platform.data.validate``

Hard schema mismatches raise ``DataValidationError``; informational counts (series count,
structurally all-zero series) are only printed.
"""

from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from forecasting_platform.config import settings

# PLAN.md section 3.1 — test.csv/sample_submission.csv share one row there but are two files.
EXPECTED_FILES = (
    "train.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
    "test.csv",
    "sample_submission.csv",
)
TRAIN_SCHEMA = {
    "id": ptypes.is_integer_dtype,
    "date": ptypes.is_object_dtype,  # parsed separately below
    "store_nbr": ptypes.is_integer_dtype,
    "family": ptypes.is_object_dtype,
    "sales": ptypes.is_float_dtype,
    "onpromotion": ptypes.is_integer_dtype,
}
EXPECTED_DATE_RANGE = (pd.Timestamp("2013-01-01"), pd.Timestamp("2017-08-15"))
EXPECTED_N_SERIES = 54 * 33


class DataValidationError(ValueError):
    """Raised on a hard mismatch between the raw data and the expected schema."""


def validate_train_frame(
    train: pd.DataFrame,
    expected_date_range: tuple[pd.Timestamp, pd.Timestamp] | None = EXPECTED_DATE_RANGE,
) -> dict[str, object]:
    """Check ``train.csv``'s columns, dtypes and (optionally) date range; return summary counts.

    Pass ``expected_date_range=None`` to skip the date-range check (e.g. for the synthetic fixture).
    """
    if list(train.columns) != list(TRAIN_SCHEMA):
        raise DataValidationError(
            f"train.csv columns {list(train.columns)} != expected {list(TRAIN_SCHEMA)}"
        )
    for col, is_expected_dtype in TRAIN_SCHEMA.items():
        if not is_expected_dtype(train[col]):
            raise DataValidationError(
                f"train.csv column {col!r} has dtype {train[col].dtype}, "
                f"expected {is_expected_dtype.__name__}"
            )
    try:
        dates = pd.to_datetime(train["date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"train.csv 'date' is not YYYY-MM-DD: {exc}") from exc

    date_min, date_max = dates.min(), dates.max()
    if expected_date_range is not None and (date_min, date_max) != expected_date_range:
        raise DataValidationError(
            f"train.csv date range {date_min.date()}..{date_max.date()} != expected "
            f"{expected_date_range[0].date()}..{expected_date_range[1].date()}"
        )

    series_totals = train.groupby(["store_nbr", "family"])["sales"].apply(lambda s: s.abs().sum())
    return {
        "n_rows": len(train),
        "date_min": date_min.date(),
        "date_max": date_max.date(),
        "n_series": len(series_totals),
        "n_all_zero_series": int((series_totals == 0).sum()),
        "pct_zero_rows": float((train["sales"] == 0).mean() * 100),
    }


def validate_raw_data(
    raw_dir: Path,
    expected_date_range: tuple[pd.Timestamp, pd.Timestamp] | None = EXPECTED_DATE_RANGE,
    expected_n_series: int | None = EXPECTED_N_SERIES,
) -> None:
    """Validate every raw file in ``raw_dir``, printing a summary; raise on hard mismatches."""
    missing = [f for f in EXPECTED_FILES if not (raw_dir / f).is_file()]
    if missing:
        raise DataValidationError(
            f"Missing files in {raw_dir}: {missing} — see data/README.md to populate it."
        )
    print(f"All {len(EXPECTED_FILES)} expected files present in {raw_dir}")

    train = pd.read_csv(raw_dir / "train.csv")
    summary = validate_train_frame(train, expected_date_range)
    print(
        f"train.csv: {summary['n_rows']:,} rows, schema OK, "
        f"dates {summary['date_min']}..{summary['date_max']}"
    )

    stores = pd.read_csv(raw_dir / "stores.csv")
    unknown_stores = sorted(set(train["store_nbr"]) - set(stores["store_nbr"]))
    if unknown_stores:
        raise DataValidationError(f"store_nbr values missing from stores.csv: {unknown_stores}")
    print(f"All {train['store_nbr'].nunique()} store_nbr values found in stores.csv")

    n_series = summary["n_series"]
    print(f"Unique (store_nbr, family) series: {n_series}")
    if expected_n_series is not None and n_series != expected_n_series:
        print(f"  WARNING: expected {expected_n_series} series, found {n_series}")
    print(
        f"Zero-inflation: {summary['pct_zero_rows']:.1f}% of rows have sales == 0; "
        f"{summary['n_all_zero_series']} series are structurally all-zero "
        "(forecast as zero, not fed to model tiers)"
    )
    print("all checks passed")


if __name__ == "__main__":
    try:
        validate_raw_data(settings.DATA_RAW_DIR)
    except DataValidationError as err:
        raise SystemExit(f"ERROR: {err}") from None
