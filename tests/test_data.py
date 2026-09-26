"""Tests for raw-data validation, using only the synthetic fixture (no Kaggle data needed)."""

from pathlib import Path

import pandas as pd
import pytest

from forecasting_platform.data.validate import (
    EXPECTED_FILES,
    DataValidationError,
    validate_raw_data,
    validate_train_frame,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture
def raw_dir(tmp_path: Path, train: pd.DataFrame) -> Path:
    """A raw dir with the fixture as train.csv plus minimal stand-ins for the other files."""
    train.to_csv(tmp_path / "train.csv", index=False)
    stores = pd.DataFrame(
        {
            "store_nbr": sorted(train["store_nbr"].unique()),
            "city": "X",
            "state": "X",
            "type": "A",
            "cluster": 1,
        }
    )
    stores.to_csv(tmp_path / "stores.csv", index=False)
    for name in set(EXPECTED_FILES) - {"train.csv", "stores.csv"}:
        (tmp_path / name).write_text("placeholder\n")
    return tmp_path


def test_fixture_schema_passes(train: pd.DataFrame) -> None:
    summary = validate_train_frame(train, expected_date_range=None)
    assert summary["n_series"] == 9
    assert summary["n_rows"] == 9 * 120
    assert summary["n_all_zero_series"] == 0


def test_date_range_mismatch_raises(train: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="date range"):
        validate_train_frame(train)  # fixture doesn't span the real 2013..2017 range


def test_missing_column_raises(train: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="columns"):
        validate_train_frame(train.drop(columns="onpromotion"), expected_date_range=None)


def test_wrong_dtype_raises(train: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="'sales'"):
        validate_train_frame(train.astype({"sales": str}), expected_date_range=None)


def test_bad_date_format_raises(train: pd.DataFrame) -> None:
    train.loc[0, "date"] = "01/01/2017"
    with pytest.raises(DataValidationError, match="YYYY-MM-DD"):
        validate_train_frame(train, expected_date_range=None)


def test_all_zero_series_counted(train: pd.DataFrame) -> None:
    train.loc[(train["store_nbr"] == 1) & (train["family"] == "FAMILY_A"), "sales"] = 0.0
    assert validate_train_frame(train, expected_date_range=None)["n_all_zero_series"] == 1


def test_validate_raw_data_passes_on_fixture_dir(
    raw_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    validate_raw_data(raw_dir, expected_date_range=None, expected_n_series=None)
    assert "all checks passed" in capsys.readouterr().out


def test_missing_file_raises(raw_dir: Path) -> None:
    (raw_dir / "oil.csv").unlink()
    with pytest.raises(DataValidationError, match="oil.csv"):
        validate_raw_data(raw_dir, expected_date_range=None, expected_n_series=None)


def test_unknown_store_raises(raw_dir: Path) -> None:
    pd.read_csv(raw_dir / "stores.csv").iloc[1:].to_csv(raw_dir / "stores.csv", index=False)
    with pytest.raises(DataValidationError, match="stores.csv"):
        validate_raw_data(raw_dir, expected_date_range=None, expected_n_series=None)
