"""Download the Kaggle "Store Sales - Time Series Forecasting" dataset.

Usage: ``uv run python -m forecasting_platform.data.download``

Credentials come from ``.env`` (``KAGGLE_USERNAME``/``KAGGLE_KEY``, or ``KAGGLE_API_TOKEN``) or the
Kaggle CLI's own config. If they're missing, or the competition rules haven't been accepted on
Kaggle, this fails with a pointer to the manual fallback in ``data/README.md``.
"""

import zipfile
from pathlib import Path

from dotenv import load_dotenv

from forecasting_platform.config import PROJECT_ROOT, settings

COMPETITION = "store-sales-time-series-forecasting"
MANUAL_FALLBACK_HINT = (
    "See data/README.md for how to set Kaggle credentials, or use the manual-download fallback "
    f"(download the zip from https://www.kaggle.com/competitions/{COMPETITION}/data and unzip it "
    f"into {settings.DATA_RAW_DIR})."
)


class KaggleDownloadError(RuntimeError):
    """Raised when the Kaggle API can't authenticate or download the competition files."""


def download_kaggle_dataset(dest_dir: Path) -> None:
    """Download the competition zip via the Kaggle API and unzip it into ``dest_dir``."""
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        # Imported lazily: importing `kaggle` itself attempts authentication.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except (Exception, SystemExit) as exc:  # the kaggle client calls sys.exit on missing creds
        raise KaggleDownloadError(
            f"Kaggle authentication failed ({exc!r}). {MANUAL_FALLBACK_HINT}"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(COMPETITION, path=str(dest_dir), quiet=False)
    except Exception as exc:
        raise KaggleDownloadError(
            f"Kaggle download failed ({exc!r}) — if this is a 403, accept the competition rules "
            f"on its Kaggle page first. {MANUAL_FALLBACK_HINT}"
        ) from exc

    zip_path = dest_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print(f"Extracted {COMPETITION} into {dest_dir}")


if __name__ == "__main__":
    try:
        download_kaggle_dataset(settings.DATA_RAW_DIR)
    except KaggleDownloadError as err:
        raise SystemExit(f"ERROR: {err}") from None
