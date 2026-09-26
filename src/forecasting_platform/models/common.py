"""Series handling shared by the fitted model tiers (statistical, ML, deep).

- **All-zero series** (no sales in the training window) are forecast as zero, not fitted.
- **Leading zeros** before a series' first sale are trimmed (not yet recorded, not demand —
  docs/eda-findings.md §2).
- **Too-short series** (fewer than ``min_length`` observations after trimming) fall back to
  seasonal naive under the calling tier's model name, so every series gets a forecast.
- Outputs are clipped at 0 (sales can't be negative).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.baseline import seasonal_naive


@dataclass
class SeriesSplit:
    fit_df: pd.DataFrame  # trimmed rows of the series to fit (all input columns kept)
    all_zero_ids: list[str]
    short_ids: list[str]
    last_date: pd.Series  # per unique_id, last training date

    def summary(self) -> str:
        return (
            f"{self.fit_df['unique_id'].nunique()} series fitted, {len(self.all_zero_ids)} "
            f"all-zero -> 0, {len(self.short_ids)} too short -> seasonal naive"
        )


def split_series(train_df: pd.DataFrame, min_length: int) -> SeriesSplit:
    df = train_df.sort_values(["unique_id", "ds"])
    started = df.groupby("unique_id", sort=False)["y"].transform(lambda s: s.ne(0).cummax())
    trimmed = df[started]
    all_zero = sorted(set(df["unique_id"]) - set(trimmed["unique_id"]))
    lengths = trimmed.groupby("unique_id").size()
    short = sorted(lengths[lengths < min_length].index)
    return SeriesSplit(
        fit_df=trimmed[~trimmed["unique_id"].isin(short)].reset_index(drop=True),
        all_zero_ids=all_zero,
        short_ids=short,
        last_date=df.groupby("unique_id")["ds"].max(),
    )


def fallback_forecasts(
    train_df: pd.DataFrame, split: SeriesSplit, horizon: int, model_name: str
) -> list[pd.DataFrame]:
    """Seasonal-naive forecasts for short series and zero forecasts for all-zero series."""
    frames = []
    if split.short_ids:
        # Untrimmed history: a series that just started may have < 9 points after trimming.
        short = train_df[train_df["unique_id"].isin(split.short_ids)]
        frames.append(seasonal_naive(short, horizon).assign(model_name=model_name))
    if split.all_zero_ids:
        steps = pd.to_timedelta(np.arange(1, horizon + 1), unit="D")
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": np.repeat(split.all_zero_ids, horizon),
                    "ds": np.concatenate([split.last_date[i] + steps for i in split.all_zero_ids]),
                    "model_name": model_name,
                    "yhat": 0.0,
                    "yhat_lo80": 0.0,
                    "yhat_hi80": 0.0,
                }
            )
        )
    return frames


def finalize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate tier outputs, clip at 0 and return the shared contract, sorted."""
    out = pd.concat(frames, ignore_index=True)
    for col in ("yhat", "yhat_lo80", "yhat_hi80"):
        out[col] = out[col].clip(lower=0).astype(float)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["model_name", "unique_id", "ds"], ignore_index=True)[FORECAST_COLUMNS]
