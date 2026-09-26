"""Generate the synthetic test fixture ``tests/fixtures/sample_train.csv``.

Deliberately synthetic — no rows are sampled from the real Kaggle data, since even a small
extract would be redistribution of the competition dataset. Same columns/dtypes as the real
``train.csv``: 3 fake stores x 3 fake families x 120 days, with weekly seasonality, a mild upward
trend, additive noise, occasional zero-sales days and a promotion effect.

Usage: ``uv run python scripts/generate_test_fixture.py``
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_platform.config import PROJECT_ROOT, settings

OUTPUT_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_train.csv"
STORES = [1, 2, 3]
FAMILIES = ["FAMILY_A", "FAMILY_B", "FAMILY_C"]
START_DATE = "2017-01-01"
N_DAYS = 120
WEEKLY_PROFILE = np.array([0.9, 0.85, 0.85, 0.9, 1.05, 1.25, 1.2])  # Mon..Sun multipliers


def generate_fixture(seed: int = settings.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    frames = []
    for store in STORES:
        for family in FAMILIES:
            base = rng.uniform(20, 200)
            trend = base * 0.002 * np.arange(N_DAYS)  # ~+24% over the window
            onpromotion = rng.poisson(1.5, N_DAYS) * (rng.random(N_DAYS) < 0.3)
            sales = (base + trend) * WEEKLY_PROFILE[dates.dayofweek] * (1 + 0.05 * onpromotion)
            sales = sales + rng.normal(0, 0.08 * base, N_DAYS)
            sales[rng.random(N_DAYS) < 0.05] = 0.0  # occasional zero-sales days
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates.strftime("%Y-%m-%d"),
                        "store_nbr": store,
                        "family": family,
                        "sales": np.round(np.clip(sales, 0, None), 3),
                        "onpromotion": onpromotion.astype("int64"),
                    }
                )
            )
    df = pd.concat(frames).sort_values(["date", "store_nbr", "family"], ignore_index=True)
    df.insert(0, "id", np.arange(len(df), dtype="int64"))
    return df


def main(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_fixture()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {output_path}")


if __name__ == "__main__":
    main()
