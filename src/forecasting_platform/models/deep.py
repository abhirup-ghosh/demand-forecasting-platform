"""Tier 3 — global deep learning with NHITS (PLAN.md P0.6d; theory in docs/methodology.md §1.6).

One NHITS network (``h=28``, ``input_size=56``) trained across all series via ``neuralforecast``.

Configuration choices:

- **Univariate:** NHITS learns from the sales history alone — the point of this tier is to test
  learned structure against the engineered features of tier 2 (which, on fold 1, got <1% of its
  gain from promotions and holidays anyway).
- **Per-window robust scaling** (``scaler_type="robust"``) so one network handles series whose
  scales differ by orders of magnitude.
- **Fixed training budget** ``max_steps=1000`` with library defaults otherwise. Not tuned on the
  backtest folds (fold 1 is the final holdout); tuning is P1.4.
- **CPU**: measured faster than Apple-MPS for this small network (27 vs 52 ms/step).
- **Intervals:** conformal (``PredictionIntervals``, 2 windows; neuralforecast retrains during
  calibration, roughly doubling cost).
- **Process isolation:** training runs in a spawned child process and ``neuralforecast``/``torch``
  are imported only there, because PyTorch's and LightGBM's OpenMP runtimes crash or deadlock when
  loaded in the same process on macOS (see ``forecasting_platform.isolation``).

All-zero / leading-zero / too-short series are handled by ``models.common``.
"""

import logging
import warnings

import pandas as pd

from forecasting_platform.config import settings
from forecasting_platform.isolation import run_isolated
from forecasting_platform.models.common import fallback_forecasts, finalize, split_series

logger = logging.getLogger(__name__)

MODEL_NAME = "NHITS"
CONFORMAL_WINDOWS = 2
MAX_STEPS = 1000


def forecast_deep(
    train_df: pd.DataFrame,
    horizon: int = settings.FORECAST_HORIZON,
    input_size: int | None = None,
    max_steps: int = MAX_STEPS,
    seed: int = settings.RANDOM_SEED,
    accelerator: str = "cpu",
    isolate: bool = True,
) -> pd.DataFrame:
    """Global NHITS forecasts for every series in ``train_df`` (long format, contract cols).

    ``input_size`` defaults to ``2 * horizon`` (56 days for the 28-day task). With ``isolate=True``
    (default) training runs in a spawned child process — see the module docstring.
    """
    args = (train_df[["unique_id", "ds", "y"]], horizon, input_size, max_steps, seed, accelerator)
    return run_isolated(_forecast_deep, *args) if isolate else _forecast_deep(*args)


def _forecast_deep(
    df: pd.DataFrame,
    horizon: int,
    input_size: int | None,
    max_steps: int,
    seed: int,
    accelerator: str,
) -> pd.DataFrame:
    nf, split = fit_deep(df, horizon, input_size, max_steps, seed, accelerator)
    frames = fallback_forecasts(df, split, horizon, MODEL_NAME)
    if nf is not None:
        frames.append(predict_deep(nf))
    return finalize(frames)


def fit_deep(
    df: pd.DataFrame,
    horizon: int = settings.FORECAST_HORIZON,
    input_size: int | None = None,
    max_steps: int = MAX_STEPS,
    seed: int = settings.RANDOM_SEED,
    accelerator: str = "cpu",
):
    """Fit NHITS in the *current* process; returns (NeuralForecast or None, SeriesSplit).

    Imports torch here — only call it in a process that never loads LightGBM (see module doc).
    ``nf`` is None when no series is long enough to fit.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.utils import PredictionIntervals

    input_size = input_size or 2 * horizon
    split = split_series(df, CONFORMAL_WINDOWS * horizon + input_size)
    logger.info("deep tier (%s): %s", MODEL_NAME, split.summary())
    if split.fit_df.empty:
        return None, split
    model = NHITS(
        h=horizon,
        input_size=input_size,
        max_steps=max_steps,
        scaler_type="robust",
        random_seed=seed,
        accelerator=accelerator,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Lightning's dataloader/worker advisories
        nf.fit(split.fit_df, prediction_intervals=PredictionIntervals(n_windows=CONFORMAL_WINDOWS))
    return nf, split


def predict_deep(nf, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contract-format forecasts from a fitted NeuralForecast (``df`` = history, if reloaded)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = nf.predict(df=df, level=[80])
    return pd.DataFrame(
        {
            "unique_id": fc["unique_id"],
            "ds": fc["ds"],
            "model_name": MODEL_NAME,
            "yhat": fc[MODEL_NAME],
            "yhat_lo80": fc[f"{MODEL_NAME}-lo-80"],
            "yhat_hi80": fc[f"{MODEL_NAME}-hi-80"],
        }
    )
