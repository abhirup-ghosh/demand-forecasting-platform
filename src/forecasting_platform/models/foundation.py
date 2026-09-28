"""Tier 4 — zero-shot foundation model, Chronos-Bolt (PLAN.md P0.6e; docs/methodology.md §1.7).

Chronos-Bolt is used **zero-shot**: no training or fine-tuning on this data. Each series' history
(leading zeros trimmed, last ``CONTEXT_LENGTH`` days) is the context; the model's 10/50/90%
quantiles map to ``yhat_lo80`` / ``yhat`` / ``yhat_hi80``.

**Scope: all series** (decided 2026-09-28). The plan originally restricted this tier to a
stratified 60-series sample to save laptop compute, but measured inference was under a second for
60 series, so it now runs on every series like the other tiers (all-zero series -> 0).
``select_chronos_sample`` / ``series_ids`` remain available for sample-based analyses.

**Model size:** ``chronos-bolt-base`` by default (Open Decision #2, resolved: runtime left ample
headroom over ``-small``). ``model_id`` selects another size; the model name records which ran.

PyTorch runs in an isolated child process (``forecasting_platform.isolation``): its OpenMP runtime
crashes/deadlocks alongside LightGBM's in one process on macOS.
"""

import logging
import time

import numpy as np
import pandas as pd

from forecasting_platform.config import settings
from forecasting_platform.isolation import run_isolated
from forecasting_platform.models.common import finalize

logger = logging.getLogger(__name__)

MODEL_NAME = "ChronosBolt"  # suffixed with the model size, e.g. "ChronosBolt-base"
CONTEXT_LENGTH = 2048  # Chronos-Bolt's maximum context
QUANTILES = [0.1, 0.5, 0.9]
BATCH_SIZE = 256


def model_name_for(model_id: str) -> str:
    """``amazon/chronos-bolt-base`` -> ``ChronosBolt-base``."""
    return f"{MODEL_NAME}-{model_id.rsplit('-', 1)[-1]}"


def select_chronos_sample(
    train_df: pd.DataFrame, n: int = settings.CHRONOS_SERIES_SAMPLE_SIZE
) -> list[str]:
    """Stratified sample: top, middle and bottom ``n // 3`` series by volume (non-zero only)."""
    volume = train_df.groupby("unique_id")["y"].sum()
    ranked = volume[volume > 0].sort_values(ascending=False)
    k = n // 3
    if len(ranked) < 3 * k:
        return sorted(ranked.index)
    mid_start = len(ranked) // 2 - k // 2
    picks = [*ranked.index[:k], *ranked.index[mid_start : mid_start + k], *ranked.index[-k:]]
    return sorted(picks)


def forecast_foundation(
    train_df: pd.DataFrame,
    horizon: int = settings.FORECAST_HORIZON,
    series_ids: list[str] | None = None,
    model_id: str = settings.CHRONOS_MODEL_ID,
    isolate: bool = True,
) -> pd.DataFrame:
    """Zero-shot Chronos-Bolt forecasts (long format, contract cols).

    Forecasts every series in ``train_df`` unless ``series_ids`` restricts it.
    """
    df = train_df[["unique_id", "ds", "y"]]
    if series_ids is not None:
        df = df[df["unique_id"].isin(series_ids)]
    args = (df, horizon, model_id)
    return run_isolated(_forecast_foundation, *args) if isolate else _forecast_foundation(*args)


def prepare_contexts(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray], list[str], pd.Series]:
    """Per-series context arrays (leading zeros trimmed, last ``CONTEXT_LENGTH`` days).

    Returns (ids with history, their contexts, ids that are all-zero, last date per series).
    """
    df = df.sort_values(["unique_id", "ds"])
    last_date = df.groupby("unique_id")["ds"].max()
    started = df.groupby("unique_id", sort=False)["y"].transform(lambda s: s.ne(0).cummax())
    history = df[started].groupby("unique_id", sort=False)["y"]
    ids = list(history.groups)
    contexts = [g.to_numpy(dtype=np.float32)[-CONTEXT_LENGTH:] for _, g in history]
    all_zero = sorted(set(last_date.index) - set(ids))
    return ids, contexts, all_zero, last_date


def quantiles_to_contract(
    ids: list[str],
    quantiles: np.ndarray,
    all_zero: list[str],
    last_date: pd.Series,
    horizon: int,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Map a (n_series, horizon, 3) array of 10/50/90% quantiles onto the shared contract."""
    steps = pd.to_timedelta(np.arange(1, horizon + 1), unit="D")
    zeros = np.zeros((len(all_zero), horizon, len(QUANTILES)))
    q = np.concatenate([quantiles.reshape(len(ids), horizon, len(QUANTILES)), zeros])
    all_ids = [*ids, *all_zero]
    out = pd.DataFrame(
        {
            "unique_id": np.repeat(all_ids, horizon),
            "ds": np.concatenate([last_date[i] + steps for i in all_ids]),
            "model_name": model_name,
            "yhat": q[:, :, 1].ravel(),
            "yhat_lo80": q[:, :, 0].ravel(),
            "yhat_hi80": q[:, :, 2].ravel(),
        }
    )
    return finalize([out])


def _forecast_foundation(df: pd.DataFrame, horizon: int, model_id: str) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline

    ids, contexts, all_zero, last_date = prepare_contexts(df)
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )
    name = model_name_for(model_id)
    start = time.perf_counter()
    batches = []
    for i in range(0, len(contexts), BATCH_SIZE):
        q, _ = pipeline.predict_quantiles(
            [torch.from_numpy(c) for c in contexts[i : i + BATCH_SIZE]],
            prediction_length=horizon,
            quantile_levels=QUANTILES,
        )
        batches.append(q.numpy())
    elapsed = time.perf_counter() - start
    logger.info("foundation tier (%s): %d series zero-shot in %.1fs", name, len(ids), elapsed)
    quantiles = np.concatenate(batches) if batches else np.empty((0, horizon, len(QUANTILES)))
    result = quantiles_to_contract(ids, quantiles, all_zero, last_date, horizon, name)
    result.attrs["inference_seconds"] = elapsed  # the child's logs don't reach the caller
    return result
