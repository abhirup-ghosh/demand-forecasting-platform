"""Forecast accuracy metrics (PLAN.md P0.7; plain-language definitions in docs/methodology.md).

- **WAPE** (headline): total absolute error / total actual volume. Well defined with zero actuals.
- **MAPE**: mean of per-row |error| / |actual| over rows with a non-zero actual only. Reported but
  never led with: it is undefined on zero-actual rows (31% of this data) and explodes on tiny ones.
- **RMSE**: in sales units; dominated by high-volume series.
- **Pinball loss**: mean quantile loss over the 10/50/90% quantiles (``yhat_lo80`` / ``yhat`` /
  ``yhat_hi80``) — scores the whole predicted distribution, not just the point forecast.
- **Interval coverage**: share of actuals inside [lo, hi]; compare with the nominal 80%.
"""

import numpy as np
import pandas as pd

from forecasting_platform.evaluation.business_cost import business_cost

ArrayLike = np.ndarray | pd.Series | list[float]
QUANTILES = (0.1, 0.5, 0.9)


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _arr(y_true), _arr(y_pred)
    total = np.abs(y).sum()
    return float(np.abs(y - p).sum() / total) if total > 0 else float("nan")


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error over non-zero actuals only (unstable; see module doc)."""
    y, p = _arr(y_true), _arr(y_pred)
    nz = y != 0
    return float(np.mean(np.abs(y[nz] - p[nz]) / np.abs(y[nz]))) if nz.any() else float("nan")


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def pinball_loss(
    y_true: ArrayLike, y_pred_quantiles: np.ndarray, quantiles: tuple[float, ...] = QUANTILES
) -> float:
    """Mean pinball loss; ``y_pred_quantiles`` has shape (n, len(quantiles))."""
    y = _arr(y_true)[:, None]
    q = np.asarray(quantiles, dtype=float)[None, :]
    diff = y - np.asarray(y_pred_quantiles, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    y = _arr(y_true)
    return float(np.mean((y >= _arr(lo)) & (y <= _arr(hi))))


def score_forecast(
    actuals: pd.DataFrame,
    forecast: pd.DataFrame,
    active_ids: set[str],
    volume_band: pd.Series | None = None,
) -> dict[str, float]:
    """All metrics for one model on one fold.

    ``actuals`` has ``unique_id, ds, y`` for the test window; ``forecast`` follows the shared
    contract. Every actual must have a forecast (raises otherwise). Coverage is computed over
    ``active_ids`` only — structurally all-zero series are trivially "covered" by a 0-width band
    and would inflate it. ``volume_band`` (unique_id -> "top"/"middle"/"bottom") adds per-band WAPE.
    """
    j = actuals[["unique_id", "ds", "y"]].merge(forecast, on=["unique_id", "ds"], how="left")
    if j["yhat"].isna().any():
        missing = j.loc[j["yhat"].isna(), "unique_id"].nunique()
        raise ValueError(f"forecast is missing values for {missing} series in the test window")
    active = j[j["unique_id"].isin(active_ids)]
    out = {
        "wape": wape(j["y"], j["yhat"]),
        "mape": mape(j["y"], j["yhat"]),
        "rmse": rmse(j["y"], j["yhat"]),
        "pinball_loss": pinball_loss(j["y"], j[["yhat_lo80", "yhat", "yhat_hi80"]].to_numpy()),
        "interval_coverage": interval_coverage(
            active["y"], active["yhat_lo80"], active["yhat_hi80"]
        ),
        "business_cost": business_cost(j["y"], j["yhat"]),
    }
    if volume_band is not None:
        banded = j.join(volume_band.rename("band"), on="unique_id")
        for band, g in banded.groupby("band"):
            out[f"wape_{band}"] = wape(g["y"], g["yhat"])
    return out
