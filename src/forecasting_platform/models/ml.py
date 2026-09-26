"""Tier 2 — global gradient-boosted trees (PLAN.md P0.6c; theory in docs/methodology.md §1.5).

One LightGBM model across all series via ``mlforecast``:

- **Target features** mirror P0.4's windows: lags 7/14/28, and rolling mean 7/28 + rolling std 7
  over ``y`` shifted by one day. Multi-step forecasts are **recursive** (mlforecast's default):
  predictions are fed back in as lags for later horizon steps.
- **Date features:** day of week, month.
- **Dynamic exogenous features:** ``onpromotion`` and the three holiday flags use their actual
  future values — promotions and holidays are planned in advance, so they're known at forecast
  time. ``oil_price`` is **not** known in advance, so over the horizon it is held at its last
  observed training value (using the realised future price would leak information).
- **Static features:** store type, store cluster, family (categorical).
- **Target transform:** raw by default; ``log_target=True`` fits on ``log1p(y)`` (Open Decision
  #4, decided empirically in P0.7). The model name records which variant ran.
- **Intervals:** conformal (``PredictionIntervals``), calibrated on rolling windows of the
  training data.
- Hyperparameters are sensible fixed defaults; tuning is P1.4.

All-zero / leading-zero / too-short series are handled by ``models.common``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import GlobalSklearnTransformer
from mlforecast.utils import PredictionIntervals
from sklearn.preprocessing import FunctionTransformer

from forecasting_platform.config import settings
from forecasting_platform.models.common import fallback_forecasts, finalize, split_series

logger = logging.getLogger(__name__)

LAGS = [7, 14, 28]
DYNAMIC_FEATURES = [
    "onpromotion",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "oil_price",
]
STATIC_FEATURES = ["store_type", "store_cluster", "family"]
CONFORMAL_WINDOWS = 2
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}


def _model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs with LightGBM-friendly dtypes (bools -> int, statics -> category)."""
    cols = ["unique_id", "ds", "y", *DYNAMIC_FEATURES, *STATIC_FEATURES]
    out = df[[c for c in cols if c in df.columns]].copy()
    for col in DYNAMIC_FEATURES[1:4]:
        out[col] = out[col].astype("int8")
    for col in STATIC_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(str).astype("category")
    return out


def _future_exog(test_df: pd.DataFrame, train_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Future exogenous values: planned (promo, holidays) as given; oil frozen at last value."""
    future = test_df[test_df["unique_id"].isin(ids)][["unique_id", "ds", *DYNAMIC_FEATURES]].copy()
    last_oil = train_df.sort_values("ds")["oil_price"].iloc[-1]
    future["oil_price"] = last_oil
    for col in DYNAMIC_FEATURES[1:4]:
        future[col] = future[col].astype("int8")
    return future


def save_feature_importance(mlf: MLForecast, model_name: str, path: Path) -> pd.DataFrame:
    """Write the fitted LightGBM model's gain/split importances to ``path`` (CSV)."""
    booster = mlf.models_[model_name].booster_
    imp = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "split": booster.feature_importance("split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(path, index=False)
    return imp


def forecast_ml(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int = settings.FORECAST_HORIZON,
    log_target: bool = False,
    importance_path: Path | None = None,
    seed: int = settings.RANDOM_SEED,
) -> pd.DataFrame:
    """Global LightGBM forecasts for every series in ``train_df`` (long format, contract cols).

    ``train_df``/``test_df`` are a fold's split of the P0.4 feature frame. Only the *exogenous*
    columns of ``test_df`` are used (never its ``y``). If ``importance_path`` is given, feature
    importances are written there as CSV.
    """
    name = "LightGBM_log1p" if log_target else "LightGBM"
    split = split_series(train_df, CONFORMAL_WINDOWS * horizon + max(LAGS))
    logger.info("ML tier (%s): %s", name, split.summary())

    frames = fallback_forecasts(train_df[["unique_id", "ds", "y"]], split, horizon, name)
    if not split.fit_df.empty:
        target_transforms = (
            [GlobalSklearnTransformer(FunctionTransformer(np.log1p, np.expm1))]
            if log_target
            else None
        )
        mlf = MLForecast(
            models={name: LGBMRegressor(random_state=seed, **LGBM_PARAMS)},
            freq="D",
            lags=LAGS,
            lag_transforms={1: [RollingMean(7), RollingMean(28), RollingStd(7)]},
            date_features=["dayofweek", "month"],
            target_transforms=target_transforms,
        )
        mlf.fit(
            _model_frame(split.fit_df),
            static_features=STATIC_FEATURES,
            prediction_intervals=PredictionIntervals(n_windows=CONFORMAL_WINDOWS, h=horizon),
        )
        ids = sorted(split.fit_df["unique_id"].unique())
        fc = mlf.predict(h=horizon, X_df=_future_exog(test_df, train_df, ids), level=[80])
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": fc["unique_id"],
                    "ds": fc["ds"],
                    "model_name": name,
                    "yhat": fc[name],
                    "yhat_lo80": fc[f"{name}-lo-80"],
                    "yhat_hi80": fc[f"{name}-hi-80"],
                }
            )
        )
        if importance_path is not None:
            save_feature_importance(mlf, name, importance_path)
    return finalize(frames)
