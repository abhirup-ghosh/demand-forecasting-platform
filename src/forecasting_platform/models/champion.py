"""Champion selection and the registered MLflow model (PLAN.md P0.8).

**Selection rule (fixed in advance, PLAN.md P0.8):** lowest mean WAPE across all backtest folds,
excluding the naive baselines (a floor, not candidates). Only models scored on every fold qualify.

**Registered model:** ``ChampionForecaster`` is an MLflow pyfunc holding the champion's real
trained weights plus the recent history it needs as context. It is *batch-scored*: on load it
runs the model once for every series and then serves forecasts from memory — the usual shape of
a daily demand-forecasting system, and here there is no newer data to re-score against anyway.
``predict`` takes a DataFrame with a ``unique_id`` column (optional ``horizon`` column, ≤ 28) and
returns rows in the shared forecast contract.

torch is imported only inside ``load_context``, so importing this module never loads it.
"""

import json

import mlflow
import pandas as pd

from forecasting_platform.config import settings
from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.models.common import finalize

BASELINES = ("Naive", "SeasonalNaive")
REGISTERED_MODEL_NAME = "demand-forecast-champion"


def select_champion(leaderboard: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Apply the P0.8 rule; returns (champion model name, ranked candidate table)."""
    n_folds = leaderboard["fold"].nunique()
    candidates = leaderboard[~leaderboard["model"].isin(BASELINES)]
    table = (
        candidates.groupby("model")
        .agg(
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
            folds=("fold", "nunique"),
            mean_coverage=("interval_coverage", "mean"),
            mean_pinball=("pinball_loss", "mean"),
            mean_business_cost=("business_cost", "mean"),
        )
        .sort_values("mean_wape")
    )
    table = table[table["folds"] == n_folds]
    if table.empty:
        raise ValueError("no non-baseline model has results for every fold")
    return str(table.index[0]), table


class ChampionForecaster(mlflow.pyfunc.PythonModel):
    """Batch-scored champion. Artifacts: ``nf_model``, ``history``, ``fallback``, ``metadata``."""

    def load_context(self, context) -> None:
        from neuralforecast import NeuralForecast

        from forecasting_platform.models.deep import predict_deep

        with open(context.artifacts["metadata"]) as fh:
            self.metadata = json.load(fh)
        nf = NeuralForecast.load(context.artifacts["nf_model"])
        history = pd.read_parquet(context.artifacts["history"])
        fallback = pd.read_parquet(context.artifacts["fallback"])
        self.forecasts = finalize([predict_deep(nf, history), fallback]).assign(
            model_name=self.metadata["champion"]
        )

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        horizon = settings.FORECAST_HORIZON
        if "horizon" in model_input.columns and len(model_input):
            horizon = int(model_input["horizon"].max())
        if not 1 <= horizon <= settings.FORECAST_HORIZON:
            raise ValueError(f"horizon must be 1..{settings.FORECAST_HORIZON}, got {horizon}")
        out = self.forecasts[self.forecasts["unique_id"].isin(model_input["unique_id"])]
        first_day = out.groupby("unique_id")["ds"].transform("min")
        out = out[(out["ds"] - first_day).dt.days < horizon]
        return out.reset_index(drop=True)[FORECAST_COLUMNS]
