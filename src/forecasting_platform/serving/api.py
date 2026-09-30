"""FastAPI forecast service (PLAN.md P0.9).

Run: ``uv run uvicorn forecasting_platform.serving.api:app --reload`` (from the repo root, so the
default SQLite MLflow store resolves), then open http://localhost:8000/docs.

The champion is loaded once at startup (``MlflowForecaster``, from ``settings.CHAMPION_MODEL_URI``)
and, being batch-scored, answers from memory. ``create_app`` takes a forecaster *factory* so tests
can inject a lightweight fake: the real champion loads PyTorch, which must never share a process
with LightGBM (see ``forecasting_platform.isolation``). If the model fails to load the service still
starts — ``/health`` stays up and ``/forecast`` answers 503 — so the failure is visible, not fatal.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from forecasting_platform.config import settings
from forecasting_platform.serving.schemas import (
    BacktestSummary,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    model_name: str
    model_uri: str

    def forecast(self, unique_id: str, horizon: int) -> pd.DataFrame:
        """Rows in the shared forecast contract; empty for an unknown ``unique_id``."""


class MlflowForecaster:
    """The registered champion (``models.champion.ChampionForecaster``) loaded via MLflow."""

    def __init__(self, model_uri: str = settings.CHAMPION_MODEL_URI) -> None:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.model_uri = model_uri
        self._model = mlflow.pyfunc.load_model(model_uri)
        self.model_name = self._model.unwrap_python_model().metadata["champion"]

    def forecast(self, unique_id: str, horizon: int) -> pd.DataFrame:
        return self._model.predict(pd.DataFrame({"unique_id": [unique_id], "horizon": [horizon]}))


def backtest_summary(leaderboard_path: Path, model_name: str) -> BacktestSummary | None:
    if not leaderboard_path.exists():
        return None
    board = pd.read_csv(leaderboard_path)
    rows = board[board["model"] == model_name]
    if rows.empty:
        return None
    return BacktestSummary(
        folds=int(rows["fold"].nunique()),
        mean_wape=float(rows["wape"].mean()),
        mean_interval_coverage=float(rows["interval_coverage"].mean()),
        mean_pinball_loss=float(rows["pinball_loss"].mean()),
        mean_business_cost=float(rows["business_cost"].mean()),
    )


def create_app(
    forecaster_factory: Callable[[], Forecaster] = MlflowForecaster,
    leaderboard_path: Path = settings.LEADERBOARD_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.forecaster = forecaster_factory()
            logger.info("Loaded champion %s", app.state.forecaster.model_name)
        except Exception:
            logger.exception("Champion model failed to load; /forecast will return 503")
            app.state.forecaster = None
        yield

    app = FastAPI(
        title="Demand Forecasting Platform API",
        description="28-day daily demand forecasts with 80% prediction intervals, served from the "
        "backtest-selected champion model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_forecaster(request: Request) -> Forecaster:
        forecaster = request.app.state.forecaster
        if forecaster is None:
            raise HTTPException(status_code=503, detail="Champion model is not loaded")
        return forecaster

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=ModelInfo)
    def models(request: Request) -> ModelInfo:
        forecaster = get_forecaster(request)
        return ModelInfo(
            champion=forecaster.model_name,
            model_uri=forecaster.model_uri,
            backtest=backtest_summary(leaderboard_path, forecaster.model_name),
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
        forecaster = get_forecaster(request)
        unique_id = f"{body.store_nbr}_{body.family}"
        rows = forecaster.forecast(unique_id, body.horizon)
        if rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown store/family combination: store_nbr={body.store_nbr}, "
                f"family={body.family!r}",
            )
        return ForecastResponse(
            unique_id=unique_id,
            model_name=forecaster.model_name,
            forecast=[
                ForecastPoint(
                    ds=r.ds.date(), yhat=r.yhat, yhat_lo80=r.yhat_lo80, yhat_hi80=r.yhat_hi80
                )
                for r in rows.itertuples()
            ],
        )

    return app


app = create_app()
