"""Pydantic request/response models for the forecast API (PLAN.md P0.9)."""

from datetime import date

from pydantic import BaseModel, Field

from forecasting_platform.config import settings


class ForecastRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, description="Store number (1-54 in the Favorita data)")
    family: str = Field(..., min_length=1, description="Product family, e.g. 'GROCERY I'")
    horizon: int = Field(
        settings.FORECAST_HORIZON,
        ge=1,
        le=settings.FORECAST_HORIZON,
        description=f"Days ahead to forecast (1-{settings.FORECAST_HORIZON})",
    )

    model_config = {
        "json_schema_extra": {"examples": [{"store_nbr": 1, "family": "GROCERY I", "horizon": 7}]}
    }


class ForecastPoint(BaseModel):
    ds: date
    yhat: float = Field(..., description="Point forecast (units sold)")
    yhat_lo80: float = Field(..., description="Lower bound of the 80% prediction interval")
    yhat_hi80: float = Field(..., description="Upper bound of the 80% prediction interval")


class ForecastResponse(BaseModel):
    unique_id: str
    model_name: str
    forecast: list[ForecastPoint]


class BacktestSummary(BaseModel):
    folds: int
    mean_wape: float
    mean_interval_coverage: float
    mean_pinball_loss: float
    mean_business_cost: float


class ModelInfo(BaseModel):
    champion: str
    model_uri: str
    backtest: BacktestSummary | None = Field(
        None, description="Mean metrics over the backtest folds (null if no leaderboard found)"
    )
