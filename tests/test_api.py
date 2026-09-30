"""API tests with FastAPI's TestClient and a fake forecaster (no MLflow model or torch needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecasting_platform.models import FORECAST_COLUMNS
from forecasting_platform.serving.api import create_app


class FakeForecaster:
    model_name = "NHITS"
    model_uri = "models:/fake/Production"

    def __init__(self) -> None:
        ds = pd.date_range("2017-08-16", periods=28)
        self.table = pd.DataFrame(
            {
                "unique_id": "1_GROCERY I",
                "ds": ds,
                "model_name": "NHITS",
                "yhat": 100.0,
                "yhat_lo80": 80.0,
                "yhat_hi80": 120.0,
            }
        )[FORECAST_COLUMNS]

    def forecast(self, unique_id: str, horizon: int) -> pd.DataFrame:
        return self.table[self.table["unique_id"] == unique_id].head(horizon)


@pytest.fixture
def leaderboard(tmp_path: Path) -> Path:
    path = tmp_path / "leaderboard.csv"
    pd.DataFrame(
        {
            "model": ["NHITS", "NHITS", "AutoETS"],
            "fold": [1, 2, 1],
            "wape": [0.10, 0.20, 0.5],
            "interval_coverage": [0.6, 0.7, 0.5],
            "pinball_loss": [20.0, 30.0, 99.0],
            "business_cost": [1e6, 3e6, 9e9],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def client(leaderboard: Path):
    with TestClient(create_app(FakeForecaster, leaderboard)) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_forecast_success(client: TestClient) -> None:
    r = client.post("/forecast", json={"store_nbr": 1, "family": "GROCERY I", "horizon": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["unique_id"] == "1_GROCERY I" and body["model_name"] == "NHITS"
    assert len(body["forecast"]) == 7
    assert body["forecast"][0] == {
        "ds": "2017-08-16",
        "yhat": 100.0,
        "yhat_lo80": 80.0,
        "yhat_hi80": 120.0,
    }


def test_forecast_default_horizon_is_28(client: TestClient) -> None:
    r = client.post("/forecast", json={"store_nbr": 1, "family": "GROCERY I"})
    assert r.status_code == 200 and len(r.json()["forecast"]) == 28


def test_forecast_unknown_combination_404(client: TestClient) -> None:
    r = client.post("/forecast", json={"store_nbr": 99, "family": "SPACESHIPS"})
    assert r.status_code == 404
    assert "Unknown store/family" in r.json()["detail"]


@pytest.mark.parametrize("horizon", [0, 29])
def test_forecast_rejects_out_of_range_horizon(client: TestClient, horizon: int) -> None:
    r = client.post("/forecast", json={"store_nbr": 1, "family": "GROCERY I", "horizon": horizon})
    assert r.status_code == 422


def test_models_reports_champion_backtest(client: TestClient) -> None:
    body = client.get("/models").json()
    assert body["champion"] == "NHITS"
    assert body["backtest"]["folds"] == 2
    assert body["backtest"]["mean_wape"] == pytest.approx(0.15)


def test_model_load_failure_returns_503_but_health_stays_up(leaderboard: Path) -> None:
    def broken():
        raise RuntimeError("no registry")

    with TestClient(create_app(broken, leaderboard)) as c:
        assert c.get("/health").status_code == 200
        r = c.post("/forecast", json={"store_nbr": 1, "family": "GROCERY I"})
        assert r.status_code == 503


def test_openapi_docs_render(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/forecast" in client.get("/openapi.json").json()["paths"]


def test_api_never_loads_torch_in_tests() -> None:
    assert "torch" not in sys.modules
