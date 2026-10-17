"""Tests for the dashboard (PLAN.md P0.10/P0.14), via Streamlit's ``AppTest`` — synthetic data only.

The main thing worth a dedicated regression test is ``PUBLIC_DEMO_MODE`` (PLAN.md Open Decision
#3): the Kaggle rules forbid shipping real sales rows to a public audience, so the public Space
must render correctly from *only* the precomputed public-demo bundle, with no raw-sales file
present at all. These fixtures deliberately point ``DATA_PROCESSED_DIR`` at a directory with no
``features.parquet`` in it — if any public-mode code path ever touched it again, the app would
throw (file not found) and ``at.exception`` would catch it; that's a sharper, more honest test
than mocking/spying on the loader function.
"""

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import forecasting_platform.config as config_module

APP_PATH = str(Path(__file__).parent.parent / "dashboard" / "app.py")
UIDS = ["1_GROCERY I", "2_BEVERAGES"]
ENV_VARS = [
    "PUBLIC_DEMO_MODE",
    "PUBLIC_DEMO_DIR",
    "DATA_PROCESSED_DIR",
    "LEADERBOARD_PATH",
    "BACKTEST_FORECAST_DIR",
]


def _leaderboard() -> pd.DataFrame:
    rows = [
        {
            "model": model,
            "fold": fold,
            "wape": wape,
            "interval_coverage": 0.7,
            "pinball_loss": 10.0,
            "business_cost": 1e6,
        }
        for model, wape in [("NHITS", 0.14), ("ChronosBolt-base", 0.15), ("SeasonalNaive", 0.17)]
        for fold in range(1, 6)
    ]
    return pd.DataFrame(rows)


def _forecast_frame(uids: list[str], ds_start: str, horizon: int, model_name: str) -> pd.DataFrame:
    ds = pd.date_range(ds_start, periods=horizon, freq="D")
    frames = [
        pd.DataFrame(
            {
                "unique_id": uid,
                "ds": ds,
                "model_name": model_name,
                "yhat": 10.0,
                "yhat_lo80": 5.0,
                "yhat_hi80": 15.0,
            }
        )
        for uid in uids
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def public_bundle(tmp_path: Path) -> Path:
    """A complete public-demo bundle: forecasts, a fold-1 tier comparison, cost totals, a
    leaderboard — and *no* raw-sales file anywhere under tmp_path."""
    bundle = tmp_path / "public_demo"
    bundle.mkdir()
    _forecast_frame(UIDS, "2017-08-16", 28, "NHITS").to_parquet(
        bundle / "forecasts.parquet", index=False
    )
    fold1 = pd.concat(
        [
            _forecast_frame(UIDS, "2017-07-19", 28, m)
            for m in ["NHITS", "ChronosBolt-base", "SeasonalNaive"]
        ],
        ignore_index=True,
    )
    fold1.to_parquet(bundle / "fold1_backtest_forecasts.parquet", index=False)
    (bundle / "business_cost_totals.json").write_text(
        json.dumps(
            {
                "NHITS": {"under": 100.0, "over": 50.0, "y": 1000.0},
                "ChronosBolt-base": {"under": 80.0, "over": 90.0, "y": 1000.0},
                "SeasonalNaive": {"under": 300.0, "over": 60.0, "y": 1000.0},
            }
        )
    )
    _leaderboard().to_csv(bundle / "leaderboard.csv", index=False)
    return bundle


@pytest.fixture
def public_env(public_bundle: Path, tmp_path: Path, monkeypatch) -> config_module:
    """Reload config with PUBLIC_DEMO_MODE on and every raw-data setting pointed at a directory
    that has no raw-sales file in it (a stand-in for the real public Space's filesystem)."""
    empty = tmp_path / "no_raw_data_here"
    empty.mkdir()
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    monkeypatch.setenv("PUBLIC_DEMO_DIR", str(public_bundle))
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(empty))
    monkeypatch.setenv("LEADERBOARD_PATH", str(public_bundle / "leaderboard.csv"))
    monkeypatch.setenv("BACKTEST_FORECAST_DIR", str(empty))
    reloaded = importlib.reload(config_module)
    assert not (empty / "features.parquet").exists()  # the invariant this test actually checks
    yield reloaded
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(config_module)


def _trace_names(element) -> list[str]:
    return [t.get("name") for t in json.loads(element.proto.spec)["data"]]


def test_public_demo_mode_renders_with_no_raw_data_present(public_env) -> None:
    assert public_env.settings.PUBLIC_DEMO_MODE is True
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Forecast Explorer",
        "Model Leaderboard",
        "Business Impact",
    ]


def test_public_demo_explorer_shows_no_actual_sales_trace(public_env) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    explorer = at.tabs[1]
    charts = explorer.get("plotly_chart")
    assert len(charts) == 2  # the forecast chart + the tier-comparison chart
    for chart in charts:
        assert "Actual sales" not in _trace_names(chart)
    forecast_names = _trace_names(charts[0])
    assert any("forecast" in n.lower() for n in forecast_names if n)
    assert any(c.value.startswith("Public demo:") for c in explorer.caption)


def test_public_demo_business_tab_uses_precomputed_totals(public_env) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    business = at.tabs[3]
    assert len(business.metric) == 3
    assert "NHITS" in business.metric[0].label  # champion from the synthetic leaderboard


def test_public_demo_dir_defaults_to_project_root_subdir() -> None:
    assert config_module.settings.PUBLIC_DEMO_MODE is False  # default, no env override
    assert config_module.settings.PUBLIC_DEMO_DIR == config_module.PROJECT_ROOT / "public_demo"
