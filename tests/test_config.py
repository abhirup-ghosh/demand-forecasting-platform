"""Tests for PROJECT_ROOT resolution (PLAN.md P0.12 Outcome).

Regression coverage for a real bug: a non-editable install (what every Dockerfile here uses, `uv
sync --no-editable`) puts config.py under `.venv/lib/python3.X/site-packages/...` instead of
`<repo>/src/forecasting_platform/`, where the ``parents[2]`` dev heuristic silently resolves to a
path inside the venv instead of the repo root — breaking every PROJECT_ROOT-derived default
(data dirs, results paths, drift report paths, ...) with no error, just wrong paths.
"""

import importlib
import os

import forecasting_platform.config as config_module

ENV_VAR = "FORECASTING_PLATFORM_ROOT"


def test_project_root_honors_env_override(tmp_path) -> None:
    """FORECASTING_PLATFORM_ROOT, when set, wins over the parents[2] dev heuristic.

    Sets/clears the real env var and reloads the module directly (rather than via monkeypatch)
    so teardown order is explicit: the var must be cleared *before* the restoring reload, or the
    module stays stuck pointing at ``tmp_path`` for every later test in this process.
    """
    assert ENV_VAR not in os.environ, "a previous test left this env var set"
    os.environ[ENV_VAR] = str(tmp_path)
    try:
        reloaded = importlib.reload(config_module)
        assert reloaded.PROJECT_ROOT == tmp_path
        assert reloaded.settings.DATA_PROCESSED_DIR == tmp_path / "data" / "processed"
        assert reloaded.settings.LEADERBOARD_PATH == tmp_path / "results" / "leaderboard.csv"
    finally:
        del os.environ[ENV_VAR]
        importlib.reload(config_module)  # restore the normal (dev heuristic) state


def test_project_root_falls_back_to_file_location_heuristic() -> None:
    """Without the env override (the normal dev/test case), PROJECT_ROOT is the repo root — the
    directory containing pyproject.toml, three levels up from this package's config.py."""
    assert ENV_VAR not in os.environ
    assert (config_module.PROJECT_ROOT / "pyproject.toml").is_file()
