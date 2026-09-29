"""Select the champion, retrain it on the full history, register it in MLflow (PLAN.md P0.8).

Usage: ``uv run python scripts/train_final_model.py``

1. Champion = lowest mean backtest WAPE in ``results/leaderboard.csv`` (baselines excluded) —
   computed by ``select_champion``, never hardcoded.
2. Retrain it on **all** data through the last date (no holdout): this is the model that serves
   forecasts for the 28 days after the dataset ends.
3. Log it as an MLflow pyfunc (``ChampionForecaster``) registered as ``demand-forecast-champion``,
   set that version's stage to ``Production`` (as PLAN.md specifies; MLflow has deprecated stages)
   and its ``production`` alias (the modern equivalent).

Only champions with a packager in ``PACKAGERS`` can be registered; if the rule ever selects another
tier the script stops with a clear message instead of silently registering something else.
Runs torch in this process — it never imports the LightGBM tier (see ``isolation.py``).
"""

import json
import logging
import os
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import ModelSignature, infer_signature  # noqa: E402
from mlflow.types import ColSpec, Schema  # noqa: E402

from forecasting_platform.config import PROJECT_ROOT, settings  # noqa: E402
from forecasting_platform.models.champion import (  # noqa: E402
    REGISTERED_MODEL_NAME,
    ChampionForecaster,
    select_champion,
)
from forecasting_platform.models.common import fallback_forecasts, finalize  # noqa: E402
from forecasting_platform.models.deep import fit_deep, predict_deep  # noqa: E402

logger = logging.getLogger("train_final_model")
LEADERBOARD = PROJECT_ROOT / "results" / "leaderboard.csv"
EXPERIMENT = "demand-forecasting-champion"
HISTORY_TAIL_DAYS = 112  # >= NHITS input_size (56); enough context for batch scoring


def package_nhits(history: pd.DataFrame, out_dir: Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Fit NHITS on the full history; write artifacts; return (artifact paths, forecast sample)."""
    horizon = settings.FORECAST_HORIZON
    nf, split = fit_deep(history, horizon)
    nf.save(str(out_dir / "nf_model"), save_dataset=False, overwrite=True)
    context = split.fit_df.groupby("unique_id", sort=False).tail(HISTORY_TAIL_DAYS)
    context[["unique_id", "ds", "y"]].to_parquet(out_dir / "history.parquet", index=False)
    finalize(fallback_forecasts(history, split, horizon, "NHITS")).to_parquet(
        out_dir / "fallback.parquet", index=False
    )
    logger.info("NHITS retrained on full history: %s", split.summary())
    artifacts = {
        "nf_model": str(out_dir / "nf_model"),
        "history": str(out_dir / "history.parquet"),
        "fallback": str(out_dir / "fallback.parquet"),
    }
    return artifacts, predict_deep(nf).head(56)


PACKAGERS = {"NHITS": (package_nhits, ["neuralforecast", "torch"])}


def main() -> None:
    board = pd.read_csv(LEADERBOARD)
    champion, table = select_champion(board)
    print("Champion selection (lowest mean WAPE, baselines excluded):")
    print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(f"-> champion: {champion}\n")
    if champion not in PACKAGERS:
        raise SystemExit(
            f"The rule selected {champion!r}, but no packager exists for it in "
            f"scripts/train_final_model.py (available: {sorted(PACKAGERS)}). Add one first."
        )
    packager, extra_reqs = PACKAGERS[champion]

    features = pd.read_parquet(settings.DATA_PROCESSED_DIR / "features.parquet")
    history = features[["unique_id", "ds", "y"]]
    train_end = history["ds"].max().date()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"champion-{champion}"):
        out_dir = Path(tmp)
        artifacts, sample = packager(history, out_dir)
        stats = table.loc[champion]
        metadata = {
            "champion": champion,
            "selection_rule": "lowest mean backtest WAPE across all folds, baselines excluded",
            "train_end": str(train_end),
            "horizon": settings.FORECAST_HORIZON,
            "backtest": {k: float(v) for k, v in stats.items()},
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        artifacts["metadata"] = str(out_dir / "metadata.json")
        table.to_csv(out_dir / "champion_selection.csv")
        mlflow.log_artifact(str(out_dir / "champion_selection.csv"))
        mlflow.log_params(
            {
                "champion": champion,
                "train_end": str(train_end),
                "n_series": history["unique_id"].nunique(),
            }
        )
        mlflow.log_metrics({f"backtest_{k}": float(v) for k, v in stats.items()})

        # Input: unique_id (required) + horizon (optional, defaults to 28 in the pyfunc).
        inputs = Schema(
            [ColSpec("string", "unique_id"), ColSpec("long", "horizon", required=False)]
        )
        outputs = infer_signature(sample[["unique_id"]], sample).outputs
        signature = ModelSignature(inputs=inputs, outputs=outputs)
        reqs = [f"{p}=={version(p)}" for p in ["mlflow", "pandas", "pyarrow", *extra_reqs]]
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChampionForecaster(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=reqs,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    client = MlflowClient()
    mv = info.registered_model_version
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", FutureWarning
        )  # stages are deprecated but PLAN.md uses them
        client.transition_model_version_stage(
            REGISTERED_MODEL_NAME, mv, "Production", archive_existing_versions=True
        )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", mv)
    print(
        f"Registered {REGISTERED_MODEL_NAME} v{mv} ({champion}, trained through {train_end}): "
        "stage=Production, alias=production"
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("lightning", "pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    main()
