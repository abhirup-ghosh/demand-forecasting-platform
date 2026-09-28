"""Run every model tier through every backtest fold (PLAN.md P0.7).

Usage: ``uv run python scripts/run_backtest.py [--folds 1 2] [--models LightGBM NHITS] [--force]``

For each fold (P0.5) x model (P0.6a-e; LightGBM both raw and log1p for Open Decision #4):

1. fit/predict on the fold's training data (forecasts cached to ``results/forecasts/`` so an
   interrupted run resumes; ``--force`` recomputes),
2. score with the shared metrics + business cost (``evaluation.metrics.score_forecast``),
3. write one row per (model, fold) to ``results/leaderboard.csv`` and log the same as an MLflow
   run (params, metrics, and a forecast-vs-actual plot for representative series).

Must run as a script (``if __name__ == "__main__"``): statsforecast uses worker processes.
"""

import argparse
import logging
import os
import time
import warnings
from collections.abc import Callable

import matplotlib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence MLflow's per-run advisory log
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting_platform.config import PROJECT_ROOT, settings  # noqa: E402
from forecasting_platform.evaluation.backtest import (  # noqa: E402
    Fold,
    generate_cv_folds,
    split_train_test,
)
from forecasting_platform.evaluation.metrics import score_forecast  # noqa: E402
from forecasting_platform.models.baseline import naive, seasonal_naive  # noqa: E402
from forecasting_platform.models.deep import forecast_deep  # noqa: E402
from forecasting_platform.models.foundation import forecast_foundation  # noqa: E402
from forecasting_platform.models.ml import forecast_ml  # noqa: E402
from forecasting_platform.models.statistical import forecast_statistical  # noqa: E402

logger = logging.getLogger("run_backtest")

RESULTS_DIR = PROJECT_ROOT / "results"
FORECAST_DIR = RESULTS_DIR / "forecasts"
LEADERBOARD = RESULTS_DIR / "leaderboard.csv"
EXPERIMENT = "demand-forecasting-backtest"
Y_COLS = ["unique_id", "ds", "y"]

# name -> (callable(train, test) -> forecast frame, model names it returns)
Runner = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
TIERS: dict[str, tuple[Runner, list[str]]] = {
    "baseline": (
        lambda tr, te: pd.concat([naive(tr[Y_COLS], 28), seasonal_naive(tr[Y_COLS], 28)]),
        ["Naive", "SeasonalNaive"],
    ),
    "statistical": (lambda tr, te: forecast_statistical(tr[Y_COLS]), ["AutoARIMA", "AutoETS"]),
    "ml": (lambda tr, te: forecast_ml(tr, te), ["LightGBM"]),
    "ml_log1p": (lambda tr, te: forecast_ml(tr, te, log_target=True), ["LightGBM_log1p"]),
    "deep": (lambda tr, te: forecast_deep(tr[Y_COLS]), ["NHITS"]),
    "foundation": (lambda tr, te: forecast_foundation(tr[Y_COLS]), ["ChronosBolt-base"]),
}


def volume_bands(train: pd.DataFrame) -> pd.Series:
    """Terciles of non-zero series by training volume: unique_id -> top/middle/bottom."""
    volume = train.groupby("unique_id")["y"].sum()
    volume = volume[volume > 0]
    return pd.qcut(volume.rank(method="first"), 3, labels=["bottom", "middle", "top"]).astype(str)


def plot_examples(train, test, fc, ids, title) -> plt.Figure:
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3.2 * len(ids)), layout="constrained")
    for ax, uid in zip(axes, ids, strict=True):
        hist = train[train["unique_id"] == uid].tail(56)
        act = test[test["unique_id"] == uid]
        f = fc[fc["unique_id"] == uid]
        ax.plot(hist["ds"], hist["y"], color="#b5b4ae", lw=1.5, label="history")
        ax.plot(act["ds"], act["y"], color="#0b0b0b", lw=1.5, label="actual")
        ax.plot(f["ds"], f["yhat"], color="#2a78d6", lw=2, label="forecast")
        ax.fill_between(
            f["ds"],
            f["yhat_lo80"],
            f["yhat_hi80"],
            color="#2a78d6",
            alpha=0.18,
            label="80% interval",
        )
        ax.set_title(uid, loc="left", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def run_tier(tier: str, fold: Fold, train, test, force: bool) -> tuple[pd.DataFrame, float]:
    path = FORECAST_DIR / f"{tier}_fold{fold.fold_id}.parquet"
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        return cached, float(cached.attrs.get("runtime_s", float("nan")))
    fn, _ = TIERS[tier]
    start = time.perf_counter()
    fc = fn(train, test)
    runtime = time.perf_counter() - start
    fc.attrs = {"runtime_s": runtime}
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path, index=False)
    return fc, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", type=int, nargs="*", help="fold ids (default: all)")
    parser.add_argument("--models", nargs="*", choices=list(TIERS), help="tiers (default: all)")
    parser.add_argument("--force", action="store_true", help="recompute cached forecasts")
    args = parser.parse_args()

    features_path = settings.DATA_PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found — run `make features` first.")
    features = pd.read_parquet(features_path)
    folds = [
        f for f in generate_cv_folds(features["ds"]) if not args.folds or f.fold_id in args.folds
    ]
    tiers = args.models or list(TIERS)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    rows = []
    for fold in folds:
        train, test = split_train_test(features, fold)
        active = set(train.groupby("unique_id")["y"].sum().loc[lambda s: s > 0].index)
        bands = volume_bands(train)
        # Representative series for plots: the largest-volume series and a middle-band one.
        examples = [
            train.groupby("unique_id")["y"].sum().idxmax(),
            bands[bands == "middle"].index[0],
        ]
        for tier in tiers:
            fc, runtime = run_tier(tier, fold, train, test, args.force)
            logger.info("fold %d %-12s %.0fs", fold.fold_id, tier, runtime)
            for model_name, g in fc.groupby("model_name"):
                scores = score_forecast(test[Y_COLS], g, active, bands)
                row = {
                    "model": model_name,
                    "tier": tier,
                    "fold": fold.fold_id,
                    "test_start": fold.test_start_date.date(),
                    "test_end": fold.test_end_date.date(),
                    "n_series": g["unique_id"].nunique(),
                    **scores,
                    "runtime_s": runtime,
                }
                rows.append(row)
                logger.info(
                    "   %-18s WAPE %.1f%%  coverage %.1f%%",
                    model_name,
                    100 * scores["wape"],
                    100 * scores["interval_coverage"],
                )
                with mlflow.start_run(run_name=f"{model_name}-fold{fold.fold_id}"):
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "tier": tier,
                            "fold": fold.fold_id,
                            "test_start": str(row["test_start"]),
                            "test_end": str(row["test_end"]),
                        }
                    )
                    mlflow.log_metrics(
                        {k: v for k, v in scores.items() if pd.notna(v)} | {"runtime_s": runtime}
                    )
                    fig = plot_examples(
                        train, test, g, examples, f"{model_name} — fold {fold.fold_id}"
                    )
                    mlflow.log_figure(fig, "forecast_vs_actual.png")
                    plt.close(fig)

    board = pd.DataFrame(rows)
    if LEADERBOARD.exists() and (args.folds or args.models):  # partial run: merge into existing
        old = pd.read_csv(LEADERBOARD)
        keep = ~old.set_index(["model", "fold"]).index.isin(
            board.set_index(["model", "fold"]).index
        )
        board = pd.concat([old[keep], board], ignore_index=True)
    board = board.sort_values(["fold", "wape"], ignore_index=True)
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(LEADERBOARD, index=False)
    summary = board.groupby("model")[["wape", "interval_coverage", "business_cost"]].mean()
    print(f"\nWrote {len(board)} rows to {LEADERBOARD}\nMean over folds:")
    print(summary.sort_values("wape").to_string(float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("lightning", "pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    main()
