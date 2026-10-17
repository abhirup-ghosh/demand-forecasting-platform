"""Export the forecasts-and-aggregates-only bundle for the public HF Space (PLAN.md P0.14).

Usage: ``uv run python scripts/export_public_demo.py``

Why this exists: the Kaggle competition rules (section 7, "Competition Data") forbid
transmitting, publishing or redistributing the dataset — including derived copies containing real
sales rows — "to any party not participating in the Competition," and restrict its use to
"Competition Use and Academic, Non-Commercial Use Only." A public-facing demo therefore can't ship
anything with a real per-row sales value in it, in the model or in the dashboard (see PLAN.md Open
Decision #3). Everything this script writes to ``settings.PUBLIC_DEMO_DIR`` is either the
platform's own output (model forecasts — numbers nobody else's data rights apply to) or an
aggregate statistic (a sum/count across many rows, not a row itself) — never a raw sales figure
for a specific store/family/day.

Writes, all under ``settings.PUBLIC_DEMO_DIR`` (default ``public_demo/``, gitignored in *this*
repo — committed instead into the HF Space's own separate git repo, see ``space/README.md``):

- ``forecasts.parquet``    — the registered champion's forecast for every series, full horizon.
  Columns: unique_id, ds, model_name, yhat, yhat_lo80, yhat_hi80. No ``y`` column at all.
- ``fold1_backtest_forecasts.parquet`` — every tier's own fold-1 (final holdout) forecast,
  concatenated. Lets the public Forecast Explorer still show a multi-tier comparison chart,
  without any ground truth overlaid (each tier's forecast is its own output, not a data copy).
- ``business_cost_totals.json`` — per model: total under-forecast units, total over-forecast
  units, total actual volume (all backtest folds, summed). These three numbers per model are
  already aggregated; the Business Impact tab's sliders only ever multiply them by a cost rate
  (linear), so the identical UI works from just these sums — it never needs the underlying rows.
- ``leaderboard.csv`` — copied as-is; already aggregate (one row per model/fold, metrics only).
- ``drift_summary.json`` — copied as-is if present; already aggregate (per-column drift scores).

Run this locally, where the real Kaggle data legitimately lives (a competition participant's own
machine) — never inside the public Space itself.
"""

import json
import shutil
import warnings

import mlflow
import pandas as pd

from forecasting_platform.config import settings

TIER_FILES = ["baseline", "statistical", "ml", "ml_log1p", "deep", "foundation"]


def export_forecasts(out_dir) -> int:
    """The champion's full-horizon forecast for every series — model output, not data."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(settings.CHAMPION_MODEL_URI)
    features = pd.read_parquet(
        settings.DATA_PROCESSED_DIR / "features.parquet", columns=["unique_id"]
    )
    ids = features["unique_id"].drop_duplicates().reset_index(drop=True)
    forecasts = model.predict(pd.DataFrame({"unique_id": ids}))
    forecasts.to_parquet(out_dir / "forecasts.parquet", index=False)
    return len(forecasts)


def export_fold1_backtest(out_dir) -> int:
    """Every tier's own fold-1 forecast — each tier's output, never ground truth."""
    frames = [
        pd.read_parquet(p)
        for tier in TIER_FILES
        for p in settings.BACKTEST_FORECAST_DIR.glob(f"{tier}_fold1.parquet")
    ]
    if not frames:
        print(
            "  (no results/forecasts/*_fold1.parquet found — skipping, run `make backtest` first)"
        )
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_dir / "fold1_backtest_forecasts.parquet", index=False)
    return len(combined)


def export_business_cost_totals(out_dir) -> dict:
    """Per-model AGGREGATE under/over-forecast unit totals — sums, not rows.

    Joins all-fold backtest forecasts with real actuals *once, locally*, then keeps only the
    resulting per-model sums — mirrors dashboard.app.error_totals(), which does this join live
    (safe there: that code only ever runs where the real data already legitimately is).
    """
    forecast_frames = []
    for tier in TIER_FILES:
        for path in sorted(settings.BACKTEST_FORECAST_DIR.glob(f"{tier}_fold*.parquet")):
            forecast_frames.append(pd.read_parquet(path))
    if not forecast_frames:
        print("  (no backtest forecasts found — skipping business_cost_totals.json)")
        return {}
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    actuals = pd.read_parquet(
        settings.DATA_PROCESSED_DIR / "features.parquet", columns=["unique_id", "ds", "y"]
    )
    joined = forecasts.merge(actuals, on=["unique_id", "ds"], how="inner")
    err = joined["y"] - joined["yhat"]
    totals = (
        joined.assign(under=err.clip(lower=0), over=(-err).clip(lower=0))
        .groupby("model_name")[["under", "over", "y"]]
        .sum()
    )
    payload = {
        model: {"under": float(r["under"]), "over": float(r["over"]), "y": float(r["y"])}
        for model, r in totals.iterrows()
    }
    (out_dir / "business_cost_totals.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    out_dir = settings.PUBLIC_DEMO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_forecasts = export_forecasts(out_dir)
    print(f"forecasts.parquet: {n_forecasts:,} rows")

    n_fold1 = export_fold1_backtest(out_dir)
    if n_fold1:
        print(f"fold1_backtest_forecasts.parquet: {n_fold1:,} rows")

    totals = export_business_cost_totals(out_dir)
    if totals:
        print(f"business_cost_totals.json: {len(totals)} models")

    if settings.LEADERBOARD_PATH.exists():
        shutil.copy(settings.LEADERBOARD_PATH, out_dir / "leaderboard.csv")
        print("leaderboard.csv: copied")
    if settings.DRIFT_SUMMARY_PATH.exists():
        shutil.copy(settings.DRIFT_SUMMARY_PATH, out_dir / "drift_summary.json")
        print("drift_summary.json: copied")

    # Self-check: fail loudly rather than silently ship a row with a real sales value.
    exported = pd.read_parquet(out_dir / "forecasts.parquet")
    assert "y" not in exported.columns, "forecasts.parquet must never carry a 'y' (actuals) column"
    print(
        f"\nWrote {out_dir}/ — no raw sales rows in any exported file (checked forecasts.parquet)."
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
