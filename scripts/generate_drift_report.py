"""Generate the Evidently data-drift report (PLAN.md P0.11).

Usage: ``uv run python scripts/generate_drift_report.py``

Reference window = fold 5's (the oldest backtest fold's) training period — everything up to and
including its ``train_end_date``, i.e. the full history a model trained for that fold would have
seen. Current window = fold 1's test period — the most recent 28 days in the dataset.

Writes ``reports/generated/drift_report.html`` (the full interactive Evidently report) and
``reports/generated/drift_summary.json`` (the top-line numbers, for the dashboard/CLI to read
without recomputing). Both are gitignored — regenerable from ``data/processed/features.parquet``.
"""

import json
import warnings

from forecasting_platform.config import settings
from forecasting_platform.evaluation.backtest import generate_cv_folds
from forecasting_platform.monitoring.drift import build_drift_report, summarize


def main() -> None:
    import pandas as pd

    features_path = settings.DATA_PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found — run `make features` first.")
    features = pd.read_parquet(features_path)

    folds = generate_cv_folds(features["ds"])
    oldest, newest = folds[-1], folds[0]
    reference = features[features["ds"] <= oldest.train_end_date]
    current = features[
        (features["ds"] >= newest.test_start_date) & (features["ds"] <= newest.test_end_date)
    ]
    print(
        f"Reference window: up to {oldest.train_end_date.date()} ({len(reference):,} rows)\n"
        f"Current window:   {newest.test_start_date.date()}..{newest.test_end_date.date()} "
        f"({len(current):,} rows)"
    )

    snapshot = build_drift_report(reference, current)
    html_path = settings.DRIFT_REPORT_PATH
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    summary = summarize(snapshot)
    summary["reference_window"] = {
        "start": str(reference["ds"].min().date()),
        "end": str(oldest.train_end_date.date()),
    }
    summary["current_window"] = {
        "start": str(newest.test_start_date.date()),
        "end": str(newest.test_end_date.date()),
    }
    settings.DRIFT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(
        f"\n{summary['n_columns_drifted']}/{summary['n_columns_checked']} columns drifted "
        f"({summary['share_columns_drifted']:.0%}, threshold "
        f"{summary['drift_share_threshold']:.0%}) "
        f"-> dataset_drift_detected={summary['dataset_drift_detected']}"
    )
    print("Most-drifted columns first (score vs this method's own threshold):")
    for c in summary["columns"][:8]:
        flag = "DRIFTED" if c["drifted"] else "ok"
        print(
            f"  [{flag:7s}] {c['column']:20s} score={c['score']:.4f} "
            f"threshold={c['threshold']}  ({c['method']})"
        )
    print(f"\nWrote {html_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
