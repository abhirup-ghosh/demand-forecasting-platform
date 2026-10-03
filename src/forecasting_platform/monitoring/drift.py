"""Lightweight data-drift report (PLAN.md P0.11).

Wraps Evidently's ``DataDriftPreset`` to compare a **reference window** (the earliest backtest
fold's training period — i.e. everything a model trained for fold 5 would have seen, see P0.5)
against the **most recent 28-day window** (fold 1's test period), over the P0.4 feature set.

This is a one-shot, point-in-time check, not continuous monitoring (that's P1.2). Its purpose is
to demonstrate train/serving-skew awareness: if the feature distributions a model would be
*trained* on look different from the most recent 28 days, that's a real signal that the model may
be stale — not a bug, but something worth flagging before trusting old backtest numbers.

Columns checked (``FEATURE_COLUMNS`` in ``features.engineering``, minus identifiers and the
target): lags/rolling stats and ``oil_price`` as numerical; ``dow``, ``month``, the three holiday
flags, ``onpromotion``, ``store_type``, ``store_cluster``, ``family`` as categorical — categorical
because they're small, discrete sets (days, months, store groups), even though some are ints/bools.
"""

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset

NUMERICAL_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "oil_price",
]
CATEGORICAL_COLUMNS = [
    "dow",
    "month",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "onpromotion",
    "store_type",
    "store_cluster",
    "family",
]
DRIFT_SHARE = 0.5  # share of columns that must drift for DriftedColumnsCount to flag overall drift
MAX_ROWS = 50_000  # cap per dataset: K-S/Z-tests don't need millions of rows, and stay fast


def _prepare(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    cols = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    out = df[cols].dropna()  # lag/rolling warm-up rows have no signal to compare
    if len(out) > MAX_ROWS:
        out = out.sample(MAX_ROWS, random_state=seed)
    return out


def build_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, seed: int = 42
) -> Snapshot:
    """Run the drift preset; both frames need every column in ``NUMERICAL_COLUMNS`` +
    ``CATEGORICAL_COLUMNS`` (the P0.4 feature frame has them all)."""
    data_definition = DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS
    )
    reference = Dataset.from_pandas(_prepare(reference_df, seed), data_definition=data_definition)
    current = Dataset.from_pandas(_prepare(current_df, seed), data_definition=data_definition)
    report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_SHARE)])
    return report.run(current_data=current, reference_data=reference)


def _column_verdict(value: float, threshold: float, method: str) -> tuple[bool, float]:
    """(drifted, severity) for one column's stattest result.

    Evidently's per-column drift tests come in two families with **opposite** drift directions,
    distinguishable only by the method name: p-value tests (K-S, Z-test, chi-square, ...) flag
    drift when ``value < threshold`` (a small p-value rejects "same distribution"); distance
    tests (Jensen-Shannon, Wasserstein, PSI, ...) flag drift when ``value > threshold`` (a large
    distance means the distributions differ). Treating every value as a p-value — as an earlier
    version of this function did — silently inverts the verdict for every distance-based column.
    ``severity`` is how many multiples past the threshold the value sits, in each method's own
    direction, so columns using different methods can still be ranked on one "most drifted" list.
    """
    is_p_value = method.endswith("p_value") or method.endswith("p-value")
    if is_p_value:
        return bool(value < threshold), threshold / max(value, 1e-300)
    return bool(value > threshold), value / threshold if threshold else float(value > 0)


def summarize(snapshot: Snapshot) -> dict:
    """Top-line numbers for the dashboard/CLI: overall drift plus the per-column results."""
    result = snapshot.dict()
    metrics = {m["metric_name"]: m for m in result["metrics"]}
    overall = next(v for k, v in metrics.items() if k.startswith("DriftedColumnsCount"))
    columns = []
    for name, m in metrics.items():
        if not name.startswith("ValueDrift"):
            continue
        value, threshold, method = m["value"], m["config"]["threshold"], m["config"]["method"]
        drifted, severity = _column_verdict(value, threshold, method)
        columns.append(
            {
                "column": m["config"]["column"],
                "drifted": drifted,
                "score": value,
                "threshold": threshold,
                "severity": severity,
                "method": method,
            }
        )
    columns.sort(key=lambda c: (not c["drifted"], -c["severity"]))
    n_drifted = int(overall["value"]["count"])
    # Cross-check our own per-column direction logic against Evidently's own overall count — if a
    # future Evidently release renames/reorders stattest methods, this catches the mismatch instead
    # of silently mislabeling columns (see _column_verdict's docstring for the bug this guards).
    recomputed = sum(c["drifted"] for c in columns)
    if recomputed != n_drifted:
        raise AssertionError(
            f"per-column drift verdicts ({recomputed} drifted) don't match Evidently's own "
            f"DriftedColumnsCount ({n_drifted}) — _column_verdict's p-value/distance direction "
            "logic may be stale for this Evidently version."
        )
    return {
        "n_columns_checked": len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS),
        "n_columns_drifted": n_drifted,
        "share_columns_drifted": float(overall["value"]["share"]),
        "drift_share_threshold": DRIFT_SHARE,
        "dataset_drift_detected": overall["value"]["share"] >= DRIFT_SHARE,
        "columns": columns,
    }
