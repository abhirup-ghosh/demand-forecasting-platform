"""Demand Forecasting Platform — interactive dashboard (PLAN.md P0.10).

Run: ``uv run streamlit run dashboard/app.py``

Tabs: Overview (EDA findings) · Forecast Explorer · Model Leaderboard · Business Impact.

``settings.DASHBOARD_STANDALONE_MODE`` picks where champion forecasts come from: in-process via the
MLflow-registered model (one container, e.g. Hugging Face Spaces) or the FastAPI service at
``settings.API_URL`` (docker-compose). This module never imports the LightGBM tier, so loading the
PyTorch-based champion here is safe (see ``forecasting_platform.isolation``).
"""

import importlib.util
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Standalone entry script: if the package isn't importable (e.g. macOS flagged the editable-install
# .pth "hidden", which Python 3.13 skips), fall back to the repo's src/. No-op when installed.
if importlib.util.find_spec("forecasting_platform") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_platform.config import settings  # noqa: E402
from forecasting_platform.models.champion import BASELINES, select_champion  # noqa: E402

# Validated categorical palette (docs/eda-findings.md charts) — fixed slot order, never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACTUAL = "#8a8984"  # neutral that reads on light and dark themes
TIER_FILES = ["baseline", "statistical", "ml", "ml_log1p", "deep", "foundation"]

st.set_page_config(page_title="Demand Forecasting Platform", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data loading (cached)
@st.cache_data(show_spinner="Loading sales history…")
def load_history() -> pd.DataFrame:
    path = settings.DATA_PROCESSED_DIR / "features.parquet"
    return pd.read_parquet(path, columns=["unique_id", "ds", "y"])


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv(settings.LEADERBOARD_PATH)


@st.cache_data(show_spinner="Loading backtest forecasts…")
def load_backtest_forecasts() -> pd.DataFrame:
    frames = []
    for tier in TIER_FILES:
        for path in sorted(settings.BACKTEST_FORECAST_DIR.glob(f"{tier}_fold*.parquet")):
            fold = int(re.search(r"fold(\d+)", path.stem).group(1))
            frames.append(pd.read_parquet(path).assign(fold=fold))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner="Scoring backtest errors…")
def error_totals() -> pd.DataFrame:
    """Per model: total under- and over-forecast units across all folds (cost is linear in both)."""
    fc = load_backtest_forecasts()
    actuals = load_history()
    j = fc.merge(actuals, on=["unique_id", "ds"], how="inner")
    err = j["y"] - j["yhat"]
    return (
        j.assign(under=err.clip(lower=0), over=(-err).clip(lower=0))
        .groupby("model_name")[["under", "over", "y"]]
        .sum()
    )


@st.cache_resource(show_spinner="Loading the champion model…")
def load_local_forecaster():
    from forecasting_platform.serving.api import MlflowForecaster

    return MlflowForecaster()


def champion_forecast(unique_id: str, horizon: int) -> tuple[pd.DataFrame, str]:
    """(forecast rows, model name) from the in-process model or the API, per settings."""
    if settings.DASHBOARD_STANDALONE_MODE:
        model = load_local_forecaster()
        return model.forecast(unique_id, horizon), model.model_name
    store, family = unique_id.split("_", 1)
    body = json.dumps({"store_nbr": int(store), "family": family, "horizon": horizon}).encode()
    req = urllib.request.Request(
        f"{settings.API_URL}/forecast", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    fc = pd.DataFrame(payload["forecast"]).assign(ds=lambda d: pd.to_datetime(d["ds"]))
    return fc, payload["model_name"]


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------- tabs
def tab_overview() -> None:
    st.markdown(
        "Daily unit sales for **1,782 store × product-family series** (54 stores, 33 families, "
        "Ecuador, 2013-01-01 → 2017-08-15). Below are the eight findings from the exploratory "
        "analysis that shaped the modelling choices — full write-up in `docs/eda-findings.md`."
    )
    md = (settings.DOCS_DIR / "eda-findings.md").read_text()
    sections = re.split(r"\n(?=## \d\. )", md)[1:]
    for section in sections:
        title, _, body = section.partition("\n")
        body = body.split("\n## Implications", 1)[0]
        images = re.findall(r"!\[[^\]]*\]\(\./(figures/[^)]+)\)", body)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)\n?", "", body).strip()
        with st.expander(title.lstrip("# "), expanded=title.startswith("## 1.")):
            for image in images:
                st.image(str(settings.DOCS_DIR / image), width="stretch")
            st.markdown(text)


def tab_explorer() -> None:
    history = load_history()
    ids = history["unique_id"].drop_duplicates()
    stores = sorted(ids.str.split("_", n=1).str[0].astype(int).unique())
    families = sorted(ids.str.split("_", n=1).str[1].unique())
    c1, c2, c3 = st.columns([1, 2, 1])
    store = c1.selectbox("Store", stores, index=stores.index(44) if 44 in stores else 0)
    family = c2.selectbox(
        "Product family",
        families,
        index=families.index("GROCERY I") if "GROCERY I" in families else 0,
    )
    horizon = c3.slider("Horizon (days)", 1, settings.FORECAST_HORIZON, settings.FORECAST_HORIZON)
    uid = f"{store}_{family}"
    series = history[history["unique_id"] == uid]

    try:
        fc, model_name = champion_forecast(uid, horizon)
    except (urllib.error.URLError, OSError, KeyError) as exc:
        source = "standalone" if settings.DASHBOARD_STANDALONE_MODE else settings.API_URL
        st.error(f"Could not get a forecast ({source}): {exc}")
        return

    st.subheader(f"Next {horizon} days — {model_name} (champion)")
    recent = series[series["ds"] > series["ds"].max() - pd.Timedelta(days=90)]
    fig = go.Figure()
    fig.add_scatter(
        x=recent["ds"], y=recent["y"], name="Actual sales", line=dict(color=ACTUAL, width=2)
    )
    fig.add_scatter(
        x=fc["ds"], y=fc["yhat_hi80"], line=dict(width=0), showlegend=False, hoverinfo="skip"
    )
    fig.add_scatter(
        x=fc["ds"],
        y=fc["yhat_lo80"],
        fill="tonexty",
        fillcolor="rgba(42,120,214,0.18)",
        line=dict(width=0),
        name="80% interval",
    )
    fig.add_scatter(
        x=fc["ds"],
        y=fc["yhat"],
        name=f"{model_name} forecast",
        line=dict(color=PALETTE[0], width=2),
    )
    fig.update_yaxes(title="Units sold")
    st.plotly_chart(style(fig), width="stretch")
    if series["y"].sum() == 0:
        st.info("This series has never recorded a sale, so every tier forecasts zero.")

    st.subheader("How the tiers compared on the final holdout (fold 1, 2017-07-19 → 08-15)")
    backtest = load_backtest_forecasts()
    fold1 = backtest[(backtest["fold"] == 1) & (backtest["unique_id"] == uid)]
    if fold1.empty:
        st.caption("No backtest forecasts found (run `scripts/run_backtest.py`).")
        return
    models = sorted(fold1["model_name"].unique())
    default = [m for m in [model_name, "ChronosBolt-base", "SeasonalNaive"] if m in models]
    chosen = st.multiselect("Tiers to overlay", models, default=default)
    window = series[series["ds"] >= pd.Timestamp("2017-06-21")]
    fig2 = go.Figure()
    fig2.add_scatter(
        x=window["ds"], y=window["y"], name="Actual sales", line=dict(color=ACTUAL, width=2)
    )
    for i, m in enumerate(chosen):
        g = fold1[fold1["model_name"] == m]
        fig2.add_scatter(
            x=g["ds"], y=g["yhat"], name=m, line=dict(color=PALETTE[i % len(PALETTE)], width=2)
        )
    fig2.add_vline(x=pd.Timestamp("2017-07-19"), line=dict(color=ACTUAL, dash="dot", width=1))
    fig2.update_yaxes(title="Units sold")
    st.plotly_chart(style(fig2, 380), width="stretch")


def tab_leaderboard() -> None:
    board = load_leaderboard()
    champion, table = select_champion(board)
    summary = (
        board.groupby("model")
        .agg(
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
            coverage_80=("interval_coverage", "mean"),
            pinball=("pinball_loss", "mean"),
            business_cost=("business_cost", "mean"),
        )
        .sort_values("mean_wape")
    )
    st.markdown(
        f"Five rolling 28-day backtest folds (2017-03-29 → 2017-08-15). Champion rule: lowest mean "
        f"WAPE, baselines excluded → **{champion}**. The 80% coverage column checks whether each "
        "model's prediction intervals are honest (target: 80%)."
    )
    colors = [
        PALETTE[0] if m == champion else ("#b5b4ae" if m in BASELINES else "#7fa9df")
        for m in summary.index
    ]
    fig = go.Figure(
        go.Bar(
            x=summary["mean_wape"] * 100,
            y=summary.index,
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data", array=summary["std_wape"] * 100, color="#8a8984", thickness=1
            ),
            text=[f"{v:.1f}%" for v in summary["mean_wape"] * 100],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}% WAPE<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Mean WAPE across folds (%) — lower is better; bars show ±1 sd")
    st.plotly_chart(style(fig, 360), width="stretch")

    st.markdown("**Per-model summary** (click a column header to sort)")
    st.dataframe(
        summary.assign(
            mean_wape=summary["mean_wape"] * 100,
            std_wape=summary["std_wape"] * 100,
            coverage_80=summary["coverage_80"] * 100,
        ),
        column_config={
            "mean_wape": st.column_config.NumberColumn("Mean WAPE %", format="%.2f"),
            "std_wape": st.column_config.NumberColumn("± sd", format="%.2f"),
            "coverage_80": st.column_config.NumberColumn("80% coverage %", format="%.1f"),
            "pinball": st.column_config.NumberColumn("Pinball loss", format="%.1f"),
            "business_cost": st.column_config.NumberColumn("3:1 cost (per fold)", format="%,.0f"),
        },
        width="stretch",
    )
    with st.expander("All (model, fold) rows"):
        st.dataframe(board, width="stretch", hide_index=True)


def tab_business() -> None:
    board = load_leaderboard()
    champion, _ = select_champion(board)
    totals = error_totals()
    st.markdown(
        "Forecast errors cost money asymmetrically: **under-forecasting** means stock-outs and "
        "lost sales; **over-forecasting** means holding cost and waste. The 3:1 default is an "
        "*illustrative* assumption for perishable grocery — set your own ratio and see what each "
        "model's errors would have cost over the five backtest folds."
    )
    c1, c2 = st.columns(2)
    cost_under = c1.slider(
        "Cost per unit under-forecast (stock-out)",
        0.0,
        10.0,
        float(settings.COST_UNDER_PER_UNIT),
        0.5,
    )
    cost_over = c2.slider(
        "Cost per unit over-forecast (excess stock)",
        0.0,
        10.0,
        float(settings.COST_OVER_PER_UNIT),
        0.5,
    )
    costs = (cost_under * totals["under"] + cost_over * totals["over"]).sort_values()
    baseline_cost = costs.get("SeasonalNaive")

    champ = totals.loc[champion]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{champion} total cost (5 folds)", f"{costs[champion]:,.0f}")
    m2.metric(
        "Units under-forecast",
        f"{champ['under']:,.0f}",
        help="Sum over all series and days of max(0, actual − forecast)",
    )
    m3.metric(
        "Units over-forecast",
        f"{champ['over']:,.0f}",
        help="Sum over all series and days of max(0, forecast − actual)",
    )
    if baseline_cost:
        saving = 1 - costs[champion] / baseline_cost
        st.success(
            f"At this ratio the champion costs **{saving:.0%} less** than the seasonal-naive "
            'baseline ("same as last week").'
        )

    ranked = costs.drop([b for b in BASELINES if b != "SeasonalNaive"], errors="ignore")
    fig = go.Figure(
        go.Bar(
            x=ranked.values,
            y=ranked.index,
            orientation="h",
            marker_color=[PALETTE[0] if m == champion else "#7fa9df" for m in ranked.index],
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Total business cost over 5 folds — lower is better")
    st.plotly_chart(style(fig, 360), width="stretch")
    if ranked.index[0] != champion:
        st.warning(
            f"At this cost ratio **{ranked.index[0]}** would be cheaper than the WAPE-selected "
            f"champion — WAPE treats both error directions equally, an asymmetric cost does not."
        )


# ---------------------------------------------------------------- page
st.title("Demand Forecasting Platform")
mode = (
    "standalone (in-process model)"
    if settings.DASHBOARD_STANDALONE_MODE
    else f"API at {settings.API_URL}"
)
st.caption(
    "28-day forecasts for 1,782 retail series · five model tiers compared by backtest · "
    f"serving: {mode}"
)
overview, explorer, leaderboard, business = st.tabs(
    ["Overview", "Forecast Explorer", "Model Leaderboard", "Business Impact"]
)
with overview:
    tab_overview()
with explorer:
    tab_explorer()
with leaderboard:
    tab_leaderboard()
with business:
    tab_business()
