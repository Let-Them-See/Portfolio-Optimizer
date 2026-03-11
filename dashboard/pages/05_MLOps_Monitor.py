"""
Page 05 — MLOps Monitor
━━━━━━━━━━━━━━━━━━━━━━━
Production ML health dashboard:
  • Data drift PSI / KS-test indicators
  • MLflow experiment run table
  • Model performance over time (MAPE, Sharpe, DirectionalAcc)
  • Retrain history log
  • Market regime detection (Bull / Bear / Sideways)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _base_layout
from dashboard.components.metrics_cards import metric_row
from dashboard.components.sidebar import render_sidebar

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
SUCCESS      = "#2ECC71"
WARNING      = "#F39C12"
DANGER       = "#E74C3C"

st.set_page_config(page_title="MLOps Monitor · Portfolio Optimizer",
                   page_icon="🔬", layout="wide")

cfg     = render_sidebar()
refresh = cfg["refresh"]

if refresh:
    st.cache_data.clear()

st.markdown("# 🔬 MLOps Monitor")
st.markdown(
    "Real-time model health, drift detection, and MLflow experiment tracking."
)

# ── API health check ──────────────────────────────────────────────────────────
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@st.cache_data(ttl=60)
def _health() -> dict:
    try:
        import httpx
        r = httpx.get(f"{API_URL}/api/v1/health", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


with st.spinner("Fetching system health …"):
    health = _health()

api_status = health.get("status", "unknown")
status_color = SUCCESS if api_status == "healthy" else \
               WARNING if "degraded" in api_status else DANGER

metric_row([
    {"label": "API Status",       "value": api_status.upper(),              "icon": "🌐"},
    {"label": "Redis",            "value": health.get("redis_status", "—"), "icon": "🔴"},
    {"label": "MLflow",           "value": health.get("mlflow_status", "—"),"icon": "🧪"},
    {"label": "Uptime",           "value": f"{health.get('uptime_seconds', 0)/3600:.1f}h","icon": "⏱️"},
    {"label": "Last Data Refresh","value": health.get("last_data_refresh","—")[:10] if health.get("last_data_refresh") else "—","icon": "📅"},
])

st.divider()

# ── Model versions ────────────────────────────────────────────────────────────
st.markdown("#### 🤖 Model Registry")
model_versions = health.get("model_versions", {})
if model_versions:
    df_mv = pd.DataFrame([
        {"Model": k, "Status": v}
        for k, v in model_versions.items()
    ])
    st.dataframe(df_mv, use_container_width=True, hide_index=True)
else:
    st.info("Model registry unavailable. Start the FastAPI server.")

st.divider()

# ── MLflow experiments ────────────────────────────────────────────────────────
st.markdown("#### 🧪 MLflow Experiment Runs")

@st.cache_data(ttl=120)
def _mlflow_runs() -> pd.DataFrame:
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_server/mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(
            experiment_names=[
                os.getenv("MLFLOW_EXPERIMENT_LSTM", "NSE_LSTM_ForecastV2"),
                os.getenv("MLFLOW_EXPERIMENT_PROPHET", "NSE_Prophet_ForecastV1"),
                os.getenv("MLFLOW_EXPERIMENT_RL", "NSE_RL_PortfolioV1"),
            ],
            order_by=["start_time DESC"],
            max_results=50,
        )
        cols = ["run_id", "experiment_id", "status", "start_time",
                "metrics.val_mape", "metrics.sharpe_ratio",
                "metrics.directional_accuracy", "tags.mlflow.runName"]
        available = [c for c in cols if c in runs.columns]
        return runs[available].head(20)
    except Exception as exc:
        return pd.DataFrame({"info": [f"MLflow unavailable: {exc}"]})


with st.spinner("Querying MLflow …"):
    runs_df = _mlflow_runs()

if not runs_df.empty and "run_id" in runs_df.columns:
    runs_df = runs_df.rename(columns={
        "tags.mlflow.runName":        "Run Name",
        "metrics.val_mape":           "MAPE",
        "metrics.sharpe_ratio":       "Sharpe",
        "metrics.directional_accuracy":"Dir. Acc",
        "start_time":                 "Time",
        "status":                     "Status",
    })
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
else:
    st.info("No MLflow runs found. Run training pipeline first: "
            "`python -m mlops.train_pipeline --all`")

# ── Drift detection ───────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📡 Data Drift Monitor")

@st.cache_data(ttl=300)
def _drift_report() -> dict:
    try:
        from mlops.drift_detection import run_drift_report
        report = run_drift_report()
        return report if report else {}
    except Exception as exc:
        return {"error": str(exc)}


drift_rep = _drift_report()

if "error" in drift_rep or not drift_rep:
    st.info(
        "Drift detection report unavailable. "
        "Ensure processed data exists and run training pipeline first."
    )
else:
    ticker = list(drift_rep.keys())[0]
    rep    = drift_rep[ticker]
    dd     = rep.get("data_drift", {})
    md     = rep.get("model_drift", {})
    regime = rep.get("market_regime", "UNKNOWN")

    recolor = {
        "Bull": SUCCESS, "Bear": DANGER, "Sideways": WARNING, "UNKNOWN": PLATINUM
    }
    st.markdown(
        f"""
        <div style='background:{SLATE_BLUE}; border-radius:8px; padding:12px 20px;
                    border-left:4px solid {recolor.get(regime, PLATINUM)};
                    margin-bottom:16px;'>
            <span style='color:{PLATINUM}; font-size:12px;'>Current Market Regime:</span>
            <span style='color:{recolor.get(regime, PLATINUM)}; font-size:18px;
                         font-weight:700; margin-left:12px;'>
                {regime}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row([
        {"label": "PSI Drift",  "value": f"{dd.get('max_psi', 0):.3f}",
         "delta": -1 if dd.get("drift_detected") else 0, "icon": "📊"},
        {"label": "KS p-value",
         "value": f"{dd.get('min_ks_pvalue', 1):.4f}",
         "icon": "🔬"},
        {"label": "Model MAPE",
         "value": f"{md.get('current_mape', 0)*100:.1f}%", "icon": "🎯"},
        {"label": "Model Drift",
         "value": "⚠️ DETECTED" if md.get("drift_detected") else "✅ OK",
         "icon": "🚨"},
    ])

    if dd.get("drift_detected") or md.get("drift_detected"):
        st.warning(
            "**Drift Detected!** Consider re-running the training pipeline: "
            "`python -m mlops.retrain_trigger --force`",
            icon="⚠️",
        )

# ── Simulated metric trend ─────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📈 Model Performance Trend (Simulated / Historical)")

import numpy as np
rng   = np.random.default_rng(0)
dates = pd.date_range("2024-01-01", periods=90, freq="B")
mape  = np.clip(10 + rng.normal(0, 1.5, 90).cumsum() * 0.05, 4, 20)
dir_a = np.clip(62 + rng.normal(0, 2, 90).cumsum() * 0.03, 50, 80)

fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(
    x=dates, y=mape,
    name="MAPE (%)", yaxis="y",
    line=dict(color=BURNT_ORANGE, width=2),
))
fig_perf.add_trace(go.Scatter(
    x=dates, y=dir_a,
    name="Directional Acc (%)", yaxis="y2",
    line=dict(color="#4FC3F7", width=2, dash="dash"),
))
fig_perf.add_hline(y=10, line=dict(color=WARNING, dash="dot", width=1),
                   annotation_text="MAPE Threshold", yref="y")
fig_perf.update_layout(
    **_base_layout(title="Model Performance Over Time (FY 2024–25)"),
    yaxis=dict(title="MAPE (%)", gridcolor=SLATE_BLUE, tickfont=dict(color=PLATINUM)),
    yaxis2=dict(title="Directional Accuracy (%)", overlaying="y", side="right",
                gridcolor=SLATE_BLUE, tickfont=dict(color=PLATINUM)),
)
st.plotly_chart(fig_perf, use_container_width=True)

# ── Retrain log ───────────────────────────────────────────────────────────────
st.markdown("#### 🔄 Retrain History")
retrain_log_path = os.getenv("RETRAIN_LOG", "mlops/retrain_log.json")
if os.path.exists(retrain_log_path):
    import json
    with open(retrain_log_path) as f:
        log = json.load(f)
    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.info("No retrain events logged yet.")
else:
    st.info(
        "No retrain log found. Scheduler writes to `mlops/retrain_log.json`. "
        "Start scheduler: `python -m mlops.retrain_trigger --schedule`"
    )
