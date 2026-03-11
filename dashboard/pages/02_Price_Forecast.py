"""
Page 02 — Price Forecast
━━━━━━━━━━━━━━━━━━━━━━━━
LSTM, Prophet, and Ensemble price forecasts for NSE stocks.
Displays:
  • Ensemble forecast with 95% confidence band
  • LSTM vs Prophet comparison
  • 30 / 60 / 90-day horizon selector
  • MC-Dropout uncertainty bars
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
import yfinance as yf

from dashboard.components.charts import forecast_chart
from dashboard.components.metrics_cards import metric_row
from dashboard.components.sidebar import render_sidebar

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
API_URL      = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Price Forecast · Portfolio Optimizer",
                   page_icon="🔮", layout="wide")

cfg        = render_sidebar()
tickers    = cfg["tickers"]
start_date = cfg["start_date"]
refresh    = cfg["refresh"]

if refresh:
    st.cache_data.clear()

st.markdown("# 🔮 Price Forecast")
st.markdown(
    "Forward price predictions using **LSTM** (deep learning), "
    "**Prophet** (Indian seasonalities), and their **ensemble**."
)

# ── Ticker selector + horizon ─────────────────────────────────────────────────
col_t, col_h, col_m = st.columns([2, 1, 1])
with col_t:
    selected_ticker = st.selectbox("Select Ticker", tickers, key="fc_ticker")
with col_h:
    days_ahead = st.selectbox("Forecast Horizon", [30, 60, 90], index=0, key="fc_days")
with col_m:
    model_choice = st.selectbox("Model", ["ensemble", "lstm", "prophet"], key="fc_model")

st.divider()


# ── Fetch historical data ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _hist(ticker: str, period: str = "2y") -> pd.Series:
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    return data["Close"].squeeze()


# ── Call API for forecasts ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _forecast_api(ticker: str, days: int, model: str) -> dict | None:
    try:
        import httpx
        payload = {"ticker": ticker, "days_ahead": days, "model": model}
        r = httpx.post(f"{API_URL}/api/v1/predict/price", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


with st.spinner(f"Generating {days_ahead}-day forecast for **{selected_ticker}** …"):
    hist_series = _hist(selected_ticker)
    forecast    = _forecast_api(selected_ticker, days_ahead, model_choice)

if hist_series is None or hist_series.empty:
    st.error("Could not fetch historical data. Check ticker symbol.")
    st.stop()

# ── Build forecast series ─────────────────────────────────────────────────────
def _to_series(points: list[dict]) -> pd.Series | None:
    if not points:
        return None
    dates  = pd.to_datetime([p["date"] for p in points])
    prices = [p["price"] for p in points]
    return pd.Series(prices, index=dates)


lstm_s, prophet_s, ensemble_s = None, None, None
if forecast:
    lstm_s     = _to_series(forecast.get("lstm_forecast", []))
    prophet_s  = _to_series(forecast.get("prophet_forecast", []))
    ensemble_s = _to_series(forecast.get("ensemble_forecast", []))

# ── KPI row ───────────────────────────────────────────────────────────────────
current_price = float(hist_series.iloc[-1])
metrics = [
    {"label": "Current Price",   "value": f"₹{current_price:,.2f}", "icon": "💹"},
    {"label": "Forecast Horizon","value": f"{days_ahead}d",         "icon": "📅"},
]
if ensemble_s is not None and len(ensemble_s):
    target = float(ensemble_s.iloc[-1])
    upside = (target - current_price) / current_price * 100
    metrics += [
        {"label": "Ensemble Target", "value": f"₹{target:,.2f}",    "icon": "🎯"},
        {"label": "Potential Upside", "value": f"{upside:+.1f}%",
         "delta": upside, "icon": "📈"},
    ]
if forecast:
    metrics.append(
        {"label": "Model Confidence",
         "value": f"{forecast.get('model_confidence_score', 0)*100:.0f}%",
         "icon": "🤖"}
    )
metric_row(metrics)

st.divider()

# ── Forecast chart ────────────────────────────────────────────────────────────
fig = forecast_chart(
    history=hist_series.tail(252),
    lstm_fc=lstm_s,
    prophet_fc=prophet_s,
    ensemble_fc=ensemble_s,
    ticker=selected_ticker,
)
st.plotly_chart(fig, use_container_width=True)

# ── Confidence band (ensemble) ────────────────────────────────────────────────
if forecast and forecast.get("ensemble_forecast"):
    import plotly.graph_objects as go
    from dashboard.components.charts import _base_layout

    pts = forecast["ensemble_forecast"]
    dates  = pd.to_datetime([p["date"] for p in pts])
    lowers = [p.get("confidence_lower", p["price"]) for p in pts]
    uppers = [p.get("confidence_upper", p["price"]) for p in pts]
    mids   = [p["price"] for p in pts]

    band_fig = go.Figure()
    band_fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(dates)),
        y=uppers + list(reversed(lowers)),
        fill="toself",
        fillcolor=f"rgba(194,77,44,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
    ))
    band_fig.add_trace(go.Scatter(
        x=dates, y=mids,
        line=dict(color=BURNT_ORANGE, width=2.5),
        name="Ensemble Forecast",
    ))
    band_fig.update_layout(
        **_base_layout(title=f"{selected_ticker} — Ensemble Forecast with 95% CI")
    )
    st.plotly_chart(band_fig, use_container_width=True)

# ── Raw forecast table ────────────────────────────────────────────────────────
with st.expander("📋 Raw Forecast Data"):
    if forecast and forecast.get("ensemble_forecast"):
        df_fc = pd.DataFrame(forecast["ensemble_forecast"])
        df_fc["date"] = pd.to_datetime(df_fc["date"]).dt.strftime("%d %b %Y")
        df_fc.columns = ["Date", "Price (₹)", "Lower Bound (₹)", "Upper Bound (₹)"]
        st.dataframe(df_fc.set_index("Date"), use_container_width=True)
    else:
        st.info("API forecast data not available. Ensure the FastAPI server is running "
                f"on {API_URL}.")
