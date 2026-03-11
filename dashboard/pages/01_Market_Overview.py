"""
Page 01 — Market Overview
━━━━━━━━━━━━━━━━━━━━━━━━━
Live NSE dashboard:
  • NIFTY50 / SENSEX index banner
  • Sector performance treemap
  • Top 5 gainers & losers
  • Individual candlestick charts with SMA overlays
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'dashboard', 'api', etc. are importable
# when Streamlit executes each page as a standalone script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from dashboard.components.charts import candlestick_chart, sector_heatmap
from dashboard.components.metrics_cards import index_banner, metric_row
from dashboard.components.sidebar import NSE_UNIVERSE, render_sidebar

st.set_page_config(page_title="Market Overview · Portfolio Optimizer",
                   page_icon="🏠", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
cfg = render_sidebar()
tickers    = cfg["tickers"]
start_date = cfg["start_date"]
end_date   = cfg["end_date"]
refresh    = cfg["refresh"]

st.markdown("# 🏠 Market Overview")
st.markdown("Live NSE / BSE market snapshot — price action, sectors, top movers.")

# ── Index banner ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_indices() -> dict:
    idx = yf.download(["^NSEI", "^BSESN"], period="2d", progress=False, auto_adjust=True)["Close"]
    n50 = float(idx["^NSEI"].iloc[-1])
    n50_chg = float(idx["^NSEI"].pct_change().iloc[-1] * 100)
    snx = float(idx["^BSESN"].iloc[-1])
    snx_chg = float(idx["^BSESN"].pct_change().iloc[-1] * 100)
    return dict(n50=n50, n50_chg=n50_chg, snx=snx, snx_chg=snx_chg)

if refresh:
    st.cache_data.clear()

with st.spinner("Fetching index data …"):
    indices = _fetch_indices()

now_utc = pd.Timestamp.utcnow()
mkt_hr  = now_utc.hour + 5.5 / 60  # rough IST
market_status = "OPEN" if 9.25 <= mkt_hr <= 15.5 else "PRE_OPEN" if 9.0 <= mkt_hr < 9.25 else "CLOSED"

index_banner(
    nifty_val=indices["n50"],
    nifty_chg=indices["n50_chg"],
    sensex_val=indices["snx"],
    sensex_chg=indices["snx_chg"],
    status=market_status,
)

# ── Stock price data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_stocks(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end,
                      progress=False, auto_adjust=True)
    return raw

with st.spinner("Loading stock data …"):
    raw_data = _fetch_stocks(
        tuple(tickers),
        str(start_date),
        str(end_date),
    )

# ── Gainers / Losers ──────────────────────────────────────────────────────────
if "Close" in raw_data.columns.get_level_values(0) if isinstance(raw_data.columns, pd.MultiIndex) else True:
    try:
        close = raw_data["Close"] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
        daily_chg = close.pct_change().iloc[-1].dropna().sort_values()
        top_gain = daily_chg.tail(3)
        top_lose = daily_chg.head(3)

        st.markdown("### 📊 Today's Movers")
        col_g, col_l = st.columns(2)
        with col_g:
            st.markdown("**🟢 Top Gainers**")
            metric_row([
                {"label": t, "value": f"₹—", "delta": round(v * 100, 2), "icon": "▲"}
                for t, v in top_gain.items()
            ])
        with col_l:
            st.markdown("**🔴 Top Losers**")
            metric_row([
                {"label": t, "value": f"₹—", "delta": round(v * 100, 2), "icon": "▼"}
                for t, v in top_lose.items()
            ])
    except Exception:
        pass

# ── Sector heatmap ────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Sector Performance")
sector_map = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS"],
    "FMCG/Consumer": ["ASIANPAINT.NS", "TITAN.NS"],
    "Auto": ["MARUTI.NS"],
    "Energy": ["ONGC.NS", "RELIANCE.NS"],
    "NBFC": ["BAJFINANCE.NS"],
    "New-Age": ["ZOMATO.NS"],
}
try:
    universe_data = yf.download(NSE_UNIVERSE, period="2d", progress=False, auto_adjust=True)["Close"]
    chg = universe_data.pct_change().iloc[-1].dropna()
    sector_perf = {}
    for sec, tkrs in sector_map.items():
        vals = [float(chg.get(t, 0)) * 100 for t in tkrs if t in chg.index]
        sector_perf[sec] = round(sum(vals) / len(vals), 2) if vals else 0.0
    st.plotly_chart(sector_heatmap(sector_perf), use_container_width=True)
except Exception as e:
    st.warning(f"Sector heatmap unavailable: {e}")

# ── Candlestick charts ────────────────────────────────────────────────────────
st.markdown("### 🕯️ Price Charts")
ticker_tabs = st.tabs(tickers)
for tab, ticker in zip(ticker_tabs, tickers):
    with tab:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                df_t = raw_data.xs(ticker, axis=1, level=1).copy()
            else:
                df_t = raw_data.copy()

            # SMA overlays
            df_t["SMA_20"] = df_t["Close"].rolling(20).mean()
            df_t["SMA_50"] = df_t["Close"].rolling(50).mean()

            fig = candlestick_chart(df_t, ticker)
            st.plotly_chart(fig, use_container_width=True)

            # Quick stats row
            price_now = float(df_t["Close"].iloc[-1])
            price_7d  = float(df_t["Close"].iloc[-6]) if len(df_t) >= 6 else price_now
            chg_7d    = (price_now - price_7d) / price_7d * 100

            metric_row([
                {"label": "Last Close",     "value": f"₹{price_now:,.2f}"},
                {"label": "7-Day Change",   "value": f"{chg_7d:+.2f}%",  "delta": chg_7d},
                {"label": "52W High",       "value": f"₹{df_t['High'].max():,.2f}"},
                {"label": "52W Low",        "value": f"₹{df_t['Low'].min():,.2f}"},
            ])
        except Exception as ex:
            st.error(f"Could not render chart for {ticker}: {ex}")
