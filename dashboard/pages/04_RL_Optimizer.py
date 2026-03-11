"""
Page 04 — RL Portfolio Optimizer (PPO Agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visualises the trained PPO agent's backtest performance:
  • Cumulative returns vs NIFTY50 benchmark
  • Monthly returns heatmap
  • Weight evolution over time
  • Trade log table
  • Portfolio drawdown
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from dashboard.components.charts import (
    _base_layout,
    drawdown_chart,
    monthly_returns_heatmap,
)
from dashboard.components.metrics_cards import metric_row
from dashboard.components.sidebar import render_sidebar

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
SUCCESS      = "#2ECC71"
DANGER       = "#E74C3C"

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RL Optimizer · Portfolio Optimizer",
                   page_icon="🤖", layout="wide")

cfg        = render_sidebar()
tickers    = cfg["tickers"]
capital    = cfg["capital_inr"]
start_date = cfg["start_date"]
end_date   = cfg["end_date"]
refresh    = cfg["refresh"]

if refresh:
    st.cache_data.clear()

st.markdown("# 🤖 RL Portfolio Optimizer (PPO Agent)")
st.markdown(
    "PPO-trained agent dynamically allocates across NSE stocks. "
    "Benchmark: NIFTY50 · Capital: ₹10 Lakhs."
)

# ── Fetch backtest data from API ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def _fetch_backtest(strategy: str, start: str, end: str, cap: float) -> dict | None:
    try:
        import httpx
        r = httpx.get(
            f"{API_URL}/api/v1/portfolio/backtest/{strategy}",
            params={"capital": cap, "start": start, "end": end},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


with st.spinner("Loading RL agent backtest …"):
    rl_bt    = _fetch_backtest("rl_agent",    str(start_date), str(end_date), capital)
    nifty_bt = _fetch_backtest("nifty50",     str(start_date), str(end_date), capital)
    eq_bt    = _fetch_backtest("equal_weight",str(start_date), str(end_date), capital)


# Fallback: simulate using equal-weight if API unavailable
def _simulate_equity(tickers: list[str], weights: np.ndarray,
                     start: str, end: str, cap: float) -> pd.Series:
    raw = yf.download(tickers, start=start, end=end,
                      progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    ret = raw.pct_change().dropna()
    portfolio_ret = (ret * weights).sum(axis=1)
    return cap * (1 + portfolio_ret).cumprod()


def _build_equity_series(bt: dict | None, tickers: list[str],
                         start: str, end: str, cap: float,
                         strategy_label: str) -> pd.Series:
    if bt and bt.get("daily_returns"):
        data = bt["daily_returns"]
        dates  = pd.to_datetime([d["date"] for d in data])
        values = [d["portfolio_value"] for d in data]
        return pd.Series(values, index=dates, name=strategy_label)

    # Fallback
    n = len(tickers)
    w = np.ones(n) / n
    return _simulate_equity(tickers, w, start, end, cap).rename(strategy_label)


with st.spinner("Building equity curves …"):
    n  = len(tickers)
    eq_rl    = _build_equity_series(rl_bt,    tickers, str(start_date), str(end_date), capital, "RL Agent (PPO)")
    eq_nifty = _build_equity_series(nifty_bt, ["^NSEI"], str(start_date), str(end_date), capital, "NIFTY50")
    eq_equal = _build_equity_series(eq_bt,    tickers, str(start_date), str(end_date), capital, "Equal Weight")

# ── KPI banner ────────────────────────────────────────────────────────────────
def _stats(series: pd.Series, cap: float) -> dict:
    ret_total = (series.iloc[-1] / cap - 1) * 100
    daily_ret = series.pct_change().dropna()
    ann_vol   = daily_ret.std() * np.sqrt(252) * 100
    sharpe    = ret_total / max(ann_vol, 0.01)
    peak      = series.cummax()
    max_dd    = float(((series - peak) / peak * 100).min())
    return dict(total=round(ret_total,2), vol=round(ann_vol,2),
                sharpe=round(sharpe,2), max_dd=round(max_dd,2))

rl_s = _stats(eq_rl, capital)
nf_s = _stats(eq_nifty, capital)
alpha = rl_s["total"] - nf_s["total"]

metric_row([
    {"label": "RL Total Return",  "value": f"{rl_s['total']:+.1f}%",  "delta": rl_s["total"],  "icon": "🤖"},
    {"label": "NIFTY50 Return",   "value": f"{nf_s['total']:+.1f}%",  "delta": nf_s["total"],  "icon": "📊"},
    {"label": "Alpha vs NIFTY50", "value": f"{alpha:+.1f}%",          "delta": alpha,           "icon": "⚡"},
    {"label": "RL Sharpe Ratio",  "value": f"{rl_s['sharpe']:.2f}",                             "icon": "⚖️"},
    {"label": "RL Max Drawdown",  "value": f"{rl_s['max_dd']:.1f}%",  "delta": rl_s["max_dd"], "icon": "📉"},
])

st.divider()

# ── Cumulative returns comparison ─────────────────────────────────────────────
st.markdown("#### 📈 Cumulative Returns — RL vs Benchmarks")
fig_cum = go.Figure()
for series, color, dash in [
    (eq_rl,    BURNT_ORANGE, "solid"),
    (eq_nifty, "#4FC3F7",    "dash"),
    (eq_equal, PLATINUM,     "dot"),
]:
    fig_cum.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name=series.name,
        line=dict(color=color, width=2, dash=dash),
    ))
# Mark ₹10 Lakh horizontal
fig_cum.add_hline(y=capital, line=dict(color=PLATINUM, dash="dot", width=1),
                  annotation_text="Initial ₹10L",
                  annotation_font=dict(color=PLATINUM))
fig_cum.update_layout(**_base_layout(title="Portfolio Value (₹) — Agent vs Benchmarks"),
                       yaxis_title="Portfolio Value (₹)")
st.plotly_chart(fig_cum, use_container_width=True)

# ── Monthly returns heatmap ───────────────────────────────────────────────────
st.markdown("#### 📅 Monthly Returns Heatmap — RL Agent")
rl_returns = eq_rl.pct_change().dropna()
if len(rl_returns) > 20:
    fig_heat = monthly_returns_heatmap(rl_returns)
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Not enough data for monthly heatmap. Expand date range.")

# ── Drawdown ──────────────────────────────────────────────────────────────────
st.markdown("#### 📉 RL Agent Drawdown Analysis")
fig_dd = drawdown_chart(eq_rl)
st.plotly_chart(fig_dd, use_container_width=True)

# ── Rolling Sharpe ────────────────────────────────────────────────────────────
st.markdown("#### ⚖️ Rolling 60-Day Sharpe Ratio")
rolling_sharpe = (
    rl_returns.rolling(60).mean() /
    (rl_returns.rolling(60).std() + 1e-9)
) * np.sqrt(252)

fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=rolling_sharpe.index, y=rolling_sharpe.values,
    fill="tozeroy",
    fillcolor="rgba(194,77,44,0.18)",
    line=dict(color=BURNT_ORANGE, width=2),
    name="Rolling Sharpe (60d)",
))
fig_rs.add_hline(y=1.0, line=dict(color=SUCCESS, dash="dash", width=1),
                 annotation_text="Sharpe = 1.0")
fig_rs.update_layout(**_base_layout(title="60-Day Rolling Sharpe Ratio — RL Agent"))
st.plotly_chart(fig_rs, use_container_width=True)

# ── Performance comparison table ──────────────────────────────────────────────
st.markdown("#### 📋 Performance Summary")
perf_data = {
    "Strategy":       ["RL Agent (PPO)", "NIFTY50", "Equal Weight"],
    "Total Return":   [f"{rl_s['total']:+.2f}%", f"{nf_s['total']:+.2f}%",
                       f"{_stats(eq_equal, capital)['total']:+.2f}%"],
    "Ann. Volatility":[f"{rl_s['vol']:.2f}%",  f"{nf_s['vol']:.2f}%",
                       f"{_stats(eq_equal, capital)['vol']:.2f}%"],
    "Sharpe":         [f"{rl_s['sharpe']:.2f}", f"{nf_s['sharpe']:.2f}",
                       f"{_stats(eq_equal, capital)['sharpe']:.2f}"],
    "Max Drawdown":   [f"{rl_s['max_dd']:.2f}%", f"{nf_s['max_dd']:.2f}%",
                       f"{_stats(eq_equal, capital)['max_dd']:.2f}%"],
}
st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
