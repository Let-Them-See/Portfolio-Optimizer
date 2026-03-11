"""
Page 03 — Portfolio Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━
Modern Portfolio Theory engine:
  • Efficient frontier (5 000-portfolio Monte Carlo)
  • Sharpe-optimal allocation with Indian ₹ values
  • Portfolio donut chart
  • Correlation heatmap
  • 3-year backtest of optimal portfolio
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from dashboard.components.charts import (
    _base_layout,
    allocation_pie,
    drawdown_chart,
    efficient_frontier_chart,
)
from dashboard.components.metrics_cards import metric_row
from dashboard.components.sidebar import render_sidebar

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
API_URL      = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Portfolio Builder · Portfolio Optimizer",
                   page_icon="💼", layout="wide")

cfg        = render_sidebar()
tickers    = cfg["tickers"]
capital    = cfg["capital_inr"]
risk       = cfg["risk"]
start_date = cfg["start_date"]
end_date   = cfg["end_date"]
refresh    = cfg["refresh"]

if refresh:
    st.cache_data.clear()

st.markdown("# 💼 Portfolio Builder")
st.markdown(
    "Sharpe-maximising allocation via **Monte Carlo MPT** · "
    f"Capital: ₹{capital/1e5:.1f} Lakhs · Risk: **{risk.upper()}**"
)

# ── Fetch price history ───────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def _fetch_close(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].dropna(how="all")
    return raw[["Close"]].dropna()


with st.spinner("Fetching price data …"):
    close_df = _fetch_close(tuple(tickers), str(start_date), str(end_date))

if close_df.empty or len(close_df) < 30:
    st.error("Insufficient data. Widen your date range or change tickers.")
    st.stop()

returns = close_df.pct_change().dropna()

# ── Monte-Carlo MPT ───────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def _mpt(tickers: tuple, ret_arr: bytes, max_w: float, n_sim: int = 5000):
    """Run MPT simulation and return optimal weights + scatter arrays."""
    ret_df = pd.read_parquet(ret_arr) if isinstance(ret_arr, bytes) else ret_arr
    mu     = ret_df.mean() * 252
    sigma  = ret_df.cov() * 252
    n      = len(tickers)
    rng    = np.random.default_rng(42)

    sim_r, sim_v, sim_s = [], [], []
    all_w = []
    for _ in range(n_sim):
        w = rng.dirichlet(np.ones(n))
        w = np.clip(w, 0, max_w)
        w /= w.sum()
        p_r = float(np.dot(w, mu))
        p_v = float(np.sqrt(w @ sigma.values @ w))
        sim_r.append(p_r * 100)
        sim_v.append(p_v * 100)
        sim_s.append(p_r / (p_v + 1e-9))
        all_w.append(w)

    best = int(np.argmax(sim_s))
    opt_weights = {t: round(float(all_w[best][i]), 4) for i, t in enumerate(tickers)}
    return np.array(sim_r), np.array(sim_v), np.array(sim_s), opt_weights


max_w_map = {"low": 0.15, "medium": 0.25, "high": 0.40}
with st.spinner("Running Monte Carlo optimisation (5 000 portfolios) …"):
    sim_ret, sim_vol, sim_shr, opt_weights = _mpt(
        tuple(tickers), returns, max_w_map[risk]
    )

# ── Compute stats ─────────────────────────────────────────────────────────────
mu     = returns.mean() * 252
sigma  = returns.cov() * 252
w_arr  = np.array([opt_weights[t] for t in tickers])
p_ret  = float(np.dot(w_arr, mu)) * 100
p_vol  = float(np.sqrt(w_arr @ sigma.values @ w_arr)) * 100
sharpe = p_ret / max(p_vol, 1e-5)

metric_row([
    {"label": "Expected Annual Return",    "value": f"{p_ret:.1f}%",   "delta": p_ret,    "icon": "📈"},
    {"label": "Annual Volatility (Risk)",  "value": f"{p_vol:.1f}%",   "delta": -p_vol,   "icon": "📉"},
    {"label": "Sharpe Ratio",              "value": f"{sharpe:.2f}",                       "icon": "⚖️"},
    {"label": "Portfolio Capital",         "value": f"₹{capital/1e5:.1f}L",               "icon": "💰"},
])

st.divider()

# ── Two-column layout: frontier + donut ───────────────────────────────────────
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("#### Efficient Frontier")
    fig_ef = efficient_frontier_chart(sim_ret, sim_vol, sim_shr, p_ret, p_vol)
    st.plotly_chart(fig_ef, use_container_width=True)

with c2:
    st.markdown("#### Portfolio Allocation")
    fig_pie = allocation_pie(opt_weights, capital)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Allocation table ──────────────────────────────────────────────────────────
st.markdown("#### 📋 Detailed Allocation")
alloc_data = {
    "Ticker":          list(opt_weights.keys()),
    "Weight (%)":      [f"{v*100:.2f}%" for v in opt_weights.values()],
    "Allocation (₹)":  [f"₹{v*capital:,.0f}" for v in opt_weights.values()],
    "Allocation (L)":  [f"₹{v*capital/1e5:.3f}L" for v in opt_weights.values()],
}
st.dataframe(pd.DataFrame(alloc_data), use_container_width=True, hide_index=True)

# ── Correlation matrix ────────────────────────────────────────────────────────
st.markdown("#### 🔗 Correlation Matrix")
corr = returns.corr()
fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[[0, "#E74C3C"], [0.5, DEEP_NAVY], [1, "#2ECC71"]],
    zmid=0,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    textfont=dict(color=PLATINUM, size=9),
    colorbar=dict(title="r", tickfont=dict(color=PLATINUM)),
))
fig_corr.update_layout(**_base_layout(title="Return Correlation Matrix"))
st.plotly_chart(fig_corr, use_container_width=True)

# ── Historical backtest of optimal weights ────────────────────────────────────
st.markdown("#### 📊 Historical Portfolio Backtest")
portfolio_returns = (returns * w_arr).sum(axis=1)
equity            = capital * (1 + portfolio_returns).cumprod()
fig_dd = drawdown_chart(equity)
st.plotly_chart(fig_dd, use_container_width=True)

# Summary stats
total_ret = (equity.iloc[-1] / capital - 1) * 100
ann_vol2  = portfolio_returns.std() * np.sqrt(252) * 100
peak      = equity.cummax()
max_dd    = float(((equity - peak) / peak * 100).min())
calmar    = abs(total_ret / max_dd) if max_dd != 0 else 0.0
metric_row([
    {"label": "Total Return",   "value": f"{total_ret:.1f}%", "delta": total_ret},
    {"label": "Ann. Volatility","value": f"{ann_vol2:.1f}%"},
    {"label": "Max Drawdown",   "value": f"{max_dd:.1f}%",    "delta": max_dd},
    {"label": "Calmar Ratio",   "value": f"{calmar:.2f}"},
])
