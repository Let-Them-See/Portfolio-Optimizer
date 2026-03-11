"""
Reusable Plotly chart helpers — Palette-enforced
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design tokens:
  DEEP_NAVY   = "#1A273A"
  SLATE_BLUE  = "#3E4A62"
  BURNT_ORANGE= "#C24D2C"
  PLATINUM    = "#D9D9D7"

All charts follow the strict 4-colour palette.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ─────────────────────────────────────────────────────────────
DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
SUCCESS      = "#2ECC71"
WARNING      = "#F39C12"
DANGER       = "#E74C3C"

# Sequential palette for multi-line charts
MULTI_COLORS = [BURNT_ORANGE, "#4FC3F7", "#81C784", "#FFD54F", "#CE93D8", "#EF9A9A"]


def _base_layout(**kwargs) -> dict:
    """Return a base Plotly layout dict enforcing the palette."""
    return dict(
        paper_bgcolor=DEEP_NAVY,
        plot_bgcolor=DEEP_NAVY,
        font=dict(color=PLATINUM, family="Inter, DM Sans, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor=SLATE_BLUE,
            bordercolor=PLATINUM,
            borderwidth=1,
            font=dict(color=PLATINUM),
        ),
        xaxis=dict(
            gridcolor=SLATE_BLUE,
            linecolor=SLATE_BLUE,
            tickfont=dict(color=PLATINUM),
            title_font=dict(color=PLATINUM),
        ),
        yaxis=dict(
            gridcolor=SLATE_BLUE,
            linecolor=SLATE_BLUE,
            tickfont=dict(color=PLATINUM),
            title_font=dict(color=PLATINUM),
        ),
        **kwargs,
    )


# ── Candlestick chart ─────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, ticker: str, title: str = "") -> go.Figure:
    """
    OHLCV candlestick with 20/50-day SMA overlay.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns + DatetimeIndex.
        ticker: Stock symbol for title.
        title: Optional override title.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            increasing_line_color=SUCCESS,
            decreasing_line_color=DANGER,
            name="Price",
        ),
        row=1, col=1,
    )
    if "SMA_20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA-20",
                                 line=dict(color=BURNT_ORANGE, width=1.5)), row=1, col=1)
    if "SMA_50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA-50",
                                 line=dict(color="#4FC3F7", width=1.5)), row=1, col=1)
    colors = [SUCCESS if c >= o else DANGER
              for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, showlegend=False),
        row=2, col=1,
    )
    layout = _base_layout(title=title or f"{ticker} — Price & Volume")
    layout.update(
        xaxis2=dict(gridcolor=SLATE_BLUE, linecolor=SLATE_BLUE, tickfont=dict(color=PLATINUM)),
        yaxis2=dict(gridcolor=SLATE_BLUE, linecolor=SLATE_BLUE, tickfont=dict(color=PLATINUM),
                    title_text="Volume"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_layout(**layout)
    return fig


# ── Forecast chart ────────────────────────────────────────────────────────────
def forecast_chart(
    history: pd.Series,
    lstm_fc:    Optional[pd.Series] = None,
    prophet_fc: Optional[pd.Series] = None,
    ensemble_fc:Optional[pd.Series] = None,
    ticker: str = "",
) -> go.Figure:
    """
    Overlay historical close with LSTM, Prophet and Ensemble forecasts.
    Includes confidence band for ensemble.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, name="Historical",
        line=dict(color=PLATINUM, width=2),
    ))
    if lstm_fc is not None:
        fig.add_trace(go.Scatter(
            x=lstm_fc.index, y=lstm_fc.values, name="LSTM",
            line=dict(color=BURNT_ORANGE, width=2, dash="dot"),
        ))
    if prophet_fc is not None:
        fig.add_trace(go.Scatter(
            x=prophet_fc.index, y=prophet_fc.values, name="Prophet",
            line=dict(color="#4FC3F7", width=2, dash="dash"),
        ))
    if ensemble_fc is not None:
        fig.add_trace(go.Scatter(
            x=ensemble_fc.index, y=ensemble_fc.values, name="Ensemble",
            line=dict(color=SUCCESS, width=2.5),
        ))
    fig.update_layout(**_base_layout(title=f"{ticker} — Price Forecast"))
    return fig


# ── Efficient frontier ────────────────────────────────────────────────────────
def efficient_frontier_chart(
    sim_returns: np.ndarray,
    sim_vols: np.ndarray,
    sim_sharpes: np.ndarray,
    opt_ret: float,
    opt_vol: float,
) -> go.Figure:
    """
    Monte-Carlo efficient frontier scatter with optimal portfolio star.

    Args:
        sim_returns: Array of simulated portfolio returns.
        sim_vols: Array of simulated portfolio volatilities.
        sim_sharpes: Array of simulated Sharpe ratios (for colour coding).
        opt_ret: Optimal portfolio expected return.
        opt_vol: Optimal portfolio volatility.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_vols, y=sim_returns,
        mode="markers",
        marker=dict(
            color=sim_sharpes,
            colorscale=[[0, SLATE_BLUE], [0.5, PLATINUM], [1, BURNT_ORANGE]],
            size=4, opacity=0.6, showscale=True,
            colorbar=dict(title="Sharpe", tickfont=dict(color=PLATINUM)),
        ),
        name="Simulated Portfolios",
    ))
    fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret],
        mode="markers+text",
        marker=dict(color=BURNT_ORANGE, size=18, symbol="star"),
        text=["Optimal"], textposition="top center",
        name="Optimal Portfolio",
    ))
    fig.update_layout(
        **_base_layout(title="Efficient Frontier — Monte Carlo Simulation"),
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Expected Annual Return (%)",
    )
    return fig


# ── Portfolio allocation pie ──────────────────────────────────────────────────
def allocation_pie(weights: Dict[str, float], capital: float) -> go.Figure:
    """Donut chart of portfolio allocation."""
    labels  = list(weights.keys())
    values  = [w * capital / 1e5 for w in weights.values()]
    colors  = (MULTI_COLORS * ((len(labels) // len(MULTI_COLORS)) + 1))[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=DEEP_NAVY, width=2)),
        textinfo="label+percent",
        textfont=dict(color=PLATINUM),
    ))
    fig.update_layout(
        **_base_layout(title="Portfolio Allocation (₹ Lakhs)"),
        showlegend=True,
    )
    return fig


# ── Drawdown chart ────────────────────────────────────────────────────────────
def drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """Equity curve + underwater drawdown chart."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name="Equity Curve", fill="tozeroy",
        line=dict(color=BURNT_ORANGE, width=2),
        fillcolor="rgba(194,77,44,0.15)",
    ), row=1, col=1)
    peak = equity_curve.cummax()
    dd   = ((equity_curve - peak) / peak) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown %", fill="tozeroy",
        line=dict(color=DANGER, width=1.5),
        fillcolor="rgba(231,76,60,0.25)",
    ), row=2, col=1)
    layout = _base_layout(title="Equity Curve & Drawdown")
    layout.update(
        yaxis_title="Portfolio Value (₹)",
        yaxis2=dict(title_text="Drawdown %", gridcolor=SLATE_BLUE,
                    linecolor=SLATE_BLUE, tickfont=dict(color=PLATINUM),
                    title_font=dict(color=PLATINUM)),
        xaxis2=dict(gridcolor=SLATE_BLUE, linecolor=SLATE_BLUE,
                    tickfont=dict(color=PLATINUM)),
    )
    fig.update_layout(**layout)
    return fig


# ── Monthly returns heatmap ───────────────────────────────────────────────────
def monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """
    Calendar heatmap of monthly returns (rows=Year, cols=Month).
    """
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    df_m = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month
    pivot = df_m.pivot(index="year", columns="month", values="ret")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    z    = pivot.values
    x    = [month_labels[m-1] for m in pivot.columns]
    y    = [str(yr) for yr in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, DANGER], [0.5, SLATE_BLUE], [1, SUCCESS]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(color=PLATINUM, size=10),
        showscale=True,
        colorbar=dict(title="Return %", tickfont=dict(color=PLATINUM)),
    ))
    fig.update_layout(**_base_layout(title="Monthly Returns Heatmap (%)"))
    return fig


# ── Sector heatmap ────────────────────────────────────────────────────────────
def sector_heatmap(sector_perf: Dict[str, float]) -> go.Figure:
    """Treemap of sector daily performance."""
    sectors = list(sector_perf.keys())
    values  = [abs(v) + 0.5 for v in sector_perf.values()]  # area ∝ abs change
    colors  = [BURNT_ORANGE if v >= 0 else DANGER for v in sector_perf.values()]
    texts   = [f"{s}<br>{v:+.2f}%" for s, v in sector_perf.items()]

    fig = go.Figure(go.Treemap(
        labels=sectors,
        values=values,
        parents=[""] * len(sectors),
        text=texts,
        textinfo="text",
        marker=dict(colors=colors, line=dict(color=DEEP_NAVY, width=2)),
    ))
    fig.update_layout(**_base_layout(title="Sector Performance (Today)"))
    return fig


# ── Loss curve ────────────────────────────────────────────────────────────────
def training_loss_chart(train_loss: list, val_loss: list) -> go.Figure:
    """LSTM training / validation loss curves."""
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                             line=dict(color=BURNT_ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                             line=dict(color="#4FC3F7", width=2, dash="dash")))
    fig.update_layout(
        **_base_layout(title="LSTM Training Loss"),
        xaxis_title="Epoch",
        yaxis_title="Huber Loss",
    )
    return fig


# ── Feature importance bar ────────────────────────────────────────────────────
def feature_importance_chart(importances: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart for permutation feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [i[0] for i in sorted_items]
    values   = [i[1] for i in sorted_items]
    colors   = [BURNT_ORANGE if v > 0 else DANGER for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **_base_layout(title="Feature Importance (Permutation)"),
        xaxis_title="Importance Score (↓ Loss)",
        height=max(300, 30 * len(features)),
    )
    return fig
