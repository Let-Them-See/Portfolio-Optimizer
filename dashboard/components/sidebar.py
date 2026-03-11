"""
Shared sidebar component — ticker selector + config
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Renders a consistent sidebar across all dashboard pages:
  • Ticker multi-select (NSE universe)
  • Date range picker
  • Capital input (₹ Lakhs)
  • Risk tolerance selector
  • Data refresh button

Returns a config dict for consumption by each page.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, Any

import streamlit as st

NSE_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "TITAN.NS",
    "MARUTI.NS", "ONGC.NS", "ZOMATO.NS",
]

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"


def render_sidebar() -> Dict[str, Any]:
    """
    Render the main sidebar and return selected configuration.

    Returns:
        A dict with keys:
            tickers     (list[str])
            start_date  (date)
            end_date    (date)
            capital_inr (float)
            risk        (str)
            refresh     (bool)
    """
    with st.sidebar:
        # ── Logo / branding ──────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style='text-align:center; padding:16px 0 8px 0;'>
                <span style='font-size:28px; color:{BURNT_ORANGE}; font-weight:800;'>📈</span>
                <div style='font-size:18px; font-weight:700;
                            color:{PLATINUM}; letter-spacing:1px;'>
                    Portfolio Optimizer
                </div>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.6;'>
                    NSE · BSE · India
                </div>
            </div>
            <hr style='border:1px solid {SLATE_BLUE}; margin:4px 0 12px 0;'/>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<p style='color:{PLATINUM}; font-size:13px; font-weight:600;'"
            f">⚙️ Configuration</p>",
            unsafe_allow_html=True,
        )

        # ── Ticker selection ──────────────────────────────────────────────────
        tickers = st.multiselect(
            "Select NSE Tickers",
            options=NSE_UNIVERSE,
            default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
            help="Choose 2–12 NSE stocks to analyse.",
            max_selections=12,
        )
        if len(tickers) < 1:
            st.warning("Select at least 1 ticker.", icon="⚠️")
            tickers = ["RELIANCE.NS"]

        # ── Date range ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From",
                value=date.today() - timedelta(days=365),
                min_value=date(2015, 1, 1),
                max_value=date.today(),
            )
        with col2:
            end_date = st.date_input(
                "To",
                value=date.today(),
                min_value=start_date,
                max_value=date.today(),
            )

        # ── Capital ────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        capital_lakhs = st.number_input(
            "Capital (₹ Lakhs)",
            min_value=0.5, max_value=10_000.0,
            value=10.0, step=0.5,
            help="Investment capital in Indian Rupees (₹ Lakhs). 10 = ₹10,00,000.",
        )
        capital_inr = capital_lakhs * 1e5

        # ── Risk tolerance ────────────────────────────────────────────────────
        risk = st.select_slider(
            "Risk Tolerance",
            options=["low", "medium", "high"],
            value="medium",
        )

        # ── Refresh button ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button(
            "🔄 Refresh Data",
            use_container_width=True,
            help="Pull latest NSE data and recompute forecasts.",
        )

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <hr style='border:1px solid {SLATE_BLUE}; margin:16px 0 8px 0;'/>
            <div style='font-size:10px; color:{PLATINUM}; opacity:0.45;
                        text-align:center; line-height:1.6;'>
                NSE market closes at 15:30 IST<br>
                Data via yfinance · Models: LSTM + Prophet + RL<br>
                v1.0.0 · FY 2024–25
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "tickers":     tickers,
        "start_date":  start_date,
        "end_date":    end_date,
        "capital_inr": capital_inr,
        "risk":        risk,
        "refresh":     refresh,
    }
