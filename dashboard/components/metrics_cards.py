"""
Metric card components for the Streamlit dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides styled KPI tiles with correct colour-palette enforcement.

Design tokens:
  DEEP_NAVY    #1A273A  —  card background
  SLATE_BLUE   #3E4A62  —  card border / hover
  BURNT_ORANGE #C24D2C  —  positive / accent values
  PLATINUM     #D9D9D7  —  label / neutral text
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"
SUCCESS      = "#2ECC71"
DANGER       = "#E74C3C"


def _delta_colour(delta: float | None) -> str:
    if delta is None:
        return PLATINUM
    return SUCCESS if delta >= 0 else DANGER


def metric_card(
    label: str,
    value: str,
    delta: Optional[float] = None,
    delta_suffix: str = "%",
    icon: str = "",
    help_text: str = "",
) -> None:
    """
    Render a single styled KPI metric card.

    Args:
        label: Card title (e.g. "Sharpe Ratio").
        value: Primary display value (pre-formatted string).
        delta: Optional numeric change (positive=green, negative=red).
        delta_suffix: Suffix appended to delta display.
        icon: Optional emoji prefix.
        help_text: Tooltip text.
    """
    delta_html = ""
    if delta is not None:
        sign  = "▲" if delta >= 0 else "▼"
        color = _delta_colour(delta)
        delta_html = (
            f"<div style='font-size:12px; color:{color}; margin-top:4px;'>"
            f"{sign} {abs(delta):.2f}{delta_suffix}</div>"
        )

    title_line = f"{icon} {label}" if icon else label

    card_html = f"""
    <div style='
        background:{DEEP_NAVY};
        border:1px solid {SLATE_BLUE};
        border-radius:10px;
        padding:16px 18px;
        min-height:90px;
        display:flex;
        flex-direction:column;
        justify-content:center;
    ' title='{help_text}'>
        <div style='font-size:11px; color:{PLATINUM}; opacity:0.65;
                    text-transform:uppercase; letter-spacing:0.8px;'>
            {title_line}
        </div>
        <div style='font-size:24px; font-weight:700; color:{BURNT_ORANGE};
                    margin-top:6px; line-height:1;'>
            {value}
        </div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def metric_row(metrics: list[dict]) -> None:
    """
    Render a horizontal row of KPI metric cards.

    Args:
        metrics: List of dicts, each accepted by :func:`metric_card`.
                 Keys: label, value, delta (opt), delta_suffix (opt),
                       icon (opt), help_text (opt).
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                delta_suffix=m.get("delta_suffix", "%"),
                icon=m.get("icon", ""),
                help_text=m.get("help_text", ""),
            )


def index_banner(
    nifty_val: float,
    nifty_chg: float,
    sensex_val: float,
    sensex_chg: float,
    status: str,
) -> None:
    """
    Render the top index banner: NIFTY50 | SENSEX | Market Status.

    Args:
        nifty_val: Current NIFTY50 index value.
        nifty_chg: Day change percentage.
        sensex_val: Current SENSEX value.
        sensex_chg: Day change percentage.
        status: 'OPEN', 'CLOSED', or 'PRE_OPEN'.
    """
    n_color  = SUCCESS if nifty_chg >= 0 else DANGER
    s_color  = SUCCESS if sensex_chg >= 0 else DANGER
    n_arrow  = "▲" if nifty_chg >= 0 else "▼"
    s_arrow  = "▲" if sensex_chg >= 0 else "▼"
    status_color = SUCCESS if status == "OPEN" else (WARNING if status == "PRE_OPEN" else DANGER)
    WARNING  = "#F39C12"
    st.markdown(
        f"""
        <div style='
            background:{SLATE_BLUE}; border-radius:10px;
            padding:12px 24px; display:flex; align-items:center;
            justify-content:space-around; margin-bottom:16px;
            border:1px solid {BURNT_ORANGE}33;
        '>
            <div style='text-align:center;'>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.6;'>NIFTY 50</div>
                <div style='font-size:20px; font-weight:700; color:{PLATINUM};'>
                    {nifty_val:,.2f}
                </div>
                <div style='font-size:13px; color:{n_color};'>
                    {n_arrow} {abs(nifty_chg):.2f}%
                </div>
            </div>
            <div style='width:1px; height:40px; background:{PLATINUM}33;'></div>
            <div style='text-align:center;'>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.6;'>SENSEX</div>
                <div style='font-size:20px; font-weight:700; color:{PLATINUM};'>
                    {sensex_val:,.2f}
                </div>
                <div style='font-size:13px; color:{s_color};'>
                    {s_arrow} {abs(sensex_chg):.2f}%
                </div>
            </div>
            <div style='width:1px; height:40px; background:{PLATINUM}33;'></div>
            <div style='text-align:center;'>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.6;'>MARKET</div>
                <div style='font-size:18px; font-weight:700; color:{status_color};'>
                    {status}
                </div>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.5;'>
                    NSE · 09:15–15:30 IST
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
