"""
Streamlit Dashboard — Real-Time Stock Portfolio Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry-point (app.py).
Run: streamlit run dashboard/app.py

Pages (auto-discovered by Streamlit):
  01_Market_Overview.py
  02_Price_Forecast.py
  03_Portfolio_Builder.py
  04_RL_Optimizer.py
  05_MLOps_Monitor.py

Design palette (enforced globally via CSS injection):
  #1A273A  Deep Navy   — backgrounds
  #3E4A62  Slate Blue  — surfaces / cards
  #C24D2C  Burnt Orange— accents / CTAs
  #D9D9D7  Platinum    — text / borders
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path so all packages resolve correctly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer · NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/vedant",
        "Report a bug": "https://github.com/vedant",
        "About": (
            "**Real-Time Stock Portfolio Optimizer**\n\n"
            "LSTM + Prophet + RL ensemble for NSE/BSE India.\n\n"
            "v1.0.0 · FY 2024–25"
        ),
    },
)

# ── Global CSS injection (palette-enforced) ───────────────────────────────────
DEEP_NAVY    = "#1A273A"
SLATE_BLUE   = "#3E4A62"
BURNT_ORANGE = "#C24D2C"
PLATINUM     = "#D9D9D7"

st.markdown(
    f"""
    <style>
    /* ── Root tokens ── */
    :root {{
        --deep-navy:    {DEEP_NAVY};
        --slate-blue:   {SLATE_BLUE};
        --burnt-orange: {BURNT_ORANGE};
        --platinum:     {PLATINUM};
    }}

    /* ── Global background ── */
    .stApp, .main, .block-container {{
        background-color: {DEEP_NAVY} !important;
        color: {PLATINUM} !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {SLATE_BLUE} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {PLATINUM} !important;
    }}

    /* ── Headers ── */
    h1, h2, h3, h4, h5 {{
        color: {PLATINUM} !important;
        font-family: 'Inter', 'DM Sans', sans-serif !important;
    }}
    h1 {{ border-bottom: 2px solid {BURNT_ORANGE}; padding-bottom: 8px; }}

    /* ── Metric tiles ── */
    [data-testid="stMetric"] {{
        background: {SLATE_BLUE};
        border: 1px solid {BURNT_ORANGE}44;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {PLATINUM} !important; opacity: 0.65; }}
    [data-testid="stMetricValue"] {{ color: {BURNT_ORANGE} !important; font-weight: 700; }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: {BURNT_ORANGE} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{ opacity: 0.85 !important; }}

    /* ── Selectbox / Multiselect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {SLATE_BLUE} !important;
        border-color: {BURNT_ORANGE}66 !important;
        color: {PLATINUM} !important;
    }}

    /* ── Slider ── */
    .stSlider > div {{ color: {PLATINUM} !important; }}
    .stSlider [data-testid="stSliderThumb"] {{
        background: {BURNT_ORANGE} !important;
    }}

    /* ── DataFrames ── */
    .stDataFrame {{ background: {SLATE_BLUE} !important; }}
    .stDataFrame th {{
        background: {BURNT_ORANGE} !important;
        color: #fff !important;
    }}
    .stDataFrame td {{ color: {PLATINUM} !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {SLATE_BLUE} !important;
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {PLATINUM} !important;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {BURNT_ORANGE} !important;
    }}

    /* ── Spinner / Progress ── */
    .stSpinner > div {{ border-top-color: {BURNT_ORANGE} !important; }}
    .stProgress > div > div {{
        background: {BURNT_ORANGE} !important;
    }}

    /* ── Divider ── */
    hr {{ border-color: {SLATE_BLUE} !important; }}

    /* ── Number input ── */
    .stNumberInput input {{
        background: {SLATE_BLUE} !important;
        color: {PLATINUM} !important;
        border-color: {BURNT_ORANGE}55 !important;
    }}

    /* ── Toast / info / warning ── */
    .stAlert {{
        background: {SLATE_BLUE} !important;
        color: {PLATINUM} !important;
        border-left: 4px solid {BURNT_ORANGE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Landing page (when no specific page is selected) ──────────────────────────
st.markdown(
    f"""
    <div style='text-align:center; padding:40px 0 20px 0;'>
        <span style='font-size:56px;'>📈</span>
        <h1 style='font-size:2.4rem; font-weight:800; color:{PLATINUM};
                   margin:0; letter-spacing:1px;'>
            Real-Time Stock Portfolio Optimizer
        </h1>
        <p style='font-size:1.1rem; color:{PLATINUM}; opacity:0.65; margin-top:8px;'>
            NSE · BSE · India &nbsp;|&nbsp; LSTM + Prophet + RL Ensemble
            &nbsp;|&nbsp; FY 2024–25
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
pages = [
    ("🏠", "Market Overview",   "Live NIFTY50, SENSEX, sector heatmap, top movers"),
    ("🔮", "Price Forecast",     "LSTM + Prophet + Ensemble forward prediction"),
    ("💼", "Portfolio Builder",  "MPT efficient frontier, allocation, Monte Carlo"),
    ("🤖", "RL Optimizer",       "PPO agent backtest vs NIFTY50 benchmark"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], pages):
    with col:
        st.markdown(
            f"""
            <div style='background:{SLATE_BLUE}; border:1px solid {BURNT_ORANGE}44;
                        border-radius:12px; padding:20px; text-align:center;
                        min-height:130px;'>
                <div style='font-size:28px;'>{icon}</div>
                <div style='font-size:14px; font-weight:700; color:{PLATINUM};
                            margin-top:8px;'>{title}</div>
                <div style='font-size:11px; color:{PLATINUM}; opacity:0.55;
                            margin-top:6px; line-height:1.4;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Select a page from the **sidebar** to begin.", icon="ℹ️")

# ── Quick status bar ──────────────────────────────────────────────────────────
api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
try:
    import httpx
    r = httpx.get(f"{api_url}/api/v1/health/ping", timeout=2)
    api_ok = r.status_code == 200
except Exception:
    api_ok = False

st.markdown(
    f"""
    <div style='margin-top:20px; padding:8px 16px;
                background:{SLATE_BLUE}; border-radius:8px;
                display:flex; gap:24px; align-items:center;'>
        <span style='font-size:11px; color:{PLATINUM}; opacity:0.6;'>System Status:</span>
        <span style='font-size:11px; color:{"#2ECC71" if api_ok else "#E74C3C"};'>
            {'● API Online' if api_ok else '● API Offline'}
        </span>
        <span style='font-size:11px; color:{PLATINUM}; opacity:0.4;'>
            Endpoint: {api_url}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
