"""
Shared pytest fixtures for all test modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def nse_tickers():
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    ]


@pytest.fixture
def synthetic_ohlcv(nse_tickers):
    """2 years of synthetic OHLCV for 8 NSE tickers (multi-level columns)."""
    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    dfs = {}
    for ticker in nse_tickers:
        price = 1000 * np.cumprod(1 + rng.normal(0.0005, 0.018, n))
        dfs[ticker] = pd.DataFrame(
            {
                "Open":   price * (1 - rng.uniform(0, 0.005, n)),
                "High":   price * (1 + rng.uniform(0, 0.010, n)),
                "Low":    price * (1 - rng.uniform(0, 0.010, n)),
                "Close":  price,
                "Volume": rng.integers(100_000, 5_000_000, n, dtype=int).astype(float),
            },
            index=dates,
        )

    return dfs


@pytest.fixture
def synthetic_single_df():
    """Single-ticker DataFrame with Close + engineered features."""
    rng   = np.random.default_rng(99)
    n     = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    price = 1500 * np.cumprod(1 + rng.normal(0.0004, 0.016, n))

    returns = np.diff(price, prepend=price[0]) / price
    df = pd.DataFrame(
        {
            "Close":      price,
            "Open":       price * (1 - rng.uniform(0, 0.003, n)),
            "High":       price * (1 + rng.uniform(0, 0.007, n)),
            "Low":        price * (1 - rng.uniform(0, 0.007, n)),
            "Volume":     rng.integers(5e5, 3e6, n, dtype=int).astype(float),
            "Returns":    returns,
            "SMA_20":     pd.Series(price).rolling(20).mean().values,
            "SMA_50":     pd.Series(price).rolling(50).mean().values,
            "EMA_12":     pd.Series(price).ewm(span=12).mean().values,
            "RSI_14":     50 + rng.normal(0, 10, n),
            "MACD":       rng.normal(0, 2, n),
            "MACD_Signal": rng.normal(0, 2, n),
            "Vol_10":     pd.Series(returns).rolling(10).std().values,
            "Vol_20":     pd.Series(returns).rolling(20).std().values,
            "BB_Upper":   price * 1.02,
            "BB_Lower":   price * 0.98,
            "OBV":        rng.normal(1e6, 1e5, n),
            "Lag_1":      np.roll(returns, 1),
        },
        index=dates,
    )
    return df.fillna(0)
