"""
Data Ingestion & Feature Engineering Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE Equity Universe | yfinance Backend
Production-Grade | Quant Research Desk

Author: Portfolio Optimizer — NSE Edition
Standard: Google-style docstrings, PEP 484 type hints
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/data_ingestion.log", mode="a"),
    ],
)
logger = logging.getLogger("DataIngestion")

# ── Constants ─────────────────────────────────────────────────────────────────
NSE_UNIVERSE: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "TITAN.NS", "MARUTI.NS", "ONGC.NS", "ZOMATO.NS",
]
BENCHMARK_TICKERS: List[str] = ["^NSEI", "^BSESN"]
DATA_START: str = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
DATA_END: str = datetime.today().strftime("%Y-%m-%d")

RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
BENCH_DIR = Path(os.getenv("DATA_BENCHMARKS_DIR", "data/benchmarks"))


# ── Directory Bootstrap ───────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    """Create required data directories if they do not exist."""
    for d in (RAW_DIR, PROC_DIR, BENCH_DIR, Path("logs")):
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Data directory tree verified.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Raw Data Acquisition
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    ticker: str,
    start: str = DATA_START,
    end: str = DATA_END,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Download 5-year OHLCV data for a single NSE ticker via yfinance.

    Args:
        ticker: NSE ticker symbol e.g. ``"TCS.NS"``.
        start:  ISO date string for start of range.
        end:    ISO date string for end of range.
        retries: Number of retry attempts on network failure.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed
        by date, or ``None`` if all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning("Empty data for %s (attempt %d)", ticker, attempt)
                time.sleep(2 ** attempt)
                continue
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            df["Ticker"] = ticker
            logger.info(
                "%-20s | %d rows | %s → %s",
                ticker,
                len(df),
                df.index.min().date(),
                df.index.max().date(),
            )
            return df
        except Exception as exc:
            logger.error("Attempt %d failed for %s: %s", attempt, ticker, exc)
            time.sleep(2 ** attempt)
    logger.critical("All retries exhausted for %s", ticker)
    return None


def pull_nse_universe(
    tickers: List[str] = NSE_UNIVERSE,
) -> Dict[str, pd.DataFrame]:
    """Batch-download OHLCV data for the entire NSE equity universe.

    Args:
        tickers: List of NSE ticker symbols.

    Returns:
        Dictionary mapping ticker → raw OHLCV DataFrame.
    """
    _ensure_dirs()
    raw_data: Dict[str, pd.DataFrame] = {}
    logger.info("Starting NSE universe pull — %d tickers", len(tickers))

    for ticker in tickers:
        df = fetch_ohlcv(ticker)
        if df is not None:
            raw_data[ticker] = df
            out_path = RAW_DIR / f"{ticker.replace('.', '_')}_raw.parquet"
            df.to_parquet(out_path)
            logger.debug("Saved raw: %s", out_path)
        time.sleep(0.5)  # Polite rate-limiting

    logger.info("Universe pull complete — %d/%d succeeded", len(raw_data), len(tickers))
    return raw_data


def pull_benchmarks() -> Dict[str, pd.DataFrame]:
    """Download NIFTY50 and SENSEX benchmark data.

    Returns:
        Dictionary with keys ``"^NSEI"`` and ``"^BSESN"``.
    """
    benchmarks: Dict[str, pd.DataFrame] = {}
    for ticker in BENCHMARK_TICKERS:
        df = fetch_ohlcv(ticker)
        if df is not None:
            benchmarks[ticker] = df
            out_path = BENCH_DIR / f"{ticker.replace('^', '').replace('.', '_')}_benchmark.parquet"
            df.to_parquet(out_path)
    logger.info("Benchmarks downloaded: %s", list(benchmarks.keys()))
    return benchmarks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Technical Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average.

    Args:
        series: Price series.
        window: Rolling window in trading days.

    Returns:
        SMA series of same length.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        series: Price series.
        span:   EMA span parameter.

    Returns:
        EMA series.
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) via Wilder's smoothing.

    Args:
        series: Close price series.
        period: RSI lookback period (default 14).

    Returns:
        RSI values bounded [0, 100].
    """
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line.

    Args:
        series: Close price series.
        fast:   Fast EMA period.
        slow:   Slow EMA period.
        signal: Signal EMA period.

    Returns:
        Tuple of (MACD_line, Signal_line).
    """
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """Compute Bollinger Band upper and lower bounds.

    Args:
        series:  Close price series.
        window:  Rolling window (default 20).
        num_std: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (upper_band, lower_band).
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + num_std * std, sma - num_std * std


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR).

    A volatility indicator used extensively in position sizing
    at institutional desks.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: ATR lookback period.

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    Args:
        close:  Close price series.
        volume: Volume series.

    Returns:
        OBV series (cumulative directional volume).
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full technical feature engineering to raw OHLCV data.

    Computes all 18 features required by downstream ML models.
    Drops rows with NaN from initial rolling windows.

    Args:
        df: Raw OHLCV DataFrame with columns
            [Open, High, Low, Close, Volume].

    Returns:
        Feature-enriched DataFrame ready for model training.
    """
    if df is None or df.empty:
        raise ValueError("Cannot engineer features on empty DataFrame.")

    out = df.copy()
    c = out["Close"]
    h = out["High"]
    lo = out["Low"]
    v = out["Volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    out["Daily_Return"] = c.pct_change()
    out["Log_Return"] = np.log(c / c.shift(1))

    # ── Moving Averages ───────────────────────────────────────────────────────
    out["SMA_20"]  = compute_sma(c, 20)
    out["SMA_50"]  = compute_sma(c, 50)
    out["SMA_200"] = compute_sma(c, 200)
    out["EMA_12"]  = compute_ema(c, 12)
    out["EMA_26"]  = compute_ema(c, 26)

    # ── Momentum Indicators ───────────────────────────────────────────────────
    out["MACD"], out["MACD_Signal"] = compute_macd(c)
    out["RSI_14"] = compute_rsi(c, 14)

    # ── Volatility Bands ──────────────────────────────────────────────────────
    out["Bollinger_Upper"], out["Bollinger_Lower"] = compute_bollinger_bands(c)

    # ── Volatility Metrics ────────────────────────────────────────────────────
    out["ATR_14"]       = compute_atr(h, lo, c, 14)
    out["Volatility_30d"] = out["Log_Return"].rolling(30).std() * np.sqrt(252)

    # ── Volume Indicators ─────────────────────────────────────────────────────
    out["OBV"]          = compute_obv(c, v)
    vol_ma30            = v.rolling(30).mean()
    out["Volume_Ratio"] = v / vol_ma30.replace(0, np.nan)

    # ── Momentum ──────────────────────────────────────────────────────────────
    out["Price_Momentum_10d"] = c.pct_change(10)

    # ── Drop NaN from rolling warmup ─────────────────────────────────────────
    out.dropna(inplace=True)

    logger.debug(
        "Features engineered | shape=%s | NaN=%d",
        out.shape, out.isnull().sum().sum(),
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data Quality Reporting
# ══════════════════════════════════════════════════════════════════════════════

def data_quality_report(
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Generate a structured data quality report for all tickers.

    Checks null counts, date coverage, price anomalies, and
    volume anomalies — consistent with Risk Data Quality standards
    at institutional operations desks.

    Args:
        data: Dictionary mapping ticker → processed DataFrame.

    Returns:
        Quality report DataFrame with one row per ticker.
    """
    rows = []
    for ticker, df in data.items():
        null_total = df.isnull().sum().sum()
        price_anomalies = int(
            ((df["Close"] / df["Close"].shift(1) - 1).abs() > 0.20).sum()
        )
        volume_anomalies = int(
            (df["Volume"] > df["Volume"].rolling(30).mean() * 5).sum()
        )
        rows.append({
            "Ticker":          ticker,
            "Start_Date":      df.index.min().date(),
            "End_Date":        df.index.max().date(),
            "Trading_Days":    len(df),
            "Null_Count":      null_total,
            "Price_Anomalies": price_anomalies,
            "Volume_Anomalies": volume_anomalies,
            "Status":          "CLEAN" if null_total == 0 else "WARN",
        })

    report = pd.DataFrame(rows)
    logger.info("\n%s", report.to_string(index=False))
    return report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NIFTY50 Return Regressor (for Prophet)
# ══════════════════════════════════════════════════════════════════════════════

def compute_nifty_returns(benchmark_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """Extract NIFTY50 daily returns for use as a Prophet regressor.

    Args:
        benchmark_data: Dictionary containing ``"^NSEI"`` data.

    Returns:
        Series of NIFTY50 daily returns indexed by date.
    """
    nifty = benchmark_data.get("^NSEI")
    if nifty is None or nifty.empty:
        logger.warning("NIFTY50 data unavailable — returning zero series.")
        return pd.Series(dtype=float, name="NIFTY50_return")
    returns = nifty["Close"].pct_change().rename("NIFTY50_return")
    returns.dropna(inplace=True)
    return returns


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Full Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline() -> Dict[str, pd.DataFrame]:
    """Execute the complete data ingestion and feature engineering pipeline.

    Steps:
        1. Pull raw OHLCV for all 12 NSE tickers
        2. Pull NIFTY50 + SENSEX benchmarks
        3. Apply technical feature engineering to each ticker
        4. Save processed Parquet files
        5. Generate and log data quality report

    Returns:
        Dictionary mapping ticker → fully processed DataFrame.
    """
    logger.info("=" * 60)
    logger.info("  NSE Portfolio Optimizer — Data Pipeline v1.0")
    logger.info("  Universe: %d stocks | Period: 5 Years", len(NSE_UNIVERSE))
    logger.info("=" * 60)

    # Step 1 & 2 — Pull raw data
    raw_data = pull_nse_universe()
    benchmark_data = pull_benchmarks()
    nifty_returns = compute_nifty_returns(benchmark_data)

    # Step 3 & 4 — Feature engineering + save
    processed: Dict[str, pd.DataFrame] = {}
    for ticker, df in raw_data.items():
        try:
            feat_df = engineer_features(df)
            # Merge NIFTY50 returns as market-beta regressor
            feat_df = feat_df.join(nifty_returns, how="left")
            feat_df["NIFTY50_return"].fillna(0, inplace=True)

            out_path = PROC_DIR / f"{ticker.replace('.', '_')}_processed.parquet"
            feat_df.to_parquet(out_path)
            processed[ticker] = feat_df
            logger.info("Processed & saved: %-20s | %d features", ticker, feat_df.shape[1])

        except Exception as exc:
            logger.error("Feature engineering failed for %s: %s", ticker, exc)

    # Step 5 — Quality report
    _ = data_quality_report(processed)

    logger.info("Pipeline complete — %d tickers ready for modelling.", len(processed))
    return processed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Load Helpers (for downstream modules)
# ══════════════════════════════════════════════════════════════════════════════

def load_processed(ticker: str) -> pd.DataFrame:
    """Load a previously processed ticker Parquet file.

    Args:
        ticker: NSE ticker symbol e.g. ``"TCS.NS"``.

    Returns:
        Processed DataFrame with all engineered features.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
    """
    path = PROC_DIR / f"{ticker.replace('.', '_')}_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found for {ticker}. Run run_full_pipeline() first."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def load_all_processed() -> Dict[str, pd.DataFrame]:
    """Load all available processed Parquet files.

    Returns:
        Dictionary mapping ticker → processed DataFrame.
    """
    data: Dict[str, pd.DataFrame] = {}
    for path in sorted(PROC_DIR.glob("*_processed.parquet")):
        ticker = path.stem.replace("_processed", "").replace("_", ".", 1)
        # Handle tickers like ASIANPAINT.NS (only first _ → .)
        # Re-derive ticker from filename more robustly
        raw_name = path.stem.replace("_processed", "")
        # Convert back: last _NS → .NS
        if raw_name.endswith("_NS"):
            ticker = raw_name[:-3] + ".NS"
        elif raw_name.endswith("_BO"):
            ticker = raw_name[:-3] + ".BO"
        else:
            ticker = raw_name
        data[ticker] = pd.read_parquet(path)
    logger.info("Loaded %d processed datasets.", len(data))
    return data


if __name__ == "__main__":
    processed_data = run_full_pipeline()
    print(f"\nReady: {list(processed_data.keys())}")
