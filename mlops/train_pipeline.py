"""
End-to-End Training Pipeline Orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Coordinates: Data Pull → Feature Eng → LSTM → Prophet → RL → Eval
MLflow experiment lifecycle management

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | TrainPipeline | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("TrainPipeline")

NSE_UNIVERSE: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "TITAN.NS", "MARUTI.NS", "ONGC.NS", "ZOMATO.NS",
]


def run_data_pipeline() -> Dict:
    """Execute data ingestion and feature engineering for NSE universe.

    Returns:
        Dictionary mapping ticker → processed DataFrame.
    """
    logger.info("STEP 1/4 — Data Ingestion & Feature Engineering")
    from data.data_ingestion import run_full_pipeline
    return run_full_pipeline()


def run_lstm_training(
    data: Dict,
    tickers: Optional[List[str]] = None,
) -> Dict:
    """Train LSTM models for specified tickers.

    Args:
        data:    Processed data dictionary.
        tickers: Subset of tickers to train; defaults to full universe.

    Returns:
        Dictionary mapping ticker → (model, dataset, metrics).
    """
    logger.info("STEP 2/4 — LSTM Model Training")
    from models.lstm_model import train_lstm

    tickers = tickers or [t for t in NSE_UNIVERSE if t in data]
    results = {}
    for ticker in tickers:
        if ticker not in data:
            logger.warning("No processed data for %s — skipping.", ticker)
            continue
        try:
            logger.info("Training LSTM: %s", ticker)
            model, dataset, metrics = train_lstm(data[ticker], ticker)
            results[ticker] = (model, dataset, metrics)
        except Exception as exc:
            logger.error("LSTM training failed for %s: %s", ticker, exc)
    return results


def run_prophet_training(
    data: Dict,
    tickers: Optional[List[str]] = None,
) -> Dict:
    """Train Prophet models for specified tickers.

    Args:
        data:    Processed data dictionary.
        tickers: Subset of tickers to train; defaults to full universe.

    Returns:
        Dictionary mapping ticker → (model, forecast, metrics).
    """
    logger.info("STEP 3/4 — Prophet Model Training")
    from models.prophet_model import train_prophet

    tickers = tickers or [t for t in NSE_UNIVERSE if t in data]
    results = {}
    for ticker in tickers:
        if ticker not in data:
            continue
        try:
            logger.info("Training Prophet: %s", ticker)
            model, forecast, metrics = train_prophet(data[ticker], ticker)
            results[ticker] = (model, forecast, metrics)
        except Exception as exc:
            logger.error("Prophet training failed for %s: %s", ticker, exc)
    return results


def run_rl_training(
    data: Dict,
    nifty_returns=None,
    total_timesteps: int = 500_000,
) -> object:
    """Train PPO RL portfolio agent.

    Args:
        data:            Processed data dictionary.
        nifty_returns:   NIFTY50 daily return series.
        total_timesteps: PPO training timesteps.

    Returns:
        Trained PPO model.
    """
    logger.info("STEP 4/4 — RL Agent (PPO) Training")
    from models.rl_agent import train_rl_agent
    return train_rl_agent(data, nifty_returns, total_timesteps=total_timesteps)


def run_full_training_pipeline(
    skip_data: bool = False,
    tickers: Optional[List[str]] = None,
    rl_timesteps: int = 500_000,
) -> Dict:
    """Execute the complete end-to-end training pipeline.

    Sequence:
        1. Data pipeline
        2. LSTM training (all tickers)
        3. Prophet training (all tickers)
        4. RL agent training (full universe)

    Args:
        skip_data:    If True, load from cached Parquet files.
        tickers:      Subset of tickers to train models for.
        rl_timesteps: Total PPO training timesteps.

    Returns:
        Dictionary with keys: data, lstm_results, prophet_results, rl_model.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  NSE Portfolio Optimizer — Full Training Pipeline")
    logger.info("  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    # Step 1 — Data
    if skip_data:
        from data.data_ingestion import load_all_processed
        data = load_all_processed()
    else:
        data = run_data_pipeline()

    # Extract NIFTY returns for RL
    nifty_returns = None
    try:
        from data.data_ingestion import pull_benchmarks, compute_nifty_returns
        benchmarks = pull_benchmarks()
        nifty_returns = compute_nifty_returns(benchmarks)
    except Exception as exc:
        logger.warning("Could not fetch NIFTY benchmark: %s", exc)

    # Step 2 — LSTM
    lstm_results = run_lstm_training(data, tickers)

    # Step 3 — Prophet
    prophet_results = run_prophet_training(data, tickers)

    # Step 4 — RL
    rl_model = run_rl_training(data, nifty_returns, rl_timesteps)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("  Pipeline Complete — Elapsed: %.1f min", elapsed / 60)
    logger.info("  LSTM:    %d models trained", len(lstm_results))
    logger.info("  Prophet: %d models trained", len(prophet_results))
    logger.info("  RL:      PPO agent ready")
    logger.info("=" * 60)

    return {
        "data":            data,
        "lstm_results":    lstm_results,
        "prophet_results": prophet_results,
        "rl_model":        rl_model,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NSE Portfolio Optimizer Training Pipeline")
    parser.add_argument("--skip-data",   action="store_true", help="Use cached data")
    parser.add_argument("--rl-steps",    type=int, default=500_000)
    parser.add_argument("--tickers",     nargs="+", default=None)
    args = parser.parse_args()

    run_full_training_pipeline(
        skip_data=args.skip_data,
        tickers=args.tickers,
        rl_timesteps=args.rl_steps,
    )
