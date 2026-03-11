"""
Prophet Forecaster — NSE Equity Universe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indian Market Seasonalities + Technical Regressors
30-Day Uncertainty-Cone Forecasting
MLflow Artifact Logging

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | Prophet | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/prophet_model.log", mode="a"),
    ],
)
logger = logging.getLogger("ProphetModel")

PROPHET_DIR = Path("models/saved/prophet")
PROPHET_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_HORIZON: int = int(os.getenv("PROPHET_FORECAST_HORIZON", 30))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DataFrame Preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_prophet_df(
    df: pd.DataFrame,
    regressor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert a processed OHLCV DataFrame to Prophet's (ds, y, ...) format.

    Prophet requires exactly two base columns: ``ds`` (datetime) and
    ``y`` (target). Additional regressors are passed in as extra columns.

    Args:
        df:             Processed feature DataFrame indexed by Date.
        regressor_cols: Technical indicator columns to add as regressors.

    Returns:
        DataFrame ready for Prophet.fit(), with ds, y, and any regressors.
    """
    if regressor_cols is None:
        regressor_cols = ["RSI_14", "MACD", "Volume_Ratio", "NIFTY50_return"]

    available = [c for c in regressor_cols if c in df.columns]

    prophet_df = df[["Close"] + available].copy()
    prophet_df = prophet_df.reset_index()
    prophet_df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Fill minor gaps in regressors
    for col in available:
        prophet_df[col].fillna(method="ffill", inplace=True)
        prophet_df[col].fillna(0, inplace=True)

    prophet_df.dropna(subset=["ds", "y"], inplace=True)
    return prophet_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Indian Market Seasonality Building
# ══════════════════════════════════════════════════════════════════════════════

def _add_indian_seasonalities(model: Prophet) -> Prophet:
    """Register Indian equity market custom seasonalities into Prophet.

    Market behaviour in India is substantially shaped by:
    - Union Budget (Feb): Sector rotation and volatility spikes
    - Q1/Q2/Q3/Q4 Earnings Seasons: Momentum and mean-reversion clusters
    - Diwali Rally (Oct-Nov): Retail sentiment upswing in consumer stocks
    - FII / DII quarterly rebalancing: Institutional liquidity cycles

    Args:
        model: Prophet instance to modify in-place.

    Returns:
        Modified Prophet instance with custom seasonalities registered.
    """
    # Budget season: January–February (high fiscal policy uncertainty)
    model.add_seasonality(
        name="budget_season",
        period=365.25,
        fourier_order=5,
        condition_name="is_budget_season",
    )
    # Earnings season: April, July, October, January
    model.add_seasonality(
        name="earnings_season",
        period=91.3125,
        fourier_order=7,
        condition_name="is_earnings_season",
    )
    # Diwali effect: October–November (festive spending boom)
    model.add_seasonality(
        name="diwali_effect",
        period=365.25,
        fourier_order=3,
        condition_name="is_diwali_period",
    )
    # FII quarterly rebalancing: month-end + quarter-end flows
    model.add_seasonality(
        name="fii_quarterly",
        period=91.3125,
        fourier_order=4,
        condition_name="is_fii_rebalance",
    )
    return model


def _add_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append boolean condition columns for Indian seasonality rules.

    Args:
        df: DataFrame with a ``ds`` datetime column.

    Returns:
        DataFrame with four additional boolean condition columns.
    """
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    df["is_budget_season"]  = ((ds.dt.month == 1) | (ds.dt.month == 2)).astype(float)
    df["is_earnings_season"] = (
        ds.dt.month.isin([1, 4, 7, 10])
    ).astype(float)
    df["is_diwali_period"]  = (
        (ds.dt.month == 10) | (ds.dt.month == 11)
    ).astype(float)
    df["is_fii_rebalance"]  = (
        (ds.dt.day >= 25) | (ds.dt.day <= 5)
    ).astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model Build & Training
# ══════════════════════════════════════════════════════════════════════════════

def build_prophet_model(
    regressor_cols: Optional[List[str]] = None,
) -> Prophet:
    """Construct a fully configured Prophet model for NSE equity forecasting.

    Includes Indian market seasonalities, multiplicative mode for
    equity price dynamics, and technical indicator regressors.

    Args:
        regressor_cols: List of external regressor column names.

    Returns:
        Configured (but not yet fitted) Prophet model.
    """
    if regressor_cols is None:
        regressor_cols = ["RSI_14", "MACD", "Volume_Ratio", "NIFTY50_return"]

    model = Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        interval_width=0.90,
        n_changepoints=25,
    )

    # Add Indian-specific seasonalities
    model = _add_indian_seasonalities(model)

    # Add external regressors
    for col in regressor_cols:
        model.add_regressor(col, standardize=True)

    return model


def train_prophet(
    df: pd.DataFrame,
    ticker: str,
    mlflow_experiment: str = "portfolio_optimizer_nse",
) -> Tuple[Prophet, pd.DataFrame, Dict]:
    """Train a Prophet model for one NSE ticker with full MLflow logging.

    Performs:
    - Data preparation + condition columns
    - Model fitting
    - 30-day forward forecast generation
    - Prophet cross-validation (horizon=30d)
    - MLflow artifact logging

    Args:
        df:                Processed feature DataFrame.
        ticker:            NSE ticker symbol.
        mlflow_experiment: MLflow experiment name.

    Returns:
        Tuple of (fitted_model, forecast_df, metrics_dict).
    """
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"Prophet_{ticker}") as run:
        mlflow.set_tags({
            "model_type": "Prophet",
            "ticker":     ticker,
            "market":     "NSE",
            "seasonality": "Indian+Global",
        })

        regressor_cols = ["RSI_14", "MACD", "Volume_Ratio", "NIFTY50_return"]
        prophet_df = prepare_prophet_df(df, regressor_cols)
        prophet_df = _add_condition_columns(prophet_df)

        # 80/20 temporal split for evaluation
        split_idx = int(len(prophet_df) * 0.80)
        train_df  = prophet_df.iloc[:split_idx].copy()
        test_df   = prophet_df.iloc[split_idx:].copy()

        # ── Fit ──────────────────────────────────────────────────────────────
        available_regressors = [c for c in regressor_cols if c in train_df.columns]
        model = build_prophet_model(available_regressors)
        model.fit(train_df)

        # ── Evaluation on holdout ─────────────────────────────────────────────
        future_eval = model.make_future_dataframe(
            periods=len(test_df), freq="B"
        )
        future_eval = _add_condition_columns(future_eval)
        for col in available_regressors:
            future_eval[col] = prophet_df[col].reindex(future_eval.index).fillna(0).values[:len(future_eval)]

        forecast_eval = model.predict(future_eval)
        merged = forecast_eval[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            prophet_df[["ds", "y"]], on="ds", how="inner"
        )
        y_true = merged["y"].values
        y_pred = merged["yhat"].values

        mae   = float(mean_absolute_error(y_true, y_pred))
        rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape  = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        coverage = float(np.mean(
            (y_true >= merged["yhat_lower"].values) &
            (y_true <= merged["yhat_upper"].values)
        ) * 100)

        metrics = {
            "prophet_mae":           mae,
            "prophet_rmse":          rmse,
            "prophet_mape":          mape,
            "uncertainty_coverage":  coverage,
        }
        mlflow.log_params({
            "changepoint_prior_scale": 0.15,
            "seasonality_prior_scale": 10.0,
            "interval_width":          0.90,
            "forecast_horizon":        FORECAST_HORIZON,
            "regressors":              ",".join(available_regressors),
        })
        mlflow.log_metrics(metrics)

        logger.info(
            "%s | MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  Coverage=%.1f%%",
            ticker, mae, rmse, mape, coverage,
        )

        # ── 30-Day Forward Forecast ───────────────────────────────────────────
        future       = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="B")
        future       = _add_condition_columns(future)
        last_vals    = prophet_df[available_regressors].iloc[-1]
        for col in available_regressors:
            future[col] = prophet_df[col].reindex(future.index).fillna(last_vals[col]).values[:len(future)]

        forecast = model.predict(future)

        # ── Save model pickle ─────────────────────────────────────────────────
        pkl_path = PROPHET_DIR / f"prophet_{ticker.replace('.', '_')}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(pkl_path))

        logger.info("Prophet model saved: %s | run_id=%s", pkl_path, run.info.run_id)

    return model, forecast, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Batch Training
# ══════════════════════════════════════════════════════════════════════════════

def train_all_tickers(
    data: Dict[str, pd.DataFrame],
) -> Dict[str, Tuple[Prophet, pd.DataFrame, Dict]]:
    """Train Prophet models for all tickers in the NSE universe.

    Args:
        data: Dictionary mapping ticker → processed DataFrame.

    Returns:
        Dictionary mapping ticker → (model, forecast, metrics).
    """
    results: Dict[str, Tuple[Prophet, pd.DataFrame, Dict]] = {}
    for ticker, df in data.items():
        try:
            logger.info("Training Prophet for %s ...", ticker)
            result = train_prophet(df, ticker)
            results[ticker] = result
        except Exception as exc:
            logger.error("Prophet training failed for %s: %s", ticker, exc)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Inference Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_prophet_model(ticker: str) -> Prophet:
    """Load a saved Prophet pickle for a specific NSE ticker.

    Args:
        ticker: NSE ticker symbol.

    Returns:
        Loaded Prophet model.

    Raises:
        FileNotFoundError: If no pickle exists for the ticker.
    """
    path = PROPHET_DIR / f"prophet_{ticker.replace('.', '_')}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No Prophet model for {ticker} at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_prophet_forecast(
    model: Prophet,
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Generate a forward forecast from a loaded Prophet model.

    Args:
        model:   Fitted Prophet model.
        df:      Most recent feature DataFrame (for regressor continuation).
        horizon: Forecast horizon in trading days.

    Returns:
        Forecast DataFrame with columns [date, price, lower, upper].
    """
    future = model.make_future_dataframe(periods=horizon, freq="B")
    future = _add_condition_columns(future)
    regressor_cols = ["RSI_14", "MACD", "Volume_Ratio", "NIFTY50_return"]
    prophet_df = prepare_prophet_df(df, regressor_cols)

    for col in regressor_cols:
        if col in prophet_df.columns:
            last_val = prophet_df[col].iloc[-1]
            future[col] = prophet_df[col].reindex(future.index).fillna(last_val).values[:len(future)]
        else:
            future[col] = 0.0

    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
    result.rename(columns={
        "ds": "date", "yhat": "price",
        "yhat_lower": "lower", "yhat_upper": "upper",
    }, inplace=True)
    result["model"] = "Prophet"
    return result.reset_index(drop=True)


def get_decomposition(
    model: Prophet,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Prophet trend + seasonality decomposition.

    Args:
        model: Fitted Prophet model.
        df:    Historical feature DataFrame.

    Returns:
        DataFrame with trend, weekly, yearly, and custom seasonality components.
    """
    prophet_df = prepare_prophet_df(df)
    prophet_df = _add_condition_columns(prophet_df)
    regressor_cols = ["RSI_14", "MACD", "Volume_Ratio", "NIFTY50_return"]
    for col in regressor_cols:
        if col in df.columns:
            prophet_df[col] = df[col].values[:len(prophet_df)]

    forecast = model.predict(prophet_df)
    components = ["ds", "trend", "weekly", "yearly"]
    custom_cols = [c for c in ["budget_season", "earnings_season", "diwali_effect"] if c in forecast.columns]
    return forecast[components + custom_cols].copy()
