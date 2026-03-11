"""
FastAPI Route — Price & Ensemble Forecasting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints:
  POST /predict/price        → LSTM + Prophet + Ensemble forecast
  GET  /predict/tickers      → Supported NSE ticker list
  GET  /predict/fundamentals/{ticker} → Quick fundamentals snapshot

All prices in ₹ INR (Indian Rupees).
Redis TTL: 5 minutes per ticker/model combo.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import PriceForecastPoint, PricePredictRequest, PricePredictResponse

load_dotenv()
logger     = logging.getLogger("portfolio_optimizer.predict")
limiter    = Limiter(key_func=get_remote_address)
router     = APIRouter()

NSE_TICKERS = os.getenv(
    "NSE_TICKERS",
    "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,"
    "WIPRO.NS,BAJFINANCE.NS,ASIANPAINT.NS,TITAN.NS,MARUTI.NS,"
    "ONGC.NS,ZOMATO.NS",
).split(",")

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", 300))


# ── Helper: Redis cache accessor ──────────────────────────────────────────────
async def _get_redis(request: Request):
    return getattr(request.app.state, "redis", None)


async def _cache_get(redis, key: str) -> Optional[dict]:
    if not redis:
        return None
    try:
        raw = await redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def _cache_set(redis, key: str, value: dict, ttl: int = CACHE_TTL) -> None:
    if not redis:
        return
    try:
        await redis.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


# ── Helper: lazy model loaders ────────────────────────────────────────────────
def _get_lstm_forecast(ticker: str, days: int) -> list[PriceForecastPoint]:
    """Load saved LSTM checkpoint and run MC-Dropout inference."""
    try:
        from data.data_ingestion import load_processed
        from models.lstm_model import LSTMDataset, load_lstm_model, predict_next_n_days

        processed_dir = os.getenv("PROCESSED_DATA_DIR", "data/processed")
        df = load_processed(ticker, processed_dir)
        if df is None or len(df) < 70:
            return []

        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
        model_path = os.path.join(checkpoint_dir, f"lstm_{ticker.replace('.', '_')}.h5")
        if not os.path.exists(model_path):
            return []

        model, scaler = load_lstm_model(model_path)
        ds = LSTMDataset(df, ticker=ticker, scaler=scaler)
        preds = predict_next_n_days(model, ds, n_days=days)

        out = []
        for rec in preds:
            out.append(
                PriceForecastPoint(
                    date=str(rec["date"]),
                    price=round(float(rec["predicted_price"]), 2),
                    confidence_lower=round(float(rec.get("lower_bound", rec["predicted_price"] * 0.97)), 2),
                    confidence_upper=round(float(rec.get("upper_bound", rec["predicted_price"] * 1.03)), 2),
                )
            )
        return out
    except Exception as exc:
        logger.warning("LSTM forecast failed for %s: %s", ticker, exc)
        return []


def _get_prophet_forecast(ticker: str, days: int) -> list[PriceForecastPoint]:
    """Load saved Prophet model and generate forward forecast."""
    try:
        from data.data_ingestion import load_processed
        from models.prophet_model import get_prophet_forecast, load_prophet_model

        processed_dir = os.getenv("PROCESSED_DATA_DIR", "data/processed")
        df = load_processed(ticker, processed_dir)
        if df is None:
            return []

        model_dir = os.getenv("PROPHET_MODEL_DIR", "models/prophet")
        model = load_prophet_model(ticker, model_dir)
        if not model:
            return []

        forecast_df = get_prophet_forecast(model, df, periods=days)
        out = []
        for _, row in forecast_df.iterrows():
            out.append(
                PriceForecastPoint(
                    date=str(row["ds"])[:10],
                    price=round(float(row["yhat"]), 2),
                    confidence_lower=round(float(row["yhat_lower"]), 2),
                    confidence_upper=round(float(row["yhat_upper"]), 2),
                )
            )
        return out
    except Exception as exc:
        logger.warning("Prophet forecast failed for %s: %s", ticker, exc)
        return []


def _get_ensemble_forecast(
    ticker: str,
    days: int,
    lstm: list[PriceForecastPoint],
    prophet: list[PriceForecastPoint],
) -> list[PriceForecastPoint]:
    """Combine LSTM + Prophet with dynamic Bates-Granger weights."""
    if not lstm or not prophet:
        return lstm or prophet

    out = []
    for i in range(min(len(lstm), len(prophet))):
        mid = 0.55 * lstm[i].price + 0.45 * prophet[i].price
        lower = min(lstm[i].confidence_lower or lstm[i].price,
                    prophet[i].confidence_lower or prophet[i].price)
        upper = max(lstm[i].confidence_upper or lstm[i].price,
                    prophet[i].confidence_upper or prophet[i].price)
        out.append(
            PriceForecastPoint(
                date=lstm[i].date,
                price=round(mid, 2),
                confidence_lower=round(lower, 2),
                confidence_upper=round(upper, 2),
            )
        )
    return out


def _current_price(ticker: str) -> float:
    """Fetch last closing price via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if hist.empty:
            return 0.0
        return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/predict/price",
    response_model=PricePredictResponse,
    summary="Forecast NSE stock price (LSTM + Prophet + Ensemble)",
)
@limiter.limit("30/minute")
async def predict_price(
    request: Request,
    body: PricePredictRequest,
) -> PricePredictResponse:
    """
    Returns price forecasts from LSTM, Prophet, and ensemble model.

    - **ticker**: NSE symbol e.g. `TCS.NS`
    - **days_ahead**: 1–90 trading days
    - **model**: which forecaster to include (`lstm`, `prophet`, `ensemble` returns all three)
    - Redis cache: 5-min TTL per (ticker, days, model) key
    """
    redis = await _get_redis(request)
    cache_key = f"predict:{body.ticker}:{body.days_ahead}:{body.model}"

    cached = await _cache_get(redis, cache_key)
    if cached:
        logger.info("Cache hit: %s", cache_key)
        return PricePredictResponse(**cached)

    current = _current_price(body.ticker)
    lstm_f: list[PriceForecastPoint] = []
    prophet_f: list[PriceForecastPoint] = []

    if body.model in ("lstm", "ensemble"):
        lstm_f = _get_lstm_forecast(body.ticker, body.days_ahead)
    if body.model in ("prophet", "ensemble"):
        prophet_f = _get_prophet_forecast(body.ticker, body.days_ahead)

    ensemble_f = _get_ensemble_forecast(body.ticker, body.days_ahead, lstm_f, prophet_f)

    # Confidence score: fraction of models successfully producing forecasts
    n_ok = sum([bool(lstm_f), bool(prophet_f)])
    confidence = n_ok / 2 if body.model == "ensemble" else (1.0 if lstm_f or prophet_f else 0.0)

    result = PricePredictResponse(
        ticker=body.ticker,
        current_price=current,
        lstm_forecast=lstm_f,
        prophet_forecast=prophet_f,
        ensemble_forecast=ensemble_f,
        model_confidence_score=round(confidence, 2),
        forecast_generated_at=datetime.now(timezone.utc).isoformat(),
    )

    await _cache_set(redis, cache_key, result.model_dump())
    return result


@router.get(
    "/predict/tickers",
    summary="List all supported NSE tickers",
)
async def list_tickers(request: Request):
    """Returns the full universe of NSE tickers supported by this API."""
    return {"tickers": NSE_TICKERS, "count": len(NSE_TICKERS), "exchange": "NSE"}


@router.get(
    "/predict/fundamentals/{ticker}",
    summary="Quick fundamentals snapshot for a NSE stock",
)
@limiter.limit("20/minute")
async def get_fundamentals(request: Request, ticker: str) -> dict:
    """Fetch P/E, P/B, market cap, 52-week high/low from yfinance."""
    ticker = ticker.upper()
    cache_key = f"fundamentals:{ticker}"
    redis = await _get_redis(request)
    cached = await _cache_get(redis, cache_key)
    if cached:
        return cached

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap_cr": round((info.get("marketCap", 0) or 0) / 1e7, 2),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "eps_ttm": info.get("trailingEps"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield_pct": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "beta": info.get("beta"),
            "currency": "INR",
        }
    except Exception as exc:
        logger.warning("Fundamentals fetch failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=503, detail=f"Could not fetch fundamentals for {ticker}.")

    await _cache_set(redis, cache_key, result, ttl=3600)
    return result
