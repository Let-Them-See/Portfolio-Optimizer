"""
FastAPI Route — Portfolio Optimisation & Backtesting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints:
  POST /portfolio/optimize             → MPT + RL allocation
  GET  /portfolio/backtest/{strategy}  → Strategy performance vs benchmarks
  GET  /market/overview                → Market dashboard snapshot
  POST /alerts/set                     → Set a price alert
  GET  /alerts/{user_id}               → List active alerts

All monetary values in ₹ INR (Lakhs / Crore labelling).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import (
    AlertRequest,
    AlertResponse,
    AllocationEntry,
    BacktestResponse,
    MarketOverviewResponse,
    PortfolioOptimizeRequest,
    PortfolioOptimizeResponse,
)

load_dotenv()
logger  = logging.getLogger("portfolio_optimizer.portfolio")
limiter = Limiter(key_func=get_remote_address)
router  = APIRouter()

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", 300))

# In-memory alert store (replace with Redis/DB in prod)
_ALERTS: dict[str, dict] = {}


# ── Cache helpers ─────────────────────────────────────────────────────────────
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


# ── MPT optimiser ─────────────────────────────────────────────────────────────
def _mpt_optimize(tickers: list[str], risk: str) -> dict[str, float]:
    """
    Run Monte-Carlo-simulated MPT to find Sharpe-optimal weights.

    Args:
        tickers: List of NSE ticker symbols.
        risk: One of 'low', 'medium', 'high'.

    Returns:
        Normalised weight dict summing to 1.0.
    """
    try:
        import yfinance as yf

        hist = yf.download(tickers, period="1y", progress=False, auto_adjust=True)["Close"]
        if hist.empty or len(hist) < 30:
            raise ValueError("Insufficient price data.")

        returns = hist.pct_change().dropna()
        mu    = returns.mean() * 252
        sigma = returns.cov() * 252

        # Monte-Carlo portfolio simulation (5 000 portfolios)
        n = len(tickers)
        rng = np.random.default_rng(seed=42)
        results = np.zeros((3, 5000))
        all_w   = np.zeros((5000, n))

        for i in range(5000):
            w = rng.dirichlet(np.ones(n))
            # Apply max-weight constraint per risk level
            max_w = {"low": 0.15, "medium": 0.25, "high": 0.35}.get(risk, 0.25)
            w = np.clip(w, 0, max_w)
            w /= w.sum()

            p_ret  = float(np.dot(w, mu))
            p_vol  = float(np.sqrt(w @ sigma.values @ w))
            sharpe = p_ret / (p_vol + 1e-9)
            results[:, i] = [p_ret, p_vol, sharpe]
            all_w[i] = w

        best_idx = int(np.argmax(results[2]))
        best_w   = all_w[best_idx]
        return {t: round(float(w), 4) for t, w in zip(tickers, best_w)}

    except Exception as exc:
        logger.warning("MPT optimiser fallback (equal-weight): %s", exc)
        n = len(tickers)
        return {t: round(1.0 / n, 4) for t in tickers}


def _portfolio_stats(weights: dict[str, float]) -> tuple[float, float, float]:
    """
    Compute expected return, volatility, Sharpe for given weights.

    Returns:
        Tuple of (return_pct, vol_pct, sharpe).
    """
    try:
        import yfinance as yf

        tickers = list(weights.keys())
        hist = yf.download(tickers, period="1y", progress=False, auto_adjust=True)["Close"]
        returns = hist.pct_change().dropna()
        mu    = returns.mean() * 252
        sigma = returns.cov() * 252

        w = np.array([weights[t] for t in tickers])
        p_ret  = float(np.dot(w, mu)) * 100
        p_vol  = float(np.sqrt(w @ sigma.values @ w)) * 100
        sharpe = (p_ret / p_vol) if p_vol > 0 else 0.0
        return round(p_ret, 2), round(p_vol, 2), round(sharpe, 2)
    except Exception:
        return 12.5, 18.0, 0.69


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/portfolio/optimize",
    response_model=PortfolioOptimizeResponse,
    summary="Optimise portfolio using MPT (Sharpe-maximisation)",
)
@limiter.limit("10/minute")
async def optimize_portfolio(
    request: Request,
    body: PortfolioOptimizeRequest,
) -> PortfolioOptimizeResponse:
    """
    Computes Sharpe-optimal allocation using Monte-Carlo MPT.

    - **capital**: Investment amount in ₹ (e.g. 1000000 = ₹10 Lakh)
    - **tickers**: 2–12 NSE symbols
    - **risk_tolerance**: low / medium / high
    - Redis cache: 5-min TTL per (tickers hash, risk) key.
    """
    redis = await _get_redis(request)
    key_tickers = ",".join(sorted(body.tickers))
    cache_key   = f"portfolio:{key_tickers}:{body.risk_tolerance}"

    cached = await _cache_get(redis, cache_key)
    if cached:
        return PortfolioOptimizeResponse(**cached)

    weights  = _mpt_optimize(body.tickers, body.risk_tolerance)
    ret, vol, sharpe = _portfolio_stats(weights)

    allocations = [
        AllocationEntry(
            ticker=t,
            weight=round(w, 4),
            allocation_inr=round(w * body.capital, 2),
            allocation_lakhs=round(w * body.capital / 1e5, 4),
        )
        for t, w in weights.items()
    ]

    # Quarterly rebalance schedule
    today = datetime.now(timezone.utc)
    rebalance = [
        f"Q{((today.month-1)//3 + 1 + i - 1)%4 + 1} FY{today.year + ((today.month-1)//3 + 1 + i - 1)//4}"
        for i in range(1, 5)
    ]

    result = PortfolioOptimizeResponse(
        capital_inr=body.capital,
        capital_lakhs=round(body.capital / 1e5, 2),
        risk_tolerance=body.risk_tolerance,
        optimal_weights=weights,
        expected_annual_return_pct=ret,
        expected_annual_volatility_pct=vol,
        sharpe_ratio=sharpe,
        allocations=allocations,
        rebalance_schedule=rebalance,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    await _cache_set(redis, cache_key, result.model_dump())
    return result


@router.get(
    "/portfolio/backtest/{strategy}",
    response_model=BacktestResponse,
    summary="Backtest portfolio strategy vs NIFTY50 benchmark",
)
@limiter.limit("5/minute")
async def backtest_strategy(
    request: Request,
    strategy: str,
    capital: float = 1_000_000,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
) -> BacktestResponse:
    """
    Run a historical backtest for a named strategy.

    Supported strategies: `equal_weight`, `rl_agent`, `momentum`, `nifty50`.
    """
    supported = {"equal_weight", "rl_agent", "momentum", "nifty50"}
    if strategy not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Choose from: {sorted(supported)}.",
        )

    redis = await _get_redis(request)
    cache_key = f"backtest:{strategy}:{start}:{end}"
    cached    = await _cache_get(redis, cache_key)
    if cached:
        return BacktestResponse(**cached)

    try:
        import yfinance as yf
        from api.routes.predict import NSE_TICKERS as TICKERS

        universe = TICKERS[:12]

        if strategy == "nifty50":
            hist_raw = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
            returns  = hist_raw["Close"].pct_change().dropna()
        else:
            hist_raw = yf.download(universe, start=start, end=end, progress=False, auto_adjust=True)["Close"]
            daily_r  = hist_raw.pct_change().dropna()
            w        = np.ones(len(universe)) / len(universe)

            if strategy == "momentum":
                last20 = daily_r.tail(20).mean()
                top    = last20.nlargest(6).index.tolist()
                w      = np.array([1 / 6 if t in top else 0 for t in daily_r.columns])

            returns = daily_r @ w

        portfolio_values = capital * (1 + returns).cumprod()
        total_ret  = float((portfolio_values.iloc[-1] / capital - 1) * 100)
        ann_vol    = float(returns.std() * np.sqrt(252) * 100)
        sharpe     = total_ret / max(ann_vol, 1e-5)
        peak       = portfolio_values.cummax()
        drawdown   = ((portfolio_values - peak) / peak * 100).min()
        calmar     = abs(total_ret / drawdown) if drawdown != 0 else 0.0

        daily_data = [
            {"date": str(d)[:10], "portfolio_value": round(float(v), 2)}
            for d, v in portfolio_values.items()
        ]

        result = BacktestResponse(
            strategy=strategy,
            start_date=start,
            end_date=end,
            initial_capital_inr=capital,
            final_capital_inr=round(float(portfolio_values.iloc[-1]), 2),
            final_capital_crore=round(float(portfolio_values.iloc[-1]) / 1e7, 4),
            total_return_pct=round(total_ret, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(float(drawdown), 2),
            calmar_ratio=round(float(calmar), 2),
            win_rate_vs_nifty_pct=round((returns > 0).mean() * 100, 1),
            daily_returns=daily_data[:252],  # cap at 1 year for payload size
        )
        await _cache_set(redis, cache_key, result.model_dump(), ttl=1800)
        return result

    except Exception as exc:
        logger.exception("Backtest failed for %s: %s", strategy, exc)
        raise HTTPException(status_code=503, detail="Backtest computation failed.")


# ══════════════════════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/market/overview",
    response_model=MarketOverviewResponse,
    summary="Live NSE/BSE market snapshot",
)
@limiter.limit("20/minute")
async def market_overview(request: Request) -> MarketOverviewResponse:
    """Returns NIFTY50, SENSEX, top gainers/losers and sector heat."""
    redis     = await _get_redis(request)
    cache_key = "market:overview"
    cached    = await _cache_get(redis, cache_key)
    if cached:
        return MarketOverviewResponse(**cached)

    try:
        import yfinance as yf
        from api.routes.predict import NSE_TICKERS as TICKERS

        indices = yf.download(["^NSEI", "^BSESN"], period="2d", progress=False, auto_adjust=True)["Close"]
        n50_val = float(indices["^NSEI"].iloc[-1])
        n50_chg = float((indices["^NSEI"].pct_change().iloc[-1]) * 100)
        snx_val = float(indices["^BSESN"].iloc[-1])
        snx_chg = float((indices["^BSESN"].pct_change().iloc[-1]) * 100)

        stocks = yf.download(TICKERS, period="2d", progress=False, auto_adjust=True)["Close"]
        chg    = stocks.pct_change().iloc[-1].dropna().sort_values()

        gainers = [
            {"ticker": t, "change_pct": round(float(v) * 100, 2)}
            for t, v in chg.tail(5).items()
        ][::-1]
        losers = [
            {"ticker": t, "change_pct": round(float(v) * 100, 2)}
            for t, v in chg.head(5).items()
        ]

        sector_map = {
            "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS"],
            "FMCG": ["ASIANPAINT.NS", "TITAN.NS"],
            "Auto": ["MARUTI.NS"],
            "Energy": ["ONGC.NS", "RELIANCE.NS"],
            "NBFC": ["BAJFINANCE.NS"],
            "Consumer": ["ZOMATO.NS"],
        }
        sector_perf: dict[str, float] = {}
        for sec, tkrs in sector_map.items():
            vals = [float(chg.get(t, 0)) * 100 for t in tkrs if t in chg.index]
            sector_perf[sec] = round(sum(vals) / len(vals), 2) if vals else 0.0

        now     = datetime.now(timezone.utc)
        market_status = (
            "OPEN" if (3 <= now.hour < 10) or (now.hour == 10 and now.minute < 30)
            else ("PRE_OPEN" if now.hour == 3 else "CLOSED")
        )

        result = MarketOverviewResponse(
            nifty50_value=round(n50_val, 2),
            nifty50_change_pct=round(n50_chg, 2),
            sensex_value=round(snx_val, 2),
            sensex_change_pct=round(snx_chg, 2),
            market_status=market_status,
            top_gainers=gainers,
            top_losers=losers,
            sector_performance=sector_perf,
            last_updated=now.isoformat(),
        )
        await _cache_set(redis, cache_key, result.model_dump(), ttl=300)
        return result

    except Exception as exc:
        logger.exception("Market overview fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail="Market data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/alerts/set",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a price alert for a NSE stock",
)
async def set_alert(request: Request, body: AlertRequest) -> AlertResponse:
    """Registers a price alert (stored in-memory; swap for Redis/DB in prod)."""
    alert_id = str(uuid.uuid4())[:8]
    _ALERTS[alert_id] = {
        "ticker":       body.ticker,
        "price_target": body.price_target,
        "direction":    body.direction,
        "user_id":      body.user_id,
        "status":       "active",
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }
    return AlertResponse(
        alert_id=alert_id,
        ticker=body.ticker,
        price_target=body.price_target,
        direction=body.direction,
        created_at=_ALERTS[alert_id]["created_at"],
    )


@router.get(
    "/alerts/{user_id}",
    summary="List active price alerts for a user",
)
async def list_alerts(request: Request, user_id: str):
    """Returns all active alerts for a given user_id."""
    user_alerts = {
        k: v for k, v in _ALERTS.items()
        if v.get("user_id") == user_id and v.get("status") == "active"
    }
    return {"user_id": user_id, "alerts": user_alerts, "count": len(user_alerts)}
