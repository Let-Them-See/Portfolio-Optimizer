"""
Tests — FastAPI Endpoint Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Covers: auth, predict, portfolio, market, health endpoints.
Uses httpx AsyncClient for real async FastAPI tests.

Run: pytest tests/test_api.py -v
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ── App import ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="module")
async def client():
    """Async test client for the FastAPI app."""
    from api.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as ac:
        yield ac


# ── /auth/token ────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_auth_success(client):
    """Valid credentials should return a JWT."""
    resp = await client.post(
        "/auth/token",
        data={"username": "analyst", "password": "portfolio@2025"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


@pytest.mark.anyio
async def test_auth_failure(client):
    """Wrong password should return 401."""
    resp = await client.post(
        "/auth/token",
        data={"username": "analyst", "password": "wrong-password"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 401


# ── /api/v1/health ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_health_ping(client):
    resp = await client.get("/api/v1/health/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


@pytest.mark.anyio
async def test_health_full(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "api_version" in body
    assert "model_versions" in body
    assert "uptime_seconds" in body


@pytest.mark.anyio
async def test_health_models(client):
    resp = await client.get("/api/v1/health/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "lstm" in body
    assert "prophet" in body
    assert "rl_ppo" in body


# ── /api/v1/predict ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_predict_tickers(client):
    resp = await client.get("/api/v1/predict/tickers")
    assert resp.status_code == 200
    body = resp.json()
    assert "tickers" in body
    assert len(body["tickers"]) > 0
    assert "TCS.NS" in body["tickers"]


@pytest.mark.anyio
async def test_predict_price_schema(client):
    """Predict endpoint should return valid schema (models may not be trained)."""
    resp = await client.post(
        "/api/v1/predict/price",
        json={"ticker": "TCS.NS", "days_ahead": 30, "model": "ensemble"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "TCS.NS"
    assert "current_price" in body
    assert "forecast_generated_at" in body
    assert isinstance(body["model_confidence_score"], float)


@pytest.mark.anyio
async def test_predict_invalid_ticker(client):
    """Non-NSE ticker should fail validation."""
    resp = await client.post(
        "/api/v1/predict/price",
        json={"ticker": "AAPL", "days_ahead": 30, "model": "lstm"},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_days_out_of_range(client):
    """days_ahead > 90 should fail validation."""
    resp = await client.post(
        "/api/v1/predict/price",
        json={"ticker": "TCS.NS", "days_ahead": 200, "model": "lstm"},
    )
    assert resp.status_code == 422


# ── /api/v1/portfolio ───────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_portfolio_optimize(client):
    resp = await client.post(
        "/api/v1/portfolio/optimize",
        json={
            "capital": 1_000_000,
            "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
            "risk_tolerance": "medium",
            "investment_horizon": 252,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "optimal_weights" in body
    assert "sharpe_ratio" in body
    assert "allocations" in body

    total_weight = sum(body["optimal_weights"].values())
    assert abs(total_weight - 1.0) < 0.02  # weights sum to ~1


@pytest.mark.anyio
async def test_portfolio_too_few_tickers(client):
    """Single-ticker portfolio should fail validation."""
    resp = await client.post(
        "/api/v1/portfolio/optimize",
        json={"capital": 100_000, "tickers": ["TCS.NS"], "risk_tolerance": "low"},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_portfolio_backtest_nifty(client):
    resp = await client.get(
        "/api/v1/portfolio/backtest/nifty50",
        params={"capital": 1_000_000, "start": "2023-01-01", "end": "2023-12-31"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "total_return_pct" in body
    assert "sharpe_ratio" in body


@pytest.mark.anyio
async def test_portfolio_backtest_invalid_strategy(client):
    resp = await client.get("/api/v1/portfolio/backtest/unknown_strategy")
    assert resp.status_code == 400


# ── /api/v1/market/overview ────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_market_overview(client):
    resp = await client.get("/api/v1/market/overview")
    # May be 200 or 503 depending on yfinance connectivity
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "nifty50_value" in body
        assert "market_status" in body
        assert body["market_status"] in ("OPEN", "CLOSED", "PRE_OPEN")


# ── /api/v1/alerts ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_alert_set_and_list(client):
    resp = await client.post(
        "/api/v1/alerts/set",
        json={"ticker": "TCS.NS", "price_target": 4000, "direction": "above", "user_id": "test_user"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "alert_id" in body
    assert body["status"] == "active"

    list_resp = await client.get("/api/v1/alerts/test_user")
    assert list_resp.status_code == 200
    list_body = list_resp.json()
    assert list_body["count"] >= 1


# ── Root ────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Real-Time Stock Portfolio Optimizer"
    assert body["version"] == "1.0.0"
