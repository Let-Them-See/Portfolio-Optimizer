"""
Pydantic Request/Response Schemas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full type-safe API contracts for all endpoints.
Indian market conventions: INR, NSE tickers.

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PricePredictRequest(BaseModel):
    """Request body for price prediction endpoint."""

    ticker: str = Field(
        ...,
        description="NSE ticker symbol e.g. TCS.NS",
        examples=["TCS.NS"],
    )
    days_ahead: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Forecast horizon in trading days (1–90)",
    )
    model: Literal["lstm", "prophet", "ensemble"] = Field(
        default="ensemble",
        description="Model to use for forecasting",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Ensure ticker ends with .NS or .BO (NSE/BSE only)."""
        v = v.upper().strip()
        if not (v.endswith(".NS") or v.endswith(".BO") or v.startswith("^")):
            raise ValueError("Ticker must be an NSE (.NS), BSE (.BO), or index (^) symbol.")
        return v


class PortfolioOptimizeRequest(BaseModel):
    """Request body for portfolio optimisation endpoint."""

    capital: float = Field(
        ...,
        gt=0,
        description="Investment capital in Indian Rupees (₹)",
        examples=[1_000_000],
    )
    tickers: List[str] = Field(
        ...,
        min_length=2,
        max_length=12,
        description="List of NSE ticker symbols",
    )
    risk_tolerance: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Investor risk appetite level",
    )
    investment_horizon: int = Field(
        default=252,
        ge=30,
        le=1260,
        description="Investment horizon in trading days",
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        """Normalise and validate each ticker symbol."""
        return [t.upper().strip() for t in v]


class AlertRequest(BaseModel):
    """Request body for price alert creation."""

    ticker: str = Field(..., description="NSE ticker symbol")
    price_target: float = Field(..., gt=0, description="Target price in ₹")
    direction: Literal["above", "below"] = Field(
        ...,
        description="Alert when price goes above or below target",
    )
    user_id: Optional[str] = Field(default=None)


class TokenRequest(BaseModel):
    """JWT token request (login)."""
    username: str
    password: str


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PriceForecastPoint(BaseModel):
    """Single day price forecast with confidence bounds."""
    date: str
    price: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class PricePredictResponse(BaseModel):
    """Response from price prediction endpoint."""
    ticker: str
    current_price: float
    currency: str = "INR"
    exchange: str = "NSE"
    lstm_forecast: List[PriceForecastPoint] = []
    prophet_forecast: List[PriceForecastPoint] = []
    ensemble_forecast: List[PriceForecastPoint] = []
    model_confidence_score: float
    forecast_generated_at: str


class AllocationEntry(BaseModel):
    """Single stock portfolio allocation entry."""
    ticker: str
    weight: float
    allocation_inr: float
    allocation_lakhs: float


class PortfolioOptimizeResponse(BaseModel):
    """Response from portfolio optimisation endpoint."""
    status: str = "success"
    capital_inr: float
    capital_lakhs: float
    risk_tolerance: str
    optimal_weights: Dict[str, float]
    expected_annual_return_pct: float
    expected_annual_volatility_pct: float
    sharpe_ratio: float
    allocations: List[AllocationEntry]
    rebalance_schedule: List[str]
    generated_at: str


class MarketOverviewResponse(BaseModel):
    """Response from market overview endpoint."""
    nifty50_value: float
    nifty50_change_pct: float
    sensex_value: float
    sensex_change_pct: float
    market_status: Literal["OPEN", "CLOSED", "PRE_OPEN"]
    top_gainers: List[Dict]
    top_losers: List[Dict]
    sector_performance: Dict[str, float]
    last_updated: str


class BacktestResponse(BaseModel):
    """Response from backtesting endpoint."""
    strategy: str
    start_date: str
    end_date: str
    initial_capital_inr: float
    final_capital_inr: float
    final_capital_crore: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_vs_nifty_pct: float
    daily_returns: List[Dict]


class AlertResponse(BaseModel):
    """Response confirming alert creation."""
    alert_id: str
    ticker: str
    price_target: float
    direction: str
    status: str = "active"
    created_at: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    api_version: str
    model_versions: Dict[str, str]
    last_data_refresh: str
    mlflow_status: str
    redis_status: str
    uptime_seconds: float
    timestamp: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
