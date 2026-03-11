"""
FastAPI Application — Real-Time Stock Portfolio Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade API with:
  • JWT Bearer auth (python-jose)
  • Rate limiting via SlowAPI (100 req/min)
  • Redis response caching (5-min TTL)
  • Gzip compression middleware
  • Structured CORS for fintech deployments
  • OpenAPI docs at /docs  (Redoc at /redoc)
  • Startup lifespan events for model warm-up

Author : portfolio_optimizer
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

import redis.asyncio as aioredis
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import health, portfolio, predict
from api.schemas import TokenRequest, TokenResponse

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("portfolio_optimizer.api")

# ── Auth config ───────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-super-secret-key")
ALGORITHM  = "HS256"
TOKEN_TTL  = int(os.getenv("JWT_EXPIRY_MINUTES", 60)) * 60  # seconds

oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

# Demo user store (replace with DB in production)
_DEMO_USERS: Dict[str, str] = {
    "analyst": _hash_password(os.getenv("DEMO_PASSWORD", "portfolio@2025")),
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ── Redis client (global, reused via lifespan) ────────────────────────────────
redis_client: aioredis.Redis | None = None
_start_time: float = time.monotonic()


# ── Lifespan events ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; clean up on shutdown."""
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established: %s", redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled.", exc)
        redis_client = None

    # Warm-up: pre-load env vars so first request is fast
    app.state.start_time = time.monotonic()
    app.state.redis = redis_client
    logger.info("Portfolio Optimizer API started.")
    yield

    # Shutdown
    if redis_client:
        await redis_client.aclose()
        logger.info("Redis connection closed.")
    logger.info("Portfolio Optimizer API stopped.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Stock Portfolio Optimizer",
    description=(
        "Production-grade NSE/BSE portfolio management API.\n\n"
        "Features LSTM + Prophet + RL ensemble forecasting, "
        "MPT portfolio optimisation, and full MLOps monitoring.\n\n"
        "**Design palette:** Navy `#1A273A` · Slate `#3E4A62` · "
        "Orange `#C24D2C` · Platinum `#D9D9D7`"
    ),
    version="1.0.0",
    contact={
        "name": "Portfolio Optimizer",
        "email": "support@portfolio-optimizer.in",
    },
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware stack ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Inject X-Process-Time header into every response."""
    t0 = time.monotonic()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.monotonic() - t0) * 1000:.2f}ms"
    return response


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _create_access_token(username: str) -> str:
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str | None:
    """Decode JWT; returns username or None (anonymous allowed on public routes)."""
    if not token:
        return None
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return data.get("sub")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Auth routes ──────────────────────────────────────────────────────────────
@app.post(
    "/auth/token",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Obtain JWT access token",
)
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue a JWT token for valid username/password credentials."""
    hashed = _DEMO_USERS.get(form_data.username)
    if not hashed or not _verify_password(form_data.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = _create_access_token(form_data.username)
    return TokenResponse(access_token=token, expires_in=TOKEN_TTL)


# ── Router mounts ─────────────────────────────────────────────────────────────
app.include_router(health.router,    tags=["Health"],    prefix="/api/v1")
app.include_router(predict.router,   tags=["Forecasting"], prefix="/api/v1")
app.include_router(portfolio.router, tags=["Portfolio"], prefix="/api/v1")


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Real-Time Stock Portfolio Optimizer",
        "version": "1.0.0",
        "exchange": "NSE/BSE India",
        "docs": "/docs",
        "status": "running",
    }


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
        workers=int(os.getenv("API_WORKERS", 2)),
    )
