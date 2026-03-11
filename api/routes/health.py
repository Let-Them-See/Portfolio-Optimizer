"""
FastAPI Route — Health & System Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints:
  GET /health         → Full system health check
  GET /health/ping    → Minimal liveness probe (no dependencies)
  GET /health/models  → Individual model version status
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse

logger = logging.getLogger("portfolio_optimizer.health")
router = APIRouter()

_start_time = time.monotonic()


def _check_redis(request: Request) -> str:
    """Return 'connected' or 'unavailable' based on app state."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return "unavailable"
    return "connected"


def _model_versions() -> dict[str, str]:
    """Scan checkpoint/model directories for available saved models."""
    versions: dict[str, str] = {}
    checkpoint_dir  = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
    prophet_dir     = os.getenv("PROPHET_MODEL_DIR", "models/prophet")
    rl_dir          = os.getenv("RL_MODEL_DIR", "models/rl")

    for d, prefix, label in [
        (checkpoint_dir, "lstm_", "LSTM"),
        (prophet_dir,    "prophet_", "Prophet"),
        (rl_dir,         "ppo_portfolio", "RL_PPO"),
    ]:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.startswith(prefix)]
            versions[label] = f"{len(files)} models loaded" if files else "not trained"
        else:
            versions[label] = "directory missing"

    return versions


def _last_data_refresh() -> str:
    """Return the mtime of the raw data directory as ISO string."""
    raw_dir = os.getenv("RAW_DATA_DIR", "data/raw")
    if os.path.isdir(raw_dir):
        entries = os.listdir(raw_dir)
        if entries:
            latest = max(
                os.path.getmtime(os.path.join(raw_dir, f)) for f in entries
            )
            return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()
    return "never"


def _mlflow_status() -> str:
    """Quick MLflow reachability check."""
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_server/mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        return "connected"
    except Exception:
        return "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/health/ping",
    summary="Liveness probe — no external dependencies",
    include_in_schema=True,
)
async def ping():
    """Minimal heartbeat — returns 200 instantly. Use for k8s liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Full system health check",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Comprehensive system health status including:
    - API uptime
    - Redis connectivity
    - MLflow connectivity
    - Available trained model versions
    - Last data refresh timestamp
    """
    uptime  = time.monotonic() - getattr(request.app.state, "start_time", _start_time)
    redis   = _check_redis(request)
    mlflow_ = _mlflow_status()
    models  = _model_versions()
    refresh = _last_data_refresh()

    overall = "healthy"
    if redis == "unavailable":
        overall = "degraded"
    if all(v in ("not trained", "directory missing") for v in models.values()):
        overall = "warning — no models trained yet"

    return HealthResponse(
        status=overall,
        api_version="1.0.0",
        model_versions=models,
        last_data_refresh=refresh,
        mlflow_status=mlflow_,
        redis_status=redis,
        uptime_seconds=round(uptime, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/health/models",
    summary="Individual model status and version details",
)
async def model_status():
    """Detailed per-model training status and file inventory."""
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
    prophet_dir    = os.getenv("PROPHET_MODEL_DIR", "models/prophet")
    rl_dir         = os.getenv("RL_MODEL_DIR", "models/rl")

    def _scan(directory: str, ext: str) -> list[str]:
        if not os.path.isdir(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(ext)]

    lstm_files    = _scan(checkpoint_dir, ".h5")
    prophet_files = _scan(prophet_dir,    ".pkl")
    rl_files      = _scan(rl_dir,         ".zip")

    return {
        "lstm": {
            "count": len(lstm_files),
            "files": lstm_files,
            "checkpoint_dir": checkpoint_dir,
        },
        "prophet": {
            "count": len(prophet_files),
            "files": prophet_files,
            "model_dir": prophet_dir,
        },
        "rl_ppo": {
            "count": len(rl_files),
            "files": rl_files,
            "model_dir": rl_dir,
        },
        "ensemble": {
            "status": "active" if lstm_files and prophet_files else "requires lstm + prophet",
        },
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
