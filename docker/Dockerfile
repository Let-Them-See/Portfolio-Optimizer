# ── Portfolio Optimizer — Docker Multi-Stage Build ────────────────────────────
# Builds a lean production image that serves the FastAPI backend.
# The Streamlit dashboard and MLflow are managed via docker-compose.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps needed to compile numpy / scipy / pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libhdf5-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install wheel
RUN pip install --upgrade pip wheel

# Install Python dependencies into a venv
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: API Runtime ─────────────────────────────────────────────────────
FROM python:3.10-slim AS api

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application source
COPY --chown=appuser:appuser api/        ./api/
COPY --chown=appuser:appuser models/     ./models/
COPY --chown=appuser:appuser mlops/      ./mlops/
COPY --chown=appuser:appuser data/       ./data/

# Create directories for model artefacts and logs
RUN mkdir -p /app/checkpoints /app/prophet_models /app/rl_model \
             /app/mlflow_server /app/data/processed /app/data/raw \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ping || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]


# ─── Stage 3: Dashboard Runtime ───────────────────────────────────────────────
FROM python:3.10-slim AS dashboard

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --chown=appuser:appuser dashboard/  ./dashboard/
COPY --chown=appuser:appuser models/     ./models/
COPY --chown=appuser:appuser data/       ./data/

RUN mkdir -p /app/data/processed && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
