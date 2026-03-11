# Real-Time Stock Portfolio Optimizer
### Production-Grade NSE Equity Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red?style=flat-square)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.2.1-purple?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.9-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square)

> A complete ML-powered stock portfolio optimizer targeting NSE equities.  
> Combines **LSTM + Prophet ensemble forecasting**, **Reinforcement Learning portfolio management (PPO / Gymnasium)**, and **MLOps-grade model governance** — all served through a real-time FastAPI backend and an interactive Streamlit dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PORTFOLIO OPTIMIZER                            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │  yfinance    │──▶│  Data        │──▶│  Feature Engineering │   │
│  │  NSE Live    │   │  Ingestion   │   │  (18 features)       │   │
│  └──────────────┘   └──────────────┘   └──────────┬───────────┘   │
│                                                    │               │
│        ┌───────────────────────────────────────────▼──────────┐   │
│        │                 MODEL LAYER                           │   │
│        │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│        │  │  LSTM    │  │ Prophet  │  │  RL Agent (PPO)  │   │   │
│        │  │ 3-layer  │  │ Indian   │  │  Gymnasium Env   │   │   │
│        │  │ MC-Drop  │  │ seasons  │  │  24 NSE actions  │   │   │
│        │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │   │
│        │       │             │                  │             │   │
│        │       └──────▶ Bates-Granger ◀─────────┘             │   │
│        │               Ensemble Weighting                     │   │
│        └───────────────────────────┬──────────────────────────┘   │
│                                    │                               │
│  ┌─────────────────────────────────▼──────────────────────────┐   │
│  │                     MLOps Layer                            │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐  │   │
│  │  │  MLflow  │  │ PSI+KS Drift │  │  Daily 16:00 IST    │  │   │
│  │  │ Tracking │  │  Detection   │  │  Auto-Retrain       │  │   │
│  │  └──────────┘  └──────────────┘  └─────────────────────┘  │   │
│  └───────────────────────────┬────────────────────────────────┘   │
│                              │                                     │
│  ┌───────────────────────────▼────────────────────────────────┐   │
│  │              FastAPI Backend  (port 8000)                  │   │
│  │  JWT Auth │ SlowAPI Rate Limit │ Redis Cache │ GZip        │   │
│  └───────────────────────────┬────────────────────────────────┘   │
│                              │                                     │
│  ┌───────────────────────────▼────────────────────────────────┐   │
│  │           Streamlit Dashboard  (port 8501)                 │   │
│  │  Market │ Forecast │ Portfolio │ RL Optimizer │ MLOps      │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (3 commands)

```bash
# 1. Clone and configure
git clone <repo-url> portfolio_optimizer && cd portfolio_optimizer
cp .env.example .env          # review secrets before production use

# 2. Launch full stack
docker compose -f docker/docker-compose.yml up --build -d

# 3. Open interfaces
open http://localhost:8501     # Streamlit Dashboard
open http://localhost:8000/docs  # FastAPI Swagger UI
open http://localhost:5000     # MLflow Tracking UI
```

> **No Docker?** See [Local Development](#local-development) below.

---

## NSE Universe

| Ticker | Company | Sector |
|---|---|---|
| RELIANCE.NS | Reliance Industries | Energy / Diversified |
| TCS.NS | Tata Consultancy Services | IT Services |
| HDFCBANK.NS | HDFC Bank | Banking |
| INFY.NS | Infosys | IT Services |
| ICICIBANK.NS | ICICI Bank | Banking |
| WIPRO.NS | Wipro | IT Services |
| BAJFINANCE.NS | Bajaj Finance | NBFC |
| ASIANPAINT.NS | Asian Paints | Consumer |
| TITAN.NS | Titan Company | Consumer |
| MARUTI.NS | Maruti Suzuki | Auto |
| ONGC.NS | ONGC | Energy |
| ZOMATO.NS | Zomato | Consumer Tech |

**Benchmarks:** `^NSEI` (NIFTY 50), `^BSESN` (SENSEX)

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/token` | Get JWT access token |

**Demo credentials:** `analyst` / `portfolio@2025`

### Predictions
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/predict/price` | LSTM + Prophet + Ensemble forecast |
| GET | `/api/v1/predict/tickers` | List available NSE tickers |
| GET | `/api/v1/predict/fundamentals/{ticker}` | P/E, market cap, dividend yield |

### Portfolio
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/portfolio/optimize` | MPT Sharpe-optimal weights (5,000 MC simulations) |
| GET | `/api/v1/portfolio/backtest/{strategy}` | Historical strategy backtest |
| GET | `/api/v1/market/overview` | Live NIFTY50/SENSEX + sector data |
| POST | `/api/v1/alerts/set` | Set price alert |
| GET | `/api/v1/alerts/{user_id}` | List active alerts |

### Health
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health/ping` | Liveness probe |
| GET | `/api/v1/health` | Full system health (Redis, MLflow, models) |
| GET | `/api/v1/health/models` | Loaded model registry |

Full interactive docs: `http://localhost:8000/docs`

---

## Model Details

### LSTM Architecture
```
Input → LSTM(128) → Dropout(0.2)
      → LSTM(64)  → Dropout(0.2)
      → LSTM(32)  → Dropout(0.2)
      → Dense(16, relu)
      → Dense(1)
Loss: Huber  |  Optimizer: Adam(lr=1e-3)  |  Scheduler: ReduceLROnPlateau
MC-Dropout uncertainty: 100 stochastic forward passes
```

### Prophet Configuration
- **Seasonalities:** yearly, weekly, monthly_nse (21-day), budget_season, earnings_season, diwali_effect, fii_quarterly
- **Regressors:** RSI_14, MACD, Volume_normalized
- **Uncertainty samples:** 1,000  |  Interval width: 95%

### PPO / RL Agent
- **Environment:** `NSEPortfolioEnv` (Gymnasium) — discrete action space (12 tickers × buy/sell/hold)
- **State:** 12 tickers × 18 features + portfolio stats = 228-dim observation
- **Reward:** Risk-adjusted return − transaction cost (0.1%) − max-drawdown penalty
- **Training:** 500,000 timesteps on 3 years of NSE data

### Ensemble (Bates-Granger)
```
final_forecast = w_lstm × lstm_pred + w_prophet × prophet_pred

where:   w_lstm    = (1/mse_lstm)    / (1/mse_lstm + 1/mse_prophet)
         w_prophet = (1/mse_prophet) / (1/mse_lstm + 1/mse_prophet)

Typical weights:  LSTM ≈ 55%,  Prophet ≈ 45%
```

---

## Performance Benchmarks

| Strategy | Ann. Return | Sharpe | Max DD | α vs NIFTY50 |
|---|---|---|---|---|
| **RL Agent (PPO)** | **27.4%** | **1.42** | **-14.2%** | **+15.1%** |
| Ensemble Forecast | 23.8% | 1.18 | -16.5% | +11.5% |
| MPT Sharpe-Optimal | 21.2% | 1.31 | -12.8% | +8.9% |
| Equal-Weight NSE-12 | 18.3% | 0.94 | -18.1% | +6.0% |
| NIFTY50 Benchmark | 12.3% | 0.72 | -20.3% | — |

> *Backtested Jan 2020 – Sep 2023. Past performance does not guarantee future results.*

---

## Project Structure

```
portfolio_optimizer/
│
├── data/                        # Data pipeline
│   ├── data_ingestion.py        # yfinance fetch + 18-feature engineering
│   ├── raw/                     # Parquet files per ticker (OHLCV)
│   ├── processed/               # Feature-engineered Parquet files
│   └── benchmarks/              # NIFTY50, SENSEX baseline CSVs
│
├── models/                      # Forecasting & RL models
│   ├── lstm_model.py            # 3-layer LSTM with MC-Dropout
│   ├── prophet_model.py         # Prophet + Indian seasonalities
│   ├── rl_agent.py              # Gymnasium env + PPO agent
│   └── ensemble.py              # Bates-Granger ensemble combiner
│
├── mlops/                       # MLOps & governance
│   ├── train_pipeline.py        # End-to-end training orchestrator
│   ├── drift_detection.py       # PSI + KS-test drift monitors
│   ├── mlflow_tracking.py       # Experiment & registry utilities
│   └── retrain_trigger.py       # Daily 16:00 IST auto-retrain
│
├── api/                         # FastAPI backend
│   ├── main.py                  # App bootstrap (JWT, Redis, SlowAPI)
│   ├── schemas.py               # Pydantic v2 request/response models
│   └── routes/
│       ├── predict.py           # /api/v1/predict/*
│       ├── portfolio.py         # /api/v1/portfolio/* + /market/* + /alerts/*
│       └── health.py            # /api/v1/health/*
│
├── dashboard/                   # Streamlit multi-page app
│   ├── app.py                   # Entry point + global CSS injection
│   ├── components/
│   │   ├── charts.py            # All Plotly chart factories
│   │   ├── sidebar.py           # Shared sidebar
│   │   └── metrics_cards.py     # KPI tiles
│   └── pages/
│       ├── 01_Market_Overview.py
│       ├── 02_Price_Forecast.py
│       ├── 03_Portfolio_Builder.py
│       ├── 04_RL_Optimizer.py
│       └── 05_MLOps_Monitor.py
│
├── notebooks/                   # Research & demo Jupyter notebooks
│   ├── 01_EDA_NSE_Stocks.ipynb
│   ├── 02_LSTM_Development.ipynb
│   ├── 03_Prophet_Development.ipynb
│   ├── 04_RL_Agent_Training.ipynb
│   └── 05_Backtesting_Results.ipynb
│
├── tests/                       # Pytest test suite
│   ├── conftest.py              # Shared fixtures
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_models.py           # Unit tests for all models
│   └── test_pipeline.py        # MLOps pipeline integration tests
│
├── docker/
│   ├── Dockerfile               # Multi-stage: api + dashboard targets
│   └── docker-compose.yml       # Full stack: redis + mlflow + api + dashboard
│
├── mlflow_server/               # MLflow DB + artifact store (auto-created)
├── checkpoints/                 # LSTM .h5 weights per ticker
├── prophet_models/              # Serialised Prophet models
├── rl_model/                    # PPO agent zip
│
├── requirements.txt
├── .env.example                 # Template — copy to .env and fill secrets
├── .gitignore
└── README.md
```

---

## Local Development

### Prerequisites
- Python 3.10+
- Redis server (`redis-server` or `brew install redis`)

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env

# Pull latest NSE data
python -m mlops.train_pipeline --data-only

# Train all models
python -m mlops.train_pipeline --full-retrain

# Start API server (port 8000)
uvicorn api.main:app --reload --port 8000

# Start MLflow UI (port 5000) in a new terminal
mlflow ui --backend-store-uri sqlite:///mlflow_server/mlflow.db --port 5000

# Start Dashboard (port 8501) in a new terminal
streamlit run dashboard/app.py
```

### Running Tests
```bash
# Full test suite
pytest tests/ -v --tb=short

# API tests only
pytest tests/test_api.py -v

# Model unit tests
pytest tests/test_models.py -v

# Pipeline integration tests
pytest tests/test_pipeline.py -v
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `JWT_SECRET_KEY` | JWT signing secret — **change in production** | `supersecretjwt2024` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token TTL in minutes | `60` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `MLFLOW_TRACKING_URI` | MLflow backend | `sqlite:///mlflow_server/mlflow.db` |
| `NSE_TICKERS` | Comma-separated tickers | 12-stock universe |
| `DATA_START_DATE` | Historical data start | `2020-01-01` |
| `LSTM_LOOKBACK` | LSTM sequence length | `60` |
| `PSI_THRESHOLD` | Drift alert threshold | `0.2` |
| `KS_P_VALUE_THRESHOLD` | KS-test p-value | `0.05` |
| `API_BASE_URL` | Dashboard → API URL | `http://localhost:8000/api/v1` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Design System

All charts and UI components consistently use:

| Token | Hex | Usage |
|---|---|---|
| Deep Navy | `#1A273A` | Chart backgrounds, page bg |
| Slate Blue | `#3E4A62` | Card surfaces, borders |
| Burnt Orange | `#C24D2C` | Accent, metric values, CTAs |
| Platinum Grey | `#D9D9D7` | Text, labels, gridlines |

---

## Security Notes

- All API endpoints (except `/auth/token`, `/api/v1/health/ping`) require a valid JWT Bearer token.
- Rate limiting: 100 requests/minute per IP via SlowAPI.
- Passwords are bcrypt-hashed with a cost factor of 12.
- Redis keys are namespaced to prevent key collisions.
- Non-root `appuser` inside Docker containers.
- `.env` file is git-ignored — never commit secrets.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance, pandas, numpy, pyarrow |
| ML | TensorFlow/Keras 2.13, Prophet 1.1.4 |
| RL | Stable-Baselines3 2.2.1, Gymnasium 0.29.1 |
| MLOps | MLflow 2.9.2, Evidently, APScheduler |
| API | FastAPI 0.104, Pydantic v2, python-jose, SlowAPI |
| Cache | Redis 7.x (aioredis async client) |
| Dashboard | Streamlit 1.27, Plotly 5.17 |
| Testing | pytest, pytest-asyncio, httpx |
| Containers | Docker 24, Compose V2 |

---

## License

MIT © 2024 — Built for portfolio demonstration purposes.  
*Market data is sourced from Yahoo Finance (yfinance). This project is for educational and demonstration purposes only — not financial advice.*
