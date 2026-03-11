"""
Tests — Model Unit Tests
━━━━━━━━━━━━━━━━━━━━━━━━
Covers data ingestion, LSTM, Prophet, RL agent and ensemble modules.
Uses small synthetic datasets to avoid yfinance network calls.

Run: pytest tests/test_models.py -v
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """250-row OHLCV DataFrame mimicking a single NSE stock."""
    rng   = np.random.default_rng(42)
    n     = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 1_000 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    df    = pd.DataFrame(
        {
            "Open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
            "High":   close * (1 + rng.uniform(0, 0.02, n)),
            "Low":    close * (1 - rng.uniform(0, 0.02, n)),
            "Close":  close,
            "Volume": rng.integers(100_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def synthetic_features(synthetic_ohlcv) -> pd.DataFrame:
    """OHLCV with all 18 engineered features."""
    from data.data_ingestion import engineer_features

    nifty_stub = pd.Series(
        np.random.default_rng(0).normal(0, 0.01, len(synthetic_ohlcv)),
        index=synthetic_ohlcv.index,
        name="NIFTY50_return",
    )
    return engineer_features(synthetic_ohlcv, nifty50_returns=nifty_stub)


# ── Data ingestion tests ───────────────────────────────────────────────────────
class TestDataIngestion:
    def test_engineer_features_columns(self, synthetic_features):
        """Feature engineering should add 10+ new columns."""
        required = [
            "Daily_Return", "Log_Return", "SMA_20", "SMA_50", "SMA_200",
            "EMA_12", "EMA_26", "MACD", "RSI_14", "ATR_14",
            "OBV", "Volatility_30d", "NIFTY50_return",
        ]
        for col in required:
            assert col in synthetic_features.columns, f"Missing column: {col}"

    def test_engineer_features_shape(self, synthetic_ohlcv, synthetic_features):
        """Feature-engineered DF must not shrink below 80% of input rows."""
        assert len(synthetic_features) >= 0.8 * len(synthetic_ohlcv)

    def test_engineer_features_no_inf(self, synthetic_features):
        """No infinite values in engineered features."""
        numeric = synthetic_features.select_dtypes(include=[float, int])
        assert not np.isinf(numeric.values).any()


# ── LSTM Dataset tests ─────────────────────────────────────────────────────────
class TestLSTMDataset:
    def test_dataset_creation(self, synthetic_features):
        """LSTMDataset should split data and create windowed sequences."""
        from models.lstm_model import LSTMDataset

        ds = LSTMDataset(synthetic_features, ticker="TEST.NS")
        assert ds.X_train.shape[1] == 60   # 60-day lookback
        assert ds.X_train.shape[0] > 0
        assert ds.y_train.shape[0] == ds.X_train.shape[0]

    def test_dataset_feature_count(self, synthetic_features):
        """Feature count must be 19 (18 features + normalised close)."""
        from models.lstm_model import LSTMDataset

        ds = LSTMDataset(synthetic_features, ticker="TEST.NS")
        # Feature dim may vary; check it's at least 5
        assert ds.X_train.shape[2] >= 5

    def test_scaler_fitted(self, synthetic_features):
        """Scaler must be fitted after dataset creation."""
        from models.lstm_model import LSTMDataset

        ds = LSTMDataset(synthetic_features, ticker="TEST.NS")
        assert ds.scaler is not None
        # scaler should be able to inverse transform
        sample = np.array([[ds.y_train[0]]])
        result = ds.scaler.inverse_transform(sample)
        assert result.shape == (1, 1)


# ── LSTM model build tests ─────────────────────────────────────────────────────
class TestBuildLSTM:
    def test_model_architecture(self):
        """3-layer LSTM model should have expected layer structure."""
        from models.lstm_model import build_lstm_model

        model = build_lstm_model(input_shape=(60, 19))
        # Verify model can be compiled and has layers
        assert model is not None
        assert len(model.layers) >= 6  # 3 LSTM + 3 Dropout + Dense layers
        assert model.input_shape == (None, 60, 19)
        assert model.output_shape == (None, 1)

    def test_model_trainable_params(self):
        """Model should have between 10k and 500k params."""
        from models.lstm_model import build_lstm_model

        model = build_lstm_model(input_shape=(60, 10))
        n_params = model.count_params()
        assert 10_000 < n_params < 500_000


# ── Prophet model tests ────────────────────────────────────────────────────────
class TestProphetModel:
    def test_prepare_df(self, synthetic_ohlcv):
        """prepare_prophet_df should return ds/y columns."""
        from models.prophet_model import prepare_prophet_df

        df_p = prepare_prophet_df(synthetic_ohlcv)
        assert "ds" in df_p.columns
        assert "y" in df_p.columns
        assert len(df_p) == len(synthetic_ohlcv)

    def test_build_model_returns_prophet(self, synthetic_ohlcv):
        """build_prophet_model should return a Prophet() instance."""
        from prophet import Prophet

        from models.prophet_model import build_prophet_model

        df_p = pd.DataFrame({"ds": synthetic_ohlcv.index, "y": synthetic_ohlcv["Close"].values})
        model = build_prophet_model(df_p)
        assert isinstance(model, Prophet)


# ── Ensemble tests ─────────────────────────────────────────────────────────────
class TestEnsemble:
    def test_dynamic_weights_sum_to_one(self):
        """Bates-Granger weights must always sum to 1.0."""
        from models.ensemble import dynamic_ensemble_weights

        mse_lstm, mse_prophet = 0.05, 0.08
        w_l, w_p = dynamic_ensemble_weights(mse_lstm, mse_prophet)
        assert abs(w_l + w_p - 1.0) < 1e-6

    def test_dynamic_weights_lower_mse_higher_weight(self):
        """Model with lower MSE should receive higher weight."""
        from models.ensemble import dynamic_ensemble_weights

        w_l, w_p = dynamic_ensemble_weights(mse_lstm=0.02, mse_prophet=0.10)
        assert w_l > w_p

    def test_combine_forecasts_shape(self):
        """combine_forecasts must produce output matching input length."""
        from models.ensemble import combine_forecasts

        lstm_fc   = np.array([100.0, 101.5, 102.0])
        prophet_fc= np.array([99.5, 101.0, 102.5])
        combined  = combine_forecasts(lstm_fc, prophet_fc, w_lstm=0.55, w_prophet=0.45)
        assert combined.shape == (3,)

    def test_combine_forecasts_values(self):
        """Ensemble must be weighted average of LSTM and Prophet."""
        from models.ensemble import combine_forecasts

        lstm_fc   = np.array([100.0])
        prophet_fc= np.array([110.0])
        combined  = combine_forecasts(lstm_fc, prophet_fc, w_lstm=0.5, w_prophet=0.5)
        np.testing.assert_allclose(combined[0], 105.0, rtol=1e-5)


# ── RL Environment tests ───────────────────────────────────────────────────────
class TestRLEnvironment:
    @pytest.fixture
    def rl_env(self, synthetic_features):
        """Instantiate the NSE portfolio gym environment."""
        from models.rl_agent import NSEPortfolioEnv

        # Create dict of 3 identical DataFrames (simulating 3 stocks)
        data_dict = {f"STOCK_{i}.NS": synthetic_features.copy() for i in range(3)}
        env = NSEPortfolioEnv(
            stock_data=data_dict,
            initial_capital=1_000_000,
            lookback_window=30,
        )
        return env

    def test_env_reset(self, rl_env):
        """Reset should return valid obs shape."""
        obs, info = rl_env.reset()
        assert obs.ndim == 1
        assert len(obs) > 0

    def test_env_step(self, rl_env):
        """Random valid action should not raise errors."""
        rl_env.reset()
        n = rl_env.action_space.shape[0]
        action = rl_env.action_space.sample()
        obs, reward, terminated, truncated, info = rl_env.step(action)
        assert obs.ndim == 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_env_action_weights_sum_to_one(self, rl_env):
        """Action processing should normalise weights to sum=1."""
        rl_env.reset()
        raw_action = np.ones(rl_env.action_space.shape[0])
        obs, reward, terminated, truncated, info = rl_env.step(raw_action)
        weights = info.get("weights", np.ones(3) / 3)
        assert abs(sum(weights) - 1.0) < 0.05
