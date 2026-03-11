"""
LSTM Price Predictor — NSE Equity Universe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3-Layer Stacked LSTM with Huber Loss
MLflow Experiment Tracking Integrated

Architecture:
    Input  → LSTM(128, seq) → Dropout(0.2)
           → LSTM(64, seq)  → Dropout(0.2)
           → LSTM(32)       → Dropout(0.2)
           → Dense(16, relu)
           → Dense(1)       [next-day close]

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
load_dotenv()

# ── Keras imports (TF backend) ────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | LSTM | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/lstm_model.log", mode="a"),
    ],
)
logger = logging.getLogger("LSTMModel")

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN: int      = int(os.getenv("LSTM_SEQUENCE_LENGTH", 60))
BATCH_SIZE: int   = int(os.getenv("LSTM_BATCH_SIZE", 32))
MAX_EPOCHS: int   = int(os.getenv("LSTM_MAX_EPOCHS", 100))
PATIENCE: int     = int(os.getenv("LSTM_PATIENCE", 10))
LEARNING_RATE     = 0.001
CHECKPOINT_DIR    = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS: List[str] = [
    "Close", "Volume", "Daily_Return", "Log_Return",
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "MACD_Signal", "RSI_14",
    "Bollinger_Upper", "Bollinger_Lower",
    "ATR_14", "Volatility_30d", "OBV",
    "Volume_Ratio", "Price_Momentum_10d",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Preparation
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDataset:
    """Stateful dataset builder for LSTM sequence modelling.

    Handles normalization, sequence construction, and
    train/validation/test splits following a 70/15/15 regime.

    Args:
        df:       Processed feature DataFrame.
        seq_len:  Lookback window in trading days.
        features: List of feature column names to use.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = SEQ_LEN,
        features: List[str] = FEATURE_COLS,
    ) -> None:
        available = [f for f in features if f in df.columns]
        self.df       = df[available].dropna().copy()
        self.seq_len  = seq_len
        self.features = available
        self.scaler   = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaled: Optional[np.ndarray] = None

    def _scale(self) -> np.ndarray:
        """Fit-transform all features; keep a separate target scaler."""
        self._scaled = self.scaler.fit_transform(self.df.values)
        target_idx = self.features.index("Close")
        self.target_scaler.fit(self.df[["Close"]].values)
        return self._scaled

    def build_sequences(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct (X, y) sequence arrays from scaled data.

        Returns:
            Tuple of X with shape ``(n_samples, seq_len, n_features)``
            and y with shape ``(n_samples,)`` — next-day close price.
        """
        scaled = self._scale()
        target_idx = self.features.index("Close")
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len: i])
            y.append(scaled[i, target_idx])
        return np.array(X), np.array(y)

    def split(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Split sequences into train / validation / test (70/15/15).

        Returns:
            Three tuples: (X_train, y_train), (X_val, y_val), (X_test, y_test).
        """
        X, y = self.build_sequences()
        n = len(X)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)
        return (X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:])

    def inverse_close(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled close price predictions.

        Args:
            scaled_values: Normalized predictions array.

        Returns:
            Actual price values in original scale (INR).
        """
        return self.target_scaler.inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(
    seq_len: int,
    n_features: int,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Construct the 3-layer stacked LSTM architecture.

    Designed for robust next-day price prediction on NSE equities.
    Uses Huber loss for outlier-resistance (earnings surprises, circuit
    breakers) — standard practice on quant desks.

    Args:
        seq_len:       Input sequence length (trading days).
        n_features:    Number of input features.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model ready for training.
    """
    inp = Input(shape=(seq_len, n_features), name="lstm_input")

    x = LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x = Dropout(0.20, name="dropout_1")(x)

    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Dropout(0.20, name="dropout_2")(x)

    x = LSTM(32, return_sequences=False, name="lstm_3")(x)
    x = Dropout(0.20, name="dropout_3")(x)

    x = Dense(16, activation="relu", name="dense_hidden")(x)
    out = Dense(1, name="price_output")(x)

    model = Model(inputs=inp, outputs=out, name="NSE_LSTM_Predictor")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    logger.info("LSTM architecture compiled | params=%s", model.count_params())
    model.summary(print_fn=logger.debug)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Training Harness
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(
    df: pd.DataFrame,
    ticker: str,
    mlflow_experiment: str = "portfolio_optimizer_nse",
) -> Tuple[Model, LSTMDataset, Dict]:
    """Full training cycle with MLflow logging for a single NSE ticker.

    Implements early stopping, learning rate annealing, and model
    checkpointing — mirroring best practices on quantitative research
    infrastructure teams.

    Args:
        df:                Processed feature DataFrame for one ticker.
        ticker:            NSE ticker symbol (used for logging/naming).
        mlflow_experiment: MLflow experiment name.

    Returns:
        Tuple of (trained_model, dataset_object, metrics_dict).
    """
    import git  # optional — graceful fallback
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha[:8]
    except Exception:
        git_hash = "unknown"

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"LSTM_{ticker}") as run:
        # ── Tag run ───────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type":   "LSTM",
            "ticker":       ticker,
            "data_version": "v1.0",
            "git_commit":   git_hash,
            "framework":    "TensorFlow/Keras",
            "market":       "NSE",
        })

        # ── Build dataset ─────────────────────────────────────────────────────
        dataset = LSTMDataset(df, seq_len=SEQ_LEN)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = dataset.split()

        logger.info(
            "%s | Train=%d  Val=%d  Test=%d | Features=%d",
            ticker, len(X_tr), len(X_val), len(X_te), X_tr.shape[2],
        )

        # ── Log hyperparams ───────────────────────────────────────────────────
        params = {
            "seq_len":       SEQ_LEN,
            "batch_size":    BATCH_SIZE,
            "max_epochs":    MAX_EPOCHS,
            "patience":      PATIENCE,
            "learning_rate": LEARNING_RATE,
            "lstm_units":    "128-64-32",
            "dropout":       0.20,
            "loss":          "Huber(delta=1.0)",
            "n_features":    X_tr.shape[2],
            "train_samples": len(X_tr),
        }
        mlflow.log_params(params)

        # ── Build & train model ───────────────────────────────────────────────
        model = build_lstm_model(SEQ_LEN, X_tr.shape[2])
        ckpt_path = str(CHECKPOINT_DIR / f"lstm_{ticker.replace('.', '_')}.h5")

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=1,
            ),
            ModelCheckpoint(
                filepath=ckpt_path, monitor="val_loss",
                save_best_only=True, verbose=0,
            ),
        ]

        class _MLflowCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: dict = None) -> None:  # type: ignore[override]
                if logs:
                    mlflow.log_metrics(
                        {k: float(v) for k, v in logs.items()}, step=epoch
                    )

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks + [_MLflowCallback()],
            verbose=0,
        )

        # ── Evaluate on test set ──────────────────────────────────────────────
        metrics = _evaluate(model, dataset, X_te, y_te)
        mlflow.log_metrics(metrics)
        logger.info(
            "%s | RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  DirAcc=%.1f%%",
            ticker, metrics["test_rmse"], metrics["test_mae"],
            metrics["test_mape"], metrics["directional_accuracy"] * 100,
        )

        # ── Log model artifact ────────────────────────────────────────────────
        input_example = X_tr[:1]
        mlflow.keras.log_model(
            model, artifact_path="lstm_model",
            input_example=input_example,
        )
        mlflow.log_artifact(ckpt_path)

        # ── Permutation feature importance ────────────────────────────────────
        fi = _permutation_importance(model, X_te, y_te, dataset.features)
        fi_df = pd.DataFrame(fi, index=["importance"]).T.sort_values("importance", ascending=False)
        fi_path = f"logs/lstm_feature_importance_{ticker.replace('.', '_')}.csv"
        fi_df.to_csv(fi_path)
        mlflow.log_artifact(fi_path)

        logger.info("MLflow run complete | run_id=%s", run.info.run_id)

    return model, dataset, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Evaluation Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(
    model: Model,
    dataset: LSTMDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE, MAE, MAPE, Directional Accuracy, Sharpe (signal).

    Args:
        model:   Trained Keras model.
        dataset: Dataset object (for inverse-scaling).
        X_test:  Test input sequences.
        y_test:  Test target values (scaled).

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_true = dataset.inverse_close(y_test)
    y_pred = dataset.inverse_close(y_pred_scaled)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    # Directional accuracy
    dir_true = np.diff(y_true) > 0
    dir_pred = np.diff(y_pred) > 0
    dir_acc  = float(np.mean(dir_true == dir_pred))

    # Sharpe of predicted trading signal
    pnl = np.where(dir_pred, np.diff(y_true), -np.diff(y_true))
    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(252))

    return {
        "test_rmse":            rmse,
        "test_mae":             mae,
        "test_mape":            mape,
        "directional_accuracy": dir_acc,
        "signal_sharpe":        sharpe,
    }


def _permutation_importance(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Compute permutation-based feature importance.

    Args:
        model:         Trained Keras model.
        X_test:        Test input array.
        y_test:        Test targets array.
        feature_names: Names of features in input order.
        n_repeats:     Number of permutation repeats per feature.

    Returns:
        Dictionary mapping feature name → mean importance score.
    """
    base_loss = float(np.mean((model.predict(X_test, verbose=0).flatten() - y_test) ** 2))
    importance: Dict[str, float] = {}

    for fi, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            idx = np.random.permutation(len(X_perm))
            X_perm[:, :, fi] = X_perm[idx, :, fi]
            perm_loss = float(
                np.mean((model.predict(X_perm, verbose=0).flatten() - y_test) ** 2)
            )
            scores.append(perm_loss - base_loss)
        importance[name] = float(np.mean(scores))

    return importance


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Inference Utilities
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_n_days(
    model: Model,
    dataset: LSTMDataset,
    n_days: int = 30,
) -> pd.DataFrame:
    """Generate multi-step ahead price forecasts using recursive prediction.

    Args:
        model:   Trained LSTM model.
        dataset: Dataset object with fitted scalers.
        n_days:  Number of trading days to forecast.

    Returns:
        DataFrame with columns [date, price, confidence_lower,
        confidence_upper, model].
    """
    scaled = dataset.scaler.transform(dataset.df.values)
    seq = scaled[-SEQ_LEN:].copy()
    target_idx = dataset.features.index("Close")

    predictions = []
    last_date = dataset.df.index[-1]
    mc_runs = 50  # Monte Carlo dropout for confidence intervals

    for day in range(n_days):
        # MC Dropout inference
        mc_preds = []
        for _ in range(mc_runs):
            inp = seq[np.newaxis, ...]
            p = float(model(inp, training=True).numpy().flatten()[0])
            mc_preds.append(p)

        mean_pred = float(np.mean(mc_preds))
        std_pred  = float(np.std(mc_preds))

        next_seq = seq.copy()
        next_seq = np.roll(next_seq, -1, axis=0)
        next_seq[-1, target_idx] = mean_pred
        seq = next_seq

        next_date = last_date + pd.tseries.offsets.BDay(day + 1)
        price       = dataset.inverse_close(np.array([mean_pred]))[0]
        price_lower = dataset.inverse_close(np.array([mean_pred - 1.96 * std_pred]))[0]
        price_upper = dataset.inverse_close(np.array([mean_pred + 1.96 * std_pred]))[0]

        predictions.append({
            "date":              next_date,
            "price":             round(float(price), 2),
            "confidence_lower":  round(float(price_lower), 2),
            "confidence_upper":  round(float(price_upper), 2),
            "model":             "LSTM",
        })

    return pd.DataFrame(predictions)


def load_lstm_model(ticker: str) -> Model:
    """Load a saved LSTM model checkpoint for a specific ticker.

    Args:
        ticker: NSE ticker symbol.

    Returns:
        Loaded Keras LSTM model.

    Raises:
        FileNotFoundError: If no checkpoint exists for this ticker.
    """
    path = CHECKPOINT_DIR / f"lstm_{ticker.replace('.', '_')}.h5"
    if not path.exists():
        raise FileNotFoundError(f"No LSTM checkpoint for {ticker} at {path}")
    return load_model(str(path), compile=False)


if __name__ == "__main__":
    from data.data_ingestion import load_processed
    ticker = "TCS.NS"
    df = load_processed(ticker)
    model, dataset, metrics = train_lstm(df, ticker)
    print(f"\nTraining complete for {ticker}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
