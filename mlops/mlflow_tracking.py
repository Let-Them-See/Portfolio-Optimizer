"""
MLflow Experiment Tracking Utilities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralised experiment management for all model types.
Model Registry with Staging → Production promotion flow.
Run comparison, artifact management, signature logging.

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.keras
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | MLflow | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/mlflow_tracking.log", mode="a"),
    ],
)
logger = logging.getLogger("MLflowTracking")

# ── MLflow Configuration ──────────────────────────────────────────────────────
TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_server/mlflow.db")
EXPERIMENT_NAME   = os.getenv("MLFLOW_EXPERIMENT_NAME", "portfolio_optimizer_nse")
ARTIFACT_ROOT     = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlflow_artifacts")

mlflow.set_tracking_uri(TRACKING_URI)


def initialise_mlflow() -> MlflowClient:
    """Bootstrap MLflow tracking server and create experiment if needed.

    Returns:
        Initialised MlflowClient object.
    """
    import pathlib
    pathlib.Path("mlflow_server").mkdir(exist_ok=True)
    pathlib.Path("logs").mkdir(exist_ok=True)

    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            exp_id = client.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=ARTIFACT_ROOT,
            )
            logger.info("Created MLflow experiment: %s (id=%s)", EXPERIMENT_NAME, exp_id)
        else:
            logger.info("Using existing experiment: %s (id=%s)", EXPERIMENT_NAME, exp.experiment_id)
    except Exception as exc:
        logger.error("MLflow init error: %s", exc)

    mlflow.set_experiment(EXPERIMENT_NAME)
    return client


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Run Logging Helpers
# ══════════════════════════════════════════════════════════════════════════════

def log_training_run(
    model_type: str,
    ticker: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[List[str]] = None,
    model_obj: Any = None,
    input_example: Any = None,
) -> str:
    """Log a complete training run to MLflow with all required metadata.

    Attaches: git commit hash, data version, timestamp, market context.

    Args:
        model_type:    One of ``"LSTM"``, ``"Prophet"``, ``"PPO"``.
        ticker:        NSE ticker (or ``"universe"`` for RL).
        params:        Hyperparameter dictionary.
        metrics:       Evaluation metric dictionary.
        artifacts:     Optional list of local file paths to log.
        model_obj:     Optional Keras/sklearn model to log as artifact.
        input_example: Optional sample input for model signature.

    Returns:
        MLflow run_id string.
    """
    run_name = f"{model_type}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tags({
            "model_type":   model_type,
            "ticker":       ticker,
            "market":       "NSE",
            "data_version": "v1.0",
            "fiscal_year":  "FY2024-25",
            "timestamp":    datetime.now().isoformat(),
        })

        # Params + metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Artifacts
        if artifacts:
            for path in artifacts:
                if os.path.exists(path):
                    mlflow.log_artifact(path)

        # Model
        if model_obj is not None and model_type == "LSTM":
            try:
                if input_example is not None:
                    sig = infer_signature(input_example, model_obj.predict(input_example))
                    mlflow.keras.log_model(
                        model_obj, artifact_path="model",
                        signature=sig, input_example=input_example,
                    )
                else:
                    mlflow.keras.log_model(model_obj, artifact_path="model")
            except Exception as exc:
                logger.warning("Could not log Keras model: %s", exc)

        run_id = run.info.run_id
        logger.info("Logged run: %s (id=%s)", run_name, run_id)

    return run_id


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Registry
# ══════════════════════════════════════════════════════════════════════════════

def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> str:
    """Register a trained model in the MLflow Model Registry.

    Args:
        run_id:        MLflow run ID containing the model artifact.
        model_name:    Registry name (e.g., ``"NSE_LSTM_TCS"``).
        artifact_path: Artifact subfolder path.

    Returns:
        Model version string.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            name=model_name,
            description=f"NSE equity model — {model_name}",
        )

    version_info = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered: %s v%s", model_name, version_info.version)
    return str(version_info.version)


def promote_model(
    model_name: str,
    version: str,
    stage: str = "Production",
) -> None:
    """Promote a model version to Staging or Production in the Registry.

    Args:
        model_name: Registry model name.
        version:    Version number string.
        stage:      Target stage — ``"Staging"``, ``"Production"``,
                    or ``"Archived"``.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info("Promoted %s v%s → %s", model_name, version, stage)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Run Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_runs(
    n_runs: int = 10,
    metric: str = "test_rmse",
) -> pd.DataFrame:
    """Retrieve and compare the most recent MLflow training runs.

    Args:
        n_runs: Number of most recent runs to compare.
        metric: Primary sort metric.

    Returns:
        DataFrame with run details, params, and key metrics.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        logger.warning("Experiment %s not found.", EXPERIMENT_NAME)
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=n_runs,
    )

    rows = []
    for run in runs:
        rows.append({
            "run_id":    run.info.run_id[:8],
            "model":     run.data.tags.get("model_type", "—"),
            "ticker":    run.data.tags.get("ticker", "—"),
            "test_rmse": run.data.metrics.get("test_rmse", None),
            "test_mape": run.data.metrics.get("test_mape", None),
            "prophet_mape": run.data.metrics.get("prophet_mape", None),
            "signal_sharpe": run.data.metrics.get("signal_sharpe", None),
            "status":    run.info.status,
            "timestamp": datetime.fromtimestamp(
                run.info.start_time / 1000
            ).strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(rows)
    logger.info("Run comparison:\n%s", df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Log Retrain Event
# ══════════════════════════════════════════════════════════════════════════════

def log_retrain_event(
    trigger_reason: str,
    drift_report: Dict,
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
) -> str:
    """Log a model retrain event for audit trail and monitoring.

    Args:
        trigger_reason:  Human-readable reason (e.g., ``"PSI drift detected"``).
        drift_report:    Full drift report dictionary from DriftDetector.
        before_metrics:  Model performance metrics before retraining.
        after_metrics:   Model performance metrics after retraining.

    Returns:
        MLflow run_id for the retrain event.
    """
    with mlflow.start_run(run_name=f"Retrain_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        mlflow.set_tags({
            "event_type":     "retrain",
            "trigger_reason": trigger_reason,
            "market":        "NSE",
        })
        mlflow.log_param("trigger_reason", trigger_reason)

        for k, v in before_metrics.items():
            mlflow.log_metric(f"before_{k}", v)
        for k, v in after_metrics.items():
            mlflow.log_metric(f"after_{k}", v)

        improvement = {
            k: round(before_metrics.get(k, 0) - after_metrics.get(k, 0), 4)
            for k in after_metrics
        }
        for k, v in improvement.items():
            mlflow.log_metric(f"improvement_{k}", v)

        logger.info(
            "Retrain event logged | Trigger: %s | RMSE improvement: %.4f",
            trigger_reason,
            improvement.get("test_rmse", 0),
        )
        return run.info.run_id
