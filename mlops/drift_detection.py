"""
Data & Model Drift Detection Monitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PSI + KS-Test statistical data drift
MAPE/Directional accuracy model drift
Market regime change detection (Bull/Bear/Sideways)

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | DriftDetect | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/drift_detection.log", mode="a"),
    ],
)
logger = logging.getLogger("DriftDetection")

# ── Thresholds ─────────────────────────────────────────────────────────────────
PSI_THRESHOLD        = float(0.2)
KS_PVALUE_THRESHOLD  = float(0.05)
MAPE_DEGRADATION_PCT = float(0.20)   # 20% relative degradation
DIR_ACC_MIN          = float(0.50)   # below this = drift alert


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Population Stability Index (PSI)
# ══════════════════════════════════════════════════════════════════════════════

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.10 → No significant shift
    PSI 0.10–0.20 → Moderate shift (monitor)
    PSI > 0.20  → Significant drift (alert)

    Reference: Siddiqi (2006) — Credit Risk Scorecards, Wiley.

    Args:
        reference: Historical (reference) distribution values.
        current:   New (current) distribution values.
        n_bins:    Number of equal-width buckets.

    Returns:
        PSI score as a float.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Define bins from reference distribution
    breakpoints = np.linspace(ref_clean.min(), ref_clean.max(), n_bins + 1)
    breakpoints[0]  -= 1e-7
    breakpoints[-1] += 1e-7

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    # Avoid division by zero
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(ref_clean))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(cur_clean))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run PSI + KS-test for every feature column between two periods.

    Args:
        reference_df:  Historical feature DataFrame (reference window).
        current_df:    Recent feature DataFrame (current window).
        feature_cols:  Columns to test; defaults to all numeric columns.

    Returns:
        DataFrame with one row per feature: [feature, psi, ks_stat,
        ks_pvalue, drift_flag, severity].
    """
    if feature_cols is None:
        feature_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) < 30 or len(cur) < 30:
            continue

        psi         = compute_psi(ref, cur)
        ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)

        drift_flag = bool(psi > PSI_THRESHOLD or ks_pvalue < KS_PVALUE_THRESHOLD)
        severity   = "HIGH" if psi > 0.25 else ("MEDIUM" if psi > PSI_THRESHOLD else "LOW")

        rows.append({
            "feature":    col,
            "psi":        round(psi, 4),
            "ks_stat":    round(float(ks_stat), 4),
            "ks_pvalue":  round(float(ks_pvalue), 4),
            "drift_flag": drift_flag,
            "severity":   severity,
        })
        if drift_flag:
            logger.warning(
                "DATA DRIFT — %-25s PSI=%.3f  KS_p=%.4f  [%s]",
                col, psi, ks_pvalue, severity,
            )

    drift_df = pd.DataFrame(rows)
    n_flagged = int(drift_df["drift_flag"].sum()) if not drift_df.empty else 0
    logger.info("Data drift scan: %d/%d features flagged.", n_flagged, len(rows))
    return drift_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Performance Drift
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_mape(
    actuals: pd.Series,
    predictions: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Compute rolling MAPE over a sliding window.

    Args:
        actuals:     Series of actual close prices.
        predictions: Series of model predictions aligned by date.
        window:      Rolling window length in trading days.

    Returns:
        Rolling MAPE series (percentage).
    """
    abs_pct_error = ((actuals - predictions).abs() / (actuals.abs() + 1e-8)) * 100
    return abs_pct_error.rolling(window=window).mean()


def detect_model_drift(
    actuals: pd.Series,
    predictions: pd.Series,
    baseline_mape: float,
    baseline_dir_acc: float,
    window: int = 30,
) -> Dict[str, object]:
    """Detect MAPE degradation and directional accuracy decline.

    Args:
        actuals:           Recent actual close prices.
        predictions:       Recent model predictions.
        baseline_mape:     Reference MAPE at model training time.
        baseline_dir_acc:  Reference directional accuracy at training.
        window:            Rolling evaluation window.

    Returns:
        Dictionary with drift flags and current metrics.
    """
    if len(actuals) < 10:
        return {"drift_detected": False, "reason": "insufficient_data"}

    current_mape = float(
        np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    )
    dir_true = (actuals.diff() > 0).dropna()
    dir_pred = (predictions.diff() > 0).dropna()
    min_len  = min(len(dir_true), len(dir_pred))
    current_dir_acc = float(
        np.mean(dir_true.values[-min_len:] == dir_pred.values[-min_len:])
    )

    mape_degradation = (current_mape - baseline_mape) / (baseline_mape + 1e-8)
    mape_drift   = bool(mape_degradation > MAPE_DEGRADATION_PCT)
    dir_drift    = bool(current_dir_acc < DIR_ACC_MIN)
    drift_flag   = mape_drift or dir_drift

    if drift_flag:
        logger.warning(
            "MODEL DRIFT — MAPE: %.2f%% (baseline=%.2f%%, Δ=%.1f%%)  "
            "DirAcc: %.1f%% (min=%.0f%%)",
            current_mape, baseline_mape, mape_degradation * 100,
            current_dir_acc * 100, DIR_ACC_MIN * 100,
        )

    return {
        "drift_detected":    drift_flag,
        "current_mape":      round(current_mape, 3),
        "baseline_mape":     round(baseline_mape, 3),
        "mape_degradation":  round(mape_degradation * 100, 2),
        "current_dir_acc":   round(current_dir_acc * 100, 2),
        "mape_drift":        mape_drift,
        "direction_drift":   dir_drift,
        "timestamp":         datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Market Regime Detection
# ══════════════════════════════════════════════════════════════════════════════

def classify_market_regime(
    nifty_returns: pd.Series,
    window: int = 60,
) -> str:
    """Detect current market regime: Bull, Bear, or Sideways.

    Uses a dual-SMA crossover + volatility filter. Regime classification
    is a prerequisite for concept drift detection in financial ML systems.

    Args:
        nifty_returns: Daily NIFTY50 return series.
        window:        Rolling window for regime estimation.

    Returns:
        One of ``"Bull"``, ``"Bear"``, ``"Sideways"``.
    """
    if len(nifty_returns) < window:
        return "Sideways"

    recent = nifty_returns.tail(window)
    cum_return = float((1 + recent).prod() - 1)
    volatility = float(recent.std() * np.sqrt(252))

    if cum_return > 0.08 and volatility < 0.25:
        regime = "Bull"
    elif cum_return < -0.08 or (cum_return < -0.04 and volatility > 0.25):
        regime = "Bear"
    else:
        regime = "Sideways"

    logger.info(
        "Market Regime: %-10s  CumRet=%.1f%%  AnnVol=%.1f%%",
        regime, cum_return * 100, volatility * 100,
    )
    return regime


def detect_concept_drift(
    nifty_returns: pd.Series,
    previous_regime: str,
) -> Dict[str, object]:
    """Detect concept drift by identifying market regime transitions.

    A regime shift from Bull→Bear or Bear→Bull constitutes concept drift
    and should trigger model retraining.

    Args:
        nifty_returns:    Current NIFTY50 daily return series.
        previous_regime:  The regime the model was trained under.

    Returns:
        Dictionary with drift flag and regime details.
    """
    current_regime = classify_market_regime(nifty_returns)
    regime_changed = current_regime != previous_regime

    if regime_changed:
        logger.warning(
            "CONCEPT DRIFT — Regime change: %s → %s",
            previous_regime, current_regime,
        )

    return {
        "concept_drift":    regime_changed,
        "previous_regime":  previous_regime,
        "current_regime":   current_regime,
        "regime_changed":   regime_changed,
        "timestamp":        datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Full Drift Report
# ══════════════════════════════════════════════════════════════════════════════

def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    actuals: pd.Series,
    predictions: pd.Series,
    baseline_mape: float,
    baseline_dir_acc: float,
    nifty_returns: pd.Series,
    previous_regime: str = "Bull",
) -> Dict:
    """Generate a comprehensive drift monitoring report.

    Combines data drift (PSI+KS), model drift (MAPE+Dir),
    and concept drift (regime change detection).

    Args:
        reference_df:      Reference feature window (~6 months).
        current_df:        Current feature window (~30 days).
        actuals:           Recent actual prices.
        predictions:       Recent model predictions.
        baseline_mape:     MAPE at training time.
        baseline_dir_acc:  Directional accuracy at training time.
        nifty_returns:     Recent NIFTY50 daily returns.
        previous_regime:   Regime label at last training.

    Returns:
        Nested dictionary with all drift signals and metadata.
    """
    data_drift   = detect_data_drift(reference_df, current_df)
    model_drift  = detect_model_drift(actuals, predictions, baseline_mape, baseline_dir_acc)
    concept      = detect_concept_drift(nifty_returns, previous_regime)

    any_drift = (
        bool(data_drift["drift_flag"].any()) if not data_drift.empty else False
        or model_drift.get("drift_detected", False)
        or concept.get("concept_drift", False)
    )

    report = {
        "drift_detected":  any_drift,
        "data_drift":      data_drift.to_dict(orient="records"),
        "model_drift":     model_drift,
        "concept_drift":   concept,
        "scan_timestamp":  datetime.now().isoformat(),
        "recommendation":  "RETRAIN" if any_drift else "STABLE",
    }

    logger.info(
        "Drift Report — Status: %-10s  Data=%s  Model=%s  Concept=%s",
        report["recommendation"],
        bool(data_drift["drift_flag"].any()) if not data_drift.empty else False,
        model_drift.get("drift_detected"),
        concept.get("concept_drift"),
    )
    return report
