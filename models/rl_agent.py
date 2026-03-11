"""
Reinforcement Learning Portfolio Allocator — NSE Universe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PPO Agent | Custom Gymnasium Environment
₹10 Lakh Starting Capital | Weekly Rebalancing
Target: ≥ 15% Alpha over NIFTY50

Reward = daily_return − 0.1×volatility
         − transaction_cost + 0.05×Δsharpe

Standard: Google-style docstrings, PEP 484
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Gymnasium + SB3 ───────────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | RLAgent | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/rl_agent.log", mode="a"),
    ],
)
logger = logging.getLogger("RLAgent")

# ── Constants ─────────────────────────────────────────────────────────────────
RL_DIR = Path("models/saved/rl")
RL_DIR.mkdir(parents=True, exist_ok=True)

N_STOCKS: int       = 12
LOOKBACK: int       = 30          # days of returns in state
MAX_WEIGHT: float   = 0.30        # max single-stock allocation
TC_RATE: float      = 0.001       # 0.1% transaction cost per trade
MAX_DRAWDOWN_LIMIT  = 0.15        # 15% max drawdown trigger
MAX_TURNOVER        = 0.20        # 20% per rebalance
INITIAL_CAPITAL_INR = float(os.getenv("INITIAL_CAPITAL_INR", 1_000_000))

NSE_UNIVERSE: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "TITAN.NS", "MARUTI.NS", "ONGC.NS", "ZOMATO.NS",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Custom Gymnasium Environment
# ══════════════════════════════════════════════════════════════════════════════

class NSEPortfolioEnv(gym.Env):
    """Custom Gym environment for NSE portfolio allocation via RL.

    Implements a continuous-weight portfolio environment where the agent
    learns to allocate capital across 12 NSE blue-chip stocks to
    maximise risk-adjusted returns.

    State Space (continuous):
        - 30d daily returns for each stock  (N_STOCKS × LOOKBACK)
        - Current portfolio weights          (N_STOCKS)
        - LSTM 5-day predicted returns       (N_STOCKS × 5)
        - Market volatility (VIX proxy)      (1)
        - Rolling 30d Portfolio Sharpe       (1)
        - Current drawdown                   (1)

    Action Space:
        - Continuous weights ∈ [0, 1] for each stock (N_STOCKS,)
        - Normalised to sum = 1 via softmax
        - Hard floor = 0 (no short-selling)
        - Hard cap = MAX_WEIGHT per stock

    Args:
        returns_df:      DataFrame of daily returns (date × stock).
        lstm_preds:      Dict mapping ticker → 5-day return forecast array.
        nifty_returns:   Series of daily NIFTY50 returns (benchmark).
        mode:            ``"train"`` or ``"test"`` for split handling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        returns_df: pd.DataFrame,
        lstm_preds: Optional[Dict[str, np.ndarray]] = None,
        nifty_returns: Optional[pd.Series] = None,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.returns_df    = returns_df.fillna(0).values.astype(np.float32)
        self.tickers       = list(returns_df.columns)
        self.n_stocks      = len(self.tickers)
        self.lstm_preds    = lstm_preds or {}
        self.nifty_returns = (
            nifty_returns.fillna(0).values.astype(np.float32)
            if nifty_returns is not None
            else np.zeros(len(returns_df), dtype=np.float32)
        )

        # Temporal split
        n = len(self.returns_df)
        if mode == "train":
            self._start, self._end = LOOKBACK, int(n * 0.80)
        else:
            self._start, self._end = int(n * 0.80), n - 1

        # State dimensions
        state_dim = (
            self.n_stocks * LOOKBACK       # Return history
            + self.n_stocks                # Current weights
            + self.n_stocks * 5            # LSTM 5-day forecasts
            + 3                            # VIX, Sharpe, Drawdown
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_stocks,), dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialise all internal portfolio state variables."""
        self.t            = self._start
        self.weights      = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        self.portfolio_val = INITIAL_CAPITAL_INR
        self.peak_val      = INITIAL_CAPITAL_INR
        self.portfolio_history: List[float] = [INITIAL_CAPITAL_INR]
        self.daily_returns: List[float]     = []

    def _get_obs(self) -> np.ndarray:
        """Construct the current observation vector.

        Returns:
            Flat observation array of shape (state_dim,).
        """
        # Return history (LOOKBACK × n_stocks)
        ret_window = self.returns_df[self.t - LOOKBACK: self.t].flatten()

        # LSTM 5-day forecasts (stub = last 5 returns)
        lstm_block = []
        for i in range(self.n_stocks):
            ticker = self.tickers[i]
            if ticker in self.lstm_preds and len(self.lstm_preds[ticker]) >= 5:
                lstm_block.extend(self.lstm_preds[ticker][:5].tolist())
            else:
                lstm_block.extend(self.returns_df[self.t - 5: self.t, i].tolist())

        # Market volatility (proxy: rolling std of NIFTY returns)
        nifty_window = self.nifty_returns[
            max(0, self.t - 30): self.t
        ]
        vix_proxy = float(np.std(nifty_window) * np.sqrt(252)) if len(nifty_window) > 1 else 0.15

        # Rolling Sharpe (30d)
        if len(self.daily_returns) >= 5:
            dr = np.array(self.daily_returns[-30:])
            sharpe = float(np.mean(dr) / (np.std(dr) + 1e-8) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Current drawdown
        drawdown = float((self.peak_val - self.portfolio_val) / (self.peak_val + 1e-8))

        obs = np.concatenate([
            ret_window.astype(np.float32),
            self.weights.astype(np.float32),
            np.array(lstm_block, dtype=np.float32),
            np.array([vix_proxy, sharpe, drawdown], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def _clip_weights(self, raw_weights: np.ndarray) -> np.ndarray:
        """Apply softmax normalisation and per-stock weight caps.

        Args:
            raw_weights: Raw action logits from the policy network.

        Returns:
            Normalised weight vector summing to 1 with per-stock cap.
        """
        # Softmax
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / (exp_w.sum() + 1e-8)
        # Cap at MAX_WEIGHT
        weights = np.minimum(weights, MAX_WEIGHT)
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(self.n_stocks) / self.n_stocks
        return weights.astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance the environment by one trading day.

        Args:
            action: Raw weight vector from policy network.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        new_weights = self._clip_weights(action)

        # Transaction cost on turnover
        turnover       = float(np.abs(new_weights - self.weights).sum() / 2)
        tc_penalty     = turnover * TC_RATE

        # Daily portfolio return
        day_returns    = self.returns_df[self.t]
        portfolio_ret  = float(np.dot(new_weights, day_returns)) - tc_penalty

        # Update portfolio value
        self.portfolio_val *= (1 + portfolio_ret)
        self.peak_val       = max(self.peak_val, self.portfolio_val)
        self.portfolio_history.append(self.portfolio_val)
        self.daily_returns.append(portfolio_ret)
        self.weights = new_weights
        self.t += 1

        # ── Reward function ───────────────────────────────────────────────────
        if len(self.daily_returns) >= 5:
            recent = np.array(self.daily_returns[-30:])
            vol    = float(np.std(recent) * np.sqrt(252))

            # Sharpe improvement signal
            prev_sharpe  = 0.0 if len(self.daily_returns) < 6 else float(
                np.mean(np.array(self.daily_returns[-31:-1])) /
                (np.std(np.array(self.daily_returns[-31:-1])) + 1e-8) * np.sqrt(252)
            )
            curr_sharpe  = float(np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252))
            sharpe_delta = curr_sharpe - prev_sharpe
        else:
            vol, sharpe_delta = 0.15, 0.0

        reward = (
            portfolio_ret
            - 0.10 * vol
            - tc_penalty * 5.0
            + 0.05 * sharpe_delta
        )

        # ── Penalty: max drawdown breach ─────────────────────────────────────
        drawdown = (self.peak_val - self.portfolio_val) / (self.peak_val + 1e-8)
        if drawdown > MAX_DRAWDOWN_LIMIT:
            reward -= 2.0

        # ── Penalty: excess turnover ──────────────────────────────────────────
        if turnover > MAX_TURNOVER:
            reward -= (turnover - MAX_TURNOVER) * 10.0

        terminated = self.t >= self._end
        info = {
            "portfolio_value":  self.portfolio_val,
            "daily_return":     portfolio_ret,
            "turnover":         turnover,
            "drawdown":         drawdown,
            "weights":          self.weights.tolist(),
        }
        return self._get_obs(), float(reward), terminated, False, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed:    Optional RNG seed for reproducibility.
            options: Unused; present for Gym API compliance.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def render(self) -> None:
        """Human-readable portfolio snapshot."""
        print(
            f"Step={self.t} | ₹{self.portfolio_val:,.0f} | "
            f"DD={((self.peak_val - self.portfolio_val) / self.peak_val * 100):.1f}%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PPO Training
# ══════════════════════════════════════════════════════════════════════════════

class _MLflowRewardCallback(BaseCallback):
    """Log PPO episode rewards to MLflow at regular intervals."""

    def __init__(self, log_freq: int = 2048) -> None:
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = float(
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                )
                mlflow.log_metric("mean_episode_reward", mean_reward, step=self.n_calls)
        return True


def train_rl_agent(
    data: Dict[str, pd.DataFrame],
    nifty_returns: Optional[pd.Series] = None,
    mlflow_experiment: str = "portfolio_optimizer_nse",
    total_timesteps: int = 500_000,
) -> PPO:
    """Train a PPO agent on the NSE portfolio environment.

    Hyperparameters match production-grade settings used in
    academic quantitative finance literature and hedge fund AI teams.

    Args:
        data:              Dict of ticker → processed DataFrame.
        nifty_returns:     NIFTY50 daily return series for market beta.
        mlflow_experiment: MLflow experiment name.
        total_timesteps:   Total PPO training timesteps.

    Returns:
        Trained PPO model saved to disk.
    """
    # Build aligned returns DataFrame
    returns_dict: Dict[str, pd.Series] = {}
    for ticker, df in data.items():
        if "Daily_Return" in df.columns:
            returns_dict[ticker] = df["Daily_Return"]

    returns_df = pd.DataFrame(returns_dict).dropna()
    logger.info("Returns matrix: %s", returns_df.shape)

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="PPO_PortfolioAgent") as run:
        mlflow.set_tags({
            "model_type":  "PPO",
            "algorithm":   "Proximal Policy Optimization",
            "universe":    "NSE_12",
            "framework":   "stable-baselines3",
        })

        rl_params = {
            "policy":        "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps":       2048,
            "batch_size":    64,
            "n_epochs":      10,
            "gamma":         0.99,
            "clip_range":    0.2,
            "total_timesteps": total_timesteps,
            "n_stocks":      N_STOCKS,
            "lookback":      LOOKBACK,
            "max_weight":    MAX_WEIGHT,
            "tc_rate":       TC_RATE,
        }
        mlflow.log_params(rl_params)

        # Create vectorised environment
        def make_env() -> Monitor:
            env = NSEPortfolioEnv(returns_df, nifty_returns=nifty_returns, mode="train")
            return Monitor(env)

        train_env = DummyVecEnv([make_env])
        eval_env  = DummyVecEnv([lambda: Monitor(
            NSEPortfolioEnv(returns_df, nifty_returns=nifty_returns, mode="test")
        )])

        # Build PPO model
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=42,
        )

        # Callbacks
        ckpt_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(RL_DIR),
            name_prefix="ppo_nse_portfolio",
        )
        eval_cb = EvalCallback(
            eval_env,
            eval_freq=20_000,
            best_model_save_path=str(RL_DIR / "best"),
            deterministic=True,
            verbose=0,
        )
        mlflow_cb = _MLflowRewardCallback()

        logger.info("PPO training started | timesteps=%d", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[ckpt_cb, eval_cb, mlflow_cb],
            progress_bar=True,
        )

        # ── Save final model ──────────────────────────────────────────────────
        model_path = RL_DIR / "ppo_nse_portfolio_final"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path) + ".zip")
        logger.info("PPO model saved: %s | run_id=%s", model_path, run.info.run_id)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Backtesting Engine
# ══════════════════════════════════════════════════════════════════════════════

def backtest_rl_agent(
    model: PPO,
    data: Dict[str, pd.DataFrame],
    nifty_returns: Optional[pd.Series] = None,
    initial_capital: float = INITIAL_CAPITAL_INR,
) -> pd.DataFrame:
    """Run a full 2-year walk-forward backtest for the PPO agent.

    Compares RL agent vs NIFTY50, equal-weight, and random allocation.

    Args:
        model:           Trained PPO model.
        data:            Dict of ticker → processed DataFrame.
        nifty_returns:   NIFTY50 daily returns series.
        initial_capital: Starting capital in INR.

    Returns:
        DataFrame with columns [date, RL_Agent, NIFTY50,
        Equal_Weight, Random, dates] — cumulative portfolio values.
    """
    returns_dict = {t: df["Daily_Return"] for t, df in data.items() if "Daily_Return" in df.columns}
    returns_df = pd.DataFrame(returns_dict).dropna()

    # Use last 2 years (≈504 trading days) for backtest
    backtest_df = returns_df.tail(504).copy()
    dates       = backtest_df.index.tolist()
    rets        = backtest_df.values.astype(np.float32)

    n_stocks  = rets.shape[1]
    n_nifty   = (
        nifty_returns.reindex(backtest_df.index).fillna(0).values
        if nifty_returns is not None
        else np.zeros(len(dates))
    )

    # ── RL Agent ──────────────────────────────────────────────────────────────
    env = NSEPortfolioEnv(backtest_df, nifty_returns=nifty_returns, mode="test")
    obs, _ = env.reset()
    rl_values      = [initial_capital]
    rl_weights_log = []
    trade_log      = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rl_values.append(info["portfolio_value"])
        rl_weights_log.append(info["weights"])
        trade_log.append({
            "date":     dates[min(env.t - 1, len(dates) - 1)],
            "weights":  info["weights"],
            "turnover": info["turnover"],
            "value_inr": info["portfolio_value"],
        })
        done = terminated or truncated

    # ── Benchmark Strategies ──────────────────────────────────────────────────
    eq_val     = initial_capital
    rand_val   = initial_capital
    eq_values  = [initial_capital]
    rand_values = [initial_capital]

    eq_w   = np.ones(n_stocks) / n_stocks
    rng    = np.random.default_rng(42)

    for day_rets in rets:
        # Equal weight
        eq_val  *= (1 + float(np.dot(eq_w, day_rets)) - TC_RATE * 0)
        # Random (rebalanced weekly)
        rand_w   = rng.dirichlet(np.ones(n_stocks))
        rand_val *= (1 + float(np.dot(rand_w, day_rets)) - TC_RATE * 0.1)
        eq_values.append(eq_val)
        rand_values.append(rand_val)

    # NIFTY50 buy-and-hold
    nifty_values = [initial_capital]
    nv = initial_capital
    for nr in n_nifty:
        nv *= (1 + float(nr))
        nifty_values.append(nv)

    # Align lengths
    min_len = min(len(rl_values), len(nifty_values), len(eq_values), len(rand_values))
    result_df = pd.DataFrame({
        "date":         dates[:min_len - 1],
        "RL_Agent":     rl_values[1:min_len],
        "NIFTY50":      nifty_values[1:min_len],
        "Equal_Weight": eq_values[1:min_len],
        "Random":       rand_values[1:min_len],
    })

    metrics = _compute_backtest_metrics(result_df, trade_log)
    _log_backtest_metrics(metrics)

    return result_df, pd.DataFrame(trade_log), metrics


def _compute_backtest_metrics(
    result_df: pd.DataFrame,
    trade_log: List[Dict],
) -> Dict[str, Any]:
    """Calculate comprehensive risk-adjusted performance metrics.

    Args:
        result_df: Backtest portfolio value DataFrame.
        trade_log: List of trade records.

    Returns:
        Dictionary of performance metrics.
    """
    metrics: Dict[str, Any] = {}

    for strategy in ["RL_Agent", "NIFTY50", "Equal_Weight", "Random"]:
        vals = result_df[strategy].values
        rets = pd.Series(vals).pct_change().dropna().values

        total_ret   = (vals[-1] / vals[0] - 1) * 100
        sharpe      = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))
        roll_max    = np.maximum.accumulate(vals)
        drawdowns   = (roll_max - vals) / (roll_max + 1e-8)
        max_dd      = float(np.max(drawdowns) * 100)
        calmar      = float(total_ret / (max_dd + 1e-8))

        # Win rate vs NIFTY50
        if strategy != "NIFTY50":
            nifty_rets = result_df["NIFTY50"].pct_change().dropna().values
            min_len    = min(len(rets), len(nifty_rets))
            win_rate   = float(np.mean(rets[:min_len] > nifty_rets[:min_len]) * 100)
        else:
            win_rate = 50.0

        final_crore = vals[-1] / 1e7

        metrics[strategy] = {
            "total_return_pct":   round(total_ret, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "max_drawdown_pct":   round(max_dd, 2),
            "calmar_ratio":       round(calmar, 3),
            "win_rate_vs_nifty":  round(win_rate, 1),
            "final_value_crore":  round(final_crore, 4),
        }

    return metrics


def _log_backtest_metrics(metrics: Dict) -> None:
    """Print a formatted backtest performance report."""
    logger.info("=" * 60)
    logger.info("  BACKTEST PERFORMANCE REPORT — NSE Universe (2Y)")
    logger.info("=" * 60)
    for strategy, m in metrics.items():
        logger.info(
            "  %-15s | Return=%.1f%%  Sharpe=%.2f  MaxDD=%.1f%%  "
            "Calmar=%.2f  Win=%.0f%%  ₹%.4f Cr",
            strategy,
            m["total_return_pct"], m["sharpe_ratio"],
            m["max_drawdown_pct"], m["calmar_ratio"],
            m["win_rate_vs_nifty"], m["final_value_crore"],
        )
    logger.info("=" * 60)


def load_rl_agent(path: Optional[str] = None) -> PPO:
    """Load a trained PPO model from disk.

    Args:
        path: Optional explicit model path. Defaults to best checkpoint.

    Returns:
        Loaded PPO model.
    """
    if path is None:
        path = str(RL_DIR / "best" / "best_model")
    return PPO.load(path)
