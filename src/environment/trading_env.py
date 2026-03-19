"""
Advanced Trading Environment for RL.

Upgrades over baseline:
  - Long AND short positions (dual-signal)
  - Realistic slippage + transaction costs
  - Dynamic position sizing (from actor)
  - Risk manager: max drawdown halt, Kelly sizing cap
  - Multiple reward modes: shaped | sharpe | sortino
  - Rolling Sharpe reward for risk-adjusted optimisation
  - Richer info dict for logging
  - EMA-based online feature normalisation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


# ──────────────────────────────────────────────────────────────
#  Risk Manager
# ──────────────────────────────────────────────────────────────

class RiskManager:
    """
    Monitors portfolio health and enforces risk limits.

    Features:
      - Max drawdown halt (episode ends if exceeded)
      - Kelly criterion position sizing
      - Rolling win-rate tracking
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.5,
        risk_free_rate: float = 0.04,
    ):
        self.max_dd       = max_drawdown_limit
        self.kelly_frac   = kelly_fraction
        self.max_pos      = max_position_pct
        self.rf           = risk_free_rate

        self.peak_equity  = 1.0
        self.equity       = 1.0
        self.trade_rets: List[float] = []

    def reset(self) -> None:
        self.peak_equity = 1.0
        self.equity      = 1.0
        self.trade_rets  = []

    def update(self, trade_return: float, position_size: float) -> float:
        """Apply trade return (already sized) and return actual equity change."""
        actual = trade_return * position_size
        self.equity *= (1.0 + actual)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.trade_rets.append(actual)
        return actual

    def drawdown(self) -> float:
        return (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)

    def is_halted(self) -> bool:
        return self.drawdown() > self.max_dd

    def kelly_size(self, win_prob: float, avg_win: float, avg_loss: float) -> float:
        """Fractional Kelly position size."""
        if avg_loss <= 0 or avg_win <= 0:
            return self.kelly_frac * 0.5
        b = avg_win / avg_loss
        k = (b * win_prob - (1 - win_prob)) / b
        k = max(0.0, min(k, 1.0))
        return min(k * self.kelly_frac, self.max_pos)

    def rolling_sharpe(self, window: int = 20) -> float:
        rets = self.trade_rets[-window:]
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        excess = arr.mean() - (self.rf / 252)
        std = arr.std() + 1e-8
        return float(excess / std)

    def rolling_sortino(self, window: int = 20) -> float:
        rets = self.trade_rets[-window:]
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        excess = arr.mean() - (self.rf / 252)
        neg = arr[arr < 0]
        downside = neg.std() + 1e-8 if len(neg) > 1 else 1e-8
        return float(excess / downside)


# ──────────────────────────────────────────────────────────────
#  Advanced Trading Environment
# ──────────────────────────────────────────────────────────────

class AdvancedTradingEnvironment:
    """
    Simulates realistic trade management with RL control.

    State:  (lookback_window, n_features) — EMA-normalised OHLCV + indicators
    Action: discrete TP/SL displacement  +  continuous position size from actor
    Reward: configurable — shaped | sharpe | sortino
    """

    def __init__(
        self,
        data_file: str,
        entry_points_file: str,
        state_cols: List[str],
        max_steps: int = 10,
        lookback_window: int = 20,
        ema_alpha: float = 0.05,
        initial_tp: float = 0.03,
        initial_sl: float = -0.03,
        tp_sl_width: float = 0.03,
        slippage_bps: float = 5.0,
        transaction_cost: float = 0.0015,
        reward_type: str = "sharpe",
        reward_window: int = 20,
        risk_config: Optional[Dict] = None,
    ):
        # ── Load data ───────────────────────────────────────────
        self.data          = pd.read_csv(data_file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        self.entry_points  = pd.read_csv(entry_points_file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

        if "signal" not in self.entry_points.columns:
            # Default: all long
            self.entry_points["signal"] = -1

        self.state_cols     = state_cols
        self.max_steps      = max_steps
        self.lookback_window = lookback_window
        self.initial_tp     = initial_tp
        self.initial_sl     = initial_sl
        self.tp_sl_width    = tp_sl_width
        self.slippage_bps   = slippage_bps / 10_000.0
        self.transaction_cost = transaction_cost
        self.reward_type    = reward_type
        self.reward_window  = reward_window

        # ── Online normalisation ────────────────────────────────
        self.ema_alpha   = ema_alpha
        self.ema_means   = np.zeros(len(state_cols))
        self.ema_vars    = np.ones(len(state_cols))
        self._norm_ready = False

        # ── Risk manager ────────────────────────────────────────
        rc = risk_config or {}
        self.risk = RiskManager(
            max_drawdown_limit = rc.get("max_drawdown_limit", 0.15),
            kelly_fraction     = rc.get("kelly_fraction", 0.25),
            max_position_pct   = rc.get("max_position_pct", 0.5),
            risk_free_rate     = rc.get("risk_free_rate", 0.04),
        )

        self.total_entry_points = len(self.entry_points)
        self._state_dim = len(state_cols)
        print(f"[Env] {len(self.data)} market rows | {self.total_entry_points} entry points | reward={reward_type}")

    # ── Gym-like interface ───────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def reset(self) -> np.ndarray:
        self.current_entry_idx = 0
        self.last_exit_date    = pd.Timestamp.min
        self.episode_trades: List[Dict] = []
        self.episode_reward    = 0.0
        self.trades_completed  = 0
        self.position_open     = False

        self.risk.reset()

        self._start_new_trade()
        return self._get_state()

    def step(
        self,
        action_value: float,
        position_size: float = 0.5,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Args:
            action_value:  TP/SL center displacement
            position_size: fraction of capital [0, 1] from actor

        Returns:
            next_state, reward, done, info
        """
        if not self.position_open:
            return self._zero_state(), 0.0, True, self._episode_info("no_position")

        # Risk check — halt if max drawdown breached
        if self.risk.is_halted():
            return self._zero_state(), -1.0, True, self._episode_info("risk_halt")

        self.current_step += 1
        next_idx = self.entry_data_idx + self.current_step

        # Run out of data
        if next_idx >= len(self.data):
            ret = self._calc_ret(self.data.iloc[self.current_data_idx]["close"])
            reward = self._calc_reward(ret, False, position_size)
            self._close_trade(ret, "no_more_data", position_size)
            return self._next_or_end(reward)

        self.current_data_idx = next_idx
        price = self.data.iloc[self.current_data_idx]["close"]
        ret   = self._calc_ret(price)

        # ── TP / SL / max-steps check ────────────────────────────
        if ret >= self.tp:
            reward = self._calc_reward(ret, True, position_size)
            self._close_trade(ret, "tp_hit", position_size)
            return self._next_or_end(reward)

        if ret <= self.sl:
            reward = self._calc_reward(ret, True, position_size)
            self._close_trade(ret, "sl_hit", position_size)
            return self._next_or_end(reward)

        if self.current_step >= self.max_steps:
            reward = self._calc_reward(ret, False, position_size)
            self._close_trade(ret, "max_steps", position_size)
            return self._next_or_end(reward)

        # ── Continue trade — update TP/SL ───────────────────────
        self.tp = action_value + self.tp_sl_width
        self.sl = action_value - self.tp_sl_width
        reward  = self._calc_reward(ret, False, position_size)

        return self._get_state(), reward, False, {
            "ret": ret, "tp": self.tp, "sl": self.sl,
            "step": self.current_step, "position": self._pos_label(),
            "pos_size": position_size,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _start_new_trade(self) -> None:
        valid = self.entry_points[self.entry_points["date"] > self.last_exit_date]
        if valid.empty:
            self.position_open = False
            return

        ep = valid.iloc[0]
        self.entry_date  = ep["date"]
        self.signal      = int(ep["signal"])   # -1 = long, +1 = short

        match = self.data[self.data["date"] >= self.entry_date]
        if match.empty:
            self.position_open = False
            return

        idx = match.index[0]
        self.entry_data_idx    = idx
        self.current_data_idx  = idx
        raw_price              = self.data.iloc[idx]["close"]

        # Apply slippage at entry
        if self.signal == -1:   # long — buy at slightly higher price
            self.entry_price = raw_price * (1.0 + self.slippage_bps)
        else:                   # short — sell at slightly lower price
            self.entry_price = raw_price * (1.0 - self.slippage_bps)

        self.tp           = self.initial_tp
        self.sl           = self.initial_sl
        self.current_step = 0
        self.position_open = True

    def _calc_ret(self, price: float) -> float:
        """Raw % return from entry (direction-adjusted for short)."""
        raw = (price - self.entry_price) / (self.entry_price + 1e-8)
        return raw if self.signal == -1 else -raw

    def _calc_reward(self, ret: float, tp_sl_hit: bool, pos_size: float) -> float:
        """Compute reward based on configured type."""
        if self.reward_type == "shaped":
            base = float(np.exp(ret / 10) - 1.0 + ret)
            bonus = ret if tp_sl_hit else 0.0
            return (base + bonus) * pos_size

        if self.reward_type == "sharpe":
            return self.risk.rolling_sharpe(self.reward_window)

        if self.reward_type == "sortino":
            return self.risk.rolling_sortino(self.reward_window)

        # fallback
        return ret * pos_size

    def _close_trade(self, raw_ret: float, reason: str, pos_size: float) -> None:
        # Direction-adjusted return after costs
        adj = raw_ret if self.signal == -1 else -raw_ret
        adj -= self.transaction_cost

        # Risk manager update (applies position sizing)
        actual = self.risk.update(adj, pos_size)

        slip_cost = abs(self.entry_price - self.data.iloc[self.entry_data_idx]["close"]) / \
                    (self.data.iloc[self.entry_data_idx]["close"] + 1e-8)

        self.episode_trades.append({
            "entry_date":        self.entry_date,
            "entry_price":       self.entry_price,
            "signal":            self.signal,
            "position":          self._pos_label(),
            "raw_return":        raw_ret,
            "adj_return":        adj,
            "actual_return":     actual,
            "pos_size":          pos_size,
            "steps":             self.current_step,
            "reason":            reason,
            "transaction_cost":  self.transaction_cost,
            "slippage_cost":     slip_cost,
            "equity":            self.risk.equity,
            "drawdown":          self.risk.drawdown(),
        })
        self.trades_completed += 1
        self.last_exit_date = self.data.iloc[self.current_data_idx]["date"]
        self._start_new_trade()

    def _next_or_end(self, reward: float) -> Tuple[np.ndarray, float, bool, Dict]:
        done = not self.position_open
        if done:
            return self._zero_state(), reward, True, self._episode_info("done")
        return self._get_state(), reward, False, {
            "continuing": True, "trades_done": self.trades_completed
        }

    def _get_state(self) -> np.ndarray:
        if not self.position_open:
            return self._zero_state()

        end   = self.current_data_idx
        start = max(0, end - self.lookback_window + 1)
        raw   = self.data.iloc[start:end + 1][self.state_cols].values.astype(np.float32)

        # Pad if window not full yet
        if len(raw) < self.lookback_window:
            pad = np.zeros((self.lookback_window - len(raw), self._state_dim), dtype=np.float32)
            raw = np.vstack([pad, raw])

        # Vectorised EMA normalisation (update on each new step)
        norm = np.zeros_like(raw)
        for t in range(raw.shape[0]):
            mask = ~np.isnan(raw[t])
            v = np.where(mask, raw[t], self.ema_means)
            self.ema_means = (1 - self.ema_alpha) * self.ema_means + self.ema_alpha * v
            diff = v - self.ema_means
            self.ema_vars  = (1 - self.ema_alpha) * self.ema_vars  + self.ema_alpha * diff ** 2
            norm[t] = (v - self.ema_means) / (np.sqrt(self.ema_vars) + 1e-8)

        return np.clip(norm, -5.0, 5.0)   # clip extreme values

    def _zero_state(self) -> np.ndarray:
        return np.zeros((self.lookback_window, self._state_dim), dtype=np.float32)

    def _pos_label(self) -> str:
        return "LONG" if self.signal == -1 else "SHORT"

    def _episode_info(self, reason: str) -> Dict:
        trades = self.episode_trades
        total  = sum(t["actual_return"] for t in trades)
        avg    = total / len(trades) if trades else 0.0
        wins   = [t for t in trades if t["actual_return"] > 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        return {
            "reason":           reason,
            "trades_completed": self.trades_completed,
            "total_return":     total,
            "avg_return":       avg,
            "win_rate":         win_rate,
            "sharpe":           self.risk.rolling_sharpe(),
            "sortino":          self.risk.rolling_sortino(),
            "max_drawdown":     self.risk.drawdown(),
            "final_equity":     self.risk.equity,
            "episode_trades":   trades,
        }
