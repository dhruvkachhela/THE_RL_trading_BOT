"""
Live Paper Trading Interface.

Connects a trained SAC agent to a real-time data feed for paper trading.

Architecture:
  DataFeed → FeaturePipeline → Agent.select_action → OrderManager → Logger

Supports:
  - CSV file feed (replaying historical data at configurable speed)
  - Hook for a real broker API (binance / alpaca / ccxt)
  - Position tracking with P&L
  - Graceful shutdown via SIGINT

Usage:
    python scripts/live_trade.py config/training_config.yaml models/best_model.pth
    python scripts/live_trade.py config/training_config.yaml models/best_model.pth --mode replay
"""

import sys
import time
import signal
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent ))

from src.agent.sac_agent  import AdvancedSACAgent
from src.utils.features   import FeaturePipeline


# ──────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class LivePosition:
    signal:         int          # -1 long, +1 short
    entry_price:    float
    entry_time:     datetime
    tp:             float
    sl:             float
    pos_size:       float
    step:           int = 0
    entry_idx:      int = 0

    def is_long(self) -> bool:
        return self.signal == -1

    def current_return(self, price: float) -> float:
        raw = (price - self.entry_price) / (self.entry_price + 1e-8)
        return raw if self.is_long() else -raw


@dataclass
class Trade:
    signal:          int
    entry_price:     float
    exit_price:      float
    entry_time:      datetime
    exit_time:       datetime
    return_pct:      float
    pos_size:        float
    steps:           int
    reason:          str


@dataclass
class PortfolioState:
    equity:          float
    cash:            float
    peak_equity:     float
    trades:          List[Trade] = field(default_factory=list)

    def drawdown(self) -> float:
        return (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)

    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.return_pct > 0) / len(self.trades)

    def sharpe(self, window: int = 20) -> float:
        rets = [t.return_pct for t in self.trades[-window:]]
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        return float(arr.mean() / (arr.std() + 1e-8))


# ──────────────────────────────────────────────────────────────
#  Data Feed (CSV replay)
# ──────────────────────────────────────────────────────────────

class CSVDataFeed:
    """
    Replays a CSV file row by row, simulating a live data stream.
    In production replace with a WebSocket feed from your broker.
    """

    def __init__(self, data_path: str, entries_path: str, speed: float = 0.0):
        """
        Args:
            data_path:    Path to market data CSV
            entries_path: Path to entry signals CSV
            speed:        Seconds to wait between bars (0 = as fast as possible)
        """
        self.data    = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        self.entries = pd.read_csv(entries_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if "signal" not in self.entries.columns:
            self.entries["signal"] = -1
        self.speed   = speed
        self.idx     = 0
        self._entry_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        if self.idx >= len(self.data):
            raise StopIteration

        row  = self.data.iloc[self.idx]
        bar  = row.to_dict()
        bar["is_entry"] = False
        bar["entry_signal"] = 0

        # Check if this bar has an entry signal
        while (self._entry_idx < len(self.entries) and
               self.entries.iloc[self._entry_idx]["date"] <= row["date"]):
            ep = self.entries.iloc[self._entry_idx]
            if ep["date"] == row["date"]:
                bar["is_entry"]     = True
                bar["entry_signal"] = int(ep["signal"])
            self._entry_idx += 1

        self.idx += 1
        if self.speed > 0:
            time.sleep(self.speed)
        return bar

    def remaining(self) -> int:
        return len(self.data) - self.idx


# ──────────────────────────────────────────────────────────────
#  Paper Trader
# ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Runs the trained SAC agent in paper-trading mode.

    At each bar:
      1. Update rolling feature window
      2. If in position: agent decides new TP/SL, check exit conditions
      3. If signal bar and no position: open new trade
      4. Log P&L and metrics
    """

    SLIPPAGE    = 0.0005   # 5 bps
    TRANS_COST  = 0.0015   # 0.15%

    def __init__(
        self,
        agent:          AdvancedSACAgent,
        cfg:            Dict,
        initial_capital: float = 10_000.0,
    ):
        self.agent    = agent
        self.cfg      = cfg
        ec            = cfg["environment"]
        mc            = cfg["model"]

        self.state_cols     = ec["state_columns"]
        self.lookback       = ec["lookback_window"]
        self.initial_tp     = ec["initial_tp"]
        self.initial_sl     = ec["initial_sl"]
        self.tp_sl_width    = ec["tp_sl_width"]
        self.max_steps      = ec["max_trade_steps"]
        self.ema_alpha      = ec["ema_alpha_norm"]

        self.ema_means = np.zeros(len(self.state_cols))
        self.ema_vars  = np.ones(len(self.state_cols))

        self.window: deque = deque(maxlen=self.lookback)

        self.position: Optional[LivePosition] = None
        self.portfolio = PortfolioState(
            equity       = initial_capital,
            cash         = initial_capital,
            peak_equity  = initial_capital,
        )

        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self._log: List[Dict] = []

    # ── Main loop ────────────────────────────────────────────────────

    def run(self, feed: CSVDataFeed) -> None:
        print("\n" + "="*65)
        print("  SAC AGENT — PAPER TRADING")
        print("="*65)
        print(f"  Initial capital: ${self.portfolio.equity:,.2f}")
        print(f"  State cols:      {self.state_cols}")
        print(f"  Lookback:        {self.lookback} bars")
        print("="*65 + "\n")

        for bar in feed:
            if self._shutdown:
                break
            self._process_bar(bar)

        self._print_summary()
        self._save_log()

    def _process_bar(self, bar: Dict) -> None:
        # Update feature window
        feats = self._extract_features(bar)
        if feats is None:
            return
        self._update_ema(feats)
        self.window.append(self._normalise(feats))

        if len(self.window) < self.lookback:
            return   # warm-up

        state = np.array(list(self.window), dtype=np.float32)   # [T, F]
        price = float(bar["close"])
        ts    = bar["date"]

        # ── Active position management ──────────────────────────
        if self.position is not None:
            self.position.step += 1
            ret  = self.position.current_return(price)
            done = False
            reason = "continuing"

            if ret >= self.position.tp:
                done, reason = True, "tp_hit"
            elif ret <= self.position.sl:
                done, reason = True, "sl_hit"
            elif self.position.step >= self.max_steps:
                done, reason = True, "max_steps"

            if done:
                self._close_position(price, ts, reason, ret)
            else:
                # Agent updates TP/SL
                action_val, _, pos_size = self.agent.select_action(state, deterministic=True)
                self.position.tp = action_val + self.tp_sl_width
                self.position.sl = action_val - self.tp_sl_width
                self._log_step(bar, ret, reason)

        # ── New trade entry ─────────────────────────────────────
        elif bar.get("is_entry") and self.position is None:
            sig = int(bar.get("entry_signal", -1))
            self._open_position(price, ts, sig, state)

    def _open_position(self, price: float, ts, signal: int, state: np.ndarray) -> None:
        slipped = price * (1 + self.SLIPPAGE) if signal == -1 else price * (1 - self.SLIPPAGE)
        _, _, pos_size = self.agent.select_action(state, deterministic=True)
        pos_size = max(0.05, min(pos_size, 0.5))

        self.position = LivePosition(
            signal      = signal,
            entry_price = slipped,
            entry_time  = ts,
            tp          = self.initial_tp,
            sl          = self.initial_sl,
            pos_size    = pos_size,
        )
        direction = "LONG" if signal == -1 else "SHORT"
        print(f"  [{ts}] OPEN {direction}  @ {slipped:.4f}  size={pos_size:.2f}  "
              f"TP={self.initial_tp:.3f}  SL={self.initial_sl:.3f}")

    def _close_position(self, price: float, ts, reason: str, ret: float) -> None:
        p   = self.position
        adj = (ret if p.is_long() else -ret) - self.TRANS_COST
        pnl = adj * p.pos_size

        self.portfolio.equity *= (1.0 + pnl)
        self.portfolio.peak_equity = max(self.portfolio.peak_equity, self.portfolio.equity)

        trade = Trade(
            signal      = p.signal,
            entry_price = p.entry_price,
            exit_price  = price,
            entry_time  = p.entry_time,
            exit_time   = ts,
            return_pct  = adj,
            pos_size    = p.pos_size,
            steps       = p.step,
            reason      = reason,
        )
        self.portfolio.trades.append(trade)

        direction = "LONG" if p.is_long() else "SHORT"
        sign = "+" if pnl >= 0 else ""
        print(f"  [{ts}] CLOSE {direction}  @ {price:.4f}  "
              f"ret={sign}{adj:.4f}  pnl={sign}{pnl:.4f}  "
              f"equity=${self.portfolio.equity:,.2f}  reason={reason}")

        self._log.append({
            "time":       str(ts),
            "side":       direction,
            "entry":      p.entry_price,
            "exit":       price,
            "return_pct": adj,
            "pos_size":   p.pos_size,
            "pnl":        pnl,
            "equity":     self.portfolio.equity,
            "drawdown":   self.portfolio.drawdown(),
            "reason":     reason,
            "steps":      p.step,
        })

        self.position = None

    def _extract_features(self, bar: Dict) -> Optional[np.ndarray]:
        try:
            return np.array([float(bar.get(c, 0.0)) for c in self.state_cols], dtype=np.float32)
        except Exception:
            return None

    def _update_ema(self, feats: np.ndarray) -> None:
        a = self.ema_alpha
        self.ema_means = (1 - a) * self.ema_means + a * feats
        diff = feats - self.ema_means
        self.ema_vars  = (1 - a) * self.ema_vars  + a * diff ** 2

    def _normalise(self, feats: np.ndarray) -> np.ndarray:
        return np.clip((feats - self.ema_means) / (np.sqrt(self.ema_vars) + 1e-8), -5.0, 5.0)

    def _log_step(self, bar: Dict, ret: float, reason: str) -> None:
        pass  # Extend for verbose per-step logging

    def _handle_shutdown(self, *_) -> None:
        print("\n[PaperTrader] Shutdown requested…")
        self._shutdown = True

    def _print_summary(self) -> None:
        p = self.portfolio
        print("\n" + "="*65)
        print("  PAPER TRADING SESSION SUMMARY")
        print("="*65)
        print(f"  Final equity   : ${p.equity:,.2f}")
        print(f"  Total return   : {(p.equity / (p.equity / (1 + sum(t.return_pct for t in p.trades))) - 1) if p.trades else 0:.2%}")
        print(f"  Total trades   : {len(p.trades)}")
        print(f"  Win rate       : {p.win_rate():.2%}")
        print(f"  Sharpe (last)  : {p.sharpe():.4f}")
        print(f"  Max drawdown   : {p.drawdown():.2%}")
        print("="*65 + "\n")

    def _save_log(self) -> None:
        if not self._log:
            return
        out_dir = Path("results/")
        out_dir.mkdir(exist_ok=True)
        ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path    = out_dir / f"paper_trade_{ts_str}.csv"
        pd.DataFrame(self._log).to_csv(path, index=False)
        print(f"[PaperTrader] Log saved → {path}")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    config_path  = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.pth"
    capital      = float(sys.argv[3]) if len(sys.argv) > 3 else 10_000.0
    speed        = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # 0 = max speed replay

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ec  = cfg["environment"]
    mc  = cfg["model"]
    ac  = cfg["agent"]
    rc  = cfg.get("risk", {})

    action_values = np.array(mc["action_values"], dtype=np.float32)
    state_dim     = len(ec["state_columns"])

    agent = AdvancedSACAgent(
        state_dim             = state_dim,
        action_dim            = len(action_values),
        sequence_length       = mc["sequence_length"],
        hidden_dim            = mc["hidden_dim"],
        lr                    = ac["learning_rate"],
        gamma                 = ac["gamma"],
        tau                   = ac["tau"],
        action_values         = action_values,
        target_entropy_factor = ac["target_entropy_factor"],
        initial_log_alpha     = ac["initial_log_alpha"],
        n_step                = ac.get("n_step_returns", 3),
        dropout               = mc.get("dropout", 0.1),
        n_heads               = mc.get("n_heads", 4),
        n_transformer_layers  = mc.get("n_transformer_layers", 2),
        use_position_sizing   = rc.get("position_sizing", True),
    )
    agent.load(weights_path)

    feed = CSVDataFeed(
        data_path    = cfg["data"]["training_data"],
        entries_path = cfg["data"]["entry_points"],
        speed        = speed,
    )

    trader = PaperTrader(agent, cfg, initial_capital=capital)
    trader.run(feed)


if __name__ == "__main__":
    main()
