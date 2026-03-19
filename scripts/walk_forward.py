"""
Walk-Forward Validation Framework.

Industry-standard method to evaluate trading strategies without lookahead bias.

Method:
  1. Split the full timeline into N windows
  2. Each window has a TRAIN period followed by a TEST period
  3. Train fresh agent on train period, evaluate on test period
  4. Aggregate OOS (out-of-sample) metrics across all windows
  5. Report combined equity curve + performance statistics

This is the correct way to evaluate ML trading strategies —
a single train/test split is almost always overfit in finance.

Usage:
    python scripts/walk_forward.py config/training_config.yaml
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent ))

from src.agent.sac_agent         import AdvancedSACAgent
from src.environment.trading_env import AdvancedTradingEnvironment
from src.memory.replay_buffer    import PrioritizedReplayBuffer, NStepBuffer


# ──────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_idx:    int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    trades:        List[Dict] = field(default_factory=list)
    sharpe:        float = 0.0
    sortino:       float = 0.0
    total_return:  float = 0.0
    win_rate:      float = 0.0
    max_drawdown:  float = 0.0
    n_trades:      int   = 0


# ──────────────────────────────────────────────────────────────
#  Walk-forward engine
# ──────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Anchored or rolling walk-forward validation.

    Args:
        config_path:    Path to training_config.yaml
        n_windows:      Number of walk-forward windows
        train_ratio:    Fraction of each window used for training
        anchored:       If True, train window always starts from t=0 (anchored WF)
                        If False, train window slides forward (rolling WF)
        train_episodes: Episodes per window (can be less than full training)
    """

    def __init__(
        self,
        config_path:    str,
        n_windows:      int   = 5,
        train_ratio:    float = 0.7,
        anchored:       bool  = False,
        train_episodes: int   = 100,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.n_windows      = n_windows
        self.train_ratio    = train_ratio
        self.anchored       = anchored
        self.train_episodes = train_episodes
        self.config_path    = config_path

        # Load full datasets
        self.full_data = pd.read_csv(
            self.cfg["data"]["training_data"], parse_dates=["date"]
        ).sort_values("date").reset_index(drop=True)

        self.full_entries = pd.read_csv(
            self.cfg["data"]["entry_points"], parse_dates=["date"]
        ).sort_values("date").reset_index(drop=True)

        if "signal" not in self.full_entries.columns:
            self.full_entries["signal"] = -1

        self.results: List[WindowResult] = []

    def run(self) -> Dict:
        """Run all walk-forward windows and return aggregated metrics."""
        print("\n" + "="*65)
        print(f"  Walk-Forward Validation  |  {self.n_windows} windows  |  "
              f"{'Anchored' if self.anchored else 'Rolling'}")
        print("="*65)

        windows = self._build_windows()

        for w_idx, (train_slice, test_slice) in enumerate(windows):
            print(f"\n── Window {w_idx+1}/{self.n_windows} ──────────────────────────────")
            result = self._run_window(w_idx, train_slice, test_slice)
            self.results.append(result)
            print(f"   OOS Sharpe={result.sharpe:+.3f}  "
                  f"Return={result.total_return:+.4f}  "
                  f"Trades={result.n_trades}  "
                  f"WR={result.win_rate:.2%}  "
                  f"MaxDD={result.max_drawdown:.2%}")

        return self._aggregate()

    def _build_windows(self) -> List[Tuple[Tuple, Tuple]]:
        """
        Splits entry-point dates into n_windows × (train, test) date slices.
        Returns list of ((data_train_mask, ep_train_mask), (data_test_mask, ep_test_mask))
        """
        dates = self.full_entries["date"].values
        n     = len(dates)

        window_size = n // self.n_windows
        windows = []

        for i in range(self.n_windows):
            if self.anchored:
                train_end_i = int((i + 1) * window_size * self.train_ratio +
                                  i * window_size * (1 - self.train_ratio))
                test_start_i = train_end_i
            else:
                start_i      = i * window_size
                train_end_i  = start_i + int(window_size * self.train_ratio)
                test_start_i = train_end_i

            test_end_i = min((i + 1) * window_size, n)

            train_ep_dates = dates[:train_end_i] if self.anchored else dates[i * window_size:train_end_i]
            test_ep_dates  = dates[test_start_i:test_end_i]

            if len(train_ep_dates) == 0 or len(test_ep_dates) == 0:
                continue

            windows.append((
                (dates[0], train_ep_dates[-1]),   # train period
                (test_ep_dates[0], test_ep_dates[-1]),  # test period
            ))

        return windows

    def _run_window(
        self,
        w_idx: int,
        train_period: Tuple,
        test_period: Tuple,
    ) -> WindowResult:
        train_start, train_end   = str(train_period[0])[:10], str(train_period[1])[:10]
        test_start,  test_end    = str(test_period[0])[:10],  str(test_period[1])[:10]

        print(f"   Train: {train_start} → {train_end}")
        print(f"   Test:  {test_start}  → {test_end}")

        # ── Filter data to window ───────────────────────────────
        train_data = self._filter_data(self.full_data, train_start, train_end)
        train_eps  = self._filter_entries(self.full_entries, train_start, train_end)
        test_data  = self._filter_data(self.full_data, test_start, test_end)
        test_eps   = self._filter_entries(self.full_entries, test_start, test_end)

        if len(train_eps) < 5 or len(test_eps) < 2:
            print("   [SKIP] Insufficient data for this window.")
            return WindowResult(w_idx, train_start, train_end, test_start, test_end)

        # ── Save temp CSVs ──────────────────────────────────────
        tmp = Path("/tmp/wf_tmp")
        tmp.mkdir(exist_ok=True)
        train_data.to_csv(tmp / "train_data.csv", index=False)
        train_eps.to_csv(tmp  / "train_entries.csv", index=False)
        test_data.to_csv(tmp  / "test_data.csv", index=False)
        test_eps.to_csv(tmp   / "test_entries.csv", index=False)

        # ── Train agent on train window ─────────────────────────
        agent = self._build_agent()
        self._train_agent(agent, tmp / "train_data.csv", tmp / "train_entries.csv")

        # ── Evaluate on test window ─────────────────────────────
        trades = self._evaluate_agent(agent, tmp / "test_data.csv", tmp / "test_entries.csv")

        # ── Compute metrics ─────────────────────────────────────
        rets = np.array([t.get("actual_return", 0.0) for t in trades])
        sharpe  = float(rets.mean() / (rets.std() + 1e-8)) if len(rets) > 1 else 0.0
        neg     = rets[rets < 0]
        sortino = float(rets.mean() / (neg.std() + 1e-8)) if len(neg) > 1 else 0.0
        total_r = float(rets.sum())
        win_r   = float((rets > 0).mean()) if len(rets) else 0.0

        equity  = np.cumprod(1 + rets) if len(rets) else np.array([1.0])
        peaks   = np.maximum.accumulate(equity)
        max_dd  = float(((peaks - equity) / (peaks + 1e-8)).max()) if len(rets) else 0.0

        return WindowResult(
            window_idx   = w_idx,
            train_start  = train_start,
            train_end    = train_end,
            test_start   = test_start,
            test_end     = test_end,
            trades       = trades,
            sharpe       = sharpe,
            sortino      = sortino,
            total_return = total_r,
            win_rate     = win_r,
            max_drawdown = max_dd,
            n_trades     = len(trades),
        )

    def _aggregate(self) -> Dict:
        """Combine all OOS windows into a single performance report."""
        all_trades = []
        for r in self.results:
            all_trades.extend(r.trades)

        sharpes  = [r.sharpe for r in self.results if r.n_trades > 0]
        returns  = [r.total_return for r in self.results if r.n_trades > 0]
        win_rates = [r.win_rate for r in self.results if r.n_trades > 0]
        mdd      = [r.max_drawdown for r in self.results if r.n_trades > 0]

        combined_rets = np.array([t.get("actual_return", 0.0) for t in all_trades])
        neg = combined_rets[combined_rets < 0]

        report = {
            "n_windows":         len(self.results),
            "total_oos_trades":  len(all_trades),
            "avg_sharpe":        float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe":        float(np.std(sharpes))  if sharpes else 0.0,
            "avg_return":        float(np.mean(returns)) if returns else 0.0,
            "avg_win_rate":      float(np.mean(win_rates)) if win_rates else 0.0,
            "avg_max_drawdown":  float(np.mean(mdd)) if mdd else 0.0,
            "combined_sharpe":   float(combined_rets.mean() / (combined_rets.std() + 1e-8))
                                 if len(combined_rets) > 1 else 0.0,
            "combined_sortino":  float(combined_rets.mean() / (neg.std() + 1e-8))
                                 if len(neg) > 1 else 0.0,
            "window_results":    self.results,
            "all_trades":        all_trades,
        }

        self._print_report(report)
        self._save_report(report)
        return report

    def _print_report(self, r: Dict) -> None:
        sep = "="*65
        print(f"\n{sep}")
        print("  WALK-FORWARD VALIDATION — AGGREGATED OOS RESULTS")
        print(sep)
        print(f"  Windows evaluated     : {r['n_windows']}")
        print(f"  Total OOS trades      : {r['total_oos_trades']}")
        print(f"  Combined Sharpe       : {r['combined_sharpe']:+.4f}")
        print(f"  Combined Sortino      : {r['combined_sortino']:+.4f}")
        print(f"  Avg window Sharpe     : {r['avg_sharpe']:+.4f} ± {r['std_sharpe']:.4f}")
        print(f"  Avg window return     : {r['avg_return']:+.4f}")
        print(f"  Avg win rate          : {r['avg_win_rate']:.2%}")
        print(f"  Avg max drawdown      : {r['avg_max_drawdown']:.2%}")
        print(f"\n  Per-window breakdown:")
        for wr in r["window_results"]:
            print(f"    W{wr.window_idx+1}  {wr.test_start}→{wr.test_end}  "
                  f"Sharpe {wr.sharpe:+.3f}  Ret {wr.total_return:+.4f}  "
                  f"N={wr.n_trades}  WR={wr.win_rate:.2%}")
        print(sep + "\n")

    def _save_report(self, r: Dict) -> None:
        out_dir = Path(self.cfg["data"].get("results_path", "results/"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        rows = []
        for wr in r["window_results"]:
            rows.append({
                "window": wr.window_idx + 1,
                "test_start": wr.test_start,
                "test_end": wr.test_end,
                "sharpe": wr.sharpe,
                "sortino": wr.sortino,
                "total_return": wr.total_return,
                "win_rate": wr.win_rate,
                "max_drawdown": wr.max_drawdown,
                "n_trades": wr.n_trades,
            })
        pd.DataFrame(rows).to_csv(out_dir / "wf_window_summary.csv", index=False)

        # All OOS trades
        if r["all_trades"]:
            pd.DataFrame(r["all_trades"]).to_csv(out_dir / "wf_all_trades.csv", index=False)

        print(f"[WF] Results saved → {out_dir}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _filter_data(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)

    def _filter_entries(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)

    def _build_agent(self) -> AdvancedSACAgent:
        mc  = self.cfg["model"]
        ac  = self.cfg["agent"]
        ec  = self.cfg["environment"]
        rc  = self.cfg.get("risk", {})
        action_values = np.array(mc["action_values"], dtype=np.float32)
        state_dim = len(ec["state_columns"])
        return AdvancedSACAgent(
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
            grad_clip             = ac.get("grad_clip", 1.0),
            dropout               = mc.get("dropout", 0.1),
            n_heads               = mc.get("n_heads", 4),
            n_transformer_layers  = mc.get("n_transformer_layers", 2),
            use_position_sizing   = rc.get("position_sizing", True),
        )

    def _build_env(self, data_path: str, entries_path: str) -> AdvancedTradingEnvironment:
        ec  = self.cfg["environment"]
        rc  = self.cfg.get("risk", {})
        rwc = self.cfg.get("reward", {})
        return AdvancedTradingEnvironment(
            data_file         = str(data_path),
            entry_points_file = str(entries_path),
            state_cols        = ec["state_columns"],
            max_steps         = ec["max_trade_steps"],
            lookback_window   = ec["lookback_window"],
            ema_alpha         = ec["ema_alpha_norm"],
            initial_tp        = ec["initial_tp"],
            initial_sl        = ec["initial_sl"],
            tp_sl_width       = ec["tp_sl_width"],
            slippage_bps      = ec.get("slippage_bps", 5.0),
            transaction_cost  = ec.get("transaction_cost", 0.0015),
            reward_type       = rwc.get("type", "sharpe"),
            risk_config       = rc,
        )

    def _train_agent(
        self,
        agent: AdvancedSACAgent,
        data_path: str,
        entries_path: str,
    ) -> None:
        rbc = self.cfg["replay_buffer"]
        ac  = self.cfg["agent"]
        tc  = self.cfg["training"]

        env     = self._build_env(data_path, entries_path)
        per_buf = PrioritizedReplayBuffer(rbc["capacity"] // 4, rbc["per_alpha"])
        n_buf   = NStepBuffer(ac.get("n_step_returns", 3), ac["gamma"], per_buf)

        batch_size = tc["batch_size"]
        warmup     = min(500, tc.get("warmup_steps", 1000) // 2)
        episodes   = self.train_episodes

        # Warmup
        state = env.reset()
        for _ in range(warmup):
            ai  = np.random.randint(len(agent.action_values))
            av  = float(agent.action_values[ai])
            ns, r, done, _ = env.step(av, 0.3)
            n_buf.push(state, ai, r, ns, done)
            state = ns if not done else env.reset()

        # Training episodes
        for ep in range(episodes):
            state = env.reset()
            done  = False
            while not done:
                av, ai, ps = agent.select_action(state, deterministic=False)
                ns, r, done, _ = env.step(av, ps)
                n_buf.push(state, ai, r, ns, done)
                if per_buf.is_ready(batch_size):
                    agent.update(per_buf, batch_size, beta=0.6)
                state = ns

        print(f"   Training complete ({episodes} eps)")

    def _evaluate_agent(
        self,
        agent: AdvancedSACAgent,
        data_path: str,
        entries_path: str,
    ) -> List[Dict]:
        env   = self._build_env(data_path, entries_path)
        state = env.reset()
        done  = False
        info  = {}
        while not done:
            av, _, ps = agent.select_action(state, deterministic=True)
            state, _, done, info = env.step(av, ps)
        return info.get("episode_trades", [])


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg_path  = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    n_windows = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    validator = WalkForwardValidator(
        config_path    = cfg_path,
        n_windows      = n_windows,
        train_ratio    = 0.7,
        anchored       = False,
        train_episodes = 100,
    )
    validator.run()
