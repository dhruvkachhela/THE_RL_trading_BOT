"""
Advanced Backtesting Script.

Loads trained weights, runs deterministic evaluation and prints:
  - Portfolio equity curve
  - Sharpe / Sortino / Calmar ratios
  - Max drawdown + recovery time
  - Win rate, profit factor
  - Long vs Short breakdown
  - Per-trade table (optional)
  - Saves results CSV + equity-curve PNG
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent ))

from src.agent.sac_agent         import AdvancedSACAgent
from src.environment.trading_env import AdvancedTradingEnvironment


# ──────────────────────────────────────────────────────────────
#  Performance Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(trades: list, initial_capital: float = 10_000.0, rf: float = 0.04) -> dict:
    if not trades:
        return {}

    equity   = initial_capital
    equity_h = [initial_capital]
    rets     = []

    for t in trades:
        r = t["actual_return"]
        rets.append(r)
        equity *= (1.0 + r)
        equity_h.append(equity)

    equity_h = np.array(equity_h)
    rets_arr = np.array(rets)

    # ── Drawdown ────────────────────────────────────────────────
    peaks    = np.maximum.accumulate(equity_h)
    dd_curve = (peaks - equity_h) / (peaks + 1e-8)
    max_dd   = float(dd_curve.max())

    # ── Recovery time (bars from max DD trough to new peak) ────
    trough_idx = int(np.argmax(dd_curve))
    recovery   = 0
    if trough_idx < len(equity_h) - 1:
        future = equity_h[trough_idx:]
        new_peak = np.argmax(future >= peaks[trough_idx])
        recovery = int(new_peak)

    # ── Returns stats ───────────────────────────────────────────
    mean_r   = float(rets_arr.mean())
    std_r    = float(rets_arr.std() + 1e-8)
    wins     = rets_arr[rets_arr > 0]
    losses   = rets_arr[rets_arr < 0]

    sharpe   = mean_r / std_r
    sortino  = mean_r / (losses.std() + 1e-8) if len(losses) > 1 else 0.0

    # Calmar ratio
    calmar   = mean_r / (max_dd + 1e-8)

    # Profit factor
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss   = float(abs(losses.sum())) if len(losses) else 1e-8
    pf           = gross_profit / gross_loss

    total_return = (equity - initial_capital) / initial_capital

    # ── Long / Short breakdown ──────────────────────────────────
    long_t  = [t for t in trades if t.get("signal") == -1]
    short_t = [t for t in trades if t.get("signal") == 1]

    def avg_ret(ts): return float(np.mean([t["actual_return"] for t in ts])) if ts else 0.0
    def wr(ts):      return float(sum(1 for t in ts if t["actual_return"] > 0) / max(len(ts), 1))

    # ── Reasons breakdown ───────────────────────────────────────
    from collections import Counter
    reason_counts = Counter(t["reason"] for t in trades)

    return {
        "initial_capital": initial_capital,
        "final_equity":    equity,
        "total_return":    total_return,
        "total_trades":    len(trades),
        "win_rate":        wr(trades),
        "profit_factor":   pf,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "calmar":          calmar,
        "max_drawdown":    max_dd,
        "recovery_trades": recovery,
        "mean_return":     mean_r,
        "std_return":      std_r,
        "gross_profit":    gross_profit,
        "gross_loss":      gross_loss,
        "long_trades":     len(long_t),
        "long_avg_return": avg_ret(long_t),
        "long_win_rate":   wr(long_t),
        "short_trades":    len(short_t),
        "short_avg_return":avg_ret(short_t),
        "short_win_rate":  wr(short_t),
        "reason_counts":   dict(reason_counts),
        "equity_curve":    equity_h.tolist(),
    }


def print_report(metrics: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  ADVANCED RL TRADING AGENT — BACKTEST REPORT")
    print(sep)
    print(f"  Initial capital   : ${metrics['initial_capital']:>12,.2f}")
    print(f"  Final equity      : ${metrics['final_equity']:>12,.2f}")
    print(f"  Total return      : {metrics['total_return']:>12.2%}")
    print()
    print(f"  Total trades      : {metrics['total_trades']:>12}")
    print(f"  Win rate          : {metrics['win_rate']:>12.2%}")
    print(f"  Profit factor     : {metrics['profit_factor']:>12.3f}")
    print()
    print(f"  Sharpe ratio      : {metrics['sharpe']:>12.4f}")
    print(f"  Sortino ratio     : {metrics['sortino']:>12.4f}")
    print(f"  Calmar ratio      : {metrics['calmar']:>12.4f}")
    print(f"  Max drawdown      : {metrics['max_drawdown']:>12.2%}")
    print(f"  Recovery (trades) : {metrics['recovery_trades']:>12}")
    print()
    print("  POSITION BREAKDOWN")
    print(f"  Long  trades      : {metrics['long_trades']:>5}  "
          f"Avg ret {metrics['long_avg_return']:+.4f}  WR {metrics['long_win_rate']:.2%}")
    print(f"  Short trades      : {metrics['short_trades']:>5}  "
          f"Avg ret {metrics['short_avg_return']:+.4f}  WR {metrics['short_win_rate']:.2%}")
    print()
    print("  EXIT REASONS")
    for reason, count in metrics["reason_counts"].items():
        pct = count / metrics["total_trades"]
        print(f"    {reason:<20} {count:>5}  ({pct:.1%})")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def backtest(config_path: str, weights_path: str, initial_capital: float = 10_000.0) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ec  = cfg["environment"]
    mc  = cfg["model"]
    ac  = cfg["agent"]
    rc  = cfg.get("risk", {})
    rwc = cfg.get("reward", {})

    action_values = np.array(mc["action_values"], dtype=np.float32)

    env = AdvancedTradingEnvironment(
        data_file         = cfg["data"]["training_data"],
        entry_points_file = cfg["data"]["entry_points"],
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

    agent = AdvancedSACAgent(
        state_dim             = env.state_dim,
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

    # ── Run deterministic episode ────────────────────────────────
    print(f"\n[Backtest] Running with weights: {weights_path}")
    state = env.reset()
    done  = False

    while not done:
        action_val, _, pos_size = agent.select_action(state, deterministic=True)
        state, _, done, info    = env.step(action_val, pos_size)

    trades = info.get("episode_trades", [])
    metrics = compute_metrics(trades, initial_capital)
    print_report(metrics)

    # ── Save results ─────────────────────────────────────────────
    out_dir = Path(cfg["data"].get("results_path", "results/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(trades)
    csv_path = out_dir / "backtest_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Backtest] Results saved -> {csv_path}")

    # ── Equity curve (matplotlib) ────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        eq_curve = np.array(metrics["equity_curve"])
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        axes[0].plot(eq_curve, color="#1f77b4", linewidth=1.5, label="Equity")
        axes[0].axhline(initial_capital, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_title("Portfolio Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        peaks = np.maximum.accumulate(eq_curve)
        dd    = (peaks - eq_curve) / (peaks + 1e-8) * 100
        axes[1].fill_between(range(len(dd)), -dd, 0, color="red", alpha=0.4, label="Drawdown %")
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("Drawdown %")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "equity_curve.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[Backtest] Equity curve saved -> {fig_path}")
        plt.close()
    except ImportError:
        print("[Backtest] matplotlib not installed — skipping plot")


if __name__ == "__main__":
    cfg_path     = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.pth"
    capital      = float(sys.argv[3]) if len(sys.argv) > 3 else 10_000.0
    backtest(cfg_path, weights_path, capital)
