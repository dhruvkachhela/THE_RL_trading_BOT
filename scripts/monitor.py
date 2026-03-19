"""
Training Monitor — Real-time terminal dashboard.

Reads the TensorBoard event files (or a simple JSON log) written by
train.py and renders a live dashboard in the terminal.

Features:
  - Rolling Sharpe / return / drawdown charts (ASCII sparklines)
  - Agent metrics: alpha, entropy, critic loss, actor loss
  - Best episode tracker
  - ETA + estimated total training time

Usage:
    python scripts/monitor.py logs/       # watch a log directory
    python scripts/monitor.py results/backtest_results.csv   # post-run analysis
"""

import sys
import os
import json
import time
import math
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from collections import deque

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
#  ASCII Sparkline
# ──────────────────────────────────────────────────────────────

_SPARK = " ▁▂▃▄▅▆▇█"

def sparkline(values: List[float], width: int = 30) -> str:
    """Render a list of floats as a unicode sparkline of `width` chars."""
    if not values:
        return " " * width
    vals = list(values)[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn + 1e-8
    chars = [_SPARK[int((v - mn) / rng * (len(_SPARK) - 1))] for v in vals]
    # Pad to width
    chars = [" "] * (width - len(chars)) + chars
    return "".join(chars)


def colour(text: str, code: int) -> str:
    """ANSI colour wrap (only when stdout is a TTY)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t):  return colour(t, 32)
def red(t):    return colour(t, 31)
def cyan(t):   return colour(t, 36)
def yellow(t): return colour(t, 33)
def bold(t):   return colour(t, 1)


# ──────────────────────────────────────────────────────────────
#  Simple JSON log reader (written by train.py if TB unavailable)
# ──────────────────────────────────────────────────────────────

class TrainingLogReader:
    """
    train.py can be configured to dump a JSON log per episode.
    This reader loads and tracks it.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self._pattern = str(self.log_dir / "training_log.jsonl")
        self._last_size = 0

        # History
        self.episodes:   List[int]   = []
        self.sharpes:    deque       = deque(maxlen=200)
        self.returns:    deque       = deque(maxlen=200)
        self.drawdowns:  deque       = deque(maxlen=200)
        self.win_rates:  deque       = deque(maxlen=200)
        self.alphas:     deque       = deque(maxlen=200)
        self.entropies:  deque       = deque(maxlen=200)
        self.c_losses:   deque       = deque(maxlen=200)
        self.a_losses:   deque       = deque(maxlen=200)
        self.best_sharpe = -999.0
        self.best_ep     = 0
        self.start_time  = None

    def poll(self) -> bool:
        """Read any new lines from the log file. Returns True if new data."""
        if not Path(self._pattern).exists():
            return False

        new_size = os.path.getsize(self._pattern)
        if new_size == self._last_size:
            return False
        self._last_size = new_size

        with open(self._pattern) as f:
            lines = f.readlines()

        for line in lines[len(self.episodes):]:
            try:
                d = json.loads(line.strip())
                ep = d.get("episode", len(self.episodes) + 1)
                self.episodes.append(ep)
                self.sharpes.append(d.get("sharpe", 0.0))
                self.returns.append(d.get("total_return", 0.0))
                self.drawdowns.append(d.get("max_drawdown", 0.0))
                self.win_rates.append(d.get("win_rate", 0.0))
                self.alphas.append(d.get("alpha", 0.0))
                self.entropies.append(d.get("entropy", 0.0))
                self.c_losses.append(d.get("critic_loss", 0.0))
                self.a_losses.append(d.get("actor_loss", 0.0))

                s = d.get("sharpe", 0.0)
                if s > self.best_sharpe:
                    self.best_sharpe = s
                    self.best_ep     = ep

                if self.start_time is None and len(self.episodes) == 1:
                    self.start_time = datetime.now()
            except Exception:
                pass
        return True


# ──────────────────────────────────────────────────────────────
#  Dashboard renderer
# ──────────────────────────────────────────────────────────────

class Dashboard:
    WIDTH = 72

    def __init__(self, log_dir: str, total_episodes: int = 500):
        self.reader = TrainingLogReader(log_dir)
        self.total_eps = total_episodes

    def _bar(self, value: float, max_val: float, width: int = 20, fill: str = "█") -> str:
        n = max(0, min(int(value / (max_val + 1e-8) * width), width))
        return fill * n + "░" * (width - n)

    def _fmt(self, v: float, decimals: int = 4, sign: bool = True) -> str:
        s = f"{v:+.{decimals}f}" if sign else f"{v:.{decimals}f}"
        return green(s) if v >= 0 else red(s)

    def render(self) -> None:
        r = self.reader
        ep_done = len(r.episodes)
        if ep_done == 0:
            print("  Waiting for training data…")
            return

        # ETA
        eta_str = "—"
        if r.start_time and ep_done > 0:
            elapsed = (datetime.now() - r.start_time).total_seconds()
            spe     = elapsed / ep_done
            remain  = max(0, self.total_eps - ep_done) * spe
            eta_str = str(timedelta(seconds=int(remain)))

        sep = "─" * self.WIDTH

        os.system("clear" if os.name != "nt" else "cls")

        print(bold(f"  {'SAC Trading Agent — Live Monitor':^{self.WIDTH-4}}"))
        print(f"  {sep}")
        print(f"  Episodes: {ep_done}/{self.total_eps}  "
              f"ETA: {cyan(eta_str)}  "
              f"Best Sharpe: {green(f'{r.best_sharpe:+.4f}')} (ep {r.best_ep})")
        print(f"  {sep}")

        # ── Sharpe sparkline ────────────────────────────────────
        sharpe_now = r.sharpes[-1] if r.sharpes else 0.0
        print(f"  Sharpe  {self._fmt(sharpe_now)}  │{sparkline(list(r.sharpes))}│")

        ret_now = r.returns[-1] if r.returns else 0.0
        print(f"  Return  {self._fmt(ret_now)}  │{sparkline(list(r.returns))}│")

        dd_now = r.drawdowns[-1] if r.drawdowns else 0.0
        print(f"  Drawdwn {red(f'{dd_now:.4f}'):>10}  │{sparkline(list(r.drawdowns))}│")

        wr_now = r.win_rates[-1] if r.win_rates else 0.0
        print(f"  WinRate {cyan(f'{wr_now:.2%}'):>10}  │{sparkline(list(r.win_rates))}│")

        print(f"  {sep}")

        # ── Agent metrics ───────────────────────────────────────
        al_now = r.alphas[-1]    if r.alphas    else 0.0
        en_now = r.entropies[-1] if r.entropies else 0.0
        cl_now = r.c_losses[-1]  if r.c_losses  else 0.0
        acl_now= r.a_losses[-1]  if r.a_losses  else 0.0

        print(f"  α (temp)  {yellow(f'{al_now:.5f}'):>10}  │{sparkline(list(r.alphas))}│")
        print(f"  Entropy   {cyan(f'{en_now:.4f}'):>10}  │{sparkline(list(r.entropies))}│")
        print(f"  CriticL   {red(f'{cl_now:.4f}'):>10}  │{sparkline(list(r.c_losses))}│")
        print(f"  ActorL    {red(f'{acl_now:.4f}'):>10}  │{sparkline(list(r.a_losses))}│")

        print(f"  {sep}")

        # ── Rolling averages ────────────────────────────────────
        w = 20
        avg_sh = np.mean(list(r.sharpes)[-w:])  if r.sharpes  else 0.0
        avg_wr = np.mean(list(r.win_rates)[-w:]) if r.win_rates else 0.0
        avg_dd = np.mean(list(r.drawdowns)[-w:]) if r.drawdowns else 0.0
        avg_re = np.mean(list(r.returns)[-w:])   if r.returns   else 0.0

        print(f"  Rolling {w}-ep averages:")
        print(f"    Sharpe  {self._fmt(avg_sh)}   Return  {self._fmt(avg_re)}")
        print(f"    WinRate {cyan(f'{avg_wr:.2%}'):>10}   MaxDD   {red(f'{avg_dd:.2%}'):>10}")
        print(f"  {sep}")
        print(f"  Updated: {datetime.now().strftime('%H:%M:%S')}")

    def watch(self, refresh_secs: float = 2.0) -> None:
        print("[Monitor] Watching for training data…  (Ctrl-C to stop)")
        try:
            while True:
                self.reader.poll()
                self.render()
                time.sleep(refresh_secs)
        except KeyboardInterrupt:
            print("\n[Monitor] Stopped.")


# ──────────────────────────────────────────────────────────────
#  Post-run analysis mode (read a results CSV)
# ──────────────────────────────────────────────────────────────

def analyse_results(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n[Monitor] Analysing {csv_path}  ({len(df)} trades)\n")

    rets  = df["actual_return"].values if "actual_return" in df.columns else df.iloc[:, 0].values
    equity = np.cumprod(1 + rets) * 10_000.0
    peaks  = np.maximum.accumulate(equity)
    dd     = (peaks - equity) / (peaks + 1e-8) * 100
    neg    = rets[rets < 0]

    sharpe  = rets.mean() / (rets.std() + 1e-8)
    sortino = rets.mean() / (neg.std() + 1e-8) if len(neg) else 0.0
    calmar  = rets.mean() / (dd.max() / 100 + 1e-8)

    sep = "─" * 50
    print(bold("  PERFORMANCE SUMMARY"))
    print(f"  {sep}")
    print(f"  Trades          : {len(df)}")
    print(f"  Win Rate        : {(rets > 0).mean():.2%}")
    print(f"  Avg Return      : {rets.mean():+.4f}")
    print(f"  Sharpe Ratio    : {sharpe:+.4f}")
    print(f"  Sortino Ratio   : {sortino:+.4f}")
    print(f"  Calmar Ratio    : {calmar:+.4f}")
    print(f"  Max Drawdown    : {dd.max():.2f}%")
    print(f"  Final Equity    : ${equity[-1]:,.2f}")

    if "reason" in df.columns:
        print(f"\n  EXIT REASONS:")
        for r, c in df["reason"].value_counts().items():
            print(f"    {r:<20} {c:>5}  ({c/len(df):.1%})")

    print(f"\n  Return Distribution:")
    pct = np.percentile(rets, [5, 25, 50, 75, 95])
    print(f"    p5={pct[0]:+.4f}  p25={pct[1]:+.4f}  p50={pct[2]:+.4f}  "
          f"p75={pct[3]:+.4f}  p95={pct[4]:+.4f}")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Trading Agent Monitor")
    parser.add_argument("path", help="Log directory OR results CSV path")
    parser.add_argument("--total-eps", type=int, default=500, help="Total training episodes")
    parser.add_argument("--refresh",   type=float, default=2.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.path.endswith(".csv"):
        analyse_results(args.path)
    else:
        dash = Dashboard(args.path, total_episodes=args.total_eps)
        dash.watch(refresh_secs=args.refresh)
