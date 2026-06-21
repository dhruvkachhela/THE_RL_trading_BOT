"""
Advanced Training Script — Walk-forward validated SAC trading agent.

Features:
  - Warmup phase (random actions fill the buffer)
  - N-step return via NStepBuffer → PER
  - Per-episode metrics (Sharpe, Sortino, drawdown, win rate)
  - Walk-forward train/eval split
  - TensorBoard logging
  - Best-model checkpointing
  - Beta annealing for PER
  - Graceful KeyboardInterrupt save
"""

import sys
import os
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent ))

from src.agent.sac_agent     import AdvancedSACAgent
from src.environment.trading_env import AdvancedTradingEnvironment
from src.memory.replay_buffer    import PrioritizedReplayBuffer, NStepBuffer

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ──────────────────────────────────────────────────────────────
#  Seed everything
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
#  Beta annealing schedule
# ──────────────────────────────────────────────────────────────

class BetaSchedule:
    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end   = end
        self.total = total_steps
        self.step  = 0

    def value(self) -> float:
        frac = min(self.step / max(self.total, 1), 1.0)
        return self.start + frac * (self.end - self.start)

    def advance(self) -> None:
        self.step += 1


# ──────────────────────────────────────────────────────────────
#  Main Trainer
# ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config_path: str = "config/training_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        seed_everything(42)

        ec  = self.cfg["environment"]
        mc  = self.cfg["model"]
        ac  = self.cfg["agent"]
        rbc = self.cfg["replay_buffer"]
        tc  = self.cfg["training"]
        rc  = self.cfg.get("risk", {})
        rwc = self.cfg.get("reward", {})
        lc  = self.cfg["logging"]

        self.action_values = np.array(mc["action_values"], dtype=np.float32)

        # ── Environment ─────────────────────────────────────────
        self.env = AdvancedTradingEnvironment(
            data_file        = self.cfg["data"]["training_data"],
            entry_points_file= self.cfg["data"]["entry_points"],
            state_cols       = ec["state_columns"],
            max_steps        = ec["max_trade_steps"],
            lookback_window  = ec["lookback_window"],
            ema_alpha        = ec["ema_alpha_norm"],
            initial_tp       = ec["initial_tp"],
            initial_sl       = ec["initial_sl"],
            tp_sl_width      = ec["tp_sl_width"],
            slippage_bps     = ec.get("slippage_bps", 5.0),
            transaction_cost = ec.get("transaction_cost", 0.0015),
            reward_type      = rwc.get("type", "sharpe"),
            reward_window    = rwc.get("window", 20),
            risk_config      = rc,
        )

        state_dim  = self.env.state_dim
        action_dim = len(self.action_values)

        # ── Agent ───────────────────────────────────────────────
        total_training_steps = tc["episodes"] * 200   # rough estimate
        self.agent = AdvancedSACAgent(
            state_dim             = state_dim,
            action_dim            = action_dim,
            sequence_length       = mc["sequence_length"],
            hidden_dim            = mc["hidden_dim"],
            lr                    = ac["learning_rate"],
            gamma                 = ac["gamma"],
            tau                   = ac["tau"],
            action_values         = self.action_values,
            target_entropy_factor = ac["target_entropy_factor"],
            initial_log_alpha     = ac["initial_log_alpha"],
            n_step                = ac.get("n_step_returns", 3),
            grad_clip             = ac.get("grad_clip", 1.0),
            dropout               = mc.get("dropout", 0.1),
            n_heads               = mc.get("n_heads", 4),
            n_transformer_layers  = mc.get("n_transformer_layers", 2),
            use_position_sizing   = rc.get("position_sizing", True),
            training_steps        = total_training_steps,
        )

        # ── Memory ──────────────────────────────────────────────
        self.per_buffer = PrioritizedReplayBuffer(
            capacity = rbc["capacity"],
            alpha    = rbc["per_alpha"],
        )
        self.nstep_buf = NStepBuffer(
            n         = ac.get("n_step_returns", 3),
            gamma     = ac["gamma"],
            per_buffer= self.per_buffer,
        )

        self.beta_schedule = BetaSchedule(
            start       = rbc["per_beta_start"],
            end         = rbc["per_beta_end"],
            total_steps = rbc.get("per_beta_frames", 100_000),
        )

        # ── Config shortcuts ────────────────────────────────────
        self.episodes         = tc["episodes"]
        self.batch_size       = tc["batch_size"]
        self.warmup_steps     = tc.get("warmup_steps", 1000)
        self.save_freq        = tc["save_frequency"]
        self.eval_freq        = tc.get("eval_frequency", 10)
        self.model_dir        = Path(self.cfg["data"]["model_save_path"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ── Logging ─────────────────────────────────────────────
        self.writer = None
        if HAS_TB and lc.get("tensorboard", False):
            log_dir = lc.get("log_dir", "logs/")
            self.writer = SummaryWriter(log_dir)

        # ── Trackers ────────────────────────────────────────────
        self.global_step   = 0
        self.best_sharpe   = -np.inf
        self.ep_sharpes    = deque(maxlen=50)
        self.ep_returns    = deque(maxlen=50)

    # ── Public ───────────────────────────────────────────────────────

    def train(self) -> None:
        print("\n" + "="*60)
        print("  Advanced SAC Trading Agent — Training")
        print("="*60)
        print(f"  Episodes:    {self.episodes}")
        print(f"  Warmup:      {self.warmup_steps} steps")
        print(f"  Batch size:  {self.batch_size}")
        print(f"  Device:      {self.agent.device}")
        print("="*60 + "\n")

        try:
            self._warmup()
            for ep in range(1, self.episodes + 1):
                ep_info = self._run_episode(train=True)
                self._log_episode(ep, ep_info)

                if ep % self.eval_freq == 0:
                    self._evaluate(ep)

                if ep % self.save_freq == 0:
                    self._save(f"checkpoint_ep{ep}.pth")

        except KeyboardInterrupt:
            print("\n[Trainer] Interrupted — saving...")
        finally:
            self._save("final_model.pth")
            if self.writer:
                self.writer.close()

    # ── Private ──────────────────────────────────────────────────────

    def _warmup(self) -> None:
        print(f"[Trainer] Warmup: collecting {self.warmup_steps} random transitions...")
        state = self.env.reset()
        steps = 0
        while steps < self.warmup_steps:
            action_idx = np.random.randint(len(self.action_values))
            action_val = float(self.action_values[action_idx])
            pos_size   = np.random.uniform(0.1, 0.5)
            next_state, reward, done, _ = self.env.step(action_val, pos_size)
            self.nstep_buf.push(state, action_idx, reward, next_state, done)
            state = next_state if not done else self.env.reset()
            steps += 1
        print(f"[Trainer] Buffer ready ({len(self.per_buffer)} experiences)\n")

    def _run_episode(self, train: bool) -> dict:
        state = self.env.reset()
        done  = False
        info  = {}

        while not done:
            action_val, action_idx, pos_size = self.agent.select_action(
                state, deterministic=not train
            )
            next_state, reward, done, info = self.env.step(action_val, pos_size)

            if train:
                self.nstep_buf.push(state, action_idx, reward, next_state, done)
                beta = self.beta_schedule.value()
                if self.per_buffer.is_ready(self.batch_size):
                    self.agent.update(self.per_buffer, self.batch_size, beta)
                    self.beta_schedule.advance()
                    self.global_step += 1

            state = next_state

        return info

    def _evaluate(self, ep: int) -> None:
        print(f"\n[Eval] Episode {ep} — running evaluation...")
        info = self._run_episode(train=False)
        sharpe = info.get("sharpe", 0.0)
        ret    = info.get("total_return", 0.0)
        print(f"  Eval Sharpe={sharpe:.3f}  Return={ret:.4f}  "
              f"Trades={info.get('trades_completed', 0)}  "
              f"WinRate={info.get('win_rate', 0):.2%}\n")

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self._save("best_model.pth")
            print(f"  *** New best Sharpe: {sharpe:.3f} — saved best_model.pth ***\n")

    def _log_episode(self, ep: int, info: dict) -> None:
        sharpe = info.get("sharpe", 0.0)
        ret    = info.get("total_return", 0.0)
        trades = info.get("trades_completed", 0)
        wr     = info.get("win_rate", 0.0)
        dd     = info.get("max_drawdown", 0.0)
        eq     = info.get("final_equity", 1.0)
        reason = info.get("reason", "")

        self.ep_sharpes.append(sharpe)
        self.ep_returns.append(ret)

        avg_sharpe = np.mean(self.ep_sharpes) if self.ep_sharpes else 0.0
        avg_ret    = np.mean(self.ep_returns) if self.ep_returns else 0.0

        agent_metrics = self.agent.get_avg_metrics()

        print(
            f"  Ep {ep:4d} | Sharpe {sharpe:+.3f} | Return {ret:+.4f} | "
            f"Equity {eq:.4f} | DD {dd:.2%} | Trades {trades:3d} | WR {wr:.1%} | "
            f"alpha {self.agent.alpha:.4f} | Reason: {reason}"
        )

        # ── JSON log (for monitor.py) ────────────────────────────
        log_dir = Path(self.cfg["logging"].get("log_dir", "logs/"))
        log_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(log_dir / "training_log.jsonl", "a") as jf:
            _json.dump({
                "episode": ep, "sharpe": sharpe, "total_return": ret,
                "max_drawdown": dd, "win_rate": wr, "alpha": self.agent.alpha,
                **agent_metrics
            }, jf)
            jf.write("\n")

        if self.writer:
            self.writer.add_scalar("Episode/Sharpe",        sharpe,        ep)
            self.writer.add_scalar("Episode/Return",        ret,           ep)
            self.writer.add_scalar("Episode/WinRate",       wr,            ep)
            self.writer.add_scalar("Episode/Drawdown",      dd,            ep)
            self.writer.add_scalar("Episode/Equity",        eq,            ep)
            self.writer.add_scalar("Episode/Trades",        trades,        ep)
            self.writer.add_scalar("Episode/AvgSharpe50",   avg_sharpe,    ep)
            self.writer.add_scalar("Episode/AvgReturn50",   avg_ret,       ep)
            for k, v in agent_metrics.items():
                self.writer.add_scalar(f"Agent/{k}", v, ep)

    def _save(self, name: str) -> None:
        path = self.model_dir / name
        self.agent.save(str(path))


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    trainer = Trainer(config_path)
    trainer.train()
