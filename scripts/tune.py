"""
Hyperparameter Optimisation using Optuna.

Searches over:
  - Learning rate
  - Hidden dimension
  - N-step returns
  - Gamma (discount factor)
  - Tau (soft-update)
  - Reward type
  - TP/SL width
  - Transformer heads
  - Dropout

Objective: maximise OOS Sharpe ratio (single walk-forward window).

Usage:
    pip install optuna
    python scripts/tune.py config/training_config.yaml --trials 50
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[Tune] optuna not installed. Run: pip install optuna")

from src.agent.sac_agent         import AdvancedSACAgent
from src.environment.trading_env import AdvancedTradingEnvironment
from src.memory.replay_buffer    import PrioritizedReplayBuffer, NStepBuffer


# ──────────────────────────────────────────────────────────────
#  Single-trial evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_params(base_cfg: Dict, params: Dict, episodes: int = 30) -> float:
    """
    Train an agent with the given params for `episodes` episodes and
    return the Sharpe ratio on the last evaluation run.
    """
    cfg = deepcopy(base_cfg)

    # Apply trial params
    cfg["agent"]["learning_rate"]       = params["lr"]
    cfg["model"]["hidden_dim"]          = params["hidden_dim"]
    cfg["agent"]["n_step_returns"]      = params["n_step"]
    cfg["agent"]["gamma"]               = params["gamma"]
    cfg["agent"]["tau"]                 = params["tau"]
    cfg["environment"]["tp_sl_width"]   = params["tp_sl_width"]
    cfg["model"]["n_heads"]             = params["n_heads"]
    cfg["model"]["dropout"]             = params["dropout"]
    cfg["reward"]["type"]               = params["reward_type"]

    ec  = cfg["environment"]
    mc  = cfg["model"]
    ac  = cfg["agent"]
    rc  = cfg.get("risk", {})
    rwc = cfg.get("reward", {})
    rbc = cfg["replay_buffer"]
    tc  = cfg["training"]

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
        grad_clip             = ac.get("grad_clip", 1.0),
        dropout               = mc.get("dropout", 0.1),
        n_heads               = mc.get("n_heads", 4),
        n_transformer_layers  = mc.get("n_transformer_layers", 2),
        use_position_sizing   = rc.get("position_sizing", True),
    )

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

    per_buf = PrioritizedReplayBuffer(rbc["capacity"] // 4, rbc["per_alpha"])
    n_buf   = NStepBuffer(ac.get("n_step_returns", 3), ac["gamma"], per_buf)
    batch   = tc["batch_size"]

    # Quick warmup
    state = env.reset()
    for _ in range(300):
        ai = np.random.randint(len(action_values))
        av = float(action_values[ai])
        ns, r, done, _ = env.step(av, 0.3)
        n_buf.push(state, ai, r, ns, done)
        state = ns if not done else env.reset()

    # Train
    for _ in range(episodes):
        state = env.reset()
        done  = False
        while not done:
            av, ai, ps = agent.select_action(state)
            ns, r, done, _ = env.step(av, ps)
            n_buf.push(state, ai, r, ns, done)
            if per_buf.is_ready(batch):
                agent.update(per_buf, batch, beta=0.6)
            state = ns

    # Evaluate (deterministic)
    state = env.reset()
    done  = False
    info  = {}
    while not done:
        av, _, ps = agent.select_action(state, deterministic=True)
        state, _, done, info = env.step(av, ps)

    return info.get("sharpe", 0.0)


# ──────────────────────────────────────────────────────────────
#  Optuna objective
# ──────────────────────────────────────────────────────────────

def make_objective(base_cfg: Dict, episodes: int):
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "lr":          trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "hidden_dim":  trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            "n_step":      trial.suggest_int("n_step", 1, 5),
            "gamma":       trial.suggest_float("gamma", 0.95, 0.999),
            "tau":         trial.suggest_float("tau", 0.001, 0.02),
            "tp_sl_width": trial.suggest_float("tp_sl_width", 0.01, 0.06),
            "n_heads":     trial.suggest_categorical("n_heads", [2, 4, 8]),
            "dropout":     trial.suggest_float("dropout", 0.0, 0.3),
            "reward_type": trial.suggest_categorical("reward_type", ["sharpe", "sortino", "shaped"]),
        }
        try:
            return evaluate_params(base_cfg, params, episodes)
        except Exception as e:
            print(f"[Tune] Trial failed: {e}")
            return -999.0

    return objective


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def run_tuning(config_path: str, n_trials: int = 50, episodes_per_trial: int = 30) -> None:
    if not HAS_OPTUNA:
        print("Install optuna first: pip install optuna")
        return

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    print(f"\n[Tune] Starting Optuna search: {n_trials} trials × {episodes_per_trial} episodes")

    study = optuna.create_study(
        direction  = "maximize",
        study_name = "sac_trading",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        make_objective(base_cfg, episodes_per_trial),
        n_trials  = n_trials,
        n_jobs    = 1,
        show_progress_bar = True,
    )

    best = study.best_trial
    print("\n" + "="*60)
    print("  BEST HYPERPARAMETERS")
    print("="*60)
    print(f"  Sharpe (objective): {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<20} {v}")
    print("="*60)

    # Save best params to YAML
    out_cfg = deepcopy(base_cfg)
    out_cfg["agent"]["learning_rate"]     = best.params["lr"]
    out_cfg["model"]["hidden_dim"]        = best.params["hidden_dim"]
    out_cfg["agent"]["n_step_returns"]    = best.params["n_step"]
    out_cfg["agent"]["gamma"]             = best.params["gamma"]
    out_cfg["agent"]["tau"]               = best.params["tau"]
    out_cfg["environment"]["tp_sl_width"] = best.params["tp_sl_width"]
    out_cfg["model"]["n_heads"]           = best.params["n_heads"]
    out_cfg["model"]["dropout"]           = best.params["dropout"]
    out_cfg["reward"]["type"]             = best.params["reward_type"]

    out_path = Path(config_path).parent / "best_config.yaml"
    import yaml as _yaml
    with open(out_path, "w") as f:
        _yaml.dump(out_cfg, f, default_flow_style=False)
    print(f"\n[Tune] Best config saved → {out_path}")

    # Save all trials
    trials_df = study.trials_dataframe()
    out_dir   = Path(base_cfg["data"].get("results_path", "results/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    print(f"[Tune] All trials saved → {out_dir / 'optuna_trials.csv'}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    n_trials    = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    episodes    = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    run_tuning(config_path, n_trials, episodes)
