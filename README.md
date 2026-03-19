# Advanced RL Trading Agent

A production-grade **Soft Actor-Critic (SAC)** reinforcement learning trading agent.

## What's New vs Baseline

| Feature | Baseline | This |
|---|---|---|
| Network | CNN-LSTM | CNN + BiLSTM + **Transformer** |
| Q-network | Standard | **Dueling** (V + A) |
| Returns | 1-step TD | **N-step TD** (n=3) |
| Memory | List PER | **Segment-tree PER** O(log N) |
| Reward | Shaped exp | **Sharpe / Sortino** |
| Risk | None | **Max-DD halt, Kelly sizing** |
| Positions | Long only | **Long + Short** |
| Costs | None | **Slippage + transaction costs** |
| Grad updates | None | **Gradient clipping** |
| LR schedule | Fixed | **Cosine annealing** |
| Checkpointing | Basic | **Best-model by Sharpe** |

## Project Structure

```
advanced_rl_trader/
├── src/
│   ├── agent/
│   │   ├── networks.py       # CNN + BiLSTM + Transformer + Dueling heads
│   │   └── sac_agent.py      # SAC with n-step, grad clip, LR schedule
│   ├── environment/
│   │   └── trading_env.py    # Long/Short env + RiskManager
│   └── memory/
│       └── replay_buffer.py  # Segment-tree PER + N-step buffer
├── scripts/
│   ├── train.py              # Full training loop
│   └── backtest.py           # Backtest + metrics + equity curve
├── config/
│   └── training_config.yaml
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Train
python scripts/train.py config/training_config.yaml

# Backtest
python scripts/backtest.py config/training_config.yaml models/best_model.pth 10000
```

## Data Format

**training_data CSV** — must have columns: `date, close` + all `state_columns` from config.

**entry_points CSV** — must have: `date, signal` where `signal = -1` (long) or `+1` (short).

## Key Concepts

### N-step Returns
Instead of `Q(s,a) ← r + γ·V(s')`, we use:
`Q(s,a) ← r₀ + γr₁ + γ²r₂ + γ³·V(s₊₃)`

Reduces variance, speeds up credit assignment across multi-step trades.

### Dueling Networks
`Q(s,a) = V(s) + A(s,a) - mean(A(s,·))`

Decouples "how good is this state" from "which action is best", 
especially useful when many actions have similar value.

### Risk-adjusted Reward
Rolling Sharpe reward: `mean(returns[-20:]) / std(returns[-20:])`

Agent directly optimises risk-adjusted performance rather than raw profit.

### Kelly Position Sizing
`f* = (b·p - q) / b`  then scaled by kelly_fraction (0.25)

Agent learns optimal bet sizing alongside TP/SL management.
