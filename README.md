# Advanced RL Trading Agent

A production-grade **Soft Actor-Critic (SAC)** reinforcement learning trading agent.

## Performance Benchmark

Results on out-of-sample test data with realistic execution friction (5 bps slippage + 15 bps transaction commission):

| Metric | Result |
| :--- | :--- |
| **Total Return** | **+10.96%** |
| **Profit Factor** | **1.307** |
| **Win Rate** | **55.77%** (52 trades) |
| **Max Drawdown** | **7.19%** |
| **Sharpe Ratio** | **+0.1208** |
| **Sortino Ratio** | **+0.2747** |
| **Short Win Rate** | **62.07%** |
| **Long Win Rate** | **47.83%** |

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
THE_RL_trading_BOT/
├── config/
│   ├── training_config.yaml          # Full production training configuration
│   ├── medium_test_config.yaml       # Balanced 50-episode training config
│   ├── quick_test_config.yaml        # Fast smoke-test configuration
│   └── no_transformer_test_config.yaml # Ablation study config
├── data/
│   ├── trainalt_data.csv             # Continuous OHLCV candles + 16 indicators
│   └── deflection_data.csv           # Macro deflection entry alert points
├── src/
│   ├── agent/
│   │   ├── networks.py               # CNN + BiLSTM + Transformer + Dueling heads
│   │   └── sac_agent.py              # SAC with n-step, grad clip, LR schedule
│   ├── environment/
│   │   └── trading_env.py            # Long/Short env + RiskManager
│   ├── memory/
│   │   └── replay_buffer.py          # Segment-tree PER + N-step buffer
│   └── utils/
│       └── features.py               # Feature pipeline & technical indicators
├── scripts/
│   ├── train.py                      # Full training loop
│   ├── backtest.py                   # Backtest + metrics + equity curve
│   ├── walk_forward.py               # Out-of-sample rolling cross-validation
│   ├── tune.py                       # Optuna hyperparameter optimization
│   ├── live_trade.py                 # Live / paper trading interface
│   └── monitor.py                    # Terminal sparklines dashboard
├── models/
│   └── best_model.pth                # Optimal trained weights checkpoint
├── results/
│   ├── backtest_results.csv          # Trade-by-trade execution log
│   └── equity_curve.png              # Backtest equity curve plot
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Train
python scripts/train.py config/medium_test_config.yaml

# Backtest
python scripts/backtest.py config/medium_test_config.yaml models/best_model.pth 10000

# Walk-forward cross-validation
python scripts/walk_forward.py config/medium_test_config.yaml

# Terminal dashboard monitor
python scripts/monitor.py logs/
```

## Data Format

**training_data CSV** — must have columns: `date, close` + all `state_columns` from config.

**entry_points CSV** — must have: `date, signal` where `signal = -1` (long) or `+1` (short).

## Key Concepts

### N-step Returns
Instead of $Q(s,a) \leftarrow r + \gamma \cdot V(s')$, we use:
$Q(s,a) \leftarrow r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 \cdot V(s_{+3})$

Reduces variance and accelerates credit assignment across multi-step trades.

### Dueling Networks
$Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s,\cdot))$

Decouples "how good is this state" from "which action is best", especially useful when many actions have similar value in flat markets.

### Risk-adjusted Reward
Rolling Sharpe reward: $\text{mean}(\text{returns}[-20:]) / \text{std}(\text{returns}[-20:])$

Agent directly optimizes risk-adjusted performance rather than raw profit.

### Kelly Position Sizing
$f^* = (b \cdot p - q) / b$, scaled by `kelly_fraction` (0.20) and capped at 25% max capital risk.

Agent learns optimal bet sizing alongside TP/SL management.

## Detailed Documentation Guides

* [PROJECT_ARCHITECTURE_VISUAL_GUIDE.md](file:///c:/Users/dhruv/Downloads/GROWN%20WINGS/THE_RL_trading_BOT/PROJECT_ARCHITECTURE_VISUAL_GUIDE.md): Complete visual flowcharts and tensor shape pipelines.
* [COMPLETE_SYSTEM_AND_INTERVIEW_DEFENSE.md](file:///c:/Users/dhruv/Downloads/GROWN%20WINGS/THE_RL_trading_BOT/COMPLETE_SYSTEM_AND_INTERVIEW_DEFENSE.md): System engineering guide and 14 technical quant/RL interview defense questions.
