# Advanced RL Trading Agent Network Architecture - Summary

## 🏗️ **Core Design Philosophy**

This network implements a **hierarchical feature extractor** combined with **dueling actor-critic** architecture specifically designed for financial trading. The design philosophy emphasizes:

1. **Multi-scale temporal processing** (CNN → BiLSTM → Transformer) to capture patterns at different time scales
2. **Parameter efficiency** through shared feature extractor between actor and critic
3. **Interpretability** via explicit action and value decomposition
4. **Trading-specific adaptations** like position sizing and risk-aware action spaces

## 🔑 **Key Innovations**

### **1. Three-Stage Temporal Feature Extractor**
```
Market Data [B, T=20, F=16]
        │
        ├─→ **CNN Stage**: Local pattern detection (3&5 kernel concat)
        │     → [B, T, H]  (candlesticks, bar formations)
        │
        ├─→ **BiLSTM Stage**: Medium-term temporal modeling  
        │     → [B, T, H]  (trends, mean-reversion windows)
        │
        └─→ **Transformer Stage**: Long-range context with CLS token
              → [B, H]     (regime detection, structural breaks)
```

**Why this works for trading:**
- CNN: Finds repetitive price patterns regardless of absolute position
- BiLSTM: Understands directional momentum and persistence
- Transformer: Attends to similar past market states (e.g., "this looks like March 2020")

### **2. Dueling Actor-Critic with Position Sizing**
```
FEATURES [B, 256]
        │
        ├─→ **ACTOR**:
        │    │
        │    ├─→ **Action Head**: Discrete TP/SL decisions [17 actions]
        │    │                   (When to take profit/stop loss)
        │    │
        │    └─→ **Size Head**: Continuous position sizing [0,1]
        │                      (How much to risk - Kelly-inspired)
        │
        └─→ **CRITIC** (Twin Dueling):
             │
             ├─→ **Value Stream**: State quality V(s) [1]
             │                   ("How good is this market state?")
             │
             └─→ **Advantage Stream**: Relative action quality [17]
                       ("How much better is each specific action?")
                         
             Q(s,a) = V(s) + A(s,a) - meanₐ[A(s,a)]
```

**Trading advantages:**
- Separates market assessment from action selection
- Enables learning of sophisticated risk management
- Handles cases where many TP/SL combinations have similar value
- Jointly learns timing (when) and sizing (how much)

### **3. Positional Encoding for Temporal Awareness**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Critical for trading because:**
- Same indicator pattern has different meaning at different times
- Enables learning of temporal contexts (e.g., "RSI>70 during uptrend vs downtrend")
- Allows transformer to understand sequence order without recurrence

### **4. Integration with Advanced SAC Features**
The networks work seamlessly with:
- **N-step returns** (n=3): Better credit assignment across multi-step trades
- **Prioritized Experience Replay**: Focus on surprising experiences
- **Automatic entropy tuning**: Adaptive exploration-exploitation balance
- **Gradient clipping**: Stable training with RNN/Transformer components
- **Cosine LR annealing**: Fine-tuning to optimal solutions

## 📊 **Architecture Specifications**

| Component | Input Shape | Output Shape | Key Parameters |
|-----------|-------------|--------------|----------------|
| **Input** | [B, 20, 16] | - | 16 technical indicators |
| **Temporal Feature Extractor** | [B, 20, 16] | [B, 256] | CNN(3,5) + BiLSTM(2L) + Transformer(2L,4H) |
| **Actor Network** | [B, 256] | [B, 17] + [B, 1] | Action logits + Position size |
| **Critic Network** | [B, 256] | [B, 17] | Q-values via Dueling (V+A→Q) |
| **Positional Encoding** | [B, T+1, H] | [B, T+1, H] | Sinusoidal, d_model=256 |

## 💡 **How to Interpret the Networks**

### **When examining `networks.py`: Follow the DATA**

1. **TemporalFeatureExtractor.forward()**:
   - Watch tensor shapes: [B,T,F] → [B,T,H] → [B,T+1,H] → [B,H]
   - Note the CLS token extraction: `return xt[:, 0, :]`
   - See where positional encoding is added: `xt = self.pos_enc(xt)`

2. **ActorNetwork.forward()**:
   - Shared features → two separate heads
   - Action head: logits for discrete TP/SL selection
   - Size head: continuous [0,1] output via sigmoid

3. **CriticNetwork.forward()**:
   - Shared features → dueling head
   - Value stream: single output V(s)
   - Advantage stream: 17 outputs A(s,a) for each action
   - Combination: `v + a - a.mean(dim=1, keepdim=True)`

### **Key Dimensions to Track**

- **Batch size (B)**: Varies during training (typically 32-128)
- **Sequence length (T)**: 20 (matches lookback_window)
- **Features (F)**: 16 technical indicators from config
- **Hidden dimension (H)**: 256 (model width)
- **Actions**: 17 discrete TP/SL displacements [-0.07 to 0.07]
- **CLS token**: Adds 1 to sequence length for transformer ([B,21,H])

## 🔄 **Information Flow During Trading Decision**

```
Raw Market Indicators
           │
           ▼
[20 time steps] × [16 features]  ← EMA-normalized in environment
           │
           ▼
TEMPORAL FEATURE EXTRACTOR
   CNN: "What local patterns do I see?" 
   BiLSTM: "What's the medium-term trend?"
   Transformer + CLS: "What similar market regimes does this resemble?"
           │
           ▼
     [256-dim market summary]
           │
     ┌─────┼───────┐
     ▼       ▼     ▼
ACTOR:                             CRITIC:
 ┌─────┴─────┐                 ┌─────┴─────┐
 │           │                 │           │
 ▼           ▼                 ▼           │
[17 LOGITS] [SIZE FRACTION]  [VALUE]   [17 ADVANTAGES]
 │       │               │       │
 │       ▼               │       │
 │   [0,1] POSITION      │       │
 │                       │       │
 │    WHEN to trade?     │  HOW good is state?
 │    WHICH direction?   │       │
 │    HOW MUCH to risk?  │  HOW much better is each action?
 └───────┬───────┘       │       │
         ▼               ▼       │
   EXECUTE TRADE      COMBINE: Q = V + A - mean(A)
                              │
                              ▼
                       [17] Q-VALUES
                              │
                              ▼
                    Used for TD-target in learning
```

## 🎯 **Trading-Specific Design Choices**

### **Action Space Design**
- **17 discrete actions**: Fine-grained TP/SL control [-0.07 to +0.07 in 0.01 increments]
- **Why not continuous?**: Easier exploration, matches typical trader thinking in discrete steps
- **Center displacement**: Action value shifts both TP and SL symmetrically around entry

### **Position Sizing Integration**
- **Continuous [0,1] output**: Fraction of capital to risk
- **Learned jointly with timing**: Agent discovers optimal timing+sizing combinations
- **Compatible with Kelly**: RiskManager applies Kelly criterion to actor's size output

### **Reward Function Flexibility**
Configured via `training_config.yaml`:
- `reward.type: "sharpe"` → Optimizes risk-adjusted returns directly
- `reward.type: "sortino"` → Focuses on downside protection
- `reward.type: "shaped"` → Traditional profit shaping + TP/SL bonuses

## ⚙️ **Practical Implications**

### **For Understanding the Code**
1. **Start with shapes**: Always check tensor dimensions match comments
2. **Follow the CLS token**: It's the bottleneck that aggregates sequence info
3. **Watch for sharing**: Actor and Critic share the first half of networks
4. **Note the dueling**: Value and advantage streams are processed separately

### **For Modification & Experimentation**
1. **Change sequence length**: Update both config and PositionalEncoding max_len
2. **Adjust hidden dimension**: Affects all downstream layers proportionally
3. **Experiment with positional encodings**: Try learned or relative versions
4. **Modify action space**: Change `action_values` in config for different granularity
5. **Toggle position sizing**: Set `use_position_sizing: false` in risk config

### **Performance Characteristics**
- **Forward pass**: ~2-5ms on modern GPU (batch=32)
- **Parameters**: ~1.5M total (much smaller than vision/NLP models)
- **Memory**: Efficient due to moderate hidden dimensions and sequence length
- **Scalability**: Easy to increase capacity by adjusting hidden_dim or layers

## 📚 **Further Reading & Connections**

### **Foundational Papers**
1. **Transformer**: Vaswani et al. 2017 - "Attention is All You Need"
2. **Dueling Networks**: Wang et al. 2016 - "Dueling Network Architectures for Deep RL"
3. **SAC**: Haarnoja et al. 2018 - "Soft Actor-Critic Algorithms and Applications"
4. **PER**: Schaul et al. 2016 - "Prioritized Experience Replay"

### **Trading-Specific RL**
1. **Moody & Saffell**: Learning to Trade via Direct Reinforcement
2. **Deng et al.**: Deep Direct Reinforcement Learning for Financial Signal Trading
3. **Jiang et al.**: Deep Reinforcement Learning for Stock Trading

### **Architecture Variations to Explore**
- **CNN-only**: For ultra-local pattern detection
- **Transformer-only**: For pure attention-based modeling  
- **Different positional encodings**: Learned, relative, or rotary (RoPE)
- **Attention visualization**: To see what time steps the model focuses on
- **Feature importance**: Via SHAP or integrated gradients on the 16 inputs

## ✅ **Verification Checklist**

When reading or modifying `networks.py`, verify:

- [ ] Tensor shapes match between layers (especially after permutes/reshape)
- [ ] CLS token is properly prepended and extracted
- [ ] Positional encoding is added before transformer self-attention
- [ ] Dueling combination follows: Q = V + A - mean(A)
- [ ] Actor outputs both discrete logits and continuous size (if enabled)
- [ ] Critic uses dueling architecture (separate value/advantage streams)
- [ ] Layer norms and dropouts are in expected places
- [ ] Activation functions match documentation (GeLU, not ReLU)
- [ ] Skip connections/residuals are present where expected (in transformer)

This architecture represents a thoughtful integration of modern deep learning techniques with domain-specific financial trading requirements. The modular design allows for experimentation while maintaining a solid foundation proven effective in the trading domain.

**Next steps**: Examine how these networks are used in `sac_agent.py` to see the complete learning loop!