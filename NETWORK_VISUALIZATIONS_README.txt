# NETWORK ARCHITECTURE VISUALIZATIONS
=====================================

This directory contains multiple visual explanations of the network architecture in `src/agent/networks.py` to help you understand how the components connect and work together.

## 📁 **FILES PROVIDED**

### 1. **NETWORK_ARCHITECTURE_EXPLAINED.md** (Main Explanation)
- Comprehensive markdown document with detailed explanations
- Includes ASCII diagrams showing data flow and component relationships
- Covers all five major components:
  * PositionalEncoding
  * TemporalFeatureExtractor (CNN → BiLSTM → Transformer)
  * DuelingHead
  * ActorNetwork
  * CriticNetwork
- Best for: Deep understanding of concepts and implementation details

### 2. **NETWORK_DIAGRAM_SIMPLE.txt** (Quick Reference)
- Compact ASCII diagram showing overall architecture
- Focuses on connections and data flow
- Best for: Quick reference while reading code

### 3. **TENSOR_SHAPES_REFERENCE.txt** (Shape Guide)
- Detailed tensor shapes at each network stage
- Shows dimensions for batch size 32
- Includes environment interface shapes
- Best for: Debugging shape mismatches and understanding data flow

### 4. **NETWORK_CONNECTIONS_DIAGRAM.txt** (Connection Focus)
- Connection-oriented ASCII art
- Shows how information flows between components
- Highlights weight sharing and key transformations
- Best for: Understanding the neural computation graph

### 5. **ARCHITECTURE_SUMMARY.md** (Executive Summary)
- High-level overview of design philosophy
- Key innovations and trading-specific adaptations
- Practical implications for usage and modification
- Best for: Getting the big picture quickly

## 🔍 **HOW TO USE THESE VISUALIZATIONS**

### **When Reading the Code:**
1. Open `src/agent/networks.py` in your editor
2. Keep `NETWORK_ARCHITECTURE_EXPLAINED.md` or `NETWORK_DIAGRAM_SIMPLE.txt` visible
3. As you read each class, refer to the corresponding section in the visualizations
4. Use `TENSOR_SHAPES_REFERENCE.txt` to verify tensor dimensions match expectations

### **When Debugging:**
1. If you see shape errors, consult `TENSOR_SHAPES_REFERENCE.txt`
2. If you're unsure about data flow, check `NETWORK_CONNECTIONS_DIAGRAM.txt`
3. For conceptual questions, review `ARCHITECTURE_SUMMARY.md`

### **When Modifying/Experimenting:**
1. Read `ARCHITECTURE_SUMMARY.md` for design philosophy
2. Check `NETWORK_ARCHITECTURE_EXPLAINED.md` for implementation details
3. Use the tensor shapes guide to ensure your changes maintain compatibility
4. Refer to the simple diagram to understand how your changes affect overall flow

## 🎯 **KEY COMPONENTS TO UNDERSTAND**

### **PositionalEncoding** (`lines 24-44`)
- Injects sequence order into transformer inputs
- Uses sinusoidal functions: sin/cos of different frequencies
- Critical for transformer to understand temporal order

### **TemporalFeatureExtractor** (`lines 50-134`)
- Three-stage hierarchical encoder:
  1. **CNN**: Local pattern detection (kernels 3 & 5)
  2. **BiLSTM**: Medium-term temporal modeling
  3. **Transformer**: Long-range context with CLS token
- Outputs [B, 256] features shared by Actor and Critic

### **DuelingHead** (`lines 140-172`)
- Decomposes Q(s,a) = V(s) + A(s,a) - mean[A(s,a)]
- Value stream: How good is this state?
- Advantage stream: How much better is each action?
- Used in Critic network for better value estimation

### **ActorNetwork** (`lines 178-230`)
- Policy network: state → action + position size
- Action head: discrete TP/SL selection (17 actions)
- Size head: continuous position sizing [0,1] (if enabled)

### **CriticNetwork** (`lines 236-261`)
- Q-value network: state → Q-values for each action
- Uses shared feature extractor with Actor
- Implements dueling architecture (twin networks for reduced overestimation)

## 💡 **QUICK REFERENCE: DATA FLOW**

```
Market Data [B,20,16]
        │
        ▼
Temporal Feature Extractor
   [CNN → BiLSTM → Transformer(+CLS+PosEnc)]
        │
        ▼
 [B,256] Shared Features
        │
    ┌───┴───┐
    ▼       ▼
 Actor    Critic
 (Policy)  (Value)
  │         │
  ▼         ▼
[logits,size]  [Q-values]
  │         │
  ▼         ▼
Action     Used for
Selection  Learning
```

## 📐 **KEY TENSOR SHAPES (batch=32)**

| Stage | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| Input | [32, 20, 16] | - | Market data + indicators |
| TFE Out | [32, 20, 16] | [32, 256] | Shared features |
| Actor Logits | [32, 256] | [32, 17] | TP/SL action selection |
| Actor Size | [32, 256] | [32, 1] | Position size [0,1] |
| Critic Q-vals | [32, 256] | [32, 17] | Value of each action |

## 🔗 **WHERE TO FIND RELATED CODE**

- **Network definition**: `src/agent/networks.py`
- **Agent using networks**: `src/agent/sac_agent.py`
- **Environment providing states**: `src/environment/trading_env.py`
- **Training loop**: `scripts/train.py`
- **Configuration**: `config/training_config.yaml`

## 💭 **TIPS FOR DEEP UNDERSTANDING**

1. **Follow the tensor shapes**: Always verify dimensions match between layers
2. **Track the CLS token**: It's the information bottleneck that aggregates the sequence
3. **Notice weight sharing**: Actor and Critic share the TemporalFeatureExtractor
4. **Understand dueling**: The V+A−mean(A) decomposition is key to stable learning
5. **See positional encoding**: Added right before transformer self-attention
6. **Check activation functions**: Mostly GeLU with LayerNorm throughout
7. **Note dropout locations**: Applied after each major transformation for regularization

These visualizations should give you multiple perspectives on the same architecture, helping you build both intuitive and precise understanding of how this sophisticated trading agent processes market data to make trading decisions.

Happy learning! 🧠