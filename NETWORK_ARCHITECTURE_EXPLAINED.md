# Advanced RL Trading Agent Network Architecture - Visual Explanation

This document provides a visual and conceptual explanation of the neural network architecture in `networks.py` to help you understand how the components connect and work together.

## 🏗️ **Overall Architecture Overview**

```
                    INPUT: Market Data Tensor
                    Shape: [Batch, Sequence_Length=20, Features=16]
                                                             ┌─────────────────────┐
                                                             │  Action Logits    │
                                                             │  [Batch, 17]     │
                    ┌─────────────────────┐                  │ (TP/SL Displacement)│
                    │  Actor Network      │◄─────────────────┤                     │
                    │                     │                  └──────────────┬──────┘
                    │  ┌─────────────┐    │                                 │
          ┌─────────┤  │ Feature     │    │                                 │
          │         │  │ Extractor   │    │                                 ▼
          │         │  │ (CNN→BiLSTM│    │                  ┌─────────────────────┐
          │         │  │  →Transformer)│    │                  │ Position Size Head  │
          │         │  │             │    │                  │ [Batch, 1] (Sigmoid)  │
          │         │  └─────────────┘    │                  └──────────────┬──────┘
          │         │                     │                                 │
 INPUT ─┤         │                     │                                   │
         │         │                     │                                   ▼
         │         │                     │                           ┌─────────────────┐
         │         │                     └─────► Critic Network     │  Value Stream   │
         │         │                           (Twin Dueling)      │  [Batch, 1]     │
         │         │                              │               │                 │
         │         │                              │          ┌────▼────┐            │
INPUT ─┤         │                              │          │Adv. Stream│            │
         │         │                              │          │[Batch,17]│            │
         │         │                              │          └────┬───┘            │
         │         │                              │               │                │
         │         │                              │          ┌──────────────┐    │
         │         │                              │          │Q(s,a) = V + A - mean(A)│
         │         │                              │          └──────────────┘    │
         │         │                              │               ▼                │
         │         │                              │          ┌────────────────┐  │
         │         │                              │          │  Q-values    │  │
         │         │                              │          │ [Batch,17]  │  │
         │         │                              │          └──────────────┘  │
         │         │                              │                    │         │
         │         │                              │                    ▼         │
         │         │                              │               ◄────────────┘ │
         │         │                              │                              │
         │         │                              │           Critic Output      │
         │         │                              │             [Batch,17]       │
         │         │                              └──────────────────────────────┘
         │         │                                       ▲
         │         │                                       │
         │         │                              ┌────────▼────────┐
         │         │                              │  Advantage      │
         │         │                              │  (for Actor)    │
         │         │                              └────────────────┘
         │         │
         │         ▼
         │  ┌─────────────────┐
         └─►│Feature Extractor│◄─────┐
            │(CNN→BiLSTM→Trans)│      │
            └─────────────────┘      │
                                     │
                           ┌─────────▼─────────┐
                           │Positional Encoding│
                           │(Added before Trans)│
                           └───────────────────┘
```

## 🧩 **Component-by-Component Breakdown**

### 1. **PositionalEncoding** (`lines 24-44`)
**Purpose**: Injects sequence order information into transformer inputs

```
Input:  [Batch_size, Seq_Len, Hidden_Dim]  (e.g., [32, 21, 256])
                          │
                          ▼
     PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
     PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
                          │
                          ▼
     Output: [Batch_size, Seq_Len, Hidden_Dim]  (Same shape!)
                          │
                          ▼
              Added to transformer input
```

**Why sequence_length+1?** Because we prepend a CLS token:
- Market features: 20 time steps
- + CLS token: 1 time step  
- Total sequence length: 21

### 2. **TemporalFeatureExtractor** (`lines 50-134`)
**Three-Stage Hierarchical Encoder**

```
                    INPUT: [Batch, T=20, F=16]
                                  │
        ┌────────────────────────┴─────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌─────────────┐                                ┌────────────────┐
│   CNN Stage │                                │                │
│(Kernel-3 &  │       CONCATENATE              │                │
│ Kernel-5 )  │◄───────────────────────────────┤                │
└─────┬───────┘                                │                │
      │                                        │                │
      ▼                                        ▼                │
┌─────────────┐                        ┌────────────────┐    │
│ LayerNorm   │                        │                │    │
│ & Dropout   │◄───────────────────────┤                │    │
└─────┬───────┘                        │                │    │
      │                                │                │    │
      ▼                                ▼                │    │
┌─────────────┐                ┌────────────────┐    │    │
│ Permute:    │                │                │    │    │
│ [B,F,T]→    │                │  BiLSTM Stage  │    │    │
│ [B,T,F]     │                │(2 layers, bidir)◄┘    │    │
└─────┬───────┘                └─────────┬───────┘    │    │
      │                                  │              │    │
      ▼                                  ▼              │    │
┌─────────────┐                ┌────────────────┐    │    │
│ LayerNorm   │                │                │    │    │
│ & Dropout   │◄───────────────┤                │    │    │
└─────┬───────┘                └──────┬─────────┘    │    │
      │                                │              │    │
      ▼                                ▼              │    │
┌─────────────┐                ┌────────────────┐    │    │
│             │                │                │    │    │
│  TRANSFORMER│                │                │    │    │
│  ENCODER    │                │                │    │    │
│ (2 layers,  │                │                │    │    │
│  4 heads,   │                │                │    │    │
│  CLS token) │◄───────────────┤                │    │    │
└─────┬───────┘                └────────────────┘    │    │
      │                                               │    │
      ▼                                               │    │
┌────────────────────┐                        ┌────────▼────┐
│   CLS Token Out    │◄───────────────────────┤    ...      │
│ [Batch, Hidden]    │                        │             │
└────────────────────┘                        └─────────────┘
                              OUTPUT: [Batch, Hidden_Dim=256]
```

### 3. **DuelingHead** (`lines 140-172`)
**Value-Advantage Decomposition**

```
                    INPUT: [Batch, Hidden_Dim=256]
                                  │
                    ┌─────────────┴─────────────┐
                    │                             │
                    ▼                             ▼
            ┌─────────────┐               ┌─────────────┐
            │ Value Stream│               │ Advantage   │
            │ (V(s))      │               │ Stream      │
            │             │               │ (A(s,a))    │
            │ Linear 256→ │               │ Linear 256→ │
            │ Hidden,GELU,│               │ Hidden,GELU,│
            │ LN,Dropout, │               │ LN,Dropout, │
            │ Linear→1    │               │ Linear→17   │
            └─────┬───────┘               └──────┬──────┘
                  │                              │
                  ▼                              ▼
               [Batch,1]                   [Batch,17]
                  │                              │
                  │           ┌──────────────────┴──────────────────┐
                  │           │                                     │
                  │           ▼                                     ▼
                  │        [Batch,1]                    [Batch,17]  │
                  │           │                              │      │
                  │           │           mean over actions    │      │
                  │           │                    ────────────►    │
                  │           │                              │      │
                  │           │           subtract mean        │      │
                  │           │                    ◄──────────┘      │
                  │           │                              │      │
                  │           ▼                              ▼      │
                  │        [Batch,1]                   [Batch,17]  │
                  │           │                              │      │
                  │           └───────────┐    ┌──────────────┘      │
                  │                       ▼    ▼                       │
                  │                 ┌───────────────┐                 │
                  │                 │    Q(s,a) =   │                 │
                  │                 │ V(s) + A(s,a) │                 │
                  │                 │   - mean(A)   │                 │
                  │                 └───────────────┘                 │
                  │                       │                           │
                  │                       ▼                           │
                  │                 ┌───────────────┐                 │
                  │                 │  Q-values     │                 │
                  │                 │ [Batch,17]    │                 │
                  │                 └───────────────┘                 │
                  └───────────────────────────────────────────────────┘
                                        │
                                        ▼
                              OUTPUT: [Batch, 17] (Q-values for each action)
```

### 4. **ActorNetwork** (`lines 178-230`)
**Policy Network: State → Action + Position Size**

```
                    INPUT: [Batch, T=20, F=16]
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐        ┌─────────────────┐
           │Feature Extractor│        │  (Shared        │
           │(CNN→BiLSTM→Trans)│        │   with Critic)  │
           └───────┬─────────┘        └─────────────────┘
                   │                           │
                   ▼                           ▼
            ┌─────────────┐           ┌─────────────┐
            │ Action Head   │         │Size Head    │
            │               │         │(if enabled) │
            │ Linear 256→   │         │ Linear 256→ │
            │ Hidden,GELU,  │         │ Hidden/2→   │
            │ LN,Dropout,   │         │ Hidden/4→   │
            │ Linear→17     │         │ Linear→1    │
            └─────┬───────┘         │ Sigmoid     │
                  │                 └──────┬──────┘
                  │                        │
                  ▼                        ▼
        [Batch,17]            [Batch,1] (Position Size)
        (Logits)              [0,1] fraction
                                  │
                                  ▼
                          Final Output:
                          ├── Action Logits: [Batch, 17]
                          └── Position Size: [Batch, 1]
```

### 5. **CriticNetwork** (`lines 236-261`)
**Q-Value Network: State → Q-values (Using Dueling)**

```
                    INPUT: [Batch, T=20, F=16]
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐        ┌─────────────┐
           │Feature Extractor│        │  (Shared    │
           │(CNN→BiLSTM→Trans)│        │   architecture│
           └───────┬─────────┘        │   as Actor) │
                   │                  └─────────────┘
                   ▼                            │
            ┌─────────────┐                    │
            │Dueling Head   │◄─────────────────┘
            │(V+A→Q)        │
            └─────┬───────┘
                  │
                  ▼
           ┌─────────────┐
           │ Q-values    │
           │ [Batch,17]  │
           └─────────────┘
```

## 📊 **Data Flow During Forward Pass**

### **Actor Network Forward Pass** (`lines 220-229`)
```python
def forward(self, x):  # x: [Batch, 20, 16]
    feat = self.encoder(x)  # → [Batch, 256]  (TemporalFeatureExtractor output)
    logits = self.action_head(feat)  # → [Batch, 17]
    pos_size = self.size_head(feat)  # → [Batch, 1]  (if use_position_sizing=True)
    return logits, pos_size
```

### **Critic Network Forward Pass** (`lines 258-260`)
```python
def forward(self, x):  # x: [Batch, 20, 16]
    return self.dueling(self.encoder(x))  # → [Batch, 17]
```

### **TemporalFeatureExtractor Forward Pass** (`lines 106-133`)
```python
def forward(self, x):  # x: [Batch, T=20, F=16]
    B, T, F = x.shape
    
    # STAGE 1: Multi-scale CNN
    xc = x.permute(0, 2, 1)           # [B, F, T]
    c3 = F.gelu(self.conv3(xc))       # [B, H/2, T]
    c5 = F.gelu(self.conv5(xc))       # [B, H/2, T]
    xc = torch.cat([c3, c5], dim=1)   # [B, H, T]
    xc = self.conv_drop(self.conv_ln(xc))  # [B, H, T]
    xc = xc.permute(0, 2, 1)          # [B, T, H]
    
    # STAGE 2: Bidirectional LSTM
    xl, _ = self.lstm(xc)             # [B, T, H]
    xl = self.lstm_ln(xl)             # [B, T, H]
    
    # STAGE 3: Transformer with CLS token
    cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
    xt = torch.cat([cls, xl], dim=1)        # [B, T+1, H]
    xt = self.pos_enc(xt)                   # [B, T+1, H]  (ADD POSITIONAL ENCODING)
    xt = self.transformer(xt)               # [B, T+1, H]
    
    return xt[:, 0, :]                      # [B, H]  (RETURN CLS TOKEN ONLY)
```

## 🔑 **Key Architectural Insights**

### **Why This Hybrid Architecture?**
1. **CNN Layer**: Detects local, translation-invariant patterns (specific bar formations, candlestick patterns)
2. **BiLSTM Layer**: Captures directional temporal dependencies and medium-term trends
3. **Transformer Layer**: Models long-range dependencies and can attend to structurally similar past market regimes
4. **CLS Token**: Provides a fixed-size, aggregated representation ideal for downstream tasks

### **Why Dueling Networks?**
In trading, many different TP/SL combinations might have similar expected value (especially in ranging markets). The dueling architecture allows the network to:
- Learn accurate state values (V(s)) regardless of which specific action is taken
- Learn relative advantages of each action (A(s,a))
- Generalize better to unseen state-action pairs

### **Why Positional Encoding?**
Transformers are permutation-invariant without positional information. The sinusoidal encoding:
- Provides information about absolute and relative positions
- Allows the model to distinguish between similar patterns occurring at different times
- Enables learning of temporal patterns like momentum vs mean-reversion that depend on historical context

### **Why N-step Returns + PER?**
- **N-step returns** (n=3): Reduce variance in TD targets while maintaining some bias reduction
- **Prioritized Experience Replay**: Focuses learning on surprising experiences (high TD-error)
- **Segment Trees**: Make PER operations O(log N) instead of O(N)

## 📈 **Information Flow Summary**

```
Raw Market Data 
        │
        ▼
[20 time steps × 16 features]  ← Online EMA normalization (in environment)
        │
        ▼
TemporalFeatureExtractor:
        │
        ├─ CNN: Local pattern extraction (3&5 kernels)
        │
        ├─ BiLSTM: Medium-term temporal modeling
        │
        ├─ +CLS token + Positional Encoding
        │
        └─ Transformer: Long-range dependencies → [256-dim features]
        │
        ▼
Shared 256-dim Feature Representation
        │
        ├─→ Actor Network:
        │    │
        │    ├─→ Action Head: [17] TP/SL displacement logits
        │    │
        │    └─→ Size Head: [1] position size fraction [0,1]
        │
        └─→ Critic Network:
             │
             ├─→ Value Stream: [1] state value V(s)
             │
             └─→ Advantage Stream: [17] action advantages A(s,a)
                  │
                  ▼
             Q(s,a) = V(s) + A(s,a) - meanₐ[A(s,a)] → [17] Q-values
```

## 💡 **How to Use This Understanding**

1. **When reading the code**: Follow the data flow dimensions as shown above
2. **When debugging**: Check tensor shapes at each stage match the expectations
3. **When modifying**: Understand which components share weights (feature extractors) vs. which are separate
4. **When extending**: The modular design makes it easy to swap components (e.g., try different positional encodings)

This architecture represents a sophisticated combination of:
- **CNN** for local spatial patterns in time series
- **RNN/LSTM** for sequential/temporal dependencies  
- **Transformer** for global context and attention mechanisms
- **Dueling Networks** for better value function approximation
- **Actor-Critic** framework for policy optimization

The result is a network that can effectively process complex financial time series data while learning both when to trade (discrete actions) and how much to risk (continuous position sizing).