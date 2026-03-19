"""
Advanced Neural Network Architectures for RL Trading Agent.

Improvements over baseline:
  - Hybrid CNN-LSTM + Transformer (multi-head self-attention)
  - Dueling network heads (V + A → Q)
  - Layer normalisation throughout
  - Dropout for regularisation
  - Positional encoding for transformer
  - Separate position-sizing head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────
#  Positional Encoding
# ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────
#  Core Feature Extractor  (CNN → LSTM → Transformer)
# ──────────────────────────────────────────────────────────────

class TemporalFeatureExtractor(nn.Module):
    """
    Three-stage temporal encoder:
      1. Conv1D — local pattern extraction (kernel-3 and kernel-5 in parallel)
      2. Bidirectional LSTM — medium-range temporal context
      3. Transformer encoder — global self-attention over the sequence
    """

    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Stage 1: Multi-scale CNN ────────────────────────────
        self.conv3 = nn.Conv1d(state_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(state_dim, hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_ln = nn.LayerNorm([hidden_dim, sequence_length])
        self.conv_drop = nn.Dropout(dropout)

        # ── Stage 2: Bidirectional LSTM ─────────────────────────
        lstm_input = hidden_dim
        self.lstm = nn.LSTM(
            lstm_input,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_ln = nn.LayerNorm(hidden_dim)

        # ── Stage 3: Transformer encoder ────────────────────────
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=sequence_length + 1, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # ── CLS token (learnable) ────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, features]
        Returns:
            features: [batch, hidden_dim]
        """
        B, T, F = x.shape

        # Stage 1 — CNN
        xc = x.permute(0, 2, 1)                     # [B, F, T]
        c3 = F.gelu(self.conv3(xc))
        c5 = F.gelu(self.conv5(xc))
        xc = torch.cat([c3, c5], dim=1)             # [B, hidden, T]
        xc = self.conv_drop(self.conv_ln(xc))
        xc = xc.permute(0, 2, 1)                     # [B, T, hidden]

        # Stage 2 — LSTM
        xl, _ = self.lstm(xc)                        # [B, T, hidden]
        xl = self.lstm_ln(xl)

        # Stage 3 — Transformer with CLS
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, hidden]
        xt = torch.cat([cls, xl], dim=1)             # [B, T+1, hidden]
        xt = self.pos_enc(xt)
        xt = self.transformer(xt)                    # [B, T+1, hidden]

        return xt[:, 0, :]                           # CLS token → [B, hidden]


# ──────────────────────────────────────────────────────────────
#  Dueling Head  (V + Advantage → Q)
# ──────────────────────────────────────────────────────────────

class DuelingHead(nn.Module):
    """
    Dueling network decomposition:
        Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    Separates learning 'how good is this state' from
    'how much better is each action'.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value_stream(x)                    # [B, 1]
        a = self.adv_stream(x)                      # [B, action_dim]
        return v + a - a.mean(dim=1, keepdim=True)  # Q values


# ──────────────────────────────────────────────────────────────
#  Actor Network
# ──────────────────────────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    Policy network: state → action logits (+ optional position size).

    action_logits → Categorical distribution for TP/SL displacement
    pos_size      → sigmoid output in [0, 1] for position sizing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        use_position_sizing: bool = True,
    ):
        super().__init__()
        self.use_position_sizing = use_position_sizing

        self.encoder = TemporalFeatureExtractor(
            state_dim, sequence_length, hidden_dim, n_heads, n_transformer_layers, dropout
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

        if use_position_sizing:
            self.size_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:   [B, action_dim]
            pos_size: [B, 1]  (fraction of capital to risk)
        """
        feat = self.encoder(x)
        logits = self.action_head(feat)
        pos_size = self.size_head(feat) if self.use_position_sizing else torch.ones(x.size(0), 1, device=x.device)
        return logits, pos_size


# ──────────────────────────────────────────────────────────────
#  Critic Network (Dueling)
# ──────────────────────────────────────────────────────────────

class CriticNetwork(nn.Module):
    """
    Q-value network with dueling heads.
    Uses the same TemporalFeatureExtractor backbone.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TemporalFeatureExtractor(
            state_dim, sequence_length, hidden_dim, n_heads, n_transformer_layers, dropout
        )
        self.dueling = DuelingHead(hidden_dim, action_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values: [B, action_dim]"""
        return self.dueling(self.encoder(x))
