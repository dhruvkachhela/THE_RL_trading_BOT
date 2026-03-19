"""
Feature Engineering Pipeline.

Computes all technical indicators used as state features:
  - Stochastic Oscillator  (sto_osc)
  - MACD                   (macd)
  - ADX                    (adx)
  - On-Balance Volume      (obv)
  - Normalised ATR         (n_atr)
  - Log Return             (log_ret)
  - RSI                    (rsi)
  - Bollinger %B           (bb_pct)
  - Volume Ratio           (volume_ratio)
  - Regime label           (regime)   — optional HMM-style volatility regime

All functions accept a pandas DataFrame with columns: open, high, low, close, volume
and return the same DataFrame with new indicator columns appended.

Usage:
    from src.utils.features import FeaturePipeline
    df = FeaturePipeline().transform(df)
"""

import numpy as np
import pandas as pd
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
#  Low-level indicator functions
# ──────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Returns MACD histogram (MACD line - signal line)."""
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.Series:
    """Returns %K smoothed (D line)."""
    lo  = df["low"].rolling(k_period).min()
    hi  = df["high"].rolling(k_period).max()
    k   = 100.0 * (df["close"] - lo) / (hi - lo + 1e-8)
    return k.rolling(d_period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Returns ADX (trend strength 0–100)."""
    tr   = _true_range(df)
    atr  = tr.rolling(period).mean()

    up   = df["high"].diff()
    down = -df["low"].diff()

    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_di  = 100.0 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / (atr + 1e-8)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr + 1e-8)

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(period).mean()
    return adx


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume, normalised by rolling std."""
    direction = np.sign(df["close"].diff().fillna(0))
    raw_obv   = (direction * df["volume"]).cumsum()
    # Normalise to make it stationary
    return (raw_obv - raw_obv.rolling(20).mean()) / (raw_obv.rolling(20).std() + 1e-8)


def compute_atr_norm(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR normalised by close price (%)."""
    atr = _true_range(df).rolling(period).mean()
    return atr / (df["close"] + 1e-8)


def compute_log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).fillna(0.0)


def compute_bollinger_pct(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger %B: 0 = lower band, 1 = upper band."""
    sma  = _sma(close, window)
    std  = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return (close - lower) / (upper - lower + 1e-8)


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by rolling mean volume."""
    return volume / (volume.rolling(window).mean() + 1e-8)


def compute_volatility_regime(close: pd.Series, window: int = 20, n_regimes: int = 3) -> pd.Series:
    """
    Simple volatility regime label (0=low, 1=medium, 2=high).
    Uses rolling realised volatility percentile rank.
    """
    log_ret = np.log(close / close.shift(1))
    rv      = log_ret.rolling(window).std()
    rank    = rv.rank(pct=True)
    labels  = pd.cut(rank, bins=n_regimes, labels=False).fillna(0).astype(float)
    return labels


# ──────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    Computes all technical indicators and appends them to an OHLCV DataFrame.

    Required input columns: date, open, high, low, close, volume
    Optional input column : newsapi  (sentiment score, passed through unchanged)

    Args:
        include_regime: add volatility regime label column
    """

    REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

    DEFAULT_COLS = [
        "sto_osc", "macd", "adx", "obv", "n_atr",
        "log_ret", "rsi", "bb_pct", "volume_ratio",
    ]

    def __init__(self, include_regime: bool = False):
        self.include_regime = include_regime

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features and return an augmented DataFrame.
        Rows with NaN (warm-up period) are dropped.
        """
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy().sort_values("date").reset_index(drop=True)

        df["sto_osc"]      = compute_stochastic(df)
        df["macd"]         = compute_macd(df["close"])
        df["adx"]          = compute_adx(df)
        df["obv"]          = compute_obv(df)
        df["n_atr"]        = compute_atr_norm(df)
        df["log_ret"]      = compute_log_return(df["close"])
        df["rsi"]          = compute_rsi(df["close"])
        df["bb_pct"]       = compute_bollinger_pct(df["close"])
        df["volume_ratio"] = compute_volume_ratio(df["volume"])

        if self.include_regime:
            df["regime"] = compute_volatility_regime(df["close"])

        # Passthrough: newsapi sentiment (if present)
        # Already in df — no computation needed

        # Drop warm-up NaNs
        feature_cols = self.DEFAULT_COLS[:]
        if self.include_regime:
            feature_cols.append("regime")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        print(f"[Features] {len(df)} rows after warm-up drop | "
              f"Columns: {feature_cols}")
        return df

    def get_feature_names(self, with_newsapi: bool = True) -> List[str]:
        cols = self.DEFAULT_COLS[:]
        if with_newsapi:
            cols.insert(6, "newsapi")   # matches original config order
        if self.include_regime:
            cols.append("regime")
        return cols


# ──────────────────────────────────────────────────────────────
#  CLI helper
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python features.py input.csv output.csv")
        sys.exit(1)

    raw = pd.read_csv(sys.argv[1], parse_dates=["date"])
    out = FeaturePipeline(include_regime=True).transform(raw)
    out.to_csv(sys.argv[2], index=False)
    print(f"Saved → {sys.argv[2]}")
