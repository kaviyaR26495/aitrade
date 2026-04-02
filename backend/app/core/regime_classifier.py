"""Market Regime Classifier & Data Quality Filter.

Implements Stage 0 of the three-stage pipeline:
  1. Multi-signal regime detection (6 regimes)
  2. Data quality scoring per candle
  3. Transition detection

Regime IDs:
  0 = Bullish + Low-Vol  (clean uptrend)
  1 = Bullish + High-Vol (volatile rally)
  2 = Neutral + Low-Vol  (sideways chop)
  3 = Neutral + High-Vol (whipsaw)
  4 = Bearish + Low-Vol  (clean downtrend)
  5 = Bearish + High-Vol (crash/panic)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Trend detection weights ────────────────────────────────────────────
W_SMA_CROSS = 0.40
W_ADX = 0.15
W_RSI = 0.15
W_MACD = 0.15
W_BB = 0.15

# Trend enum mapping
BULLISH, NEUTRAL, BEARISH = "bullish", "neutral", "bearish"
HIGH_VOL, LOW_VOL = "high", "low"

# Regime ID mapping
REGIME_MAP = {
    (BULLISH, LOW_VOL): 0,
    (BULLISH, HIGH_VOL): 1,
    (NEUTRAL, LOW_VOL): 2,
    (NEUTRAL, HIGH_VOL): 3,
    (BEARISH, LOW_VOL): 4,
    (BEARISH, HIGH_VOL): 5,
}


# ── Trend Signals ──────────────────────────────────────────────────────

def _sma_cross_signal(df: pd.DataFrame) -> pd.Series:
    """
    SMA crossover system (40% weight).
    Returns: +1 (bullish), 0 (neutral), -1 (bearish) per row.
    """
    sma50 = df["sma_50"]
    sma200 = df["sma_200"]
    close = df["close"]

    # Relative difference between SMAs
    sma_diff_pct = (sma50 - sma200) / sma200 * 100

    signal = pd.Series(0.0, index=df.index)

    # Bullish: SMA50 > SMA200 AND close > SMA50
    bullish = (sma50 > sma200) & (close > sma50)
    signal[bullish] = 1.0

    # Bearish: SMA50 < SMA200 AND close < SMA50
    bearish = (sma50 < sma200) & (close < sma50)
    signal[bearish] = -1.0

    # Neutral: mixed signals or SMAs within 2%
    within_2pct = sma_diff_pct.abs() < 2.0
    signal[within_2pct] = 0.0

    return signal


def _adx_signal(df: pd.DataFrame) -> pd.Series:
    """
    ADX trend strength (15% weight).
    ADX > 25 → trending (+1 or -1 from adx_pos/neg), ADX < 20 → neutral.
    """
    signal = pd.Series(0.0, index=df.index)

    trending = df["adx"] > 25
    ranging = df["adx"] < 20

    # Direction from DI+/DI-
    di_bullish = df["adx_pos"] > df["adx_neg"]
    signal[trending & di_bullish] = 1.0
    signal[trending & ~di_bullish] = -1.0
    signal[ranging] = 0.0

    return signal


def _rsi_signal(df: pd.DataFrame) -> pd.Series:
    """
    RSI bias (15% weight).
    RSI > 0.60 → bullish, RSI < 0.40 → bearish, else neutral.
    Note: rsi is already 0-1 scale.
    """
    rsi = df["rsi"]
    signal = pd.Series(0.0, index=df.index)
    signal[rsi > 0.60] = 1.0
    signal[rsi < 0.40] = -1.0
    return signal


def _macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    MACD histogram direction (15% weight).
    3 consecutive positive bars → bullish, 3 negative → bearish.
    """
    hist = df["macd_hist"]
    signal = pd.Series(0.0, index=df.index)

    # Rolling 3-bar consistency
    pos_streak = (hist > 0).rolling(3).sum()
    neg_streak = (hist < 0).rolling(3).sum()

    signal[pos_streak >= 3] = 1.0
    signal[neg_streak >= 3] = -1.0

    return signal


def _bb_signal(df: pd.DataFrame) -> pd.Series:
    """
    Bollinger Band directional bias (15% weight).
    Close above BB mid → bullish, below → bearish.
    """
    close = df["close"]
    bb_mid = df["bbl"] if "bbl" in df.columns else df.get("bb_mid")
    if bb_mid is None:
        return pd.Series(0.0, index=df.index)

    signal = pd.Series(0.0, index=df.index)
    signal[close > bb_mid] = 1.0
    signal[close < bb_mid] = -1.0
    return signal


# ── Volatility Signals ─────────────────────────────────────────────────

def _classify_volatility(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    ATR-based volatility classification with BB width confirmation.

    High-Vol: ATR_pct > rolling 90th percentile (over lookback days)
    """
    # ATR as percentage of close
    if "atr" not in df.columns:
        # Calculate ATR(14) if not present
        df = _compute_atr(df, period=14)

    atr_pct = (df["atr"] / df["close"]) * 100

    # Rolling 90th percentile
    rolling_p90 = atr_pct.rolling(lookback, min_periods=50).quantile(0.90)

    # BB width as confirming signal
    if all(c in df.columns for c in ["bbl_h", "bbl_l", "bbl"]):
        bb_width = (df["bbl_h"] - df["bbl_l"]) / df["bbl"]
        bb_p90 = bb_width.rolling(lookback, min_periods=50).quantile(0.90)
        bb_high = bb_width > bb_p90
    else:
        bb_high = pd.Series(False, index=df.index)

    # Historical volatility (20-day rolling std of log returns)
    log_ret = np.log(df["close"]).diff()
    hvol = log_ret.rolling(20).std()
    hvol_p90 = hvol.rolling(lookback, min_periods=50).quantile(0.90)

    # Primary: ATR-based, confirmed by BB width or historical vol
    atr_high = atr_pct > rolling_p90
    hvol_high = hvol > hvol_p90

    # Majority vote (2 of 3 = high vol)
    vol_votes = atr_high.astype(int) + bb_high.astype(int) + hvol_high.astype(int)

    df["_volatility"] = np.where(vol_votes >= 2, HIGH_VOL, LOW_VOL)

    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ATR if not present."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.rolling(period).mean()
    return df


# ── Main Classifier ────────────────────────────────────────────────────

def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime for each candle.

    Input: DataFrame with OHLCV + indicators (sma_50, sma_200, adx, adx_pos,
           adx_neg, rsi, macd_hist, bbl/bb_mid, bbl_h/bb_upper, bbl_l/bb_lower, close).

    Output: DataFrame with additional columns:
       trend, volatility, regime_id, regime_confidence, is_transition
    """
    df = df.copy()

    # Ensure required indicators exist
    required = {"close", "sma_50", "sma_200", "adx", "adx_pos", "adx_neg", "rsi", "macd_hist"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for regime classification: {missing}")

    # ── Trend: weighted majority vote of 5 signals ──
    sma_sig = _sma_cross_signal(df)
    adx_sig = _adx_signal(df)
    rsi_sig = _rsi_signal(df)
    macd_sig = _macd_signal(df)
    bb_sig = _bb_signal(df)

    # Weighted combination (-1 to +1)
    trend_score = (
        W_SMA_CROSS * sma_sig
        + W_ADX * adx_sig
        + W_RSI * rsi_sig
        + W_MACD * macd_sig
        + W_BB * bb_sig
    )

    # Map to trend enum
    df["trend"] = np.where(
        trend_score > 0.2, BULLISH,
        np.where(trend_score < -0.2, BEARISH, NEUTRAL)
    )

    # Confidence: how strongly signals agree (0-1)
    # Take absolute value of weighted score — closer to 1.0 = stronger agreement
    signals = pd.DataFrame({
        "sma": sma_sig, "adx": adx_sig, "rsi": rsi_sig, "macd": macd_sig, "bb": bb_sig
    })
    # Count how many agree with the majority direction
    bullish_count = (signals > 0).sum(axis=1)
    bearish_count = (signals < 0).sum(axis=1)
    max_agreement = pd.concat([bullish_count, bearish_count], axis=1).max(axis=1)
    df["regime_confidence"] = (max_agreement / 5.0).clip(0.2, 1.0)

    # ── Volatility: ATR + BB width + hist vol ──
    df = _classify_volatility(df)
    df["volatility"] = df["_volatility"]
    df = df.drop(columns=["_volatility"])

    # ── Combined regime ID ──
    df["regime_id"] = df.apply(
        lambda row: REGIME_MAP.get((row["trend"], row["volatility"]), 2), axis=1
    )

    # ── Transition detection ──
    df["is_transition"] = False
    regime_changed = df["regime_id"] != df["regime_id"].shift(1)
    # Mark ±2 candles around transitions
    for offset in range(-2, 3):
        shifted = regime_changed.shift(offset)
        if shifted is not None:
            df["is_transition"] = df["is_transition"] | shifted.fillna(False)

    return df


# ── Data Quality Scoring ───────────────────────────────────────────────

def compute_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality score per candle (0-1). Deductions for:
    - Circuit hits (daily move > 10%)
    - Low volume days (< 20th pctile of 50-day rolling)
    - Data gaps (missing previous trading day)
    - Regime transitions
    - Extreme outliers (> 3σ from 50-day mean)
    """
    df = df.copy()
    score = pd.Series(1.0, index=df.index)

    # ── Circuit hit: daily move > 10% ──
    daily_return = df["close"].pct_change().abs()
    circuit_hit = daily_return > 0.10
    score[circuit_hit] -= 0.3

    # ── Volume anomaly: volume < 20th percentile of 50-day rolling ──
    vol_p20 = df["volume"].rolling(50, min_periods=10).quantile(0.20)
    low_volume = df["volume"] < vol_p20
    score[low_volume] -= 0.3

    # ── Data gap: check if date gap > 3 calendar days (skipping weekends) ──
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        day_gaps = dt.diff().dt.days
        has_gap = day_gaps > 4  # > 4 to skip normal weekends
        score[has_gap] -= 0.2

    # ── Regime transition ──
    if "is_transition" in df.columns:
        score[df["is_transition"]] -= 0.2

    # ── Extreme outlier: daily return > 3σ from 50-day rolling mean ──
    rolling_mean = daily_return.rolling(50, min_periods=10).mean()
    rolling_std = daily_return.rolling(50, min_periods=10).std()
    outlier = daily_return > (rolling_mean + 3 * rolling_std)
    score[outlier] -= 0.1

    df["quality_score"] = score.clip(0.0, 1.0)
    return df


# ── Convenience Functions ──────────────────────────────────────────────

def classify_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: classify regimes + compute quality scores."""
    df = classify_regimes(df)
    df = compute_quality_scores(df)
    return df


def get_quality_filtered_data(
    df: pd.DataFrame,
    min_quality: float = 0.8,
    regime_ids: list[int] | None = None,
    exclude_transitions: bool = True,
) -> pd.DataFrame:
    """
    Filter data for RL training:
    - quality_score >= min_quality
    - Optionally filter by regime IDs
    - Optionally exclude transition periods
    """
    mask = df["quality_score"] >= min_quality

    if exclude_transitions and "is_transition" in df.columns:
        mask &= ~df["is_transition"]

    if regime_ids is not None:
        mask &= df["regime_id"].isin(regime_ids)

    return df[mask].reset_index(drop=True)


def get_regime_segments(
    df: pd.DataFrame,
    min_length: int = 20,
) -> list[tuple[int, pd.DataFrame]]:
    """
    Split data into contiguous regime segments.
    Returns list of (regime_id, segment_df) tuples.
    Only includes segments with at least min_length rows.
    """
    segments = []
    if "regime_id" not in df.columns:
        return segments

    # Identify contiguous blocks
    regime_changes = df["regime_id"] != df["regime_id"].shift(1)
    block_id = regime_changes.cumsum()

    for _, group in df.groupby(block_id):
        if len(group) >= min_length:
            rid = group["regime_id"].iloc[0]
            segments.append((rid, group.reset_index(drop=True)))

    return segments


def regime_summary(df: pd.DataFrame) -> dict:
    """Summary statistics of regime classification."""
    if "regime_id" not in df.columns:
        return {}

    regime_breakdown = {}
    for rid in range(6):
        subset = df[df["regime_id"] == rid]
        regime_breakdown[rid] = {
            "count": len(subset),
            "pct": len(subset) / len(df) * 100 if len(df) > 0 else 0,
            "avg_quality": float(subset["quality_score"].mean()) if "quality_score" in subset.columns and len(subset) > 0 else None,
            "avg_confidence": float(subset["regime_confidence"].mean()) if "regime_confidence" in subset.columns and len(subset) > 0 else None,
        }

    quality_tiers = {}
    if "quality_score" in df.columns:
        quality_tiers = {
            "high": int((df["quality_score"] >= 0.8).sum()),
            "medium": int(((df["quality_score"] >= 0.5) & (df["quality_score"] < 0.8)).sum()),
            "low": int((df["quality_score"] < 0.5).sum()),
        }

    avg_quality = float(df["quality_score"].mean()) if "quality_score" in df.columns else None
    
    total_transitions = int(df["is_transition"].sum()) if "is_transition" in df.columns else 0

    return {
        "total_candles": len(df),
        "total_transitions": total_transitions,
        "avg_quality": avg_quality,
        "quality_tiers": quality_tiers,
        "regime_breakdown": regime_breakdown,
    }
