"""Data normalization pipeline.

Ported from pytrade's df_creator.py normalization logic.

Three normalization strategies applied per column category:
1. Log-return z-score  — price-like columns (nOpen, nHigh, nClose, nLow, kama, vwkama, srsi, sma_5/12/24/100/200)
2. MinMax scaling       — volume (n_volume), obv (can be negative — not safe for log)
3. Pct-change from close — bb_upper, bb_lower, bb_mid (no longer weekly_sma_50)
4. Excluded (kept as-is) — rsi, TGRB, macd*, adx*, dow, month, day, regime features,
                           and all new stationary ML features (already bounded)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── ML feature lists ────────────────────────────────────────────────────────
# These are the ONLY columns passed to ML models (RL env, KNN, LSTM).
# All are stationary (bounded ratios, percentages, or 0-1 scaled).
ML_DAILY_FEATURES: list[str] = [
    # Candle body structure (already normalised ratios)
    "tgrb_top", "tgrb_green", "tgrb_red", "tgrb_bottom",
    # Stationary trend distance
    "dist_sma_50", "dist_sma_200",
    # Short / medium / long momentum (% returns)
    "roc_1", "roc_5", "roc_20",
    # Volatility regime
    "atr_pct", "realized_vol_20",
    # Bollinger Band position & width (0-1 bounded)
    "bb_pctb", "bb_width",
    # Institutional flow & liquidity
    "cmf_20", "dist_vwap_5",
    # Bounded oscillators / normalised signals
    "rsi", "macd_hist_norm",
    # Trend strength (0-1 normalised via /100)
    "adx_norm", "adx_pos_norm", "adx_neg_norm",
    # ── Market context (computed by market_context.py) ────────────────
    # Institutional flow — pre-normalised z-scores, pass through
    "fii_net_norm", "dii_net_norm",
    # Sector breadth — fraction of sector stocks above SMA-50 (0-1)
    "sector_breadth_pct",
    # Relative strength vs NIFTY 50 over 5 days (clipped ±20)
    "rs_5d",
    # Sector-average 5-day ROC (smoothed)
    "sector_roc_avg",
    # Binary: 1.0 if broadmarket index is above its 200-SMA, else 0.0
    "market_above_sma200",
    # ── Cyclical date encoding (sin/cos, bounded [-1, +1]) ────────────
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "day_sin", "day_cos",
]

ML_WEEKLY_FEATURES: list[str] = [
    "weekly_rsi",
    "weekly_macd_hist_norm",
    "weekly_roc_1",
    "weekly_adx_norm", "weekly_adx_pos_norm", "weekly_adx_neg_norm",
]

# Combined list — used by get_feature_columns() and as the final model input filter.
ML_FEATURES: list[str] = ML_DAILY_FEATURES + ML_WEEKLY_FEATURES  # 38 total


# Columns excluded from normalization (already scaled or categorical)
EXCLUDE_COLS = {
    "open", "high", "low", "close", "volume", "adj_close",
    "date", "stock_id",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_neg", "adx_pos",
    "rsi",
    "tgrb_top", "tgrb_green", "tgrb_red", "tgrb_bottom",
    # Cyclical date features — already bounded [-1, +1], pass through
    "dow_sin", "dow_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    # Regime features are pre-scaled (0/1 binary or 0-1 float); skip log-return treatment
    "regime_trend_bullish", "regime_trend_bearish", "regime_trend_neutral",
    "regime_vol_high", "regime_confidence",
    "trend", "volatility", "regime_id", "quality_score", "is_transition",
    # Market context features — pre-normalized, pass through as-is
    "fii_net_norm", "dii_net_norm", "sector_breadth_pct",
    "rs_5d", "sector_roc_avg", "market_above_sma200",
    # ── Weekly stationary ML features (all bounded, pass through) ────────
    "weekly_rsi",
    "weekly_macd_hist_norm",
    "weekly_roc_1",
    "weekly_adx_norm", "weekly_adx_pos_norm", "weekly_adx_neg_norm",
    # ── New daily stationary ML features (bounded ratios, pass through) ──
    "dist_sma_50", "dist_sma_200",
    "roc_1", "roc_5", "roc_20",
    "atr_pct", "realized_vol_20",
    "bb_pctb", "bb_width",
    "cmf_20", "dist_vwap_5",
    "macd_hist_norm",
    "adx_norm", "adx_pos_norm", "adx_neg_norm",
}

# MinMax-scaled columns (obv can be negative → cannot use log-return)
MINMAX_COLS = {"n_volume", "obv"}

# Percentage-change-from-close columns
PCTCHG_COLS = {"bb_upper", "bb_lower", "bb_mid"}

# Reference window for z-score normalization
ZSCORE_WINDOW = 100


def normalize_dataframe(
    df: pd.DataFrame,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Apply the full pytrade normalization pipeline.

    1. Identify log-return columns (all numeric cols not in exclude/minmax/pctchg)
    2. Replace zeros with ffill
    3. Log-return z-score normalize
    4. MinMax scale volume
    5. Pct-change from close for bollinger/sma_50

    Returns normalized DataFrame.
    """
    df = df.copy()

    # --- Step 1: Identify log-return columns ---
    all_exclude = EXCLUDE_COLS | MINMAX_COLS | PCTCHG_COLS
    log_return_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in all_exclude
    ]

    # --- Step 2: Ensure no zeros/negatives before log (ffill + bfill + clip) ---
    # Do NOT drop rows — that compounds across columns and empties the DataFrame.
    for col in log_return_cols:
        s = df[col].replace({0: np.nan}).ffill().bfill()
        # After fill, clip to a small positive value so log(x) is always finite
        df[col] = s.clip(lower=1e-8).fillna(1e-8)

    if df.empty:
        return df

    # --- Step 3: Log-return z-score normalization (causal rolling window) ---
    # Each row is normalised using only the preceding `zscore_window` rows,
    # so no future statistics leak into earlier observations.
    for col in log_return_cols:
        log_ret = np.log(df[col]).diff()
        rolling_mean = log_ret.rolling(zscore_window, min_periods=2).mean()
        rolling_std = (
            log_ret.rolling(zscore_window, min_periods=2).std()
            .fillna(1.0)
            .replace(0, 1.0)
        )
        df[col] = (log_ret - rolling_mean) / rolling_std

    # Drop first row (NaN from diff)
    df = df.iloc[1:].reset_index(drop=True)

    # --- Step 4: Rolling min-max scale volume and obv (causal, no lookahead) ---
    # Using a rolling window avoids fitting on the full dataset which would
    # expose future max/min values to earlier timesteps.
    for col in MINMAX_COLS:
        if col in df.columns and len(df) > 0:
            vals = df[col].fillna(0)
            rolling_min = vals.rolling(zscore_window, min_periods=1).min()
            rolling_max = vals.rolling(zscore_window, min_periods=1).max()
            denom = (rolling_max - rolling_min).replace(0, 1.0)
            df[col] = ((vals - rolling_min) / denom).clip(0.0, 1.0).astype(np.float32)

    # --- Step 5: Pct-change from close ---
    if "close" in df.columns:
        for col in PCTCHG_COLS:
            if col in df.columns:
                df[col] = ((df["close"] - df[col]) / df["close"]) * 100

    # --- Step 6: Final cleanup ---
    df = df.fillna(0)

    # Convert float columns to float32
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df


def prepare_model_input(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    seq_len: int = 15,
) -> np.ndarray:
    """
    Convert normalized DataFrame to numpy array suitable for model input.

    If feature_cols is None, uses all numeric non-raw columns.

    Returns shape (n_samples, seq_len, n_features) for sequential models,
    or (n_samples, n_features) for flat models.
    """
    if feature_cols is None:
        exclude = {"open", "high", "low", "close", "volume", "adj_close", "date", "stock_id"}
        feature_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]

    data = df[feature_cols].values.astype(np.float32)

    if seq_len > 1:
        # Create sliding windows
        sequences = []
        for i in range(seq_len, len(data) + 1):
            sequences.append(data[i - seq_len: i])
        return np.array(sequences, dtype=np.float32)
    else:
        return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ML feature columns present in df.

    Returns only columns from ML_FEATURES (the 26 stationary, bounded features)
    that actually exist in df.  This acts as the filter between the full
    indicator DataFrame (which retains legacy raw columns for UI charts and
    regime classification) and the model observation space.
    """
    return [f for f in ML_FEATURES if f in df.columns]
