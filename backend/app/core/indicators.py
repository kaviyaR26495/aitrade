"""Technical indicator calculations.

Ported from pytrade's df_creator.py — computes all indicators on an OHLCV
DataFrame and returns it with new columns appended.

Required libraries: ta, pandas_ta, numpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


# Which indicator groups to compute (matches pytrade obs_col keys)
ALL_INDICATOR_GROUPS = [
    "TGRB", "rsi", "srsi", "kama", "vwkama",
    "obv", "bbl", "macd", "adx", "sma", "atr",
]

WARMUP_ROWS = 30  # drop first N rows after indicator calc


def calc_vw_kama(df: pd.DataFrame, length: int = 10) -> pd.Series:
    """Volume-Weighted KAMA (from pytrade common.py)."""
    vw_price = (df["close"] * df["volume"]) / df["volume"].rolling(window=length).sum()
    temp_kama = ta.momentum.kama(df["close"], window=length, fillna=True)

    sc = (temp_kama.diff() / temp_kama.shift(1)).abs()
    vw_kama = vw_price.copy()

    for i in range(length, len(df)):
        idx = df.index[i]
        idx_prev = df.index[i - 1]
        vw_kama.iloc[i] = vw_kama.iloc[i - 1] + sc.iloc[i] * (vw_price.iloc[i] - vw_kama.iloc[i - 1])

    return vw_kama


def compute_tgrb(df: pd.DataFrame) -> pd.DataFrame:
    """Candle body pattern: tgrb_top, tgrb_green, tgrb_red, tgrb_bottom."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df["tgrb_top"] = np.where(
        o >= c,
        (h - o) / o,
        (h - c) / c,
    )
    df["tgrb_green"] = np.where(o >= c, 0, (c - o) / o)
    df["tgrb_red"] = np.where(o >= c, (o - c) / c, 0)
    df["tgrb_bottom"] = np.where(
        o >= c,
        (c - l) / l,
        (o - l) / l,
    )
    return df


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """RSI (0-1 scale)."""
    df["rsi"] = ta.momentum.rsi(df["close"], fillna=True) / 100
    return df


def compute_srsi(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic RSI."""
    df["srsi"] = ta.momentum.stochrsi(df["close"], window=14, fillna=True)
    return df


def compute_kama(df: pd.DataFrame) -> pd.DataFrame:
    """Kaufman Adaptive Moving Average."""
    df["kama"] = ta.momentum.kama(df["close"], window=10, fillna=True)
    return df


def compute_vwkama(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-Weighted KAMA."""
    df["vwkama"] = calc_vw_kama(df, 10)
    return df


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume (using ta library)."""
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv"] = df["obv"].fillna(0)
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD, Signal, Histogram."""
    df["macd"] = ta.trend.macd(df["close"], fillna=True)
    df["macd_signal"] = ta.trend.macd_signal(df["close"], fillna=True)
    df["macd_hist"] = ta.trend.macd_diff(df["close"], fillna=True)
    return df


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    """SMA at multiple windows plus EMA-20."""
    for w in [5, 12, 24, 50, 100, 200]:
        df[f"sma_{w}"] = ta.trend.sma_indicator(df["close"], window=w, fillna=True)
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20, fillna=True)
    return df


def compute_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands (20, 2) — bb_upper, bb_lower, bb_mid."""
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    return df


def compute_adx(df: pd.DataFrame) -> pd.DataFrame:
    """ADX with positive/negative directional indicators."""
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
    df["adx_neg"] = ta.trend.adx_neg(df["high"], df["low"], df["close"], window=14, fillna=True)
    df["adx_pos"] = ta.trend.adx_pos(df["high"], df["low"], df["close"], window=14, fillna=True)
    return df


def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range (14)."""
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
    return df


# Mapping from group names to compute functions
INDICATOR_FUNCS = {
    "TGRB": compute_tgrb,
    "rsi": compute_rsi,
    "srsi": compute_srsi,
    "kama": compute_kama,
    "vwkama": compute_vwkama,
    "obv": compute_obv,
    "macd": compute_macd,
    "sma": compute_sma,
    "bbl": compute_bollinger,
    "adx": compute_adx,
    "atr": compute_atr,
}


def compute_all_indicators(
    df: pd.DataFrame,
    groups: list[str] | None = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """
    Compute all (or selected) indicator groups on an OHLCV DataFrame.

    Expected input columns (lowercase): date, open, high, low, close, volume
    """
    df = df.copy()

    # Ensure required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Fill NaN with forward-fill then 1 (pytrade pattern)
    df = df.ffill().fillna(1)

    # Add date features
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df["dow"] = dt.dt.dayofweek / 10
        df["month"] = dt.dt.month / 100
        df["day"] = dt.dt.day / 100

    # Copy raw OHLCV to n-prefixed columns (for normalization later)
    df["n_open"] = df["open"]
    df["n_high"] = df["high"]
    df["n_close"] = df["close"]
    df["n_low"] = df["low"]
    df["n_volume"] = df["volume"]

    # Compute indicator groups
    selected = groups or ALL_INDICATOR_GROUPS
    for group in selected:
        func = INDICATOR_FUNCS.get(group)
        if func:
            df = func(df)

    # Drop warmup rows
    if drop_warmup and len(df) > WARMUP_ROWS:
        df = df.iloc[WARMUP_ROWS:].reset_index(drop=True)

    return df


def get_indicator_columns(groups: list[str] | None = None) -> list[str]:
    """Return list of indicator column names for given groups."""
    cols = []
    selected = groups or ALL_INDICATOR_GROUPS

    if "TGRB" in selected:
        cols.extend(["tgrb_top", "tgrb_green", "tgrb_red", "tgrb_bottom"])
    if "rsi" in selected:
        cols.append("rsi")
    if "srsi" in selected:
        cols.append("srsi")
    if "kama" in selected:
        cols.append("kama")
    if "vwkama" in selected:
        cols.append("vwkama")
    if "obv" in selected:
        cols.append("obv")
    if "macd" in selected:
        cols.extend(["macd", "macd_signal", "macd_hist"])
    if "sma" in selected:
        cols.extend([f"sma_{w}" for w in [5, 12, 24, 50, 100, 200]] + ["ema_20"])
    if "bbl" in selected:
        cols.extend(["bb_upper", "bb_lower", "bb_mid"])
    if "adx" in selected:
        cols.extend(["adx", "adx_neg", "adx_pos"])
    if "atr" in selected:
        cols.append("atr")

    return cols
