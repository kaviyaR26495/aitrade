"""Technical indicator calculations.

Ported from pytrade's df_creator.py — computes all indicators on an OHLCV
DataFrame and returns it with new columns appended.

Required libraries: ta, numpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


# Which indicator groups to compute (matches pytrade obs_col keys).
# Legacy groups are kept for UI chart overlays (sma_*, bb_*, etc.).
# New stationary ML groups (dist_sma … adx_norm) are appended additively.
ALL_INDICATOR_GROUPS = [
    "TGRB", "rsi", "srsi", "kama", "vwkama",
    "obv", "bbl", "macd", "adx", "sma", "atr",
    # ── Stationary ML features (additive, order matters for dependencies) ──
    "dist_sma",       # needs sma_50 / sma_200 (computed internally)
    "roc",            # self-contained
    "atr_pct",        # depends on: atr
    "bb_norm",        # self-contained
    "macd_hist_norm", # depends on: macd
    "cmf",            # self-contained
    "vwap_dist",      # self-contained
    "adx_norm",       # depends on: adx (raw 0-100 kept for regime_classifier)
]

# Stationary weekly indicators.  Non-stationary weekly_sma_50 is dropped;
# macd_hist_norm and adx_norm replace their raw equivalents.
WEEKLY_INDICATOR_GROUPS = ["rsi", "macd", "roc", "adx", "macd_hist_norm", "adx_norm"]

# Final column names present in the daily DataFrame after the weekly merge.
WEEKLY_INDICATOR_COLS = [
    "weekly_rsi",
    "weekly_macd_hist_norm",
    "weekly_roc_1",
    "weekly_adx_norm", "weekly_adx_pos_norm", "weekly_adx_neg_norm",
]

# SMA_200 requires 200-bar warmup; drop those rows from the live fallback path.
WARMUP_ROWS = 200


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
    """ADX with positive/negative directional indicators (raw 0-100 scale).
    Kept as-is so regime_classifier thresholds (adx > 25) continue to work.
    """
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
    df["adx_neg"] = ta.trend.adx_neg(df["high"], df["low"], df["close"], window=14, fillna=True)
    df["adx_pos"] = ta.trend.adx_pos(df["high"], df["low"], df["close"], window=14, fillna=True)
    return df


def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range (14). Kept for internal + regime use."""
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
    return df


# ── New stationary ML-ready compute functions ─────────────────────────────

def compute_dist_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage distance from SMA-50 and SMA-200 (stationary trend features)."""
    sma50 = ta.trend.sma_indicator(df["close"], window=50, fillna=True)
    sma200 = ta.trend.sma_indicator(df["close"], window=200, fillna=True)
    df["dist_sma_50"] = ((df["close"] - sma50) / sma50.replace(0, np.nan)).fillna(0)
    df["dist_sma_200"] = ((df["close"] - sma200) / sma200.replace(0, np.nan)).fillna(0)
    return df


def compute_roc(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of Change for 1, 5, and 20 bars (daily/weekly/monthly momentum)."""
    for window in [1, 5, 20]:
        df[f"roc_{window}"] = (
            ta.momentum.ROCIndicator(close=df["close"], window=window, fillna=True).roc()
        )
    return df


def compute_atr_pct(df: pd.DataFrame) -> pd.DataFrame:
    """ATR as % of close + annualised realised volatility (20-day).
    Depends on: atr group must run first.
    """
    if "atr" not in df.columns:
        df = compute_atr(df)
    close = df["close"].replace(0, np.nan)
    df["atr_pct"] = (df["atr"] / close).fillna(0)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_20"] = (
        log_ret.rolling(window=20, min_periods=2).std() * np.sqrt(252)
    ).fillna(0)
    return df


def compute_bb_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Band %B (0-1 price position) and Band Width (expansion/contraction)."""
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
    df["bb_pctb"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()
    return df


def compute_macd_hist_norm(df: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram normalised by close price (stationary MACD signal).
    Depends on: macd group must run first.
    """
    if "macd_hist" not in df.columns:
        df = compute_macd(df)
    close = df["close"].replace(0, np.nan)
    df["macd_hist_norm"] = (df["macd_hist"] / close).fillna(0)
    return df


def compute_cmf(df: pd.DataFrame) -> pd.DataFrame:
    """Chaikin Money Flow (20-day) — institutional accumulation/distribution."""
    df["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=20,
        fillna=True,
    ).chaikin_money_flow()
    return df


def compute_vwap_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage distance from rolling 5-day VWAP."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tvol = typical * df["volume"]
    rolling_vwap = (
        tvol.rolling(window=5, min_periods=1).sum()
        / df["volume"].rolling(window=5, min_periods=1).sum()
    )
    rolling_vwap = rolling_vwap.replace(0, np.nan)
    df["dist_vwap_5"] = ((df["close"] - rolling_vwap) / rolling_vwap).fillna(0)
    return df


def compute_adx_norm(df: pd.DataFrame) -> pd.DataFrame:
    """ADX normalised to 0-1 for ML models.
    Regime classifier keeps separate raw adx > 25 thresholds — do not modify
    the raw adx/adx_pos/adx_neg columns here.
    Depends on: adx group must run first.
    """
    if "adx" not in df.columns:
        df = compute_adx(df)
    df["adx_norm"] = df["adx"] / 100.0
    df["adx_pos_norm"] = df["adx_pos"] / 100.0
    df["adx_neg_norm"] = df["adx_neg"] / 100.0
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
    # New stationary ML features
    "dist_sma": compute_dist_sma,
    "roc": compute_roc,
    "atr_pct": compute_atr_pct,
    "bb_norm": compute_bb_normalized,
    "macd_hist_norm": compute_macd_hist_norm,
    "cmf": compute_cmf,
    "vwap_dist": compute_vwap_dist,
    "adx_norm": compute_adx_norm,
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

    # Forward-fill mid-series gaps (e.g. holidays, stale ticks).
    # For leading NaN on short-history / recently-listed stocks, back-fill
    # OHLC from the first real bar — avoids the artificial close=1 artefact
    # that inflates ROC and distance-SMA features during the IPO era.
    # Volume is zeroed rather than back-filled (no traded volume is correct).
    df = df.ffill()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].bfill()
    df["volume"] = df["volume"].fillna(0)

    # Add date features — sin/cos cyclical encoding so the network sees
    # Monday and Friday as adjacent in circular space (not 0.0 vs 0.4),
    # and January/December as adjacent (not 0.01 vs 0.12).
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        dow   = dt.dt.dayofweek        # 0=Mon … 4=Fri
        month = dt.dt.month            # 1–12
        day   = dt.dt.day              # 1–31
        df["dow_sin"]   = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"]   = np.cos(2 * np.pi * dow / 5)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
        df["day_sin"]   = np.sin(2 * np.pi * (day - 1) / 31)
        df["day_cos"]   = np.cos(2 * np.pi * (day - 1) / 31)

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


def compute_weekly_indicators(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a high-signal, stationary subset of indicators on a *weekly* OHLCV
    DataFrame and rename every output column with a ``weekly_`` prefix.

    Returns a DataFrame with columns:
        date, weekly_rsi, weekly_macd_hist_norm, weekly_roc_1,
        weekly_adx_norm, weekly_adx_pos_norm, weekly_adx_neg_norm.

    Only ``date`` and the indicator columns are returned; raw OHLCV columns are
    dropped so the result is safe to merge into a daily DataFrame without
    column collisions.  The ``drop_warmup`` flag is set to False so the caller
    retains the full weekly series for forward-filling purposes.
    """
    if weekly_df.empty:
        return pd.DataFrame(columns=["date"] + WEEKLY_INDICATOR_COLS)

    # Compute on a clean copy with warm-up rows kept (we need the full series
    # so that forward-filling into daily rows doesn't lose the early weeks).
    wdf = compute_all_indicators(
        weekly_df.copy(),
        groups=WEEKLY_INDICATOR_GROUPS,
        drop_warmup=False,
    )

    # Build rename map: stationary columns only
    daily_to_weekly: dict[str, str] = {
        col: f"weekly_{col}"
        for col in [
            "rsi",
            "macd_hist_norm",
            "roc_1",
            "adx_norm", "adx_pos_norm", "adx_neg_norm",
        ]
        if col in wdf.columns
    }

    # Rename and keep only date + weekly indicator columns
    wdf = wdf.rename(columns=daily_to_weekly)
    keep = ["date"] + [c for c in WEEKLY_INDICATOR_COLS if c in wdf.columns]
    return wdf[keep].reset_index(drop=True)


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
    # New stationary ML features
    if "dist_sma" in selected:
        cols.extend(["dist_sma_50", "dist_sma_200"])
    if "roc" in selected:
        cols.extend(["roc_1", "roc_5", "roc_20"])
    if "atr_pct" in selected:
        cols.extend(["atr_pct", "realized_vol_20"])
    if "bb_norm" in selected:
        cols.extend(["bb_pctb", "bb_width"])
    if "macd_hist_norm" in selected:
        cols.append("macd_hist_norm")
    if "cmf" in selected:
        cols.append("cmf_20")
    if "vwap_dist" in selected:
        cols.append("dist_vwap_5")
    if "adx_norm" in selected:
        cols.extend(["adx_norm", "adx_pos_norm", "adx_neg_norm"])

    return cols
