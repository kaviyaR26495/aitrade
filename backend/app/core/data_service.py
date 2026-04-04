"""Data service — orchestrates data pipeline, indicator computation, and DB storage.

Sits between the API routes and the lower-level modules (data_pipeline, indicators, normalizer).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.indicators import compute_all_indicators, get_indicator_columns
from app.core.normalizer import normalize_dataframe, get_feature_columns, prepare_model_input
from app.core.data_pipeline import ohlcv_to_dataframe, sync_stock_ohlcv, sync_all_stocks
from app.db import crud

logger = logging.getLogger(__name__)

# Mapping from indicator DataFrame columns to DB model columns
INDICATOR_COL_MAP = {
    "sma_5": "sma_5",
    "sma_12": "sma_12",
    "sma_24": "sma_24",
    "sma_50": "sma_50",
    "sma_100": "sma_100",
    "sma_200": "sma_200",
    "rsi": "rsi",
    "srsi": "srsi",
    "macd": "macd",
    "macd_signal": "macd_signal",
    "macd_hist": "macd_hist",
    "adx": "adx",
    "adx_pos": "adx_pos",
    "adx_neg": "adx_neg",
    "kama": "kama",
    "vwkama": "vwkama",
    "obv": "obv",
    "bbl_h": "bb_upper",
    "bbl_l": "bb_lower",
    "bbl": "bb_mid",
    "Top": "tgrb_top",
    "Green": "tgrb_green",
    "Red": "tgrb_red",
    "Bottom": "tgrb_bottom",
}


async def compute_and_store_indicators(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
    groups: list[str] | None = None,
) -> int:
    """
    Compute technical indicators for a stock and store in DB.

    1. Fetch OHLCV from DB
    2. Convert to DataFrame
    3. Compute indicators
    4. Map columns to DB schema
    5. Bulk upsert to stock_indicators
    """
    ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval)
    if not ohlcv_rows:
        logger.warning("No OHLCV data for stock_id=%d interval=%s", stock_id, interval)
        return 0

    df = ohlcv_to_dataframe(list(ohlcv_rows))
    if len(df) < 50:
        logger.warning("Insufficient data for indicators (got %d rows)", len(df))
        return 0

    # Compute indicators
    df = compute_all_indicators(df, groups=groups)

    # Build DB rows
    rows = []
    for _, row in df.iterrows():
        db_row = {
            "stock_id": stock_id,
            "date": row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"],
            "interval": interval,
        }
        for src_col, db_col in INDICATOR_COL_MAP.items():
            if src_col in row.index:
                val = row[src_col]
                db_row[db_col] = float(val) if pd.notna(val) else None
        rows.append(db_row)

    count = await crud.bulk_upsert_indicators(db, rows)
    logger.info("Upserted %d indicator rows for stock_id=%d", count, stock_id)
    return count


async def get_stock_features(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
    start_date: date | None = None,
    end_date: date | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Get full feature matrix for a stock (OHLCV + indicators + regimes).

    Returns a DataFrame with all features, optionally normalized.
    """
    ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval, start_date, end_date)
    if not ohlcv_rows:
        return pd.DataFrame()

    df = ohlcv_to_dataframe(list(ohlcv_rows))
    if len(df) < 50:
        return pd.DataFrame()

    df = compute_all_indicators(df)

    # Join regime features if available
    regime_rows = await crud.get_regimes(db, stock_id, interval, start_date, end_date)
    if regime_rows:
        regime_data = []
        for r in regime_rows:
            trend_val = r.trend.value if hasattr(r.trend, 'value') else str(r.trend)
            vol_val = r.volatility.value if hasattr(r.volatility, 'value') else str(r.volatility)
            regime_data.append({
                "date": r.date,
                "regime_id": r.regime_id or 0,
                "regime_confidence": r.regime_confidence or 0.5,
                "quality_score": r.quality_score or 0.5,
                "regime_trend_bullish": 1.0 if trend_val == "bullish" else 0.0,
                "regime_trend_bearish": 1.0 if trend_val == "bearish" else 0.0,
                "regime_trend_neutral": 1.0 if trend_val == "neutral" else 0.0,
                "regime_vol_high": 1.0 if vol_val == "high" else 0.0,
                "is_transition": float(r.is_transition) if r.is_transition is not None else 0.0,
            })
        regime_df = pd.DataFrame(regime_data)
        if "date" in df.columns:
            regime_df["date"] = pd.to_datetime(regime_df["date"])
            df = df.merge(regime_df, on="date", how="left")
        # Fill NaN regime columns with defaults
        regime_fill = {
            "regime_id": 0, "regime_confidence": 0.5, "quality_score": 0.5,
            "regime_trend_bullish": 0.0, "regime_trend_bearish": 0.0,
            "regime_trend_neutral": 1.0,
            "regime_vol_high": 0.0, "is_transition": 0.0,
        }
        for col, default in regime_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

    if normalize:
        df = normalize_dataframe(df)

    return df


async def get_model_ready_data(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
    seq_len: int = 15,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get normalized feature DataFrame and feature column list for model input.

    Returns (df, feature_cols) — ready for prepare_model_input().
    """
    df = await get_stock_features(db, stock_id, interval, start_date, end_date, normalize=True)
    if df.empty:
        return df, []

    feature_cols = get_feature_columns(df)
    return df, feature_cols


async def sync_and_compute(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
) -> dict:
    """Full pipeline: sync OHLCV from Kite → compute indicators → store all."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        return {"error": f"Stock {stock_id} not found"}

    ohlcv_count = await sync_stock_ohlcv(db, stock, interval)

    indicator_count = 0
    try:
        indicator_count = await compute_and_store_indicators(db, stock_id, interval)
    except Exception as ind_err:
        logger.error(
            "Indicator computation failed for stock_id=%d (%s): %s",
            stock_id, stock.symbol, ind_err,
        )

    return {
        "stock_id": stock_id,
        "symbol": stock.symbol,
        "ohlcv_synced": ohlcv_count,
        "indicators_computed": indicator_count,
    }
