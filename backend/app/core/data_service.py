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
    "atr": "atr",
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
    ohlcv_rows = await crud.get_ohlcv_as_dicts(db, stock_id, interval)
    if not ohlcv_rows:
        logger.warning("No OHLCV data for stock_id=%d interval=%s", stock_id, interval)
        return 0

    df = ohlcv_to_dataframe(ohlcv_rows)
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

    Uses a single SQL JOIN via ``crud.get_full_stock_features()`` to fetch
    OHLCV, pre-computed indicator values, and regime labels from the DB cache
    in one round-trip.  This replaces the old pattern of fetching OHLCV and
    then re-computing MACD, RSI, Bollinger Bands etc. on the fly, which wasted
    CPU on every training and prediction call.

    Falls back to live indicator computation (``compute_all_indicators``) only
    when the indicators table has no rows for this stock (e.g. just after a
    fresh sync before the nightly indicator job has run).

    Returns a DataFrame with all features, optionally normalized.
    """
    rows = await crud.get_full_stock_features(db, stock_id, interval, start_date, end_date)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 50:
        return pd.DataFrame()

    # If the DB has no pre-computed indicator values yet (indicator columns all
    # None), fall back to live computation so training isn't silently broken.
    indicator_sentinel = "rsi"  # lightweight column present for all synced stocks
    if indicator_sentinel not in df.columns or df[indicator_sentinel].isna().all():
        logger.warning(
            "stock_id=%d has no pre-computed indicators — "
            "running compute_all_indicators() as fallback",
            stock_id,
        )
        df = compute_all_indicators(df)

    # Default-fill regime columns if the regime table was empty / partially filled
    regime_fill = {
        "regime_id": 0, "regime_confidence": 0.5, "quality_score": 0.5,
        "regime_trend_bullish": 0.0, "regime_trend_bearish": 0.0,
        "regime_trend_neutral": 1.0,
        "regime_vol_high": 0.0, "is_transition": 0.0,
    }
    for col, default in regime_fill.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # Enrich with market context (FII/DII flow + sector breadth)
    try:
        from app.core.market_context import enrich_with_market_context
        stock_obj = await crud.get_stock_by_id(db, stock_id)
        if stock_obj is not None:
            df = await enrich_with_market_context(df, db, stock_obj)
    except Exception as _mc_exc:
        logger.warning("Market context enrichment skipped: %s", _mc_exc)

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


async def get_sector_features(
    db: AsyncSession,
    sector: str,
    interval: str = "day",
    start_date: date | None = None,
    end_date: date | None = None,
    normalize: bool = True,
    min_stocks: int = 2,
) -> pd.DataFrame:
    """Pool feature data from all active stocks in a sector.

    Returns a single concatenated DataFrame with an added ``stock_id`` column so
    downstream training code can track which rows belong to which instrument.
    This gives the RL / LSTM / KNN models an order-of-magnitude more training
    data by learning universal mechanics across correlated stocks rather than
    memorising one stock's historical quirks.

    Parameters
    ----------
    min_stocks : int
        Minimum number of stocks required; returns empty DataFrame when fewer
        stocks exist so callers can fall back to single-stock training.
    """
    stocks = await crud.get_stocks_by_sector(db, sector)

    if len(stocks) < min_stocks:
        logger.warning(
            "Sector '%s' has %d active stocks (need >= %d). "
            "Returning empty DataFrame — caller should fall back to single-stock training.",
            sector, len(stocks), min_stocks,
        )
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = await get_stock_features(
                db, stock.id, interval, start_date, end_date, normalize=normalize
            )
            if not df.empty:
                df["stock_id"] = stock.id
                df["symbol"]   = stock.symbol
                dfs.append(df)
        except Exception as exc:
            logger.warning("Skipping stock %s for sector pool: %s", stock.symbol, exc)

    if not dfs:
        return pd.DataFrame()

    pooled = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Pooled sector '%s': %d/%d stocks, %d rows total",
        sector, len(dfs), len(stocks), len(pooled),
    )
    return pooled


async def compute_and_store_regimes(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
) -> int:
    """Compute market regimes and quality scores for a stock and store in DB."""
    from app.core.regime_classifier import classify_and_score

    # Fetch full feature matrix (needed by regime classifier)
    df = await get_stock_features(db, stock_id, interval, normalize=False)
    if df.empty or len(df) < 50:
        logger.warning("Insufficient data for regimes (stock_id=%d)", stock_id)
        return 0

    # Classify regimes and compute scores
    df = classify_and_score(df, use_gmm=True)

    # Build DB rows
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "stock_id": stock_id,
            "date": row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"],
            "interval": interval,
            "trend": row["trend"],
            "volatility": row["volatility"],
            "regime_id": int(row["regime_id"]),
            "regime_confidence": float(row["regime_confidence"]),
            "quality_score": float(row["quality_score"]),
            "is_transition": bool(row["is_transition"]),
        })

    count = await crud.bulk_upsert_regimes(db, rows)
    logger.info("Upserted %d regime rows for stock_id=%d", count, stock_id)
    return count


async def sync_and_compute(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
) -> dict:
    """Full pipeline: sync OHLCV from Kite → compute indicators → classify regimes → store all."""
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

    regime_count = 0
    try:
        regime_count = await compute_and_store_regimes(db, stock_id, interval)
    except Exception as reg_err:
        logger.error(
            "Regime classification failed for stock_id=%d (%s): %s",
            stock_id, stock.symbol, reg_err,
        )

    return {
        "stock_id": stock_id,
        "symbol": stock.symbol,
        "ohlcv_synced": ohlcv_count,
        "indicators_computed": indicator_count,
        "regimes_classified": regime_count,
    }
