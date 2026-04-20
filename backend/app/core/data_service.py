"""Data service — orchestrates data pipeline, indicator computation, and DB storage.

Sits between the API routes and the lower-level modules (data_pipeline, indicators, normalizer).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel
from app.core.indicators import compute_all_indicators, get_indicator_columns, compute_weekly_indicators, WEEKLY_INDICATOR_COLS
from app.core.normalizer import normalize_dataframe, get_feature_columns, prepare_model_input
from app.core.data_pipeline import ohlcv_to_dataframe, sync_stock_ohlcv, sync_all_stocks
from app.db import crud
import json
import requests
import time as _time
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

_NSE_PRESET_ENDPOINTS: dict[str, tuple[str, str | None]] = {
    "nifty_50": ("index", "NIFTY 50"),
    "nifty_100": ("index", "NIFTY 100"),
    "nifty_500": ("index", "NIFTY 500"),
    "nifty_bank": ("index", "NIFTY BANK"),
    "nifty_it": ("index", "NIFTY IT"),
    "nifty_pharma": ("index", "NIFTY PHARMA"),
    "nifty_auto": ("index", "NIFTY AUTO"),
    "nifty_fmcg": ("index", "NIFTY FMCG"),
    "nifty_metal": ("index", "NIFTY METAL"),
    "nifty_psu_bank": ("index", "NIFTY PSU BANK"),
    "nifty_financial": ("index", "NIFTY FINANCIAL SERVICES"),
    "nifty_realty": ("index", "NIFTY REALTY"),
    "nifty_energy": ("index", "NIFTY ENERGY"),
    "nifty_midcap50": ("index", "NIFTY MIDCAP 50"),
    "nse_etf": ("etf", None),
}

_NSE_PRESET_CACHE: dict[str, tuple[float, set[str]]] = {}
_NSE_PRESET_CACHE_TTL = 15 * 60.0


def _normalize_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    return symbol.strip().upper() or None


def _fetch_nse_preset_symbols(category: str) -> set[str]:
    cached = _NSE_PRESET_CACHE.get(category)
    now = _time.monotonic()
    if cached and (now - cached[0]) < _NSE_PRESET_CACHE_TTL:
        return set(cached[1])

    endpoint = _NSE_PRESET_ENDPOINTS.get(category)
    if not endpoint:
        raise HTTPException(status_code=400, detail=f"Unsupported universe category: {category}")

    kind, index_name = endpoint
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)

    try:
        if kind == "index":
            response = session.get(
                "https://www.nseindia.com/api/equity-stockIndices",
                params={"index": index_name},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            symbols = {
                normalized
                for item in payload.get("data", [])
                if (normalized := _normalize_symbol(item.get("symbol")))
                and normalized != _normalize_symbol(index_name)
            }
        else:
            response = session.get("https://www.nseindia.com/api/etf", timeout=20)
            response.raise_for_status()
            payload = response.json()
            symbols = {
                normalized
                for item in payload.get("data", [])
                if (normalized := _normalize_symbol(item.get("symbol")))
            }

        _NSE_PRESET_CACHE[category] = (now, set(symbols))
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch NSE symbols for {category}: {e}")
        # Return empty on failure to avoid crashing the caller
        return set()

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
    "bb_upper": "bb_upper",
    "bb_lower": "bb_lower",
    "bb_mid": "bb_mid",
    "tgrb_top": "tgrb_top",
    "tgrb_green": "tgrb_green",
    "tgrb_red": "tgrb_red",
    "tgrb_bottom": "tgrb_bottom",
    "ema_20": "ema_20",
    "atr": "atr",
    # New stationary ML features
    "dist_sma_50": "dist_sma_50",
    "dist_sma_200": "dist_sma_200",
    "roc_1": "roc_1",
    "roc_5": "roc_5",
    "roc_20": "roc_20",
    "atr_pct": "atr_pct",
    "realized_vol_20": "realized_vol_20",
    "bb_pctb": "bb_pctb",
    "bb_width": "bb_width",
    "cmf_20": "cmf_20",
    "dist_vwap_5": "dist_vwap_5",
    "macd_hist_norm": "macd_hist_norm",
    "adx_norm": "adx_norm",
    "adx_pos_norm": "adx_pos_norm",
    "adx_neg_norm": "adx_neg_norm",
}


def merge_weekly_features(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge weekly technical indicators into a daily OHLCV DataFrame.

    Algorithm
    ---------
    1. Compute weekly indicators (RSI, MACD, SMA-50, ADX) on *weekly_df*.
    2. Align weekly values onto daily dates using ``pd.merge_asof`` with
       ``direction='backward'`` — each daily row receives the most recent
       *prior* week's indicator value (never the current week, so there is
       zero lookahead bias).
    3. Forward-fill any remaining NaNs (e.g. early rows before the first
       complete weekly bar).
    4. Fill with neutral defaults if weekly data is missing entirely.

    Parameters
    ----------
    daily_df  : unsorted or sorted daily OHLCV + indicator DataFrame.
    weekly_df : raw weekly OHLCV DataFrame (will have indicators computed internally).

    Returns
    -------
    daily_df with WEEKLY_INDICATOR_COLS appended as new columns.
    """
    if weekly_df is None or weekly_df.empty:
        logger.debug("merge_weekly_features: no weekly data — filling with neutral defaults.")
        for col in WEEKLY_INDICATOR_COLS:
            daily_df[col] = 0.0
        return daily_df

    try:
        w_indicators = compute_weekly_indicators(weekly_df)
    except Exception as exc:
        logger.warning("merge_weekly_features: weekly indicator computation failed (%s) — using defaults.", exc)
        for col in WEEKLY_INDICATOR_COLS:
            daily_df[col] = 0.0
        return daily_df

    if w_indicators.empty or len(w_indicators) < 2:
        for col in WEEKLY_INDICATOR_COLS:
            daily_df[col] = 0.0
        return daily_df

    # Ensure both DataFrames have datetime date columns, sorted ascending.
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    w_indicators["date"] = pd.to_datetime(w_indicators["date"])

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    w_indicators = w_indicators.sort_values("date").reset_index(drop=True)

    # ── Incomplete-bar guard ───────────────────────────────────────────────
    # During live trading (Mon–Thu) the most recent weekly bar in the DB is
    # the *current* week whose candle is still open and whose indicators are
    # still actively changing.  Drop it so daily rows never receive a partial
    # weekly RSI/MACD value.  This is a no-op for historical data because every
    # past weekly bar is already fully closed.
    if not w_indicators.empty:
        from datetime import date as _date
        today = pd.Timestamp(_date.today())
        last_week_date = w_indicators["date"].iloc[-1]
        # ISO weekday: Mon=0 … Fri=4 … Sun=6
        # If today falls inside the same ISO week as the last bar → bar is open
        if (today - last_week_date).days < 7 and today.weekday() < 5:
            w_indicators = w_indicators.iloc[:-1].reset_index(drop=True)
            logger.debug(
                "merge_weekly_features: dropped last unclosed weekly bar (%s) — today is %s (%s)",
                last_week_date.date(), today.date(), today.strftime("%A"),
            )
        if len(w_indicators) < 2:
            for col in WEEKLY_INDICATOR_COLS:
                daily_df[col] = 0.0
            return daily_df

    # merge_asof: for each daily row find the last weekly bar whose date
    # is <= the daily date.  This is the strict no-lookahead condition.
    merged = pd.merge_asof(
        daily_df,
        w_indicators,
        on="date",
        direction="backward",
        suffixes=("", "_weekly_raw"),
    )

    # Forward-fill any NaN at the start of the series (before the first full weekly bar).
    weekly_cols_present = [c for c in WEEKLY_INDICATOR_COLS if c in merged.columns]
    merged[weekly_cols_present] = merged[weekly_cols_present].ffill().fillna(0.0)

    logger.debug(
        "merge_weekly_features: merged %d weekly bars into %d daily rows (%d weekly cols)",
        len(w_indicators), len(merged), len(weekly_cols_present),
    )
    return merged


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
    use_weekly_context: bool = True,
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

    When ``use_weekly_context=True`` (the default) and ``interval='day'``,
    weekly OHLCV is fetched and weekly indicators are forward-filled into every
    daily row before normalization.  This gives models a multi-timeframe view
    of the weekly trend without any lookahead bias.

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

    # ── Weekly multi-timeframe enrichment ──────────────────────────────
    # Only applied when requesting daily features; weekly-on-weekly doesn't
    # add information and would require weekly-of-weekly data.
    if use_weekly_context and interval == "day":
        try:
            weekly_ohlcv_rows = await crud.get_ohlcv_as_dicts(db, stock_id, "week")
            if weekly_ohlcv_rows:
                from app.core.data_pipeline import ohlcv_to_dataframe
                weekly_df = ohlcv_to_dataframe(weekly_ohlcv_rows)
                df = merge_weekly_features(df, weekly_df)
                logger.debug(
                    "stock_id=%d: weekly context merged (%d weekly bars)",
                    stock_id, len(weekly_ohlcv_rows),
                )
            else:
                logger.debug("stock_id=%d: no weekly OHLCV rows found — skipping weekly enrichment", stock_id)
                for col in WEEKLY_INDICATOR_COLS:
                    df[col] = 0.0
        except Exception as _wk_exc:
            logger.warning("Weekly context enrichment failed for stock_id=%d: %s", stock_id, _wk_exc)
            for col in WEEKLY_INDICATOR_COLS:
                if col not in df.columns:
                    df[col] = 0.0

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


async def prepare_model_input(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
    seq_len: int = 15,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Fetch, compute, merge, and normalize data for a single stock."""
    df = await get_stock_features(db, stock_id, interval, normalize=True)
    if df.empty:
        return None, []
    
    feature_cols = get_feature_columns()
    return df, feature_cols


# ── Universe Resolution ───────────────────────────────────────────────────

class UniverseConfig(BaseModel):
    """Stock universe configuration."""
    category: str = "nifty_50"
    custom_symbols: list[str] = []


async def get_universe_stocks(db: AsyncSession) -> list:
    """Read the stock_universe setting and resolve into a list of Stock objects."""
    raw = await crud.get_setting(db, "stock_universe")
    if raw:
        try:
            cfg = UniverseConfig(**json.loads(raw))
        except Exception:
            cfg = UniverseConfig()
    else:
        cfg = UniverseConfig()
        
    return await resolve_universe_stocks(db, cfg)


async def resolve_universe_stocks(db: AsyncSession, cfg: UniverseConfig) -> list:
    """Resolve a UniverseConfig into a list of Stock objects from the DB."""
    from app.db.models import Stock
    from sqlalchemy import select as sa_select
    
    # 1. Get symbols
    if cfg.category == "custom":
        symbols = {s.upper() for s in cfg.custom_symbols if s.strip()}
    else:
        symbols = _fetch_nse_preset_symbols(cfg.category)

    if cfg.custom_symbols:
        symbols.update(s.upper() for s in cfg.custom_symbols if s.strip())

    if not symbols:
        return []

    # 2. Query DB
    result = await db.execute(
        sa_select(Stock)
        .where(Stock.is_active == True, Stock.symbol.in_(symbols))
        .order_by(Stock.symbol)
    )
    return list(result.scalars().all())


async def get_model_ready_data(
    db: AsyncSession,
    stock_id: int,
    interval: str = "day",
    seq_len: int = 15,
    start_date: date | None = None,
    end_date: date | None = None,
    use_weekly_context: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get normalized feature DataFrame and feature column list for model input.
    """
    df = await get_stock_features(
        db, stock_id, interval, start_date, end_date,
        normalize=True, use_weekly_context=use_weekly_context,
    )
    if df.empty:
        return df, []

    # FIX: Ensure inference features strictly match RL training features!
    trend = df.get("trend", pd.Series(["neutral"] * len(df), index=df.index))
    vol = df.get("volatility", pd.Series(["low"] * len(df), index=df.index))
    
    df["regime_trend_bullish"] = (trend == "bullish").astype(float)
    df["regime_trend_bearish"] = (trend == "bearish").astype(float)
    df["regime_trend_neutral"] = (trend == "neutral").astype(float)
    df["regime_vol_high"] = (vol == "high").astype(float)

    feature_cols = get_feature_columns(df)
    
    # Ensure regime columns are explicitly included in feature_cols
    regime_feats = [
        "regime_trend_bullish", "regime_trend_bearish", 
        "regime_trend_neutral", "regime_vol_high", "regime_confidence"
    ]
    for col in regime_feats:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

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
    force_full: bool = False,
) -> dict:
    """Full pipeline: sync OHLCV from Kite → compute indicators → classify regimes → store all."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        return {"error": f"Stock {stock_id} not found"}

    ohlcv_count = await sync_stock_ohlcv(db, stock, interval, force_full)

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
