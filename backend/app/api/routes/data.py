from __future__ import annotations

import logging
import time as _time
from datetime import date
from typing import Annotated

import requests

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud
from app.core import data_service, data_pipeline
from app.core import zerodha as _zerodha

router = APIRouter()

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
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch NSE universe for {category}: {exc}",
        )

    _NSE_PRESET_CACHE[category] = (now, symbols)
    return set(symbols)


# ── Schemas ────────────────────────────────────────────────────────────

class StockOut(BaseModel):
    id: int
    symbol: str
    exchange: str
    kite_id: int | None = None
    sector: str | None = None
    is_active: bool

    class Config:
        from_attributes = True


class SyncRequest(BaseModel):
    stock_ids: list[int] | None = None
    interval: str = "day"
    force_full: bool = False


class SyncResult(BaseModel):
    stock_id: int
    symbol: str
    ohlcv_synced: int
    indicators_computed: int


class OHLCVOut(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float


class IndicatorOut(BaseModel):
    date: date
    rsi: float | None = None
    srsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    adx: float | None = None
    adx_pos: float | None = None
    adx_neg: float | None = None
    kama: float | None = None
    vwkama: float | None = None
    obv: float | None = None
    sma_5: float | None = None
    sma_12: float | None = None
    sma_24: float | None = None
    sma_50: float | None = None
    sma_100: float | None = None
    sma_200: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_mid: float | None = None
    tgrb_top: float | None = None
    tgrb_green: float | None = None
    tgrb_red: float | None = None
    tgrb_bottom: float | None = None

    class Config:
        from_attributes = True


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("/stocks", response_model=list[StockOut])
async def list_stocks(
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(True),
):
    """List all tracked stocks."""
    if active_only:
        stocks = await crud.get_all_active_stocks(db)
    else:
        from sqlalchemy import select
        from app.db.models import Stock
        result = await db.execute(select(Stock).order_by(Stock.symbol))
        stocks = result.scalars().all()
    return stocks


@router.get("/stocks/{stock_id}/ohlcv", response_model=list[OHLCVOut])
async def get_stock_ohlcv(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    interval: str = Query("day"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Get OHLCV data for a stock."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    rows = await crud.get_ohlcv(db, stock_id, interval, start_date, end_date)
    return [
        OHLCVOut(
            date=r.date,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            adj_close=r.adj_close,
            volume=r.volume,
        )
        for r in rows
    ]


@router.get("/stocks/{stock_id}/indicators", response_model=list[IndicatorOut])
async def get_stock_indicators(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    interval: str = Query("day"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Get computed indicators for a stock."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    rows = await crud.get_indicators(db, stock_id, interval, start_date, end_date)
    return rows


@router.post("/sync", response_model=list[SyncResult])
async def sync_data(
    req: SyncRequest,
    db: AsyncSession = Depends(get_db),
):
    """Trigger OHLCV + indicator sync for selected stocks."""
    if req.stock_ids:
        stock_ids = req.stock_ids
    else:
        stocks = await crud.get_all_active_stocks(db)
        stock_ids = [s.id for s in stocks]

    results = []
    for sid in stock_ids:
        try:
            result = await data_service.sync_and_compute(db, sid, req.interval)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            results.append(SyncResult(**result))
        except HTTPException:
            raise
        except Exception as e:
            # Fetch symbol for a meaningful error row (best-effort)
            try:
                fallback_stock = await crud.get_stock_by_id(db, sid)
                fallback_sym = fallback_stock.symbol if fallback_stock else "?"
            except Exception:
                fallback_sym = "?"
            logger.error("Sync failed for stock_id=%d (%s): %s", sid, fallback_sym, e)
            results.append(SyncResult(stock_id=sid, symbol=fallback_sym, ohlcv_synced=-1, indicators_computed=-1))

    return results


@router.post("/sync/stocks")
async def sync_stock_list(db: AsyncSession = Depends(get_db)):
    """Fetch and populate the NSE stock list from Kite API."""
    count = await data_pipeline.populate_stock_list(db)
    return {"stocks_populated": count}


@router.post("/sync/holidays")
async def sync_holidays(db: AsyncSession = Depends(get_db)):
    """Fetch and store NSE holidays."""
    count = await data_pipeline.sync_holidays(db)
    return {"holidays_synced": count}


@router.get("/stocks/{stock_id}/features")
async def get_stock_features(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    interval: str = Query("day"),
    normalize: bool = Query(True),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Get full feature matrix (OHLCV + indicators, optionally normalized)."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    df = await data_service.get_stock_features(
        db, stock_id, interval, start_date, end_date, normalize=normalize
    )
    if df.empty:
        return {"columns": [], "data": [], "rows": 0}

    return {
        "columns": df.columns.tolist(),
        "data": df.to_dict(orient="records"),
        "rows": len(df),
    }


# ── Stock Universe ─────────────────────────────────────────────────────


class UniverseConfig(BaseModel):
    """Stock universe configuration."""
    category: str = "nifty_50"
    # nifty_50 | nifty_100 | nifty_500
    # nifty_bank | nifty_it | nifty_pharma | nifty_auto | nifty_fmcg
    # nifty_metal | nifty_psu_bank | nifty_financial | nifty_realty | nifty_energy
    # nifty_midcap50 | nse_etf | custom
    custom_symbols: list[str] = []  # extra symbols added by user


class PresetSymbolsOut(BaseModel):
    category: str
    symbols: list[str]
    resolved_count: int


class UniverseOut(BaseModel):
    category: str
    custom_symbols: list[str]
    resolved_count: int


@router.get("/universe")
async def get_universe(db: AsyncSession = Depends(get_db)):
    """Get the current stock universe config."""
    import json
    raw = await crud.get_setting(db, "stock_universe")
    if raw:
        try:
            data = json.loads(raw)
            cfg = UniverseConfig(**data)
        except Exception:
            cfg = UniverseConfig()
    else:
        cfg = UniverseConfig()

    # Resolve count
    count = await _resolve_universe_count(db, cfg)
    return UniverseOut(
        category=cfg.category,
        custom_symbols=cfg.custom_symbols,
        resolved_count=count,
    )


@router.get("/universe/presets/{category}", response_model=PresetSymbolsOut)
async def get_universe_preset_symbols(category: str):
    """Fetch the live NSE symbols for a preset universe category."""
    symbols = sorted(_fetch_nse_preset_symbols(category))
    return PresetSymbolsOut(
        category=category,
        symbols=symbols,
        resolved_count=len(symbols),
    )


@router.put("/universe")
async def set_universe(
    cfg: UniverseConfig,
    db: AsyncSession = Depends(get_db),
):
    """Update the stock universe config."""
    import json
    await crud.set_setting(db, "stock_universe", json.dumps(cfg.model_dump()))
    count = await _resolve_universe_count(db, cfg)
    return UniverseOut(
        category=cfg.category,
        custom_symbols=cfg.custom_symbols,
        resolved_count=count,
    )


@router.get("/stocks/universe", response_model=list[StockOut])
async def list_universe_stocks(db: AsyncSession = Depends(get_db)):
    """List stocks filtered by the active universe."""
    import json
    raw = await crud.get_setting(db, "stock_universe")
    if raw:
        try:
            cfg = UniverseConfig(**json.loads(raw))
        except Exception:
            cfg = UniverseConfig()
    else:
        cfg = UniverseConfig()

    return await _resolve_universe_stocks(db, cfg)


async def _get_universe_symbols(cfg: UniverseConfig) -> set[str]:
    """Return the set of symbols for the universe."""
    if cfg.category == "custom":
        base = {s.upper() for s in cfg.custom_symbols if s.strip()}
    else:
        base = _fetch_nse_preset_symbols(cfg.category)

    if cfg.custom_symbols:
        base.update(s.upper() for s in cfg.custom_symbols if s.strip())

    return base


async def _resolve_universe_stocks(
    db: AsyncSession, cfg: UniverseConfig
) -> list:
    from app.db.models import Stock
    symbols = await _get_universe_symbols(cfg)
    if not symbols:
        return []

    from sqlalchemy import select as sa_select
    result = await db.execute(
        sa_select(Stock)
        .where(Stock.is_active == True, Stock.symbol.in_(symbols))
        .order_by(Stock.symbol)
    )
    return list(result.scalars().all())


async def _resolve_universe_count(db: AsyncSession, cfg: UniverseConfig) -> int:
    stocks = await _resolve_universe_stocks(db, cfg)
    return len(stocks)


# ── Zerodha Instruments ────────────────────────────────────────────────

_inst_cache: dict[str, list] = {}
_inst_cache_ts: dict[str, float] = {}
_INST_TTL = 3600.0  # 1-hour TTL


class InstrumentOut(BaseModel):
    symbol: str
    name: str
    instrument_token: int
    lot_size: int
    exchange: str


@router.get("/instruments", response_model=list[InstrumentOut])
async def get_zerodha_instruments(
    exchange: str = Query("NSE"),
):
    """Return all EQ instruments for an exchange from Zerodha, cached 1 h.

    Requires Zerodha to be authenticated first (Settings → Authenticate Token).
    """
    key = exchange.upper()
    now = _time.monotonic()
    if key in _inst_cache and (now - _inst_cache_ts.get(key, 0.0)) < _INST_TTL:
        return _inst_cache[key]

    try:
        raw = _zerodha.get_instruments(exchange)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Zerodha instruments unavailable: {exc}. Authenticate first in Settings.",
        )

    result = [
        InstrumentOut(
            symbol=str(inst["tradingsymbol"]),
            name=str(inst.get("name") or ""),
            instrument_token=int(inst["instrument_token"]),
            lot_size=int(inst.get("lot_size") or 1),
            exchange=str(inst.get("exchange") or exchange),
        )
        for inst in raw
        if inst.get("instrument_type") == "EQ"
    ]
    result.sort(key=lambda x: x.symbol)

    _inst_cache[key] = result
    _inst_cache_ts[key] = now
    return result


@router.delete("/instruments/cache")
async def clear_instrument_cache():
    """Bust the in-memory instrument cache to force a fresh Zerodha fetch."""
    _inst_cache.clear()
    _inst_cache_ts.clear()
    return {"cleared": True}
