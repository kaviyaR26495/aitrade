"""Point-in-Time (PIT) Fundamental Data Pipeline.

Ingests stock-level fundamental metrics by scraping Screener.in (SSR HTML,
no login required for key ratios) and stores one row per (stock, date) in
``stock_fundamentals_pit``.  A separate step computes ML-ready z-scores
into ``stock_fundamental_zscores``.

PIT guarantee
-------------
The ingestion function only inserts a row for *today's* date — it never
back-fills historical dates.  This prevents look-ahead bias: when the ML
models are trained on past windows, each row contains only the fundamental
data that was known on that calendar date.

Quarterly Intelligent Refresh (QIR)
------------------------------------
Fundamentals change at most 4×/year (quarterly earnings).  A full HTTP
scrape of 500 stocks daily is unnecessary and risks IP bans.  QIR splits
pending stocks into two groups:

* **need_scrape** – no PIT row in the current calendar quarter, OR the
  stock has an NSE board meeting / earnings event within ±7 days of today.
  These stocks get a live Screener.in scrape.

* **copy_forward** – already has a row in the current quarter but not
  today's date.  A new row is inserted with today's date copying the most
  recent quarter values (``source="scrnr_copy_fwd"``).  Zero HTTP requests.

Typical day: ~95% copy-forward, ~5% scrape.  Quarter start days and
earnings-season weeks flip the ratio but remain bounded and polite.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Stock,
    StockFundamentalPIT,
    StockFundamentalZScore,
    FundamentalSectorStats,
    now_ist,
)

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ── Rotating user-agents ───────────────────────────────────────────────
_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
]

# ── NSE → Screener symbol overrides ───────────────────────────────────
# Zerodha/NSE symbols that differ from Screener.in slugs.
_NSE_TO_SCREENER: dict[str, str] = {
    "M&M": "MM",
    "M&MFIN": "MMFIN",
    "L&T": "LT",
    "L&TFH": "LTFH",
    "AT&T": "ATT",
    "I&MINDIA": "IMINDIA",
    "MOTHERSUMI": "MSUMI",
    "MRF": "MRF",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "RELIANCE": "RELIANCE",
    "HDFCAMC": "HDFCAMC",
    "TRENT": "TRENT",
    "NIFTY50": "NIFTY",
    "NIFTYBEES": "NIFTYBEES",
    "JUNIORBEES": "JUNIORBEES",
}

_EMPTY_METRICS: dict = {
    k: None
    for k in ("pe_ratio", "forward_pe", "pb_ratio", "dividend_yield", "roe", "debt_equity")
}


# ── Symbol resolution ──────────────────────────────────────────────────

def _resolve_screener_symbol(symbol: str) -> str:
    """Convert an NSE/Zerodha symbol to a Screener.in URL slug.

    1. Strip .NS / .BSE suffixes.
    2. Apply manual override table for known mismatches.
    3. Remove & characters (residual edge cases).
    """
    clean = symbol.strip().upper()
    # Strip exchange suffixes
    for suffix in (".NS", ".BSE", ".BO"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    # Apply manual override
    if clean in _NSE_TO_SCREENER:
        return _NSE_TO_SCREENER[clean]
    # Remove & which is invalid in a URL path
    clean = clean.replace("&", "")
    return clean


# ── Screener.in HTML parser ────────────────────────────────────────────

def _parse_key_ratio(soup: BeautifulSoup, label_pattern: str) -> float | None:
    """Extract a numeric value from Screener's key-ratios sidebar.

    The sidebar HTML looks like:
        <li class="flex ...">
            <span class="name">Stock P/E</span>
            <span class="number">24.1</span>
        </li>
    """
    try:
        span = soup.find("span", string=re.compile(label_pattern, re.IGNORECASE))
        if span is None:
            return None
        parent = span.find_parent("li")
        if parent is None:
            return None
        num_span = parent.find("span", class_="number")
        if num_span is None:
            return None
        text = num_span.get_text(strip=True)
        # Strip currency symbols, commas, percent signs, trailing spaces
        text = re.sub(r"[₹,%\s]", "", text)
        return float(text) if text else None
    except (ValueError, AttributeError):
        return None


def _parse_debt_equity(soup: BeautifulSoup) -> float | None:
    """Extract Debt/Equity from the Ratios table (most recent year column).

    The ratios section has rows like:
        <tr><td class="text">Debt / Equity</td><td>0.02</td>...<td>0.03</td></tr>
    We take the last non-empty <td> which is the most recent annual figure.
    Returns None for banks/holding companies where the section is absent.
    """
    try:
        section = soup.find("section", id="ratios")
        if section is None:
            return None
        for row in section.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            label = cells[0].get_text(strip=True).lower()
            if "debt" in label and "equity" in label:
                # Walk cells right-to-left to find the most recent non-empty value
                for cell in reversed(cells[1:]):
                    text = re.sub(r"[₹,%\s]", "", cell.get_text(strip=True))
                    if text:
                        try:
                            return float(text)
                        except ValueError:
                            continue
        return None
    except Exception:
        return None


async def _scrape_screener(symbol: str, client: httpx.AsyncClient) -> dict:
    """Fetch and parse fundamental metrics for one stock from Screener.in.

    Tries the consolidated page first; falls back to standalone on 404.
    Returns a dict with keys matching StockFundamentalPIT columns.
    All values are None on any parse/network failure (never raises).
    """
    slug = _resolve_screener_symbol(symbol)
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        url = f"https://www.screener.in/company/{slug}/consolidated/"
        response = await client.get(url, headers=headers, follow_redirects=True)
        if response.status_code == 404:
            url = f"https://www.screener.in/company/{slug}/"
            response = await client.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        _log.warning("Screener fetch failed for %s (slug=%s): %s", symbol, slug, exc)
        return dict(_EMPTY_METRICS)

    try:
        soup = BeautifulSoup(response.text, "lxml")

        pe_ratio = _parse_key_ratio(soup, r"^Stock P/E$")
        roe_raw = _parse_key_ratio(soup, r"^ROE$")
        roce_raw = _parse_key_ratio(soup, r"^ROCE$")
        div_yield_raw = _parse_key_ratio(soup, r"^Dividend Yield$")
        current_price = _parse_key_ratio(soup, r"^Current Price$")
        book_value = _parse_key_ratio(soup, r"^Book Value$")
        debt_equity = _parse_debt_equity(soup)

        # Screener shows ROE/ROCE as a percentage (18.5 means 18.5%).
        # Prefer ROE; fall back to ROCE if ROE absent.
        roe_pct = roe_raw if roe_raw is not None else roce_raw
        roe = _safe_float(roe_pct / 100.0) if roe_pct is not None else None

        # Dividend yield also shown as percentage on Screener.
        dividend_yield = _safe_float(div_yield_raw / 100.0) if div_yield_raw is not None else None

        # P/B = Current Price / Book Value (not always directly listed).
        pb_ratio: float | None = None
        if current_price is not None and book_value is not None and book_value > 0:
            pb_ratio = _safe_float(current_price / book_value)

        metrics = {
            "pe_ratio": _safe_float(pe_ratio),
            "forward_pe": None,          # not available on Screener free tier
            "pb_ratio": pb_ratio,
            "dividend_yield": dividend_yield,
            "roe": roe,
            "debt_equity": _safe_float(debt_equity),
        }
        _log.debug("Scraped %s (slug=%s): %s", symbol, slug, metrics)
        return metrics

    except Exception as exc:
        _log.warning("Screener parse failed for %s: %s", symbol, exc)
        return dict(_EMPTY_METRICS)


async def fetch_stock_fundamentals(symbol: str) -> dict:
    """Fetch fundamental metrics for a single NSE stock via Screener.in.

    Convenience wrapper around _scrape_screener for ad-hoc / test use.
    Creates a short-lived httpx client — for bulk ingestion use
    ingest_fundamentals_pit which shares a single client across all stocks.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        return await _scrape_screener(symbol, client)


# ── Quarterly Intelligent Refresh (QIR) helpers ───────────────────────

def _current_quarter(d: date) -> tuple[int, int]:
    """Return (year, quarter_number) for a given date.  Q1=Jan-Mar, Q4=Oct-Dec."""
    return d.year, (d.month - 1) // 3 + 1


def _earnings_flagged_stocks(stocks: Sequence[Stock], as_of_date: date) -> set[int]:
    """Return stock_ids that have an NSE board meeting within ±7 days of as_of_date.

    Uses nsepython to fetch the corporate event calendar.  Returns an empty
    set on any failure so QIR degrades gracefully to quarter-boundary logic.
    """
    try:
        import nsepython as nse

        calendar = nse.nse_get_event_calendar()   # returns list of dicts
        if not isinstance(calendar, list):
            return set()

        window_start = (as_of_date - timedelta(days=7)).isoformat()
        window_end = (as_of_date + timedelta(days=7)).isoformat()

        # Build a set of NSE symbols with upcoming board meetings
        event_symbols: set[str] = set()
        for event in calendar:
            purpose = str(event.get("purpose", "")).lower()
            event_date_str = str(event.get("date", ""))
            if "board meeting" not in purpose and "financial result" not in purpose:
                continue
            if window_start <= event_date_str <= window_end:
                sym = event.get("symbol", "")
                if sym:
                    event_symbols.add(sym.upper())

        sym_to_id = {s.symbol.upper(): s.id for s in stocks}
        return {sym_to_id[sym] for sym in event_symbols if sym in sym_to_id}

    except Exception as exc:
        _log.debug("NSE event calendar lookup failed (QIR will use quarter boundary only): %s", exc)
        return set()


async def ingest_fundamentals_pit(
    db: AsyncSession,
    stocks: Sequence[Stock],
    as_of_date: date | None = None,
) -> int:
    """Ingest today's fundamental snapshot for a list of stocks.

    Each stock gets exactly one row with ``date = as_of_date``.
    If a row for (stock_id, date) already exists it is skipped (idempotent).

    Uses Quarterly Intelligent Refresh (QIR) to minimise HTTP requests:
    - Stocks with no data this calendar quarter are fully scraped.
    - Stocks with a board meeting this week are fully scraped (fresh data).
    - All other stocks receive a copy-forward row from the most recent
      quarter data (zero HTTP requests).

    Scrapes are bounded by asyncio.Semaphore(3) over a single shared
    httpx.AsyncClient to avoid IP bans.

    Returns the number of new rows inserted.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    cur_year, cur_q = _current_quarter(as_of_date)
    quarter_start = date(cur_year, (cur_q - 1) * 3 + 1, 1)

    # ── 1. Stocks already ingested today (idempotent guard) ───────────
    today_result = await db.execute(
        select(StockFundamentalPIT.stock_id).where(
            StockFundamentalPIT.date == as_of_date
        )
    )
    already_today: set[int] = {row[0] for row in today_result.all()}
    pending = [s for s in stocks if s.id not in already_today]

    if not pending:
        _log.info("Fundamental PIT: all %d stocks already ingested for %s", len(stocks), as_of_date)
        return 0

    # ── 2. Stocks that already have a row this quarter ─────────────────
    quarter_result = await db.execute(
        select(StockFundamentalPIT.stock_id).where(
            StockFundamentalPIT.date >= quarter_start,
            StockFundamentalPIT.date < as_of_date,
        )
    )
    has_quarter_data: set[int] = {row[0] for row in quarter_result.all()}

    # ── 3. Earnings-flagged stocks (near board meetings) ──────────────
    earnings_ids = await asyncio.to_thread(_earnings_flagged_stocks, pending, as_of_date)

    # ── 4. QIR split ──────────────────────────────────────────────────
    need_scrape: list[Stock] = []
    copy_forward: list[Stock] = []
    for stock in pending:
        if stock.id not in has_quarter_data or stock.id in earnings_ids:
            need_scrape.append(stock)
        else:
            copy_forward.append(stock)

    _log.info(
        "Fundamental PIT QIR: %d scrape / %d copy-forward / %d skip (already today)",
        len(need_scrape),
        len(copy_forward),
        len(already_today),
    )

    inserted = 0

    # ── 5. Copy-forward rows (no HTTP) ────────────────────────────────
    if copy_forward:
        # Fetch the most recent row per stock_id from this quarter
        cf_ids = [s.id for s in copy_forward]
        recent_result = await db.execute(
            select(StockFundamentalPIT)
            .where(
                StockFundamentalPIT.stock_id.in_(cf_ids),
                StockFundamentalPIT.date >= quarter_start,
            )
            .order_by(StockFundamentalPIT.date.desc())
        )
        # Pick the most recent row per stock
        recent_by_stock: dict[int, StockFundamentalPIT] = {}
        for row in recent_result.scalars().all():
            if row.stock_id not in recent_by_stock:
                recent_by_stock[row.stock_id] = row

        for stock in copy_forward:
            src = recent_by_stock.get(stock.id)
            if src is None:
                # No quarter data found despite check — promote to scrape
                need_scrape.append(stock)
                continue
            db.add(StockFundamentalPIT(
                stock_id=stock.id,
                date=as_of_date,
                source="scrnr_copy_fwd",
                ingested_at=now_ist(),
                pe_ratio=src.pe_ratio,
                forward_pe=src.forward_pe,
                pb_ratio=src.pb_ratio,
                dividend_yield=src.dividend_yield,
                roe=src.roe,
                debt_equity=src.debt_equity,
            ))
            inserted += 1

    # ── 6. Live scrape (concurrent, semaphore-bounded) ─────────────────
    if need_scrape:
        semaphore = asyncio.Semaphore(3)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        async with httpx.AsyncClient(timeout=20.0, limits=limits) as client:
            async def _fetch_one(stock: Stock) -> tuple[Stock, dict]:
                async with semaphore:
                    metrics = await _scrape_screener(stock.symbol, client)
                    return stock, metrics

            results = await asyncio.gather(*[_fetch_one(s) for s in need_scrape])

        for stock, metrics in results:
            db.add(StockFundamentalPIT(
                stock_id=stock.id,
                date=as_of_date,
                source="screener_spider",
                ingested_at=now_ist(),
                **metrics,
            ))
            inserted += 1

    await db.commit()
    _log.info(
        "Fundamental PIT ingestion complete: %d new rows for date=%s "
        "(%d scraped, %d copy-forwarded)",
        inserted,
        as_of_date,
        len(need_scrape),
        inserted - len(need_scrape),
    )
    return inserted


async def compute_sector_stats(
    db: AsyncSession,
    as_of_date: date | None = None,
) -> None:
    """Compute sector-level PE averages for today and upsert into
    ``fundamental_sector_stats``.  Called after ``ingest_fundamentals_pit``."""
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    # Pull today's fundamentals joined with sector
    result = await db.execute(
        select(
            Stock.sector,
            StockFundamentalPIT.pe_ratio,
        )
        .join(StockFundamentalPIT, StockFundamentalPIT.stock_id == Stock.id)
        .where(StockFundamentalPIT.date == as_of_date)
        .where(StockFundamentalPIT.pe_ratio.isnot(None))
        .where(StockFundamentalPIT.pe_ratio > 0)
    )
    rows = result.all()
    if not rows:
        return

    df = pd.DataFrame(rows, columns=["sector", "pe_ratio"])
    grouped = df.groupby("sector")["pe_ratio"]

    for sector, pe_series in grouped:
        pe_arr = pe_series.dropna().values
        if len(pe_arr) < 2:
            continue
        avg = float(np.mean(pe_arr))
        std = float(np.std(pe_arr, ddof=1))

        existing = await db.execute(
            select(FundamentalSectorStats).where(
                FundamentalSectorStats.sector == sector,
                FundamentalSectorStats.date == as_of_date,
            )
        )
        row = existing.scalar_one_or_none()
        if row is None:
            row = FundamentalSectorStats(
                sector=sector,
                date=as_of_date,
                computed_at=now_ist(),
            )
            db.add(row)
        row.sector_pe_avg = avg
        row.sector_pe_std = std
        row.stock_count = int(len(pe_arr))
        row.computed_at = now_ist()

    await db.commit()


async def compute_valuation_zscores(
    db: AsyncSession,
    stock: Stock,
    as_of_date: date | None = None,
    lookback_years: int = 3,
) -> StockFundamentalZScore | None:
    """Compute and persist ML-ready z-scores for a single stock.

    - ``pe_zscore_3y``      : (today_PE - rolling_mean_3yr) / rolling_std_3yr, clipped ±3
    - ``pe_zscore_sector``  : (today_PE - sector_avg) / sector_std, clipped ±3
    - ``roe_norm``          : ROE clipped to [0, 1.0] (100%) — Screener stores as decimal (0.18 = 18%)
    - ``debt_equity_norm``  : D/E clipped to [0, 5], divided by 5, then inverted (1 = low debt)

    Returns None if insufficient data exists.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    cutoff = as_of_date - timedelta(days=lookback_years * 365)

    # Fetch 3-year history of PE for this stock
    result = await db.execute(
        select(StockFundamentalPIT.date, StockFundamentalPIT.pe_ratio)
        .where(StockFundamentalPIT.stock_id == stock.id)
        .where(StockFundamentalPIT.date >= cutoff)
        .where(StockFundamentalPIT.date <= as_of_date)
        .order_by(StockFundamentalPIT.date)
    )
    pit_rows = result.all()
    if not pit_rows:
        return None

    today_row = next((r for r in reversed(pit_rows) if r.date == as_of_date), None)
    if today_row is None or today_row.pe_ratio is None:
        return None

    today_pe = today_row.pe_ratio
    pe_history = [r.pe_ratio for r in pit_rows if r.pe_ratio is not None]

    # 3-year z-score
    pe_zscore_3y: float | None = None
    if len(pe_history) >= 10:
        mean_3y = float(np.mean(pe_history))
        std_3y = float(np.std(pe_history, ddof=1))
        if std_3y > 0:
            pe_zscore_3y = float(np.clip((today_pe - mean_3y) / std_3y, -3.0, 3.0))

    # Sector z-score (today's sector stats)
    pe_zscore_sector: float | None = None
    if stock.sector:
        sector_result = await db.execute(
            select(FundamentalSectorStats).where(
                FundamentalSectorStats.sector == stock.sector,
                FundamentalSectorStats.date == as_of_date,
            )
        )
        sector_stats = sector_result.scalar_one_or_none()
        if sector_stats and sector_stats.sector_pe_std and sector_stats.sector_pe_std > 0:
            pe_zscore_sector = float(
                np.clip(
                    (today_pe - sector_stats.sector_pe_avg) / sector_stats.sector_pe_std,
                    -3.0,
                    3.0,
                )
            )

    # ROE normalisation: stored as decimal (0.18 = 18%) — clamp to [0, 1]
    roe_norm: float | None = None
    roe_raw = today_row.pe_ratio  # NOTE: we re-fetch the full row below
    # Fetch full row for non-PE fields
    full_result = await db.execute(
        select(StockFundamentalPIT).where(
            StockFundamentalPIT.stock_id == stock.id,
            StockFundamentalPIT.date == as_of_date,
        )
    )
    full_row = full_result.scalar_one_or_none()
    if full_row and full_row.roe is not None:
        roe_norm = float(np.clip(full_row.roe, 0.0, 1.0))  # stored as 0–1 decimal

    debt_equity_norm: float | None = None
    if full_row and full_row.debt_equity is not None:
        # D/E: clipped to [0,5], /5 → 0-1, then invert so 1=low debt is "good"
        de_scaled = float(np.clip(full_row.debt_equity, 0.0, 5.0)) / 5.0
        debt_equity_norm = round(1.0 - de_scaled, 4)

    # Upsert z-score row
    existing = await db.execute(
        select(StockFundamentalZScore).where(
            StockFundamentalZScore.stock_id == stock.id,
            StockFundamentalZScore.date == as_of_date,
        )
    )
    z_row = existing.scalar_one_or_none()
    if z_row is None:
        z_row = StockFundamentalZScore(
            stock_id=stock.id,
            date=as_of_date,
            computed_at=now_ist(),
        )
        db.add(z_row)

    z_row.pe_zscore_3y = pe_zscore_3y
    z_row.pe_zscore_sector = pe_zscore_sector
    z_row.roe_norm = roe_norm
    z_row.debt_equity_norm = debt_equity_norm
    z_row.computed_at = now_ist()

    await db.commit()
    return z_row


async def ingest_and_score(
    db: AsyncSession,
    stocks: list[Stock],
    as_of_date: date | None = None,
) -> dict:
    """Full pipeline: ingest PIT → compute sector stats → compute z-scores.

    Returns a summary dict suitable for the API response.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    inserted = await ingest_fundamentals_pit(db, stocks, as_of_date)
    await compute_sector_stats(db, as_of_date)

    scored = 0
    for stock in stocks:
        z = await compute_valuation_zscores(db, stock, as_of_date)
        if z is not None:
            scored += 1

    return {
        "date": as_of_date.isoformat(),
        "total_stocks": len(stocks),
        "pit_rows_inserted": inserted,
        "zscores_computed": scored,
    }


# ── Internal helpers ───────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/Inf/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
