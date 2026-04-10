"""Market context enrichment — FII/DII flow and sector breadth.

Adds three leading-indicator columns that go beyond per-stock lagging indicators:
  fii_net_norm       — FII net trade rolling z-score (institutional buying pressure)
  dii_net_norm       — DII net trade rolling z-score (domestic buffer / outflow)
  sector_breadth_pct — fraction of sector stocks trading above SMA-50 (0–1)

All I/O paths are non-fatal: failures fill with neutral defaults so the rest of
the feature pipeline remains unaffected.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.db import crud

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.db.models import Stock

logger = logging.getLogger(__name__)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _parse_nse_date(s: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    m = series.rolling(window, min_periods=5).mean()
    s = series.rolling(window, min_periods=5).std().replace(0, 1.0)
    return ((series - m) / s).fillna(0.0).clip(-3.0, 3.0)


async def fetch_fii_dii(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch FII/DII net trade from NSE API.

    Returns a DataFrame with columns: date, fii_net, dii_net,
    fii_net_norm, dii_net_norm.  Falls back to zeros on any error.
    """
    zero_index = pd.bdate_range(start=start_date, end=end_date)
    zero_df = pd.DataFrame({
        "date": zero_index.date,
        "fii_net": 0.0,
        "dii_net": 0.0,
        "fii_net_norm": 0.0,
        "dii_net_norm": 0.0,
    })

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Prime session cookie (NSE requires a prior visit to the homepage)
            await client.get("https://www.nseindia.com/", headers=_NSE_HEADERS)
            resp = await client.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                headers=_NSE_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

        rows = []
        for item in data:
            raw_date = (
                item.get("tradeDate")
                or item.get("date")
                or item.get("DD-MMM-YYYY", "")
            )
            d = _parse_nse_date(str(raw_date))
            if d is None:
                continue
            fii = _safe_float(
                item.get("fiiNetTrade") or item.get("fii_net") or item.get("FII_NET")
            )
            dii = _safe_float(
                item.get("diiNetTrade") or item.get("dii_net") or item.get("DII_NET")
            )
            rows.append({"date": d, "fii_net": fii, "dii_net": dii})

        if not rows:
            return zero_df

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df["fii_net_norm"] = _rolling_zscore(df["fii_net"])
        df["dii_net_norm"] = _rolling_zscore(df["dii_net"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df.reset_index(drop=True)

    except Exception as exc:
        logger.warning("FII/DII fetch failed (%s); using zero fill.", exc)
        return zero_df


async def compute_sector_breadth(
    db: "AsyncSession",
    sector: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fraction of sector stocks trading above their SMA-50 per trading day.

    Returns DataFrame with columns: date, sector_breadth_pct.
    Falls back to a constant 0.5 when fewer than 3 sector stocks are available.
    """
    stocks = await crud.get_stocks_by_sector(db, sector)

    bdate_index = pd.bdate_range(start=start_date, end=end_date)
    fallback_df = pd.DataFrame({
        "date": bdate_index.date,
        "sector_breadth_pct": 0.5,
    })

    if len(stocks) < 3:
        return fallback_df

    date_above: dict[date, int] = {}
    date_total: dict[date, int] = {}

    for stock in stocks:
        try:
            ind_rows = await crud.get_indicators(db, stock.id, "day", start_date, end_date)
            for row in ind_rows:
                d = row.date if isinstance(row.date, date) else row.date.date()
                date_total[d] = date_total.get(d, 0) + 1
                close_val = getattr(row, "close", None)
                sma50_val = getattr(row, "sma_50", None)
                if close_val is not None and sma50_val is not None and sma50_val > 0:
                    if close_val > sma50_val:
                        date_above[d] = date_above.get(d, 0) + 1
        except Exception as exc:
            logger.debug("Skipping stock %s for sector breadth: %s", stock.symbol, exc)
            continue

    if not date_total:
        return fallback_df

    dates_sorted = sorted(date_total.keys())
    breadth_vals = [date_above.get(d, 0) / date_total[d] for d in dates_sorted]
    df = pd.DataFrame({"date": dates_sorted, "sector_breadth_pct": breadth_vals})
    df["sector_breadth_pct"] = (
        df["sector_breadth_pct"].rolling(5, min_periods=1).mean()
    )
    return df.reset_index(drop=True)


async def enrich_with_market_context(
    df: pd.DataFrame,
    db: "AsyncSession",
    stock: "Stock",
) -> pd.DataFrame:
    """Attach fii_net_norm, dii_net_norm, sector_breadth_pct to df.

    All enrichment steps are non-fatal — failures fill with neutral defaults:
      fii_net_norm / dii_net_norm  → 0.0
      sector_breadth_pct           → 0.5

    Expects df to have a 'date' column (datetime-compatible or date objects).
    """
    df = df.copy()
    df["fii_net_norm"] = 0.0
    df["dii_net_norm"] = 0.0
    df["sector_breadth_pct"] = 0.5
    df["rs_5d"] = 0.0
    df["sector_roc_avg"] = 0.0
    df["market_above_sma200"] = 0.5

    date_col = pd.to_datetime(df["date"]).dt.date
    if date_col.empty:
        return df

    start_date = date_col.min()
    end_date   = date_col.max()

    # ── FII / DII ──────────────────────────────────────────────────────
    try:
        fii_df = await fetch_fii_dii(start_date, end_date)
        if not fii_df.empty:
            fii_df["date"] = pd.to_datetime(fii_df["date"])
            df["_date_dt"] = pd.to_datetime(df["date"])
            merged = df.merge(
                fii_df[["date", "fii_net_norm", "dii_net_norm"]],
                left_on="_date_dt",
                right_on="date",
                how="left",
                suffixes=("", "_fii"),
            )
            for col in ("fii_net_norm", "dii_net_norm"):
                src = f"{col}_fii" if f"{col}_fii" in merged.columns else col
                if src in merged.columns:
                    df[col] = merged[src].fillna(0.0).values
            df = df.drop(columns=["_date_dt"], errors="ignore")
    except Exception as exc:
        logger.warning("FII/DII enrichment failed (%s); using zeros.", exc)
        df = df.drop(columns=["_date_dt"], errors="ignore")

    # ── Sector Breadth ─────────────────────────────────────────────────
    sector = getattr(stock, "sector", None)
    if sector:
        try:
            breadth_df = await compute_sector_breadth(db, sector, start_date, end_date)
            if not breadth_df.empty:
                breadth_df["date"] = pd.to_datetime(breadth_df["date"])
                df["_date_dt"] = pd.to_datetime(df["date"])
                merged = df.merge(
                    breadth_df[["date", "sector_breadth_pct"]],
                    left_on="_date_dt",
                    right_on="date",
                    how="left",
                    suffixes=("", "_sb"),
                )
                src = (
                    "sector_breadth_pct_sb"
                    if "sector_breadth_pct_sb" in merged.columns
                    else "sector_breadth_pct"
                )
                if src in merged.columns:
                    df["sector_breadth_pct"] = merged[src].fillna(0.5).values
                df = df.drop(columns=["_date_dt"], errors="ignore")
        except Exception as exc:
            logger.warning("Sector breadth enrichment failed (%s); using 0.5.", exc)
            df = df.drop(columns=["_date_dt"], errors="ignore")

    # ── Relative Strength to benchmark ────────────────────────────────
    try:
        rs = await compute_rs_5d(db, df, settings.BENCHMARK_SYMBOL)
        df["rs_5d"] = pd.Series(rs, index=df.index).fillna(0.0).values
    except Exception as exc:
        logger.warning("rs_5d enrichment failed (%s); using 0.0.", exc)

    # ── Sector average ROC ─────────────────────────────────────────────
    sector = getattr(stock, "sector", None)
    if sector:
        try:
            sroc_df = await compute_sector_roc_avg(db, sector, start_date, end_date)
            if not sroc_df.empty:
                sroc_df["date"] = pd.to_datetime(sroc_df["date"])
                df["_date_dt"] = pd.to_datetime(df["date"])
                merged = df.merge(sroc_df[["date", "sector_roc_avg"]], left_on="_date_dt", right_on="date", how="left", suffixes=("", "_sr"))
                src = "sector_roc_avg_sr" if "sector_roc_avg_sr" in merged.columns else "sector_roc_avg"
                if src in merged.columns:
                    df["sector_roc_avg"] = merged[src].fillna(0.0).values
                df = df.drop(columns=["_date_dt"], errors="ignore")
        except Exception as exc:
            logger.warning("sector_roc_avg enrichment failed (%s); using 0.0.", exc)
            df = df.drop(columns=["_date_dt"], errors="ignore")

    # ── Market regime flag (index above 200-SMA) ───────────────────────
    try:
        mrf_df = await compute_market_regime_flag(db, start_date, end_date, settings.BENCHMARK_SYMBOL)
        if not mrf_df.empty:
            mrf_df["date"] = pd.to_datetime(mrf_df["date"])
            df["_date_dt"] = pd.to_datetime(df["date"])
            merged = df.merge(mrf_df[["date", "market_above_sma200"]], left_on="_date_dt", right_on="date", how="left", suffixes=("", "_mrf"))
            src = "market_above_sma200_mrf" if "market_above_sma200_mrf" in merged.columns else "market_above_sma200"
            if src in merged.columns:
                df["market_above_sma200"] = merged[src].fillna(0.5).values
            df = df.drop(columns=["_date_dt"], errors="ignore")
    except Exception as exc:
        logger.warning("market_above_sma200 enrichment failed (%s); using 0.5.", exc)
        df = df.drop(columns=["_date_dt"], errors="ignore")

    return df


# ── New contextual features ────────────────────────────────────────────

async def compute_rs_5d(
    db: "AsyncSession",
    df: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.Series:
    """Relative Strength vs benchmark over 5 trading days.

    ``rs_5d = stock_roc_5 - benchmark_roc_5``

    Measures whether a stock is generating alpha independent of the market
    tide.  Clipped to ±20 to cap influence of extreme one-day gaps.

    Requires 'roc_5' column in *df* (computed by indicators.py).
    Returns a Series aligned to df.index; falls back to 0.0 when data
    is unavailable.
    """
    if "roc_5" not in df.columns:
        return pd.Series(0.0, index=df.index)

    date_col = pd.to_datetime(df["date"]).dt.date
    start_date = date_col.min()
    end_date = date_col.max()

    try:
        rows = await crud.get_index_ohlcv(db, benchmark_symbol, "day", start_date, end_date)
        if len(rows) < 6:
            return pd.Series(0.0, index=df.index)

        bdf = pd.DataFrame([{"date": r.date, "close": r.close} for r in rows])
        bdf = bdf.sort_values("date").reset_index(drop=True)
        bdf["bench_roc_5"] = bdf["close"].pct_change(5) * 100

        bdf["date"] = pd.to_datetime(bdf["date"])
        df_dt = pd.to_datetime(df["date"])
        merged = pd.Series(df.index).map(dict(zip(df_dt, df.index)))  # identity
        tmp = pd.DataFrame({"date": df_dt, "roc_5": df["roc_5"].values}).merge(
            bdf[["date", "bench_roc_5"]], on="date", how="left"
        )
        rs = (tmp["roc_5"] - tmp["bench_roc_5"].fillna(0.0)).clip(-20, 20)
        return rs.values
    except Exception as exc:
        logger.warning("rs_5d computation failed (%s); using 0.0.", exc)
        return pd.Series(0.0, index=df.index)


async def compute_sector_roc_avg(
    db: "AsyncSession",
    sector: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Average 5-day ROC across all sector peers per trading day.

    Captures sector-level momentum: a stock inside a rising sector has
    tailwind even if its own ROC_5 is temporarily flat.

    Returns DataFrame with columns: date, sector_roc_avg.
    Fallback: 0.0 when fewer than 3 sector stocks are available or on error.
    """
    fallback_df = pd.DataFrame({
        "date": pd.bdate_range(start=start_date, end=end_date).date,
        "sector_roc_avg": 0.0,
    })

    stocks = await crud.get_stocks_by_sector(db, sector)
    if len(stocks) < 3:
        return fallback_df

    date_vals: dict[date, list[float]] = {}
    for stock in stocks:
        try:
            ind_rows = await crud.get_indicators(db, stock.id, "day", start_date, end_date)
            for row in ind_rows:
                d = row.date if isinstance(row.date, date) else row.date.date()
                val = getattr(row, "roc_5", None)
                if val is not None and not pd.isna(val):
                    date_vals.setdefault(d, []).append(float(val))
        except Exception as exc:
            logger.debug("Skipping stock %s for sector_roc_avg: %s", stock.symbol, exc)

    if not date_vals:
        return fallback_df

    dates_sorted = sorted(date_vals.keys())
    avgs = [sum(date_vals[d]) / len(date_vals[d]) for d in dates_sorted]
    out = pd.DataFrame({"date": dates_sorted, "sector_roc_avg": avgs})
    out["sector_roc_avg"] = out["sector_roc_avg"].rolling(5, min_periods=1).mean()
    return out.reset_index(drop=True)


async def compute_market_regime_flag(
    db: "AsyncSession",
    start_date: date,
    end_date: date,
    benchmark_symbol: str,
) -> pd.DataFrame:
    """Binary flag: 1.0 when benchmark close > its 200-day SMA, else 0.0.

    A stock-picking model should know whether it is operating in a bull market
    (index above 200-SMA) or a bear market — independent of the individual
    stock's own trend indicators.

    Returns DataFrame with columns: date, market_above_sma200.
    Fallback: 0.5 (neutral / unknown) when fewer than 200 bars are available.
    """
    bdate_index = pd.bdate_range(start=start_date, end=end_date)
    fallback_df = pd.DataFrame({
        "date": bdate_index.date,
        "market_above_sma200": 0.5,
    })

    try:
        # Fetch extra history so the 200-SMA is warm by start_date
        from datetime import timedelta
        fetch_start = start_date - timedelta(days=300)
        rows = await crud.get_index_ohlcv(db, benchmark_symbol, "day", fetch_start, end_date)
        if len(rows) < 201:
            return fallback_df

        bdf = pd.DataFrame([{"date": r.date, "close": r.close} for r in rows])
        bdf = bdf.sort_values("date").reset_index(drop=True)
        bdf["sma_200"] = bdf["close"].rolling(200, min_periods=200).mean()
        bdf["market_above_sma200"] = np.where(bdf["close"] > bdf["sma_200"], 1.0, 0.0)
        # NaN rows (before 200 bars) → neutral 0.5
        bdf.loc[bdf["sma_200"].isna(), "market_above_sma200"] = 0.5

        out = bdf[bdf["date"] >= start_date][["date", "market_above_sma200"]].reset_index(drop=True)
        return out
    except Exception as exc:
        logger.warning("market_above_sma200 computation failed (%s); using 0.5.", exc)
        return fallback_df
