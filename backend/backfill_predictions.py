"""
backfill_predictions.py
=======================
Checks ensemble_predictions coverage for Apr 1–30 2026 (all trading days
found in the OHLCV table), generates + stores predictions for any missing
dates, then calls the compound backtest API and prints the report.

Usage (from backend/ dir with venv active):
    python backfill_predictions.py
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import select, func

from app.db.database import async_session_factory
from app.db.models import EnsemblePrediction, StockOHLCV, Stock
from app.ml import predictor

START = date(2026, 4, 1)
END   = date(2026, 4, 30)


# ── helpers ──────────────────────────────────────────────────────────────────

async def get_trading_dates(db) -> list[date]:
    """All distinct dates present in OHLCV for the requested range."""
    res = await db.execute(
        select(StockOHLCV.date)
        .where(StockOHLCV.date >= START, StockOHLCV.date <= END)
        .distinct()
        .order_by(StockOHLCV.date)
    )
    return [r for (r,) in res.all()]


async def get_prediction_coverage(db, trading_dates: list[date]) -> dict[date, int]:
    """Returns {date: stock_count} for predictions already in DB."""
    if not trading_dates:
        return {}
    res = await db.execute(
        select(EnsemblePrediction.date, func.count(EnsemblePrediction.id))
        .where(EnsemblePrediction.date.in_(trading_dates))
        .group_by(EnsemblePrediction.date)
    )
    return {r.date: r[1] for r in res.all()}


async def get_universe_count(db) -> int:
    from app.core.data_service import get_universe_stocks
    stocks = await get_universe_stocks(db)
    return len(stocks)


# ── main logic ────────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*60}")
    print(f"  Backfill & Backtest: {START}  →  {END}")
    print(f"{'='*60}\n")

    async with async_session_factory() as db:
        trading_dates = await get_trading_dates(db)

    if not trading_dates:
        print("❌  No OHLCV data found for Apr 2026. Please sync OHLCV data first.")
        return

    print(f"📅  Trading days in OHLCV: {len(trading_dates)}")
    for d in trading_dates:
        print(f"     {d}")

    # ── Check coverage ────────────────────────────────────────────────────
    async with async_session_factory() as db:
        coverage = await get_prediction_coverage(db, trading_dates)
        universe_count = await get_universe_count(db)

    print(f"\n🌐  Universe stocks: {universe_count}")
    print(f"\n📊  Existing prediction coverage:")

    missing_dates: list[date] = []
    for d in trading_dates:
        cnt = coverage.get(d, 0)
        pct = (cnt / universe_count * 100) if universe_count else 0
        status = "✅" if pct >= 50 else ("⚠️ " if cnt > 0 else "❌")
        print(f"     {status}  {d}  →  {cnt}/{universe_count} stocks  ({pct:.0f}%)")
        if pct < 50:
            missing_dates.append(d)

    if not missing_dates:
        print("\n✅  All trading days have sufficient predictions. Skipping generation.")
    else:
        print(f"\n🔄  Generating predictions for {len(missing_dates)} date(s) with <50% coverage...")
        for target_date in missing_dates:
            print(f"\n   ▶  Generating for {target_date} ...")
            try:
                async with async_session_factory() as db:
                    result = await predictor.run_daily_predictions(
                        db,
                        target_date=target_date,
                        interval="day",
                        agreement_required=False,  # looser gate so more signals populate
                    )
                n = len(result.get("results", []))
                print(f"      ✅  {target_date}: {n} predictions generated & stored.")
            except Exception as exc:
                print(f"      ❌  {target_date}: generation failed — {exc}")

        # Re-check after backfill
        print("\n📊  Coverage after backfill:")
        async with async_session_factory() as db:
            coverage2 = await get_prediction_coverage(db, trading_dates)
        for d in trading_dates:
            cnt = coverage2.get(d, 0)
            pct = (cnt / universe_count * 100) if universe_count else 0
            status = "✅" if pct >= 50 else ("⚠️ " if cnt > 0 else "❌")
            print(f"     {status}  {d}  →  {cnt}/{universe_count} stocks  ({pct:.0f}%)")

    # ── Run compound backtest via HTTP API ────────────────────────────────
    print(f"\n🚀  Running compound backtest via API  ({START} → {END}) ...")
    import httpx
    payload = {
        "start_date": str(START),
        "end_date": str(END),
        "initial_capital": 100000.0,
        "stoploss_pct": 5.0,
        "target_pct": 10.0,
        "min_confidence": 0.50,
        "max_positions_per_day": 10,
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post("http://localhost:8000/api/backtest/run-compound", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            initial = payload["initial_capital"]
            final = data.get("final_capital", 0)
            profit = data.get("profit_booked", 0)
            profit_pct = (profit / initial * 100) if initial else 0
            n_trades = data.get("total_trades", 0)
            trade_log = data.get("trade_log", [])

            print(f"\n{'='*60}")
            print(f"  BACKTEST REPORT  (ID: {data.get('id', '?')})")
            print(f"{'='*60}")
            print(f"  Period           : {START}  →  {END}")
            print(f"  Initial Capital  : ₹{initial:,.2f}")
            print(f"  Final Capital    : ₹{final:,.2f}")
            print(f"  Profit Booked    : ₹{profit:,.2f}")
            print(f"  Profit %         : {profit_pct:.2f}%")
            print(f"  Total Trades     : {n_trades}")

            if trade_log:
                # Build daily transaction summary
                daily: dict[str, list] = defaultdict(list)
                for t in trade_log:
                    daily[t["entry_date"]].append(("BUY",  t["symbol"], t["entry_price"], t["quantity"]))
                    daily[t["exit_date"]].append( ("SELL", t["symbol"], t["exit_price"],  t["quantity"], t["pnl"]))

                print(f"\n{'─'*60}")
                print(f"  DAY-BY-DAY TRANSACTIONS")
                print(f"{'─'*60}")
                for day in sorted(daily.keys()):
                    print(f"\n  📅  {day}")
                    for tx in daily[day]:
                        if tx[0] == "BUY":
                            _, sym, price, qty = tx
                            val = price * qty
                            print(f"       🟢  BUY   {sym:15s}  {qty:5d} @ ₹{price:,.2f}  =  ₹{val:,.2f}")
                        else:
                            _, sym, price, qty, pnl = tx
                            val = price * qty
                            pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
                            print(f"       🔴  SELL  {sym:15s}  {qty:5d} @ ₹{price:,.2f}  =  ₹{val:,.2f}  P&L: {pnl_str}")

                # Holdings at end (exit_reason == end_of_backtest)
                holdings = [t for t in trade_log if t.get("exit_reason") == "end_of_backtest"]
                if holdings:
                    holdings_val = sum(t["exit_price"] * t["quantity"] for t in holdings)
                    print(f"\n  {'─'*58}")
                    print(f"  📦  HOLDINGS AT END (still open — valued at end-date close)")
                    print(f"  {'─'*58}")
                    for h in holdings:
                        val = h["exit_price"] * h["quantity"]
                        print(f"       {h['symbol']:15s}  {h['quantity']:5d} @ ₹{h['exit_price']:,.2f}  =  ₹{val:,.2f}")
                    print(f"       {'─'*40}")
                    print(f"       Total holdings value : ₹{holdings_val:,.2f}")
            else:
                print("\n  ⚠️  No trades were executed.")
                print("       Possible reasons:")
                print("       • Predictions have action=HOLD or low confidence")
                print("       • No BUY signals matched regime/confidence filters")
                print("       • OHLCV data missing for Apr 2026 stocks")

            debug = data.get("debug", {})
            if debug:
                print(f"\n  {'─'*58}")
                print(f"  DEBUG")
                print(f"       OHLCV dates found      : {debug.get('all_dates', '?')}")
                print(f"       Prediction dates found : {debug.get('preds_map_keys', '?')}")
                print(f"       Total predictions      : {debug.get('preds_count', '?')}")
        else:
            print(f"❌  API returned {resp.status_code}: {resp.text[:500]}")
    except Exception as exc:
        print(f"❌  Backtest API call failed: {exc}")
        raise

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
