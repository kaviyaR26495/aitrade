"""Historical prediction backfill script.

Runs the KNN+LSTM ensemble on each trading day in a date range and stores the
results in ensemble_predictions — exactly as the daily Celery job does — so
the compound backtester has signals for dates before April 2026.

Usage (run from backend/):
    python scripts/backfill_predictions.py
    python scripts/backfill_predictions.py --start 2025-01-01 --end 2026-03-31
    python scripts/backfill_predictions.py --start 2025-01-01 --ensemble-config-id 40
    python scripts/backfill_predictions.py --start 2024-01-01 --no-skip-existing
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import date, datetime, timedelta

from sqlalchemy import select, text, func, distinct

# ── bootstrap path so we can import app.* ────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import async_session_factory
from app.db.models import EnsemblePrediction, EnsembleConfig, StockOHLCV
from app.ml.predictor import run_daily_predictions

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,   # suppress SQLAlchemy noise
)
# Show our own progress messages at INFO
logger = logging.getLogger("backfill")
logger.setLevel(logging.INFO)
logging.getLogger("app.ml.predictor").setLevel(logging.WARNING)
logging.getLogger("app.core").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


async def get_trading_dates(start: date, end: date) -> list[date]:
    """Return the distinct trading days in [start, end] that have OHLCV data."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(distinct(StockOHLCV.date))
            .where(StockOHLCV.date >= start, StockOHLCV.date <= end, StockOHLCV.interval == "day")
            .order_by(StockOHLCV.date)
        )
        return [row[0] for row in result.all()]


async def get_dates_with_predictions(dates: list[date], ensemble_config_id: int) -> set[date]:
    """Return the subset of *dates* already covered by the given ensemble config."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(distinct(EnsemblePrediction.date))
            .where(
                EnsemblePrediction.date.in_(dates),
                EnsemblePrediction.ensemble_config_id == ensemble_config_id,
            )
        )
        return {row[0] for row in result.all()}


async def get_latest_ensemble_config_id() -> int:
    """Return the id of the most recently created ensemble config."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(EnsembleConfig.id)
            .order_by(EnsembleConfig.created_at.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise RuntimeError("No EnsembleConfig found in DB. Train a model first.")
        return row


async def run_backfill(
    start: date,
    end: date,
    ensemble_config_id: int,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    logger.info("=" * 60)
    logger.info("Backfill: %s → %s  (ensemble_config=%d)", start, end, ensemble_config_id)
    logger.info("skip_existing=%s  dry_run=%s", skip_existing, dry_run)
    logger.info("=" * 60)

    # 1. Discover all trading dates in range
    logger.info("Querying trading dates from stock_ohlcv ...")
    all_dates = await get_trading_dates(start, end)
    if not all_dates:
        logger.error("No OHLCV trading dates found for range %s → %s", start, end)
        return
    logger.info("Found %d trading days in range", len(all_dates))

    # 2. Filter out already-done dates
    dates_to_process = list(all_dates)
    if skip_existing:
        already_done = await get_dates_with_predictions(all_dates, ensemble_config_id)
        dates_to_process = [d for d in all_dates if d not in already_done]
        skipped = len(all_dates) - len(dates_to_process)
        if skipped:
            logger.info("Skipping %d dates that already have predictions", skipped)
    logger.info("Dates to process: %d", len(dates_to_process))

    if not dates_to_process:
        logger.info("Nothing to do — all dates already covered.")
        return

    if dry_run:
        logger.info("[DRY RUN] Would process: %s ... %s", dates_to_process[0], dates_to_process[-1])
        return

    # 3. Process each date
    total = len(dates_to_process)
    t_start = time.monotonic()

    for idx, target_date in enumerate(dates_to_process, start=1):
        t0 = time.monotonic()

        # ETA estimate
        elapsed = t0 - t_start
        avg_sec = elapsed / (idx - 1) if idx > 1 else 0
        remaining = avg_sec * (total - idx + 1)
        eta_str = str(timedelta(seconds=int(remaining))) if idx > 1 else "?"

        logger.info(
            "[%d/%d] %s  (elapsed=%s  ETA=%s)",
            idx, total, target_date,
            str(timedelta(seconds=int(elapsed))),
            eta_str,
        )

        try:
            async with async_session_factory() as db:
                result = await run_daily_predictions(
                    db=db,
                    target_date=target_date,
                    interval="day",
                    ensemble_config_id=ensemble_config_id,
                    # Disable live-trading guards — not relevant for historical backfill
                    sector_guard=False,
                    weekly_confluence_filter=False,
                )
            saved = result.get("predictions_made", 0)
            errors = result.get("errors", 0) if isinstance(result.get("errors"), int) else len(result.get("error_details", []))
            duration = time.monotonic() - t0
            logger.info(
                "  ✓ date=%s  saved=%s  errors=%s  %.1fs",
                target_date, saved, errors, duration,
            )
        except Exception as exc:
            logger.error("  ✗ date=%s  FAILED: %s", target_date, exc, exc_info=True)
            # Continue to next date rather than aborting the whole backfill
            continue

    total_elapsed = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info(
        "Backfill complete: %d dates in %s",
        len(dates_to_process),
        str(timedelta(seconds=int(total_elapsed))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical ensemble predictions")
    parser.add_argument(
        "--start", type=date.fromisoformat, default=date(2025, 1, 1),
        metavar="YYYY-MM-DD", help="Start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=date(2026, 3, 31),
        metavar="YYYY-MM-DD", help="End date (default: 2026-03-31)",
    )
    parser.add_argument(
        "--ensemble-config-id", type=int, default=None,
        help="Ensemble config ID to use (default: latest)",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false", default=True,
        help="Re-process dates that already have predictions",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just show what would be processed, don't write to DB",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Resolve ensemble config
    if args.ensemble_config_id is None:
        cfg_id = await get_latest_ensemble_config_id()
        logger.info("Using latest ensemble config: id=%d", cfg_id)
    else:
        cfg_id = args.ensemble_config_id

    await run_backfill(
        start=args.start,
        end=args.end,
        ensemble_config_id=cfg_id,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
