import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import async_session_factory
from app.ml.predictor import run_target_price_predictions

async def run():
    async with async_session_factory() as db:
        res = await run_target_price_predictions(db, stock_ids=[144, 145])
        print("Created:", res.get("created"))
        print("Rejected:", len(res.get("rejected", [])))
        print("Rejection reasons sample:", res.get("rejected", [])[:20])
        print("Errors:", res.get("errors", [])[:5])

asyncio.run(run())
