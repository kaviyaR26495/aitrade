import asyncio
from app.db.database import async_session_factory
from app.ml.predictor import run_target_price_predictions

async def run():
    async with async_session_factory() as db:
        res = await run_target_price_predictions(db, stock_ids=[144]) # HDFCBANK or some known stock id
        print("Created:", res.get("created"))
        print(res.get("rejected", [])[:5])
        print(res.get("errors", [])[:5])

asyncio.run(run())
