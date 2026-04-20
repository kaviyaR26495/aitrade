import asyncio
from app.db.database import async_session_factory
from app.ml.predictor import run_target_price_predictions
from sqlalchemy import text

async def run():
    async with async_session_factory() as db:
        # Check stock ids
        res = await db.execute(text("SELECT id FROM stocks LIMIT 2"))
        ids = [r[0] for r in res.fetchall()]
        print("Testing ids:", ids)
        
        try:
            res = await run_target_price_predictions(db, stock_ids=ids)
            print("CREATED:", res.get("created"))
            print("REJECTED:", len(res.get("rejected", [])))
            print("REASONS:", set(r.get("reason", "") for r in res.get("rejected", [])))
        except Exception as e:
            print("ERROR", e)

asyncio.run(run())
