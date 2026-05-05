import asyncio
from app.api.deps import get_db
from app.db.database import async_session
from sqlalchemy import text

async def main():
    async with async_session() as db:
        res = await db.execute(text("SELECT date FROM stock_ohlcv LIMIT 1"))
        d1 = res.scalar()
        print("OHLCV date type:", type(d1), d1)

        res2 = await db.execute(text("SELECT date FROM ensemble_predictions LIMIT 1"))
        d2 = res2.scalar()
        print("Ensemble date type:", type(d2), d2)
        
        res3 = await db.execute(text("SELECT action, confidence FROM ensemble_predictions WHERE action=1 LIMIT 5"))
        print("Actions=1:", res3.fetchall())

asyncio.run(main())
