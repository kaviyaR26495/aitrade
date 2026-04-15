import asyncio
from sqlalchemy import select
from app.db.database import async_session_factory
from app.db.models import EnsemblePrediction
from datetime import date

async def check():
    async with async_session_factory() as db:
        q = select(EnsemblePrediction).order_by(EnsemblePrediction.date.desc()).limit(10)
        res = await db.execute(q)
        preds = res.scalars().all()
        if not preds:
            print("No predictions found!")
            return
        for p in preds:
            print(f"Date: {p.date}, ID: {p.id}, Symbol ID: {p.stock_id}, Regime: {p.regime_id}")

if __name__ == "__main__":
    asyncio.run(check())
