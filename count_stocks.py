import asyncio
from app.db.database import async_session_factory
from app.db.models import Stock
from sqlalchemy import select

async def run():
    async with async_session_factory() as db:
        res = await db.execute(select(Stock).where(Stock.is_active == True))
        print("Active stocks:", len(res.scalars().all()))

asyncio.run(run())
