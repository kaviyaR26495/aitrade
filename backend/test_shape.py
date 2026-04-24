import asyncio
from datetime import date
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.db.database import async_session_factory
from app.api.routes.backtest import _run_live_inference
from app.config import settings

async def main():
    async with async_session_factory() as session:
        try:
            # mock test for _run_live_inference
            dates = [date(2026, 4, d) for d in range(1, 23)]
            pred = await _run_live_inference(session, 3, 'day', dates, date(2026,4,1), date(2026,4,22))
            print('Preds:', len(pred))
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(main())
