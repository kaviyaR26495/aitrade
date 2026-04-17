import asyncio
from backend.app.db.database import async_session_factory
from backend.app.db import crud
from sqlalchemy import select
from backend.app.db.models import EnsembleConfig

async def main():
    async with async_session_factory() as db:
        res = await db.execute(select(EnsembleConfig.created_at).order_by(EnsembleConfig.created_at.desc()).limit(1))
        print("Ensemble:", res.scalar_one_or_none())
        print("Setting:", await crud.get_setting(db, "last_auto_retrain_at"))

asyncio.run(main())
