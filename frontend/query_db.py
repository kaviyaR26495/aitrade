import asyncio
from backend.app.db.database import async_session_factory
from backend.app.db import crud

async def main():
    async with async_session_factory() as db:
        print(await crud.get_setting(db, "last_auto_retrain_at"))

asyncio.run(main())
