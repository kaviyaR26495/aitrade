import asyncio
from app.db.database import engine, Base
from app.db import models  # VERY IMPORTANT (loads tables)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_tables())