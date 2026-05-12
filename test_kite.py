import asyncio
from app.db.database import async_session_factory
from app.db import crud
from app.core.zerodha import get_kite

async def main():
    async with async_session_factory() as db:
        token = await crud.get_setting(db, "KITE_ACCESS_TOKEN")
        print("Token:", token)
        kite = get_kite()
        kite.set_access_token(token)
        try:
            h = kite.holdings()
            print("Holdings:", h)
        except Exception as e:
            print("Error type:", type(e))
            print("Error details:", str(e))
            print("Dir:", dir(e))

if __name__ == "__main__":
    asyncio.run(main())
