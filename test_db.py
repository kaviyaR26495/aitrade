import asyncio
from app.db.database import async_session_factory
from app.db import crud
from app.ml.rl_trainer import _run_training_background
import math

async def test():
    async with async_session_factory() as db:
        await crud.update_rl_model_completed(
            db,
            model_id=1, # assuming it doesn't matter or fails fast
            total_reward=-math.inf,
            sharpe_ratio=math.nan,
            model_path="/tmp/test"
        )
        print("Success")

asyncio.run(test())
