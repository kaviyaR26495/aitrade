import asyncio
from app.db.database import async_session_factory
from app.db import crud
from sqlalchemy import select
from app.db.models import RLModel, KNNModel, LSTMModel, EnsembleConfig

async def fix_ensemble():
    async with async_session_factory() as db:
        # Check if we already have a config
        existing = await db.execute(select(EnsembleConfig))
        if len(existing.scalars().all()) > 0:
            print("Ensemble configuration already exists.")
            return

        # Find latest models
        rl = (await db.execute(select(RLModel).order_by(RLModel.id.desc()))).scalars().first()
        knn = (await db.execute(select(KNNModel).order_by(KNNModel.id.desc()))).scalars().first()
        lstm = (await db.execute(select(LSTMModel).order_by(LSTMModel.id.desc()))).scalars().first()

        if not (rl and knn and lstm):
            print(f"Missing models: RL={rl}, KNN={knn}, LSTM={lstm}")
            return

        print(f"Linking RL#{rl.id}, KNN#{knn.id}, LSTM#{lstm.id}...")
        
        ensemble = await crud.create_ensemble_config(
            db,
            name=f"Ensemble_{rl.name}",
            knn_model_id=knn.id,
            lstm_model_id=lstm.id,
            knn_weight=0.5,
            lstm_weight=0.5,
            agreement_required=True,
            interval=knn.interval or "day",
        )
        print(f"Created EnsembleConfig #{ensemble.id}: {ensemble.name}")

if __name__ == "__main__":
    asyncio.run(fix_ensemble())
