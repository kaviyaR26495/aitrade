import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.db.models import Stock, EnsembleConfig, RLModel, KNNModel, LSTMModel
from app.config import settings

async def main():
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with async_session() as session:
        stocks = await session.execute(select(Stock).limit(5))
        print("Stocks:", [(s.id, s.symbol) for s in stocks.scalars()])

        ensembles = await session.execute(select(EnsembleConfig).order_by(EnsembleConfig.created_at.desc()).limit(1))
        for m in ensembles.scalars(): print("Ensemble:", m.id)

        rl = await session.execute(select(RLModel).order_by(RLModel.created_at.desc()).limit(1))
        for m in rl.scalars(): print("RL:", m.id)
        
        knn = await session.execute(select(KNNModel).order_by(KNNModel.created_at.desc()).limit(1))
        for m in knn.scalars(): print("KNN:", m.id)
        
        lstm = await session.execute(select(LSTMModel).order_by(LSTMModel.created_at.desc()).limit(1))
        for m in lstm.scalars(): print("LSTM:", m.id)

asyncio.run(main())
