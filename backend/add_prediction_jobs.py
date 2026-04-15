import asyncio
from sqlalchemy import text, inspect
from app.db.database import engine

async def migrate():
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_jobs (
                id VARCHAR(50) PRIMARY KEY,
                status VARCHAR(20) DEFAULT 'running',
                progress INTEGER DEFAULT 0,
                total_stocks INTEGER DEFAULT 0,
                completed_stocks INTEGER DEFAULT 0,
                error TEXT,
                batch_id VARCHAR(50),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """))
        print("Prediction jobs table created.")

if __name__ == "__main__":
    asyncio.run(migrate())
