from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure trading_env is importable regardless of editable-install quirks
_trading_env_src = Path(__file__).resolve().parents[2] / "trading_env"
if str(_trading_env_src) not in sys.path:
    sys.path.insert(0, str(_trading_env_src))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    _log = logging.getLogger(__name__)

    # Startup — restore Zerodha access token from DB if available
    try:
        from app.db.database import async_session_factory
        from app.db import crud
        from app.core import zerodha as _zerodha
        async with async_session_factory() as db:
            token = await crud.get_setting(db, "KITE_ACCESS_TOKEN")
            if token:
                _zerodha.set_access_token(token)
                _log.info("Zerodha access token restored from DB.")
    except Exception as exc:
        _log.warning("Could not restore Zerodha token on startup: %s", exc)

    # Startup — sync stock list from Zerodha (tests connectivity too)
    try:
        from app.db.database import async_session_factory
        from app.core import zerodha as _zerodha, data_pipeline
        if _zerodha.is_authenticated():
            async with async_session_factory() as db:
                count = await data_pipeline.populate_stock_list(db)
                _log.info("Startup stock list sync: %d instruments upserted.", count)
        else:
            _log.info("Startup stock list sync skipped — Zerodha not authenticated yet.")
    except Exception as exc:
        _log.warning("Startup stock list sync failed (non-fatal): %s", exc)

    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title="AItrade",
    description="AI Trading Platform — RL + KNN + LSTM Ensemble",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ────────────────────────────────────────────────
from app.api.routes import auth, config as config_routes, data, regime, models, backtest, trading, portfolio, chat  # noqa: E402

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(config_routes.router, prefix="/api/config", tags=["config"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(regime.router, prefix="/api/regime", tags=["regime"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
