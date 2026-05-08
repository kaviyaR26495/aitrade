from __future__ import annotations

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

    # Startup — check Celery broker connectivity (non-blocking)
    try:
        from app.workers.celery_app import celery_app as _celery
        _celery.control.inspect(timeout=1.0).ping()
        _log.info("Celery broker reachable.")
    except Exception:
        _log.warning("Celery broker not reachable — scheduled tasks will not run until Redis is available.")

    # Startup — mark any pipeline jobs that were left in "running" state as failed.
    # Background tasks are lost on server restart; without this the UI shows them
    # stuck as "in progress" forever.
    try:
        from app.db.database import async_session_factory as _asf
        from sqlalchemy import update as _upd, select as _sel
        from app.db.models import PipelineJob as _PJ
        import json as _json
        async with _asf() as _db:
            _res = await _db.execute(_sel(_PJ).where(_PJ.status.in_(["running", "queued"])))
            _orphans = _res.scalars().all()
            for _j in _orphans:
                _stages = list(_j.stages or [])
                for _s in _stages:
                    if _s.get("status") in ("running", "pending"):
                        _s["status"] = "failed"
                        _s["message"] = "Server restarted — pipeline task was lost. Use Resume to continue."
                await _db.execute(
                    _upd(_PJ).where(_PJ.id == _j.id).values(
                        status="failed",
                        stages=_stages,
                        error="Server restarted mid-run — task lost. Use Resume to continue.",
                    )
                )
            if _orphans:
                await _db.commit()
                _log.warning(
                    "Startup: marked %d orphaned pipeline job(s) as failed: %s",
                    len(_orphans), [j.id for j in _orphans],
                )
    except Exception as _exc:
        _log.warning("Startup pipeline orphan cleanup failed (non-fatal): %s", _exc)

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
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://103.197.114.36:5173",
        "http://103.197.114.36:8000",
        "http://nueroalgo.in",
        "http://www.nueroalgo.in",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import traceback
import logging
from fastapi import Request

error_logger = logging.getLogger("http_errors")
error_logger.setLevel(logging.ERROR)
fh = logging.FileHandler("http_errors.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"))
error_logger.addHandler(fh)

@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            error_logger.error(f"HTTP {response.status_code} | {request.method} {request.url}")
        return response
    except Exception as exc:
        error_logger.error(f"HTTP 500 | {request.method} {request.url} | Exception: {str(exc)}\n{traceback.format_exc()}")
        raise

# ── Route registration ────────────────────────────────────────────────
from app.api.routes import auth, config as config_routes, data, regime, models, backtest, trading, portfolio, chat, pipeline, indices, training, fundamentals, sentiment, agents, backfill  # noqa: E402

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(config_routes.router, prefix="/api/config", tags=["config"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(regime.router, prefix="/api/regime", tags=["regime"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(indices.router, prefix="/api/indices", tags=["indices"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(fundamentals.router, prefix="/api/fundamentals", tags=["fundamentals"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(agents.router)
app.include_router(backfill.router, prefix="/api/backfill", tags=["backfill"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
