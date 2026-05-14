"""Celery application factory.

Broker  : Redis  (redis://localhost:6379/0)
Backend : Redis  (redis://localhost:6379/1)

Two queues:
  data  — I/O-heavy tasks (OHLCV sync, fundamentals, sentiment)
  ml    — CPU/GPU-heavy tasks (retraining, predictions)

Concurrency note
----------------
All Celery tasks run in a *separate process* from the FastAPI server, so
they cannot share SQLAlchemy async sessions.  Each task creates its own
``asyncio.run(...)`` context and opens a fresh DB session via
``async_session_factory()``.
"""
from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init
from kombu import Queue

# Allow overriding via environment variable (useful in Docker)
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "aitrade",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Result expiry
    result_expires=86400,  # 24 hours
    # Worker behaviour
    worker_prefetch_multiplier=1,   # don't pre-fetch ML tasks — they're heavy
    task_acks_late=True,            # ack only after task completes
    worker_max_tasks_per_child=50,  # recycle workers to avoid memory leaks after GPU ops
    # Queues
    task_default_queue="data",
    task_queues=(
        Queue("data"),
        Queue("ml"),
    ),
    task_routes={
        "app.workers.tasks.task_nightly_sync": {"queue": "data"},
        "app.workers.tasks.task_fundamental_sync": {"queue": "data"},
        "app.workers.tasks.task_morning_sentiment": {"queue": "data"},
        "app.workers.tasks.task_morning_predictions": {"queue": "ml"},
        "app.workers.tasks.task_monthly_retrain": {"queue": "ml"},
        "app.workers.tasks.task_morning_auth_check": {"queue": "data"},
        "app.workers.tasks.task_morning_trade_start": {"queue": "data"},
    },
    # Beat schedule (IST = UTC+5:30)
    beat_schedule={
        # Morning auth check — 08:00 IST (02:30 UTC) Mon-Fri
        # Safety net: alerts via Telegram if Android auto-login hasn't fired yet
        "morning-auth-check": {
            "task": "app.workers.tasks.task_morning_auth_check",
            "schedule": crontab(hour=2, minute=30, day_of_week="1-5"),
        },
        # Nightly OHLCV + indicator sync — 18:30 IST (13:00 UTC)
        "nightly-sync": {
            "task": "app.workers.tasks.task_nightly_sync",
            "schedule": crontab(hour=13, minute=0),
        },
        # Weekly fundamental refresh — Sunday 20:00 IST (14:30 UTC)
        "weekly-fundamental-sync": {
            "task": "app.workers.tasks.task_fundamental_sync",
            "schedule": crontab(hour=14, minute=30, day_of_week=0),
        },
        # Morning sentiment — 08:45 IST (03:15 UTC)
        "morning-sentiment": {
            "task": "app.workers.tasks.task_morning_sentiment",
            "schedule": crontab(hour=3, minute=15),
        },
        # Morning predictions — 09:00 IST (03:30 UTC)
        "morning-predictions": {
            "task": "app.workers.tasks.task_morning_predictions",
            "schedule": crontab(hour=3, minute=30),
        },
        # TPML signal generation — 09:05 IST (03:35 UTC), Mon-Fri
        "tpml-signals": {
            "task": "app.workers.tasks.task_morning_tpml_signals",
            "schedule": crontab(hour=3, minute=35, day_of_week="1-5"),
            "options": {"queue": "ml"},
        },
        # Trailing stop updates — every 5 min, 09:15-15:30 IST (03:45-10:00 UTC), Mon-Fri
        "trailing-stop-update": {
            "task": "app.workers.tasks.task_trailing_stop_update",
            "schedule": crontab(minute="*/5", hour="3-10", day_of_week="1-5"),
            "options": {"queue": "ml"},
        },
        # Monthly retraining — 1st of each month at 22:00 IST (16:30 UTC)
        "monthly-retrain": {
            "task": "app.workers.tasks.task_monthly_retrain",
            "schedule": crontab(hour=16, minute=30, day_of_month=1),
        },
    },
    timezone="UTC",
)


@worker_process_init.connect
def _dispose_engine_on_fork(**kwargs):
    """Replace the inherited SQLAlchemy async engine with a NullPool one after fork.

    Celery uses prefork workers. The child inherits the parent's connection pool
    whose internal futures are tied to the parent's event loop. Using NullPool
    means each asyncio.run() call gets a fresh connection that is closed
    immediately after use — no pool state to conflict across event loop
    boundaries. This prevents 'Future attached to a different loop' errors.
    """
    import os
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    import app.db.database as _db_module

    # Dispose old pooled connections from parent process
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_db_module.engine.dispose())
        loop.close()
    except Exception:
        pass

    # Replace engine + session factory with NullPool variants safe for forked workers
    from app.config import settings
    new_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        poolclass=NullPool,
    )
    _db_module.engine = new_engine
    _db_module.async_session_factory = async_sessionmaker(
        new_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
