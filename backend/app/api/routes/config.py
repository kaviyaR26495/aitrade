import os
import shutil
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud
from app.config import settings

router = APIRouter()


class SettingUpdate(BaseModel):
    key: str
    value: str


@router.get("/")
async def get_config(db: AsyncSession = Depends(get_db)):
    """Return all app settings from DB."""
    from sqlalchemy import select
    from app.db.models import AppSetting
    result = await db.execute(select(AppSetting))
    settings = result.scalars().all()
    return {s.property: s.value for s in settings}


@router.get("/{key}")
async def get_setting(key: str, db: AsyncSession = Depends(get_db)):
    """Get a specific setting."""
    value = await crud.get_setting(db, key)
    if value is None:
        return {"key": key, "value": None}
    return {"key": key, "value": value}


@router.put("/")
async def update_config(
    req: SettingUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a single app setting."""
    await crud.set_setting(db, req.key, req.value)
    return {"key": req.key, "value": req.value, "status": "updated"}


@router.put("/batch")
async def update_config_batch(
    settings: list[SettingUpdate],
    db: AsyncSession = Depends(get_db),
):
    """Update multiple settings at once."""
    for s in settings:
        await crud.set_setting(db, s.key, s.value)
    return {"updated": len(settings)}


@router.delete("/cleanup/logs")
async def cleanup_logs(background_tasks: BackgroundTasks):
    """Queue deletion of all training log files and return immediately."""
    log_dir = Path("/home/karthi/work/aitrade/backend/logs/training")
    if not log_dir.exists():
        return {"status": "no training log directory", "queued": 0}

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return {"status": "nothing to delete", "queued": 0}

    def _delete_logs(files: list[Path]) -> None:
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

    background_tasks.add_task(_delete_logs, log_files)
    return {"status": "queued", "queued": len(log_files)}


@router.delete("/cleanup/models")
async def cleanup_models(
    background_tasks: BackgroundTasks,
    keep_latest: int = 3,
):
    """Queue deletion of old model artifacts and return immediately."""
    base = Path("/home/karthi/work/aitrade/backend/model_artifacts")
    distill_dir = base / "distill"
    rl_dir = base / "rl"

    def get_version(p: Path) -> int:
        try:
            return int("".join(filter(str.isdigit, p.name)) or "0")
        except ValueError:
            return 0

    to_delete: list[Path] = []

    if distill_dir.exists():
        groups: dict[str, list[Path]] = {"knn": [], "lstm": []}
        for d in distill_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith("knn_"):
                groups["knn"].append(d)
            elif d.name.startswith("lstm_"):
                groups["lstm"].append(d)
        for dirs in groups.values():
            sorted_dirs = sorted(dirs, key=get_version, reverse=True)
            to_delete.extend(sorted_dirs[keep_latest:])

    if rl_dir.exists():
        pipeline_dirs = [d for d in rl_dir.iterdir() if d.is_dir() and d.name.startswith("Pipeline_")]
        sorted_pipelines = sorted(pipeline_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
        to_delete.extend(sorted_pipelines[keep_latest:])

    if not to_delete:
        return {"status": "nothing to delete", "queued": 0}

    def _delete_dirs(dirs: list[Path]) -> None:
        for d in dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass

    background_tasks.add_task(_delete_dirs, to_delete)
    return {"status": "queued", "queued": len(to_delete), "kept_per_type": keep_latest}
