from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud

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
