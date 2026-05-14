import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import delete
from app.core import zerodha
from app.api.deps import get_db
from app.db import crud
from app.db.models import AppSetting
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

KITE_TOKEN_KEY = "KITE_ACCESS_TOKEN"


class AutoLoginConfig(BaseModel):
    zerodha_user_id: str
    zerodha_password: str
    zerodha_totp_secret: str
    enabled: bool = True


@router.get("/login-url")
async def get_login_url(db: AsyncSession = Depends(get_db)):
    """Get Zerodha login URL for OAuth flow — uses DB-stored credentials if available."""
    try:
        from app.config import settings
        api_key = await crud.get_setting(db, "KITE_API_KEY") or settings.KITE_API_KEY
        zerodha_ip = await crud.get_setting(db, "ZERODHA_IP") or settings.ZERODHA_IP
        
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="Kite API Key not configured. Set it in Settings → Zerodha first."
            )
            
        kite = zerodha.refresh_kite(api_key, zerodha_ip)
        url = kite.login_url()
        return {"login_url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def zerodha_callback(request_token: str, state: str | None = None, db: AsyncSession = Depends(get_db)):
    """Handle Zerodha OAuth callback — exchange request_token for access_token."""
    try:
        from app.config import settings
        from app.core import zerodha as _zerodha
        
        # Ensure we use the same API SECRET configured in UI
        api_secret = await crud.get_setting(db, "KITE_API_SECRET") or settings.KITE_API_SECRET
        if not api_secret:
             raise HTTPException(status_code=400, detail="Kite API Secret not configured.")

        data = await asyncio.to_thread(_zerodha.generate_session, request_token, api_secret)
        # Persist token so it survives server restarts
        await crud.set_setting(db, KITE_TOKEN_KEY, data["access_token"])
        
        return {
            "status": "authenticated",
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")


@router.get("/status")
async def auth_status(db: AsyncSession = Depends(get_db)):
    """Check whether a Zerodha access token is stored and valid.

    This route performs several checks:
    1. If a token is in DB but not active in memory (e.g. server restart), it restores it.
    2. It calls Zerodha API (profile) to verify the token is still valid.
    3. If invalid, it purges the token from DB so the UI shows 'Not Connected'.
    """
    token = await crud.get_setting(db, KITE_TOKEN_KEY)
    if not token:
        return {"authenticated": False}

    # Ensure memory instance has the token if it's in DB
    if not zerodha.is_authenticated():
        zerodha.set_access_token(token)

    # Perform live check
    is_valid = await zerodha.is_authenticated_live()

    if not is_valid:
        # Token expired or invalid — purge it
        logger.warning("Clearing expired/invalid Zerodha token from DB")
        await db.execute(delete(AppSetting).where(AppSetting.property == KITE_TOKEN_KEY))
        await db.commit()
        # Also clear from memory
        try:
            zerodha.get_kite().access_token = None
        except:
            pass
        return {"authenticated": False}

    return {"authenticated": True}


@router.post("/trigger-morning-trade")
async def trigger_morning_trade(db: AsyncSession = Depends(get_db)):
    """Called by the Android auto-login service after successful Zerodha authentication.

    Verifies the token is active then queues the morning trade-start Celery task.
    This fires portfolio reconciliation and trade signal generation.
    """
    token = await crud.get_setting(db, KITE_TOKEN_KEY)
    if not token:
        raise HTTPException(status_code=401, detail="Zerodha not authenticated — no token found")

    if not zerodha.is_authenticated():
        zerodha.set_access_token(token)

    is_valid = await zerodha.is_authenticated_live()
    if not is_valid:
        raise HTTPException(status_code=401, detail="Zerodha token is expired or invalid")

    try:
        from app.workers.tasks import task_morning_trade_start
        result = task_morning_trade_start.delay()
        logger.info("Morning trade start queued: task_id=%s", result.id)
        return {"status": "trade_start_queued", "task_id": result.id}
    except Exception as exc:
        logger.error("Failed to queue morning trade task: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to queue trade task: {exc}")


@router.post("/save-auto-login")
async def save_auto_login_config(config: AutoLoginConfig, db: AsyncSession = Depends(get_db)):
    """Store Zerodha auto-login credentials in the DB.

    The Android app also stores these in EncryptedSharedPreferences locally.
    SECURITY: Credentials stored encrypted via AppSetting. Only the backend process can read them.
    """
    await crud.set_setting(db, "ZERODHA_AUTO_LOGIN_USER_ID", config.zerodha_user_id)
    await crud.set_setting(db, "ZERODHA_AUTO_LOGIN_ENABLED", str(config.enabled).lower())
    if config.zerodha_password:
        await crud.set_setting(db, "ZERODHA_AUTO_LOGIN_PASSWORD", config.zerodha_password)
    if config.zerodha_totp_secret:
        await crud.set_setting(db, "ZERODHA_AUTO_LOGIN_TOTP_SECRET", config.zerodha_totp_secret)

    logger.info("Auto-login config saved for user: %s", config.zerodha_user_id)
    return {"status": "saved", "enabled": config.enabled}


@router.get("/auto-login-status")
async def get_auto_login_status(db: AsyncSession = Depends(get_db)):
    """Return whether auto-login is configured and enabled."""
    user_id = await crud.get_setting(db, "ZERODHA_AUTO_LOGIN_USER_ID") or ""
    enabled = (await crud.get_setting(db, "ZERODHA_AUTO_LOGIN_ENABLED") or "false") == "true"
    has_password = bool(await crud.get_setting(db, "ZERODHA_AUTO_LOGIN_PASSWORD"))
    has_totp = bool(await crud.get_setting(db, "ZERODHA_AUTO_LOGIN_TOTP_SECRET"))
    return {
        "configured": bool(user_id and has_password and has_totp),
        "enabled": enabled,
        "user_id": user_id,
    }


@router.get("/users")
async def get_shared_users(db: AsyncSession = Depends(get_db)):
    """Fetch the shared users list from DB settings."""
    users_json = await crud.get_setting(db, "SHARED_USERS")
    if not users_json:
        return []
    import json
    try:
        return json.loads(users_json)
    except:
        return []


@router.post("/users")
async def save_shared_users(users: list, db: AsyncSession = Depends(get_db)):
    """Save the shared users list to DB settings."""
    import json
    await crud.set_setting(db, "SHARED_USERS", json.dumps(users))
    return {"status": "success"}


@router.get("/roles")
async def get_shared_roles(db: AsyncSession = Depends(get_db)):
    """Fetch the shared roles list from DB settings."""
    roles_json = await crud.get_setting(db, "SHARED_ROLES")
    if not roles_json:
        return []
    import json
    try:
        return json.loads(roles_json)
    except:
        return []


@router.post("/roles")
async def save_shared_roles(roles: list, db: AsyncSession = Depends(get_db)):
    """Save the shared roles list to DB settings."""
    import json
    await crud.set_setting(db, "SHARED_ROLES", json.dumps(roles))
    return {"status": "success"}
