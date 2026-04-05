from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import zerodha
from app.api.deps import get_db
from app.db import crud

router = APIRouter()

KITE_TOKEN_KEY = "KITE_ACCESS_TOKEN"


@router.get("/login-url")
async def get_login_url():
    """Get Zerodha login URL for OAuth flow."""
    try:
        kite = zerodha.get_kite()
        url = kite.login_url()
        return {"login_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def zerodha_callback(request_token: str, db: AsyncSession = Depends(get_db)):
    """Handle Zerodha OAuth callback — exchange request_token for access_token."""
    try:
        data = await zerodha.async_generate_session(request_token)
        # Persist token so it survives server restarts
        await crud.set_setting(db, KITE_TOKEN_KEY, data["access_token"])
        return {
            "status": "authenticated",
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")


@router.get("/status")
async def auth_status(db: AsyncSession = Depends(get_db)):
    """Check whether a Zerodha access token is stored."""
    token = await crud.get_setting(db, KITE_TOKEN_KEY)
    return {"authenticated": bool(token)}
