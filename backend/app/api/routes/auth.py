import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import zerodha
from app.api.deps import get_db
from app.db import crud

router = APIRouter()

KITE_TOKEN_KEY = "KITE_ACCESS_TOKEN"


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
async def zerodha_callback(request_token: str, db: AsyncSession = Depends(get_db)):
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
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")


@router.get("/status")
async def auth_status(db: AsyncSession = Depends(get_db)):
    """Check whether a Zerodha access token is stored."""
    token = await crud.get_setting(db, KITE_TOKEN_KEY)
    return {"authenticated": bool(token)}
