"""Chat API — SSE streaming endpoint + provider discovery."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.crud import get_setting
from app.config import settings as app_settings
from app.core.chatbot_context import build_system_prompt
from app.core.llm_providers import PROVIDERS, get_ollama_models, get_gemini_models, stream_chat

router = APIRouter()


# ── Request / response models ─────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str | None = None
    model: str | None = None
    page: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Stream LLM response via SSE."""
    # Resolve provider + model from request or settings
    provider = req.provider or await get_setting(db, "llm_provider") or "openai"
    model = req.model or await get_setting(db, "llm_model") or "gpt-4.1-nano"

    # Resolve API key — DB first, then .env fallback
    SETTINGS_KEY_MAP = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key = await get_setting(db, f"llm_api_key_{provider}")
    if not api_key:
        attr = SETTINGS_KEY_MAP.get(provider, "")
        api_key = getattr(app_settings, attr, "") if attr else ""
    if not api_key and provider != "ollama":
        async def error_stream():
            yield f"data: {json.dumps({'error': f'No API key configured for {provider}. Go to Settings → Chat Assistant to add one.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    ollama_url = await get_setting(db, "ollama_base_url") or "http://localhost:11434"

    # Build system prompt with page context
    system_prompt = build_system_prompt(req.page)

    # Convert messages to dicts
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async def event_generator():
        try:
            async for chunk in stream_chat(
                provider=provider,
                api_key=api_key or "",
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                ollama_base_url=ollama_url,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/providers")
async def list_providers(db: AsyncSession = Depends(get_db)):
    """Return available providers and their model lists."""
    result: dict[str, Any] = {}
    for name, info in PROVIDERS.items():
        result[name] = {"models": list(info["models"])}

    # Populate Ollama models dynamically
    ollama_url = await get_setting(db, "ollama_base_url") or "http://localhost:11434"
    result["ollama"]["models"] = await get_ollama_models(ollama_url)

    # Populate Gemini models dynamically
    gemini_key = await get_setting(db, "llm_api_key_gemini")
    if not gemini_key:
        gemini_key = getattr(app_settings, "GEMINI_API_KEY", "")
    if gemini_key:
        result["gemini"]["models"] = await get_gemini_models(gemini_key)

    return result


@router.get("/status")
async def chat_status(db: AsyncSession = Depends(get_db)):
    """Return current LLM configuration status."""
    provider = await get_setting(db, "llm_provider") or "openai"
    model = await get_setting(db, "llm_model") or "gpt-4.1-nano"
    SETTINGS_KEY_MAP = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key = await get_setting(db, f"llm_api_key_{provider}")
    if not api_key:
        attr = SETTINGS_KEY_MAP.get(provider, "")
        api_key = getattr(app_settings, attr, "") if attr else ""
    ollama_url = await get_setting(db, "ollama_base_url") or "http://localhost:11434"

    configured = bool(api_key) if provider != "ollama" else True

    return {
        "configured": configured,
        "provider": provider,
        "model": model,
        "ollama_base_url": ollama_url,
    }
