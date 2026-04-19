"""Agentic Investment Committee — shared types and BaseAgent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Signal = Literal["BUY", "SELL", "HOLD", "ABSTAIN"]


@dataclass
class AgentVerdict:
    agent_name: str
    signal: Signal
    confidence: float          # [0, 1]
    reasoning: str
    metadata: dict = field(default_factory=dict)


async def _resolve_llm(db) -> tuple[str, str, str, str]:
    """Return (provider, model, api_key, ollama_url) from AppSetting."""
    from app.db.crud import get_setting
    from app.core.llm_providers import PROVIDERS
    from app.config import settings as app_settings

    SETTINGS_KEY_MAP = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    provider = await get_setting(db, "llm_provider") or "openai"
    model = await get_setting(db, "llm_model") or "gpt-4.1-nano"
    api_key = await get_setting(db, f"llm_api_key_{provider}")
    if not api_key:
        attr = SETTINGS_KEY_MAP.get(provider, "")
        api_key = getattr(app_settings, attr, "") if attr else ""
    ollama_url = await get_setting(db, "ollama_base_url") or "http://localhost:11434"
    return provider, model, api_key or "", ollama_url


async def llm_complete(db, messages: list[dict], system_prompt: str = "") -> str:
    """Non-streaming LLM call — accumulates all chunks into a single string."""
    from app.core.llm_providers import stream_chat

    provider, model, api_key, ollama_url = await _resolve_llm(db)
    parts: list[str] = []
    async for chunk in stream_chat(
        provider=provider,
        api_key=api_key,
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        ollama_base_url=ollama_url,
    ):
        parts.append(chunk)
    return "".join(parts)
