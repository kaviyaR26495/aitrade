"""Multi-provider LLM abstraction — streams text chunks from OpenAI, Anthropic, Gemini, or Ollama."""
from __future__ import annotations

from typing import AsyncIterator

# ── Provider handlers ─────────────────────────────────────────────────

async def stream_openai(
    api_key: str,
    model: str,
    messages: list[dict],
    system_prompt: str,
    base_url: str | None = None,
) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    stream = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        stream=True,
        temperature=0.7,
        max_tokens=1024,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def stream_anthropic(
    api_key: str,
    model: str,
    messages: list[dict],
    system_prompt: str,
) -> AsyncIterator[str]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)

    async with client.messages.stream(
        model=model,
        system=system_prompt,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_gemini(
    api_key: str,
    model: str,
    messages: list[dict],
    system_prompt: str,
) -> AsyncIterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Convert messages to Gemini Content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
        max_output_tokens=1024,
    )

    async for chunk in await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            yield chunk.text


# ── Provider registry ─────────────────────────────────────────────────

PROVIDERS = {
    "openai": {
        "handler": "openai",
        "models": [
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
            "o4-mini",
        ],
    },
    "anthropic": {
        "handler": "anthropic",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
        ],
    },
    "gemini": {
        "handler": "gemini",
        "models": [
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "nano-banana-pro-preview",
            "lyria-3-pro-preview",
            "deep-research-pro-preview-12-2025"
        ],
    },
    "ollama": {
        "handler": "openai",  # Ollama uses OpenAI-compatible API
        "models": [],  # Populated dynamically
    },
}


async def get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Fetch installed Ollama models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{base_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


_gemini_cache: dict[str, tuple[float, list[str]]] = {}

async def get_gemini_models(api_key: str) -> list[str]:
    """Fetch available Gemini models via the new google.genai SDK (cached 1 h)."""
    import asyncio
    import time

    if not api_key:
        return PROVIDERS["gemini"]["models"]

    now = time.time()
    if api_key in _gemini_cache and now - _gemini_cache[api_key][0] < 3600:
        return _gemini_cache[api_key][1]

    def fetch() -> list[str]:
        from google import genai

        client = genai.Client(api_key=api_key)
        models: list[str] = []
        try:
            for m in client.models.list():
                methods = getattr(m, "supported_generation_methods", None) or []
                if "generateContent" in methods:
                    name = m.name.replace("models/", "").replace("tunedModels/", "")
                    if name:
                        models.append(name)
            result = sorted(set(models), reverse=True)
            if result:
                _gemini_cache[api_key] = (time.time(), result)
            return result or PROVIDERS["gemini"]["models"]
        except Exception:
            return PROVIDERS["gemini"]["models"]

    return await asyncio.to_thread(fetch)


async def stream_chat(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict],
    system_prompt: str,
    ollama_base_url: str = "http://localhost:11434",
) -> AsyncIterator[str]:
    """Route to the correct provider handler and yield text chunks."""
    if provider == "openai":
        async for chunk in stream_openai(api_key, model, messages, system_prompt):
            yield chunk
    elif provider == "anthropic":
        async for chunk in stream_anthropic(api_key, model, messages, system_prompt):
            yield chunk
    elif provider == "gemini":
        async for chunk in stream_gemini(api_key, model, messages, system_prompt):
            yield chunk
    elif provider == "ollama":
        base = f"{ollama_base_url}/v1"
        async for chunk in stream_openai("ollama", model, messages, system_prompt, base_url=base):
            yield chunk
    else:
        yield f"Unknown provider: {provider}"
