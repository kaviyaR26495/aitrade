"""Sentiment Triage Funnel.

Two-stage pipeline:
  Stage 1 — FinBERT Filter (GPU):
      Fetch headlines → discard ~65% neutral noise using ProsusAI/finbert.
  Stage 2 — LLM Judgment:
      Escalate only non-neutral headlines to the configured reasoning LLM
      for a structured Financial Impact Score in [-1, +1].

The aggregated daily score is persisted in ``stock_sentiment``.

Design decisions
----------------
- FinBERT model is loaded once at module level and kept on GPU.
- LLM calls reuse the provider already configured in app Settings
  (``llm_provider`` + ``llm_model`` + corresponding API key).
- All network errors are caught; a stock with no news returns a
  neutral score (0.0) so the ML pipeline is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Stock, StockSentiment, now_ist

_log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

# ── FinBERT singleton ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_finbert():
    """Load ProsusAI/finbert pipeline once and cache on the GPU.

    Lazy-loaded on first call so the import does not block FastAPI startup
    when the transformers package is available but a GPU task is not needed.
    """
    from transformers import pipeline as hf_pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    _log.info("Loading FinBERT on device=%s", "GPU" if device == 0 else "CPU")
    return hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512,
        top_k=None,  # return all 3 class scores
    )


# ── TypedDicts ─────────────────────────────────────────────────────────

class HeadlineResult(TypedDict):
    title: str
    source: str
    published: str
    finbert_label: str      # positive | negative | neutral
    finbert_score: float    # confidence [0,1] for the predicted label
    finbert_signed: float   # signed: +score if positive, -score if negative, 0 if neutral


class SentimentSummary(TypedDict):
    stock_id: int
    date: str
    headline_count: int
    neutral_filtered_count: int
    avg_finbert_score: float | None
    llm_impact_score: float | None
    llm_summary: str | None
    non_neutral_headlines: list[HeadlineResult]


# ── Headline fetching ──────────────────────────────────────────────────

async def fetch_headlines(symbol: str, max_results: int = 20) -> list[dict]:
    """Fetch recent headlines for a stock from Google News RSS.

    Returns a list of dicts with keys: title, source, published.
    Falls back to an empty list on any error.
    """
    import feedparser

    query = f"{symbol} NSE stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    def _blocking_parse() -> list[dict]:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:max_results]:
            results.append({
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", "Unknown") if isinstance(entry.get("source"), dict) else str(entry.get("source", "Unknown")),
                "published": str(entry.get("published", "")),
            })
        return results

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _blocking_parse)
    except Exception as exc:
        _log.warning("Headline fetch failed for %s: %s", symbol, exc)
        return []


# ── FinBERT triage ─────────────────────────────────────────────────────

def finbert_triage(raw_headlines: list[dict]) -> list[HeadlineResult]:
    """Run FinBERT on a list of headlines and discard neutral ones.

    Returns only non-neutral headlines enriched with sentiment scores.
    Runs synchronously — caller should delegate to an executor for async use.
    """
    if not raw_headlines:
        return []

    pipe = _load_finbert()
    titles = [h["title"] for h in raw_headlines if h.get("title")]
    if not titles:
        return []

    try:
        outputs = pipe(titles, batch_size=16)
    except Exception as exc:
        _log.warning("FinBERT inference failed: %s", exc)
        return []

    results: list[HeadlineResult] = []
    for headline, scores in zip(raw_headlines, outputs):
        if not scores:
            continue
        # Scores is a list of {label, score} for all 3 classes
        best = max(scores, key=lambda x: x["score"])
        label = best["label"].lower()   # positive | negative | neutral
        conf = float(best["score"])

        # Skip neutral headlines — this is the core triage step
        if label == "neutral":
            continue

        signed = conf if label == "positive" else -conf
        results.append(
            HeadlineResult(
                title=headline["title"],
                source=headline.get("source", ""),
                published=headline.get("published", ""),
                finbert_label=label,
                finbert_score=conf,
                finbert_signed=signed,
            )
        )
    return results


# ── LLM Financial Impact Score ─────────────────────────────────────────

_IMPACT_PROMPT = """\
You are a quantitative financial analyst specialising in the Indian stock market.
Analyse the following news headlines about a stock and return a structured JSON response.

Headlines:
{headlines}

Return ONLY valid JSON with this exact schema (no markdown, no extra text):
{{
  "impact_score": <float between -1.0 and 1.0>,
  "event_type": "<string: earnings | regulatory | macro | merger | insider | sector | other>",
  "urgency_hours": <int: how many hours until the market impact is expected, 0 if immediate>,
  "summary": "<one sentence summary of the key financial implication>"
}}

Scoring guide:
  +1.0 = strongly bullish (beat earnings, major contract win, regulatory approval)
   0.5 = moderately bullish
   0.0 = ambiguous or mixed signals
  -0.5 = moderately bearish
  -1.0 = strongly bearish (profit warning, regulatory ban, fraud allegation)
"""


async def llm_financial_impact_score(
    headlines: list[HeadlineResult],
    db: AsyncSession,
) -> dict[str, Any]:
    """Escalate non-neutral headlines to the configured LLM for a structured
    Financial Impact Score.

    Returns a dict: {impact_score, event_type, urgency_hours, summary}.
    Falls back to {impact_score: 0.0, ...} on any error.
    """
    if not headlines:
        return {"impact_score": 0.0, "event_type": "other", "urgency_hours": 0, "summary": "No non-neutral headlines."}

    # Resolve LLM provider from DB settings
    try:
        from app.db import crud
        from app.core.llm_providers import get_provider

        provider_name = await crud.get_setting(db, "llm_provider") or "openai"
        model_name = await crud.get_setting(db, "llm_model") or "gpt-4o-mini"
        api_key = await crud.get_setting(db, f"{provider_name}_api_key") or ""

        provider = get_provider(provider_name, model_name, api_key)
    except Exception as exc:
        _log.warning("Could not initialise LLM provider for sentiment: %s", exc)
        return {"impact_score": 0.0, "event_type": "other", "urgency_hours": 0, "summary": "LLM unavailable."}

    headline_text = "\n".join(
        f"- [{h['finbert_label'].upper()}] {h['title']}" for h in headlines
    )
    prompt = _IMPACT_PROMPT.format(headlines=headline_text)

    try:
        response_text = ""
        async for chunk in provider.stream(prompt):
            response_text += chunk

        # Extract JSON from response
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in LLM response")

        result = json.loads(match.group())
        result["impact_score"] = float(
            max(-1.0, min(1.0, result.get("impact_score", 0.0)))
        )
        return result
    except Exception as exc:
        _log.warning("LLM impact scoring failed: %s", exc)
        # Fallback: average FinBERT signed scores
        avg = sum(h["finbert_signed"] for h in headlines) / len(headlines)
        return {
            "impact_score": round(float(avg), 4),
            "event_type": "other",
            "urgency_hours": 0,
            "summary": "LLM failed; FinBERT average used.",
        }


# ── Aggregation & Persistence ──────────────────────────────────────────

async def run_sentiment_for_stock(
    db: AsyncSession,
    stock: Stock,
    as_of_date: date | None = None,
    max_headlines: int = 20,
) -> SentimentSummary:
    """Full sentiment pipeline for one stock:
    fetch → FinBERT triage → LLM judgment → persist.

    Idempotent: if a row for (stock_id, date) already exists it is updated
    in-place, allowing re-runs to refresh stale sentiment data.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    # Fetch headlines
    raw = await fetch_headlines(stock.symbol, max_results=max_headlines)
    total = len(raw)

    # FinBERT triage in executor (CPU-heavy even with GPU offload)
    loop = asyncio.get_event_loop()
    non_neutral: list[HeadlineResult] = await loop.run_in_executor(
        None, finbert_triage, raw
    )
    neutral_count = total - len(non_neutral)

    # Average FinBERT signed score across non-neutral headlines
    avg_finbert: float | None = None
    if non_neutral:
        avg_finbert = round(
            sum(h["finbert_signed"] for h in non_neutral) / len(non_neutral), 4
        )

    # LLM judgment
    llm_result = await llm_financial_impact_score(non_neutral, db)
    llm_score = llm_result.get("impact_score")
    llm_summary = llm_result.get("summary")

    # Persist
    existing = await db.execute(
        select(StockSentiment).where(
            StockSentiment.stock_id == stock.id,
            StockSentiment.date == as_of_date,
        )
    )
    row = existing.scalar_one_or_none()
    if row is None:
        row = StockSentiment(stock_id=stock.id, date=as_of_date, ingested_at=now_ist())
        db.add(row)

    row.headline_count = total
    row.neutral_filtered_count = neutral_count
    row.avg_finbert_score = avg_finbert
    row.llm_impact_score = llm_score
    row.llm_summary = llm_summary
    row.ingested_at = now_ist()
    await db.commit()

    return SentimentSummary(
        stock_id=stock.id,
        date=as_of_date.isoformat(),
        headline_count=total,
        neutral_filtered_count=neutral_count,
        avg_finbert_score=avg_finbert,
        llm_impact_score=llm_score,
        llm_summary=llm_summary,
        non_neutral_headlines=non_neutral,
    )


async def run_sentiment_batch(
    db: AsyncSession,
    stocks: list[Stock],
    as_of_date: date | None = None,
    concurrency: int = 5,
) -> list[SentimentSummary]:
    """Run sentiment pipeline for multiple stocks with bounded concurrency.

    ``concurrency`` limits simultaneous headline fetches + LLM calls to
    avoid rate-limiting on the news RSS and the LLM provider.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(stock: Stock) -> SentimentSummary:
        async with semaphore:
            try:
                return await run_sentiment_for_stock(db, stock, as_of_date)
            except Exception as exc:
                _log.error("Sentiment failed for %s: %s", stock.symbol, exc)
                return SentimentSummary(
                    stock_id=stock.id,
                    date=as_of_date.isoformat(),
                    headline_count=0,
                    neutral_filtered_count=0,
                    avg_finbert_score=None,
                    llm_impact_score=None,
                    llm_summary=None,
                    non_neutral_headlines=[],
                )

    return list(await asyncio.gather(*[_bounded(s) for s in stocks]))
