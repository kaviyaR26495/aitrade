"""KiteConnect wrapper with configurable IP routing.

Ported from pytrade's pyTrade_common.py — modernized for async FastAPI.
NSE only, CNC orders only (no MIS/intraday).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class CorporateActionGapError(Exception):
    """Raised when a corporate action price gap is detected for a symbol.

    Callers running inside an async route should catch this and write a
    CorporateActionBlock to the DB via crud.create_ca_block so the 48-hour
    cooldown survives server restarts.
    """

    def __init__(self, symbol: str, gap_pct: float) -> None:
        self.symbol = symbol
        self.gap_pct = gap_pct
        super().__init__(f"Corporate action gap {gap_pct:.1f}% detected for {symbol}")


def send_emergency_alert(message: str) -> None:
    """Fire-and-forget Telegram alert for critical trading failures.

    Sends a message via Telegram Bot API.  Silently logs and returns if
    TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not configured so the caller
    is never blocked by a misconfigured alert channel.
    """
    import urllib.parse
    import urllib.request

    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        logger.warning("Emergency alert suppressed (Telegram not configured): %s", message)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    try:
        with urllib.request.urlopen(url, data=payload, timeout=5) as resp:
            if resp.status != 200:
                logger.error("Telegram alert HTTP %s for message: %s", resp.status, message)
    except Exception as exc:
        logger.error("Telegram alert failed: %s — original message: %s", exc, message)


def detect_corporate_action_gap(
    symbol: str,
    exchange: str = "NSE",
    threshold_pct: float | None = None,
) -> tuple[bool, float]:
    """Detect an abnormal open-vs-prev-close price gap suggesting a corporate action.

    Uses Kite OHLC quote which returns today's open and the previous day's close
    in a single call.  No external API is needed.

    Corporate action price impacts:
      - Stock split 1:2   → ~50% price drop
      - Bonus shares 1:1  → ~50% price drop
      - Large dividend     → 8–15% drop
      - Small dividend     → < 5% (ignored, below default threshold)

    Parameters
    ----------
    threshold_pct : Override ``settings.CORPORATE_ACTION_GAP_PCT`` for testing.

    Returns
    -------
    (is_gap_detected: bool, gap_pct: float)
        ``is_gap_detected`` is True when ``|open/prev_close − 1| ≥ threshold``.
        ``gap_pct`` is the raw percentage magnitude (always ≥ 0).
    """
    if threshold_pct is None:
        threshold_pct = settings.CORPORATE_ACTION_GAP_PCT

    try:
        key = f"{exchange}:{symbol}"
        data = get_kite().ohlc(key)
        if not data or key not in data:
            logger.warning("No OHLC data for %s — skipping gap check", key)
            return False, 0.0

        ohlc = data[key]["ohlc"]
        prev_close = float(ohlc["close"])   # Kite OHLC "close" = previous day's close
        today_open = float(ohlc["open"])

        if prev_close <= 0:
            return False, 0.0

        gap_pct = abs(today_open - prev_close) / prev_close * 100.0
        if gap_pct >= threshold_pct:
            logger.warning(
                "Corporate action gap detected for %s: open=%.2f, prev_close=%.2f, gap=%.2f%%"
                " (threshold=%.1f%%) — blocking for 48 h",
                symbol, today_open, prev_close, gap_pct, threshold_pct,
            )
            return True, gap_pct

    except Exception as exc:
        # Never block a trade due to a failed gap check — return clean
        logger.warning("Gap detection failed for %s: %s", symbol, exc)

    return False, 0.0


# Lazy import — kiteconnect may not be installed during tests
_kite = None


def get_kite() -> Any:
    """Get or create KiteConnect instance (singleton)."""
    global _kite
    if _kite is not None:
        return _kite

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise RuntimeError("kiteconnect package not installed. Run: pip install kiteconnect")

    _kite = KiteConnect(api_key=settings.KITE_API_KEY)

    # Configurable IP proxy
    if settings.ZERODHA_IP:
        _kite.root = f"https://{settings.ZERODHA_IP}"
        logger.info("Using custom Zerodha IP: %s", settings.ZERODHA_IP)

    return _kite


def set_access_token(access_token: str):
    kite = get_kite()
    kite.set_access_token(access_token)


def is_authenticated() -> bool:
    """Return True if a non-empty access token has been set on the Kite instance."""
    try:
        return bool(getattr(get_kite(), 'access_token', None))
    except Exception:
        return False


def generate_session(request_token: str) -> dict:
    kite = get_kite()
    data = kite.generate_session(request_token, api_secret=settings.KITE_API_SECRET)
    kite.set_access_token(data["access_token"])
    return data


def get_login_url() -> str:
    kite = get_kite()
    return kite.login_url()


# ── Market Data ────────────────────────────────────────────────────────

def get_ltp(symbol: str, exchange: str = "NSE") -> float:
    """Fetch Last Traded Price."""
    kite = get_kite()
    key = f"{exchange}:{symbol}"
    data = kite.ltp(key)
    return float(data[key]["last_price"])


def get_close_price(symbol: str, exchange: str = "NSE") -> float:
    """Fetch close price from OHLC quote."""
    kite = get_kite()
    key = f"{exchange}:{symbol}"
    data = kite.ohlc(key)
    return float(data[key]["ohlc"]["close"])


def get_quote(symbol: str, exchange: str = "NSE") -> dict:
    kite = get_kite()
    key = f"{exchange}:{symbol}"
    data = kite.quote(key)
    return data[key]
 

def fetch_historical_data(
    instrument_token: int,
    from_date: datetime,
    to_date: datetime,
    interval: str = "day",
) -> list[dict]:
    """
    Fetch historical OHLCV data from Kite API.

    Uses 2000-day chunking (ported from pytrade's fetch_chunk pattern).
    Interval: 'day' or 'week' only.
    """
    kite = get_kite()
    all_data = []

    # Chunk into 2000-day windows (Kite API limit)
    chunk_days = 2000
    current_from = from_date

    while current_from < to_date:
        current_to = min(current_from + timedelta(days=chunk_days), to_date)
        try:
            records = kite.historical_data(
                instrument_token,
                current_from,
                current_to,
                interval,
            )
            all_data.extend(records)
        except Exception as e:
            err_str = str(e)
            logger.error(
                "Error fetching history for token=%s from=%s to=%s: %s",
                instrument_token, current_from, current_to, e,
            )
            # Re-raise auth errors so callers can surface them to the user
            if 'token' in err_str.lower() or 'access' in err_str.lower() or 'permission' in err_str.lower():
                raise
        current_from = current_to + timedelta(days=1)

    return all_data


def get_instruments(exchange: str = "NSE") -> list[dict]:
    """Fetch all instruments for an exchange."""
    kite = get_kite()
    return kite.instruments(exchange)


# ── Holdings & Positions ───────────────────────────────────────────────

EXCLUDED_SYMBOLS = {"YESBANK", "ADANIENSOL"}


def get_holdings() -> list[dict]:
    """Get current holdings, filtered for active holdings only."""
    kite = get_kite()
    holdings = kite.holdings()
    return [
        h for h in holdings
        if (h.get("quantity", 0) + h.get("t1_quantity", 0)) > 0
        and h.get("tradingsymbol") not in EXCLUDED_SYMBOLS
    ]


def get_positions() -> list[dict]:
    """Get current net positions."""
    kite = get_kite()
    return kite.positions().get("net", [])


def build_portfolio_snapshot() -> dict:
    """Fetch live broker state and return a reconciliation snapshot dict.

    This is the *single source of truth* for cash and holdings that the
    sizing algorithms must use.  It should be called every morning at
    08:30 IST *before* any orders are placed.  Calling it overwrites the
    local DB ledger with the absolute truth from Zerodha so Kelly /
    Vol-Target sizing never operates on stale or hallucinated cash values.

    Returns
    -------
    dict with fields:
        cash_available    — free margin (equity segment)
        opening_balance   — total opening balance (equity)
        holdings_value    — sum(quantity × last_price) across all CNC holdings
        unrealized_pnl    — sum of Kite-reported P&L across all holdings
        holdings          — raw list from kite.holdings() (filtered)
        positions         — raw list from kite.positions()[\"net\"]
    """
    kite = get_kite()

    # ── Cash from margins API ───────────────────────────────────────
    try:
        margins = kite.margins("equity")
        cash_available = float(margins.get("available", {}).get("live_balance", 0))
        opening_balance = float(margins.get("available", {}).get("opening_balance", 0))
    except Exception as exc:
        logger.error("Failed to fetch margins: %s", exc)
        raise

    # ── Holdings ────────────────────────────────────────────────────
    holdings = get_holdings()
    holdings_value = sum(
        h.get("quantity", 0) * h.get("last_price", h.get("average_price", 0))
        for h in holdings
    )
    unrealized_pnl = sum(h.get("pnl", 0) for h in holdings)

    # ── Positions ───────────────────────────────────────────────────
    positions = get_positions()

    logger.info(
        "Portfolio snapshot: cash=%.2f holdings_value=%.2f unrealized_pnl=%.2f "
        "(%d holdings, %d positions)",
        cash_available, holdings_value, unrealized_pnl, len(holdings), len(positions),
    )

    return {
        "cash_available": cash_available,
        "opening_balance": opening_balance,
        "holdings_value": holdings_value,
        "unrealized_pnl": unrealized_pnl,
        "holdings": holdings,
        "positions": positions,
    }


# ── Order Placement ────────────────────────────────────────────────────

def get_tag() -> list[str]:
    """Generate order tags: master + monthly tag (ported from pytrade)."""
    now = datetime.now()
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    return ["aitrade", f"ai{month_map[now.month]}{now.year % 100}"]


def parse_symbol(symbol: str) -> tuple[str, str]:
    """Parse symbol to (clean_symbol, exchange). NSE only."""
    if symbol.endswith(".NS"):
        return symbol.replace(".NS", ""), "NSE"
    return symbol, "NSE"


def place_order(
    symbol: str,
    transaction_type: str,  # "BUY" or "SELL"
    quantity: int,
    price: float | None = None,
    exchange: str = "NSE",
    product: str = "CNC",
    order_type: str = "LIMIT",
    variety: str = "regular",
) -> str | None:
    """Place a CNC order on NSE. Returns Zerodha order_id."""
    kite = get_kite()

    if symbol in EXCLUDED_SYMBOLS:
        logger.warning("Skipping excluded symbol: %s", symbol)
        return None

    tags = get_tag()

    try:
        order_id = kite.place_order(
            variety=variety,
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=product,
            order_type=order_type,
            price=price,
            validity="DAY",
            tag=tags[1],
        )
        logger.info("Order placed: %s %s %s qty=%d @ %s → %s",
                     transaction_type, symbol, exchange, quantity, price, order_id)
        return str(order_id)
    except Exception as e:
        logger.error("Order failed: %s %s — %s", transaction_type, symbol, e)
        raise


def place_gtt_order(
    symbol: str,
    exchange: str,
    avg_price: float,
    quantity: int,
    sell_pct: float,
    stoploss_pct: float,
    tick_size: float = 0.05,
) -> int | None:
    """Place GTT OCO (stoploss + target) order. Ported from pytrade."""
    kite = get_kite()

    # Calculate tick-adjusted prices
    target_price = round(avg_price * (1 + sell_pct / 100) / tick_size) * tick_size
    stoploss_price = round(avg_price * (1 - stoploss_pct / 100) / tick_size) * tick_size

    try:
        gtt_id = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_OCO,
            tradingsymbol=symbol,
            exchange=exchange,
            trigger_values=[stoploss_price, target_price],
            last_price=avg_price,
            orders=[
                {
                    "exchange": exchange,
                    "tradingsymbol": symbol,
                    "transaction_type": "SELL",
                    "quantity": quantity,
                    "order_type": "LIMIT",
                    "product": "CNC",
                    "price": stoploss_price,
                },
                {
                    "exchange": exchange,
                    "tradingsymbol": symbol,
                    "transaction_type": "SELL",
                    "quantity": quantity,
                    "order_type": "LIMIT",
                    "product": "CNC",
                    "price": target_price,
                },
            ],
        )
        logger.info("GTT placed for %s: SL=%.2f Target=%.2f → %s",
                     symbol, stoploss_price, target_price, gtt_id)
        return gtt_id
    except Exception as e:
        logger.error("GTT failed for %s — %s", symbol, e)
        raise


def exit_all_positions() -> list[str]:
    """Emergency kill switch — exit all holdings at market price."""
    kite = get_kite()
    order_ids = []
    holdings = get_holdings()

    for h in holdings:
        qty = h.get("quantity", 0) + h.get("t1_quantity", 0)
        if qty > 0:
            try:
                oid = kite.place_order(
                    variety="regular",
                    exchange=h.get("exchange", "NSE"),
                    tradingsymbol=h["tradingsymbol"],
                    transaction_type="SELL",
                    quantity=qty,
                    product="CNC",
                    order_type="MARKET",
                    validity="DAY",
                )
                order_ids.append(str(oid))
                logger.info("Exit: SELL %s qty=%d → %s", h["tradingsymbol"], qty, oid)
            except Exception as e:
                logger.error("Exit failed: %s — %s", h["tradingsymbol"], e)

    return order_ids


# ── Limit-Chase Order ──────────────────────────────────────────────────
# Prevents unfilled LIMIT orders from missing the trade when the stock
# runs away.  Algorithm:
#
#   1. Fetch current bid (BUY) or ask (SELL) from the live order book.
#   2. Place a LIMIT at the bid/ask.
#   3. Poll status every ``poll_interval_s`` seconds.
#   4. If still OPEN after ``retry_after_s``: cancel, re-fetch bid/ask,
#      resubmit — up to ``max_attempts`` retries.
#   5. Final attempt (or slippage budget exhausted): MARKET order.


def _get_bid_ask(symbol: str, exchange: str) -> tuple[float, float]:
    """Return (best_bid, best_ask) from the live order book depth."""
    kite = get_kite()
    key = f"{exchange}:{symbol}"
    data = kite.quote(key)
    depth = data[key].get("depth", {})
    bids = depth.get("buy", [])
    asks = depth.get("sell", [])
    best_bid = float(bids[0]["price"]) if bids else float(data[key]["last_price"])
    best_ask = float(asks[0]["price"]) if asks else float(data[key]["last_price"])
    return best_bid, best_ask


def _cancel_order(order_id: str, variety: str = "regular") -> None:
    try:
        get_kite().cancel_order(variety=variety, order_id=order_id)
        logger.info("Cancelled order %s", order_id)
    except Exception as exc:
        logger.warning("Cancel failed for %s: %s", order_id, exc)


def _order_status(order_id: str) -> str:
    """Return Kite order status string e.g. 'COMPLETE', 'OPEN', 'CANCELLED'.

    Uses ``order_history(order_id=...)`` which targets a single order rather
    than ``orders()`` which downloads the full day's order book.  With 3+
    concurrent limit-chase loops polling every 5 s the old implementation
    would exhaust Kite's 3-req/s rate limit immediately.
    """
    try:
        history = get_kite().order_history(order_id=order_id)
        if history:
            return str(history[-1].get("status", "UNKNOWN")).upper()
    except Exception as exc:
        logger.warning("Could not fetch order status for %s: %s", order_id, exc)
    return "UNKNOWN"


def _get_fill_price(order_id: str, fallback: float) -> float:
    """Return the average fill price from order_history, or *fallback* on error."""
    try:
        history = get_kite().order_history(order_id=order_id)
        if history:
            avg = float(history[-1].get("average_price", 0))
            if avg > 0:
                return avg
    except Exception as exc:
        logger.warning("Could not fetch fill price for %s: %s", order_id, exc)
    return fallback


def place_limit_chase_order(
    symbol: str,
    transaction_type: str,          # "BUY" or "SELL"
    quantity: int,
    exchange: str = "NSE",
    product: str = "CNC",
    max_slippage_pct: float = 0.3,  # give up if price drifts > 0.3 % from first quote
    max_attempts: int = 5,
    retry_after_s: float = 120.0,   # re-price after 2 minutes of no fill
    poll_interval_s: float = 5.0,   # order-status poll frequency
) -> dict | None:
    """Place a LIMIT order at the current bid/ask and chase the price if unfilled.

    Solves the adverse-selection problem where an unfilled LIMIT order means
    the trade only executes if price reverses against you.

    Parameters
    ----------
    max_slippage_pct : Maximum tolerated price drift from the initial quote.
        If the best bid/ask has moved more than this percentage from the first
        observation, the function gives up on LIMIT and submits MARKET to
        guarantee a fill.
    max_attempts     : Limit-reprice iterations before falling back to MARKET.
    retry_after_s    : Seconds to wait for a fill before cancelling and repricing.
    poll_interval_s  : Order-status poll interval in seconds.

    Returns
    -------
    dict | None — ``{"order_id": str, "fill_price": float}`` on success, None on
    failure.  ``fill_price`` is the actual average price reported by Zerodha
    order_history.  Callers **must** use this value — not the original quote —
    when placing a protective GTT stoploss so the stoploss/target prices are
    anchored to the true fill cost.
    """
    import time

    if symbol in EXCLUDED_SYMBOLS:
        logger.warning("Skipping excluded symbol: %s", symbol)
        return None

    # Last-resort ex-date guard: if the execute-signal route was bypassed
    # (e.g. called directly in a script), still refuse to BUY on gap days.
    # Raises CorporateActionGapError so async callers can write a 48-h DB block.
    if transaction_type == "BUY":
        gap_hit, gap_pct = detect_corporate_action_gap(symbol, exchange)
        if gap_hit:
            raise CorporateActionGapError(symbol, gap_pct)

    kite = get_kite()
    tags = get_tag()

    # Snapshot the initial reference price for drift detection
    bid0, ask0 = _get_bid_ask(symbol, exchange)
    ref_price = ask0 if transaction_type == "BUY" else bid0
    if ref_price <= 0:
        logger.error("Cannot place order for %s — zero reference price", symbol)
        return None

    max_drift = ref_price * max_slippage_pct / 100
    attempt = 0
    current_order_id: str | None = None

    while attempt < max_attempts:
        attempt += 1

        bid, ask = _get_bid_ask(symbol, exchange)
        limit_price = ask if transaction_type == "BUY" else bid

        # Drift check
        if abs(limit_price - ref_price) > max_drift:
            logger.warning(
                "%s %s: price drifted %.3f%% beyond tolerance — switching to MARKET",
                transaction_type, symbol, abs(limit_price - ref_price) / ref_price * 100,
            )
            break

        try:
            current_order_id = str(kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type="LIMIT",
                price=round(limit_price, 2),
                validity="DAY",
                tag=tags[1],
            ))
            logger.info(
                "LimitChase attempt %d/%d: %s %s qty=%d @ %.2f → %s",
                attempt, max_attempts, transaction_type, symbol,
                quantity, limit_price, current_order_id,
            )
        except Exception as exc:
            logger.error("LimitChase place failed (attempt %d): %s", attempt, exc)
            time.sleep(poll_interval_s)
            continue

        # Poll until filled, externally cancelled, or timeout
        elapsed = 0.0
        while elapsed < retry_after_s:
            time.sleep(poll_interval_s)
            elapsed += poll_interval_s
            status = _order_status(current_order_id)
            if status == "COMPLETE":
                fill_price = _get_fill_price(current_order_id, fallback=limit_price)
                logger.info(
                    "LimitChase FILLED: %s %s qty=%d @ %.2f (attempt %d)",
                    transaction_type, symbol, quantity, fill_price, attempt,
                )
                return {"order_id": current_order_id, "fill_price": fill_price}
            if status in ("CANCELLED", "REJECTED"):
                logger.warning("Order %s was %s externally", current_order_id, status)
                current_order_id = None
                break

        # Before cancelling, check for partial fills so the re-submit uses
        # only the *remaining* (unfilled) quantity — prevents over-leveraging.
        if current_order_id:
            try:
                history = kite.order_history(order_id=current_order_id)
                if history:
                    filled = int(history[-1].get("filled_quantity", 0))
                    if filled > 0:
                        logger.info(
                            "LimitChase partial fill: %d/%d shares already filled for %s",
                            filled, quantity, current_order_id,
                        )
                        quantity -= filled
                        if quantity <= 0:
                            logger.info(
                                "Order %s fully filled across partial fills — done.",
                                current_order_id,
                            )
                            fill_price = _get_fill_price(current_order_id, fallback=limit_price)
                            return {"order_id": current_order_id, "fill_price": fill_price}
            except Exception as hist_exc:
                logger.error(
                    "Failed to fetch order history for %s: %s", current_order_id, hist_exc
                )

            _cancel_order(current_order_id)
            current_order_id = None

    # ── MARKET fallback ───────────────────────────────────────────────
    logger.warning(
        "LimitChase exhausted %d attempts for %s %s — falling back to MARKET (remaining qty=%d)",
        max_attempts, transaction_type, symbol, quantity,
    )
    try:
        market_id = str(kite.place_order(
            variety="regular",
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=product,
            order_type="MARKET",
            validity="DAY",
            tag=tags[1],
        ))
        logger.info(
            "LimitChase MARKET fallback: %s %s qty=%d → %s",
            transaction_type, symbol, quantity, market_id,
        )
        # For a MARKET order we use the last traded price as fill_price
        fill_price = _get_fill_price(market_id, fallback=get_ltp(symbol, exchange))
        return {"order_id": market_id, "fill_price": fill_price}
    except Exception as exc:
        logger.error("LimitChase MARKET fallback failed for %s: %s", symbol, exc)
        return None


# ── Async Wrappers ─────────────────────────────────────────────────────────────
# All KiteConnect network calls are synchronous (requests-based).  These wrappers
# run each blocking function in a thread-pool via asyncio.to_thread so that the
# FastAPI event loop is never blocked.

async def async_get_ltp(symbol: str, exchange: str = "NSE") -> float:
    return await asyncio.to_thread(get_ltp, symbol, exchange)


async def async_get_holdings() -> list[dict]:
    return await asyncio.to_thread(get_holdings)


async def async_get_positions() -> list[dict]:
    return await asyncio.to_thread(get_positions)


async def async_build_portfolio_snapshot() -> dict:
    return await asyncio.to_thread(build_portfolio_snapshot)


async def async_generate_session(request_token: str) -> dict:
    return await asyncio.to_thread(generate_session, request_token)


async def async_detect_corporate_action_gap(
    symbol: str,
    exchange: str = "NSE",
    threshold_pct: float | None = None,
) -> tuple[bool, float]:
    return await asyncio.to_thread(detect_corporate_action_gap, symbol, exchange, threshold_pct)


async def async_place_order(
    symbol: str,
    transaction_type: str,
    quantity: int,
    price: float | None = None,
    exchange: str = "NSE",
    product: str = "CNC",
    order_type: str = "LIMIT",
    variety: str = "regular",
) -> str | None:
    return await asyncio.to_thread(
        place_order, symbol, transaction_type, quantity,
        price, exchange, product, order_type, variety,
    )


async def async_place_gtt_order(
    symbol: str,
    exchange: str,
    avg_price: float,
    quantity: int,
    sell_pct: float,
    stoploss_pct: float,
    tick_size: float = 0.05,
) -> int | None:
    return await asyncio.to_thread(
        place_gtt_order, symbol, exchange, avg_price, quantity,
        sell_pct, stoploss_pct, tick_size,
    )


async def async_place_limit_chase_order(
    symbol: str,
    transaction_type: str,
    quantity: int,
    exchange: str = "NSE",
    product: str = "CNC",
    max_attempts: int = 5,
) -> dict | None:
    return await asyncio.to_thread(
        place_limit_chase_order, symbol, transaction_type,
        quantity, exchange, product, max_attempts,
    )


async def async_exit_all_positions() -> list[str]:
    return await asyncio.to_thread(exit_all_positions)

