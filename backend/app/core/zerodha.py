"""KiteConnect wrapper with configurable IP routing.

Ported from pytrade's pyTrade_common.py — modernized for async FastAPI.
NSE only, CNC orders only (no MIS/intraday).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

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
