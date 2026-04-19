"""Support / Resistance zone detection engine.

Combines four methods to identify price zones where buying or selling
pressure historically concentrates, then merges and ranks them.

Methods
-------
1. Fractal Pivots   – Williams-style N-bar high/low pivots
2. Fibonacci        – Retracements from swing H/L (23.6 / 38.2 / 50 / 61.8)
3. Volume Profile   – KDE of volume-weighted price → peaks = HVN zones
4. Pivot Points     – Classic floor-trader pivots (S1/S2/R1/R2 + POC)

The public API is ``compute_sr_zones(df, current_price, atr)`` which
returns an ``SRResult`` with merged zones, nearest S/R, and breakout
projections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────

class ZoneType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class DetectionMethod(str, Enum):
    FRACTAL = "fractal"
    FIBONACCI = "fibonacci"
    VOLUME_PROFILE = "volume_profile"
    PIVOT_POINTS = "pivot_points"


@dataclass
class SRZone:
    """A single support or resistance zone."""
    price_low: float
    price_high: float
    zone_type: ZoneType
    strength: float  # 0-1 normalised score
    methods: List[DetectionMethod] = field(default_factory=list)
    touch_count: int = 1
    last_tested_idx: int = 0

    @property
    def midpoint(self) -> float:
        return (self.price_low + self.price_high) / 2.0

    @property
    def width(self) -> float:
        return self.price_high - self.price_low


@dataclass
class SRResult:
    """Output of the S/R engine for a single stock."""
    zones: List[SRZone]
    nearest_support: Optional[SRZone]
    nearest_resistance: Optional[SRZone]
    # Distance as fraction of current price
    support_distance_pct: float = 0.0
    resistance_distance_pct: float = 0.0
    # Breakout projections (measured-move targets)
    upside_breakout_target: float = 0.0
    downside_breakout_target: float = 0.0


# ── 1. Fractal Pivots ────────────────────────────────────────────────────

def _fractal_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    n: int = 5,
) -> tuple[list[float], list[float]]:
    """Williams fractal: a local high/low that is the extreme of 2n+1 bars."""
    supports: list[float] = []
    resistances: list[float] = []
    length = len(highs)
    for i in range(n, length - n):
        # Resistance: bar i has the highest high in window
        if highs[i] == np.max(highs[i - n : i + n + 1]):
            resistances.append(float(highs[i]))
        # Support: bar i has the lowest low in window
        if lows[i] == np.min(lows[i - n : i + n + 1]):
            supports.append(float(lows[i]))
    return supports, resistances


# ── 2. Fibonacci Retracements ─────────────────────────────────────────────

_FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


def _fibonacci_levels(
    swing_high: float,
    swing_low: float,
) -> list[float]:
    """Fibonacci retracement levels between a swing high and swing low."""
    span = swing_high - swing_low
    if span <= 0:
        return []
    return [swing_high - ratio * span for ratio in _FIB_RATIOS]


# ── 3. Volume Profile (KDE) ──────────────────────────────────────────────

def _volume_profile_zones(
    closes: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = 80,
    peak_prominence: float = 0.15,
) -> list[float]:
    """Find High-Volume Nodes (HVN) via a volume-weighted price histogram.

    Instead of scipy KDE (heavy dependency), we use a simple histogram +
    Gaussian smoothing approach.
    """
    if len(closes) < 20:
        return []

    price_min, price_max = float(np.min(closes)), float(np.max(closes))
    if price_max <= price_min:
        return []

    # Apply exponential time-decay: older volume gets ~13% weight, recent gets 100%
    decay_factor = np.exp(np.linspace(-2.0, 0.0, len(volumes)))
    decayed_volumes = volumes * decay_factor

    # Build volume-weighted histogram
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    bin_width = (price_max - price_min) / n_bins

    for price, vol in zip(closes, decayed_volumes):
        idx = int((price - price_min) / bin_width)
        idx = min(idx, n_bins - 1)
        hist[idx] += vol

    # Gaussian smooth (σ=2 bins) to reduce noise
    kernel_size = 5
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / 2.0)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")

    # Find peaks (local maxima above prominence threshold)
    max_val = smoothed.max()
    if max_val == 0:
        return []

    threshold = max_val * peak_prominence
    peaks: list[float] = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
            and smoothed[i] > threshold
        ):
            # Price at bin centre
            peaks.append(price_min + (i + 0.5) * bin_width)

    return peaks


# ── 4. Classic Pivot Points ───────────────────────────────────────────────

def _pivot_points(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> dict[str, float]:
    """Classic floor-trader pivot levels from the prior period."""
    pp = (prev_high + prev_low + prev_close) / 3.0
    return {
        "PP": pp,
        "S1": 2 * pp - prev_high,
        "S2": pp - (prev_high - prev_low),
        "R1": 2 * pp - prev_low,
        "R2": pp + (prev_high - prev_low),
    }


# ── Zone merging ─────────────────────────────────────────────────────────

def _merge_zones(
    zones: list[SRZone],
    atr: float,
    merge_tolerance_atr: float = 0.5,
) -> list[SRZone]:
    """Merge overlapping zones within ``merge_tolerance_atr × ATR``."""
    if not zones:
        return []

    tol = atr * merge_tolerance_atr
    sorted_zones = sorted(zones, key=lambda z: z.midpoint)
    merged: list[SRZone] = [sorted_zones[0]]

    for zone in sorted_zones[1:]:
        prev = merged[-1]
        if zone.midpoint - prev.midpoint <= tol:
            # Merge: widen bounds, accumulate strength/methods/touches
            prev.price_low = min(prev.price_low, zone.price_low)
            prev.price_high = max(prev.price_high, zone.price_high)
            prev.strength = min(1.0, prev.strength + zone.strength * 0.5)
            prev.touch_count += zone.touch_count
            for m in zone.methods:
                if m not in prev.methods:
                    prev.methods.append(m)
        else:
            merged.append(zone)

    return merged


# ── Touch-count refresh ──────────────────────────────────────────────────

def _count_touches(
    zones: list[SRZone],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: float,
    touch_tolerance_atr: float = 0.3,
) -> None:
    """Count how many times price has tested each zone (in-place update)."""
    tol = atr * touch_tolerance_atr
    for zone in zones:
        mid = zone.midpoint
        count = 0
        last_idx = 0
        for i in range(len(closes)):
            # Price came within tolerance of the zone midpoint
            if abs(lows[i] - mid) <= tol or abs(highs[i] - mid) <= tol:
                count += 1
                last_idx = i
        zone.touch_count = max(zone.touch_count, count)
        zone.last_tested_idx = last_idx


# ── Polarity classification ──────────────────────────────────────────────

def _classify_polarity(
    zones: list[SRZone],
    current_price: float,
) -> None:
    """Assign support/resistance based on position relative to price.

    A broken support becomes resistance (polarity flip) and vice versa.
    """
    for zone in zones:
        if zone.midpoint < current_price:
            zone.zone_type = ZoneType.SUPPORT
        else:
            zone.zone_type = ZoneType.RESISTANCE


# ── Main public API ──────────────────────────────────────────────────────

def compute_sr_zones(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    *,
    lookback: int = 250,
    fractal_n: int = 5,
    vp_bins: int = 80,
) -> SRResult:
    """Compute merged S/R zones from the last ``lookback`` bars.

    Parameters
    ----------
    df : DataFrame
        Must contain ``open, high, low, close, volume`` columns.
    current_price : float
        Latest traded price (for polarity classification).
    atr : float
        Current 14-day ATR (used as merge tolerance unit).
    lookback : int
        Number of recent bars to analyse.
    fractal_n : int
        Half-window size for Williams fractals.
    vp_bins : int
        Number of histogram bins for volume profile.

    Returns
    -------
    SRResult
    """
    tail = df.tail(lookback)
    if len(tail) < 30:
        return SRResult(zones=[], nearest_support=None, nearest_resistance=None)

    highs = tail["high"].values.astype(np.float64)
    lows = tail["low"].values.astype(np.float64)
    closes = tail["close"].values.astype(np.float64)
    volumes = tail["volume"].values.astype(np.float64)

    half_atr = atr * 0.5  # zone width

    raw_zones: list[SRZone] = []

    # ── 1. Fractal pivots ─────────────────────────────────────────────
    sup_prices, res_prices = _fractal_pivots(highs, lows, n=fractal_n)
    for p in sup_prices:
        raw_zones.append(SRZone(
            price_low=p - half_atr,
            price_high=p + half_atr,
            zone_type=ZoneType.SUPPORT,
            strength=0.3,
            methods=[DetectionMethod.FRACTAL],
        ))
    for p in res_prices:
        raw_zones.append(SRZone(
            price_low=p - half_atr,
            price_high=p + half_atr,
            zone_type=ZoneType.RESISTANCE,
            strength=0.3,
            methods=[DetectionMethod.FRACTAL],
        ))

    # ── 2. Fibonacci from swing H/L of lookback ──────────────────────
    swing_high = float(np.max(highs))
    swing_low = float(np.min(lows))
    fib_levels = _fibonacci_levels(swing_high, swing_low)
    for p in fib_levels:
        raw_zones.append(SRZone(
            price_low=p - half_atr,
            price_high=p + half_atr,
            zone_type=ZoneType.SUPPORT,  # re-classified by polarity
            strength=0.25,
            methods=[DetectionMethod.FIBONACCI],
        ))

    # ── 3. Volume profile ─────────────────────────────────────────────
    vp_levels = _volume_profile_zones(closes, volumes, n_bins=vp_bins)
    for p in vp_levels:
        raw_zones.append(SRZone(
            price_low=p - half_atr,
            price_high=p + half_atr,
            zone_type=ZoneType.SUPPORT,
            strength=0.35,
            methods=[DetectionMethod.VOLUME_PROFILE],
        ))

    # ── 4. Floor-trader pivots (from last 20 bars as "prior period") ──
    recent = tail.tail(20)
    if len(recent) >= 5:
        pp_levels = _pivot_points(
            prev_high=float(recent["high"].max()),
            prev_low=float(recent["low"].min()),
            prev_close=float(recent["close"].iloc[-1]),
        )
        for _label, p in pp_levels.items():
            raw_zones.append(SRZone(
                price_low=p - half_atr,
                price_high=p + half_atr,
                zone_type=ZoneType.SUPPORT,
                strength=0.2,
                methods=[DetectionMethod.PIVOT_POINTS],
            ))

    # ── Merge + enrich ────────────────────────────────────────────────
    merged = _merge_zones(raw_zones, atr)
    _count_touches(merged, highs, lows, closes, atr)
    _classify_polarity(merged, current_price)

    # Boost strength for multi-method and high-touch zones
    for z in merged:
        method_bonus = min(0.3, 0.1 * (len(z.methods) - 1))
        touch_bonus = min(0.3, 0.05 * z.touch_count)
        z.strength = min(1.0, z.strength + method_bonus + touch_bonus)

    # Sort by strength descending
    merged.sort(key=lambda z: z.strength, reverse=True)

    # ── Nearest S/R ───────────────────────────────────────────────────
    supports = [z for z in merged if z.zone_type == ZoneType.SUPPORT]
    resistances = [z for z in merged if z.zone_type == ZoneType.RESISTANCE]

    nearest_sup: Optional[SRZone] = None
    nearest_res: Optional[SRZone] = None
    sup_dist_pct = 0.0
    res_dist_pct = 0.0

    if supports:
        nearest_sup = max(supports, key=lambda z: z.midpoint)
        sup_dist_pct = (current_price - nearest_sup.midpoint) / current_price

    if resistances:
        nearest_res = min(resistances, key=lambda z: z.midpoint)
        res_dist_pct = (nearest_res.midpoint - current_price) / current_price

    # ── Breakout projections (measured-move) ──────────────────────────
    upside_target = 0.0
    downside_target = 0.0
    if nearest_res is not None:
        # Measured move = distance from nearest support to resistance, added
        base = nearest_sup.midpoint if nearest_sup else current_price
        upside_target = nearest_res.midpoint + (nearest_res.midpoint - base)
    if nearest_sup is not None:
        base = nearest_res.midpoint if nearest_res else current_price
        downside_target = nearest_sup.midpoint - (base - nearest_sup.midpoint)

    return SRResult(
        zones=merged,
        nearest_support=nearest_sup,
        nearest_resistance=nearest_res,
        support_distance_pct=sup_dist_pct,
        resistance_distance_pct=res_dist_pct,
        upside_breakout_target=upside_target,
        downside_breakout_target=downside_target,
    )


# ── Feature vector for ML consumption ────────────────────────────────────

def sr_features(result: SRResult, current_price: float, atr: float) -> dict[str, float]:
    """Extract a flat dict of S/R features for the ML feature vector.

    Returns 6 features:
        sr_support_dist    – % distance to nearest support (positive = above)
        sr_resistance_dist – % distance to nearest resistance (positive = below)
        sr_support_strength  – strength of nearest support zone (0-1)
        sr_resistance_strength – strength of nearest resistance zone (0-1)
        sr_zone_count      – number of zones within ±2 ATR of price (density)
        sr_rr_ratio        – risk/reward ratio (distance-to-resistance / distance-to-support)
    """
    sup_dist = result.support_distance_pct
    res_dist = result.resistance_distance_pct
    sup_str = result.nearest_support.strength if result.nearest_support else 0.0
    res_str = result.nearest_resistance.strength if result.nearest_resistance else 0.0

    # Zone density: count zones within ±2 ATR
    zone_count = sum(
        1 for z in result.zones
        if abs(z.midpoint - current_price) <= 2.0 * atr
    )

    # R:R ratio (clamped)
    if sup_dist > 0 and res_dist > 0:
        rr = res_dist / sup_dist
    else:
        rr = 1.0
    rr = min(rr, 10.0)

    return {
        "sr_support_dist": round(sup_dist, 6),
        "sr_resistance_dist": round(res_dist, 6),
        "sr_support_strength": round(sup_str, 4),
        "sr_resistance_strength": round(res_str, 4),
        "sr_zone_count": float(zone_count),
        "sr_rr_ratio": round(rr, 4),
    }
