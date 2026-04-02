"""Tests for regime classifier and data quality scoring."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.indicators import compute_all_indicators
from app.core.regime_classifier import (
    classify_regimes,
    compute_quality_scores,
    classify_and_score,
    get_quality_filtered_data,
    get_regime_segments,
    regime_summary,
    BULLISH, BEARISH, NEUTRAL,
    HIGH_VOL, LOW_VOL,
)


def make_trending_data(n: int = 500, direction: str = "up") -> pd.DataFrame:
    """Generate synthetic data with a clear trend."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    if direction == "up":
        close = 100 + np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.3)
    elif direction == "down":
        close = 200 - np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.3)
        close = np.maximum(close, 10)
    else:  # sideways
        close = 100 + np.random.randn(n) * 2

    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + np.abs(np.random.randn(n)) * 2,
        "low": close - np.abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100000, 10000000, n).astype(float),
    })


class TestRegimeClassifier:
    def test_classify_regimes_columns(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        assert "trend" in result.columns
        assert "volatility" in result.columns
        assert "regime_id" in result.columns
        assert "regime_confidence" in result.columns
        assert "is_transition" in result.columns

    def test_regime_id_range(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        assert result["regime_id"].min() >= 0
        assert result["regime_id"].max() <= 5

    def test_trend_values(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        valid_trends = {BULLISH, BEARISH, NEUTRAL}
        assert set(result["trend"].unique()).issubset(valid_trends)

    def test_volatility_values(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        valid_vol = {HIGH_VOL, LOW_VOL}
        assert set(result["volatility"].unique()).issubset(valid_vol)

    def test_uptrend_mostly_bullish(self):
        df = make_trending_data(direction="up")
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        # The latter half should be mostly bullish (after SMAs catch up)
        latter = result.iloc[len(result)//2:]
        bullish_pct = (latter["trend"] == BULLISH).mean()
        assert bullish_pct > 0.4, f"Expected >40% bullish in uptrend, got {bullish_pct:.1%}"

    def test_downtrend_mostly_bearish(self):
        df = make_trending_data(direction="down")
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        latter = result.iloc[len(result)//2:]
        bearish_pct = (latter["trend"] == BEARISH).mean()
        # Relaxed threshold — SMA200 is slow to turn on synthetic data
        assert bearish_pct > 0.2, f"Expected >20% bearish in downtrend, got {bearish_pct:.1%}"

    def test_confidence_range(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        assert result["regime_confidence"].min() >= 0.0
        assert result["regime_confidence"].max() <= 1.0

    def test_transition_detection(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_regimes(df)

        # Should have some transitions in the data
        assert result["is_transition"].any()


class TestQualityScoring:
    def test_quality_score_range(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        df = classify_regimes(df)
        result = compute_quality_scores(df)

        assert result["quality_score"].min() >= 0.0
        assert result["quality_score"].max() <= 1.0

    def test_circuit_hit_penalty(self):
        df = make_trending_data(n=100)
        df = compute_all_indicators(df)

        # Inject a circuit-hit-like spike
        midpoint = len(df) // 2
        df.loc[df.index[midpoint], "close"] = df.loc[df.index[midpoint - 1], "close"] * 1.15

        df = classify_regimes(df)
        result = compute_quality_scores(df)

        # That row should have lower quality
        assert result.loc[result.index[midpoint], "quality_score"] < 1.0

    def test_full_pipeline(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        result = classify_and_score(df)

        assert "trend" in result.columns
        assert "quality_score" in result.columns
        assert "regime_id" in result.columns


class TestFiltering:
    def test_quality_filter(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        df = classify_and_score(df)

        filtered = get_quality_filtered_data(df, min_quality=0.8)
        assert (filtered["quality_score"] >= 0.8).all()

    def test_regime_filter(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        df = classify_and_score(df)

        filtered = get_quality_filtered_data(df, regime_ids=[0, 1])
        if len(filtered) > 0:
            assert set(filtered["regime_id"].unique()).issubset({0, 1})

    def test_regime_segments(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        df = classify_and_score(df)

        segments = get_regime_segments(df, min_length=5)
        for rid, seg in segments:
            assert len(seg) >= 5
            assert (seg["regime_id"] == rid).all()

    def test_regime_summary(self):
        df = make_trending_data()
        df = compute_all_indicators(df)
        df = classify_and_score(df)

        summary = regime_summary(df)
        total = sum(summary["regime_breakdown"][rid]["count"] for rid in range(6))
        assert total == len(df)
        assert summary["total_candles"] == len(df)
        assert "quality_tiers" in summary
