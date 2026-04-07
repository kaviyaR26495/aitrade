"""Tests for indicator computation and normalization."""
import numpy as np
import pandas as pd
import pytest

# Ensure backend is importable
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.indicators import (
    compute_all_indicators,
    compute_tgrb,
    compute_rsi,
    compute_sma,
    compute_bollinger,
    compute_macd,
    compute_adx,
    compute_obv,
    get_indicator_columns,
    WARMUP_ROWS,
)
from app.core.normalizer import normalize_dataframe, get_feature_columns, prepare_model_input


def make_sample_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)  # prevent negatives

    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + np.abs(np.random.randn(n)) * 2,
        "low": close - np.abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100000, 10000000, n).astype(float),
    })


class TestIndicators:
    def test_compute_all_indicators(self):
        df = make_sample_ohlcv()
        result = compute_all_indicators(df)

        # Should have dropped warmup rows
        assert len(result) == len(df) - WARMUP_ROWS

        # Should have all indicator columns
        expected_cols = get_indicator_columns()
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_tgrb_values(self):
        df = make_sample_ohlcv()
        result = compute_tgrb(df.copy())

        # TGRB values should be percentages (can be negative for Bottom)
        assert "tgrb_top" in result.columns
        assert "tgrb_green" in result.columns
        assert "tgrb_red" in result.columns
        assert "tgrb_bottom" in result.columns

        # Green and Red should be mutually exclusive (one is 0 per row)
        for _, row in result.head(10).iterrows():
            assert row["tgrb_green"] == 0 or row["tgrb_red"] == 0

    def test_rsi_range(self):
        df = make_sample_ohlcv()
        result = compute_rsi(df.copy())

        # RSI should be between 0 and 1 (divided by 100)
        valid = result["rsi"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_sma_computed(self):
        df = make_sample_ohlcv()
        result = compute_sma(df.copy())

        for w in [5, 12, 24, 50, 100, 200]:
            assert f"sma_{w}" in result.columns
            # SMA values should be close to close prices
            valid = result[f"sma_{w}"].dropna()
            assert len(valid) > 0

    def test_bollinger_bands(self):
        df = make_sample_ohlcv()
        result = compute_bollinger(df.copy())

        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_mid" in result.columns

        # Upper band > Lower band
        valid_h = result["bb_upper"].dropna()
        valid_l = result["bb_lower"].dropna()
        assert (valid_h.values[-50:] > valid_l.values[-50:]).all()

    def test_macd(self):
        df = make_sample_ohlcv()
        result = compute_macd(df.copy())

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_adx(self):
        df = make_sample_ohlcv()
        result = compute_adx(df.copy())

        assert "adx" in result.columns
        assert "adx_pos" in result.columns
        assert "adx_neg" in result.columns

    def test_obv(self):
        df = make_sample_ohlcv()
        result = compute_obv(df.copy())
        assert "obv" in result.columns

    def test_selective_groups(self):
        df = make_sample_ohlcv()
        result = compute_all_indicators(df, groups=["rsi", "sma"])

        assert "rsi" in result.columns
        assert "sma_50" in result.columns
        # Should NOT have MACD since we didn't select it
        assert "macd" not in result.columns

    def test_no_warmup_drop(self):
        df = make_sample_ohlcv()
        result = compute_all_indicators(df, drop_warmup=False)
        assert len(result) == len(df)


class TestNormalizer:
    def test_normalize_produces_float32(self):
        df = make_sample_ohlcv()
        df = compute_all_indicators(df)
        result = normalize_dataframe(df)

        float_cols = result.select_dtypes(include=[np.floating]).columns
        for col in float_cols:
            assert result[col].dtype == np.float32, f"{col} is {result[col].dtype}"

    def test_normalize_no_nans(self):
        df = make_sample_ohlcv()
        df = compute_all_indicators(df)
        result = normalize_dataframe(df)

        assert not result.isnull().any().any(), "NaN values found after normalization"

    def test_feature_columns(self):
        df = make_sample_ohlcv()
        df = compute_all_indicators(df)
        result = normalize_dataframe(df)
        feat_cols = get_feature_columns(result)

        # Should not include raw OHLCV
        assert "open" not in feat_cols
        assert "close" not in feat_cols
        assert "volume" not in feat_cols
        assert len(feat_cols) > 10

    def test_prepare_model_input_sequential(self):
        df = make_sample_ohlcv()
        df = compute_all_indicators(df)
        df = normalize_dataframe(df)

        seq_len = 15
        data = prepare_model_input(df, seq_len=seq_len)

        assert data.ndim == 3
        assert data.shape[1] == seq_len
        assert data.dtype == np.float32

    def test_prepare_model_input_flat(self):
        df = make_sample_ohlcv()
        df = compute_all_indicators(df)
        df = normalize_dataframe(df)

        data = prepare_model_input(df, seq_len=1)

        assert data.ndim == 2
        assert data.dtype == np.float32
