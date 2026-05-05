"""SQLAlchemy ORM models — all 16 tables from ARCHITECTURE.md."""
from __future__ import annotations

import enum
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    """Current time in Asia/Kolkata (IST, UTC+5:30).

    Use as the ``default`` for all DateTime columns so timestamps stored in
    MySQL always represent IST wall-clock time, preventing off-by-one day
    errors in nightly reconciliation and date-filtered queries.
    """
    return datetime.now(IST)

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


# ── Enums ──────────────────────────────────────────────────────────────

class IntervalEnum(str, enum.Enum):
    day = "day"
    week = "week"


class TrendEnum(str, enum.Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class VolatilityEnum(str, enum.Enum):
    high = "high"
    low = "low"


class ModelStatus(str, enum.Enum):
    pending = "pending"
    training = "training"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"
    paused = "paused"


class OrderStatus(str, enum.Enum):
    pending = "pending"
    placed = "placed"        # legacy alias — prefer submitted
    completed = "completed"  # legacy alias — prefer filled
    cancelled = "cancelled"
    rejected = "rejected"
    submitted = "submitted"   # sent to broker, awaiting confirmation
    partial_fill = "partial_fill"  # partially filled
    filled = "filled"         # fully filled & confirmed


# ── Stocks & Calendar ──────────────────────────────────────────────────

class Stock(Base):
    __tablename__ = "stocks_list"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")
    kite_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tick_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class NSEHoliday(Base):
    __tablename__ = "nse_holidays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trading_date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    week_day: Mapped[str | None] = mapped_column(String(20), nullable=True)
    description: Mapped[str | None] = mapped_column(String(200), nullable=True)


# ── OHLCV Cache ────────────────────────────────────────────────────────

class StockOHLCV(Base):
    __tablename__ = "stock_ohlcv"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "interval", name="uq_ohlcv"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)


class IndexOHLCV(Base):
    """OHLCV cache for market indices (NIFTY 50, BANK NIFTY, etc.).

    Separate from StockOHLCV: indices have no stock_id FK and are identified
    only by their symbol string (e.g. "NIFTY 50").
    """
    __tablename__ = "index_ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "date", "interval", name="uq_index_ohlcv"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, default="day")
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)


# ── Indicators Cache ───────────────────────────────────────────────────

class StockIndicator(Base):
    __tablename__ = "stock_indicators"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "interval", name="uq_indicator"),
        Index("ix_indicator_stock_date", "stock_id", "date", "interval"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)

    # SMA
    sma_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_12: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_24: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_100: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    # EMA
    ema_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    # RSI
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    srsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    # MACD
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ADX
    adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_pos: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_neg: Mapped[float | None] = mapped_column(Float, nullable=True)
    # KAMA
    kama: Mapped[float | None] = mapped_column(Float, nullable=True)
    vwkama: Mapped[float | None] = mapped_column(Float, nullable=True)
    # OBV
    obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Bollinger Bands
    bb_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    # TGRB candle structure
    tgrb_top: Mapped[float | None] = mapped_column(Float, nullable=True)
    tgrb_green: Mapped[float | None] = mapped_column(Float, nullable=True)
    tgrb_red: Mapped[float | None] = mapped_column(Float, nullable=True)
    tgrb_bottom: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ATR (for regime classifier)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ── New stationary ML features (additive) ─────────────────────────────
    # Stationary trend distance
    dist_sma_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_sma_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Rate of change (momentum)
    roc_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Volatility regime
    atr_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_vol_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Bollinger Band position & width
    bb_pctb: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Institutional flow & liquidity
    cmf_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_vwap_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Normalised signals
    macd_hist_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ADX normalised to 0-1 (raw adx/adx_pos/adx_neg kept for regime_classifier)
    adx_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_pos_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_neg_norm: Mapped[float | None] = mapped_column(Float, nullable=True)


# ── Regime Classification ──────────────────────────────────────────────

class StockRegime(Base):
    __tablename__ = "stock_regimes"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "interval", name="uq_regime"),
        Index("ix_regime_stock_date", "stock_id", "date", "interval"),
        Index("ix_regime_id_date", "regime_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    trend: Mapped[TrendEnum] = mapped_column(Enum(TrendEnum), nullable=False)
    volatility: Mapped[VolatilityEnum] = mapped_column(Enum(VolatilityEnum), nullable=False)
    regime_id: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-5
    regime_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_transition: Mapped[bool] = mapped_column(Boolean, default=False)


# ── RL Models ──────────────────────────────────────────────────────────

class RLModel(Base):
    __tablename__ = "rl_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)  # PPO, RecurrentPPO, A2C, DDPG, TD3, SAC
    hyperparams: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    regime_filter: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    total_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


class RLTrainingRun(Base):
    __tablename__ = "rl_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    timestep: Mapped[int] = mapped_column(Integer, nullable=False)
    episode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Golden Patterns ────────────────────────────────────────────────────

class GoldenPattern(Base):
    __tablename__ = "golden_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    dataset_filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=BUY, -1=SELL, 0=HOLD
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_at_capture: Mapped[float | None] = mapped_column(Float, nullable=True)  # raw ATR % at pattern date


# ── KNN Models ─────────────────────────────────────────────────────────

class KNNModel(Base):
    __tablename__ = "knn_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    k_neighbors: Mapped[int] = mapped_column(Integer, default=5)
    feature_combination: Mapped[str | None] = mapped_column(String(200), nullable=True)
    seq_len: Mapped[int] = mapped_column(Integer, default=15)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    regime_filter: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    norm_params_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── LSTM Models ────────────────────────────────────────────────────────

class LSTMModel(Base):
    __tablename__ = "lstm_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    hidden_size: Mapped[int] = mapped_column(Integer, default=128)
    num_layers: Mapped[int] = mapped_column(Integer, default=2)
    dropout: Mapped[float] = mapped_column(Float, default=0.3)
    seq_len: Mapped[int] = mapped_column(Integer, default=15)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    regime_filter: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    norm_params_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Ensemble ───────────────────────────────────────────────────────────

class EnsembleConfig(Base):
    __tablename__ = "ensemble_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    knn_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("knn_models.id"), nullable=False)
    lstm_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("lstm_models.id"), nullable=False)
    knn_weight: Mapped[float] = mapped_column(Float, default=0.5)
    lstm_weight: Mapped[float] = mapped_column(Float, default=0.5)
    agreement_required: Mapped[bool] = mapped_column(Boolean, default=True)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    backtest_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Predictions (DB cached) ───────────────────────────────────────────

class KNNPrediction(Base):
    __tablename__ = "knn_predictions"
    __table_args__ = (
        UniqueConstraint("knn_model_id", "stock_id", "date", "interval", name="uq_knn_pred"),
        Index("ix_knn_pred_date_action", "date", "action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knn_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("knn_models.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    action: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=BUY, 0=HOLD, -1=SELL
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    proba_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    proba_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    proba_hold: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matched_pattern_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class LSTMPrediction(Base):
    __tablename__ = "lstm_predictions"
    __table_args__ = (
        UniqueConstraint("lstm_model_id", "stock_id", "date", "interval", name="uq_lstm_pred"),
        Index("ix_lstm_pred_date_conf", "date", "interval", "confidence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lstm_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("lstm_models.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    action: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    proba_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    proba_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    proba_hold: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class EnsemblePrediction(Base):
    __tablename__ = "ensemble_predictions"
    __table_args__ = (
        UniqueConstraint("batch_id", "stock_id", name="uq_ensemble_batch_stock"),
        Index("ix_ensemble_pred_batch", "batch_id"),
        Index("ix_ensemble_pred_date_conf", "date", "interval", "confidence"),
        Index("ix_ensemble_pred_run_at", "run_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(50), nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)
    ensemble_config_id: Mapped[int] = mapped_column(Integer, ForeignKey("ensemble_configs.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    action: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    knn_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    knn_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    lstm_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lstm_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    agreement: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


# ── Per-Stock Ensemble Weight Calibration ────────────────────────────

class StockEnsembleWeights(Base):
    """Per-stock KNN/LSTM weight calibration, computed by per_stock_optimal_weights().

    Rows are keyed by (ensemble_config_id, stock_id).  A new row is written
    each time the calibration job runs; the most-recent row is used at
    prediction time.  This lets each stock carry its own optimal ratio
    rather than relying on the global EnsembleConfig defaults.
    """
    __tablename__ = "stock_ensemble_weights"
    __table_args__ = (
        Index("ix_sew_config_stock", "ensemble_config_id", "stock_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ensemble_config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ensemble_configs.id"), nullable=False
    )
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("stocks_list.id"), nullable=False
    )
    knn_weight: Mapped[float] = mapped_column(Float, nullable=False)
    lstm_weight: Mapped[float] = mapped_column(Float, nullable=False)
    calibration_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Backtest Results ───────────────────────────────────────────────────

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)  # rl, knn, lstm, ensemble
    model_id: Mapped[int] = mapped_column(Integer, nullable=False)
    stock_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=True)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    trades_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trade_log: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)



class CompoundBacktestResult(Base):
    __tablename__ = "compound_backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    final_capital: Mapped[float] = mapped_column(Float, nullable=False)
    profit_booked: Mapped[float] = mapped_column(Float, nullable=False)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)  # SL, target, regimes, max_positions
    equity_curve: Mapped[dict] = mapped_column(JSON, nullable=False)
    trade_log: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Trade Orders ───────────────────────────────────────────────────────

class TradeOrder(Base):
    __tablename__ = "trade_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    ensemble_prediction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ensemble_predictions.id"), nullable=True
    )
    trade_signal_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("trade_signals.id"), nullable=True
    )
    variety: Mapped[str] = mapped_column(String(20), default="regular")  # regular, amo, gtt
    transaction_type: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.pending)
    zerodha_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tag: Mapped[str | None] = mapped_column(String(20), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_ist)
    filled_quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_reconciled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ── App Settings ───────────────────────────────────────────────────────

class AppSetting(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    property: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)


# ── Pipeline Jobs (Autopilot Persistence) ──────────────────────────────

class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # UUID
    status: Mapped[str] = mapped_column(String(50), default="pending")
    current_stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stages: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # List of stage names
    symbols: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # List of symbols
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist)


# ── Daily Portfolio Reconciliation Snapshot ────────────────────────────
# Written by the 08:30 IST reconciliation job.  The live position-sizing
# algorithms (Kelly, Vol-Target) must read cash and holdings exclusively
# from the most recent row here \u2014 never from a local in-memory ledger.

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (Index("ix_portfolio_snapshot_date", "snapshot_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    # Cash figures fetched directly from Kite margins API
    cash_available: Mapped[float] = mapped_column(Float, nullable=False)
    opening_balance: Mapped[float] = mapped_column(Float, nullable=False)
    # MTM value of all CNC holdings at the time of reconciliation
    holdings_value: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    # Full broker-sourced snapshots stored as JSON for audit
    holdings_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    positions_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    reconciled_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Corporate Action Block ──────────────────────────────────────────────
# Persists a 48-hour trading block when an ex-date gap is detected for a
# symbol.  Checked at every BUY entry-point; written the first time the
# gap is detected so subsequent restarts still honour the cooldown.

class CorporateActionBlock(Base):
    __tablename__ = "corporate_action_blocks"
    __table_args__ = (
        Index("ix_ca_block_symbol_expiry", "symbol", "blocked_until"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False, default="NSE")
    # Detected open-vs-prev-close gap that triggered the block
    gap_pct: Mapped[float] = mapped_column(Float, nullable=False)
    blocked_until: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Prediction Jobs (Async Tracking) ──────────────────────────────────

class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # UUID
    status: Mapped[str] = mapped_column(String(20), default="running")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total_stocks: Mapped[int] = mapped_column(Integer, default=0)
    completed_stocks: Mapped[int] = mapped_column(Integer, default=0)
    batch_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist)


# ── Point-in-Time Fundamental Data ────────────────────────────────────
# PIT guarantee: each row represents what was *known* on that specific date.
# Never backfill past dates — only insert forward as new data arrives.

class StockFundamentalPIT(Base):
    """Daily snapshot of fundamental metrics — one row per (stock, date)."""
    __tablename__ = "stock_fundamentals_pit"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_fundamental_pit"),
        Index("ix_fundamental_pit_stock_date", "stock_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    # Valuation ratios
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)       # trailingPE
    forward_pe: Mapped[float | None] = mapped_column(Float, nullable=True)     # forwardPE
    pb_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)       # priceToBook
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True) # dividendYield
    # Quality metrics
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)            # returnOnEquity (0–1)
    debt_equity: Mapped[float | None] = mapped_column(Float, nullable=True)    # debtToEquity
    # Source tag: 'yfinance' | 'nsepython'
    source: Mapped[str] = mapped_column(String(20), default="yfinance")
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


class FundamentalSectorStats(Base):
    """Daily sector-level PE aggregates for cross-sectional z-score computation."""
    __tablename__ = "fundamental_sector_stats"
    __table_args__ = (
        UniqueConstraint("sector", "date", name="uq_sector_stats"),
        Index("ix_sector_stats_date", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sector: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    sector_pe_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_pe_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    stock_count: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


class StockFundamentalZScore(Base):
    """Derived, ML-ready bounded z-scores — refreshed nightly."""
    __tablename__ = "stock_fundamental_zscores"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_fundamental_zscore"),
        Index("ix_fundamental_zscore_stock_date", "stock_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    # Z-scores: bounded ±3, zero-filled when insufficient history
    pe_zscore_3y: Mapped[float | None] = mapped_column(Float, nullable=True)    # vs stock's own 3yr rolling mean
    pe_zscore_sector: Mapped[float | None] = mapped_column(Float, nullable=True) # vs sector peers today
    # Normalised quality signals (0–1)
    roe_norm: Mapped[float | None] = mapped_column(Float, nullable=True)         # ROE clipped 0–100%, then /100
    debt_equity_norm: Mapped[float | None] = mapped_column(Float, nullable=True) # D/E clipped 0–5, then /5, inverted
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Sentiment Data ─────────────────────────────────────────────────────

class StockSentiment(Base):
    """Daily aggregated news sentiment per stock — produced by the
    FinBERT-triage + LLM-judgment pipeline."""
    __tablename__ = "stock_sentiment"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_sentiment"),
        Index("ix_sentiment_stock_date", "stock_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    headline_count: Mapped[int] = mapped_column(Integer, default=0)
    neutral_filtered_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_finbert_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # raw FinBERT [-1,1]
    llm_impact_score: Mapped[float | None] = mapped_column(Float, nullable=True)   # LLM impact [-1,1]
    llm_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Multi-Horizon LSTM Predictions ────────────────────────────────────

class LSTMHorizonModel(Base):
    """Registry entry for a trained multi-horizon LSTM model."""
    __tablename__ = "lstm_horizon_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_rl_model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=True)
    hidden_size: Mapped[int] = mapped_column(Integer, default=256)
    num_layers: Mapped[int] = mapped_column(Integer, default=2)
    seq_len: Mapped[int] = mapped_column(Integer, default=15)
    horizon: Mapped[int] = mapped_column(Integer, default=10)   # forecast steps
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    norm_params_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


class LSTMHorizonPrediction(Base):
    """10-day rolling forecast sequences — one row per (model, stock, date)."""
    __tablename__ = "lstm_horizon_predictions"
    __table_args__ = (
        UniqueConstraint("model_id", "stock_id", "prediction_date", name="uq_horizon_pred"),
        Index("ix_horizon_pred_date", "prediction_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("lstm_horizon_models.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False)
    # Per-horizon action (0=HOLD, 1=BUY, 2=SELL) and confidence (0-1)
    h1_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h1_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h2_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h2_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h3_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h3_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h4_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h4_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h5_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h5_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h6_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h6_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h7_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h7_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h8_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h8_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h9_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h9_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    h10_action: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h10_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Fraction of h1–h10 steps agreeing with h1 direction
    trend_durability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)


# ── Regime-Stratified Ensemble Map ────────────────────────────────────

class RegimeEnsembleMap(Base):
    """Maps each of the 6 regime IDs to its dedicated EnsembleConfig.

    The prediction engine uses this table to select the correct
    regime-stratified model pair rather than the global ensemble.
    Falls back to global ensemble when no row exists for a regime.
    """
    __tablename__ = "regime_ensemble_map"

    regime_id: Mapped[int] = mapped_column(Integer, primary_key=True)   # 0-5
    ensemble_config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ensemble_configs.id"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist)


# ── Trade Signals (Target-Price Pipeline Output) ──────────────────────

class SignalStatus(str, enum.Enum):
    pending = "pending"        # signal generated, not yet executed
    active = "active"          # order placed, position open
    target_hit = "target_hit"  # closed at target
    sl_hit = "sl_hit"          # closed at stoploss
    expired = "expired"        # κ-decay made R:R unviable, signal dropped
    cancelled = "cancelled"    # manually cancelled


class TradeSignal(Base):
    """A BUY signal with target price, stoploss, and meta-classifier score.

    Produced by the signal synthesiser; consumed by the OMS for order
    placement and by the trailing-stop engine for dynamic SL updates.
    """
    __tablename__ = "trade_signals"
    __table_args__ = (
        UniqueConstraint("stock_id", "signal_date", name="uq_signal_stock_date"),
        Index("ix_signal_status", "status"),
        Index("ix_signal_date", "signal_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)

    # ── Price levels ──
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    stoploss_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_stoploss: Mapped[float | None] = mapped_column(Float, nullable=True)  # trailing SL

    # ── Confidence & quality scores ──
    pop_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # meta-classifier PoP (0-1)
    fqs_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # fundamental quality (0-1)
    confluence_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # raw ensemble confluence
    execution_cost_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # spread+slip as % of price

    # ── κ-decay tracking ──
    initial_rr_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_rr_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_since_signal: Mapped[int] = mapped_column(Integer, default=0)

    # ── Model source references ──
    lstm_mu: Mapped[float | None] = mapped_column(Float, nullable=True)       # LSTM predicted mean return
    lstm_sigma: Mapped[float | None] = mapped_column(Float, nullable=True)    # LSTM predicted uncertainty
    knn_median_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    knn_win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Trailing stop state ──
    is_trailing_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trailing_updates_count: Mapped[int] = mapped_column(Integer, default=0)
    last_gtt_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # ── Status & metadata ──
    status: Mapped[SignalStatus] = mapped_column(Enum(SignalStatus), default=SignalStatus.pending)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist)
