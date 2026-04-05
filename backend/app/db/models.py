"""SQLAlchemy ORM models — all 16 tables from ARCHITECTURE.md."""
from __future__ import annotations

import enum
from datetime import date, datetime

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
    LargeBinary,
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
    placed = "placed"
    completed = "completed"
    cancelled = "cancelled"
    rejected = "rejected"


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

    __table_args__ = (
        UniqueConstraint("stock_id", "date", "interval", name="uq_ohlcv"),
        Index("ix_ohlcv_stock_date", "stock_id", "date", "interval"),
    )


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RLTrainingRun(Base):
    __tablename__ = "rl_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    timestep: Mapped[int] = mapped_column(Integer, nullable=False)
    episode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── Golden Patterns ────────────────────────────────────────────────────

class GoldenPattern(Base):
    __tablename__ = "golden_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rl_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("rl_models.id"), nullable=False)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[IntervalEnum] = mapped_column(Enum(IntervalEnum), nullable=False)
    feature_window: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=BUY, -1=SELL, 0=HOLD
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
        UniqueConstraint("ensemble_config_id", "stock_id", "date", "interval", name="uq_ensemble_pred"),
        Index("ix_ensemble_pred_date_conf", "date", "interval", "confidence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
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
    calibrated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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


# ── Trade Orders ───────────────────────────────────────────────────────

class TradeOrder(Base):
    __tablename__ = "trade_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks_list.id"), nullable=False)
    ensemble_prediction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ensemble_predictions.id"), nullable=True
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
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── App Settings ───────────────────────────────────────────────────────

class AppSetting(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    property: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
