from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.api.deps import get_db
from app.config import settings
from app.db import crud
from app.db.models import BacktestResult as BacktestResultModel
from app.db.models import KNNPrediction, LSTMPrediction, EnsemblePrediction, StockOHLCV, GoldenPattern

logger = logging.getLogger(__name__)
router = APIRouter()


class BacktestRequest(BaseModel):
    model_type: str  # "rl", "knn", "lstm", "ensemble"
    model_id: int | None = None
    knn_name: str | None = None
    lstm_name: str | None = None
    stock_ids: list[int] | None = None
    stock_id: int | None = None
    interval: str = "day"
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 100_000.0
    stoploss_pct: float = 5.0
    target_pct: float | None = None
    min_confidence: float = 0.50
    max_positions: int = 10


class BacktestSummary(BaseModel):
    id: int
    model_type: str
    model_id: int
    stock_id: int | None = None
    symbol: str | None = None
    interval: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    total_return: float | None
    win_rate: float | None
    max_drawdown: float | None
    sharpe: float | None
    profit_factor: float | None
    trades_count: int | None
    created_at: str | None = None

    class Config:
        from_attributes = True


@router.post("/run")
async def run_backtest(
    req: BacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run backtest for a model on given date range."""
    valid_types = {"rl", "knn", "lstm", "ensemble"}
    if req.model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {valid_types}")

    stock_id = req.stock_id
    if not stock_id and req.stock_ids:
        stock_id = req.stock_ids[0]

    if not stock_id:
        raise HTTPException(status_code=400, detail="stock_id is required")

    start_d = req.start_date or date(2020, 1, 1)
    end_d = req.end_date or date.today()

    # 1. Fetch OHLCV for the exact backtest window
    ohlcv = await crud.get_ohlcv(db, stock_id, req.interval, start_d, end_d)
    if not ohlcv:
        raise HTTPException(status_code=400, detail=f"No OHLCV data found for stock {stock_id} in range {start_d} – {end_d}")

    dates = [x.date for x in ohlcv]
    close_prices = np.array([x.close for x in ohlcv])
    high_prices  = np.array([x.high  for x in ohlcv])
    low_prices   = np.array([x.low   for x in ohlcv])
    open_prices  = np.array([x.open  for x in ohlcv])
    # 20-day rolling average daily traded value (close × volume) for the ADV cap.
    # Computed from raw arrays inline to avoid an extra DB round-trip.
    _traded_val = np.array([float(x.close) * float(x.volume or 0) for x in ohlcv])
    _adv_window = 20
    adv_values = np.array([
        float(np.mean(_traded_val[max(0, j - _adv_window): j + 1]))
        for j in range(len(_traded_val))
    ])

    # 2. Try to fetch predictions from DB first
    pred_dict: dict = {}

    # For ensemble: query EnsemblePrediction directly — these carry the final
    # ensemble signal (post agreement-filter, sector-guard, weekly-confluence)
    # that the live prediction UI shows.  Querying raw KNNPrediction first was wrong
    # because KNN rows are pre-ensemble and may all be HOLD even when the ensemble says BUY.
    if req.model_type == "ensemble":
        q = select(EnsemblePrediction).where(
            EnsemblePrediction.stock_id == stock_id,
            EnsemblePrediction.interval == req.interval,
            EnsemblePrediction.date.in_(dates),
        )
        if req.model_id:
            q = q.where(EnsemblePrediction.ensemble_config_id == req.model_id)
        res = await db.execute(q)
        for p in res.scalars():
            # EnsemblePrediction.action: 0=HOLD, 1=BUY, 2=SELL
            # Backtester expects: 0=HOLD, 1=BUY, -1=SELL — map 2→-1
            mapped_action = -1 if p.action == 2 else p.action
            pred_dict[p.date] = {
                "action": mapped_action,
                "confidence": p.confidence,
                "regime_id": p.regime_id,
            }

    elif req.model_type == "knn":
        q = select(KNNPrediction).where(
            KNNPrediction.stock_id == stock_id,
            KNNPrediction.interval == req.interval,
            KNNPrediction.date.in_(dates),
        )
        if req.model_id:
            q = q.where(KNNPrediction.knn_model_id == req.model_id)
        res = await db.execute(q)
        for p in res.scalars():
            pred_dict[p.date] = {"action": p.action, "confidence": p.confidence, "regime_id": p.regime_id, "matched_pattern_indices": p.matched_pattern_ids}

    elif req.model_type == "lstm":
        q = select(LSTMPrediction).where(
            LSTMPrediction.stock_id == stock_id,
            LSTMPrediction.interval == req.interval,
            LSTMPrediction.date.in_(dates),
        )
        if req.model_id:
            q = q.where(LSTMPrediction.lstm_model_id == req.model_id)
        res = await db.execute(q)
        for p in res.scalars():
            pred_dict[p.date] = {"action": p.action, "confidence": p.confidence, "regime_id": p.regime_id}

    # Diagnostic: log signal breakdown from DB
    _buy_cnt = sum(1 for v in pred_dict.values() if v["action"] == 1)
    _buy_above_thresh = sum(1 for v in pred_dict.values() if v["action"] == 1 and v["confidence"] >= req.min_confidence)
    logger.info(
        "Backtest stock=%d type=%s dates=%d db_preds=%d BUY=%d BUY_above_min_conf(%.2f)=%d",
        stock_id, req.model_type, len(dates), len(pred_dict),
        _buy_cnt, req.min_confidence, _buy_above_thresh,
    )

    # 3. If no predictions in DB, run live inference from model artifacts
    coverage = len(pred_dict) / max(len(dates), 1)
    if coverage < 0.1 and req.model_type != "rl":
        logger.info(
            "Only %.0f%% predictions in DB for stock %d — running live inference from model artifacts",
            coverage * 100, stock_id,
        )
        pred_dict = await _run_live_inference(
            db, stock_id, req.interval, dates, start_d, end_d,
            knn_name=req.knn_name, lstm_name=req.lstm_name
        )

    predictions = [pred_dict.get(d, {"action": 0, "confidence": 0.0, "regime_id": None}) for d in dates]

    # 4. Execute backtest simulation
    from app.ml.backtester import BacktestConfig, run_backtest as ml_run_backtest
    config = BacktestConfig(
        initial_capital=req.initial_capital,
        stoploss_pct=req.stoploss_pct,
        target_pct=req.target_pct,
        min_confidence=req.min_confidence,
        max_positions=req.max_positions,
    )
    bt_result = ml_run_backtest(
        predictions, close_prices, dates, config,
        low_prices=low_prices,
        open_prices=open_prices,
        high_prices=high_prices,
        adv_values=adv_values,
    )

    # 5. Persist backtest record — resolve the true model_id if caller did not send one
    resolved_model_id = req.model_id
    if not resolved_model_id:
        # Auto-resolve: use the latest model of the given type so the record
        # always carries a real FK-compatible id which the delete cascade can match.
        try:
            if req.model_type == "knn":
                from app.db.models import KNNModel as _KNN
                _r = await db.execute(select(_KNN).order_by(_KNN.created_at.desc()).limit(1))
                _m = _r.scalar_one_or_none()
                resolved_model_id = _m.id if _m else 0
            elif req.model_type == "lstm":
                from app.db.models import LSTMModel as _LSTM
                _r = await db.execute(select(_LSTM).order_by(_LSTM.created_at.desc()).limit(1))
                _m = _r.scalar_one_or_none()
                resolved_model_id = _m.id if _m else 0
            elif req.model_type == "ensemble":
                from app.db.models import EnsembleConfig as _EC
                _r = await db.execute(select(_EC).order_by(_EC.created_at.desc()).limit(1))
                _m = _r.scalar_one_or_none()
                resolved_model_id = _m.id if _m else 0
            elif req.model_type == "rl":
                from app.db.models import RLModel as _RL
                _r = await db.execute(select(_RL).order_by(_RL.created_at.desc()).limit(1))
                _m = _r.scalar_one_or_none()
                resolved_model_id = _m.id if _m else 0
        except Exception:
            resolved_model_id = 0

    record = BacktestResultModel(
        model_type=req.model_type,
        model_id=resolved_model_id or 0,
        stock_id=stock_id,
        interval=req.interval,
        start_date=start_d,
        end_date=end_d,
        total_return=bt_result.total_return_pct / 100.0,
        win_rate=bt_result.win_rate / 100.0,
        max_drawdown=bt_result.max_drawdown_pct / 100.0,
        sharpe=bt_result.sharpe_ratio,
        profit_factor=bt_result.profit_factor if bt_result.profit_factor != float("inf") else 999.0,
        trades_count=bt_result.total_trades,
        trade_log=bt_result.trade_log,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    # 6. Build equity curve for the frontend chart
    equity_data = []
    if bt_result.equity_curve and close_prices[0] > 0:
        base_price = float(close_prices[0])
        for idx, (dt, eq) in enumerate(zip(dates, bt_result.equity_curve)):
            benchmark = req.initial_capital * (float(close_prices[idx]) / base_price)
            equity_data.append({"date": str(dt), "equity": round(eq, 2), "benchmark": round(benchmark, 2)})

    return {
        "id": record.id,
        "model_type": record.model_type,
        "model_id": record.model_id,
        "stock_id": record.stock_id,
        "interval": record.interval,
        "start_date": str(record.start_date),
        "end_date": str(record.end_date),
        "total_return": record.total_return,
        "win_rate": record.win_rate,
        "max_drawdown": record.max_drawdown,
        "sharpe": record.sharpe,
        "sharpe_ratio": bt_result.sharpe_ratio,
        "profit_factor": record.profit_factor,
        "trades_count": record.trades_count,
        "trade_log": record.trade_log,
        "equity_curve": equity_data,
        "total_trades": bt_result.total_trades,
        "winning_trades": bt_result.winning_trades,
        "losing_trades": bt_result.losing_trades,
        "buy_hold_return": bt_result.buy_hold_return_pct / 100.0,
    }


async def _run_live_inference(
    db,
    stock_id: int,
    interval: str,
    target_dates: list,
    start_d: date,
    end_d: date,
    knn_name: str | None = None,
    lstm_name: str | None = None,
) -> dict:
    """
    Load specific or latest KNN+LSTM distilled models from disk and run sliding-window
    inference over the requested date range. Returns a {date: prediction_dict} map.

    Model output class encoding: 0=HOLD, 1=BUY, 2=SELL
    Backtester encoding:         0=HOLD, 1=BUY, -1=SELL   (2 → -1 mapping applied here)
    """
    distill_dir = settings.MODEL_DIR / "distill"
    if not distill_dir.exists():
        logger.warning("No distill directory found at %s", distill_dir)
        return {}

    def _latest_model_path(distill_dir: Path, prefix: str, filename: str) -> Path | None:
        """Find the newest model file in the highest-versioned model directory under distill_dir."""
        candidates = sorted(
            [d for d in distill_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda p: int("".join(filter(str.isdigit, p.name)) or "0"),
        )
        # Derive glob pattern from the filename extension (e.g. *.joblib, *.pt)
        suffix = Path(filename).suffix  # ".joblib" or ".pt"
        for candidate in reversed(candidates):
            # First try the exact fixed name (backward compat)
            exact = candidate / filename
            if exact.exists():
                return exact
            # Then find the newest timestamped file with the same extension
            versioned = sorted(candidate.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime)
            if versioned:
                return versioned[-1]
        return None

    # Resolve actual model paths
    if knn_name:
        knn_base_dir = distill_dir / knn_name
        knn_path = _latest_model_path(knn_base_dir.parent, prefix=knn_name, filename="knn_model.joblib")
        if not knn_path:
            logger.error("Requested KNN model not found in: %s", knn_base_dir)
            knn_path = None
    else:
        knn_path = _latest_model_path(distill_dir, prefix="knn", filename="knn_model.joblib")

    if lstm_name:
        lstm_base_dir = distill_dir / lstm_name
        lstm_path = _latest_model_path(lstm_base_dir.parent, prefix=lstm_name, filename="lstm_model.pt")
        if not lstm_path:
            logger.error("Requested LSTM model not found in: %s", lstm_base_dir)
            lstm_path = None
    else:
        lstm_path = _latest_model_path(distill_dir, prefix="lstm", filename="lstm_model.pt")

    if not knn_path or not lstm_path:
        logger.warning("Missing required model artifacts for live inference")
        return {}

    # Load models
    from app.ml.knn_distiller import load_knn_model, predict_knn
    from app.ml.lstm_distiller import load_lstm_model, predict_lstm

    try:
        knn_model = load_knn_model(str(knn_path))
        lstm_model = load_lstm_model(str(lstm_path))
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        return {}

    # Fetch features with warmup so every backtest date has a full seq_len window
    seq_len = settings.DEFAULT_SEQ_LEN_DAILY if interval == "day" else settings.DEFAULT_SEQ_LEN_WEEKLY
    # Indicators like SMA_200 need ~200 trading days of warmup; fetch extra data
    warmup_start = start_d - timedelta(days=400)

    from app.core.data_service import get_model_ready_data, get_stock_features
    from app.core.normalizer import prepare_model_input

    df, feature_cols = await get_model_ready_data(
        db, stock_id, interval, seq_len=seq_len, start_date=warmup_start, end_date=end_d,
    )
    if df.empty or len(df) < seq_len + 1:
        logger.warning("Insufficient feature data for live inference on stock %d", stock_id)
        return {}

    # Build a date→ATR% lookup from raw (unnormalized) close prices for the ATR gate.
    # ATR% is approximated as the 14-bar rolling mean of |ΔClose|/Close × 100.
    _atr_by_date: dict = {}
    try:
        df_raw = await get_stock_features(
            db, stock_id, interval, start_date=warmup_start, end_date=end_d, normalize=False
        )
        if not df_raw.empty and "close" in df_raw.columns:
            _raw_close = df_raw["close"].astype(float)
            _abs_ret = _raw_close.pct_change().abs() * 100.0  # |ΔC/C| in %
            _atr_pct_series = _abs_ret.rolling(14, min_periods=1).mean()
            _raw_dates = (
                df_raw["date"].dt.date.tolist()
                if "date" in df_raw.columns and hasattr(df_raw["date"].iloc[0], "date")
                else list(df_raw["date"])
            )
            _atr_by_date = {d: v for d, v in zip(_raw_dates, _atr_pct_series)}
    except Exception as _atr_exc:
        logger.warning("ATR gate data fetch failed (gate disabled): %s", _atr_exc)

    # date column name
    date_col = "date" if "date" in df.columns else df.index.name or None
    if date_col and date_col in df.columns:
        all_dates = list(df[date_col])
    else:
        # try index
        all_dates = list(df.index)

    X_all = prepare_model_input(df, feature_cols, seq_len=seq_len)  # (n_windows, seq_len, features)
    if len(X_all) == 0:
        return {}

    # Load KNN norm_params for inference-time Z-score normalization
    # (must match the StandardScaler fitted during training)
    from app.ml.knn_distiller import load_knn_norm_params
    knn_norm_params = load_knn_norm_params(knn_path.parent)

    # Run individual models
    knn_preds_raw, knn_probs_raw = predict_knn(knn_model, X_all, norm_params=knn_norm_params)
    lstm_preds_raw, lstm_probs_raw = predict_lstm(lstm_model, X_all)

    # Resolve matched patterns from KNN for explainability
    matched_patterns = []
    if hasattr(knn_model, "_query"):
        _, indices = knn_model._query(np.ascontiguousarray(X_all.reshape(len(X_all), -1), dtype=np.float32))
        # indices is (n_samples, k)
        # We need to map these back to GoldenPattern IDs if possible.
        # This requires the KNN model to have a mapping of its training indices to DB IDs.
        # If not available, we'll indicate indices.
        matched_patterns = indices.tolist()

    # KNN may have been trained on only [1, 2] classes → expand to 3-column prob [HOLD, BUY, SELL]
    knn_classes = list(knn_model.classes_)  # e.g. [1, 2]
    if knn_probs_raw.shape[1] != 3:
        expanded = np.zeros((len(knn_probs_raw), 3), dtype=np.float32)
        for col_idx, cls in enumerate(knn_classes):
            if 0 <= cls <= 2:
                expanded[:, cls] = knn_probs_raw[:, col_idx]
        knn_probs_raw = expanded

    # Run ensemble combination
    from app.ml.ensemble import ensemble_predict
    preds_list = ensemble_predict(
        knn_preds_raw, knn_probs_raw,
        lstm_preds_raw, lstm_probs_raw,
        knn_weight=0.5, lstm_weight=0.5,
        agreement_required=True,
    )

    # Each window[i] covers data rows [i, i+seq_len-1] → prediction is for all_dates[i+seq_len-1]
    # Build rolling-median ATR for the gate: suppress signals on panic/crash days
    # where current ATR% > 3× the 60-bar rolling median (market is unusually wild).
    _sorted_atr_dates = sorted(_atr_by_date.keys())
    _atr_values = [_atr_by_date[d] for d in _sorted_atr_dates]
    _atr_median_by_date: dict = {}
    _window = 60
    for _j, _d in enumerate(_sorted_atr_dates):
        _slice = _atr_values[max(0, _j - _window): _j + 1]
        _valid = [v for v in _slice if v == v and v > 0]  # exclude NaN and zero
        if _valid:
            _sorted_slice = sorted(_valid)
            _mid = len(_sorted_slice) // 2
            _median = (
                _sorted_slice[_mid]
                if len(_sorted_slice) % 2 == 1
                else (_sorted_slice[_mid - 1] + _sorted_slice[_mid]) / 2.0
            )
            _atr_median_by_date[_d] = _median

    result: dict = {}
    target_date_set = set(target_dates)
    for i, p in enumerate(preds_list):
        date_idx = i + seq_len - 1
        if date_idx >= len(all_dates):
            break
        d = all_dates[date_idx]
        # Normalise to a plain date object
        if hasattr(d, "date"):
            d = d.date()
        elif isinstance(d, str):
            from datetime import datetime
            try:
                d = datetime.strptime(d, "%Y-%m-%d").date()
            except:
                pass
        if d not in target_date_set:
            continue
        raw_action = p["action"]  # 0=HOLD, 1=BUY, 2=SELL
        mapped_action = -1 if raw_action == 2 else raw_action  # convert SELL → -1 for backtester

        # ATR gate: suppress BUY/SELL during panic/crash days (unusually high volatility).
        # Skip signal when current ATR% > 3× the 60-bar rolling median ATR%.
        if mapped_action != 0 and _atr_median_by_date:
            _curr_atr = _atr_by_date.get(d, 0.0)
            _med_atr = _atr_median_by_date.get(d, 0.0)
            if _med_atr > 0 and _curr_atr > 3.0 * _med_atr:
                logger.debug("ATR gate suppressed %s signal on %s (atr=%.2f%%, median=%.2f%%)",
                             "BUY" if mapped_action == 1 else "SELL", d, _curr_atr, _med_atr)
                mapped_action = 0

        result[d] = {
            "action": mapped_action,
            "confidence": p["confidence"],
            "regime_id": None,
            "matched_pattern_indices": p.get("matched_pattern_indices")
        }

    logger.info("Live inference produced %d predictions for %d target dates", len(result), len(target_dates))
    return result


@router.get("/{backtest_id}/trades/{trade_idx}/patterns")
async def get_trade_patterns(
    backtest_id: int,
    trade_idx: int,
    db: AsyncSession = Depends(get_db),
):
    """Fetch golden patterns that matched a specific trade in a backtest."""
    res = await db.execute(select(BacktestResultModel).where(BacktestResultModel.id == backtest_id))
    bt = res.scalar_one_or_none()
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    if not bt.trade_log or trade_idx >= len(bt.trade_log):
        raise HTTPException(status_code=404, detail="Trade not found")

    trade = bt.trade_log[trade_idx]
    pattern_indices = trade.get("matched_pattern_indices")
    if not pattern_indices:
        return {"patterns": []}

    # If it's a dict (from DB JSON), it might need conversion
    if isinstance(pattern_indices, dict):
        # depending on how it was stored
        pattern_indices = list(pattern_indices.values())

    # Fetch Golden Patterns from DB
    # We don't have a direct mapping from KNN index to GoldenPattern.id yet in the code,
    # so we'll fetch patterns for the same stock/interval as a proxy for search.
    # IN A REAL SCENARIO: The KNN model should store the GoldenPattern.id in its training data.
    q = select(GoldenPattern).where(
        GoldenPattern.stock_id == bt.stock_id,
        GoldenPattern.interval == bt.interval
    ).limit(100) # Safety limit
    
    res = await db.execute(q)
    all_patterns = res.scalars().all()
    
    # Filter by indices if we assume the training set was just these patterns
    # This is a heuristic for this implementation.
    matched = []
    for idx in pattern_indices:
        if 0 <= idx < len(all_patterns):
            p = all_patterns[idx]
            matched.append({
                "id": p.id,
                "date": str(p.date),
                "pnl_pct": p.pnl_percent,
                "confidence": p.confidence,
                "label": "BUY" if p.label == 1 else "SELL" if p.label == -1 else "HOLD"
            })

    return {"patterns": matched}


@router.get("/results/{backtest_id}")
async def get_backtest_results(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get backtest results."""
    result = await db.execute(
        select(BacktestResultModel).where(BacktestResultModel.id == backtest_id)
    )
    bt = result.scalar_one_or_none()
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return {
        "id": bt.id,
        "model_type": bt.model_type,
        "model_id": bt.model_id,
        "stock_id": bt.stock_id,
        "interval": bt.interval.value if hasattr(bt.interval, "value") else str(bt.interval),
        "start_date": str(bt.start_date),
        "end_date": str(bt.end_date),
        "total_return": bt.total_return,
        "win_rate": bt.win_rate,
        "max_drawdown": bt.max_drawdown,
        "sharpe": bt.sharpe,
        "sharpe_ratio": bt.sharpe,
        "profit_factor": bt.profit_factor,
        "trades_count": bt.trades_count,
        "trade_log": bt.trade_log,
    }


@router.get("/results", response_model=list[BacktestSummary])
async def list_backtest_results(
    db: AsyncSession = Depends(get_db),
    model_type: str | None = Query(None),
):
    """List all backtest results."""
    from app.db.models import Stock
    q = (
        select(BacktestResultModel, Stock.symbol)
        .outerjoin(Stock, Stock.id == BacktestResultModel.stock_id)
        .order_by(BacktestResultModel.id.desc())
    )
    if model_type:
        q = q.where(BacktestResultModel.model_type == model_type)
    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "id": bt.id,
            "model_type": bt.model_type,
            "model_id": bt.model_id,
            "stock_id": bt.stock_id,
            "symbol": symbol,
            "interval": bt.interval.value if hasattr(bt.interval, "value") else str(bt.interval),
            "start_date": str(bt.start_date),
            "end_date": str(bt.end_date),
            "total_return": bt.total_return,
            "win_rate": bt.win_rate,
            "max_drawdown": bt.max_drawdown,
            "sharpe": bt.sharpe,
            "profit_factor": bt.profit_factor,
            "trades_count": bt.trades_count,
            "created_at": str(bt.created_at) if getattr(bt, "created_at", None) else None,
        }
        for bt, symbol in rows
    ]


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a single backtest result by ID."""
    res = await db.execute(select(BacktestResultModel).where(BacktestResultModel.id == backtest_id))
    bt = res.scalar_one_or_none()
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    await db.delete(bt)
    await db.commit()
    return {"deleted": backtest_id}


class BatchDeleteRequest(BaseModel):
    ids: list[int]


@router.delete("/batch/delete")
async def delete_backtest_batch(
    req: BatchDeleteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple backtest results by a list of IDs."""
    from sqlalchemy import delete as sqla_delete
    if not req.ids:
        return {"deleted": 0}
    await db.execute(sqla_delete(BacktestResultModel).where(BacktestResultModel.id.in_(req.ids)))
    await db.commit()
    return {"deleted": len(req.ids)}
