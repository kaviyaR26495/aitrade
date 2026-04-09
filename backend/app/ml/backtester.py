"""Unified Backtesting Engine.

Works for RL, KNN, LSTM, and Ensemble models.
Simulates trading with:
- Per-date prediction + order execution
- Configurable SL/target
- Commission costs
- Regime-aware analysis

Outputs: equity curve, trade log, performance metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001  # 0.1% per trade (Zerodha CNC is ~0.03% but includes STT etc)
    stoploss_pct: float = 5.0
    target_pct: float | None = None  # None = hold until sell signal
    min_confidence: float = 0.72
    max_positions: int = 10
    position_size_pct: float = 10.0  # % of capital per trade (used by 'fixed' sizing)
    # ── Slippage simulation ────────────────────────────────────────────────
    slippage_bps: float = 5.0        # baseline bid-ask proxy (5 bps = 0.05 %)
    vol_slippage_scale: float = 0.5  # extra slippage per unit of realised daily vol
    # ── Dynamic position sizing ────────────────────────────────────────────
    position_sizing: str = "volatility_target"  # "fixed" | "kelly" | "volatility_target"
    max_single_trade_risk_pct: float = 2.0       # max % of equity risked at stoploss
    vol_target_annual: float = 0.15              # target annualised vol for vol-target method
    # ── NIFTY 200-DMA breadth kill-switch ────────────────────────────────────
    # Requires ``nifty_closes`` to be passed as a kwarg to run_backtest().
    # nifty_block_buys=False → halve size when NIFTY is below 200-DMA
    # nifty_block_buys=True  → disable all new BUYs when NIFTY below 200-DMA
    nifty_block_buys: bool = False
    # ── ADV liquidity cap ─────────────────────────────────────────────────────
    # Caps each trade at adv_cap_fraction × 20-day average daily traded value
    # (close × volume).  Prevents over-sizing into illiquid stocks.
    adv_cap_fraction: float = 0.01   # 1% of 20-day ADV value
    # ── Sector concentration guard ─────────────────────────────────────────────
    # Requires ``sectors`` to be passed as a list to run_backtest().
    # A new BUY is blocked when the sector already accounts for ≥ sector_cap
    # of total portfolio equity.
    sector_cap: float = 0.25  # maximum equity fraction in any single sector
    # ── Regime lock ───────────────────────────────────────────────────────────
    # When True: only execute a BUY/SELL signal if the current bar's regime_id
    # matches the regime_id recorded on the pattern that generated the signal.
    # Requires predictions to carry "pattern_regime_id" (set by live inference).
    regime_lock: bool = False

@dataclass
class Trade:
    entry_date: Any
    exit_date: Any | None = None
    action: str = "BUY"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    regime_id: int | None = None
    confidence: float = 0.0
    exit_reason: str = ""
    slippage_cost: float = 0.0   # total INR lost to slippage on this round-trip
    matched_pattern_indices: list[int] | None = None


@dataclass
class BacktestResult:
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    buy_hold_return_pct: float = 0.0
    total_slippage_cost: float = 0.0  # total INR drained by slippage across all trades
    gap_down_exits: int = 0           # trades where the stoploss filled below the SL price due to overnight gap


def _rolling_vol(close_prices: np.ndarray, idx: int, window: int = 20) -> float:
    """Compute realised daily volatility (std of returns) over the last ``window`` bars."""
    start = max(1, idx - window + 1)
    if idx - start < 1:
        return 0.0
    px = close_prices[start : idx + 1]
    rets = (px[1:] - px[:-1]) / np.where(px[:-1] != 0, px[:-1], 1e-8)
    return float(np.std(rets)) if len(rets) >= 2 else 0.0


def _slip_price(
    price: float,
    side: int,
    slippage_bps: float,
    realized_vol: float,
    vol_slippage_scale: float,
) -> float:
    """Return slipped fill price.  side=+1 BUY (pay more), side=-1 SELL (receive less)."""
    if slippage_bps <= 0 and vol_slippage_scale <= 0:
        return price
    slip = slippage_bps / 10_000.0 + vol_slippage_scale * realized_vol
    return float(price * (1.0 + side * slip))


def run_backtest(
    predictions: list[dict],
    close_prices: np.ndarray,
    dates: list,
    config: BacktestConfig | None = None,
    low_prices: np.ndarray | None = None,
    open_prices: np.ndarray | None = None,
    high_prices: np.ndarray | None = None,
    nifty_closes: np.ndarray | None = None,
    sectors: list[str | None] | None = None,
    adv_values: np.ndarray | None = None,
) -> BacktestResult:
    """
    Run backtest on a sequence of predictions.

    predictions : list of dicts with keys: action (0=HOLD, 1=BUY, 2=SELL), confidence, regime_id
    close_prices : daily close prices aligned with predictions
    dates        : dates aligned with predictions
    low_prices   : daily low prices (same length as close_prices).  When provided,
                   the stoploss check uses the *low* to detect intraday breaches —
                   a stock can drop through your stoploss price intraday and recover
                   to a flat close, which a close-only backtester would miss entirely.
    open_prices  : daily open prices.  When an overnight news event causes the
                   stock to gap below the stoploss price, the exit is simulated at
                   the open (not the stoploss price), matching real market behaviour.
    high_prices  : daily high prices.  When provided, the target check uses the
                   intraday high — matching a resting limit order that fills as
                   soon as price touches the target.  Exit fills at the exact
                   target price (entry × (1 + target_pct/100)), not the high.
    adv_values   : 20-day rolling average daily traded value (close × volume) in
                   INR, one entry per bar.  When provided, each BUY trade is
                   capped at BacktestConfig.adv_cap_fraction × adv_values[i]
                   to simulate realistic market-impact limits.

    nifty_closes : np.ndarray | None
        Historical NIFTY 50 daily closes aligned with ``close_prices`` (same
        index).  When provided, the 200-DMA kill-switch is applied: if today's
        NIFTY close is below its 200-DMA, new BUY position sizes are halved
        (or fully blocked, depending on BacktestConfig.nifty_block_buys).
    sectors : list[str | None] | None
        Per-bar sector label aligned with ``predictions``.  Each element is the
        sector string for the stock at that bar (constant for single-stock
        backtests, may vary for multi-stock runs).  When provided, a new BUY
        is blocked if the sector already occupies ≥ BacktestConfig.sector_cap
        of total portfolio equity.

    Returns BacktestResult with metrics and trade log.
    """
    if config is None:
        config = BacktestConfig()

    from app.ml.position_sizer import size_trade, size_trade_with_breadth, sector_concentration_multiplier  # noqa: PLC0415

    n = min(len(predictions), len(close_prices), len(dates))
    capital = config.initial_capital
    cash = capital
    positions: dict[str, dict] = {}  # stock_key → {entry_price, raw_entry_price, quantity, ...}
    equity_curve = []
    trades: list[Trade] = []
    gap_down_exits = 0

    # Normalise optional price arrays (fallback to close when absent)
    _lows  = low_prices  if low_prices  is not None else close_prices
    _opens = open_prices if open_prices is not None else close_prices
    _highs = high_prices if high_prices is not None else close_prices

    for i in range(n):
        price = float(close_prices[i])
        low   = float(_lows[i])
        open_ = float(_opens[i])
        high  = float(_highs[i])
        pred = predictions[i]
        action = pred.get("action", 0)
        confidence = pred.get("confidence", 0)
        regime_id = pred.get("regime_id")
        dt = dates[i]

        # Realised vol for slippage + position sizing
        rv = _rolling_vol(close_prices, i)
        # Sector label for this bar
        sector_label = sectors[i] if sectors is not None and i < len(sectors) else None

        # Update MTM price on all open positions (needed for sector cap calc)
        for pos in positions.values():
            pos["price"] = price

        # ── Stoploss / target checks on open positions ────────────────
        closed_keys = []
        for key, pos in positions.items():
            sl_price = pos["raw_entry_price"] * (1 - config.stoploss_pct / 100)

            # ── Overnight gap detection ───────────────────────────────
            # If today's OPEN is already below the stoploss price the stock
            # has gapped down overnight and will fill AT the open, not at sl_price.
            # If the intraday LOW (without gap) dips below sl_price, the exit
            # fills at sl_price (limit-stoploss simulation).
            gap_down = open_ < sl_price
            sl_breached = gap_down or (low < sl_price)

            if sl_breached:
                # Worst-case fill: open if gap-down, else sl_price
                raw_exit = open_ if gap_down else sl_price
                fill_exit = _slip_price(raw_exit, -1, config.slippage_bps, rv, config.vol_slippage_scale)
                exit_value = pos["quantity"] * fill_exit * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                slip_cost = (
                    (pos["entry_price"] - pos["raw_entry_price"]) * pos["quantity"]
                    + (raw_exit - fill_exit) * pos["quantity"]
                )
                cash += exit_value
                if gap_down:
                    gap_down_exits += 1
                trades.append(Trade(
                    entry_date=pos["entry_date"], exit_date=dt, action="BUY",
                    entry_price=pos["entry_price"], exit_price=fill_exit,
                    quantity=pos["quantity"], pnl=pnl, pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"), confidence=pos.get("confidence", 0),
                    exit_reason="gap_down_stoploss" if gap_down else "stoploss",
                    slippage_cost=round(slip_cost, 4),
                    matched_pattern_indices=pos.get("matched_pattern_indices"),
                ))
                closed_keys.append(key)
                continue  # don't also check target on the same bar

            elif config.target_pct and ((high - pos["raw_entry_price"]) / pos["raw_entry_price"] * 100) > config.target_pct:
                # Limit order sitting at target price fires when today's HIGH
                # touches or exceeds it.  Fill at the exact target — not the
                # intraday high — mirroring live order execution.
                fill_raw = pos["raw_entry_price"] * (1 + config.target_pct / 100)
                fill_exit = _slip_price(fill_raw, -1, config.slippage_bps, rv, config.vol_slippage_scale)
                exit_value = pos["quantity"] * fill_exit * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                slip_cost = (
                    (pos["entry_price"] - pos["raw_entry_price"]) * pos["quantity"]
                    + (fill_raw - fill_exit) * pos["quantity"]
                )
                cash += exit_value
                trades.append(Trade(
                    entry_date=pos["entry_date"], exit_date=dt, action="BUY",
                    entry_price=pos["entry_price"], exit_price=fill_exit,
                    quantity=pos["quantity"], pnl=pnl, pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"), confidence=pos.get("confidence", 0),
                    exit_reason="target", slippage_cost=round(slip_cost, 4),
                    matched_pattern_indices=pos.get("matched_pattern_indices"),
                ))
                closed_keys.append(key)

        for key in closed_keys:
            del positions[key]

        # ── New signals ────────────────────────────────────────────────
        # Regime lock: skip signal if current regime doesn't match the pattern's regime
        if config.regime_lock and action in (1, -1):
            pattern_regime = pred.get("pattern_regime_id")
            if pattern_regime is not None and regime_id is not None and pattern_regime != regime_id:
                action = 0  # suppress — wrong market regime

        if action == 1 and confidence >= config.min_confidence:
            if len(positions) < config.max_positions and price > 0:
                # Dynamic position sizing + optional NIFTY breadth kill-switch
                if nifty_closes is not None:
                    nifty_window = nifty_closes[: i + 1]
                    pos_frac = size_trade_with_breadth(
                        method=config.position_sizing,
                        realized_vol_daily=rv,
                        nifty_close_series=nifty_window,
                        trades=trades,
                        max_risk_pct=config.max_single_trade_risk_pct / 100,
                        stoploss_pct=config.stoploss_pct / 100,
                        target_vol_annual=config.vol_target_annual,
                        fallback_pct=config.position_size_pct / 100,
                        max_positions=config.max_positions,
                        half_size_below_dma=not config.nifty_block_buys,
                    )
                else:
                    pos_frac = size_trade(
                        method=config.position_sizing,
                        realized_vol_daily=rv,
                        trades=trades,
                        max_risk_pct=config.max_single_trade_risk_pct / 100,
                        stoploss_pct=config.stoploss_pct / 100,
                        target_vol_annual=config.vol_target_annual,
                        fallback_pct=config.position_size_pct / 100,
                        max_positions=config.max_positions,
                    )
                fill_entry = _slip_price(price, +1, config.slippage_bps, rv, config.vol_slippage_scale)
                # Size on total equity (cash + mark-to-market open positions),
                # then cap at available cash so we never spend margin
                current_open_value = sum(p["quantity"] * price for p in positions.values())
                total_equity = cash + current_open_value
                ideal_position_value = total_equity * pos_frac
                actual_position_value = min(ideal_position_value, cash)
                # ADV liquidity cap: never commit more than adv_cap_fraction of
                # 20-day average daily traded value (price × volume).  This
                # prevents the order from becoming the market in illiquid stocks.
                if adv_values is not None and i < len(adv_values) and adv_values[i] > 0:
                    adv_cap = float(adv_values[i]) * config.adv_cap_fraction
                    actual_position_value = min(actual_position_value, adv_cap)
                qty = int(actual_position_value / fill_entry)
                # Sector concentration guard: block the trade if the sector
                # already occupies ≥ sector_cap of total portfolio equity
                if sector_label and sectors is not None:
                    sect_mult = sector_concentration_multiplier(
                        sector_label,
                        open_positions=positions,
                        total_equity=total_equity,
                        sector_cap=config.sector_cap,
                    )
                    if sect_mult == 0.0:
                        continue  # sector cap breached — skip trade
                if qty > 0:
                    actual_cost = qty * fill_entry * (1 + config.commission_pct)
                    if actual_cost <= cash:
                        cash -= actual_cost
                        positions[f"pos_{i}"] = {
                            "entry_price": fill_entry,
                            "raw_entry_price": price,
                            "quantity": qty,
                            "entry_date": dt,
                            "regime_id": regime_id,
                            "confidence": confidence,
                            "matched_pattern_indices": pred.get("matched_pattern_indices"),
                            "sector": sector_label,
                            "price": price,  # current MTM price for sector calc
                        }

        elif action == -1 and confidence >= config.min_confidence:  # noqa: E501
            for key, pos in list(positions.items()):
                fill_exit = _slip_price(price, -1, config.slippage_bps, rv, config.vol_slippage_scale)
                exit_value = pos["quantity"] * fill_exit * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                slip_cost = (
                    (pos["entry_price"] - pos["raw_entry_price"]) * pos["quantity"]
                    + (price - fill_exit) * pos["quantity"]
                )
                cash += exit_value
                trades.append(Trade(
                    entry_date=pos["entry_date"], exit_date=dt, action="BUY",
                    entry_price=pos["entry_price"], exit_price=fill_exit,
                    quantity=pos["quantity"], pnl=pnl, pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"), confidence=pos.get("confidence", 0),
                    exit_reason="sell_signal", slippage_cost=round(slip_cost, 4),
                    matched_pattern_indices=pos.get("matched_pattern_indices"),
                ))
            positions.clear()

        # Mark-to-market equity uses raw price (no slippage on unrealised P&L)
        open_value = sum(pos["quantity"] * price for pos in positions.values())
        equity_curve.append(cash + open_value)

    # ── Close remaining positions at last price ────────────────────────
    if positions and n > 0:
        last_price = float(close_prices[n - 1])
        last_rv = _rolling_vol(close_prices, n - 1)
        for pos in positions.values():
            fill_exit = _slip_price(last_price, -1, config.slippage_bps, last_rv, config.vol_slippage_scale)
            exit_value = pos["quantity"] * fill_exit * (1 - config.commission_pct)
            pnl = exit_value - (pos["quantity"] * pos["entry_price"])
            pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
            slip_cost = (
                (pos["entry_price"] - pos["raw_entry_price"]) * pos["quantity"]
                + (last_price - fill_exit) * pos["quantity"]
            )
            cash += exit_value
            trades.append(Trade(
                entry_date=pos["entry_date"],
                exit_date=dates[-1] if dates else None,
                entry_price=pos["entry_price"],
                exit_price=fill_exit,
                quantity=pos["quantity"],
                pnl=pnl, pnl_pct=pnl_pct,
                regime_id=pos.get("regime_id"),
                exit_reason="end_of_backtest",
                slippage_cost=round(slip_cost, 4),
                matched_pattern_indices=pos.get("matched_pattern_indices"),
            ))

    # Compute metrics
    result = _compute_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=capital,
        close_prices=close_prices[:n],
        n_days=n,
    )
    result.gap_down_exits = gap_down_exits
    return result


def _compute_metrics(
    trades: list[Trade],
    equity_curve: list[float],
    initial_capital: float,
    close_prices: np.ndarray,
    n_days: int,
) -> BacktestResult:
    """Compute backtest performance metrics."""
    result = BacktestResult()
    result.equity_curve = equity_curve
    result.total_trades = len(trades)

    if not equity_curve:
        return result

    final_equity = equity_curve[-1]
    result.total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

    # Annualized return (assume 252 trading days)
    if n_days > 0:
        years = n_days / 252
        if years > 0:
            result.annual_return_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    # Sharpe ratio (daily returns)
    equity_arr = np.array(equity_curve)
    daily_returns = np.diff(equity_arr) / equity_arr[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        result.sharpe_ratio = round(
            float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)), 4
        )

    # Max drawdown
    peak = equity_arr[0]
    max_dd = 0
    for val in equity_arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown_pct = round(max_dd, 4)

    # Trade statistics
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = round(len(wins) / len(trades) * 100, 2)

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = round(total_wins / total_losses, 4) if total_losses > 0 else float("inf")

        if wins:
            result.avg_win_pct = round(float(np.mean([t.pnl_pct for t in wins])), 4)
        if losses:
            result.avg_loss_pct = round(float(np.mean([t.pnl_pct for t in losses])), 4)

        result.total_slippage_cost = round(sum(t.slippage_cost for t in trades), 2)

    # Buy and hold return
    if len(close_prices) >= 2:
        result.buy_hold_return_pct = round(
            ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100, 4
        )

    # Trade log
    def _fmt_date(d: Any) -> str:
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        return str(d)

    result.trade_log = [
        {
            "entry_date": _fmt_date(t.entry_date),
            "exit_date": _fmt_date(t.exit_date),
            "entry_price": round(float(t.entry_price), 2),
            "exit_price": round(float(t.exit_price), 2),
            "quantity": int(t.quantity),
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "regime_id": t.regime_id,
            "confidence": t.confidence,
            "exit_reason": t.exit_reason,
            "slippage_cost": round(t.slippage_cost, 4),
            "matched_pattern_indices": t.matched_pattern_indices,
        }
        for t in trades
    ]

    return result


# ─── Walk-Forward Optimisation ────────────────────────────────────────────────


def walk_forward_backtest(
    full_df: pd.DataFrame,
    train_fn,
    predict_fn,
    train_years: int = 3,
    test_months: int = 12,
    min_train_rows: int = 200,
    config: BacktestConfig | None = None,
) -> dict:
    """Strict Walk-Forward Optimisation (WFO) backtester.

    Eliminates data leakage by ensuring the model *never sees the future*:
    each fold trains on a fixed historical window and is evaluated on the
    immediately following out-of-sample period.  The test windows are
    non-overlapping — the concatenated equity curve is a pure out-of-sample
    performance record.

    Fold structure
    --------------
    Fold 0:  train [t0 .. t0+train_years],    test [t0+train_years .. +test_months]
    Fold 1:  train [t0+T .. t0+T+train_years], test [t0+T+train_years .. +test_months]
    ...
    where T = test_months (rolling by test window, not by full train period).

    Parameters
    ----------
    full_df : pd.DataFrame
        Full historical OHLCV data with a DatetimeIndex or a ``date`` column.
        Must contain a ``close`` column (or the 4th column is used as close).
    train_fn : Callable[[pd.DataFrame], Any]
        ``train_fn(train_df) -> model``
        User-supplied training routine.  Receives the training slice as a
        DataFrame with the same columns as ``full_df``.  Returns any object
        that ``predict_fn`` can accept.
    predict_fn : Callable[[Any, pd.DataFrame], list[dict]]
        ``predict_fn(model, test_df) -> list[dict]``
        User-supplied prediction routine.  Must return predictions in the
        standard format:  [{"action": int, "confidence": float, "regime_id":
        int | None}, ...], one entry per row of ``test_df``.
    train_years : int
        Length of the training window in years.
    test_months : int
        Length of the test (out-of-sample) window in months.  This is also
        the roll-forward step size.
    min_train_rows : int
        Folds with fewer than this many training rows are skipped.
    config : BacktestConfig | None
        Backtest configuration used for every test fold.  All slippage and
        position-sizing settings are applied uniformly to every fold, so the
        comparison is apples-to-apples across time periods.

    Returns
    -------
    dict with keys:
        folds                  -- per-fold results (list of dicts)
        n_folds                -- number of successfully completed folds
        out_of_sample_equity   -- concatenated equity curve across all test folds
        mean_sharpe            -- arithmetic mean Sharpe across folds
        mean_annual_return_pct -- arithmetic mean annualised return across folds
        mean_max_drawdown_pct  -- arithmetic mean max-drawdown across folds
        mean_win_rate          -- arithmetic mean win-rate across folds
        worst_drawdown_pct     -- worst single-fold max-drawdown (tail-risk measure)
        best_sharpe            -- best single-fold Sharpe
        consistency_score      -- fraction of folds with positive annual return

    Example::

        from app.ml.backtester import walk_forward_backtest, BacktestConfig

        def my_train(train_df):
            return retrained_model

        def my_predict(model, test_df):
            return model_predictions(model, test_df)

        wfo = walk_forward_backtest(
            full_df=full_historical_df,
            train_fn=my_train,
            predict_fn=my_predict,
            train_years=3,
            test_months=12,
            config=BacktestConfig(position_sizing="volatility_target"),
        )
        print(f"OOS Sharpe: {wfo['mean_sharpe']}  Consistency: {wfo['consistency_score']:.0%}")
    """
    if config is None:
        config = BacktestConfig()

    # ── Normalise DatetimeIndex ───────────────────────────────────────
    df = full_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            raise ValueError(
                "full_df must have a DatetimeIndex or a 'date' column for WFO date splitting."
            )
    df = df.sort_index()

    close_col = "close" if "close" in df.columns else df.columns[3]

    fold_results: list[dict] = []
    all_equity: list[float] = []
    fold_num = 0

    # Roll the test-window start forward by ``test_months`` each iteration
    test_start = df.index[0] + pd.DateOffset(years=train_years)

    while test_start < df.index[-1]:
        train_start = test_start - pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(months=test_months)

        # Strict wall: training data ends strictly before the first test candle
        train_slice = df.loc[train_start : test_start - pd.Timedelta(days=1)]
        test_slice = df.loc[test_start : test_end - pd.Timedelta(days=1)]

        if len(train_slice) < min_train_rows:
            logger.debug("WFO fold %d skipped — only %d training rows", fold_num, len(train_slice))
            test_start += pd.DateOffset(months=test_months)
            fold_num += 1
            continue

        if len(test_slice) < 5:
            break

        try:
            model = train_fn(train_slice.reset_index())

            test_close = test_slice[close_col].values.astype(float)
            test_dates = test_slice.index.tolist()
            predictions = predict_fn(model, test_slice.reset_index())

            result = run_backtest(
                predictions=predictions,
                close_prices=test_close,
                dates=test_dates,
                config=config,
            )

            fold_dict: dict = {
                "fold": fold_num,
                "train_start": str(train_start.date()),
                "train_end": str((test_start - pd.Timedelta(days=1)).date()),
                "test_start": str(test_start.date()),
                "test_end": str((test_end - pd.Timedelta(days=1)).date()),
                "train_rows": len(train_slice),
                "test_rows": len(test_slice),
                "result": {
                    "total_return_pct": result.total_return_pct,
                    "annual_return_pct": result.annual_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                    "buy_hold_return_pct": result.buy_hold_return_pct,
                    "total_slippage_cost": result.total_slippage_cost,
                },
                "equity_curve": result.equity_curve,
            }
            fold_results.append(fold_dict)
            all_equity.extend(result.equity_curve)

            logger.info(
                "WFO fold %d [%s \u2192 %s]: return=%.2f%%  sharpe=%.3f  dd=%.2f%%  trades=%d",
                fold_num,
                fold_dict["test_start"],
                fold_dict["test_end"],
                result.annual_return_pct,
                result.sharpe_ratio,
                result.max_drawdown_pct,
                result.total_trades,
            )

        except Exception as exc:
            logger.warning("WFO fold %d failed: %s", fold_num, exc)
            fold_results.append({
                "fold": fold_num,
                "train_start": str(train_start.date()),
                "test_start": str(test_start.date()),
                "error": str(exc),
            })

        fold_num += 1
        test_start += pd.DateOffset(months=test_months)

    if not fold_results:
        raise ValueError(
            "No WFO folds were completed.  "
            "Check that full_df spans at least train_years + test_months."
        )

    successful = [f for f in fold_results if "result" in f]

    if not successful:
        return {"folds": fold_results, "n_folds": 0, "error": "All folds failed"}

    sharpes = [f["result"]["sharpe_ratio"] for f in successful]
    returns = [f["result"]["annual_return_pct"] for f in successful]
    drawdowns = [f["result"]["max_drawdown_pct"] for f in successful]
    win_rates = [f["result"]["win_rate"] for f in successful]

    return {
        "folds": fold_results,
        "n_folds": len(successful),
        "out_of_sample_equity": all_equity,
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "mean_annual_return_pct": round(float(np.mean(returns)), 4),
        "mean_max_drawdown_pct": round(float(np.mean(drawdowns)), 4),
        "mean_win_rate": round(float(np.mean(win_rates)), 4),
        "worst_drawdown_pct": round(float(np.max(drawdowns)), 4),
        "best_sharpe": round(float(np.max(sharpes)), 4),
        "consistency_score": round(float(np.mean([r > 0 for r in returns])), 4),
    }

