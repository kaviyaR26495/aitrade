# Plan: AI Trading Platform (aitrade)

## TL;DR

Build a full-stack web application using a THREE-stage ML pipeline:

- **Stage 0 (NEW)**: Market Regime Classifier — classify all historical data into regimes (Bullish/Bearish/Neutral × High-Vol/Low-Vol) and quality-filter training data so RL only learns from clean, regime-tagged segments.
- **Stage 1**: Train RL models (PPO, DDPG, SAC, TD3, A2C) per-regime on quality data to discover profitable trade patterns
- **Stage 2**: Distill those patterns into regime-aware **KNN + LSTM** models — KNN for interpretability (matched patterns), LSTM for temporal sequence modeling — then **ensemble** their predictions (weighted by backtest accuracy) for final trading decisions

NSE only. Daily + Weekly intervals. No intraday. All data (OHLCV, indicators, regimes, predictions) cached in local MySQL for fast queries. Reuse and modernize Zerodha API + KNN + DB patterns from ~/pytrade.

---

## Architecture Overview — Three-Stage Pipeline

```
 ┌── STAGE 0: MARKET REGIME CLASSIFICATION & DATA QUALITY (Offline) ──┐
 │                                                                     │
 │  Raw OHLCV (Zerodha/DB) → Calculate Indicators → Store in DB      │
 │         │                                                           │
 │         ↓                                                           │
 │  Market Regime Classifier:                                          │
 │    ├─ Trend:      Bullish / Bearish / Neutral                      │
 │    ├─ Volatility: High-Vol / Low-Vol                                │
 │    └─ Combined:   6 regimes (Bull+HighVol, Bull+LowVol, ...)      │
 │         │                                                           │
 │         ↓                                                           │
 │  Data Quality Filter:                                               │
 │    ├─ Remove anomalies (circuit hits, gaps, low-volume days)       │
 │    ├─ Remove regime transitions (uncertain boundary periods)        │
 │    └─ Tag each candle with regime label → Store in DB              │
 │         │                                                           │
 │         ↓                                                           │
 │  Quality-classified data segments ready for RL training             │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
 ┌── STAGE 1: REGIME-SPECIFIC RL PATTERN DISCOVERY (Offline) ─────────┐
 │                                                                     │
 │  For EACH regime (or user-selected regimes):                       │
 │    Quality data segment → trading_env → RL Training                │
 │    (PPO/DDPG/SAC/TD3/A2C per regime)                              │
 │              │                                                      │
 │              ↓                                                      │
 │    Best RL model per regime replays historical data                │
 │              │                                                      │
 │              ↓                                                      │
 │    Extract "golden patterns" tagged with regime                    │
 │              │                                                      │
 │              ├──→ Train regime-aware KNN (flattened + regime)       │
 │              │                                                      │
 │              └──→ Train LSTM (sequential + regime)                  │
 │                      │                                              │
 │                      ↓                                              │
 │    Weighted Ensemble: KNN × w1 + LSTM × w2 (weights from backtest)│
 │    + save all model artifacts                                      │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
 ┌── STAGE 2: LIVE INFERENCE (Online) ────────────────────────────────┐
 │                                                                     │
 │  Live daily/weekly data (Zerodha) → DB cache → Indicators          │
 │         │                                                           │
 │         ↓                                                           │
 │  Regime Classifier → detect CURRENT regime                         │
 │         │                                                           │
 │         ├──→ KNN predict (fast, interpretable)                     │
 │         └──→ LSTM predict (temporal-aware)                         │
 │                   │                                                 │
 │                   ↓                                                 │
 │  Ensemble: weighted combination → final BUY/HOLD/SELL             │
 │         │                                                           │
 │         ↓                                                           │
 │  Store prediction in DB → Execute via Zerodha if confirmed        │
 └────────────────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
aitrade/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py             # FastAPI app entry, CORS, lifespan
│   │   ├── config.py           # Pydantic Settings (env-based config)
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py           # Zerodha OAuth login/callback
│   │   │   │   ├── data.py           # Stock data fetch, indicators, sync
│   │   │   │   ├── regime.py         # Regime classification endpoints
│   │   │   │   ├── models.py         # RL + KNN model CRUD, training trigger
│   │   │   │   ├── backtest.py       # Backtesting endpoints
│   │   │   │   ├── trading.py        # Live trade execution (buy/sell/GTT)
│   │   │   │   ├── portfolio.py      # Holdings, positions, P&L
│   │   │   │   ├── config.py         # App settings (IP, params, stock lists)
│   │   │   │   └── chat.py           # LLM chatbot: SSE streaming, provider discovery, status
│   │   │   └── deps.py              # Dependency injection (DB session, Kite client)
│   │   ├── core/
│   │   │   ├── zerodha.py            # KiteConnect wrapper (reuse from pytrade)
│   │   │   ├── data_pipeline.py      # Fetch, clean, DB cache, incremental sync
│   │   │   ├── indicators.py         # Technical indicators (SMA, EMA, RSI, MACD, ADX, BB, KAMA)
│   │   │   ├── normalizer.py         # Normalization logic with save/load params
│   │   │   ├── regime_classifier.py  # Market regime detection & data quality filtering
│   │   │   ├── llm_providers.py      # Multi-provider LLM abstraction (OpenAI, Anthropic, Gemini, Ollama)
│   │   │   └── chatbot_context.py    # System prompt builder, guide-id registry, action protocol
│   │   ├── ml/
│   │   │   ├── rl_trainer.py         # Stage 1: RL training orchestrator
│   │   │   ├── pattern_extractor.py  # Stage 1→2 bridge: replay RL, extract golden patterns
│   │   │   ├── knn_distiller.py      # Stage 2: Train KNN on extracted patterns
│   │   │   ├── lstm_trainer.py      # Stage 2: Train LSTM on extracted patterns (temporal)
│   │   │   ├── ensemble.py          # Stage 2: Weighted ensemble of KNN + LSTM
│   │   │   ├── predictor.py          # Stage 2 live: Ensemble predict next-day action
│   │   │   ├── algorithms.py         # SB3 algorithm registry (PPO, DDPG, SAC, TD3, A2C)
│   │   │   └── callbacks.py          # Training callbacks (logging, checkpointing)
│   │   ├── db/
│   │   │   ├── database.py           # SQLAlchemy async engine + session
│   │   │   ├── models.py             # ORM models
│   │   │   ├── crud.py               # DB operations (bulk insert, upsert, cache queries)
│   │   │   └── cache.py              # DB-first data access layer (DB vs API decision logic)
│   │   └── workers/
│   │       ├── celery_app.py         # Celery config for async jobs
│   │       └── tasks.py              # Training, distillation, sync, regime classification tasks
│   ├── alembic/                      # DB migrations
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── trading_env/                # Standalone pip-installable Gymnasium env library
│   ├── trading_env/
│   │   ├── __init__.py
│   │   ├── envs/
│   │   │   ├── __init__.py
│   │   │   ├── base_env.py           # Base trading environment (daily)
│   │   │   └── swing_trading_env.py  # Multi-day CNC holding simulation
│   │   ├── spaces.py                 # Custom action/observation spaces
│   │   ├── rewards.py               # Reward functions (Sharpe, profit-based, risk-adjusted)
│   │   ├── portfolio.py             # Portfolio state tracking with trade logger
│   │   └── renderer.py             # Visualization (matplotlib)
│   ├── pyproject.toml
│   ├── tests/
│   └── README.md
│
├── frontend/                   # React + Vite SPA
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx         # Overview: active models, P&L, positions
│   │   │   ├── DataManager.tsx       # Stock selection, data fetch, indicator preview
│   │   │   ├── RegimeAnalysis.tsx    # Regime classification view, data quality
│   │   │   ├── ModelStudio.tsx       # Create/train RL models, trigger distillation
│   │   │   ├── PatternLab.tsx        # View extracted golden patterns, KNN config
│   │   │   ├── Backtest.tsx          # Run & visualize backtests (RL and KNN)
│   │   │   ├── LiveTrading.tsx       # Execute trades, manage orders
│   │   │   ├── Portfolio.tsx         # Holdings, positions, order history
│   │   │   └── Settings.tsx          # Zerodha API config, LLM provider config, trading params
│   │   ├── components/
│   │   │   ├── Layout.tsx            # Main layout: Sidebar + Outlet + ChatBot + Notifications
│   │   │   ├── Sidebar.tsx           # Navigation with data-guide-id attributes
│   │   │   ├── ChatBot.tsx           # Floating LLM chat widget: SSE streaming, config, actions
│   │   │   └── ui.tsx                # Shared UI components (Card, Button, Input, Select, StatCard)
│   │   ├── hooks/
│   │   │   └── useApi.ts            # React Query hooks
│   │   ├── services/
│   │   │   └── api.ts               # API client (REST + SSE chat streaming)
│   │   ├── utils/
│   │   │   └── spotlight.ts          # UI element highlight/spotlight for chatbot guidance
│   │   └── store/
│   │       └── appStore.ts           # Zustand global state
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml          # MySQL + Redis + Backend + Frontend + Celery worker
├── ARCHITECTURE.md             # This file
└── README.md
```

---

## Phase 1: Foundation (Backend Core + DB + Trading Env Library)

### Step 1.1 — Project scaffolding & configuration

- Initialize FastAPI project with Pydantic Settings for env-based config
- Config keys: `KITE_API_KEY`, `KITE_API_SECRET`, `ZERODHA_IP` (configurable proxy/IP), `DB_URL`, `REDIS_URL`, `SECRET_KEY`
- Set up `.env.example` with all configurable values
- **NSE only, no intraday**: intervals = ['day', 'week'] — no MIS, no minute data
- **Files**: `backend/app/main.py`, `backend/app/config.py`, `backend/.env.example`, `backend/requirements.txt`

### Step 1.2 — Database schema & ORM with DB caching layer (_parallel with 1.3_)

- MySQL via SQLAlchemy async (reuse MySQL from pytrade)
- Alembic for migrations
- **DB-first caching pattern** (ported from pytrade's `db_common.py` + `df_creator.py`):
  - All OHLCV, indicators, regimes, predictions stored in DB for fast local queries
  - `cache.py` implements the decision logic: DB first → if stale/missing → fetch from Zerodha API → upsert into DB
  - Stale check: compare `MAX(date)` in DB vs `get_last_trading_day()` (reuse pytrade's holiday calendar pattern)
  - Bulk insert with `ON DUPLICATE KEY UPDATE` (reuse pytrade's batch pattern: 5000 rows/batch for OHLCV, 10000 rows/batch for indicators)
- **Tables**:
  - `stocks_list` — instrument metadata (symbol, exchange='NSE', kite_id, tick_size, lot_size, sector, is_active)
  - `nse_holidays` — trading calendar (trading_date, week_day, description) — reuse pytrade's `holidays` table pattern, fetch from NSE API
  - **`stock_ohlcv`** — historical OHLCV cached locally (stock_id FK, date, open, high, low, close, adj_close, volume, interval ENUM('day','week')) — PRIMARY KEY (stock_id, date, interval) — the core DB cache, all queries hit this first
  - **`stock_indicators`** — pre-computed indicators cached (stock_id FK, date, interval, sma_5, sma_12, sma_24, sma_50, sma_100, sma_200, ema_20, rsi, srsi, macd, macd_signal, macd_hist, adx, adx_pos, adx_neg, kama, vwkama, obv, bb_upper, bb_lower, bb_mid, tgrb_top, tgrb_green, tgrb_red, tgrb_bottom) — PRIMARY KEY (stock_id, date, interval)
  - **`stock_regimes`** — NEW: regime classification per candle (stock_id FK, date, interval, trend ENUM('bullish','bearish','neutral'), volatility ENUM('high','low'), regime_id INT [0-5 combined], regime_confidence FLOAT, quality_score FLOAT, is_transition BOOLEAN) — PRIMARY KEY (stock_id, date, interval)
  - `rl_models` — RL model registry (id, name, algorithm, hyperparams JSON, features JSON, training_config JSON, regime_filter JSON [which regimes used for training], interval, total_reward, sharpe_ratio, model_path, status, created_at)
  - `rl_training_runs` — RL training progress (rl_model_id FK, timestep, episode, reward, loss, metrics JSON, timestamp)
  - `golden_patterns` — extracted patterns (id, rl_model_id FK, stock_id FK, date, interval, feature_window BLOB, label [1/-1/0], pnl_percent, regime_id, confidence)
  - `knn_models` — distilled KNN registry (id, name, source_rl_model_id FK, k_neighbors, feature_combination, seq_len, interval, regime_filter JSON, accuracy, precision_buy, precision_sell, model_path, regime_classifier_path, norm_params_path, status, created_at)
  - `lstm_models` — distilled LSTM registry (id, name, source_rl_model_id FK, hidden_size, num_layers, dropout, seq_len, interval, regime_filter JSON, accuracy, precision_buy, precision_sell, model_path, norm_params_path, status, created_at)
  - `ensemble_configs` — ensemble weights (id, name, knn_model_id FK, lstm_model_id FK, knn_weight FLOAT, lstm_weight FLOAT, agreement_required BOOLEAN, interval, backtest_accuracy, created_at)
  - **`knn_predictions`** — daily/weekly predictions cached in DB (knn_model_id FK, stock_id FK, date, interval, action [1/0/-1], confidence, proba_buy FLOAT, proba_sell FLOAT, proba_hold FLOAT, regime_id, matched_pattern_ids JSON) — PRIMARY KEY (knn_model_id, stock_id, date, interval) — queried by trading logic for fast lookups
  - `lstm_predictions` — LSTM predictions (lstm_model_id FK, stock_id FK, date, interval, action, confidence, proba_buy, proba_sell, proba_hold, regime_id) — PRIMARY KEY (lstm_model_id, stock_id, date, interval)
  - `ensemble_predictions` — final ensemble predictions (ensemble_config_id FK, stock_id FK, date, interval, action, confidence, knn_action, knn_confidence, lstm_action, lstm_confidence, agreement BOOLEAN, regime_id) — PRIMARY KEY (ensemble_config_id, stock_id, date, interval) — **this is the primary table for trading decisions**
  - `backtest_results` — backtest outcomes for RL and KNN (id, model_type, model_id, stock_id FK, interval, start_date, end_date, total_return, win_rate, max_drawdown, sharpe, profit_factor, trades_count, trade_log JSON)
  - `trade_orders` — executed trades via Zerodha (order_id, stock_id FK, knn_model_id FK, prediction_id FK, variety, transaction_type, quantity, price, sl_price, target_price, status, zerodha_order_id, tag, timestamp)
  - `settings` — key-value config (property, value — access_token, zerodha_ip, etc.)
- **Indexes** (for fast queries, matching pytrade's PK join pattern):
  - All tables: composite PK on (stock_id, date, interval) — enables fast JOIN across ohlcv ⟷ indicators ⟷ regimes ⟷ predictions
  - `knn_predictions`: additional index on (date, action) for "get all BUY predictions for today"
  - `stock_regimes`: additional index on (regime_id, date) for "get all bullish days"
- **Prediction query pattern** (modernized from pytrade's `get_filtered_buy_stocks()`):
  ```sql
  SELECT o.symbol, o.close, p.confidence, p.proba_buy, r.trend, r.volatility
  FROM stock_ohlcv o
  JOIN knn_predictions p ON o.stock_id = p.stock_id AND o.date = p.date
  JOIN stock_regimes r ON o.stock_id = r.stock_id AND o.date = r.date
  JOIN stock_indicators i ON o.stock_id = i.stock_id AND o.date = i.date
  WHERE o.date = :today AND o.interval = 'day'
    AND p.action = 1 AND p.confidence > 0.65
    AND r.trend != 'bearish'
  ORDER BY p.confidence DESC
  ```
- **Files**: `backend/app/db/database.py`, `backend/app/db/models.py`, `backend/app/db/crud.py`, `backend/app/db/cache.py`, `backend/alembic/`

### Step 1.3 — Trading Environment Library (_parallel with 1.2_)

- Standalone pip-installable Gymnasium environment
- **NO intraday env** — only swing/positional trading on daily + weekly data
- **Critical for RL→KNN distillation**: `TradeLogger` records every action + raw feature window + P&L
- **Base Environment** (`base_env.py`):
  - **Two observation modes** (configurable):
    - `obs_mode='flat'` (default for MLP policies): flattened `seq_len × features` → 1D vector. Used by PPO-MLP, A2C, DDPG, TD3, SAC.
    - `obs_mode='sequential'` (for RecurrentPPO LSTM policy): single timestep features per step → `(num_features + regime_features,)`. The LSTM policy's internal hidden state handles the temporal context. Env feeds one candle at a time.
  - Action space: Discrete(3) — Buy(1)/Hold(0)/Sell(-1)
  - Portfolio tracking: cash, holdings, net worth, NSE costs (brokerage 0.03% + STT 0.1% sell-side + DP charges)
  - Realistic: T+1 settlement for daily, no circuit-hit candles (pre-filtered by data quality layer)
  - `TradeLogger`: records action, raw feature window, resulting P&L after `profit_horizon` days
  - **NormalizerState**: captures rolling stats per feature → serializable JSON
- **Swing Trading Env** (`swing_trading_env.py`): CNC-style, configurable holding period, SL default 5%, profit_horizon=1 (next-day) or 5 (next-week)
  - For daily interval: profit_horizon=1 (predict next trading day)
  - For weekly interval: profit_horizon=1 (predict next trading week)
- **Reward Functions** (`rewards.py`):
  - `risk_adjusted_pnl` — P&L penalized by drawdown (default)
  - `sharpe_reward` — rolling Sharpe
  - `sortino_reward` — downside-risk adjusted
  - `profit_reward` — raw P&L
- **Files**: `trading_env/` entire package

### Step 1.4 — Zerodha API Integration Layer

- KiteConnect wrapper with configurable IP routing
- Reuse from pytrade: `place_order()`, `place_GTT_order()`, `place_AMO_MIS_order()`, `get_held_holdings()`, `exit_all_with_profit()`, `get_LTP()`, `get_close_price()`
- Web-based OAuth (redirect callback, not Selenium)
- Configurable proxy/IP: `ZERODHA_IP` env var → `proxies` param in KiteConnect session
- **NSE only**: filter instruments where `exchange='NSE'`
- **Files**: `backend/app/core/zerodha.py`, `backend/app/api/routes/auth.py`

---

## Phase 2: Data Pipeline, Indicators & DB Caching

### Step 2.1 — Data fetching + DB caching service

- **DB-first fetch pattern** (ported from pytrade's `get_dfs()` decision logic):
  1. Query `stock_ohlcv` table for requested stock + date range + interval
  2. Check freshness: if `MAX(date)` < `get_last_trading_day(today)` → data is stale
  3. If stale or missing: fetch from Zerodha Kite API using 2000-day chunking (reuse pytrade's `fetch_history()`)
  4. Upsert fetched data into `stock_ohlcv` using bulk `ON DUPLICATE KEY UPDATE` (5000 rows/batch from pytrade pattern)
  5. Return DataFrame from DB
- **Incremental sync**: only fetch new candles since `MAX(date)` in DB
- **Intervals**: `day` and `week` only (no intraday)
- **Exchange**: NSE only (filter `stocks_list.exchange = 'NSE'`)
- **Stock universe**: configurable via UI, default to pytrade's 500+ NSE stocks
- **Holiday calendar**: fetch from `https://www.nseindia.com/api/holiday-master?type=trading` → store in `nse_holidays` table (reuse pytrade's `holidays_db_updater.py` pattern)
- **Scheduled sync** (Celery Beat): run daily after market close (3:45 PM IST) — update all stocks' daily data; weekly on Saturday for weekly data
- **Files**: `backend/app/core/data_pipeline.py`, `backend/app/db/cache.py`, `backend/app/api/routes/data.py`

### Step 2.2 — Technical Indicators + DB caching (_depends on 2.1_)

- Calculate indicators → store in `stock_indicators` table (reuse pytrade's `set_bulk_db_stock_indicator()` pattern — 10000 rows/batch, dynamic column matching)
- **Indicator pipeline**: after OHLCV sync, auto-compute indicators for new candles and upsert into DB
- Indicators (matching pytrade's full set):
  - SMA: 5, 12, 24, 50, 100, 200
  - EMA: 20
  - RSI(14), Stochastic RSI
  - MACD(12,26,9) + signal + histogram
  - ADX(14) + ADX_pos + ADX_neg
  - Bollinger Bands(20,2): upper, lower, mid
  - KAMA(10,2,30), VW-KAMA
  - OBV
  - TGRB candle structure: Top, Green, Red, Bottom (ratios to Close)
- **Combination system** (from pytrade): available groups [TGRB, rsi, srsi, kama, vwkama, obv, bbl, macd, adx, sma] — user selects via UI
- **Both intervals**: indicators computed for daily AND weekly data separately
- **Files**: `backend/app/core/indicators.py`, `backend/app/db/crud.py`

### Step 2.3 — Normalization Pipeline (_depends on 2.2_)

- `Normalizer` class with `fit()`/`transform()`/`save()`/`load()`:
  - TGRB: already ratio-normalized (pass-through)
  - OHLCV: log returns → Z-score (rolling 50-bar window) — reuse pytrade pattern
  - Volume: log + MinMaxScaler
  - RSI: [0,100] → [0,1]
  - MACD/ADX/KAMA: Z-score (rolling 50-bar)
  - SMA/EMA: ratio to Close (SMA/Close - 1)
- Fit on training date range only; save params in DB alongside model
- Sliding window: `seq_len=15` for daily, `seq_len=10` for weekly → flatten to 1D
- **Files**: `backend/app/core/normalizer.py`

---

## Phase 3: Market Regime Classification & Data Quality (Stage 0 — NEW)

**This is the key new layer — classify data BEFORE training so RL learns from quality, regime-specific segments instead of raw start-to-end data.**

### Step 3.1 — Regime Classification Algorithm

- **Multi-signal regime detector** (`regime_classifier.py`) using a combination of rules + unsupervised learning:

  **A. Trend Detection (3 classes: Bullish / Bearish / Neutral)**:
  - **Primary signal**: SMA crossover system
    - Bullish: SMA_50 > SMA_200 AND Close > SMA_50 (golden cross territory)
    - Bearish: SMA_50 < SMA_200 AND Close < SMA_50 (death cross territory)
    - Neutral: mixed signals (SMA_50 ≈ SMA_200 within 2%, or Close between the two SMAs)
  - **Confirming signals** (weighted vote):
    - ADX > 25 → trending (strengthens bull/bear), ADX < 20 → ranging (pushes toward neutral)
    - RSI > 60 → bullish bias, RSI < 40 → bearish bias, 40-60 → neutral
    - MACD histogram positive (3 consecutive bars) → bullish, negative → bearish
    - Price above/below Bollinger mid → directional bias
  - **Final trend**: weighted majority vote of all signals (SMA cross gets 40% weight, others 15% each)

  **B. Volatility Classification (2 classes: High-Vol / Low-Vol)**:
  - **ATR-based**: Calculate ATR(14) as % of Close → `ATR_pct`
    - High-Vol: ATR_pct > rolling 90th percentile (over last 252 trading days / 1 year)
    - Low-Vol: ATR_pct ≤ 90th percentile
  - **Confirming**: Bollinger Band Width (BB_upper - BB_lower) / BB_mid
    - Wide BBands confirm High-Vol, narrow confirm Low-Vol
  - **Historical volatility**: 20-day rolling std of log returns as additional signal

  **C. Combined Regime (6 classes, mapped to regime_id 0-5)**:

  | regime_id | Trend   | Volatility | Training value             | Common pattern                  |
  | --------- | ------- | ---------- | -------------------------- | ------------------------------- |
  | 0         | Bullish | Low-Vol    | HIGH — clean uptrend       | Best for BUY model training     |
  | 1         | Bullish | High-Vol   | MEDIUM — volatile rally    | Risky BUY, needs tight SL       |
  | 2         | Neutral | Low-Vol    | LOW — sideways chop        | Avoid training, mostly HOLD     |
  | 3         | Neutral | High-Vol   | LOW — whipsaw              | Worst data quality, skip        |
  | 4         | Bearish | Low-Vol    | HIGH — clean downtrend     | Good for SELL model training    |
  | 5         | Bearish | High-Vol   | HIGH (avoid) — crash/panic | Important to learn SELL signals |

  **D. Regime Confidence Score (0-1)**:
  - How strongly signals agree. If all 5 signals say "bullish" → confidence=1.0. If 3 say bullish, 2 neutral → confidence=0.6
  - Stored in `stock_regimes.regime_confidence`

- **Per candle output**: (trend, volatility, regime_id, confidence, is_transition) → stored in `stock_regimes` table
- **Process for both intervals**: daily AND weekly separately (weekly regime = broader market context)
- **Files**: `backend/app/core/regime_classifier.py`

### Step 3.2 — Data Quality Scoring & Filtering (_depends on 3.1_)

- **Quality score per candle** (stored in `stock_regimes.quality_score`, 0-1):
  - Deductions:
    - `-0.3`: Circuit hit (Close = Upper/Lower circuit, daily move > 10%)
    - `-0.3`: Volume anomaly (volume < 20th percentile of 50-day rolling, i.e., illiquid day)
    - `-0.2`: Data gap (previous trading day missing — stock was suspended/halted)
    - `-0.2`: Regime transition (current candle's regime differs from previous — boundary uncertainty)
    - `-0.1`: Extreme outlier (daily return > 3σ from 50-day rolling mean — could be split/bonus, not real move)
  - Base score = 1.0, apply deductions, floor at 0.0
- **Transition detection** (`is_transition` flag):
  - Mark candles where regime changes from previous candle (e.g., Bullish→Neutral)
  - Also mark ±2 candles around the transition (buffer zone)
  - These are uncertain periods — optionally exclude from training
- **Quality tiers for training**:
  - **Tier 1** (quality_score ≥ 0.8): Clean data — use for ALL training
  - **Tier 2** (0.5 ≤ quality_score < 0.8): Acceptable — use if user opts in
  - **Tier 3** (quality_score < 0.5): Poor — exclude from training by default
  - User configurable via UI: minimum quality threshold slider
- **Celery task**: After OHLCV + indicators sync, auto-run regime classification for new candles → upsert into `stock_regimes`
- **Files**: `backend/app/core/regime_classifier.py`

### Step 3.3 — Regime-Based Data Segmentation for Training (_depends on 3.2_)

- When user triggers RL training, data pipeline:
  1. Query `stock_ohlcv` JOIN `stock_indicators` JOIN `stock_regimes` for selected stocks + date range
  2. Filter: `quality_score >= min_quality_threshold` (default 0.8)
  3. Filter: `is_transition = FALSE` (skip boundary periods, configurable)
  4. **Regime selection**: user chooses which regimes to train on:
     - Option A (default): **All regimes** — train one model on all quality data, regime appended as feature. The RL model learns to behave differently per regime.
     - Option B: **Per-regime models** — train SEPARATE RL models for each regime (e.g., one PPO for Bullish+LowVol, another for Bearish+HighVol). More specialized but needs enough data per regime.
     - Option C: **Selected regimes only** — user picks specific regimes (e.g., only Bullish regimes) for a BUY-focused model
  5. Regime features added to observation space: trend (one-hot: 3 values) + volatility (binary) + confidence (float) = 5 extra features appended to each candle
  6. Pass filtered, regime-tagged data to trading_env for RL training

- **Weekly regime as macro context**: For daily training, also fetch the weekly regime for each date (the regime of the week that the day falls in). Append as additional features: `weekly_trend`, `weekly_volatility`. This gives the model both micro (daily) and macro (weekly) regime awareness.
- **Files**: `backend/app/core/data_pipeline.py`, `backend/app/ml/rl_trainer.py`

---

## Phase 4: RL Training (Stage 1 — Regime-Specific Pattern Discovery)

### Step 4.1 — RL Algorithm Registry

- Stable-Baselines3 + **sb3-contrib** for all RL algorithms:
  - **PPO** (MLP): lr=3e-4, n_steps=2048, clip=0.2 (Discrete action) — standard frame-stacked observation
  - **RecurrentPPO** (LSTM policy, from `sb3-contrib`): same hyperparams + `lstm_hidden_size=256`, `n_lstm_layers=1`, `enable_critic_lstm=True`. Observation is per-timestep (NOT flattened) — the LSTM policy maintains hidden state across steps, learning temporal dependencies within an episode. **This is the primary LSTM-in-RL option.**
  - **A2C** (MLP): lr=7e-4, n_steps=5 — frame-stacked observation
  - **DDPG**: lr=1e-3, buffer=1M, tau=0.005 (Continuous → discretize: <-0.33=SELL, >0.33=BUY) — MLP only (off-policy replay buffer incompatible with recurrent policies)
  - **TD3**: DDPG + policy_delay=2, target_noise=0.2 — MLP only
  - **SAC**: lr=3e-4, ent_coef=auto — MLP only
- **LSTM policy vs MLP policy tradeoff**:
  - MLP (PPO/A2C/DDPG/TD3/SAC): sees flattened `seq_len × features` as a single vector. Simple, fast, all algorithms available.
  - LSTM (RecurrentPPO only): processes observations one timestep at a time, maintaining hidden state across the episode. Learns temporal patterns (e.g., "3 consecutive rising RSI then plateau") that MLP with frame-stacking cannot capture. Slower training, but potentially better pattern discovery.
  - **Off-policy limitation**: DDPG/TD3/SAC use replay buffers that sample random transitions — this breaks LSTM hidden state continuity. No official SB3 support. Frame-stacking is the standard workaround.
- **UI toggle**: for PPO, user selects "MLP" or "LSTM" policy type. Other algorithms are MLP-only.
- All configurable via UI
- **Files**: `backend/app/ml/algorithms.py`

### Step 4.2 — Regime-Aware RL Training Orchestrator (_depends on 1.3, 2.3, 3.3, 4.1_)

- Celery async jobs
- **Training flow incorporating regime-classified data**:
  1. User selects via UI: stock(s), algorithm, hyperparams, date range, feature combination, reward function, **minimum quality threshold** (default 0.8), **regime filter** (which regimes to include)
  2. Backend queries DB: `stock_ohlcv JOIN stock_indicators JOIN stock_regimes WHERE quality_score >= threshold AND regime_id IN (selected) AND is_transition = FALSE`
  3. Appends regime features to observation: daily trend (one-hot 3), daily volatility (binary), weekly trend (one-hot 3), weekly volatility (binary), confidence (float) = **9 extra features** per candle
  4. Applies normalization (fit on first 80% chronologically)
  5. Creates `SwingTradingEnv` with `obs_mode='sequential'` if RecurrentPPO, else `obs_mode='flat'`
  6. For RecurrentPPO: uses `from sb3_contrib import RecurrentPPO` with `MlpLstmPolicy`. For others: standard SB3 `model.learn()`
  7. SB3 `model.learn(total_timesteps=N)` with callbacks:
     - `ProgressCallback` → WebSocket + `rl_training_runs` table
     - `CheckpointCallback` → save every M steps
     - `EvalCallback` → evaluate on held-out quality data
  8. Model saved to `.zip`, metadata (including `regime_filter` JSON) saved to `rl_models` table
  9. **Auto-trigger pattern extraction** (Step 5.1)
- **Training options**:
  - **Single universal model** (default): all selected regimes as training data, regime as feature — model learns to adapt
  - **Per-regime models**: separate training run per regime — more specialized
  - "**Train All Algorithms**" button → 6 algorithms (PPO-MLP, RecurrentPPO-LSTM, A2C, DDPG, TD3, SAC) × selected regime strategy, auto-compare
- **Dual interval**: user can train on daily data, weekly data, or BOTH (daily with weekly regime context appended)
- WebSocket `/ws/training/{task_id}` for live progress
- **Files**: `backend/app/ml/rl_trainer.py`, `backend/app/ml/callbacks.py`, `backend/app/workers/tasks.py`

### Step 4.3 — RL Model Comparison (_depends on 4.2_)

- Compare all algorithms on same quality-filtered data + same regimes
- Metrics: total return, Sharpe, max drawdown, win rate, profit factor
- Auto-rank, recommend best for distillation
- **Files**: `backend/app/ml/rl_trainer.py`, `backend/app/api/routes/models.py`

---

## Phase 5: RL→KNN+LSTM Distillation (Stage 1→2 Bridge)

### Step 5.1 — Pattern Extraction from RL (_depends on 4.2_)

- **Core: replace pytrade's `sequential()` labels with RL-discovered patterns**
- **Extraction flow**:
  1. Load best RL model → replay on FULL historical quality data (deterministic mode)
  2. Env's `TradeLogger` captures: date, action, raw feature window, P&L after `profit_horizon` days
  3. **Labeling**:
     - **Label 1** (BUY): RL chose BUY AND P&L > `min_profit_threshold` (default 1.2%)
     - **Label -1** (SELL): RL chose SELL AND P&L > threshold
     - **Label 0**: everything else
  4. Each pattern also tagged with regime_id from `stock_regimes` table
  5. Store in `golden_patterns` table
- Configurable: `min_profit_threshold`, `profit_horizon` (1 for daily, 1 for weekly), `min_confidence`
- **Files**: `backend/app/ml/pattern_extractor.py`

### Step 5.2 — KNN Distillation (_depends on 5.1_)

- Build KNN from golden patterns:
  1. Handle class imbalance: SMOTE oversampling (default), configurable
  2. Feature = `seq_len` candles × features flattened + regime features (trend one-hot + volatility + weekly regime)
  3. Train `KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)`
  4. Time-series split: first 80% train, last 20% test (chronological)
  5. Save 3 artifacts: KNN `.joblib` + regime classifier params `.json` + Normalizer `.json`
  6. Metrics and regime_filter stored in `knn_models` table
- **Files**: `backend/app/ml/knn_distiller.py`

### Step 5.3 — LSTM Distillation (_parallel with 5.2, depends on 5.1_)

- **Key advantage over KNN: LSTM preserves temporal ordering** — KNN flattens `seq_len × features` into 1D, losing the sequential structure. LSTM processes the sequence step-by-step, learning temporal dependencies (e.g., "RSI rising for 3 days then plateauing" vs "RSI was high on 3 scattered days").
- **Architecture** (PyTorch):
  1. Input: `(batch, seq_len, num_features + regime_features)` — NO flattening, keep 2D sequence
  2. LSTM layers: `nn.LSTM(input_size=num_features, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)`
  3. Final hidden state → `nn.Linear(128, 64)` → ReLU → `nn.Dropout(0.3)` → `nn.Linear(64, 3)` → Softmax
  4. Output: 3-class probabilities (SELL, HOLD, BUY)
- **Training**:
  1. Same golden patterns from Step 5.1, same chronological 80/20 split as KNN
  2. Class imbalance: weighted `CrossEntropyLoss(weight=class_weights)` — computed from label distribution
  3. Optimizer: Adam, lr=1e-3 with ReduceLROnPlateau scheduler
  4. Early stopping: patience=10 epochs on validation loss
  5. Batch size: 64, max epochs: 100
  6. Regime features appended to each timestep (not just end) — LSTM sees regime context throughout the sequence
- **Hyperparams** (configurable via UI):
  - `hidden_size`: 64 / 128 / 256 (default 128)
  - `num_layers`: 1 / 2 / 3 (default 2)
  - `dropout`: 0.1-0.5 (default 0.3)
  - `learning_rate`: 1e-4 to 1e-2 (default 1e-3)
- **Save**: PyTorch model `.pt` + Normalizer `.json` → paths in `lstm_models` table
- **Files**: `backend/app/ml/lstm_trainer.py`

### Step 5.4 — Ensemble Configuration (_depends on 5.2, 5.3_)

- **Combine KNN + LSTM for final predictions**:
  1. Run both models on the held-out 20% test data
  2. Compute per-model accuracy, precision_buy, precision_sell on test set
  3. **Weight calculation**: weight proportional to backtest accuracy on test set
     - Default: `w_knn = acc_knn / (acc_knn + acc_lstm)`, `w_lstm = acc_lstm / (acc_knn + acc_lstm)`
     - User can override weights manually via UI
  4. **Ensemble prediction**: `final_proba = w_knn * knn_proba + w_lstm * lstm_proba` → argmax → action
  5. **Agreement filter** (optional, default ON): only trade when KNN and LSTM agree on action (both say BUY or both say SELL). Disagreements → HOLD. This is a conservative safety filter.
  6. Store config in `ensemble_configs` table
- **Auto-ensemble**: after both KNN and LSTM train, auto-create ensemble config with computed weights
- **Files**: `backend/app/ml/ensemble.py`

### Step 5.5 — Validation (_depends on 5.4_)

- Backtest ensemble vs KNN-only vs LSTM-only vs source RL on held-out period
- Compare: accuracy, precision_buy, Sharpe, drawdown
- Expected: ensemble ≥ best individual model (diversity reduces variance)
- Warn if any individual model accuracy < 55% or precision_buy < 50%

---

## Phase 6: Backtesting (RL + KNN + LSTM + Ensemble)

### Step 6.1 — Unified Backtester (_depends on 4.2, 5.4_)

- Works for RL, KNN, LSTM, and **ensemble**, daily and weekly intervals
- **KNN backtest flow**: load KNN + Normalizer + regime classifier → for each date:
  1. Query `stock_ohlcv JOIN stock_indicators` from DB (fast local query)
  2. Regime classify current candle → regime features
  3. Build feature window → normalize → KNN predict + proba
  4. Trade if confidence > threshold, apply SL/target
- **LSTM backtest flow**: load LSTM `.pt` + Normalizer → for each date:
  1. Same data query as KNN
  2. Build feature window as 2D sequence (seq_len × features) — NO flattening
  3. Normalize, append regime features per timestep
  4. LSTM forward pass → softmax probabilities
  5. Trade if confidence > threshold
- **Ensemble backtest flow**: run KNN + LSTM in parallel → weighted combination → apply agreement filter
- **For KNN**: show matched historical patterns via `kneighbors()`
- **For LSTM**: show attention-like importance via gradient-based saliency (which timesteps mattered most)
- Output: equity curve, trade log, statistics (return, Sharpe, drawdown, win rate, profit factor)
- **Results stored in `backtest_results` table** for fast re-access
- **Files**: `backend/app/api/routes/backtest.py`, `backend/app/workers/tasks.py`

### Step 6.2 — Side-by-Side Comparison

- Overlay: RL vs KNN vs LSTM vs **Ensemble** vs buy-and-hold, filtered by regime period
- Shows where ensemble agrees vs disagrees — marks missed opportunities and avoided bad trades

---

## Phase 7: Live Prediction & Trading (DB-Cached Predictions)

### Step 7.1 — Daily/Weekly Prediction Service (_depends on 5.4_)

- **Uses KNN + LSTM ensemble for production** — weighted combination, optional agreement filter
- **Scheduled via Celery Beat**:
  - **Daily**: after market close (3:45 PM IST) — predict next trading day for all stocks
  - **Weekly**: Friday after close — predict next trading week
- **Prediction flow**:
  1. Ensure today's OHLCV is synced to DB (from Zerodha → `stock_ohlcv`)
  2. Ensure today's indicators are computed and in `stock_indicators`
  3. Classify today's regime → update `stock_regimes`
  4. For each stock in universe:
     a. Build feature window from DB (last `seq_len` candles from `stock_ohlcv JOIN stock_indicators`)
     b. **KNN path**: flatten → normalize with KNN normalizer → KNN predict → store in `knn_predictions`
     c. **LSTM path**: keep 2D sequence → normalize with LSTM normalizer → LSTM forward → store in `lstm_predictions`
     d. **Ensemble**: load `ensemble_configs` → `final_proba = w_knn * knn_proba + w_lstm * lstm_proba` → if `agreement_required` and KNN disagrees with LSTM → HOLD → store in `ensemble_predictions`
  5. **All predictions stored in respective DB tables** with `ON DUPLICATE KEY UPDATE`
  6. Filter: ensemble confidence > `min_trade_confidence` (default 0.65) AND regime favorable AND (if agreement mode ON) both models agree
  7. Ranked results immediately queryable from DB
- **Manual trigger**: user can re-run prediction for any stock/date via UI
- **Historical prediction tracking**: `ensemble_predictions.actual_outcome` updated next day with real P&L — allows accuracy tracking over time, per-model and ensemble
- **Query for trading**: `SELECT * FROM ensemble_predictions WHERE date = :today AND action = 1 AND confidence > 0.65 AND agreement = TRUE ORDER BY confidence DESC` — instant, all from local DB
- **Fallback**: if only KNN or only LSTM is trained, use the available model alone (no ensemble)
- **Files**: `backend/app/ml/predictor.py`, `backend/app/ml/ensemble.py`, `backend/app/api/routes/models.py`

### Step 7.2 — Trade Execution (_depends on 7.1, 1.4_)

- Based on **ensemble predictions** from DB, create Zerodha orders
- **Order types** (reuse from pytrade): Regular CNC BUY/SELL, AMO, GTT, SL
- **No MIS/intraday** — CNC only for daily/weekly predictions
- Auto SL = 5% (default for CNC), configurable target
- Monthly tags (pytrade's `get_tag()`), buy limit ₹10,000
- User confirmation required (modal in UI shows: KNN confidence, LSTM confidence, ensemble confidence, agreement status)
- Order stored in `trade_orders` table with `ensemble_prediction_id` FK back to `ensemble_predictions`
- **Files**: `backend/app/api/routes/trading.py`

### Step 7.3 — Portfolio Monitoring

- Holdings/positions via Kite API, WebSocket live prices
- Kill switch: exit all
- Track model-initiated vs manual trades
- **Files**: `backend/app/api/routes/portfolio.py`

---

## Phase 8: Frontend (React + TypeScript + shadcn/ui)

### Step 8.1 — Project setup & layout

- Vite + React + TypeScript + TailwindCSS + Zustand + TanStack React Query
- **Pages**: Dashboard, Data Manager, Regime Analysis, Model Studio, Pattern Lab, Backtest, Live Trading, Portfolio, Settings
- **Layout**: Sidebar + page content (Outlet) + ChatBot (floating widget) + Notifications
- All interactive elements tagged with `data-guide-id` for chatbot spotlight guidance
- UI components (Card, Button, Input, Select, StatCard) forward `...rest` props to support `data-*` attributes

### Step 8.2 — Settings page

- Zerodha API key/secret + OAuth trigger
- **Configurable IP address**
- Stock universe selector (NSE only)
- Feature combination checkboxes
- Trading params: SL%, target%, buy limit, min confidence threshold
- **Quality threshold slider** (0.5-1.0, default 0.8)
- seq_len (default 15 for daily, 10 for weekly)
- Interval selector: Daily / Weekly / Both
- **Chat Assistant config**: LLM provider dropdown, model selector, API key input, Ollama URL, save/test

### Step 8.3 — Data Manager page

- Stock search (NSE only), date range picker
- **Data sync status**: shows per-stock — last synced date, data freshness, "Sync Now" button
- **DB stats**: total rows in stock_ohlcv, stock_indicators, stock_regimes
- Candlestick chart + indicator overlays

### Step 8.4 — Regime Analysis page (NEW)

- **Regime Timeline**: full-width price chart with colored background bands showing regime classification over time
  - Green = Bullish+LowVol, Light green = Bullish+HighVol
  - Red = Bearish+LowVol, Dark red = Bearish+HighVol
  - Gray = Neutral+LowVol, Yellow = Neutral+HighVol
- **Regime Distribution**: pie/bar chart showing % of data in each regime
- **Quality Heatmap**: calendar view, each day colored by quality_score (green=high, red=low)
- **Transition Points**: highlighted on timeline where regime changes
- **Filter preview**: "If I set quality ≥ 0.8, how much training data do I keep?" — shows filtered vs total candle count per stock
- **Data quality stats table**: per stock — total candles, quality ≥ 0.8 count, regime breakdown, circuit hits excluded, gaps detected
- **Regime-vs-Return scatter**: for each regime, show average daily return — confirms that Bullish regimes actually had positive returns

### Step 8.5 — Model Studio page

- **RL Training tab**:
  - Algorithm picker (PPO, RecurrentPPO, A2C, DDPG, TD3, SAC), hyperparams form, stocks, date range
  - **For PPO**: toggle "LSTM Policy" → switches to RecurrentPPO with extra params (lstm_hidden_size, n_lstm_layers)
  - **Regime filter**: checkboxes for which regimes to include (with data count per regime shown)
  - **Quality slider**: min quality threshold
  - **Interval**: Daily / Weekly
  - "Train All Algorithms" button
  - WebSocket progress: reward curve, loss
  - RL comparison table: algorithm, reward, Sharpe, win rate — per regime
- **Distillation tab**:
  - Select source RL model → "Extract Patterns" → shows extraction stats (BUY/SELL/HOLD counts per regime)
  - **KNN config**: k, balancing method → "Train KNN" → metrics + confusion matrix
  - **LSTM config**: hidden_size, num_layers, dropout, lr → "Train LSTM" → training loss curve + metrics + confusion matrix
  - **"Train Both"** button → trains KNN + LSTM in parallel, auto-creates ensemble
  - Side-by-side: KNN accuracy vs LSTM accuracy, per-class precision comparison
- **Ensemble tab**:
  - Shows current ensemble weights (auto-computed + manual override slider)
  - **Agreement filter toggle** (ON/OFF, default ON)
  - Ensemble vs individual model accuracy table
  - "Re-calibrate Weights" button: re-run on latest test data

### Step 8.6 — Pattern Lab page

- Golden pattern explorer: table with filters (stock, label, regime, quality, date)
- Click pattern → mini candlestick + indicator values + RL action + P&L + regime
- PCA/t-SNE scatter colored by label
- **KNN Neighborhood Inspector**: for any prediction, show K matched patterns as mini charts + their regime + their P&L
- **LSTM Sequence Inspector**: for any prediction, show the input sequence with saliency highlighting (which timesteps and features the LSTM weighted most — via gradient × input)
- **Regime overlay**: patterns colored by regime to see if BUY patterns cluster in certain regimes

### Step 8.7 — Backtest page

- Model selector (RL/KNN/LSTM/**Ensemble**), stock, date range
- Equity curve with regime-colored background
- Trade markers on chart
- Stats: return, Sharpe, drawdown, win rate — **broken down by regime** (e.g., "Win rate in Bullish: 72%, in Neutral: 45%")
- **Ensemble insight**: highlight trades where KNN and LSTM disagreed — show which was right
- Side-by-side comparison overlays: RL vs KNN vs LSTM vs Ensemble vs buy-and-hold

### Step 8.8 — Live Trading page

- **Today's Predictions**: from `ensemble_predictions` table (instant query)
  - Ranked by ensemble confidence, shows: stock, **ensemble action + confidence**, KNN confidence, LSTM confidence, **agreement badge** (checkmark if both agree, warning if disagree), current regime, matched patterns (mini charts)
  - Filter by: regime, min confidence, action type, **agreement only**
- **Weekly Predictions tab**: same for weekly interval
- One-click order (CNC only, with confirmation modal showing KNN + LSTM + ensemble breakdown)
- Open orders, positions, recent trades
- Kill switch

### Step 8.9 — Dashboard

- Summary: total P&L, active ensemble model + accuracy (KNN acc / LSTM acc / ensemble acc), positions count
- **Current market regime indicator**: shows overall market regime (based on NIFTY 50 regime classification)
- **Model agreement rate**: last 30 days — % of predictions where KNN + LSTM agreed
- Prediction accuracy: last 30 days — predicted vs actual, broken down by regime and by model (KNN, LSTM, ensemble)
- Equity curve of live trading

---

## Phase 10: In-App LLM Chatbot with Interactive Guidance

**Context-aware assistant that helps users navigate the app, explains features, and visually guides them to UI elements.**

### Step 10.1 — Multi-Provider LLM Backend

- **4 providers supported** via `llm_providers.py`:
  - **OpenAI**: gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-4o-mini, gpt-4o, o4-mini
  - **Anthropic**: claude-sonnet-4-20250514, claude-3-5-haiku-20241022
  - **Gemini**: gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-flash
  - **Ollama**: local models (dynamic discovery via `http://localhost:11434/api/tags`)
- **Provider abstraction**: registry dict mapping provider name → async generator handler. Each yields text chunks for streaming.
- OpenAI SDK reused for Ollama with custom `base_url`
- API keys stored server-side in `settings` table (keys: `llm_provider`, `llm_model`, `llm_api_key_openai`, `llm_api_key_anthropic`, `llm_api_key_gemini`, `llm_ollama_url`)
- **Files**: `backend/app/core/llm_providers.py`

### Step 10.2 — Context-Aware System Prompt & Guide Protocol

- **`chatbot_context.py`** builds system prompts with:
  - **APP_KNOWLEDGE**: ~2K token static knowledge covering all 9 pages, workflows, terminology, and troubleshooting
  - **GUIDE_ID_REGISTRY**: maps every `data-guide-id` to its page and human description (50+ elements)
  - **Action tag protocol**: LLM instructed to emit `[ACTION:navigate=/path,highlight=guide-id]` when guiding users
  - **Page context**: `build_system_prompt(current_page)` injects page-specific instructions so the LLM knows where the user currently is
- **Guide IDs** added to all 9 pages + Sidebar via `data-guide-id` HTML attributes on interactive elements:
  - Dashboard: stat-buy-signals, stat-holdings, equity-curve, predictions-list, recent-orders, etc.
  - DataManager: sync-stocks-btn, sync-holidays-btn, stock-search, interval-select, ohlcv-chart, etc.
  - RegimeAnalysis: regime-stock-select, classify-btn, regime-pie-chart, quality-chart, etc.
  - ModelStudio: algorithm-select, train-stock-select, start-training-btn, distill-btn, ensemble-config, etc.
  - Backtest: backtest-model-type, run-backtest-btn, backtest-equity-curve, etc.
  - LiveTrading: agreement-checkbox, run-predictions-btn, predictions-table, etc.
  - Portfolio: exit-all-btn, holdings-table, positions-table
  - Settings: zerodha-api-key, zerodha-api-secret, kite-login-btn, authenticate-btn, etc.
  - Sidebar: nav-dashboard, nav-data, nav-regime, nav-models, nav-backtest, nav-trading, nav-portfolio, nav-settings
- **Files**: `backend/app/core/chatbot_context.py`, all page `.tsx` files, `Sidebar.tsx`

### Step 10.3 — Chat API Endpoints (SSE Streaming)

- **POST `/api/chat`** — SSE streaming endpoint:
  - Request body: `{ messages, provider?, model?, page? }`
  - Resolves provider/model from request or falls back to DB settings
  - Fetches API key from `settings` table (keys never exposed to frontend)
  - Builds system prompt via `build_system_prompt(page)`
  - Streams response chunks as SSE `data:` events
  - Returns `text/event-stream` content type
- **GET `/api/chat/providers`** — returns available providers with model lists; queries Ollama dynamically for local models
- **GET `/api/chat/status`** — returns current configured provider, model, and whether an API key is set
- **Files**: `backend/app/api/routes/chat.py`

### Step 10.4 — Frontend Chat Widget

- **`ChatBot.tsx`** — floating chat widget:
  - Fixed bottom-right bubble (MessageCircle icon, z-40), expands to ~400×550px panel
  - Dark theme, message thread with streaming text display
  - **Config panel** (gear icon): provider dropdown, model input, API key, Ollama URL, save button
  - **SSE streaming**: uses `fetch()` + `ReadableStream` to parse chunked responses in real-time
  - **Action tag parser**: regex `\[ACTION:([^\]]+)\]` strips tags from displayed text, executes navigation + spotlight
  - **Page awareness**: sends current route path with each message so LLM knows user's location
  - localStorage persistence for panel open/close state
- **Files**: `frontend/src/components/ChatBot.tsx`, wired into `Layout.tsx`

### Step 10.5 — Spotlight / Interactive Guidance System

- **`spotlight.ts`** utility:
  - `spotlight(guideId, duration?)`: finds element by `[data-guide-id="${guideId}"]`, scrolls into view, creates overlay + pulsing ring highlight
  - Auto-dismisses after duration (default 3500ms), click overlay to dismiss early
- **CSS animations** in `index.css`:
  - `@keyframes spotlight-pulse`: ring scale/opacity animation
  - `.spotlight-overlay`: fixed inset-0, semi-transparent black backdrop (z-[9998])
  - `.spotlight-ring`: absolute positioned, border glow, z-[9999]
- **Interaction flow**: user asks "How do I update my Zerodha API key?" → LLM responds with explanation + `[ACTION:navigate=/settings,highlight=zerodha-api-key]` → ChatBot parses action → navigates to Settings → spotlights the API key input field
- **Files**: `frontend/src/utils/spotlight.ts`, `frontend/src/index.css`

### Step 10.6 — LLM Configuration in Settings

- **Settings.tsx** has a "Chat Assistant" configuration card:
  - Provider dropdown (OpenAI / Anthropic / Gemini / Ollama)
  - Model input (text field)
  - API key input (password masked)
  - Conditional Ollama URL field (shown only when Ollama selected)
  - Save + Test Connection buttons
  - Settings persisted to `settings` table via `PUT /api/config/batch`
- **Files**: `frontend/src/pages/Settings.tsx`

### Step 10.7 — Chat API Client Functions

- Added to `frontend/src/services/api.ts`:
  - `sendChatMessage(messages, page?, provider?, model?)` — SSE fetch returning `ReadableStreamDefaultReader`
  - `getChatProviders()` → lists providers and their models
  - `getChatStatus()` → current LLM configuration status

---

## Phase 9: DevOps & Testing

### Step 9.1 — Docker Compose

- MySQL 8, Redis, FastAPI, Celery worker + beat, React (nginx)
- Volumes: model files, DB data

### Step 9.2 — Testing

- `pytest trading_env/tests/` — env correctness, TradeLogger, NormalizerState
- `pytest backend/tests/test_regime_classifier.py` — regime detection accuracy:
  - Feed known bull market data (e.g., 2020-2021 rally) → verify classified as Bullish
  - Feed known crash (March 2020) → verify Bearish + HighVol
  - Feed sideways period → verify Neutral
  - Quality scores: circuit-hit candle should score < 0.5
- `pytest backend/tests/test_normalizer.py` — save/load round-trip
- `pytest backend/tests/test_db_cache.py` — DB-first fetch: mock Kite API, verify DB is queried first, API called only when stale
- `pytest backend/tests/test_pattern_extractor.py` — golden pattern labels match actual P&L
- `pytest backend/tests/test_knn_distiller.py` — time-series split, class balancing
- `pytest backend/tests/test_lstm_trainer.py` — LSTM training convergence:
  - Train for 10 epochs on small golden pattern set → verify loss decreases
  - Save/load round-trip: predictions identical after reload
  - Input shape correctness: (batch, seq_len, features) not flattened
  - Class weight computation: verify minority class gets higher weight
- `pytest backend/tests/test_ensemble.py` — ensemble logic:
  - Weight computation from backtest accuracies
  - Agreement filter: KNN=BUY + LSTM=BUY → BUY, KNN=BUY + LSTM=SELL → HOLD (when agreement=ON)
  - Weighted proba combination is mathematically correct
  - Fallback: ensemble with only KNN available degrades to KNN-only
- Integration: end-to-end pipeline on 1 stock — data sync → indicators → regime classify → RL train → extract → distill KNN + LSTM → ensemble config → backtest → predict → result in `ensemble_predictions`
- **Files**: `backend/tests/`, `trading_env/tests/`

---

## Relevant Files from ~/pytrade to Reuse

| pytrade file                                     | Reuse in aitrade                                                                                 | What to extract                                                                                                                                                                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils/pyTrade_common.py`                        | `backend/app/core/zerodha.py`                                                                    | `place_order()`, `place_GTT_order()`, `get_held_holdings()`, `get_held_positions()`, `exit_all_with_profit()`, `get_LTP()`, `get_close_price()`, `get_tag()`, `get_symbl()`                                                                           |
| `utils/pyTrader_login.py`                        | `backend/app/api/routes/auth.py`                                                                 | OAuth flow (make web-based redirect)                                                                                                                                                                                                                  |
| `utils/df_creator.py`                            | `backend/app/core/data_pipeline.py`, `backend/app/core/normalizer.py`, `backend/app/db/cache.py` | `fetch_history()`, `fetch_chunk()`, `get_dfs()` (DB-first logic, indicator pipeline), `get_combination()`, log-return normalization, `download=False` DB query pattern                                                                                |
| `utils/db_common.py`                             | `backend/app/db/crud.py`, `backend/app/db/cache.py`                                              | `set_bulk_db_stocks()` (5000 row batch), `set_bulk_db_stock_indicator()` (10000 row batch), `get_filtered_buy_stocks()` JOIN pattern, `get_start_date_for_update()` incremental sync, `ON DUPLICATE KEY UPDATE`, `get_last_working_day()` stale check |
| `utils/common.py`                                | `backend/app/core/indicators.py`                                                                 | `calc_vw_kama()`, `is_downtrend()`, `get_sma_per_diff()`                                                                                                                                                                                              |
| `utils/addLabel.py`                              | Reference only                                                                                   | `sequential()` — NOT used for labeling, kept as baseline accuracy comparison                                                                                                                                                                          |
| `utils/constants.py`                             | `backend/app/config.py`                                                                          | `get_seq_len()`→15, `get_stoploss_per()`→5% (CNC), `get_buy_limit()`→10000, `get_str_combination()`→'TGRBmacdadx', `get_stocks()`→NSE stock list                                                                                                      |
| `DB_updater/indicator_db_updater.py`             | `backend/app/workers/tasks.py`                                                                   | Indicator sync pattern: compute + bulk upsert to DB                                                                                                                                                                                                   |
| `DB_updater/holidays_db_updater.py`              | `backend/app/core/data_pipeline.py`                                                              | NSE holiday fetch from API → `nse_holidays` table, `get_last_working_day()`                                                                                                                                                                           |
| `DB_updater/web_db_sync.py`                      | `backend/app/workers/tasks.py`                                                                   | Full DB sync pattern: `update_db_stocks()` + `update_db_stock_indicator()`                                                                                                                                                                            |
| `gym-trader-yahoo/gym_trader/envs/trader_env.py` | `trading_env/envs/base_env.py`                                                                   | `Trader` class step/reset, observation — add TradeLogger + regime features                                                                                                                                                                            |
| `knn.py` lines 68-87                             | `backend/app/ml/knn_distiller.py`                                                                | Sliding window construction                                                                                                                                                                                                                           |
| `knn_predict.py`                                 | `backend/app/ml/predictor.py`                                                                    | Production prediction flow                                                                                                                                                                                                                            |
| `knn_backtest.py`                                | `backend/app/api/routes/backtest.py`                                                             | Backtest SL/PF simulation                                                                                                                                                                                                                             |
| `Model_related/KNN_HMM_Model_creator.py`         | `backend/app/core/regime_classifier.py`                                                          | HMM regime concept (we upgrade to multi-signal classifier)                                                                                                                                                                                            |

---

## Key Technology Choices

| Component        | Technology                                                    | Rationale                                                                      |
| ---------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Backend          | FastAPI                                                       | Async, WebSocket, auto OpenAPI docs                                            |
| Frontend         | React + Vite + TypeScript + TailwindCSS                       | Rich charting, real-time updates                                               |
| RL Framework     | Stable-Baselines3 + sb3-contrib                               | PPO/A2C/DDPG/SAC/TD3 (MLP) + RecurrentPPO (LSTM policy)                        |
| KNN              | scikit-learn KNeighborsClassifier                             | Fast inference, interpretable via kneighbors()                                 |
| LSTM             | PyTorch nn.LSTM                                               | Preserves temporal sequence structure that KNN loses; SB3 already uses PyTorch |
| Ensemble         | Custom weighted average                                       | Reduces variance — KNN catches local patterns, LSTM catches temporal trends    |
| Regime Detection | Custom multi-signal (SMA cross + ADX + RSI + MACD + BB + ATR) | More robust than single-signal HMM; deterministic, debuggable                  |
| Trading Env      | Gymnasium (custom lib)                                        | Standard RL interface                                                          |
| Database         | MySQL 8                                                       | Matches pytrade, full DB caching for fast queries                              |
| Task Queue       | Celery + Redis                                                | Async training, scheduled sync/predictions                                     |
| Charts           | Lightweight Charts + Recharts                                 | Candlestick, equity curves, regime overlays                                    |
| Class Balancing  | imbalanced-learn (SMOTE)                                      | Handle 80/15/5 class distribution                                              |
| LLM Chatbot      | OpenAI / Anthropic / Gemini / Ollama SDKs + SSE streaming     | Multi-provider flexibility, API keys server-side, real-time streaming          |
| Chat UI          | Custom floating widget + spotlight overlay                    | Context-aware guidance with visual element highlighting                        |

---

## Data Flow: DB Caching Strategy

```
                    Zerodha Kite API
                          │
                          ↓ (fetch only when DB is stale/missing)
                    ┌─────────────┐
                    │ stock_ohlcv │ ← PK: (stock_id, date, interval)
                    └──────┬──────┘
                           │ (auto-compute after sync)
                    ┌──────┴──────────┐
                    │ stock_indicators │ ← PK: (stock_id, date, interval)
                    └──────┬──────────┘
                           │ (auto-classify after indicators)
                    ┌──────┴──────────┐
                    │ stock_regimes   │ ← PK: (stock_id, date, interval)
                    └──────┬──────────┘
                           │ (used for training + live prediction)
                     ┌─────┴────────┐
                     │  RL Training  │ → rl_models, rl_training_runs
                     └─────┬────────┘
                           │ (extract from RL)
                     ┌─────┴────────────┐
                     │ golden_patterns   │
                     └─────┬────────────┘
                           │ (distill both models)
                     ┌─────┴──────┐   ┌─────────────┐
                     │ knn_models  │   │ lstm_models  │
                     └─────┬──────┘   └──────┬──────┘
                           │                  │
                     ┌─────┴──────────────────┴─────┐
                     │      ensemble_configs         │ ← weights from backtest
                     └─────────────┬─────────────────┘
                                   │ (daily/weekly predictions)
            ┌──────────────────────┼──────────────────────┐
     ┌──────┴───────────┐  ┌──────┴───────────┐  ┌───────┴────────────┐
     │ knn_predictions   │  │ lstm_predictions  │  │ ensemble_predictions│
     └──────────────────┘  └──────────────────┘  └───────┬────────────┘
                                                          │ Trading decisions
                                                          │ hit THIS table
                                                   ┌──────┴──────────┐
                                                   │ trade_orders     │
                                                   └─────────────────┘

    ALL reads for training, prediction, and trading go to local MySQL first.
    Kite API is ONLY called for: fresh OHLCV sync + order execution.
```

---

## Decisions

- **Market regime classification** as pre-training layer: classify data quality + regime BEFORE RL training — models learn from clean, regime-tagged data only
- **6 regimes** (3 trend × 2 volatility) replaces pytrade's 3-state HMM — more granular, deterministic, debuggable
- **No intraday**: daily + weekly intervals only, CNC orders only
- **NSE only**: filter all data/instruments to exchange='NSE'
- **DB-first caching**: all OHLCV, indicators, regimes, predictions in MySQL — Kite API called only when stale. Reuses pytrade's proven `ON DUPLICATE KEY UPDATE` bulk pattern
- **Predictions in DB**: `ensemble_predictions` table queried for trading — no re-computation needed
- **RL as label generator** replaces pytrade's biased `sequential()` labels
- **KNN + LSTM ensemble for production**: KNN gives interpretability (matched patterns), LSTM gives temporal awareness (sequence modeling). Weighted ensemble combines strengths, agreement filter adds safety
- **LSTM architecture**: 2-layer, 128 hidden, dropout 0.3, PyTorch — shares PyTorch with SB3, processes sequences without flattening
- **Ensemble weights from backtest**: auto-computed proportional to model accuracy on test data. User can override.
- **Agreement filter default ON**: only trade when KNN AND LSTM agree — conservative, reduces false signals
- **RecurrentPPO (LSTM policy) via sb3-contrib**: PPO can use an LSTM policy that maintains hidden state across episode steps, seeing one candle at a time. Off-policy algorithms (DDPG/TD3/SAC) cannot use LSTM policies due to replay buffer incompatibility — they use frame-stacking instead. This gives us LSTM at TWO levels: (1) RL agent with LSTM policy for pattern discovery, (2) standalone LSTM for distillation.
- **Quality scoring** (0-1) filters out circuit hits, low-volume days, gaps, transitions
- **k=5 over k=2**: pytrade's k=2 overfits
- **SB3 over Ray RLlib**: simpler, supports DDPG natively
- **Weekly regime as macro context**: daily models see both daily and weekly regime features
- **LLM chatbot with interactive guidance**: multi-provider (OpenAI/Anthropic/Gemini/Ollama) chatbot that understands the full app, navigates users to pages, and visually highlights UI elements via spotlight overlay
- **SSE streaming over WebSocket for chat**: simpler protocol, works through proxies, one-directional stream sufficient for LLM responses
- **API keys stored server-side**: LLM API keys kept in `settings` DB table, never sent to frontend — backend proxies all LLM requests
- **Action tag protocol** (`[ACTION:navigate=/path,highlight=guide-id]`): allows LLM to trigger UI actions from within its text response — parsed client-side, stripped from display
- **`data-guide-id` attributes** on 50+ interactive elements: enables chatbot to reference and spotlight any UI control by stable identifier, decoupled from CSS classes or DOM structure

---

## Verification Checklist

1. **DB Caching**: mock Kite API → first call stores in DB → second call reads from DB (no API hit) → after advancing date, stale check triggers API
2. **Regime Classifier**:
   - Feed NIFTY 2020 March crash → verify Bearish+HighVol (regime_id=5)
   - Feed NIFTY 2020-2021 bull run → verify Bullish+LowVol (regime_id=0)
   - Feed choppy sideways period → verify Neutral
   - Circuit-hit candle → quality_score < 0.5
3. **Data Quality**: verify "quality ≥ 0.8" filter removes circuit hits, low-volume days, transitions
4. **RL Training**: train PPO on quality-filtered RELIANCE daily data → reward improves, model saves
5. **Pattern Extraction**: golden_patterns BUY labels correspond to actual profits, regime_id matches stock_regimes
6. **KNN Distillation**: accuracy > 55%, precision_buy > 50%
7. **LSTM Training**: loss decreases over 10+ epochs, save/load round-trip produces identical predictions, input shape is (batch, seq_len, features) not flattened
8. **Ensemble**: weighted proba = w*knn * knn + w*lstm * lstm, agreement filter correctly outputs HOLD on disagreement, fallback to single model when only one available
9. **Prediction DB Cache**: run daily prediction → verify rows in `knn_predictions`, `lstm_predictions`, AND `ensemble_predictions` → ensemble query returns results in <50ms
10. **End-to-End**: sync RELIANCE → indicators → regimes → RL train → extract → distill KNN + LSTM → ensemble → backtest → predict → result in `ensemble_predictions`
11. **Frontend**: Model Studio shows KNN + LSTM + Ensemble tabs, Live Trading shows agreement badges, Pattern Lab shows both KNN neighbors and LSTM saliency
12. `docker-compose up` → all healthy, frontend at :3000
