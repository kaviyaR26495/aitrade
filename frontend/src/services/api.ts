import axios from 'axios';
import { useAppStore } from '../store/appStore';

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

// ── API console interceptors ──────────────────────────────────────────
api.interceptors.request.use((config) => {
  const id = crypto.randomUUID();
  (config as any).__logId = id;
  (config as any).__logStart = Date.now();
  useAppStore.getState().addApiLog({
    id,
    timestamp: Date.now(),
    method: (config.method ?? 'GET').toUpperCase(),
    url: config.url ?? '',
    status: null,
    duration: null,
    requestBody: config.data ? JSON.stringify(config.data).slice(0, 500) : undefined,
  });
  return config;
});

api.interceptors.response.use(
  (response) => {
    const id = (response.config as any).__logId;
    const start = (response.config as any).__logStart;
    if (id) {
      useAppStore.getState().updateApiLog(id, {
        status: response.status,
        duration: start ? Date.now() - start : null,
        responseBody: JSON.stringify(response.data).slice(0, 800),
      });
    }
    return response;
  },
  (error) => {
    const id = error.config?.__logId;
    const start = error.config?.__logStart;
    if (id) {
      useAppStore.getState().updateApiLog(id, {
        status: error.response?.status ?? -1,
        duration: start ? Date.now() - start : null,
        responseBody: error.response
          ? JSON.stringify(error.response.data).slice(0, 800)
          : error.message,
      });
    }
    return Promise.reject(error);
  }
);

// ── Auth ──
export const getLoginUrl = () => api.get('/auth/login-url');
export const getAuthStatus = () => api.get('/auth/status');
export const zerodhaCallback = (requestToken: string, state?: string) =>
  api.get('/auth/callback', { params: { request_token: requestToken, state } });
export const triggerMorningTrade = () => api.post('/auth/trigger-morning-trade');
export const saveAutoLoginConfig = (config: {
  zerodha_user_id: string;
  zerodha_password: string;
  zerodha_totp_secret: string;
  enabled: boolean;
}) => api.post('/auth/save-auto-login', config);
export const getAutoLoginStatus = () => api.get('/auth/auto-login-status');
export const getSharedUsers = () => api.get<any[]>('/auth/users');
export const saveSharedUsers = (users: any[]) => api.post('/auth/users', users);
export const getSharedRoles = () => api.get<any[]>('/auth/roles');
export const saveSharedRoles = (roles: any[]) => api.post('/auth/roles', roles);

// ── Config ──
export const getConfig = () => api.get('/config/');
export const getHealth = () => api.get('/health');
export const getSetting = (key: string) => api.get(`/config/${key}`);
export const updateSetting = (key: string, value: string) =>
  api.put('/config/', { key, value });
export const updateConfigBatch = (settings: { key: string; value: string }[]) =>
  api.put('/config/batch', settings);

// ── Data ──
export const listStocks = () => api.get('/data/stocks');
export const getOhlcv = (stockId: number, interval?: string, startDate?: string, endDate?: string) =>
  api.get(`/data/stocks/${stockId}/ohlcv`, { params: { interval, start_date: startDate, end_date: endDate } });
export const getIndicators = (stockId: number, interval?: string, startDate?: string, endDate?: string) =>
  api.get(`/data/stocks/${stockId}/indicators`, { params: { interval, start_date: startDate, end_date: endDate } });
export const syncData = (stockIds: number[], interval?: string) =>
  api.post('/data/sync', { stock_ids: stockIds, interval });
export const syncStockList = () => api.post('/data/sync/stocks');
export const syncHolidays = () => api.post('/data/sync/holidays');
export const getFeatures = (stockId: number, interval?: string, normalize?: boolean) =>
  api.get(`/data/stocks/${stockId}/features`, { params: { interval, normalize } });

// ── Stock Universe ──
export const getUniverse = () => api.get('/data/universe');
export const setUniverse = (category: string, customSymbols: string[]) =>
  api.put('/data/universe', { category, custom_symbols: customSymbols });
export const listUniverseStocks = () => api.get('/data/stocks/universe');
export const getUniversePresetSymbols = (category: string) =>
  api.get<{ category: string; symbols: string[]; resolved_count: number }>(`/data/universe/presets/${category}`);

// ── Zerodha Instruments ──
export const getZerodhaInstruments = (exchange = 'NSE') =>
  api.get('/data/instruments', { params: { exchange } });
export const clearInstrumentCache = () =>
  api.delete('/data/instruments/cache');

// ── Regime ──
export const getRegime = (stockId: number, interval?: string) =>
  api.get(`/regime/${stockId}`, { params: { interval } });
export const getRegimeSummary = (stockId: number, interval?: string) =>
  api.get(`/regime/${stockId}/summary`, { params: { interval } });
export const classifyRegime = (stockId: number, interval?: string) =>
  api.post(`/regime/classify`, { stock_ids: [stockId], interval: interval || 'day' });

// ── Models ──
export const getAlgorithms = () => api.get('/models/algorithms');
export const getDeviceInfo = () => api.get('/models/device');
export const listRlModels = () => api.get('/models/rl');
export const getRlModel = (id: number) => api.get(`/models/rl/${id}`);
export const getRlModelLogs = (id: number) => api.get(`/models/rl/${id}/logs`);
export const stopTraining = (id: number) => api.post(`/models/rl/${id}/stop`, {});
export const pauseTraining = (id: number) => api.post(`/models/rl/${id}/pause`, {});
export const resumeTraining = (id: number) => api.post(`/models/rl/${id}/resume`, {});
export const deleteRlModel = (id: number) => api.delete(`/models/rl/${id}`);
export const getTrainingLogFile = (id: number) => api.get(`/models/rl/${id}/log-file`);
export const deleteTrainingLogFile = (id: number) => api.delete(`/models/rl/${id}/log-file`);
export const listTrainingLogFiles = () => api.get('/models/training-logs');
export const getTrainingLogsTotalSize = () => api.get('/models/training-logs/total-size');
export const deleteAllTrainingLogs = () => api.delete('/models/training-logs');
export const trainModel = (params: Record<string, unknown>) =>
  api.post('/models/train', params);
export const distillModel = (params: Record<string, unknown>) =>
  api.post('/models/distill', params);
export const getDistillLog = (knnModelId: number) => api.get(`/models/knn/${knnModelId}/log`);
export const listKnnModels = () => api.get('/models/knn');
export const deleteKnnModel = (id: number) => api.delete(`/models/knn/${id}`);
export const listLstmModels = () => api.get('/models/lstm');
export const deleteLstmModel = (id: number) => api.delete(`/models/lstm/${id}`);
export const listEnsembleConfigs = () => api.get('/models/ensemble');
export const updateEnsembleConfig = (id: number, data: { knn_weight: number; lstm_weight: number; agreement_required: boolean }) =>
  api.put(`/models/ensemble/${id}`, data);
export const deleteEnsembleConfig = (id: number) => api.delete(`/models/ensemble/${id}`);
export const getMetaClassifier = () => api.get('/models/meta-classifier');
export const exportModelBundle = (knnModelId: number, lstmModelId: number) =>
  api.get('/models/export', { params: { knn_model_id: knnModelId, lstm_model_id: lstmModelId }, responseType: 'blob' });
export const importModelBundle = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post<{ knn_model_id: number; lstm_model_id: number; ensemble_config_id: number; stub_rl_model_id: number }>(
    '/models/import', form
  );
};
export const listAllModels = () => api.get('/models/all');

// ── Backtest ──
export const runBacktest = (params: Record<string, unknown>) =>
  api.post('/backtest/run', params);
export const runCompoundBacktest = (params: Record<string, unknown>) =>
  api.post('/backtest/run-compound', params);
export const getBacktestResults = (id: number) => api.get(`/backtest/results/${id}`);
export const listBacktestResults = () => api.get('/backtest/results');
export const getCompoundBacktest = (id: number) => api.get(`/backtest/compound-results/${id}`);
export const listCompoundBacktests = () => api.get('/backtest/compound-results');
export const deleteCompoundBacktest = (id: number) => api.delete(`/backtest/compound-results/${id}`);
export const getTradePatterns = (backtestId: number, tradeIdx: number) =>
  api.get(`/backtest/${backtestId}/trades/${tradeIdx}/patterns`);
export const deleteBacktest = (id: number) => api.delete(`/backtest/${id}`);
export const deleteBacktestBatch = (ids: number[]) => api.delete('/backtest/batch/delete', { data: { ids } });

// ── Backfill ──
export const startBackfill = (params: {
  start_date: string;
  end_date: string;
  ensemble_config_id?: number | null;
  override_existing: boolean;
}) => api.post('/backfill/start', params);
export const getBackfillStatus = () => api.get('/backfill/status');
export const stopBackfill = () => api.post('/backfill/stop', {});
export const getBackfillCoverage = () => api.get('/backfill/coverage');

// ── Trading ──
export const getPredictionJob = (jobId: string) =>
  api.get(`/trading/predictions/jobs/${jobId}`);
export const cancelPredictionJob = (jobId: string) =>
  api.delete(`/trading/predictions/jobs/${jobId}`);
export const listOrders = (limit?: number) =>
  api.get('/trading/orders', { params: { limit } });

// ── TPML Signals ──
export const getSignals = (params?: { target_date?: string; date_from?: string; date_to?: string; status?: string; min_pop?: number }) =>
  api.get('/trading/signals', { params });
export const generateSignals = (params: { interval?: string; stock_ids?: number[]; target_date?: string; pop_threshold?: number }) =>
  api.post('/trading/signals/generate', params);
export const trainMhLstm = () =>
  api.post('/trading/signals/train-model', {});
export const getMhLstmStatus = () =>
  api.get<{ status: string; model_id?: number; accuracy?: number; name?: string; created_at?: string }>('/models/mh-lstm/status');
export const getTaskStatus = (taskId: string) =>
  api.get<{ task_id: string; state: string; result?: Record<string, unknown>; error?: string }>(`/trading/signals/task/${taskId}`);
export const getSignalPreview = (signalId: number) =>
  api.get<{ signal_id: number; quantity: number; entry_price: number; position_value: number }>(`/trading/signals/${signalId}/preview`);
export const executeSignal = (signalId: number, dryRun = false) =>
  api.post(`/trading/signals/${signalId}/execute`, null, { params: { dry_run: dryRun } });
export const deleteSignal = (signalId: number) =>
  api.delete(`/trading/signals/${signalId}`);
export const deleteSignalsByDate = (targetDate: string) =>
  api.delete(`/trading/signals/date/${targetDate}`);

// ── Portfolio ──
export const getHoldings = () => api.get('/portfolio/holdings');
export const getPositions = () => api.get('/portfolio/positions');
export const reconcilePortfolio = () => api.post('/portfolio/reconcile');
export const getPortfolioSnapshot = () => api.get('/portfolio/snapshot');

// ── Chat ──
export const getChatProviders = () => api.get('/chat/providers');
export const getChatStatus = () => api.get('/chat/status');

/** SSE streaming chat — returns a ReadableStream reader */
export async function sendChatMessage(
  messages: { role: string; content: string }[],
  page?: string,
  provider?: string,
  model?: string,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, page, provider, model }),
  });
  if (!res.body) throw new Error('No response body');
  return res.body.getReader();
}

export default api;
export const getGoldenPatterns = (rl_model_id?: number) =>
  api.get('/models/patterns', { params: { rl_model_id } });

// ── Pipeline ──────────────────────────────────────────────────────────────────

export interface PipelineRequest {
  symbols: string[];
  skip_sync?: boolean;
  force_sync?: boolean;
  use_regime_pooling?: boolean;
  resume_job_id?: string;
}

export interface PipelineStageStatus {
  /** 0-indexed stage number */
  stage: number;
  /** camelCase name for the stage */
  name: string;
  /** 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' */
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  /** 0–100 progress within the running stage */
  progress: number;
  message?: string;
}

export interface PipelineStatus {
  job_id: string;
  symbols: string[];
  /** Overall pipeline status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' */
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'purged';
  /** 0-indexed index of the currently active stage */
  current_stage: number;
  stages: PipelineStageStatus[];
  created_at: string;
  updated_at: string;
  error?: string;
}

export interface PipelineTerminateResult {
  job_id: string;
  status: 'terminating' | 'purged';
  records_deleted?: Record<string, number>;
  files_deleted?: string[];
}

export const startPipeline = (req: PipelineRequest) =>
  api.post<{ job_id: string }>('/pipeline/start', req);

export const getPipelineStatus = (jobId: string) =>
  api.get<PipelineStatus>(`/pipeline/status/${jobId}`);
export const getLatestPipelineJob = () =>
  api.get<PipelineStatus>('/pipeline/latest');

export const terminatePipeline = (jobId: string, purge = false) =>
  api.delete<PipelineTerminateResult>(`/pipeline/${jobId}`, { params: { purge } });

// ── Training (CT Pipeline) ────────────────────────────────────────────────────

export interface RetrainStatus {
  last_retrain_at: string | null;
  days_since_retrain: number | null;
  needs_retrain: boolean;
  has_models: boolean;
}

export const getRetrainStatus = () =>
  api.get<RetrainStatus>('/training/retrain-status');

export const triggerAutoRetrain = (lookbackYears = 2) =>
  api.post<{ message: string; lookback_years: number }>(
    '/training/auto-retrain',
    { lookback_years: lookbackYears },
  );
