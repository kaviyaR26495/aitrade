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
export const zerodhaCallback = (requestToken: string) =>
  api.get('/auth/callback', { params: { request_token: requestToken } });

// ── Config ──
export const getConfig = () => api.get('/config/');
export const getSetting = (key: string) => api.get(`/config/${key}`);
export const updateSetting = (key: string, value: string) =>
  api.put('/config/', { key, value });
export const updateConfigBatch = (settings: { key: string; value: string }[]) =>
  api.put('/config/batch', settings);

// ── Data ──
export const listStocks = () => api.get('/data/stocks');
export const getOhlcv = (stockId: number, interval?: string, startDate?: string, endDate?: string) =>
  api.get(`/data/stocks/${stockId}/ohlcv`, { params: { interval, start_date: startDate, end_date: endDate } });
export const getIndicators = (stockId: number, interval?: string) =>
  api.get(`/data/stocks/${stockId}/indicators`, { params: { interval } });
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
export const listAllModels = () => api.get('/models/all');

// ── Backtest ──
export const runBacktest = (params: Record<string, unknown>) =>
  api.post('/backtest/run', params);
export const getBacktestResults = (id: number) => api.get(`/backtest/results/${id}`);
export const listBacktestResults = () => api.get('/backtest/results');

// ── Trading ──
export const getPredictions = (params?: Record<string, unknown>) =>
  api.get('/trading/predictions', { params });
export const runPredictions = (params: Record<string, unknown>) =>
  api.post('/trading/run-predictions', params);
export const placeOrder = (params: Record<string, unknown>) =>
  api.post('/trading/order', params);
export const listOrders = (limit?: number) =>
  api.get('/trading/orders', { params: { limit } });

// ── Portfolio ──
export const getHoldings = () => api.get('/portfolio/holdings');
export const getPositions = () => api.get('/portfolio/positions');
export const getLtp = (symbol: string) => api.get(`/portfolio/ltp/${symbol}`);
export const exitAll = () => api.post('/portfolio/exit-all');

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
}

export interface PipelineStageStatus {
  /** 0-indexed stage number */
  stage: number;
  /** camelCase name for the stage */
  name: string;
  /** 'pending' | 'running' | 'completed' | 'failed' */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** 0–100 progress within the running stage */
  progress: number;
  message?: string;
}

export interface PipelineStatus {
  job_id: string;
  symbols: string[];
  /** Overall pipeline status: 'queued' | 'running' | 'completed' | 'failed' */
  status: 'queued' | 'running' | 'completed' | 'failed';
  /** 0-indexed index of the currently active stage */
  current_stage: number;
  stages: PipelineStageStatus[];
  created_at: string;
  updated_at: string;
  error?: string;
}

export const startPipeline = (req: PipelineRequest) =>
  api.post<{ job_id: string }>('/pipeline/start', req);

export const getPipelineStatus = (jobId: string) =>
  api.get<PipelineStatus>(`/pipeline/status/${jobId}`);
