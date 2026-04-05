import { useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as api from '../services/api';
import type { PipelineRequest, PipelineStatus } from '../services/api';

const TERMINAL_RL_STATUSES = new Set(['completed', 'failed', 'stopped']);

function syncRlModelStatus(qc: ReturnType<typeof useQueryClient>, modelId: number | null, status?: string | null) {
  if (!modelId || !status) return;
  qc.setQueryData(['rl-models'], (prev: any[] | undefined) => {
    if (!Array.isArray(prev)) return prev;
    return prev.map((model) => {
      if (model.id !== modelId) return model;
      if (model.status === status) return model;
      return { ...model, status };
    });
  });
}

// ── Auth hooks ──
export const useAuthStatus = () =>
  useQuery({
    queryKey: ['auth-status'],
    queryFn: () => api.getAuthStatus().then(r => r.data),
    staleTime: 5 * 60 * 1000, // re-check every 5 minutes
    retry: false,
  });

// ── Data hooks ──
export const useStocks = () =>
  useQuery({ queryKey: ['stocks'], queryFn: () => api.listStocks().then(r => r.data) });

export const useUniverseStocks = () =>
  useQuery({ queryKey: ['universe-stocks'], queryFn: () => api.listUniverseStocks().then(r => r.data) });

export const useUniverse = () =>
  useQuery({ queryKey: ['universe'], queryFn: () => api.getUniverse().then(r => r.data) });

export const useSetUniverse = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { category: string; customSymbols: string[] }) =>
      api.setUniverse(params.category, params.customSymbols),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['universe'] });
      qc.invalidateQueries({ queryKey: ['universe-stocks'] });
    },
  });
};

export const useOhlcv = (stockId: number, interval = 'day') =>
  useQuery({
    queryKey: ['ohlcv', stockId, interval],
    queryFn: () => api.getOhlcv(stockId, interval).then(r => r.data),
    enabled: !!stockId,
  });

export const useIndicators = (stockId: number, interval = 'day') =>
  useQuery({
    queryKey: ['indicators', stockId, interval],
    queryFn: () => api.getIndicators(stockId, interval).then(r => r.data),
    enabled: !!stockId,
  });

export const useSyncData = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { stockIds: number[]; interval?: string }) =>
      api.syncData(params.stockIds, params.interval),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ohlcv'] }),
  });
};

// ── Regime hooks ──
export const useRegime = (stockId: number, interval = 'day') =>
  useQuery({
    queryKey: ['regime', stockId, interval],
    queryFn: () => api.getRegime(stockId, interval).then(r => r.data),
    enabled: !!stockId,
  });

export const useRegimeSummary = (stockId: number, interval = 'day') =>
  useQuery({
    queryKey: ['regime-summary', stockId, interval],
    queryFn: () => api.getRegimeSummary(stockId, interval).then(r => r.data),
    enabled: !!stockId,
  });

// ── Model hooks ──
export const useAlgorithms = () =>
  useQuery({ queryKey: ['algorithms'], queryFn: () => api.getAlgorithms().then(r => r.data) });

export const useDeviceInfo = () =>
  useQuery({ queryKey: ['device-info'], queryFn: () => api.getDeviceInfo().then(r => r.data), staleTime: 60_000 });

export const useRlModels = () =>
  useQuery({
    queryKey: ['rl-models'],
    queryFn: () => api.listRlModels().then(r => r.data),
    refetchInterval: (query) => {
      const models: any[] = query.state.data ?? [];
      const hasActive = models.some(
        (m) => m.status === 'pending' || m.status === 'training' || m.status === 'paused',
      );
      return hasActive ? 3000 : false;
    },
  });

export const useRlModelLogs = (modelId: number | null, isActive = false) => {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['rl-model-logs', modelId],
    queryFn: () => api.getRlModelLogs(modelId!).then(r => r.data),
    enabled: !!modelId,
    refetchInterval: (q) => {
      if (!isActive) return false;
      if (q.state.data?.is_active === false) return false;
      if (TERMINAL_RL_STATUSES.has(q.state.data?.status)) return false;
      return 3000;
    },
  });
  useEffect(() => {
    const status = query.data?.status;
    if (query.data?.is_active === false && TERMINAL_RL_STATUSES.has(status)) {
      syncRlModelStatus(qc, modelId, status);
      qc.invalidateQueries({ queryKey: ['rl-models'] });
    }
  }, [modelId, query.data?.is_active, query.data?.status, qc]);
  return query;
};

export const useKnnModels = () =>
  useQuery({
    queryKey: ['knn-models'],
    queryFn: () => api.listKnnModels().then(r => r.data),
    refetchInterval: (q) => {
      const models: any[] = q.state.data ?? [];
      return models.some(m => m.status === 'training' || m.status === 'pending') ? 3000 : false;
    },
  });

export const useLstmModels = () =>
  useQuery({
    queryKey: ['lstm-models'],
    queryFn: () => api.listLstmModels().then(r => r.data),
    refetchInterval: (q) => {
      const models: any[] = q.state.data ?? [];
      return models.some(m => m.status === 'training' || m.status === 'pending') ? 3000 : false;
    },
  });

export const useTrainModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Record<string, unknown>) => api.trainModel(params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rl-models'] }),
  });
};

export const useStopTraining = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.stopTraining(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rl-models'] }),
  });
};

export const usePauseTraining = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.pauseTraining(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rl-models'] }),
  });
};

export const useResumeTraining = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.resumeTraining(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rl-models'] }),
  });
};

export const useDeleteRlModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteRlModel(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rl-models'] }),
  });
};

export const useTrainingLogFiles = () =>
  useQuery({ queryKey: ['training-log-files'], queryFn: () => api.listTrainingLogFiles().then(r => r.data), staleTime: 5000 });

export const useTrainingLogFile = (modelId: number | null, isActive: boolean) => {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['training-log-file', modelId],
    queryFn: () => api.getTrainingLogFile(modelId!).then(r => r.data),
    enabled: !!modelId,
    refetchInterval: (q) => {
      if (!isActive) return false;
      if (q.state.data?.is_active === false) return false;
      if (TERMINAL_RL_STATUSES.has(q.state.data?.status)) return false;
      return 2000;
    },
  });
  useEffect(() => {
    const status = query.data?.status;
    if (query.data?.is_active === false && TERMINAL_RL_STATUSES.has(status)) {
      syncRlModelStatus(qc, modelId, status);
      qc.invalidateQueries({ queryKey: ['rl-models'] });
    }
  }, [modelId, query.data?.is_active, query.data?.status, qc]);
  return query;
};

export const useTrainingLogsTotalSize = () =>
  useQuery({ queryKey: ['training-logs-size'], queryFn: () => api.getTrainingLogsTotalSize().then(r => r.data), staleTime: 5000 });

export const useDeleteTrainingLogFile = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteTrainingLogFile(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training-log-files'] });
      qc.invalidateQueries({ queryKey: ['training-logs-size'] });
      qc.removeQueries({ queryKey: ['training-log-file'] });
    },
  });
};

export const useDeleteAllTrainingLogs = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.deleteAllTrainingLogs(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training-log-files'] });
      qc.invalidateQueries({ queryKey: ['training-logs-size'] });
      qc.removeQueries({ queryKey: ['training-log-file'] });
    },
  });
};

export const useDistillModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Record<string, unknown>) => api.distillModel(params),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['knn-models'] });
      qc.invalidateQueries({ queryKey: ['lstm-models'] });
      qc.invalidateQueries({ queryKey: ['ensemble-configs'] });
    },
  });
};

const TERMINAL_DISTILL_STATUSES = new Set(['completed', 'failed']);

export const useDistillLog = (knnModelId: number | null, isActive = false) => {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['distill-log', knnModelId],
    queryFn: () => api.getDistillLog(knnModelId!).then(r => r.data),
    enabled: !!knnModelId,
    refetchInterval: (q) => {
      if (!isActive) return false;
      if (q.state.data?.is_active === false) return false;
      return 3000;
    },
  });
  useEffect(() => {
    if (query.data?.is_active === false) {
      qc.invalidateQueries({ queryKey: ['knn-models'] });
      qc.invalidateQueries({ queryKey: ['lstm-models'] });
      qc.invalidateQueries({ queryKey: ['ensemble-configs'] });
    }
  }, [query.data?.is_active, qc]);
  return query;
};

export const useDeleteKnnModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteKnnModel(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['knn-models'] });
      qc.invalidateQueries({ queryKey: ['ensemble-configs'] });
    },
  });
};

export const useDeleteLstmModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteLstmModel(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['lstm-models'] });
      qc.invalidateQueries({ queryKey: ['ensemble-configs'] });
    },
  });
};

export const useEnsembleConfigs = () =>
  useQuery({
    queryKey: ['ensemble-configs'],
    queryFn: () => api.listEnsembleConfigs().then(r => r.data),
  });

export const useUpdateEnsemble = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: { knn_weight: number; lstm_weight: number; agreement_required: boolean } }) =>
      api.updateEnsembleConfig(id, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ensemble-configs'] }),
  });
};

export const useDeleteEnsemble = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteEnsembleConfig(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ensemble-configs'] }),
  });
};

// ── Backtest hooks ──
export const useBacktestResults = () =>
  useQuery({ queryKey: ['backtests'], queryFn: () => api.listBacktestResults().then(r => r.data) });

export const useRunBacktest = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Record<string, unknown>) => api.runBacktest(params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['backtests'] }),
  });
};

// ── Trading hooks ──
export const usePredictions = (params?: Record<string, unknown>) =>
  useQuery({
    queryKey: ['predictions', params],
    queryFn: () => api.getPredictions(params).then(r => r.data),
  });

export const useRunPredictions = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Record<string, unknown>) => api.runPredictions(params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['predictions'] }),
  });
};

export const usePlaceOrder = () =>
  useMutation({ mutationFn: (params: Record<string, unknown>) => api.placeOrder(params) });

export const useOrders = (limit = 50) =>
  useQuery({ queryKey: ['orders', limit], queryFn: () => api.listOrders(limit).then(r => r.data) });

// ── Portfolio hooks ──
export const useHoldings = () =>
  useQuery({ queryKey: ['holdings'], queryFn: () => api.getHoldings().then(r => r.data) });

export const usePositions = () =>
  useQuery({ queryKey: ['positions'], queryFn: () => api.getPositions().then(r => r.data) });

// ── Config hooks ──
export const useConfig = () =>
  useQuery({ queryKey: ['config'], queryFn: () => api.getConfig().then(r => r.data) });

export const useUpdateSetting = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { key: string; value: string }) =>
      api.updateSetting(params.key, params.value),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['config'] }),
  });
};

// ── Zerodha Instruments hook ──
export const useZerodhaInstruments = (exchange = 'NSE') =>
  useQuery({
    queryKey: ['zerodha-instruments', exchange],
    queryFn: () => api.getZerodhaInstruments(exchange).then(r => r.data),
    staleTime: 60 * 60 * 1000, // 1 hour — mirrors server-side TTL
    retry: false,              // fail fast if not authenticated
  });;
export const useGoldenPatterns = (rl_model_id?: number) =>
  useQuery({
    queryKey: ['golden_patterns', rl_model_id],
    queryFn: () => api.getGoldenPatterns(rl_model_id).then(res => res.data),
  });

// ── Pipeline hooks ─────────────────────────────────────────────────────────

const TERMINAL_PIPELINE_STATUSES = new Set(['completed', 'failed']);

export const useStartPipeline = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: PipelineRequest) => api.startPipeline(req).then(r => r.data),
    onSuccess: (data) => {
      // Seed the status cache immediately so the poller has something to show
      qc.setQueryData(['pipeline-status', data.job_id], null);
    },
  });
};

export const usePipelineStatus = (jobId: string | null) =>
  useQuery<PipelineStatus>({
    queryKey: ['pipeline-status', jobId],
    queryFn: () => api.getPipelineStatus(jobId!).then(r => r.data),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as PipelineStatus | undefined;
      if (!data) return 2000;
      if (TERMINAL_PIPELINE_STATUSES.has(data.status)) return false;
      return 2000;
    },
  });

