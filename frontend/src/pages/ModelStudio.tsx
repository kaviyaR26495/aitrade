import { useState, useRef, useEffect } from 'react';
import { Card, Button, Select, SearchableSelect, Input, Badge, StatCard, EmptyState, PageHeader, Tabs, ListItem } from '../components/ui';
import {
  useAlgorithms, useRlModels, useKnnModels, useLstmModels,
  useUniverseStocks, useTrainModel, useDistillModel, useRlModelLogs, useDeviceInfo,
  useStopTraining, usePauseTraining, useResumeTraining, useDeleteRlModel,
} from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Brain, Cpu, Layers, ChevronDown, ChevronUp, Zap, Square, Pause, Play, Trash2, ToggleLeft, ToggleRight } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

type Tab = 'rl' | 'distill' | 'ensemble';

export default function ModelStudio() {
  const [tab, setTab] = useState<Tab>('rl');
  const { data: algorithms } = useAlgorithms();
  const { data: device } = useDeviceInfo();
  const { data: rlModels } = useRlModels();
  const { data: knnModels } = useKnnModels();
  const { data: lstmModels } = useLstmModels();
  const { data: stocks } = useUniverseStocks();
  const trainMutation = useTrainModel();
  const distillMutation = useDistillModel();
  const stopMutation = useStopTraining();
  const pauseMutation = usePauseTraining();
  const resumeMutation = useResumeTraining();
  const deleteMutation = useDeleteRlModel();
  const { addNotification } = useAppStore();

  const [algo, setAlgo] = useState('PPO');
  const [stockId, setStockId] = useState('');
  const [timesteps, setTimesteps] = useState('100000');
  const [interval, setInterval] = useState('day');
  const [minQuality, setMinQuality] = useState('0.8');
  const [regimeFilter, setRegimeFilter] = useState<number[]>([]);

  const [selectedLogModelId, setSelectedLogModelId] = useState<number | null>(null);
  const selectedLogModel = rlModels?.find((m: any) => m.id === selectedLogModelId);
  const isLogModelActive = selectedLogModel?.status === 'training' || selectedLogModel?.status === 'pending' || selectedLogModel?.status === 'paused';
  const { data: trainingLogs } = useRlModelLogs(selectedLogModelId, isLogModelActive);

  // Global log visibility toggle (default ON)
  const [logsVisible, setLogsVisible] = useState(true);
  // Id of model pending delete confirmation
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null);

  const hasGpu = device?.device === 'cuda';
  // User can force CPU even when GPU is available; default to GPU if present
  const [forceDevice, setForceDevice] = useState<'auto' | 'cpu'>('auto');
  const effectiveDevice = forceDevice === 'cpu' ? 'cpu' : (hasGpu ? 'cuda' : 'cpu');

  const [rlModelId, setRlModelId] = useState('');
  const [kNeighbors, setKNeighbors] = useState('5');
  const [lstmHidden, setLstmHidden] = useState('128');
  const [lstmLayers, setLstmLayers] = useState('2');

  const algoOptions = algorithms && Array.isArray(algorithms)
    ? algorithms.map((a: any) => ({ value: a.name, label: a.name }))
    : [{ value: 'PPO', label: 'PPO' }];

  const stockOptions = [
    { value: '', label: 'Select stock...' },
    ...(stocks?.map((s: any) => ({ value: String(s.id), label: s.symbol })) ?? []),
  ];

  const rlModelOptions = [
    { value: '', label: 'Select RL model...' },
    ...(rlModels?.map((m: any) => ({ value: String(m.id), label: `${m.algorithm} — ${m.name ?? m.id}` })) ?? []),
  ];

  const handleTrain = () => _doTrain();

  const _doTrain = () => {
    trainMutation.mutate(
      {
        stock_ids: [parseInt(stockId)],
        algorithm: algo,
        total_timesteps: parseInt(timesteps),
        interval,
        min_quality: parseFloat(minQuality),
        regime_ids: regimeFilter.length > 0 ? regimeFilter : undefined,
        device: effectiveDevice,
      },
      {
        onSuccess: (data: any) => {
          addNotification({ type: 'success', message: 'Training started' });
          // Auto-open logs for the new model
          if (data?.data?.model_id) setSelectedLogModelId(data.data.model_id);
        },
        onError: (e: any) => {
          let msg = 'Training failed';
          const d = e?.response?.data?.detail;
          if (typeof d === 'string') msg = d;
          else if (Array.isArray(d)) msg = d.map((x: any) => x.msg).join(', ');
          addNotification({ type: 'error', message: msg });
        },
      }
    );
  };

  const handleDistill = () => {
    distillMutation.mutate(
      {
        rl_model_id: parseInt(rlModelId),
        stock_ids: [parseInt(stockId)],
        interval,
        k_neighbors: parseInt(kNeighbors),
        lstm_hidden_size: parseInt(lstmHidden),
        lstm_num_layers: parseInt(lstmLayers),
      },
      {
        onSuccess: () => addNotification({ type: 'success', message: 'Distillation complete' }),
        onError: (e: any) => {
          let msg = 'Distillation failed';
          const d = e?.response?.data?.detail;
          if (typeof d === 'string') msg = d;
          else if (Array.isArray(d)) msg = d.map((x: any) => x.msg).join(', ');
          addNotification({ type: 'error', message: msg });
        },
      }
    );
  };

  const toggleRegime = (id: number) => {
    setRegimeFilter((prev) =>
      prev.includes(id) ? prev.filter((r) => r !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-8">
      <PageHeader title="Model Studio" description="Train RL agents and distill into fast inference models">
        <Tabs
          tabs={[
            { id: 'rl', label: 'RL Training' },
            { id: 'distill', label: 'Distillation' },
            { id: 'ensemble', label: 'Ensemble' },
          ]}
          activeTab={tab}
          onTabChange={(t) => setTab(t as Tab)}
          variant="pills"
        />
      </PageHeader>

      <div className="grid grid-cols-3 gap-5">
        <StatCard label="RL Models" value={rlModels?.length ?? 0} icon={<Brain size={18} />} color="var(--primary)" />
        <StatCard label="KNN Models" value={knnModels?.length ?? 0} icon={<Cpu size={18} />} color="var(--success)" />
        <StatCard label="LSTM Models" value={lstmModels?.length ?? 0} icon={<Layers size={18} />} color="var(--warning)" />
      </div>


      {tab === 'rl' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <Card title="Train RL Model">
            <div className="space-y-5">
              {/* Device selector */}
              {device && (
                <div className="flex items-center justify-between px-3 py-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface-2)]">
                  <div className="flex items-center gap-2 text-sm">
                    <Zap size={14} className={hasGpu ? 'text-emerald-400' : 'text-[var(--text-dim)]'} />
                    {hasGpu ? (
                      <span className="text-[var(--text)]">
                        {device.gpu_name}
                        <span className="text-[var(--text-dim)] ml-1.5 text-xs">{device.gpu_memory_gb} GB · CUDA {device.cuda_version}</span>
                      </span>
                    ) : (
                      <span className="text-[var(--text-dim)] text-sm">No GPU detected</span>
                    )}
                  </div>
                  {/* GPU / CPU toggle */}
                  <button
                    onClick={() => setForceDevice(d => d === 'cpu' ? 'auto' : 'cpu')}
                    className="flex items-center gap-1.5 text-xs transition-colors"
                    title={forceDevice === 'cpu' ? 'Switch to GPU training' : 'Switch to CPU training'}
                  >
                    {effectiveDevice === 'cuda' ? (
                      <>
                        <ToggleRight size={18} className="text-emerald-400" />
                        <span className="text-emerald-400 font-medium">GPU</span>
                      </>
                    ) : (
                      <>
                        <ToggleLeft size={18} className="text-[var(--text-dim)]" />
                        <span className="text-[var(--text-dim)]">{hasGpu ? 'CPU (GPU off)' : 'CPU'}</span>
                      </>
                    )}
                  </button>
                </div>
              )}
              <Select value={algo} onChange={setAlgo} options={algoOptions} label="Algorithm" data-guide-id="algorithm-select" />
              <SearchableSelect value={stockId} onChange={setStockId} options={stockOptions} label="Stock" placeholder="Search stocks..." />
              <Select value={interval} onChange={setInterval} options={[
                { value: 'day', label: 'Daily' },
                { value: 'week', label: 'Weekly' },
              ]} label="Interval" />
              <Input value={timesteps} onChange={setTimesteps} label="Total Timesteps" type="number" />
              <Input value={minQuality} onChange={setMinQuality} label="Min Quality Threshold" type="number" />

              <div>
                <label className="block text-xs text-[var(--text-dim)] mb-2 font-medium uppercase tracking-wider">Regime Filter (optional)</label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { id: 0, label: '0: Bullish (Low Vol)' },
                    { id: 1, label: '1: Bullish (High Vol)' },
                    { id: 2, label: '2: Neutral (Low Vol)' },
                    { id: 3, label: '3: Neutral (High Vol)' },
                    { id: 4, label: '4: Bearish (Low Vol)' },
                    { id: 5, label: '5: Bearish (High Vol)' }
                  ].map(({ id, label }) => (
                    <button
                      key={id}
                      onClick={() => toggleRegime(id)}
                      className={`px-3 py-1.5 rounded-[var(--radius-sm)] text-xs font-medium border transition-all duration-150 ${
                        regimeFilter.includes(id)
                          ? 'bg-[var(--primary)] border-[var(--primary)] text-[var(--bg)] shadow-sm'
                          : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--primary)] hover:text-[var(--primary)]'
                      }`}
                      title={label}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <Button onClick={handleTrain} loading={trainMutation.isPending} disabled={!stockId} data-guide-id="train-btn">
                Start Training
              </Button>
            </div>
          </Card>

          <Card
            title="Trained RL Models"
            action={
              <button
                onClick={() => setLogsVisible(v => !v)}
                className="text-xs text-[var(--text-dim)] hover:text-[var(--text)] transition-colors px-2 py-1 rounded border border-[var(--border)] hover:border-[var(--primary)]"
              >
                {logsVisible ? 'Hide logs' : 'Show logs'}
              </button>
            }
          >
            {rlModels && rlModels.length > 0 ? (
              <div className="space-y-2 max-h-[480px] overflow-y-auto">
                {rlModels.map((m: any) => {
                  const cfg = m.training_config ?? {};
                  const stockIds: number[] = cfg.stock_ids ?? [];
                  const steps: number | undefined = cfg.total_timesteps;
                  const isActive = m.status === 'training' || m.status === 'pending';
                  const isPaused = m.status === 'paused';
                  const isControllable = isActive || isPaused;
                  // Logs open by default for active models; toggle-able per model
                  const showingLogs = logsVisible && (
                    selectedLogModelId === m.id || (isActive && selectedLogModelId === null)
                  );
                  const confirmingDelete = deleteConfirmId === m.id;
                  const badgeColor =
                    m.status === 'completed' ? 'green' :
                    m.status === 'training' ? 'yellow' :
                    m.status === 'paused' ? 'yellow' :
                    m.status === 'failed' ? 'red' :
                    m.status === 'stopped' ? 'gray' : 'gray';
                  return (
                    <div key={m.id}>
                      <ListItem
                        active={showingLogs}
                        onClick={() => {
                          setLogsVisible(true);
                          setSelectedLogModelId(selectedLogModelId === m.id ? null : m.id);
                        }}
                        left={
                          <div className="w-full">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium">{m.algorithm}</span>
                              <Badge color={badgeColor}>
                                {m.status}
                              </Badge>
                              {isActive && (
                                <span className="ml-auto text-[10px] text-[var(--text-dim)] animate-pulse">● live</span>
                              )}
                            </div>
                            <div className="text-xs text-[var(--text-dim)]">
                              {stockIds.length > 0 ? `Stock IDs: ${stockIds.join(', ')}` : ''}
                              {steps ? ` · ${steps.toLocaleString()} steps` : ''}
                              {m.interval ? ` · ${m.interval}` : ''}
                            </div>
                            {m.total_reward != null && (
                              <div className="flex gap-3 mt-1 text-xs">
                                <span className="tabular-nums">Reward: <span className="text-[var(--text)]">{m.total_reward.toFixed(2)}</span></span>
                                {m.sharpe_ratio != null && <span className="tabular-nums">Best reward: <span className="text-[var(--text)]">{m.sharpe_ratio.toFixed(2)}</span></span>}
                              </div>
                            )}
                            {/* Action buttons */}
                            <div
                              className="flex items-center gap-1 mt-2"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {isActive && (
                                <>
                                  <button
                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
                                    onClick={() => stopMutation.mutate(m.id)}
                                    disabled={stopMutation.isPending}
                                    title="Stop training"
                                  >
                                    <Square size={11} /> Stop
                                  </button>
                                  <button
                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 transition-colors"
                                    onClick={() => pauseMutation.mutate(m.id)}
                                    disabled={pauseMutation.isPending}
                                    title="Pause training"
                                  >
                                    <Pause size={11} /> Pause
                                  </button>
                                </>
                              )}
                              {isPaused && (
                                <>
                                  <button
                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-green-500/10 text-green-400 hover:bg-green-500/20 transition-colors"
                                    onClick={() => resumeMutation.mutate(m.id)}
                                    disabled={resumeMutation.isPending}
                                    title="Resume training"
                                  >
                                    <Play size={11} /> Resume
                                  </button>
                                  <button
                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
                                    onClick={() => stopMutation.mutate(m.id)}
                                    disabled={stopMutation.isPending}
                                    title="Stop training"
                                  >
                                    <Square size={11} /> Stop
                                  </button>
                                </>
                              )}
                              {!isControllable && (
                                confirmingDelete ? (
                                  <>
                                    <span className="text-xs text-[var(--text-dim)] mr-1">Delete?</span>
                                    <button
                                      className="px-2 py-0.5 rounded text-xs bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                                      onClick={() => { deleteMutation.mutate(m.id); setDeleteConfirmId(null); }}
                                      disabled={deleteMutation.isPending}
                                    >
                                      Yes
                                    </button>
                                    <button
                                      className="px-2 py-0.5 rounded text-xs bg-[var(--surface-2)] text-[var(--text-dim)] hover:bg-[var(--surface-3)] transition-colors"
                                      onClick={() => setDeleteConfirmId(null)}
                                    >
                                      Cancel
                                    </button>
                                  </>
                                ) : (
                                  <button
                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
                                    onClick={() => setDeleteConfirmId(m.id)}
                                    title="Delete model"
                                  >
                                    <Trash2 size={11} /> Delete
                                  </button>
                                )
                              )}
                            </div>
                          </div>
                        }
                        right={
                          <span className="text-[var(--text-dim)] shrink-0">
                            {showingLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </span>
                        }
                      />
                      {showingLogs && (
                        <TrainingLogsPanel
                          logs={trainingLogs?.logs ?? []}
                          source={trainingLogs?.source}
                          isActive={isActive}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <EmptyState icon={<Brain size={28} />} title="No trained models" description="Configure and start training to see RL models here." />
            )}
          </Card>
        </div>
      )}

      {tab === 'distill' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <Card title="Distill RL → KNN + LSTM">
            <div className="space-y-4">
              <Select value={rlModelId} onChange={setRlModelId} options={rlModelOptions} label="Source RL Model" />

              <div className="border-t border-[var(--border)] pt-4">
                <h4 className="text-sm font-medium mb-3">KNN Settings</h4>
                <Input value={kNeighbors} onChange={setKNeighbors} label="K Neighbors" type="number" />
              </div>

              <div className="border-t border-[var(--border)] pt-4">
                <h4 className="text-sm font-medium mb-3">LSTM Settings</h4>
                <div className="grid grid-cols-2 gap-3">
                  <Input value={lstmHidden} onChange={setLstmHidden} label="Hidden Size" type="number" />
                  <Input value={lstmLayers} onChange={setLstmLayers} label="Num Layers" type="number" />
                </div>
              </div>

              <Button onClick={handleDistill} loading={distillMutation.isPending} disabled={!rlModelId} data-guide-id="distill-btn">
                Distill Model
              </Button>
            </div>
          </Card>

          <div className="space-y-4">
            <Card title="KNN Models">
              {knnModels && knnModels.length > 0 ? (
                <div className="space-y-2">
                  {knnModels.map((m: any) => (
                    <div key={m.id} className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] transition-colors text-sm">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">KNN (k={m.k_neighbors})</span>
                        <Badge color="green">{m.status}</Badge>
                      </div>
                      {m.metrics && (
                        <div className="text-xs text-[var(--text-dim)] mt-1 tabular-nums">
                          Accuracy: {(m.metrics.accuracy * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState icon={<Cpu size={24} />} title="No KNN models" description="Distill an RL model to create KNN models." />
              )}
            </Card>

            <Card title="LSTM Models">
              {lstmModels && lstmModels.length > 0 ? (
                <div className="space-y-2">
                  {lstmModels.map((m: any) => (
                    <div key={m.id} className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] transition-colors text-sm">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">LSTM (h={m.hidden_size})</span>
                        <Badge color="green">{m.status}</Badge>
                      </div>
                      {m.metrics && (
                        <div className="text-xs text-[var(--text-dim)] mt-1 tabular-nums">
                          Accuracy: {(m.metrics.accuracy * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState icon={<Layers size={24} />} title="No LSTM models" description="Distill an RL model to create LSTM models." />
              )}
            </Card>
          </div>
        </div>
      )}

      {tab === 'ensemble' && (
        <Card title="Ensemble Configuration">
          <div className="space-y-4">
            <p className="text-sm text-[var(--text-muted)]">
              The ensemble combines KNN and LSTM predictions using weighted probabilities.
              When agreement is required, the ensemble only outputs BUY/SELL when both models agree.
            </p>

            <div className="grid grid-cols-2 gap-5">
              <div className="p-4 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
                <h4 className="text-xs font-medium text-[var(--text-dim)] uppercase tracking-wider mb-2">KNN Weight</h4>
                <div className="text-2xl font-bold text-blue-400 tabular-nums">0.5</div>
              </div>
              <div className="p-4 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
                <h4 className="text-xs font-medium text-[var(--text-dim)] uppercase tracking-wider mb-2">LSTM Weight</h4>
                <div className="text-2xl font-bold text-purple-400 tabular-nums">0.5</div>
              </div>
            </div>

            <div className="p-4 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium">Agreement Required</h4>
                <Badge color="green">ON</Badge>
              </div>
              <p className="text-xs text-[var(--text-dim)]">
                When enabled, predictions where KNN and LSTM disagree are set to HOLD.
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Training Logs Panel ────────────────────────────────────────────────

interface TrainingLogEntry {
  timestep?: number;
  progress?: number;
  fps?: number;
  ep_rew_mean?: number;
  ep_len_mean?: number;
  loss?: number;
  value_loss?: number;
  net_worth?: number;
  profit_pct?: number;
  error?: string;
  info?: string;
  gpu_ok?: boolean;
  stopped?: boolean;
  paused?: boolean;
}

type LogTab = 'logs' | 'chart';

function TrainingLogsPanel({
  logs,
  source,
  isActive,
}: {
  logs: TrainingLogEntry[];
  source?: string;
  isActive?: boolean;
}) {
  const [logTab, setLogTab] = useState<LogTab>('logs');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever new log entries arrive
  useEffect(() => {
    if (logTab === 'logs' && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length, logTab]);

  const latest = logs.filter(e => e.timestep != null).at(-1);
  const progressPct = latest?.progress != null ? Math.round(latest.progress * 100) : null;

  // GPU status from first info entry
  const gpuEntry = logs.find(e => e.gpu_ok != null);

  // Chart data: only entries with timestep + reward or profit
  const chartData = logs
    .filter(e => e.timestep != null && (e.ep_rew_mean != null || e.profit_pct != null))
    .map(e => ({
      step: e.timestep,
      reward: e.ep_rew_mean,
      profit: e.profit_pct,
    }));

  return (
    <div className="mx-3 mb-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
      {/* Progress bar */}
      {progressPct != null && (
        <div className="h-1 bg-[var(--bg-input)]">
          <div
            className="h-1 bg-[var(--primary)] transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      <div className="px-3 pt-2 pb-3">
        {/* Header row */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-medium text-[var(--text-dim)] uppercase tracking-wider">
              Training {source === 'live' ? '· live' : ''}
            </span>
            {gpuEntry != null && (
              <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${gpuEntry.gpu_ok ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
                {gpuEntry.gpu_ok ? '⚡ GPU' : '⚠ CPU fallback'}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {progressPct != null && (
              <span className="text-xs tabular-nums text-[var(--primary)] font-medium">
                {progressPct}%{latest?.fps != null ? ` · ${latest.fps.toFixed(0)} fps` : ''}
              </span>
            )}
            {latest?.profit_pct != null && (
              <span className={`text-xs tabular-nums font-medium ${latest.profit_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                P&L {latest.profit_pct >= 0 ? '+' : ''}{latest.profit_pct.toFixed(2)}%
              </span>
            )}
            {/* Tab switcher */}
            <div className="flex rounded overflow-hidden border border-[var(--border)]">
              {(['logs', 'chart'] as LogTab[]).map(t => (
                <button
                  key={t}
                  onClick={() => setLogTab(t)}
                  className={`px-2 py-0.5 text-[10px] font-medium transition-colors ${logTab === t ? 'bg-[var(--primary)] text-[var(--bg)]' : 'text-[var(--text-dim)] hover:text-[var(--text)]'}`}
                >
                  {t === 'logs' ? 'Logs' : 'Chart'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Live stats bar */}
        {latest && (latest.ep_rew_mean != null || latest.loss != null || latest.net_worth != null) && (
          <div className="flex gap-4 mb-2 text-[11px] tabular-nums font-mono">
            {latest.ep_rew_mean != null && (
              <span className="text-[var(--text-dim)]">
                rew <span className={latest.ep_rew_mean >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                  {latest.ep_rew_mean.toFixed(3)}
                </span>
              </span>
            )}
            {latest.loss != null && (
              <span className="text-[var(--text-dim)]">loss <span className="text-amber-400">{latest.loss.toFixed(5)}</span></span>
            )}
            {latest.net_worth != null && (
              <span className="text-[var(--text-dim)]">
                net_worth ₹<span className="text-[var(--text)]">{latest.net_worth.toLocaleString('en-IN')}</span>
              </span>
            )}
            {latest.ep_len_mean != null && (
              <span className="text-[var(--text-dim)]">ep_len <span className="text-[var(--text)]">{latest.ep_len_mean.toFixed(0)}</span></span>
            )}
          </div>
        )}

        {/* LOG TAB */}
        {logTab === 'logs' && (
          logs.length === 0 ? (
            <p className="text-xs text-[var(--text-dim)] italic">
              {isActive ? 'Waiting for first checkpoint…' : 'No log data available.'}
            </p>
          ) : (
            <div
              ref={scrollRef}
              className="space-y-0.5 max-h-52 overflow-y-auto font-mono text-[11px] pr-1"
            >
              {logs.map((entry, i) => (
                <div
                  key={i}
                  className={`flex gap-2 leading-relaxed ${
                    entry.error
                      ? 'text-rose-400'
                      : entry.info
                      ? 'text-[var(--text-dim)] italic'
                      : 'text-[var(--text-muted)]'
                  }`}
                >
                  {entry.error ? (
                    <span>✗ {entry.error}</span>
                  ) : entry.info ? (
                    <span className={entry.gpu_ok === true ? 'text-emerald-400 not-italic font-medium' : entry.gpu_ok === false ? 'text-amber-400 not-italic' : ''}>
                      ℹ {entry.info}
                    </span>
                  ) : (
                    <>
                      <span className="shrink-0 text-[var(--text-dim)] w-14 text-right">
                        {entry.timestep?.toLocaleString() ?? '—'}
                      </span>
                      <span className="text-[var(--text-dim)]">
                        {entry.progress != null ? `${(entry.progress * 100).toFixed(1)}%` : ''}
                      </span>
                      {entry.ep_rew_mean != null && (
                        <span className={entry.ep_rew_mean >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          rew={entry.ep_rew_mean.toFixed(3)}
                        </span>
                      )}
                      {entry.profit_pct != null && (
                        <span className={entry.profit_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          profit={entry.profit_pct >= 0 ? '+' : ''}{entry.profit_pct.toFixed(2)}%
                        </span>
                      )}
                      {entry.loss != null && (
                        <span className="text-amber-400">loss={entry.loss.toFixed(5)}</span>
                      )}
                      {entry.fps != null && (
                        <span className="text-[var(--text-dim)]">{entry.fps.toFixed(0)}fps</span>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          )
        )}

        {/* CHART TAB */}
        {logTab === 'chart' && (
          chartData.length < 2 ? (
            <p className="text-xs text-[var(--text-dim)] italic py-4 text-center">
              {isActive ? 'Waiting for enough data points…' : 'No chart data available.'}
            </p>
          ) : (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
                    tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                  />
                  <YAxis
                    yAxisId="rew"
                    tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
                    width={44}
                  />
                  <YAxis
                    yAxisId="profit"
                    orientation="right"
                    tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
                    tickFormatter={(v) => `${v.toFixed(1)}%`}
                    width={44}
                  />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 }}
                    labelStyle={{ color: 'var(--text-dim)' }}
                    formatter={(value: number, name: string) =>
                      name === 'profit' ? [`${value.toFixed(2)}%`, 'Profit'] : [value.toFixed(4), 'Reward']
                    }
                    labelFormatter={(v) => `Step ${v?.toLocaleString()}`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    yAxisId="rew"
                    type="monotone"
                    dataKey="reward"
                    name="Reward"
                    stroke="#6ee7b7"
                    dot={false}
                    strokeWidth={1.5}
                    connectNulls
                  />
                  <Line
                    yAxisId="profit"
                    type="monotone"
                    dataKey="profit"
                    name="Profit %"
                    stroke="#60a5fa"
                    dot={false}
                    strokeWidth={1.5}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )
        )}
      </div>
    </div>
  );
}
