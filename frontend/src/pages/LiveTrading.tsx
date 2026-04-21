import { useMemo, useState, useRef, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Card, Button, Badge, StatCard, EmptyState, PageHeader, Table, Modal, Tooltip, SkeletonTable, type TableColumn } from '../components/ui';
import { useSignals, useGenerateSignals, useTrainMhLstm, useExecuteSignal, useOrders, useUniverseStocks, usePredictionJob, useCancelPredictionJob, useSignalPreview, useOhlcv, useIndicators } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Crosshair, Play, Shield, ShieldAlert, XCircle, Loader2, Filter, Zap, ToggleLeft, ToggleRight, BrainCircuit, CalendarDays, X, BarChart2 } from 'lucide-react';
import LightweightCandleChart, { type IndicatorSeries, type PriceLevel } from '../components/LightweightCandleChart';

type StatusFilter = 'ALL' | 'pending' | 'active' | 'target_hit' | 'sl_hit' | 'expired';

const STATUS_COLORS: Record<string, 'green' | 'blue' | 'yellow' | 'red' | 'gray'> = {
  pending: 'yellow',
  active: 'blue',
  target_hit: 'green',
  sl_hit: 'red',
  expired: 'gray',
};

export default function LiveTrading() {
  const { addNotification } = useAppStore();
  const [interval] = useState('day');
  // Date range filter — both start as empty so all recent signals load on mount.
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('ALL');
  const [minPop, setMinPop] = useState('0');
  const [symbolQuery, setSymbolQuery] = useState('');
  // Separate date for signal generation (independent of the view date-range filter)
  const [generateDate, setGenerateDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Chart pattern modal
  const [selectedSignal, setSelectedSignal] = useState<any>(null);
  const qc = useQueryClient();

  // Signal generation job tracking
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const { data: job } = usePredictionJob(activeJobId);
  const cancelJob = useCancelPredictionJob();
  // Remember the date passed to the generation job so we can snap the filter to it on completion.
  const pendingGenerateDateRef = useRef<string | null>(null);

  // Stock filter for targeted generation
  const { data: universeStocks } = useUniverseStocks();
  const [showStockSelector, setShowStockSelector] = useState(false);
  const [selectedStockIds, setSelectedStockIds] = useState<Set<number>>(new Set());
  const [stockSearchQuery, setStockSearchQuery] = useState('');

  // Execution state
  const [executeModal, setExecuteModal] = useState<any>(null);
  const [dryRun, setDryRun] = useState(true);
  const executeSignal = useExecuteSignal();
  const { data: signalPreview } = useSignalPreview(executeModal?.id ?? null);

  useEffect(() => {
    if (job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled') {
      if (job.status === 'completed') {
        pendingGenerateDateRef.current = null;
        setDateFrom('');
        setDateTo('');
        qc.invalidateQueries({ queryKey: ['signals'] });
        addNotification({ type: 'success', message: 'Signal generation completed' });
      } else if (job.status === 'cancelled') {
        addNotification({ type: 'warning', message: 'Signal generation stopped' });
      } else if (job.status === 'failed') {
        addNotification({ type: 'error', message: `Signal generation failed: ${job.error}` });
      }
      setIsSubmitting(false);
      setTimeout(() => setActiveJobId(null), 3000);
    }
  }, [job, addNotification, qc]);

  // Fetch signals — min_pop is applied client-side so changing it never
  // triggers a new network request and never hides data that was already fetched.
  const signalParams = useMemo(() => ({
    date_from: dateFrom || undefined,
    date_to: dateTo || undefined,
    status: statusFilter === 'ALL' ? undefined : statusFilter,
  }), [dateFrom, dateTo, statusFilter]);

  const { data: signals, isLoading } = useSignals(signalParams);

  const { data: orders } = useOrders(20);
  const generateSignals = useGenerateSignals();
  const trainMhLstm = useTrainMhLstm();

  // Notify when training completes or fails
  const prevTrainingState = useRef(trainMhLstm.trainingState);
  useEffect(() => {
    const prev = prevTrainingState.current;
    prevTrainingState.current = trainMhLstm.trainingState;
    if (prev === trainMhLstm.trainingState) return;
    if (trainMhLstm.trainingState === 'success' && (prev === 'queued' || prev === 'running')) {
      addNotification({ type: 'success', message: 'LSTM model training completed — you can now generate signals' });
    } else if (trainMhLstm.trainingState === 'failure') {
      addNotification({ type: 'error', message: 'LSTM model training failed — check Celery worker logs' });
    } else if (trainMhLstm.trainingState === 'queued') {
      addNotification({ type: 'info', message: 'LSTM model training queued' });
    }
  }, [trainMhLstm.trainingState, addNotification]);

  // Filtered signals — symbol search and min_pop are applied client-side.
  const filteredSignals = useMemo(() => {
    let all = signals ?? [];
    const q = symbolQuery.trim().toLowerCase();
    if (q) all = all.filter((s: any) => String(s.symbol ?? '').toLowerCase().includes(q));
    const popThreshold = Number(minPop) / 100;
    if (popThreshold > 0) all = all.filter((s: any) => (s.pop_score ?? 0) >= popThreshold);
    return all;
  }, [signals, symbolQuery, minPop]);

  // Stat aggregations
  const activeCount = filteredSignals.filter((s: any) => s.status === 'active').length;
  const pendingCount = filteredSignals.filter((s: any) => s.status === 'pending').length;
  const avgPop = filteredSignals.length > 0
    ? filteredSignals.reduce((sum: number, s: any) => sum + (s.pop_score ?? 0), 0) / filteredSignals.length
    : 0;
  const avgRR = filteredSignals.length > 0
    ? filteredSignals.reduce((sum: number, s: any) => sum + (s.initial_rr_ratio ?? 0), 0) / filteredSignals.length
    : 0;

  const signalColumns: TableColumn<any>[] = [
    {
      key: 'direction',
      label: 'Dir',
      tooltip: 'Signal direction: LONG (target > entry) or SHORT (target < entry)',
      align: 'center',
      render: (s) => {
        const isShort = s.target_price < s.entry_price;
        return (
          <Badge color={isShort ? 'red' : 'green'}>{isShort ? 'SHORT' : 'LONG'}</Badge>
        );
      },
    },
    {
      key: 'symbol',
      label: 'Symbol',
      tooltip: 'Stock ticker symbol — click row to view chart',
      render: (s) => <span className="font-medium">{s.symbol}</span>,
    },
    {
      key: 'entry_price',
      label: 'Entry',
      tooltip: 'Suggested entry price',
      align: 'right',
      mono: true,
      render: (s) => <span className="tabular-nums">{s.entry_price?.toFixed(2) ?? '—'}</span>,
    },
    {
      key: 'target_price',
      label: 'Target',
      tooltip: 'Target price (profit booking level)',
      align: 'right',
      mono: true,
      render: (s) => <span className="tabular-nums text-emerald-400">{s.target_price?.toFixed(2) ?? '—'}</span>,
    },
    {
      key: 'stoploss_price',
      label: 'SL',
      tooltip: 'Stop-loss price (current trailing SL if active)',
      align: 'right',
      mono: true,
      render: (s) => {
        const sl = s.current_stoploss ?? s.stoploss_price;
        return <span className="tabular-nums text-red-400">{sl?.toFixed(2) ?? '—'}</span>;
      },
    },
    {
      key: 'initial_rr_ratio',
      label: 'R:R',
      tooltip: 'Risk-Reward ratio (target distance / SL distance)',
      align: 'center',
      mono: true,
      render: (s) => {
        const rr = s.current_rr_ratio ?? s.initial_rr_ratio;
        return (
          <span className={`tabular-nums font-medium ${rr >= 2.5 ? 'text-emerald-400' : rr >= 1.5 ? 'text-amber-400' : 'text-red-400'}`}>
            {rr?.toFixed(1) ?? '—'}
          </span>
        );
      },
    },
    {
      key: 'pop_score',
      label: 'PoP%',
      tooltip: 'Probability of Profit from meta-classifier',
      align: 'center',
      render: (s) => (
        <span className={`font-medium tabular-nums ${s.pop_score >= 0.7 ? 'text-emerald-400' : s.pop_score >= 0.55 ? 'text-amber-400' : 'text-[var(--text-muted)]'}`}>
          {((s.pop_score ?? 0) * 100).toFixed(0)}%
        </span>
      ),
    },
    {
      key: 'fqs_score',
      label: 'FQS',
      tooltip: 'Final Quality Score — composite signal quality metric',
      align: 'center',
      mono: true,
      render: (s) => (
        <span className={`tabular-nums ${s.fqs_score >= 0.7 ? 'text-emerald-400' : 'text-[var(--text-dim)]'}`}>
          {(s.fqs_score ?? 0).toFixed(2)}
        </span>
      ),
    },
    {
      key: 'status',
      label: 'Status',
      tooltip: 'Signal lifecycle status',
      align: 'center',
      render: (s) => (
        <Badge color={STATUS_COLORS[s.status] ?? 'gray'}>{s.status}</Badge>
      ),
    },
    {
      key: 'days_since_signal',
      label: 'Days',
      tooltip: 'Trading days since signal was generated',
      align: 'center',
      mono: true,
      render: (s) => {
        const days = s.days_since_signal ?? 0;
        const stale = days > 15;
        return (
          <span className={`text-xs tabular-nums ${stale ? 'text-rose-400 font-bold' : 'text-[var(--text-dim)]'}`}>
            {days}{stale && ' (Stale)'}
          </span>
        );
      },
    },
    {
      key: 'trailing',
      label: 'Trail',
      tooltip: 'Trailing stop status (active / update count)',
      align: 'center',
      render: (s) => (
        s.is_trailing_active
          ? <Badge color="blue">↑{s.trailing_updates_count ?? 0}</Badge>
          : <span className="text-xs text-[var(--text-dim)]">—</span>
      ),
    },
    {
      key: 'execute',
      label: '',
      align: 'right',
      stopPropagation: true,
      render: (s) => (
        s.status === 'pending' ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExecuteModal(s)}
            className="h-7 px-2 text-xs text-[var(--primary)] hover:bg-[var(--primary-dim)]"
          >
            <Zap size={12} className="mr-1" />
            Execute
          </Button>
        ) : null
      ),
    },
  ];

  const handleGenerateSignals = () => {
    if (isSubmitting || !!activeJobId) return;
    setIsSubmitting(true);
    const stockIds = selectedStockIds.size > 0
      ? Array.from(selectedStockIds)
      : undefined;
    pendingGenerateDateRef.current = generateDate;  // capture before async mutation
    generateSignals.mutate(
      { interval, stock_ids: stockIds, target_date: generateDate, pop_threshold: Number(minPop) / 100 || 0.55 },
      {
        onSuccess: (res: any) => {
          setActiveJobId(res.job_id);
          addNotification({ type: 'info', message: 'Started signal generation' });
        },
        onError: () => {
          addNotification({ type: 'error', message: 'Failed to start signal generation' });
          setIsSubmitting(false);
        },
      }
    );
  };

  const handleExecuteConfirm = () => {
    if (!executeModal) return;
    executeSignal.mutate(
      { signalId: executeModal.id, dryRun },
      {
        onSuccess: (res: any) => {
          const mode = dryRun ? 'paper-trade' : 'live';
          addNotification({ type: 'success', message: `${executeModal.symbol} executed (${mode}) @ ${res.fill_price}` });
          setExecuteModal(null);
        },
        onError: (err: any) => {
          addNotification({ type: 'error', message: err?.response?.data?.detail ?? 'Execution failed' });
        },
      }
    );
  };

  return (
    <div className="space-y-8">
      <PageHeader title="Live Trading" description="Generate TPML signals and execute trades">
        <div className="flex gap-2 items-center flex-wrap">
          {/* ── Date range ─────────────────────────────────── */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] shadow-sm">
            <CalendarDays size={13} className="text-[var(--text-dim)] shrink-0" />
            <input
              type="date"
              title="From date"
              value={dateFrom}
              max={dateTo || undefined}
              onChange={(e) => setDateFrom(e.target.value)}
              className="bg-transparent border-none p-0 text-xs font-medium focus:ring-0 cursor-pointer w-[110px]"
            />
            <span className="text-[var(--text-dim)] text-xs select-none">→</span>
            <input
              type="date"
              title="To date"
              value={dateTo}
              min={dateFrom || undefined}
              onChange={(e) => setDateTo(e.target.value)}
              className="bg-transparent border-none p-0 text-xs font-medium focus:ring-0 cursor-pointer w-[110px]"
            />
            {(dateFrom || dateTo) && (
              <button
                type="button"
                title="Clear date filter"
                onClick={() => { setDateFrom(''); setDateTo(''); }}
                className="ml-1 text-[var(--text-dim)] hover:text-[var(--text)] transition-colors"
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* ── Latest quick-select ───────────────────────── */}
          <Tooltip content="Jump to the most recent signal date" side="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setDateFrom('');
                setDateTo('');
              }}
              className="h-9 gap-1.5 border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] text-xs text-[var(--text-dim)]"
            >
              Latest
            </Button>
          </Tooltip>

          {activeJobId && (
            <div className="flex items-center gap-3 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--primary-dim)] rounded-[var(--radius-sm)] shadow-sm animate-in fade-in slide-in-from-right-4">
              <div className="flex flex-col gap-0.5 min-w-[120px]">
                <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-dim)]">
                  <span>Progress</span>
                  <span>{job ? Math.max(job.progress, 1) : 0}%</span>
                </div>
                <div className="w-full bg-[var(--bg-card)] h-1.5 rounded-full overflow-hidden border border-[var(--border)]">
                  <div
                    className="h-full bg-[var(--primary)] transition-all duration-500 ease-out"
                    style={{ width: `${job ? Math.max(job.progress, 2) : 2}%` }}
                  />
                </div>
              </div>
              <div className="h-4 w-px bg-[var(--border)] mx-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (confirm('Stop signal generation?')) {
                    cancelJob.mutate(activeJobId);
                  }
                }}
                className="h-7 px-2 text-xs text-red-400 hover:bg-red-500/10 gap-1.5"
                loading={cancelJob.isPending}
              >
                {cancelJob.isPending ? <Loader2 className="animate-spin" size={12} /> : <XCircle size={12} />}
                Stop
              </Button>
            </div>
          )}

          <Tooltip content="Select specific stocks for signal generation" side="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowStockSelector(true)}
              disabled={isSubmitting || !!activeJobId}
              icon={<Filter size={14} />}
              className="h-9 gap-1.5 border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
            >
              {selectedStockIds.size > 0 ? (
                <span className="text-[var(--primary)] font-semibold">{selectedStockIds.size} selected</span>
              ) : (
                <span className="text-[var(--text-dim)]">All stocks</span>
              )}
            </Button>
          </Tooltip>

          <Tooltip content={trainMhLstm.trainingState === 'success' ? "Re-train the MultiHorizon LSTM model (Optional)" : "Train the MultiHorizon LSTM model (required before first signal generation)"} side="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                if (confirm(trainMhLstm.trainingState === 'success' ? 'Re-train the MultiHorizon LSTM model? This may take several minutes.' : 'Train the MultiHorizon LSTM model? This will run as a background job and may take several minutes.')) {
                  trainMhLstm.mutate();
                }
              }}
              disabled={trainMhLstm.trainingState === 'queued' || trainMhLstm.trainingState === 'running'}
              loading={trainMhLstm.trainingState === 'queued' || trainMhLstm.trainingState === 'running'}
              icon={
                trainMhLstm.trainingState === 'success'
                  ? <Zap size={14} className="text-green-400" />
                  : trainMhLstm.trainingState === 'failure'
                  ? <XCircle size={14} className="text-red-400" />
                  : <BrainCircuit size={14} />
              }
              className="h-9 gap-1.5 border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
            >
              {trainMhLstm.trainingState === 'queued' && 'Queued...'}
              {trainMhLstm.trainingState === 'running' && 'Training...'}
              {trainMhLstm.trainingState === 'success' && 'Model Ready'}
              {trainMhLstm.trainingState === 'failure' && 'Train Failed'}
              {trainMhLstm.trainingState === 'idle' && 'Train Model'}
            </Button>
          </Tooltip>

          {/* ── Generate-for date ──────────────────────── */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--primary-dim)] rounded-[var(--radius-sm)] shadow-sm" title="Date to generate signals for">
            <Play size={11} className="text-[var(--primary)] shrink-0" />
            <input
              type="date"
              value={generateDate}
              onChange={(e) => setGenerateDate(e.target.value)}
              className="bg-transparent border-none p-0 text-xs font-medium focus:ring-0 cursor-pointer w-[110px] text-[var(--text)]"
            />
          </div>

          <Button
            variant="primary"
            onClick={handleGenerateSignals}
            icon={<Play size={16} />}
            loading={isSubmitting || (!!activeJobId && job?.status === 'running')}
            disabled={isSubmitting || !!activeJobId}
          >
            {isSubmitting || activeJobId ? 'Generating...' : 'Generate Signals'}
          </Button>
        </div>
      </PageHeader>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-5">
        <StatCard label="Active Signals" value={activeCount} color="var(--info)" />
        <StatCard label="Pending Signals" value={pendingCount} color="var(--warning)" />
        <StatCard label="Avg PoP" value={`${(avgPop * 100).toFixed(0)}%`} color="var(--success)" />
        <StatCard label="Avg R:R" value={avgRR.toFixed(1)} color="var(--primary)" />
        <StatCard label="Total" value={filteredSignals.length} color="var(--text-dim)" />
      </div>

      {/* Filters */}
      <Card title="Signal Filters" className="p-4">
        <div className="flex flex-col lg:flex-row lg:items-center gap-3">
          <div className="flex items-center gap-2">
            {(['ALL', 'pending', 'active', 'target_hit', 'sl_hit', 'expired'] as StatusFilter[]).map((status) => {
              const isActive = statusFilter === status;
              return (
                <Button
                  key={status}
                  size="sm"
                  variant={isActive ? 'primary' : 'ghost'}
                  onClick={() => setStatusFilter(status)}
                  className={isActive ? '' : 'text-[var(--text-dim)]'}
                >
                  {status === 'ALL' ? 'All' : status.replace('_', ' ')}
                </Button>
              );
            })}
          </div>

          <div className="flex items-center gap-2 w-full lg:w-auto">
            <span className="text-xs text-[var(--text-dim)] font-medium uppercase tracking-wider">Min PoP</span>
            <input
              type="number"
              min="0"
              max="100"
              step="5"
              value={minPop}
              onChange={(e) => setMinPop(e.target.value)}
              placeholder="55"
              className="h-9 w-[80px] rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
            />
            <span className="text-xs text-[var(--text-dim)]">%</span>
          </div>

          <div className="lg:ml-auto w-full lg:w-[280px]">
            <input
              type="text"
              value={symbolQuery}
              onChange={(e) => setSymbolQuery(e.target.value)}
              placeholder="Search stock symbol..."
              className="w-full h-9 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
            />
          </div>
        </div>
      </Card>

      {/* Signals Table */}
      <Card title="Trade Signals" noPadding action={
        <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">
          {isLoading ? 'Loading...' : `${filteredSignals.length} signals`}
        </span>
      }>
        {isLoading ? (
          <div className="px-6 py-5"><SkeletonTable rows={4} /></div>
        ) : (
          <Table<any>
            columns={signalColumns}
            data={filteredSignals}
            onRowClick={(s) => setSelectedSignal(s)}
            emptyState={
              <EmptyState
                icon={<Crosshair size={32} />}
                title="No signals available"
                description='Click "Generate Signals" to create TPML trade signals.'
              />
            }
          />
        )}
      </Card>

      {/* Recent Orders */}
      <Card title="Recent Orders">
        {orders && orders.length > 0 ? (
          <div className="space-y-2">
            {orders.map((o: any) => (
              <div key={o.id} className="flex items-center justify-between py-3 px-4 rounded-[var(--radius-sm)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] transition-colors text-sm">
                <div className="flex items-center gap-2.5">
                  <Badge color={o.transaction_type === 'BUY' ? 'green' : 'red'}>{o.transaction_type}</Badge>
                  <span className="font-medium">Stock #{o.stock_id}</span>
                  <span className="text-xs text-[var(--text-dim)] tabular-nums">Qty: {o.quantity}</span>
                </div>
                <div className="flex items-center gap-2.5">
                  <Badge color={o.status === 'placed' ? 'blue' : o.status === 'complete' || o.status === 'filled' ? 'green' : 'gray'}>
                    {o.status}
                  </Badge>
                  <span className="text-xs text-[var(--text-dim)]">{o.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState icon={<Shield size={24} />} title="No recent orders" description="Execute signals above to place orders." />
        )}
      </Card>

      {/* Execute Signal Modal */}
      {executeModal && (
        <Modal
          open={!!executeModal}
          onClose={() => setExecuteModal(null)}
          title={`Execute Signal: ${executeModal.symbol}`}
          footer={
            <div className="flex gap-3">
              <Button variant="secondary" onClick={() => setExecuteModal(null)} className="flex-1">Cancel</Button>
              <Button onClick={handleExecuteConfirm} loading={executeSignal.isPending} className="flex-1">
                {dryRun ? 'Paper Trade' : 'Execute Live'}
              </Button>
            </div>
          }
        >
          <div className="space-y-4">
            <div className="flex justify-between items-center p-4 bg-[var(--bg-card)] rounded-xl border border-[var(--border)]">
              <div>
                <div className="text-sm text-[var(--text-dim)]">Symbol</div>
                <div className="text-xl font-bold">{executeModal.symbol}</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-[var(--text-dim)]">PoP</div>
                <span className="text-lg font-bold text-emerald-400">{((executeModal.pop_score ?? 0) * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 bg-[var(--bg-input)] rounded-lg border border-[var(--border)]">
                <div className="text-[10px] text-[var(--text-dim)] uppercase font-bold">Entry</div>
                <div className="text-sm font-mono font-bold mt-1">{executeModal.entry_price?.toFixed(2)}</div>
              </div>
              <div className="p-3 bg-[var(--bg-input)] rounded-lg border border-[var(--border)]">
                <div className="text-[10px] text-emerald-400 uppercase font-bold">Target</div>
                <div className="text-sm font-mono font-bold mt-1 text-emerald-400">{executeModal.target_price?.toFixed(2)}</div>
                <div className="text-[9px] text-emerald-300/60 mt-0.5 font-medium">S/R Resistance Clamp</div>
              </div>
              <div className="p-3 bg-[var(--bg-input)] rounded-lg border border-[var(--border)]">
                <div className="text-[10px] text-red-400 uppercase font-bold">Stop-Loss</div>
                <div className="text-sm font-mono font-bold mt-1 text-red-400">{executeModal.stoploss_price?.toFixed(2)}</div>
                <div className="text-[9px] text-red-300/60 mt-0.5 font-medium">ATR + Support Floor</div>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[var(--border)]">
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">R:R Ratio</span>
                <span className="font-mono font-medium">{executeModal.initial_rr_ratio?.toFixed(1)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">FQS Score</span>
                <span className="font-mono">{(executeModal.fqs_score ?? 0).toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Est. Cost</span>
                <span className="font-mono">{((executeModal.execution_cost_pct ?? 0) * 100).toFixed(2)} bps</span>
              </div>
              <div className="flex justify-between text-xs items-center pt-1 border-t border-[var(--border)]">
                <span className="text-[var(--text-muted)] font-semibold">Position Size</span>
                {signalPreview ? (
                  <span className="font-mono font-bold text-[var(--primary)]">
                    {signalPreview.quantity} shares (~₹{signalPreview.position_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                  </span>
                ) : (
                  <span className="font-mono text-[var(--text-dim)]">Calculating…</span>
                )}
              </div>
            </div>

            {/* Dry-run toggle */}
            <div
              className="flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors"
              style={{
                borderColor: dryRun ? 'var(--info)' : 'var(--danger)',
                backgroundColor: dryRun ? 'rgba(59,130,246,0.05)' : 'rgba(239,68,68,0.05)',
              }}
              onClick={() => setDryRun(!dryRun)}
            >
              <div>
                <div className="text-sm font-medium">{dryRun ? 'Paper Trade (Dry Run)' : 'Live Execution'}</div>
                <div className="text-[11px] text-[var(--text-muted)]">
                  {dryRun ? 'No real orders will be placed' : 'Will place real limit-chase BUY + GTT OCO on Zerodha'}
                </div>
              </div>
              {dryRun
                ? <ToggleLeft size={28} className="text-[var(--info)]" />
                : <ToggleRight size={28} className="text-[var(--danger)]" />
              }
            </div>
          </div>
        </Modal>
      )}

      {/* Pattern Chart Modal */}
      {selectedSignal && (
        <SignalPatternModal signal={selectedSignal} onClose={() => setSelectedSignal(null)} />
      )}

      {/* Stock Selector Modal */}
      <Modal
        open={showStockSelector}
        onClose={() => { setShowStockSelector(false); setStockSearchQuery(''); }}
        title="Select Stocks for Signal Generation"
        description="Choose specific stocks. Leave all unchecked to generate for the entire universe."
        size="lg"
        icon={<Filter size={20} className="text-[var(--primary)]" />}
        footer={
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-[var(--text-dim)]">
              {selectedStockIds.size > 0
                ? `${selectedStockIds.size} of ${universeStocks?.length ?? 0} stocks selected`
                : `All ${universeStocks?.length ?? 0} stocks will be used`}
            </span>
            <div className="flex gap-2">
              <Button variant="secondary" size="sm" onClick={() => setSelectedStockIds(new Set())}>
                Clear All
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setSelectedStockIds(new Set(universeStocks?.map((s: any) => s.id) ?? []))}
              >
                Select All
              </Button>
              <Button variant="primary" size="sm" onClick={() => { setShowStockSelector(false); setStockSearchQuery(''); }}>
                Done
              </Button>
            </div>
          </div>
        }
      >
        <div className="space-y-3">
          <input
            type="text"
            value={stockSearchQuery}
            onChange={(e) => setStockSearchQuery(e.target.value)}
            placeholder="Search symbol..."
            className="w-full h-9 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
          />
          <div className="max-h-[380px] overflow-y-auto space-y-1 pr-1">
            {(universeStocks ?? [])
              .filter((s: any) => !stockSearchQuery || String(s.symbol ?? '').toLowerCase().includes(stockSearchQuery.toLowerCase()))
              .map((s: any) => (
                <label
                  key={s.id}
                  className="flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)] hover:bg-[var(--bg-hover)] cursor-pointer transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selectedStockIds.has(s.id)}
                    onChange={(e) => {
                      setSelectedStockIds(prev => {
                        const next = new Set(prev);
                        if (e.target.checked) next.add(s.id); else next.delete(s.id);
                        return next;
                      });
                    }}
                    className="w-4 h-4 rounded border-[var(--border)] accent-[var(--primary)] cursor-pointer"
                  />
                  <span className="text-sm font-medium">{s.symbol}</span>
                  {s.sector && <span className="text-xs text-[var(--text-dim)] ml-auto">{s.sector}</span>}
                </label>
              ))}
          </div>
        </div>
      </Modal>
    </div>
  );
}

// ─── Pattern Chart Modal ──────────────────────────────────────────────────────
function SignalPatternModal({ signal, onClose }: { signal: any; onClose: () => void }) {
  const predictionDate = signal.signal_date as string;
  const startDt = new Date(predictionDate + 'T00:00:00');
  startDt.setDate(startDt.getDate() - 90);
  const startDate = startDt.toISOString().split('T')[0];
  const endDate = new Date().toISOString().split('T')[0];

  const { data: ohlcv, isLoading: loadingOhlcv } = useOhlcv(signal.stock_id, 'day', startDate, endDate);
  const { data: indicators, isLoading: loadingInd } = useIndicators(signal.stock_id, 'day', startDate, endDate);

  const chartIndicators: IndicatorSeries[] = useMemo(() => {
    const res: IndicatorSeries[] = [];
    if (indicators) {
      const sma50 = indicators.map((d: any) => ({ time: d.date, value: d.sma_50 })).filter((d: any) => d.value != null);
      const sma200 = indicators.map((d: any) => ({ time: d.date, value: d.sma_200 })).filter((d: any) => d.value != null);
      if (sma50.length)  res.push({ name: 'SMA 50',  color: '#3b82f6', data: sma50 });
      if (sma200.length) res.push({ name: 'SMA 200', color: '#eab308', data: sma200 });
    }
    return res;
  }, [indicators]);

  const chartOhlcv = useMemo(() => (ohlcv ?? []).map((d: any) => ({
    time: d.date, open: d.open, high: d.high, low: d.low, close: d.close,
  })), [ohlcv]);

  const priceLevels = useMemo(() => [
    ...(signal.entry_price ? [{ price: signal.entry_price, color: '#94a3b8', title: 'Entry', lineStyle: 1 }] : []),
    ...(signal.target_price ? [{ price: signal.target_price, color: '#10b981', title: 'Target', lineStyle: 2 }] : []),
    ...((signal.current_stoploss ?? signal.stoploss_price) ? [{ price: (signal.current_stoploss ?? signal.stoploss_price) as number, color: '#ef4444', title: 'Stop-Loss', lineStyle: 2 }] : []),
  ] as PriceLevel[], [signal.entry_price, signal.target_price, signal.current_stoploss, signal.stoploss_price]);

  const isShort = signal.target_price < signal.entry_price;

  return (
    <Modal open onClose={onClose} title={`${signal.symbol} — Pattern Analysis`} size="2xl">
      <div className="space-y-5">
        {/* Signal summary row */}
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Signal Date</div>
            <div className="text-lg font-mono font-bold">{signal.signal_date}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Direction</div>
            <Badge color={isShort ? 'red' : 'green'} >{isShort ? 'SHORT' : 'LONG'}</Badge>
          </div>
          <div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Entry</div>
            <div className="font-mono font-bold">{signal.entry_price?.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-[10px] text-emerald-400 uppercase tracking-wider font-bold">Target</div>
            <div className="font-mono font-bold text-emerald-400">{signal.target_price?.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-[10px] text-red-400 uppercase tracking-wider font-bold">Stop-Loss</div>
            <div className="font-mono font-bold text-red-400">{(signal.current_stoploss ?? signal.stoploss_price)?.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">R:R</div>
            <div className="font-mono font-bold">{(signal.current_rr_ratio ?? signal.initial_rr_ratio)?.toFixed(1)}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">PoP</div>
            <div className="font-mono font-bold text-[var(--primary)]">{((signal.pop_score ?? 0) * 100).toFixed(0)}%</div>
          </div>
        </div>

        {/* Chart */}
        <div className="relative bg-[#0b0b14] rounded-2xl border border-[var(--border)] overflow-hidden" style={{ minHeight: 480 }}>
          {loadingOhlcv || loadingInd ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <div className="w-8 h-8 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-[var(--text-dim)] animate-pulse">Loading price data…</span>
            </div>
          ) : chartOhlcv.length > 0 ? (
            <LightweightCandleChart
              ohlcv={chartOhlcv}
              indicators={chartIndicators}
              height={480}
              verticalLineDate={predictionDate}
              priceLevels={priceLevels}
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[var(--text-dim)]">
              <ShieldAlert size={32} strokeWidth={1} />
              <span className="text-xs">No OHLCV data found for this instrument.</span>
            </div>
          )}
        </div>

        {/* LSTM details */}
        {(signal.lstm_mu != null || signal.knn_win_rate != null) && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {[
              { label: 'LSTM μ return', value: signal.lstm_mu != null ? `${(signal.lstm_mu * 100).toFixed(2)}%` : '—' },
              { label: 'LSTM σ', value: signal.lstm_sigma != null ? `${(signal.lstm_sigma * 100).toFixed(2)}%` : '—' },
              { label: 'KNN median rtn', value: signal.knn_median_return != null ? `${(signal.knn_median_return * 100).toFixed(2)}%` : '—' },
              { label: 'KNN win rate', value: signal.knn_win_rate != null ? `${(signal.knn_win_rate * 100).toFixed(0)}%` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="p-3 rounded-lg bg-[var(--bg-input)] border border-[var(--border)]">
                <div className="text-[10px] text-[var(--text-dim)] uppercase font-bold">{label}</div>
                <div className="font-mono font-bold mt-1">{value}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}
