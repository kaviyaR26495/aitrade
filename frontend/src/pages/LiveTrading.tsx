import { useMemo, useState, useRef, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Card, Button, Badge, StatCard, EmptyState, PageHeader, Table, Modal, Tooltip, SkeletonTable, type TableColumn } from '../components/ui';
import { useSignals, useGenerateSignals, useTrainMhLstm, useExecuteSignal, useOrders, useUniverseStocks, usePredictionJob, useCancelPredictionJob, useSignalPreview, useOhlcv, useIndicators, useDeleteSignal, useDeleteSignalsByDate } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Crosshair, Play, Shield, ShieldAlert, XCircle, Loader2, Filter, Zap, ToggleLeft, ToggleRight, BrainCircuit, CalendarDays, BarChart2, ChevronLeft, ChevronRight, Trash } from 'lucide-react';
import LightweightCandleChart, { type IndicatorSeries, type PriceLevel } from '../components/LightweightCandleChart';

type StatusFilter = 'ALL' | 'pending' | 'active' | 'target_hit' | 'sl_hit' | 'expired';

const STATUS_COLORS: Record<string, 'green' | 'blue' | 'yellow' | 'red' | 'gray'> = {
  pending: 'yellow',
  active: 'blue',
  target_hit: 'green',
  sl_hit: 'red',
  expired: 'gray',
};

const REGIME_COLORS: Record<number, string> = {
  0: '#22c55e',
  1: '#a3e635',
  2: '#ef4444',
  3: '#991b1b',
  4: '#6b7280',
  5: '#eab308',
};

const REGIME_LABELS: Record<number, string> = {
  0: 'Bull+LowVol',
  1: 'Bull+HighVol',
  2: 'Bear+LowVol',
  3: 'Bear+HighVol',
  4: 'Neutral+LowVol',
  5: 'Neutral+HighVol',
};

export default function LiveTrading() {
  const { addNotification } = useAppStore();
  const [interval] = useState('day');
  // Date range filter — both start as empty so all recent signals load on mount.
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [isDateRangeOpen, setIsDateRangeOpen] = useState(false);
  const [calendarMonth, setCalendarMonth] = useState<Date>(() => {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), 1);
  });
  const dateRangeRef = useRef<HTMLDivElement>(null);
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
  const deleteSignal = useDeleteSignal();
  const deleteSignalsByDate = useDeleteSignalsByDate();
  const { data: signalPreview } = useSignalPreview(executeModal?.id ?? null);

  useEffect(() => {
    if (!isDateRangeOpen) return;

    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!dateRangeRef.current?.contains(target)) {
        setIsDateRangeOpen(false);
      }
    };

    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [isDateRangeOpen]);

  useEffect(() => {
    if (!isDateRangeOpen) return;
    const seed = dateFrom || dateTo;
    if (!seed) return;
    const parsed = new Date(`${seed}T00:00:00`);
    if (!Number.isNaN(parsed.getTime())) {
      setCalendarMonth(new Date(parsed.getFullYear(), parsed.getMonth(), 1));
    }
  }, [isDateRangeOpen, dateFrom, dateTo]);

  useEffect(() => {
    if (job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled') {
      if (job.status === 'completed') {
        const capturedDate = pendingGenerateDateRef.current;
        pendingGenerateDateRef.current = null;
        if (capturedDate) {
          setDateFrom(capturedDate);
          setDateTo(capturedDate);
        } else {
          setDateFrom('');
          setDateTo('');
        }
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

  const dateRangeLabel = dateFrom && dateTo
    ? `${dateFrom} to ${dateTo}`
    : dateFrom
      ? `From ${dateFrom}`
      : dateTo
        ? `Until ${dateTo}`
        : 'Date Range';

  const monthLabel = calendarMonth.toLocaleString('en-US', { month: 'long', year: 'numeric' });
  const firstWeekday = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), 1).getDay();
  const daysInMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 0).getDate();
  const monthDays = Array.from({ length: daysInMonth }, (_, i) => {
    const d = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), i + 1);
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
  });

  const handleRangeDateClick = (picked: string) => {
    if (!dateFrom || (dateFrom && dateTo)) {
      setDateFrom(picked);
      setDateTo('');
      return;
    }

    if (picked < dateFrom) {
      setDateFrom(picked);
      setDateTo('');
    } else {
      setDateTo(picked);
    }
  };

  const signalColumns: TableColumn<any>[] = [
    {
      key: 'symbol',
      label: 'Symbol',
      tooltip: 'Direction & Stock ticker symbol — click row to view chart',
      sortable: true,
      render: (s) => {
        const isShort = s.target_price < s.entry_price;
        return (
          <div className="flex items-center gap-2">
            <span className="font-medium text-[var(--text)]">{s.symbol}</span>
            <Badge color={isShort ? 'red' : 'green'} size="sm">{isShort ? 'SHORT' : 'LONG'}</Badge>
          </div>
        );
      },
    },
    {
      key: 'signal_date',
      label: 'Date',
      tooltip: 'Date when generated & days since active',
      align: 'center',
      mono: true,
      sortable: true,
      sortValue: (s) => {
        const v = s.signal_date;
        if (!v) return 0;
        const ts = Date.parse(String(v));
        return Number.isNaN(ts) ? 0 : ts;
      },
      render: (s) => {
        const v = s.signal_date;
        if (!v) return <span className="text-xs text-[var(--text-dim)]">—</span>;
        const dateText = String(v).includes('T') ? String(v).split('T')[0] : String(v);
        const parts = dateText.split('-');
        const month = parts[1];
        const day = parts[2];
        const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        const formatDt = month ? `${months[parseInt(month)-1]} ${day}` : dateText;
        
        const days = s.days_since_signal ?? 0;
        const stale = days > 15;
        
        return (
          <div className="flex flex-col items-center">
            <span className="tabular-nums text-[11px] text-[var(--text-muted)]">{formatDt}</span>
            <span className={`text-[10px] tabular-nums ${stale ? 'text-rose-400 font-bold' : 'text-[var(--text-dim)]'}`}>
              {days}d
            </span>
          </div>
        );
      },
    },
    {
      key: 'regime',
      label: 'Regime',
      tooltip: 'Market regime (Trend + Volatility)',
      align: 'center',
      sortable: true,
      sortValue: (s) => s.regime_id ?? -1,
      render: (s) => {
        const regimeId = s.regime_id ?? -1;
        if (regimeId < 0) return <span className="text-xs text-[var(--text-dim)]">—</span>;
        const label = REGIME_LABELS[regimeId] || `R${regimeId}`;
        const color = REGIME_COLORS[regimeId] || '#6b7280';
        return (
          <span 
            className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded shadow-sm whitespace-nowrap" 
            style={{ backgroundColor: `${color}15`, color: color, border: `1px solid ${color}30` }}
          >
            {label}
          </span>
        );
      },
    },
    {
      key: 'levels',
      label: 'Levels (Entry → Tgt)',
      tooltip: 'Entry → Target (Stop Loss)',
      align: 'right',
      mono: true,
      render: (s) => {
        const sl = s.current_stoploss ?? s.stoploss_price;
        return (
          <div className="flex flex-col items-end whitespace-nowrap leading-tight gap-0.5">
            <div className="text-[11px]">
              <span className="text-[var(--text-muted)]">₹{s.entry_price?.toFixed(2) ?? '—'}</span>
              <span className="text-[var(--text-dim)] mx-1">→</span>
              <span className="text-emerald-400 font-medium">₹{s.target_price?.toFixed(2) ?? '—'}</span>
            </div>
            <div className="text-[10px] text-red-500/80">
              SL ₹{sl?.toFixed(2) ?? '—'}
            </div>
          </div>
        );
      },
    },
    {
      key: 'initial_rr_ratio',
      label: 'R:R',
      tooltip: 'Risk-Reward ratio',
      align: 'center',
      mono: true,
      sortable: true,
      sortValue: (s) => s.current_rr_ratio ?? s.initial_rr_ratio,
      render: (s) => {
        const rr = s.current_rr_ratio ?? s.initial_rr_ratio;
        return (
          <span className={`tabular-nums font-semibold ${rr >= 2.5 ? 'text-emerald-400' : rr >= 1.5 ? 'text-amber-400' : 'text-red-400'}`}>
            {rr?.toFixed(1) ?? '—'}
          </span>
        );
      },
    },
    {
      key: 'pop_score',
      label: 'PoP%',
      tooltip: 'Probability of Profit (Hover for FQS)',
      align: 'center',
      sortable: true,
      render: (s) => (
        <Tooltip content={`FQS: ${(s.fqs_score ?? 0).toFixed(2)}`} side="top">
          <span className={`font-bold tabular-nums ${s.pop_score >= 0.7 ? 'text-emerald-400' : s.pop_score >= 0.55 ? 'text-amber-400' : 'text-[var(--text-muted)]'}`}>
            {((s.pop_score ?? 0) * 100).toFixed(0)}%
          </span>
        </Tooltip>
      ),
    },
    {
      key: 'status',
      label: 'Status',
      tooltip: 'Signal lifecycle status & trailing stop info',
      align: 'center',
      sortable: true,
      render: (s) => (
        <div className="flex flex-col items-center gap-1">
          <Badge color={STATUS_COLORS[s.status] ?? 'gray'} size="sm">{s.status}</Badge>
          {s.is_trailing_active && (
            <span className="text-[9px] text-blue-400 font-bold uppercase tracking-wider">
              Trail ↑{s.trailing_updates_count ?? 0}
            </span>
          )}
        </div>
      ),
    },
    {
      key: 'execute',
      label: 'Actions',
      align: 'right',
      stopPropagation: true,
      render: (s) => (
        <div className="flex items-center justify-end gap-1">
          {s.status === 'pending' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                setExecuteModal(s);
              }}
              className="h-7 px-2 text-[11px] font-semibold text-[var(--primary)] bg-[var(--primary)]/10 hover:bg-[var(--primary)] hover:text-white transition-all shadow-sm rounded-sm"
            >
              <Zap size={11} className="mr-1" />
              Exec
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              if (confirm('Delete this prediction signal?')) {
                deleteSignal.mutate(s.id);
              }
            }}
            className="h-7 px-1.5 text-[10px] text-[var(--text-muted)] hover:text-red-400 hover:bg-red-500/10 shrink-0"
            title="Delete signal"
            loading={deleteSignal.isPending && deleteSignal.variables === s.id}
          >
            {!(deleteSignal.isPending && deleteSignal.variables === s.id) && <Trash size={12} />}
          </Button>
        </div>
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
          <div ref={dateRangeRef} className="relative">
            <button
              type="button"
              onClick={() => setIsDateRangeOpen((v) => !v)}
              className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] shadow-sm text-xs font-medium text-[var(--text)] hover:bg-[var(--bg-hover)] transition-colors"
              title="Select date range"
            >
              <CalendarDays size={13} className="text-[var(--text-dim)] shrink-0" />
              <span className="truncate max-w-[180px] text-left">{dateRangeLabel}</span>
            </button>

            {isDateRangeOpen && (
              <div className="absolute top-full mt-2 left-0 z-40 w-[318px] p-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] shadow-xl">
                <div className="flex items-center justify-between mb-2">
                  <button
                    type="button"
                    onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1))}
                    className="h-7 w-7 inline-flex items-center justify-center rounded border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
                    title="Previous month"
                  >
                    <ChevronLeft size={14} />
                  </button>
                  <div className="text-xs font-semibold text-[var(--text)]">{monthLabel}</div>
                  <button
                    type="button"
                    onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1))}
                    className="h-7 w-7 inline-flex items-center justify-center rounded border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
                    title="Next month"
                  >
                    <ChevronRight size={14} />
                  </button>
                </div>

                <div className="grid grid-cols-7 gap-1 text-[10px] text-[var(--text-dim)] uppercase font-semibold mb-1">
                  {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((d) => (
                    <div key={d} className="h-6 flex items-center justify-center">{d}</div>
                  ))}
                </div>

                <div className="grid grid-cols-7 gap-1">
                  {Array.from({ length: firstWeekday }).map((_, idx) => (
                    <div key={`pad-${idx}`} className="h-8" />
                  ))}

                  {monthDays.map((d) => {
                    const dayNum = Number(d.slice(-2));
                    const isStart = dateFrom === d;
                    const isEnd = dateTo === d;
                    const inRange = !!(dateFrom && dateTo && d >= dateFrom && d <= dateTo);
                    const active = isStart || isEnd;

                    return (
                      <button
                        key={d}
                        type="button"
                        onClick={() => handleRangeDateClick(d)}
                        className={`h-8 rounded text-xs font-medium transition-colors ${
                          active
                            ? 'bg-[var(--primary)] text-black'
                            : inRange
                              ? 'bg-[var(--primary-dim)]/30 text-[var(--text)]'
                              : 'bg-[var(--bg-input)] text-[var(--text)] hover:bg-[var(--bg-hover)]'
                        }`}
                        title={d}
                      >
                        {dayNum}
                      </button>
                    );
                  })}
                </div>

                <div className="mt-2 text-[10px] text-[var(--text-dim)]">
                  {!dateFrom && !dateTo && 'Pick a start date, then an end date.'}
                  {dateFrom && !dateTo && `Start: ${dateFrom} (select end date)`}
                  {dateFrom && dateTo && `Range: ${dateFrom} to ${dateTo}`}
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => { setDateFrom(''); setDateTo(''); }}
                    className="text-xs text-[var(--text-dim)] hover:text-[var(--text)] transition-colors"
                  >
                    Clear range
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsDateRangeOpen(false)}
                    className="text-xs px-2 py-1 rounded border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
                  >
                    Done
                  </button>
                </div>
              </div>
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
          
          <Button
            variant="danger"
            onClick={() => {
              if (confirm(`Are you sure you want to delete all pending signals for ${generateDate}?`)) {
                deleteSignalsByDate.mutate(generateDate);
              }
            }}
            icon={<Trash size={16} />}
            loading={deleteSignalsByDate.isPending}
            title={`Delete all pending signals for ${generateDate}`}
          >
            Delete
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
            compact={true}
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
