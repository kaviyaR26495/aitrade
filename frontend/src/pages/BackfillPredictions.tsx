import { useState, useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Card, Button, Input, Select, Badge, StatCard, PageHeader, Checkbox } from '../components/ui';
import { useBackfillStatus, useBackfillCoverage, useStartBackfill, useStopBackfill } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Play, Square, RefreshCw, Database, CalendarDays, CheckCircle, AlertCircle, Clock } from 'lucide-react';

// ── helpers ──────────────────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  return `${m}m ${s}s`;
}

// ── component ─────────────────────────────────────────────────────────────────

export default function BackfillPredictions() {
  const { addNotification } = useAppStore();
  const qc = useQueryClient();

  // ── form state ──
  const today = new Date().toISOString().split('T')[0];
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2026-03-31');
  const [selectedConfigId, setSelectedConfigId] = useState<string>('');
  const [overrideExisting, setOverrideExisting] = useState(false);

  // ── data ──
  const { data: coverage, refetch: refetchCoverage } = useBackfillCoverage();
  const startBackfill = useStartBackfill();
  const stopBackfill = useStopBackfill();

  // Poll status only when job is running/stopping
  const [isPolling, setIsPolling] = useState(false);
  const { data: status, refetch: refetchStatus } = useBackfillStatus(isPolling);

  // Log auto-scroll
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [status?.log]);

  // Start polling when we know a job is running
  useEffect(() => {
    if (status?.status === 'running' || status?.status === 'stopping') {
      setIsPolling(true);
    } else if (status?.status === 'completed' || status?.status === 'failed' || status?.status === 'stopped') {
      setIsPolling(false);
      refetchCoverage();
    }
  }, [status?.status, refetchCoverage]);

  // Initialise polling on mount so we pick up a job that was started earlier in the session
  useEffect(() => {
    refetchStatus().then(r => {
      const s = r.data?.status;
      if (s === 'running' || s === 'stopping') setIsPolling(true);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const configs = coverage?.configs ?? [];
  const configOptions = [
    { value: '', label: 'Latest (auto)' },
    ...configs.map((c: any) => ({ value: String(c.id), label: `Config #${c.id} — ${c.name}` })),
  ];

  const isRunning = status?.status === 'running' || status?.status === 'stopping';

  const handleStart = async () => {
    try {
      await startBackfill.mutateAsync({
        start_date: startDate,
        end_date: endDate,
        ensemble_config_id: selectedConfigId ? Number(selectedConfigId) : null,
        override_existing: overrideExisting,
      });
      setIsPolling(true);
      addNotification({ type: 'success', message: 'Backfill started.' });
    } catch (e: any) {
      addNotification({ type: 'error', message: e?.response?.data?.detail ?? 'Failed to start backfill.' });
    }
  };

  const handleStop = async () => {
    try {
      await stopBackfill.mutateAsync();
      addNotification({ type: 'info', message: 'Stop signal sent — current date will finish then job will halt.' });
    } catch (e: any) {
      addNotification({ type: 'error', message: e?.response?.data?.detail ?? 'Failed to stop backfill.' });
    }
  };

  // ── coverage stats ──
  const totalCoveredDays = (coverage?.coverage ?? []).reduce((sum: number, r: any) => sum + r.distinct_days, 0);
  const allMinDate = (coverage?.coverage ?? []).reduce((min: string | null, r: any) =>
    r.min_date && (!min || r.min_date < min) ? r.min_date : min, null as string | null);
  const allMaxDate = (coverage?.coverage ?? []).reduce((max: string | null, r: any) =>
    r.max_date && (!max || r.max_date > max) ? r.max_date : max, null as string | null);

  // ── status badge ──
  const statusBadge = (s?: string) => {
    if (!s || s === 'idle') return <Badge variant="gray">Idle</Badge>;
    if (s === 'running') return <Badge variant="blue">Running</Badge>;
    if (s === 'stopping') return <Badge variant="yellow">Stopping…</Badge>;
    if (s === 'completed') return <Badge variant="green">Completed</Badge>;
    if (s === 'stopped') return <Badge variant="yellow">Stopped</Badge>;
    if (s === 'failed') return <Badge variant="red">Failed</Badge>;
    return <Badge variant="gray">{s}</Badge>;
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Backfill Predictions"
        description="Generate historical ensemble predictions for past trading days so backtesting covers the full date range."
      />

      {/* ── Stats row ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Ensemble Configs"
          value={String(configs.length)}
          icon={<Database size={16} />}
        />
        <StatCard
          label="Total Days Covered"
          value={String(totalCoveredDays)}
          icon={<CalendarDays size={16} />}
        />
        <StatCard
          label="Earliest Prediction"
          value={allMinDate ?? '—'}
          icon={<CalendarDays size={16} />}
        />
        <StatCard
          label="Latest Prediction"
          value={allMaxDate ?? '—'}
          icon={<CalendarDays size={16} />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── Left: configuration + run ─────────────────────────── */}
        <Card title="Run Backfill">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Input label="Start Date" type="date" value={startDate} onChange={setStartDate} />
              <Input label="End Date" type="date" value={endDate} onChange={setEndDate} max={today} />
            </div>

            <Select
              label="Ensemble Config"
              value={selectedConfigId}
              onChange={setSelectedConfigId}
              options={configOptions}
            />

            <div className="p-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)]">
              <Checkbox
                checked={overrideExisting}
                onChange={setOverrideExisting}
                label="Override existing predictions"
                description="When checked, dates that already have predictions for the selected config will be regenerated (overwritten). Leave unchecked to only fill missing dates."
              />
            </div>

            <div className="flex gap-2 pt-1">
              <Button
                onClick={handleStart}
                loading={startBackfill.isPending}
                disabled={isRunning || !startDate || !endDate}
                className="flex-1"
              >
                <Play size={14} className="mr-1.5" />
                {isRunning ? 'Running…' : 'Start Backfill'}
              </Button>
              {isRunning && (
                <Button variant="danger" onClick={handleStop} loading={stopBackfill.isPending}>
                  <Square size={14} className="mr-1.5" />
                  Stop
                </Button>
              )}
            </div>

            {/* ── live progress ── */}
            {status && status.status !== 'idle' && (
              <div className="space-y-3 pt-2 border-t border-[var(--border)]">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-[var(--text)]">Job Status</span>
                  {statusBadge(status.status)}
                </div>

                <div className="space-y-1.5 text-sm text-[var(--text-muted)]">
                  <div className="flex justify-between">
                    <span>Progress</span>
                    <span className="font-mono text-[var(--text)]">
                      {status.done} / {status.total} dates ({status.progress_pct}%)
                    </span>
                  </div>
                  {status.skipped > 0 && (
                    <div className="flex justify-between">
                      <span>Skipped (already done)</span>
                      <span className="font-mono text-[var(--text)]">{status.skipped}</span>
                    </div>
                  )}
                  {status.errors > 0 && (
                    <div className="flex justify-between text-rose-400">
                      <span>Errors</span>
                      <span className="font-mono">{status.errors}</span>
                    </div>
                  )}
                  {status.current_date && status.status === 'running' && (
                    <div className="flex justify-between">
                      <span>Processing</span>
                      <span className="font-mono text-[var(--text)]">{status.current_date}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span>Elapsed</span>
                    <span className="font-mono text-[var(--text)]">{formatDuration(status.elapsed_seconds ?? 0)}</span>
                  </div>
                  {status.eta_seconds != null && (
                    <div className="flex justify-between">
                      <span>ETA</span>
                      <span className="font-mono text-[var(--text)]">{formatDuration(status.eta_seconds)}</span>
                    </div>
                  )}
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 rounded-full bg-[var(--bg-input)] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${status.progress_pct}%`,
                      background: status.status === 'completed'
                        ? 'var(--color-success, #22c55e)'
                        : status.status === 'failed' || status.status === 'stopped'
                          ? 'var(--color-danger, #ef4444)'
                          : 'var(--color-primary, #6366f1)',
                    }}
                  />
                </div>

                {/* Config info */}
                {status.ensemble_config_id && (
                  <p className="text-xs text-[var(--text-muted)]">
                    Config #{status.ensemble_config_id} · {status.start_date} → {status.end_date}
                    {status.override_existing ? ' · Override ON' : ' · Skip existing'}
                  </p>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* ── Right: coverage table ─────────────────────────────── */}
        <Card
          title="Prediction Coverage"
          actions={
            <Button variant="ghost" onClick={() => { refetchCoverage(); }} className="h-7 px-2 text-xs">
              <RefreshCw size={12} className="mr-1" /> Refresh
            </Button>
          }
        >
          {!coverage ? (
            <p className="text-sm text-[var(--text-muted)] py-4 text-center">Loading…</p>
          ) : coverage.coverage.length === 0 ? (
            <p className="text-sm text-[var(--text-muted)] py-4 text-center">No predictions in database yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] text-[var(--text-muted)]">
                    <th className="text-left pb-2 font-medium">Config</th>
                    <th className="text-right pb-2 font-medium">From</th>
                    <th className="text-right pb-2 font-medium">To</th>
                    <th className="text-right pb-2 font-medium">Days</th>
                    <th className="text-right pb-2 font-medium">Rows</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {coverage.coverage.map((row: any) => {
                    const cfg = configs.find((c: any) => c.id === row.ensemble_config_id);
                    return (
                      <tr key={row.ensemble_config_id} className="text-[var(--text)]">
                        <td className="py-1.5 pr-2">
                          <span className="text-xs font-mono bg-[var(--bg-input)] px-1.5 py-0.5 rounded">
                            #{row.ensemble_config_id}
                          </span>
                          {cfg && (
                            <span className="ml-2 text-xs text-[var(--text-muted)] truncate max-w-[120px] inline-block align-middle">
                              {cfg.name}
                            </span>
                          )}
                        </td>
                        <td className="py-1.5 text-right font-mono text-xs">{row.min_date ?? '—'}</td>
                        <td className="py-1.5 text-right font-mono text-xs">{row.max_date ?? '—'}</td>
                        <td className="py-1.5 text-right font-mono">{row.distinct_days.toLocaleString()}</td>
                        <td className="py-1.5 text-right font-mono text-[var(--text-muted)]">
                          {row.total_rows.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Gap indicator */}
          {allMinDate && allMinDate > '2025-01-01' && (
            <div className="mt-3 flex items-start gap-2 p-3 rounded-[var(--radius-sm)] bg-amber-500/10 border border-amber-500/30 text-sm text-amber-300">
              <AlertCircle size={15} className="flex-shrink-0 mt-0.5" />
              <span>
                Predictions start <strong>{allMinDate}</strong> — dates before this will show no signals in backtesting.
                Run a backfill to extend coverage.
              </span>
            </div>
          )}
          {allMinDate && allMinDate <= '2025-01-01' && (
            <div className="mt-3 flex items-center gap-2 p-3 rounded-[var(--radius-sm)] bg-emerald-500/10 border border-emerald-500/30 text-sm text-emerald-300">
              <CheckCircle size={15} className="flex-shrink-0" />
              <span>Good coverage — predictions available from {allMinDate}.</span>
            </div>
          )}
        </Card>
      </div>

      {/* ── Log output ────────────────────────────────────────────── */}
      {status && status.log && status.log.length > 0 && (
        <Card
          title="Job Log"
          actions={
            <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
              {isRunning && <Clock size={12} className="animate-spin" />}
              {isRunning ? 'Live' : 'Finished'}
            </div>
          }
        >
          <div
            ref={logRef}
            className="font-mono text-xs text-[var(--text-muted)] bg-[var(--bg-input)] rounded p-3 h-64 overflow-y-auto space-y-0.5"
          >
            {status.log.map((line: string, i: number) => (
              <div
                key={i}
                className={
                  line.startsWith('✓')
                    ? 'text-emerald-400'
                    : line.startsWith('✗') || line.includes('FAILED') || line.includes('ERROR')
                      ? 'text-rose-400'
                      : line.startsWith('Backfill complete')
                        ? 'text-emerald-300 font-semibold'
                        : ''
                }
              >
                {line}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
