/**
 * AutoPilotPipeline — One-Click Multi-Stock Training Pipeline UI
 *
 * Reads the confirmed stock universe from global state, starts the
 * backend pipeline via POST /api/pipeline/start, then polls
 * GET /api/pipeline/status/:job_id every 2 s until completion or failure.
 *
 * Pipeline stages:
 *   0  Data Sync & Regime Pooling
 *   1  Offline Pre-training (CQL)
 *   2  Behavioral Cloning (BC)
 *   3  Online Fine-Tuning (AttentionPPO)
 *   4  Ensemble Distillation (KNN + LSTM)
 *   5  Ready for Live Trading
 */
import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Rocket, CheckCircle2, Loader2, AlertCircle, Circle,
  ChevronRight, X, ExternalLink, RefreshCw, Layers,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { useStartPipeline, usePipelineStatus, useUniverse } from '../hooks/useApi';
import { listUniverseStocks } from '../services/api';
import type { PipelineStatus, PipelineStageStatus } from '../services/api';
import { Button } from './ui';

// ── Stage definitions (used as fallback when backend hasn't replied yet) ──────

const STAGE_DEFS: { name: string; label: string; description: string }[] = [
  {
    name: 'data_sync',
    label: 'Data Sync & Regime Pooling',
    description: 'Fetch OHLCV history, compute indicators, and assign market regimes for every stock.',
  },
  {
    name: 'cql_pretrain',
    label: 'Offline Pre-training (CQL)',
    description: 'Train a conservative offline RL policy on historical transitions to avoid risky exploration.',
  },
  {
    name: 'bc_warmup',
    label: 'Behavioral Cloning (BC)',
    description: 'Align the PPO actor to the CQL baseline using supervised imitation before online training.',
  },
  {
    name: 'ppo_finetune',
    label: 'Online Fine-Tuning (AttentionPPO)',
    description: 'Explore and refine the policy in simulated markets with attention-based feature extraction.',
  },
  {
    name: 'ensemble_distill',
    label: 'Ensemble Distillation (KNN + LSTM)',
    description: 'Compress the RL policy into a fast KNN + LSTM ensemble ready for real-time inference.',
  },
  {
    name: 'backtest',
    label: 'Backtest & Validation',
    description: 'Simulate the trained ensemble on historical data for up to 5 stocks. Metrics are saved in the Backtest page.',
  },
  {
    name: 'ready',
    label: 'Ready for Live Trading',
    description: 'All models validated. The ensemble is deployed and awaiting execution signals.',
  },
];

// ── Sub-components ────────────────────────────────────────────────────────────

function StageRow({
  index,
  def,
  stage,
  isCurrent,
  isComplete,
  isFailed,
  isPending,
}: {
  index: number;
  def: (typeof STAGE_DEFS)[number];
  stage?: PipelineStageStatus;
  isCurrent: boolean;
  isComplete: boolean;
  isFailed: boolean;
  isPending: boolean;
}) {
  const progress = stage?.progress ?? 0;
  const message = stage?.message;

  return (
    <div className="flex items-start gap-4">
      {/* Step indicator */}
      <div className="flex flex-col items-center gap-0 shrink-0">
        <div
          className={`
            relative flex items-center justify-center w-9 h-9 rounded-full border-2 transition-all duration-500
            ${isComplete
              ? 'bg-emerald-500/15 border-emerald-500 text-emerald-400'
              : isFailed
              ? 'bg-rose-500/15 border-rose-500 text-rose-400'
              : isCurrent
              ? 'bg-indigo-500/15 border-indigo-500 text-indigo-400'
              : isPending
              ? 'bg-[var(--bg-hover)] border-[var(--border)] text-[var(--text-dim)]'
              : 'bg-[var(--bg-hover)] border-[var(--border)] text-[var(--text-dim)]'
            }
          `}
        >
          {isComplete ? (
            <CheckCircle2 size={18} className="text-emerald-400" />
          ) : isFailed ? (
            <AlertCircle size={18} className="text-rose-400" />
          ) : isCurrent ? (
            <Loader2 size={18} className="animate-spin text-indigo-400" />
          ) : (
            <span className="text-[11px] font-bold tabular-nums">{index + 1}</span>
          )}
          {/* Pulsing ring for the active step */}
          {isCurrent && (
            <span className="absolute inset-[-4px] rounded-full border border-indigo-500/40 animate-ping" />
          )}
        </div>
        {/* Connector line */}
        {index < STAGE_DEFS.length - 1 && (
          <div
            className={`w-px flex-1 min-h-[28px] mt-1 transition-colors duration-500 ${
              isComplete ? 'bg-emerald-500/40' : 'bg-[var(--border)]'
            }`}
          />
        )}
      </div>

      {/* Stage content */}
      <div className="flex-1 pb-6">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-sm font-semibold transition-colors duration-300 ${
              isComplete
                ? 'text-emerald-400'
                : isFailed
                ? 'text-rose-400'
                : isCurrent
                ? 'text-[var(--text)]'
                : 'text-[var(--text-muted)]'
            }`}
          >
            {def.label}
          </span>
          {isComplete && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-semibold">
              Done
            </span>
          )}
          {isFailed && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 font-semibold">
              Failed
            </span>
          )}
          {isCurrent && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 font-semibold animate-pulse">
              Running
            </span>
          )}
        </div>

        <p className="text-xs text-[var(--text-dim)] leading-relaxed">{def.description}</p>

        {/* Progress bar — only shown for the active stage */}
        {isCurrent && (
          <div className="mt-2.5 space-y-1">
            <div className="flex items-center justify-between text-[10px] text-[var(--text-dim)]">
              {message && <span className="italic truncate max-w-[240px]">{message}</span>}
              <span className="ml-auto font-mono tabular-nums">{progress}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-[var(--bg-active)] overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-700 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Completed bar — full green */}
        {isComplete && (
          <div className="mt-2.5 h-1.5 w-full rounded-full bg-emerald-500/20 overflow-hidden">
            <div className="h-full w-full rounded-full bg-emerald-500/60 transition-all duration-700" />
          </div>
        )}

        {/* Failed error hint */}
        {isFailed && message && (
          <p className="mt-1.5 text-[11px] text-rose-400/80 italic">{message}</p>
        )}
      </div>
    </div>
  );
}

// ── Stock chip ────────────────────────────────────────────────────────────────

function ChipList({ symbols, onRemove }: { symbols: string[]; onRemove?: (s: string) => void }) {
  const MAX_SHOW = 12;
  const shown = symbols.slice(0, MAX_SHOW);
  const overflow = symbols.length - MAX_SHOW;

  return (
    <div className="flex flex-wrap gap-1.5">
      {shown.map((sym) => (
        <span
          key={sym}
          className="inline-flex items-center gap-1 pl-2.5 pr-1.5 py-1 rounded-full
                     text-[11px] font-semibold font-mono
                     bg-[var(--primary-subtle)] text-[var(--primary)]
                     border border-[var(--primary-glow)]"
        >
          {sym}
          {onRemove && (
            <button
              type="button"
              onClick={() => onRemove(sym)}
              className="hover:text-rose-400 transition-colors leading-none ml-0.5"
              aria-label={`Remove ${sym}`}
            >
              <X size={10} />
            </button>
          )}
        </span>
      ))}
      {overflow > 0 && (
        <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold text-[var(--text-dim)] bg-[var(--bg-hover)] border border-[var(--border)]">
          +{overflow} more
        </span>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AutoPilotPipeline() {
  const navigate = useNavigate();
  const {
    pipelineUniverse,
    setPipelineUniverse,
    addNotification,
    activePipelineJobId: jobId,
    setActivePipelineJobId: setJobId,
  } = useAppStore();

  const [startError, setStartError] = useState<string | null>(null);

  const startMutation = useStartPipeline();
  const { data: statusData, isError: statusError } = usePipelineStatus(jobId);
  const { data: universe, isLoading: universeLoading } = useUniverse();

  // Sync pipelineUniverse from backend if it's currently empty
  useEffect(() => {
    if (pipelineUniverse.length === 0 && universe) {
      // Fetch the full resolved list of symbols from the backend
      listUniverseStocks().then((res) => {
        const symbols = (res.data ?? []).map((s: any) => s.symbol);
        if (symbols.length > 0) {
          setPipelineUniverse(symbols);
        }
      }).catch((e) => {
        console.error('Failed to auto-sync universe symbols', e);
      });
    }
  }, [universe, pipelineUniverse.length, setPipelineUniverse]);

  const handleSyncFromBackend = useCallback(() => {
    if (universe?.category === 'custom' && universe.custom_symbols?.length > 0) {
      setPipelineUniverse(universe.custom_symbols);
      addNotification({ type: 'info', message: 'Universe synced from saved configuration.' });
    } else {
      addNotification({ type: 'warning', message: 'No custom universe found on server.' });
    }
  }, [universe, setPipelineUniverse, addNotification]);

  const isRunning = !!jobId && statusData?.status === 'running';
  const isComplete = statusData?.status === 'completed';
  const isFailed = statusData?.status === 'failed';
  const isQueued = statusData?.status === 'queued';
  const hasStarted = !!jobId;

  const currentStage = statusData?.current_stage ?? -1;

  const handleRemoveSymbol = useCallback(
    (sym: string) => {
      setPipelineUniverse(pipelineUniverse.filter((s) => s !== sym));
    },
    [pipelineUniverse, setPipelineUniverse],
  );

  const handleStart = useCallback(() => {
    if (pipelineUniverse.length === 0) return;
    setStartError(null);
    startMutation.mutate(
      { symbols: pipelineUniverse },
      {
        onSuccess: (data) => {
          setJobId(data.job_id);
        },
        onError: (err: any) => {
          const msg =
            err?.response?.data?.detail ??
            'Failed to start pipeline. Is the backend running?';
          setStartError(msg);
          addNotification({ type: 'error', message: msg });
        },
      },
    );
  }, [pipelineUniverse, startMutation, addNotification]);

  const handleReset = useCallback(() => {
    setJobId(null);
    setStartError(null);
  }, []);

  // Build stage rows — merge backend data with local defs
  const stageRows = STAGE_DEFS.map((def, i) => {
    const backendStage = statusData?.stages?.[i];
    const stageStatus = backendStage?.status ?? 'pending';

    return {
      def,
      stage: backendStage,
      isCurrent: hasStarted && i === currentStage && (isRunning || isQueued),
      isComplete: stageStatus === 'completed',
      isFailed: stageStatus === 'failed',
      isPending: !hasStarted || stageStatus === 'pending',
    };
  });

  return (
    <div className="space-y-5">
      {/* ── Universe basket ──────────────────────────────────────── */}
      <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-card)]/60 px-5 py-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-[var(--text-muted)]" />
            <span className="text-[11px] font-bold uppercase tracking-widest text-[var(--text-muted)]">
              Stock Universe
            </span>
            <span className="ml-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--primary-subtle)] text-[var(--primary)] font-semibold">
              {universeLoading ? '...' : pipelineUniverse.length}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {!hasStarted && (
              <button
                type="button"
                onClick={handleSyncFromBackend}
                className="flex items-center gap-1.5 text-[11px] text-[var(--text-dim)] hover:text-[var(--primary)] transition-colors"
                title="Reload from backend saved universe"
              >
                <RefreshCw size={11} className={universeLoading ? 'animate-spin' : ''} />
                Sync
              </button>
            )}
            {!hasStarted && (
              <button
                type="button"
                onClick={() => navigate('/stocks')}
                className="flex items-center gap-1.5 text-[11px] text-[var(--text-dim)] hover:text-[var(--primary)] transition-colors"
              >
                Edit <ChevronRight size={11} />
              </button>
            )}
          </div>
        </div>

        {pipelineUniverse.length === 0 ? (
          <div className="py-6 text-center text-sm text-[var(--text-muted)]">
            <p>No stocks selected.</p>
            <button
              type="button"
              onClick={() => navigate('/stocks')}
              className="mt-2 text-[var(--primary)] hover:underline text-sm font-medium"
            >
              Go to Stock Selector →
            </button>
          </div>
        ) : (
          <ChipList
            symbols={pipelineUniverse}
            onRemove={!hasStarted ? handleRemoveSymbol : undefined}
          />
        )}
      </div>

      {/* ── Start / Reset button ──────────────────────────────────── */}
      {!hasStarted ? (
        <div className="flex flex-col gap-2">
          {startError && (
            <div className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius)] bg-rose-500/10 border border-rose-500/30 text-sm text-rose-400">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{startError}</span>
            </div>
          )}
          <Button
            size="lg"
            onClick={handleStart}
            disabled={pipelineUniverse.length === 0 || startMutation.isPending}
            loading={startMutation.isPending}
            className="w-full h-14 text-base font-bold tracking-wide bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-400 hover:to-violet-500 border-0 shadow-[0_0_30px_rgba(99,102,241,0.3)] disabled:opacity-40"
          >
            <Rocket size={18} />
            Start Universal Training Pipeline
          </Button>
          {pipelineUniverse.length === 0 && (
            <p className="text-center text-xs text-[var(--text-dim)]">
              Select stocks first in{' '}
              <button
                type="button"
                onClick={() => navigate('/stocks')}
                className="text-[var(--primary)] hover:underline"
              >
                Stock Selector
              </button>
            </p>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {isComplete ? (
              <CheckCircle2 size={16} className="text-emerald-400" />
            ) : isFailed ? (
              <AlertCircle size={16} className="text-rose-400" />
            ) : (
              <Loader2 size={16} className="animate-spin text-indigo-400" />
            )}
            <span className={`text-sm font-semibold ${isComplete ? 'text-emerald-400' : isFailed ? 'text-rose-400' : 'text-[var(--text)]'}`}>
              {isComplete ? 'Pipeline complete!' : isFailed ? 'Pipeline failed' : isQueued ? 'Queued…' : 'Pipeline running…'}
            </span>
            <span className="text-xs text-[var(--text-dim)] font-mono">{jobId}</span>
          </div>

          <div className="flex items-center gap-2">
            {(isComplete || isFailed) && (
              <button
                type="button"
                onClick={handleReset}
                className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
              >
                <RefreshCw size={11} />
                Run again
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Pipeline stepper ─────────────────────────────────────── */}
      {(hasStarted || true) && (
        <div
          className={`rounded-[var(--radius-lg)] border px-6 py-5 transition-colors duration-500 ${
            isComplete
              ? 'border-emerald-500/30 bg-emerald-500/5'
              : isFailed
              ? 'border-rose-500/30 bg-rose-500/5'
              : hasStarted
              ? 'border-indigo-500/30 bg-indigo-500/5'
              : 'border-[var(--border)] bg-[var(--bg-card)]/40'
          }`}
        >
          {statusError && (
            <div className="mb-4 flex items-center gap-2 text-xs text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded px-3 py-2">
              <AlertCircle size={12} />
              <span>Status polling error — retrying automatically…</span>
            </div>
          )}

          <div className="space-y-0">
            {stageRows.map((row, i) => (
              <StageRow
                key={i}
                index={i}
                def={row.def}
                stage={row.stage}
                isCurrent={row.isCurrent}
                isComplete={row.isComplete}
                isFailed={row.isFailed}
                isPending={row.isPending}
              />
            ))}
          </div>

          {/* ── Error detail ── */}
          {isFailed && statusData?.error && (
            <div className="mt-2 p-3 rounded bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 font-mono break-all">
              {statusData.error}
            </div>
          )}
        </div>
      )}

      {/* ── Success CTA ──────────────────────────────────────────── */}
      {isComplete && (
        <div className="flex flex-col items-center gap-3 py-4 px-6 rounded-[var(--radius-lg)] bg-emerald-500/10 border border-emerald-500/30">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={22} className="text-emerald-400" />
            <span className="text-base font-bold text-emerald-400">All stages complete — model is live-ready!</span>
          </div>
          <p className="text-sm text-[var(--text-muted)] text-center max-w-md">
            The ensemble is distilled and validated. Head to Live Trading to deploy signals for{' '}
            <span className="font-semibold text-emerald-300">{pipelineUniverse.join(', ')}</span>.
          </p>
          <div className="flex items-center gap-3">
            <Button
              onClick={() => navigate('/backtest')}
              className="bg-indigo-600 hover:bg-indigo-500 border-0 px-6"
            >
              View Backtest Results
            </Button>
            <Button
              size="lg"
              onClick={() => navigate('/trading')}
              className="bg-emerald-600 hover:bg-emerald-500 border-0 shadow-[0_0_20px_rgba(52,211,153,0.3)] px-8"
            >
              <ExternalLink size={16} />
              Deploy to Live Trading
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
