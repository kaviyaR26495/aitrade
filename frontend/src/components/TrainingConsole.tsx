import { useEffect, useRef, useState } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2, X, RefreshCw, Database, Activity } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import {
  useRlModels,
  useTrainingLogFile,
  useTrainingLogFiles,
  useTrainingLogsTotalSize,
  useDeleteTrainingLogFile,
  useDeleteAllTrainingLogs,
} from '../hooks/useApi';

// ── Line colouring ─────────────────────────────────────────────────────
function lineClass(line: string): string {
  if (line.includes('ERROR')) return 'text-rose-400';
  if (line.includes('STOPPED')) return 'text-amber-400';
  if (line.includes('PAUSED')) return 'text-amber-300';
  if (line.includes('DONE')) return 'text-emerald-400';
  if (line.includes('TRAIN')) return 'text-sky-300';
  if (line.includes('INFO')) return 'text-indigo-300';
  if (line.startsWith('=')) return 'text-[var(--text-dim)]';
  return 'text-[var(--text)]';
}

// ── Model selector pill ────────────────────────────────────────────────
function ModelPill({
  modelId,
  algorithm,
  isActive,
  selected,
  onClick,
}: {
  modelId: number;
  algorithm: string;
  isActive: boolean;
  selected: boolean;
  onClick: (e?: React.MouseEvent) => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-medium transition-colors shrink-0 ${
        selected
          ? 'bg-[var(--primary)] text-white'
          : 'bg-[var(--surface-2)] text-[var(--text-dim)] hover:bg-[var(--surface-3)]'
      }`}
    >
      {isActive && (
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
      )}
      #{modelId} {algorithm}
    </button>
  );
}

// ── Main console ───────────────────────────────────────────────────────
export default function TrainingConsole() {
  const {
    trainingConsoleOpen,
    trainingConsoleModelId,
    toggleTrainingConsole,
    setTrainingConsoleModelId,
  } = useAppStore();

  const [height, setHeight] = useState(340);
  const [dragging, setDragging] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [deleteAllConfirm, setDeleteAllConfirm] = useState(false);

  const { data: rlModels } = useRlModels();
  const { data: logFiles } = useTrainingLogFiles();
  const { data: totalSize } = useTrainingLogsTotalSize();
  const deleteFileMutation = useDeleteTrainingLogFile();
  const deleteAllMutation = useDeleteAllTrainingLogs();

  const models: any[] = rlModels ?? [];
  const activeModelIds = new Set(
    models.filter((m) => m.status === 'training' || m.status === 'paused').map((m) => m.id)
  );

  // Auto-open and switch to first actively training model
  const { openTrainingConsole } = useAppStore();
  useEffect(() => {
    if (activeModelIds.size > 0) {
      const firstActive = [...activeModelIds][0];
      if (!trainingConsoleModelId || !activeModelIds.has(trainingConsoleModelId)) {
        openTrainingConsole(firstActive);
      }
    }
  }, [models.map((m) => m.status).join(',')]);

  const selectedModelId = trainingConsoleModelId;
  const isActive = selectedModelId !== null && activeModelIds.has(selectedModelId);

  const { data: logData, refetch } = useTrainingLogFile(selectedModelId, isActive);

  const scrollRef = useRef<HTMLDivElement>(null);
  const prevLenRef = useRef(0);

  // Auto-scroll only when new lines arrive
  useEffect(() => {
    const lines = logData?.content?.split('\n').filter(Boolean) ?? [];
    if (lines.length > prevLenRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevLenRef.current = lines.length;
  }, [logData?.content]);

  // Drag-to-resize
  const dragStartY = useRef(0);
  const dragStartH = useRef(height);
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      const delta = dragStartY.current - e.clientY;
      setHeight(Math.max(180, Math.min(700, dragStartH.current + delta)));
    };
    const onUp = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [dragging]);

  const lines = logData?.content?.split('\n').filter(Boolean) ?? [];

  const modelsWithLogs = new Set((logFiles ?? []).map((f: any) => f.model_id));
  const displayModels = models.filter((m) => activeModelIds.has(m.id) || modelsWithLogs.has(m.id));

  const HEADER_H = 36;

  return (
    <div
      className="shrink-0 bg-[#0d1117] border-t border-[var(--border)] transition-all duration-200 select-none"
      style={{ height: trainingConsoleOpen ? height : HEADER_H }}
    >
      {/* Drag handle */}
      {trainingConsoleOpen && (
        <div
          className="h-1 w-full cursor-ns-resize hover:bg-[var(--primary)]/40 transition-colors"
          onMouseDown={(e) => {
            dragStartY.current = e.clientY;
            dragStartH.current = height;
            setDragging(true);
          }}
        />
      )}

      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 h-9 cursor-pointer"
        onClick={toggleTrainingConsole}
      >
        <Activity size={13} className="text-emerald-400 shrink-0" />
        <span className="text-[11px] font-semibold text-[var(--text)] tracking-wide">Training Console</span>

        {activeModelIds.size > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-medium animate-pulse">
            {activeModelIds.size} training
          </span>
        )}

        {totalSize && (
          <span className="text-[10px] text-[var(--text-muted)] flex items-center gap-1">
            <Database size={10} />
            {totalSize.total_human}
          </span>
        )}

        <div className="flex-1" />

        {trainingConsoleOpen && (
          <>
            {/* Model pills */}
            <div className="flex items-center gap-1 max-w-sm overflow-x-auto no-scrollbar">
              {displayModels.map((m: any) => (
                <ModelPill
                  key={m.id}
                  modelId={m.id}
                  algorithm={m.algorithm}
                  isActive={activeModelIds.has(m.id)}
                  selected={selectedModelId === m.id}
                  onClick={(e: any) => { e?.stopPropagation?.(); setTrainingConsoleModelId(m.id); }}
                />
              ))}
            </div>

            <button
              onClick={(e) => { e.stopPropagation(); refetch(); }}
              className="p-1 rounded hover:bg-white/10 text-[var(--text-muted)] transition-colors ml-1"
              title="Refresh"
            >
              <RefreshCw size={11} />
            </button>

            <button
              onClick={(e) => { e.stopPropagation(); setShowFiles((p) => !p); }}
              className={`p-1 rounded hover:bg-white/10 transition-colors ml-0.5 ${showFiles ? 'text-sky-400' : 'text-[var(--text-muted)]'}`}
              title="Log files"
            >
              <Database size={11} />
            </button>

            {selectedModelId && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteFileMutation.mutate(selectedModelId);
                  setTrainingConsoleModelId(null);
                }}
                className="p-1 rounded hover:bg-white/10 text-[var(--text-muted)] hover:text-rose-400 transition-colors"
                title="Delete this log"
              >
                <Trash2 size={11} />
              </button>
            )}
          </>
        )}

        <button
          className="p-1 rounded hover:bg-white/10 text-[var(--text-muted)] transition-colors"
          title={trainingConsoleOpen ? 'Collapse' : 'Expand'}
          onClick={(e) => { e.stopPropagation(); toggleTrainingConsole(); }}
        >
          {trainingConsoleOpen ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </button>
      </div>

      {/* Body */}
      {trainingConsoleOpen && (
        <div className="flex flex-col" style={{ height: height - HEADER_H - 4 }}>

          {/* File manager panel */}
          {showFiles && (
            <div className="bg-[#161b22] border-b border-[#30363d] px-3 py-2 shrink-0">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
                  Log Files — {totalSize?.total_human ?? '0 B'}
                </span>
                <div className="flex items-center gap-1">
                  {deleteAllConfirm ? (
                    <>
                      <span className="text-[10px] text-rose-400 mr-1">Delete all?</span>
                      <button
                        className="text-[10px] px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 hover:bg-rose-500/30"
                        onClick={() => { deleteAllMutation.mutate(); setDeleteAllConfirm(false); }}
                      >Yes</button>
                      <button
                        className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-[var(--text-muted)]"
                        onClick={() => setDeleteAllConfirm(false)}
                      >Cancel</button>
                    </>
                  ) : (
                    <button
                      className="text-[10px] flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 hover:bg-rose-500/20"
                      onClick={() => setDeleteAllConfirm(true)}
                    >
                      <Trash2 size={9} /> Delete all
                    </button>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
                {(logFiles ?? []).length === 0 && (
                  <span className="text-[10px] text-[var(--text-muted)] italic">No log files</span>
                )}
                {(logFiles ?? []).map((f: any) => (
                  <div key={f.model_id} className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-white/5 text-[10px]">
                    <span className="text-sky-300 font-mono">#{f.model_id}</span>
                    <span className="text-[var(--text-dim)]">{f.size_human}</span>
                    <span className="text-[var(--text-muted)]">{f.modified_at.slice(0, 16).replace('T', ' ')}</span>
                    <button
                      onClick={() => deleteFileMutation.mutate(f.model_id)}
                      className="ml-0.5 text-rose-400/60 hover:text-rose-400"
                    >
                      <X size={9} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Console output */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto font-mono text-[11px] leading-relaxed px-3 py-2 bg-[#0d1117]"
            style={{ userSelect: 'text' }}
          >
            {!selectedModelId ? (
              <div className="flex items-center justify-center h-full text-[var(--text-muted)] gap-2">
                <Terminal size={14} className="opacity-40" />
                <span>Select a model above to view its training log</span>
              </div>
            ) : lines.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[var(--text-muted)] gap-2">
                <Terminal size={14} className="opacity-40" />
                <span>No log data yet — waiting for training to start…</span>
              </div>
            ) : (
              (lines as string[]).map((line: string, i: number) => (
                <div key={i} className={`whitespace-pre-wrap break-all ${lineClass(line)}`}>
                  {line}
                </div>
              ))
            )}
            {isActive && (
              <div className="flex items-center gap-1.5 mt-1 text-emerald-400/60">
                <span className="animate-pulse">▋</span>
                <span className="text-[10px]">Training in progress…</span>
              </div>
            )}
          </div>

          {/* Status bar */}
          <div className="flex items-center justify-between px-3 py-1 bg-[#161b22] border-t border-[#30363d] shrink-0 text-[9px] text-[var(--text-muted)]">
            <span>{lines.length} lines</span>
            <span>{logData?.size_human ?? '—'}</span>
            <span>{isActive ? '● live' : '○ static'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
