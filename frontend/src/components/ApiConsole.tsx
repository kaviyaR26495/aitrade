import { useState, useRef, useEffect } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2 } from 'lucide-react';
import { useAppStore, type ApiLogEntry } from '../store/appStore';

// ── Category detection ─────────────────────────────────────────────────
type Category = 'All' | 'Training' | 'Data' | 'Auth' | 'Trading' | 'Portfolio' | 'Backtest' | 'Other';

const CATEGORY_PATTERNS: [Category, RegExp][] = [
  ['Training',  /\/models\//],
  ['Data',      /\/data\//],
  ['Auth',      /\/auth\//],
  ['Trading',   /\/trading\//],
  ['Portfolio', /\/portfolio\//],
  ['Backtest',  /\/backtest\//],
];

const CATEGORY_COLOR: Record<Category, string> = {
  All:       'text-[var(--text)]',
  Training:  'text-violet-400',
  Data:      'text-sky-400',
  Auth:      'text-amber-400',
  Trading:   'text-emerald-400',
  Portfolio: 'text-teal-400',
  Backtest:  'text-indigo-400',
  Other:     'text-[var(--text-dim)]',
};

function getCategory(url: string): Category {
  for (const [cat, rx] of CATEGORY_PATTERNS) {
    if (rx.test(url)) return cat;
  }
  return 'Other';
}

function statusColor(status: number | null) {
  if (status === null) return 'text-[var(--text-dim)]';
  if (status === -1) return 'text-rose-400';
  if (status >= 500) return 'text-rose-400';
  if (status >= 400) return 'text-amber-400';
  if (status >= 200) return 'text-emerald-400';
  return 'text-[var(--text-dim)]';
}

function statusLabel(status: number | null) {
  if (status === null) return '…';
  if (status === -1) return 'ERR';
  return String(status);
}

const METHOD_COLOR: Record<string, string> = {
  GET: 'text-sky-400',
  POST: 'text-emerald-400',
  PUT: 'text-amber-400',
  DELETE: 'text-rose-400',
  PATCH: 'text-violet-400',
};

function fmtTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function tryPretty(raw?: string) {
  if (!raw) return '';
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

// ── Row ────────────────────────────────────────────────────────────────

function LogRow({ entry }: { entry: ApiLogEntry }) {
  const [open, setOpen] = useState(false);
  const today = new Date().toDateString();
  const entryDate = new Date(entry.timestamp).toDateString();
  const timeLabel = entryDate === today ? fmtTime(entry.timestamp) : `${fmtDate(entry.timestamp)} ${fmtTime(entry.timestamp)}`;
  const category = getCategory(entry.url);

  return (
    <div className="border-b border-[var(--border)]/40 last:border-0">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center gap-3 px-3 py-1.5 hover:bg-[var(--bg-hover)] text-left transition-colors"
      >
        {/* Method */}
        <span className={`font-mono text-[10px] font-bold w-10 shrink-0 ${METHOD_COLOR[entry.method] ?? 'text-[var(--text-muted)]'}`}>
          {entry.method}
        </span>

        {/* Category badge */}
        <span className={`text-[9px] font-medium w-16 shrink-0 ${CATEGORY_COLOR[category]}`}>
          {category}
        </span>

        {/* URL */}
        <span className="font-mono text-[11px] text-[var(--text)] flex-1 truncate">
          {entry.url}
        </span>

        {/* Status */}
        <span className={`font-mono text-[10px] font-bold w-8 shrink-0 text-right ${statusColor(entry.status)}`}>
          {statusLabel(entry.status)}
        </span>

        {/* Duration */}
        <span className="font-mono text-[10px] text-[var(--text-dim)] w-14 shrink-0 text-right">
          {entry.duration != null ? `${entry.duration}ms` : ''}
        </span>

        {/* Time */}
        <span className="font-mono text-[10px] text-[var(--text-muted)] w-28 shrink-0 text-right">
          {timeLabel}
        </span>

        <span className="text-[var(--text-muted)] shrink-0">
          {open ? <ChevronDown size={11} /> : <ChevronUp size={11} />}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 grid grid-cols-2 gap-2 bg-[var(--bg-input)]/40">
          <div>
            <p className="text-[9px] uppercase tracking-widest text-[var(--text-muted)] mb-1 mt-2">Request</p>
            <pre className="font-mono text-[10px] text-[var(--text)] bg-[var(--bg)] rounded p-2 overflow-x-auto max-h-40 whitespace-pre-wrap break-all">
              {entry.requestBody ? tryPretty(entry.requestBody) : <span className="text-[var(--text-dim)] italic">—</span>}
            </pre>
          </div>
          <div>
            <p className="text-[9px] uppercase tracking-widest text-[var(--text-muted)] mb-1 mt-2">Response</p>
            <pre className="font-mono text-[10px] text-[var(--text)] bg-[var(--bg)] rounded p-2 overflow-x-auto max-h-40 whitespace-pre-wrap break-all">
              {entry.responseBody ? tryPretty(entry.responseBody) : <span className="text-[var(--text-dim)] italic">pending…</span>}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────

const ALL_CATEGORIES: Category[] = ['All', 'Training', 'Data', 'Auth', 'Trading', 'Portfolio', 'Backtest', 'Other'];

export default function ApiConsole() {
  const { apiLog, apiConsoleOpen, toggleApiConsole, clearApiLog } = useAppStore();
  const [activeCategory, setActiveCategory] = useState<Category>('All');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to newest entry when open
  useEffect(() => {
    if (apiConsoleOpen && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [apiLog.length, apiConsoleOpen]);

  const filtered = activeCategory === 'All'
    ? [...apiLog].reverse()
    : [...apiLog].filter((e) => getCategory(e.url) === activeCategory).reverse();

  const pendingCount = apiLog.filter((e) => e.status === null).length;
  const errorCount = apiLog.filter((e) => e.status !== null && (e.status === -1 || e.status >= 400)).length;

  return (
    <div
      className="shrink-0 bg-[var(--bg-card)] border-t border-[var(--border)] transition-all duration-200"
      style={{ height: apiConsoleOpen ? 260 : 36 }}
    >
      {/* Header bar */}
      <div className="flex items-center gap-2 px-3 h-9 select-none cursor-pointer" onClick={toggleApiConsole}>
        <Terminal size={13} className="text-[var(--primary)] shrink-0" />
        <span className="text-[11px] font-semibold text-[var(--text)] tracking-wide">API Console</span>

        {/* Live badges */}
        <span className="text-[10px] text-[var(--text-muted)] ml-1">{apiLog.length} calls</span>
        {pendingCount > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-400 font-medium animate-pulse">
            {pendingCount} pending
          </span>
        )}
        {errorCount > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-400 font-medium">
            {errorCount} error{errorCount > 1 ? 's' : ''}
          </span>
        )}

        <div className="flex-1" />

        {/* Column headers (only visible when open, but stay in bar) */}
        {apiConsoleOpen && (
          <div className="hidden sm:flex items-center gap-3 text-[9px] uppercase tracking-widest text-[var(--text-muted)] mr-4">
            <span className="w-10">Method</span>
            <span className="w-16">Category</span>
            <span className="flex-1">URL</span>
            <span className="w-8 text-right">Status</span>
            <span className="w-14 text-right">Time</span>
            <span className="w-28 text-right">Timestamp</span>
          </div>
        )}

        {/* Actions */}
        <button
          onClick={(e) => { e.stopPropagation(); clearApiLog(); }}
          className="p-1 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-rose-400 transition-colors"
          title="Clear log"
        >
          <Trash2 size={12} />
        </button>
        <button
          className="p-1 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors"
          title={apiConsoleOpen ? 'Collapse' : 'Expand'}
        >
          {apiConsoleOpen ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </button>
      </div>

      {/* Log list */}
      {apiConsoleOpen && (
        <>
          {/* Category filter pills */}
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-[var(--border)]/40 overflow-x-auto no-scrollbar">
            {ALL_CATEGORIES.map((cat) => {
              const count = cat === 'All' ? apiLog.length : apiLog.filter((e) => getCategory(e.url) === cat).length;
              if (count === 0 && cat !== 'All') return null;
              return (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-medium shrink-0 transition-colors ${
                    activeCategory === cat
                      ? 'bg-[var(--primary)] text-white'
                      : 'bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  <span className={activeCategory !== cat ? CATEGORY_COLOR[cat] : ''}>{cat}</span>
                  <span className="opacity-60">{count}</span>
                </button>
              );
            })}
          </div>

          <div
            ref={scrollRef}
            className="overflow-y-auto"
            style={{ height: 260 - 36 - 34 }}
          >
            {filtered.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[11px] text-[var(--text-muted)] gap-2">
                <Terminal size={14} className="opacity-40" />
                {apiLog.length === 0 ? 'No API calls yet — make a request to see it here' : `No ${activeCategory} calls`}
              </div>
            ) : (
              filtered.map((entry) => <LogRow key={entry.id} entry={entry} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}
