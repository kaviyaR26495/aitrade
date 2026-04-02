import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Card, Button, Input, Badge, PageHeader, Skeleton } from '../components/ui';
import {
  useZerodhaInstruments,
  useUniverseStocks,
  useSetUniverse,
} from '../hooks/useApi';
import { clearInstrumentCache, syncStockList } from '../services/api';
import { useAppStore } from '../store/appStore';
import {
  Search, X, CheckSquare, Square, RefreshCw,
  AlertTriangle, Save, ListChecks, Download,
} from 'lucide-react';

// ── Types ─────────────────────────────────────────────────────────────
interface Instrument {
  symbol: string;
  name: string;
  instrument_token: number;
  lot_size: number;
  exchange: string;
}

const MAX_VISIBLE = 250;

// ── Component ─────────────────────────────────────────────────────────
export default function StockSelector() {
  const { addNotification } = useAppStore();

  // Remote data
  const {
    data: instruments,
    isLoading: instLoading,
    isError: instError,
    error: instErrorObj,
    refetch: refetchInstruments,
  } = useZerodhaInstruments('NSE');

  const { data: universeStocks } = useUniverseStocks();
  const setUniverseMutation = useSetUniverse();

  // Local state
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [syncingList, setSyncingList] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  // Initialise selection from current universe
  useEffect(() => {
    if (universeStocks && universeStocks.length > 0) {
      setSelected(new Set(universeStocks.map((s: { symbol: string }) => s.symbol)));
    }
  }, [universeStocks]);

  // Escape clears search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSearch('');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Filtered instruments
  const filtered: Instrument[] = useMemo(() => {
    if (!instruments) return [];
    const q = search.trim().toLowerCase();
    if (!q) return instruments;
    return instruments.filter(
      (inst: Instrument) =>
        inst.symbol.toLowerCase().includes(q) ||
        inst.name.toLowerCase().includes(q),
    );
  }, [instruments, search]);

  const visible = filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  // Toggle helpers
  const toggle = useCallback((symbol: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(symbol)) next.delete(symbol);
      else next.add(symbol);
      return next;
    });
  }, []);

  const selectAllVisible = () => {
    setSelected(prev => {
      const next = new Set(prev);
      visible.forEach(i => next.add(i.symbol));
      return next;
    });
  };

  const deselectAllVisible = () => {
    setSelected(prev => {
      const next = new Set(prev);
      visible.forEach(i => next.delete(i.symbol));
      return next;
    });
  };

  const clearAll = () => setSelected(new Set());

  const allVisibleSelected =
    visible.length > 0 && visible.every(i => selected.has(i.symbol));

  // Save universe
  const handleSave = () => {
    const symbols = Array.from(selected).sort();
    setUniverseMutation.mutate(
      { category: 'custom', customSymbols: symbols },
      {
        onSuccess: () =>
          addNotification({
            type: 'success',
            message: `Universe saved — ${symbols.length} stock${symbols.length !== 1 ? 's' : ''}`,
          }),
        onError: () =>
          addNotification({ type: 'error', message: 'Failed to save universe' }),
      },
    );
  };

  // Refresh instrument list (bust cache + re-fetch)
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await clearInstrumentCache();
      await refetchInstruments();
      addNotification({ type: 'success', message: 'Instrument list refreshed' });
    } catch {
      addNotification({ type: 'error', message: 'Refresh failed' });
    } finally {
      setRefreshing(false);
    }
  };

  // Sync stock list to DB (ensures DB rows exist for selected symbols)
  const handleSyncStockList = async () => {
    setSyncingList(true);
    try {
      await syncStockList();
      addNotification({ type: 'success', message: 'Stock list synced to database' });
    } catch {
      addNotification({ type: 'error', message: 'Sync failed — authenticate Zerodha first' });
    } finally {
      setSyncingList(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────
  const selectedList = Array.from(selected).sort();

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stock Selector"
        description="Browse every NSE equity and build your trading universe"
      >
        <Button
          variant="secondary"
          size="sm"
          onClick={handleRefresh}
          loading={refreshing}
        >
          <RefreshCw size={13} />
          Refresh List
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleSyncStockList}
          loading={syncingList}
        >
          <Download size={13} />
          Sync to&nbsp;DB
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          loading={setUniverseMutation.isPending}
          disabled={selected.size === 0}
        >
          <Save size={13} />
          Save Universe ({selected.size})
        </Button>
      </PageHeader>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* ── Left: Instrument browser ─── */}
        <Card title="NSE Instruments" className="lg:col-span-2">
          {/* Toolbar */}
          <div className="flex items-center gap-3 mb-5">
            <div className="relative flex-1">
              <Search
                size={15}
                className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] pointer-events-none"
              />
              <input
                ref={searchRef}
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search by symbol or company name… (Esc clears)"
                className="
                  w-full pl-10 pr-10 h-10 rounded-[var(--radius-sm)] text-sm
                  bg-[var(--bg-input)] border border-[var(--border)]
                  text-[var(--text)] placeholder:text-[var(--text-dim)]
                  focus:outline-none focus:border-[var(--primary)] focus:shadow-[0_0_0_3px_var(--primary-subtle)]
                  hover:border-[var(--text-muted)] transition-all duration-200
                "
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 rounded text-[var(--text-dim)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)] transition-colors"
                >
                  <X size={13} />
                </button>
              )}
            </div>

            <button
              type="button"
              onClick={allVisibleSelected ? deselectAllVisible : selectAllVisible}
              className="flex items-center gap-2 h-10 px-4 rounded-[var(--radius-sm)] text-[13px] font-semibold
                         bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)]
                         hover:border-[var(--primary)] hover:text-[var(--primary)] transition-all duration-200 whitespace-nowrap flex-shrink-0"
            >
              {allVisibleSelected ? <CheckSquare size={14} /> : <Square size={14} />}
              {allVisibleSelected ? 'Deselect visible' : 'Select visible'}
            </button>
          </div>

          {/* Stats row */}
          <div className="flex items-center gap-3 mb-3">
            {instruments && (
              <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">
                {instruments.length.toLocaleString()} instruments
              </span>
            )}
            {search && (
              <span className="text-[11px] text-[var(--text-dim)]">
                · {filtered.length.toLocaleString()} match{filtered.length !== 1 ? 'es' : ''}
              </span>
            )}
            {hasMore && (
              <span className="text-[11px] text-amber-400 font-medium">
                Showing first {MAX_VISIBLE} — refine search to see more
              </span>
            )}
          </div>

          {/* Error state */}
          {instError && (
            <div className="flex items-start gap-3 p-4 rounded-[var(--radius)] bg-[var(--danger-subtle)] border border-[var(--danger)] text-sm mb-4">
              <AlertTriangle size={16} className="text-[var(--danger)] mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold text-[var(--danger)]">Cannot load instruments</p>
                <p className="text-[var(--text-muted)] mt-0.5 text-xs">
                  {(instErrorObj as any)?.response?.data?.detail ??
                    'Authenticate Zerodha in Settings first, then return here.'}
                </p>
              </div>
            </div>
          )}

          {/* Table */}
          <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)]">
            {/* Sticky header */}
            <div className="grid grid-cols-[2.25rem_1fr_2fr_5rem] gap-x-2 px-4 py-2.5
                            bg-[var(--bg-elevated)] border-b border-[var(--border)]
                            text-[11px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
              <span />
              <span>Symbol</span>
              <span>Company</span>
              <span className="text-right">Lot&nbsp;Size</span>
            </div>

            <div className="max-h-[520px] overflow-y-auto">
              {instLoading ? (
                <div className="p-4 space-y-2">
                  <Skeleton lines={10} className="h-9" />
                </div>
              ) : visible.length === 0 && !instError ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <ListChecks size={30} className="text-[var(--text-dim)] mb-3" />
                  <p className="text-sm text-[var(--text-muted)] font-medium">No instruments found</p>
                  <p className="text-xs text-[var(--text-dim)] mt-1">
                    {instruments ? 'Try a different search term' : 'Authenticate Zerodha to load the list'}
                  </p>
                </div>
              ) : (
                visible.map(inst => {
                  const isChecked = selected.has(inst.symbol);
                  return (
                    <button
                      key={inst.instrument_token}
                      type="button"
                      onClick={() => toggle(inst.symbol)}
                      className={`
                        w-full grid grid-cols-[2.25rem_1fr_2fr_5rem] gap-x-2
                        px-4 py-2.5 text-left text-sm
                        border-b border-[var(--border-light)] last:border-0
                        transition-colors duration-100 group
                        ${isChecked
                          ? 'bg-[var(--primary-subtle)]'
                          : 'hover:bg-[var(--bg-hover)]'
                        }
                      `}
                    >
                      <span className={`flex items-center ${isChecked ? 'text-[var(--primary)]' : 'text-[var(--text-dim)] group-hover:text-[var(--text-muted)]'}`}>
                        {isChecked ? <CheckSquare size={15} /> : <Square size={15} />}
                      </span>
                      <span className={`font-mono font-semibold tracking-tight text-[13px] ${isChecked ? 'text-[var(--primary)]' : 'text-[var(--text)]'}`}>
                        {inst.symbol}
                      </span>
                      <span className="text-[var(--text-muted)] truncate text-xs leading-5 self-center">
                        {inst.name || '—'}
                      </span>
                      <span className="text-right text-[var(--text-dim)] tabular-nums text-xs self-center">
                        {inst.lot_size}
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </Card>

        {/* ── Right: Universe panel ─── */}
        <div className="flex flex-col gap-5">
          {/* Summary card */}
          <Card title="Your Universe">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-[var(--text-muted)]">Selected stocks</span>
                <Badge color="blue">{selected.size}</Badge>
              </div>

              <Button
                className="w-full"
                onClick={handleSave}
                loading={setUniverseMutation.isPending}
                disabled={selected.size === 0}
              >
                <Save size={14} />
                Save Universe
              </Button>

              {selected.size > 0 && (
                <button
                  type="button"
                  onClick={clearAll}
                  className="w-full text-xs text-[var(--text-dim)] hover:text-[var(--danger)] transition-colors py-1"
                >
                  Clear all selections
                </button>
              )}

              <div className="pt-1 space-y-1.5 text-xs text-[var(--text-dim)]">
                <p>
                  Saving sets <span className="font-mono text-[var(--primary)]">category: custom</span> — these exact symbols will be used for model training, backtesting, and live trading.
                </p>
                <p>
                  To revert to an index preset (Nifty&nbsp;50, etc.) use the <span className="font-medium text-[var(--text-muted)]">Settings</span> page.
                </p>
              </div>
            </div>
          </Card>

          {/* Selected chips */}
          {selectedList.length > 0 && (
            <Card title={`Selected (${selectedList.length})`} className="flex-1">
              <div className="max-h-[400px] overflow-y-auto">
                <div className="flex flex-wrap gap-1.5">
                  {selectedList.map(sym => (
                    <span
                      key={sym}
                      className="inline-flex items-center gap-1 pl-2.5 pr-1.5 py-1 rounded-full
                                 text-[11px] font-semibold font-mono
                                 bg-[var(--primary-subtle)] text-[var(--primary)]
                                 border border-[var(--primary-glow)]"
                    >
                      {sym}
                      <button
                        type="button"
                        onClick={() => toggle(sym)}
                        className="hover:text-[var(--danger)] transition-colors ml-0.5 leading-none"
                        aria-label={`Remove ${sym}`}
                      >
                        <X size={11} />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
