import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, Badge, PageHeader, Skeleton } from '../components/ui';
import { useZerodhaInstruments, useSetUniverse, useUniverse } from '../hooks/useApi';
import { clearInstrumentCache, syncStockList, getUniversePresetSymbols } from '../services/api';
import { useAppStore } from '../store/appStore';
import {
  Search, X, CheckSquare, Square, RefreshCw,
  AlertTriangle, Save, ListChecks, Download, Rocket,
} from 'lucide-react';

interface Instrument {
  symbol: string;
  name: string;
  instrument_token: number;
  lot_size: number;
  exchange: string;
}

interface Preset {
  key: string;
  label: string;
  category: string;
  color: string;
  dot: string;
}

interface PresetGroup {
  label: string;
  presets: Preset[];
}

interface SelectionState {
  manualSelected: Set<string>;
  activePresetCategories: Set<string>;
  excludedSymbols: Set<string>;
}

const MAX_VISIBLE = 250;
const SAVED_UNIVERSE_STORAGE_KEY = 'aitrade-stock-selector-saved-universe';

function loadSavedUniverseSymbols(): string[] {
  try {
    const raw = localStorage.getItem(SAVED_UNIVERSE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((symbol): symbol is string => typeof symbol === 'string') : [];
  } catch {
    return [];
  }
}

function persistSavedUniverseSymbols(symbols: string[]) {
  try {
    localStorage.setItem(SAVED_UNIVERSE_STORAGE_KEY, JSON.stringify(symbols));
  } catch {
    // ignore storage failures
  }
}

const PRESET_GROUPS: PresetGroup[] = [
  {
    label: 'Broad Market',
    presets: [
      { key: 'nifty_50', label: 'Nifty 50', category: 'nifty_50', color: 'text-blue-300 border-blue-500', dot: 'bg-blue-400' },
      { key: 'nifty_100', label: 'Nifty 100', category: 'nifty_100', color: 'text-indigo-300 border-indigo-500', dot: 'bg-indigo-400' },
      { key: 'nifty_500', label: 'Nifty 500', category: 'nifty_500', color: 'text-amber-300 border-amber-500', dot: 'bg-amber-400' },
    ],
  },
  {
    label: 'Cap-Based',
    presets: [
      { key: 'nifty_midcap50', label: 'Midcap 50', category: 'nifty_midcap50', color: 'text-teal-300 border-teal-500', dot: 'bg-teal-400' },
    ],
  },
  {
    label: 'Sector',
    presets: [
      { key: 'nifty_bank', label: 'Banking', category: 'nifty_bank', color: 'text-emerald-300 border-emerald-500', dot: 'bg-emerald-400' },
      { key: 'nifty_psu_bank', label: 'PSU Banks', category: 'nifty_psu_bank', color: 'text-green-300 border-green-500', dot: 'bg-green-400' },
      { key: 'nifty_financial', label: 'Financial', category: 'nifty_financial', color: 'text-cyan-300 border-cyan-500', dot: 'bg-cyan-400' },
      { key: 'nifty_it', label: 'IT', category: 'nifty_it', color: 'text-sky-300 border-sky-500', dot: 'bg-sky-400' },
      { key: 'nifty_pharma', label: 'Pharma', category: 'nifty_pharma', color: 'text-pink-300 border-pink-500', dot: 'bg-pink-400' },
      { key: 'nifty_auto', label: 'Auto', category: 'nifty_auto', color: 'text-orange-300 border-orange-500', dot: 'bg-orange-400' },
      { key: 'nifty_fmcg', label: 'FMCG', category: 'nifty_fmcg', color: 'text-lime-300 border-lime-500', dot: 'bg-lime-400' },
      { key: 'nifty_metal', label: 'Metal', category: 'nifty_metal', color: 'text-stone-300 border-stone-500', dot: 'bg-stone-400' },
      { key: 'nifty_energy', label: 'Energy', category: 'nifty_energy', color: 'text-yellow-300 border-yellow-500', dot: 'bg-yellow-400' },
      { key: 'nifty_realty', label: 'Realty', category: 'nifty_realty', color: 'text-rose-300 border-rose-500', dot: 'bg-rose-400' },
    ],
  },
  {
    label: 'Themes',
    presets: [
      { key: 'nse_etf', label: 'ETFs', category: 'nse_etf', color: 'text-fuchsia-300 border-fuchsia-500', dot: 'bg-fuchsia-400' },
    ],
  },
];

export default function StockSelector() {
  const {
    addNotification,
    setPipelineUniverse,
    stockSelectorSelection,
    setStockSelectorSelection,
    resetStockSelectorSelection,
  } = useAppStore();
  const navigate = useNavigate();

  const {
    data: instruments,
    isLoading: instLoading,
    isError: instError,
    error: instErrorObj,
    refetch: refetchInstruments,
  } = useZerodhaInstruments('NSE');
  const { data: universe } = useUniverse();

  const setUniverseMutation = useSetUniverse();

  const [search, setSearch] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [syncingList, setSyncingList] = useState(false);
  const [presetLoading, setPresetLoading] = useState<string | null>(null);
  const [presetSymbolsByCategory, setPresetSymbolsByCategory] = useState<Record<string, string[]>>({});
  const hydratedSavedUniverseRef = useRef(false);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSearch('');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  useEffect(() => {
    if (hydratedSavedUniverseRef.current) return;
    
    // Always consider it hydrated if anything is in stockSelectorSelection 
    // because appStore now hydrates itself from localStorage.
    if (
      stockSelectorSelection.manualSelected.size > 0 ||
      stockSelectorSelection.activePresetCategories.size > 0 ||
      stockSelectorSelection.excludedSymbols.size > 0
    ) {
      hydratedSavedUniverseRef.current = true;
      return;
    }

    // If session is truly empty, look for permanent saved context
    const savedSymbols = loadSavedUniverseSymbols();
    if (savedSymbols.length > 0) {
      setStockSelectorSelection({
        manualSelected: new Set(savedSymbols),
        activePresetCategories: new Set(),
        excludedSymbols: new Set(),
      });
      hydratedSavedUniverseRef.current = true;
      return;
    }

    if (!universe || universe.category !== 'custom' || universe.custom_symbols.length === 0) {
      hydratedSavedUniverseRef.current = true;
      return;
    }

    setStockSelectorSelection({
      manualSelected: new Set(universe.custom_symbols),
      activePresetCategories: new Set(),
      excludedSymbols: new Set(),
    });
    hydratedSavedUniverseRef.current = true;
  }, [setStockSelectorSelection, stockSelectorSelection, universe]);

  useEffect(() => {
    if (!instruments) return;

    let cancelled = false;
    const uniqueCategories = Array.from(
      new Set(PRESET_GROUPS.flatMap(group => group.presets.map(preset => preset.category))),
    );

    const preloadPresets = async () => {
      const entries = await Promise.all(
        uniqueCategories.map(async category => {
          const response = await getUniversePresetSymbols(category);
          return [category, response.data.symbols] as const;
        }),
      );

      if (!cancelled) {
        setPresetSymbolsByCategory(Object.fromEntries(entries));
      }
    };

    preloadPresets().catch(() => {
      if (!cancelled) {
        setPresetSymbolsByCategory({});
      }
    });

    return () => {
      cancelled = true;
    };
  }, [instruments]);

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

  const selected = useMemo(() => {
    const next = new Set(stockSelectorSelection.manualSelected);

    stockSelectorSelection.activePresetCategories.forEach(category => {
      const presetSymbols = presetSymbolsByCategory[category] ?? [];
      presetSymbols.forEach(symbol => next.add(symbol));
    });

    stockSelectorSelection.excludedSymbols.forEach(symbol => next.delete(symbol));
    return next;
  }, [presetSymbolsByCategory, stockSelectorSelection]);

  const isSymbolCoveredByActivePreset = useCallback(
    (state: SelectionState, symbol: string) => {
      for (const category of state.activePresetCategories) {
        const presetSymbols = presetSymbolsByCategory[category] ?? [];
        if (presetSymbols.includes(symbol)) {
          return true;
        }
      }
      return false;
    },
    [presetSymbolsByCategory],
  );

  const toggle = useCallback((symbol: string) => {
    setStockSelectorSelection(prev => {
      const nextManual = new Set(prev.manualSelected);
      const nextExcluded = new Set(prev.excludedSymbols);
      const coveredByPreset = isSymbolCoveredByActivePreset(prev, symbol);
      const isExcluded = nextExcluded.has(symbol);
      const isManual = nextManual.has(symbol);

      if (coveredByPreset && !isExcluded) {
        nextManual.delete(symbol);
        nextExcluded.add(symbol);
      } else if (isManual) {
        nextManual.delete(symbol);
        nextExcluded.delete(symbol);
      } else {
        nextExcluded.delete(symbol);
        nextManual.add(symbol);
      }

      return {
        ...prev,
        manualSelected: nextManual,
        excludedSymbols: nextExcluded,
      };
    });
  }, [isSymbolCoveredByActivePreset, setStockSelectorSelection]);

  const selectAllVisible = () => {
    setStockSelectorSelection(prev => {
      const nextManual = new Set(prev.manualSelected);
      const nextExcluded = new Set(prev.excludedSymbols);
      visible.forEach(i => {
        nextManual.add(i.symbol);
        nextExcluded.delete(i.symbol);
      });
      return {
        ...prev,
        manualSelected: nextManual,
        excludedSymbols: nextExcluded,
      };
    });
  };

  const deselectAllVisible = () => {
    setStockSelectorSelection(prev => {
      const nextManual = new Set(prev.manualSelected);
      const nextExcluded = new Set(prev.excludedSymbols);
      visible.forEach(i => {
        nextManual.delete(i.symbol);
        if (isSymbolCoveredByActivePreset(prev, i.symbol)) {
          nextExcluded.add(i.symbol);
        } else {
          nextExcluded.delete(i.symbol);
        }
      });
      return {
        ...prev,
        manualSelected: nextManual,
        excludedSymbols: nextExcluded,
      };
    });
  };

  const clearAll = () => resetStockSelectorSelection();

  const isPresetActive = useCallback(
    (preset: Preset) => {
      return stockSelectorSelection.activePresetCategories.has(preset.category);
    },
    [stockSelectorSelection.activePresetCategories],
  );

  const applyPreset = async (preset: Preset) => {
    if (!instruments) return;
    const cached = presetSymbolsByCategory[preset.category];

    if (!cached) {
      setPresetLoading(preset.key);
    }

    try {
      const symbols = cached ?? (await getUniversePresetSymbols(preset.category)).data.symbols;
      if (!cached) {
        setPresetSymbolsByCategory(prev => ({ ...prev, [preset.category]: symbols }));
      }
      setStockSelectorSelection(prev => {
        const nextActive = new Set(prev.activePresetCategories);

        if (nextActive.has(preset.category)) {
          nextActive.delete(preset.category);
        } else {
          nextActive.add(preset.category);
        }

        return {
          ...prev,
          activePresetCategories: nextActive,
        };
      });
      addNotification({
        type: 'success',
        message: `${preset.label} toggled — ${symbols.length} stocks`,
      });
    } catch {
      addNotification({ type: 'error', message: `Failed to load ${preset.label} from NSE` });
    } finally {
      if (!cached) {
        setPresetLoading(null);
      }
    }
  };

  const allVisibleSelected = visible.length > 0 && visible.every(i => selected.has(i.symbol));

  const handleSave = () => {
    const symbols = Array.from(selected).sort();
    setUniverseMutation.mutate(
      { category: 'custom', customSymbols: symbols },
      {
        onSuccess: () => {
          persistSavedUniverseSymbols(symbols);
          setPipelineUniverse(symbols);
          addNotification({ type: 'success', message: `Universe saved — ${symbols.length} stock${symbols.length !== 1 ? 's' : ''}` });
        },
        onError: () => addNotification({ type: 'error', message: 'Failed to save universe' }),
      },
    );
  };

  const handleConfirmAndProceed = () => {
    const symbols = Array.from(selected).sort();
    setPipelineUniverse(symbols);
    setUniverseMutation.mutate(
      { category: 'custom', customSymbols: symbols },
      {
        onSuccess: () => {
          persistSavedUniverseSymbols(symbols);
          addNotification({ type: 'success', message: `Universe confirmed — ${symbols.length} stock${symbols.length !== 1 ? 's' : ''} ready for pipeline` });
          navigate('/models');
        },
        onError: () => addNotification({ type: 'error', message: 'Failed to save universe' }),
      },
    );
  };

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

  const selectedList = Array.from(selected).sort();

  return (
    <div className="space-y-8">
      <PageHeader title="Stock Selector" description="Browse every NSE equity and build your trading universe">
        <Button variant="secondary" size="sm" onClick={handleRefresh} loading={refreshing}>
          <RefreshCw size={13} />
          Refresh List
        </Button>
        <Button variant="secondary" size="sm" onClick={handleSyncStockList} loading={syncingList}>
          <Download size={13} />
          Sync to&nbsp;DB
        </Button>
        <Button size="sm" onClick={handleSave} loading={setUniverseMutation.isPending} disabled={selected.size === 0}>
          <Save size={13} />
          Save Universe ({selected.size})
        </Button>
        <Button
          size="sm"
          onClick={handleConfirmAndProceed}
          loading={setUniverseMutation.isPending}
          disabled={selected.size === 0}
          className="bg-gradient-to-r from-indigo-500 to-violet-500 hover:from-indigo-400 hover:to-violet-400 border-0"
        >
          <Rocket size={13} />
          Confirm &amp; Start Pipeline
        </Button>
      </PageHeader>

      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] px-4 py-3 space-y-2">
        <div className="flex items-center justify-between mb-0.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
            Quick Select <span className="normal-case font-normal opacity-60">(click to toggle stocks)</span>
          </span>
          <button
            type="button"
            onClick={clearAll}
            className="text-[10px] text-[var(--text-dim)] hover:text-[var(--danger)] transition-colors"
          >
            Reset to 0 ({selected.size})
          </button>
        </div>

        {PRESET_GROUPS.map(group => (
          <div key={group.label} className="flex items-center gap-2">
            <span className="text-[9px] font-semibold uppercase tracking-wider text-[var(--text-dim)] w-[72px] flex-shrink-0">
              {group.label}
            </span>
            <div className="flex flex-wrap gap-1">
              {group.presets.map(preset => {
                const active = isPresetActive(preset);
                const loading = presetLoading === preset.key;
                return (
                  <button
                    key={preset.key}
                    type="button"
                    onClick={() => applyPreset(preset)}
                    disabled={!instruments || loading}
                    aria-pressed={active}
                    title={`${active ? 'Remove' : 'Add'} ${preset.label}`}
                    className={`
                      inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                      text-[10px] font-semibold border transition-all duration-150
                      disabled:opacity-30 disabled:cursor-not-allowed
                      ${active
                        ? 'bg-[var(--primary)] border-[var(--primary)] text-white shadow-[0_0_0_1px_var(--primary-glow)]'
                        : 'bg-[var(--surface-2,var(--bg-input))] border-[var(--border)] text-[var(--text-dim)] hover:bg-[var(--bg-hover)] hover:border-[var(--text-muted)]'}
                    `}
                  >
                    {loading
                      ? <span className="w-1 h-1 rounded-full border border-current border-t-transparent animate-spin flex-shrink-0" />
                      : <span className={`w-1 h-1 rounded-full flex-shrink-0 ${active ? 'bg-white opacity-90' : `${preset.dot} opacity-70`}`} />}
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card title="NSE Instruments" className="lg:col-span-2">
          <div className="flex items-center gap-3 mb-5">
            <div className="relative flex-1">
              <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] pointer-events-none" />
              <input
                ref={searchRef}
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search by symbol or company name… (Esc clears)"
                className="w-full pl-10 pr-10 h-10 rounded-[var(--radius-sm)] text-sm bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--primary)] focus:shadow-[0_0_0_3px_var(--primary-subtle)] hover:border-[var(--text-muted)] transition-all duration-200"
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
              className="flex items-center gap-2 h-10 px-4 rounded-[var(--radius-sm)] text-[13px] font-semibold bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--primary)] hover:text-[var(--primary)] transition-all duration-200 whitespace-nowrap flex-shrink-0"
            >
              {allVisibleSelected ? <CheckSquare size={14} /> : <Square size={14} />}
              {allVisibleSelected ? 'Deselect visible' : 'Select visible'}
            </button>
          </div>

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

          {instError && (
            <div className="flex items-start gap-3 p-4 rounded-[var(--radius)] bg-[var(--danger-subtle)] border border-[var(--danger)] text-sm mb-4">
              <AlertTriangle size={16} className="text-[var(--danger)] mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold text-[var(--danger)]">Cannot load instruments</p>
                <p className="text-[var(--text-muted)] mt-0.5 text-xs">
                  {(instErrorObj as any)?.response?.data?.detail ?? 'Authenticate Zerodha in Settings first, then return here.'}
                </p>
              </div>
            </div>
          )}

          <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)]">
            <div className="grid grid-cols-[2.25rem_1fr_2fr_5rem] gap-x-2 px-4 py-2.5 bg-[var(--bg-elevated)] border-b border-[var(--border)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
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
                      className={`w-full grid grid-cols-[2.25rem_1fr_2fr_5rem] gap-x-2 px-4 py-2.5 text-left text-sm border-b border-[var(--border-light)] last:border-0 transition-all duration-100 group ${isChecked ? 'bg-[var(--primary-subtle)] shadow-[inset_3px_0_0_var(--primary)]' : 'hover:bg-[var(--bg-hover)]'}`}
                    >
                      <span className={`flex items-center ${isChecked ? 'text-[var(--primary)]' : 'text-[var(--text-dim)] group-hover:text-[var(--text-muted)]'}`}>
                        {isChecked ? <CheckSquare size={15} /> : <Square size={15} />}
                      </span>
                      <span className={`font-mono font-semibold tracking-tight text-[13px] ${isChecked ? 'text-white' : 'text-[var(--text)]'}`}>
                        {inst.symbol}
                      </span>
                      <span className={`truncate text-xs leading-5 self-center ${isChecked ? 'text-[var(--text)]' : 'text-[var(--text-muted)]'}`}>{inst.name || '—'}</span>
                      <span className="text-right text-[var(--text-dim)] tabular-nums text-xs self-center">{inst.lot_size}</span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </Card>

        <div className="flex flex-col gap-5">
          <Card title="Your Universe">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-[var(--text-muted)]">Selected stocks</span>
                <Badge color="blue">{selected.size}</Badge>
              </div>

              <Button className="w-full" onClick={handleSave} loading={setUniverseMutation.isPending} disabled={selected.size === 0}>
                <Save size={14} />
                Save Universe
              </Button>

              <Button
                className="w-full bg-gradient-to-r from-indigo-500 to-violet-500 hover:from-indigo-400 hover:to-violet-400 border-0 shadow-[0_0_20px_rgba(99,102,241,0.25)]"
                onClick={handleConfirmAndProceed}
                loading={setUniverseMutation.isPending}
                disabled={selected.size === 0}
              >
                <Rocket size={14} />
                Confirm &amp; Start Pipeline
              </Button>

              <button type="button" onClick={clearAll} className="w-full text-xs text-[var(--text-dim)] hover:text-[var(--danger)] transition-colors py-1">
                Reset to 0
              </button>

              <div className="pt-1 space-y-1.5 text-xs text-[var(--text-dim)]">
                <p>
                  Saving sets <span className="font-mono text-[var(--primary)]">category: custom</span> — these exact symbols will be used for model training, backtesting, and live trading.
                </p>
                <p>
                  <span className="font-medium text-violet-400">Confirm &amp; Start Pipeline</span> saves the universe and opens the One-Click Training Pipeline in Model Studio.
                </p>
                <p>
                  To revert to an index preset (Nifty&nbsp;50, etc.) use the <span className="font-medium text-[var(--text-muted)]">Settings</span> page.
                </p>
              </div>
            </div>
          </Card>

          {selectedList.length > 0 && (
            <Card title={`Selected (${selectedList.length})`} className="flex-1">
              <div className="max-h-[400px] overflow-y-auto">
                <div className="flex flex-wrap gap-1.5">
                  {selectedList.map((sym: string) => (
                    <span
                      key={sym}
                      className="inline-flex items-center gap-1 pl-2.5 pr-1.5 py-1 rounded-full text-[11px] font-semibold font-mono bg-[var(--primary)] text-white border border-[var(--primary-glow)] shadow-[0_0_0_1px_var(--primary-glow)]"
                    >
                      {sym}
                      <button type="button" onClick={() => toggle(sym)} className="hover:text-[var(--danger)] transition-colors ml-0.5 leading-none" aria-label={`Remove ${sym}`}>
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