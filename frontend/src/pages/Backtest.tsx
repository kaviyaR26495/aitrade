import { useState, useEffect, useMemo, useRef } from 'react';
import { Card, Button, Select, SearchableSelect, Input, Badge, StatCard, EmptyState, PageHeader, Table, Modal, Checkbox } from '../components/ui';
import { useUniverseStocks, useBacktestResults, useBacktestDetail, useRunBacktest, useOhlcv, useIndicators, useDeleteBacktestBatch } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import LightweightCandleChart from '../components/LightweightCandleChart';
import { Square, BarChart3, Play, Search, X, Trash2, ChevronDown, ChevronRight } from 'lucide-react';

export default function Backtest() {
  const { data: stocks } = useUniverseStocks();
  const { data: backtests } = useBacktestResults();
  const runBacktest = useRunBacktest();
  const deleteBatch = useDeleteBacktestBatch();
  const { addNotification } = useAppStore();

  // ── Run form state ──
  const [modelType, setModelType] = useState('ensemble');
  const [knnName, setKnnName] = useState('');
  const [lstmName, setLstmName] = useState('');
  const [stockId, setStockId] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [stoplossPct, setStoplossPct] = useState('5.0');
  const [targetPct, setTargetPct] = useState('');
  const [minConfidence, setMinConfidence] = useState('0.50');
  const [isBulkRunning, setIsBulkRunning] = useState(false);
  const shouldStopRef = useRef(false);  // ref so the async loop always reads the latest value
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; symbol: string } | null>(null);

  // ── Table filters ──
  const [btSearch, setBtSearch] = useState('');
  const [btModelFilter, setBtModelFilter] = useState('');
  const [showZeroTrades, setShowZeroTrades] = useState(false);
  // ── Selection & delete ──
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [confirmDelete, setConfirmDelete] = useState<{ ids: number[]; label: string } | null>(null);
  const [sortKey, setSortKey] = useState<string>('total_return');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  // ── Detail modal ──
  const [btDetailId, setBtDetailId] = useState<number | null>(null);
  const [btModalTab, setBtModalTab] = useState<'chart' | 'trades'>('chart');
  const { data: btDetailData, isLoading: btDetailLoading } = useBacktestDetail(btDetailId);

  const btDetail = btDetailData as any;
  const { data: btModalOhlcv, isLoading: btModalOhlcvLoading } = useOhlcv(
    btDetail?.stock_id,
    btDetail?.interval || 'day',
    btDetail?.start_date,
    btDetail?.end_date,
  );
  const { data: btModalIndicatorData } = useIndicators(
    btDetail?.stock_id,
    btDetail?.interval || 'day',
    btDetail?.start_date,
    btDetail?.end_date,
  );

  // Default dates: 1 year ago → today
  useEffect(() => {
    if (!startDate) {
      const end = new Date();
      const start = new Date();
      start.setFullYear(end.getFullYear() - 1);
      setEndDate(end.toISOString().split('T')[0]);
      setStartDate(start.toISOString().split('T')[0]);
    }
  }, []);

  const stockOptions = [
    { value: '', label: 'Select stock...' },
    ...(stocks?.map((s: any) => ({ value: String(s.id), label: s.symbol })) ?? []),
  ];

  // Group backtests by YYYY-MM-DD of created_at (the day the run was triggered)
  const groupedBacktests = useMemo(() => {
    if (!backtests) return [];
    const filtered = (backtests as any[]).filter(bt => {
      if (!showZeroTrades && (bt.trades_count === 0 || bt.trades_count == null)) return false;
      const searchLower = btSearch.toLowerCase();
      const matchSearch = !btSearch || (
        (bt.symbol ?? '').toLowerCase().includes(searchLower) ||
        (bt.model_type ?? '').toLowerCase().includes(searchLower)
      );
      const matchModel = !btModelFilter || bt.model_type === btModelFilter;
      return matchSearch && matchModel;
    });
    // Group by run date (created_at day)
    const map = new Map<string, any[]>();
    for (const bt of filtered) {
      const runDay = bt.created_at ? bt.created_at.slice(0, 10) : 'Unknown';
      if (!map.has(runDay)) map.set(runDay, []);
      map.get(runDay)!.push(bt);
    }
    // Sort groups newest first
    return Array.from(map.entries()).sort((a, b) => b[0].localeCompare(a[0]));
  }, [backtests, btSearch, btModelFilter, showZeroTrades]);

  const filteredBacktests = useMemo(() => groupedBacktests.flatMap(([, rows]) => rows), [groupedBacktests]);

  const toggleSelect = (id: number) => setSelectedIds(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const toggleGroup = (runDay: string, rows: any[]) => {
    const allSelected = rows.every(r => selectedIds.has(r.id));
    setSelectedIds(prev => {
      const next = new Set(prev);
      rows.forEach(r => allSelected ? next.delete(r.id) : next.add(r.id));
      return next;
    });
  };

  const handleDeleteConfirmed = async () => {
    if (!confirmDelete) return;
    try {
      await deleteBatch.mutateAsync(confirmDelete.ids);
      setSelectedIds(new Set());
      addNotification({ type: 'success', message: `Deleted ${confirmDelete.ids.length} backtest(s)` });
    } catch {
      addNotification({ type: 'error', message: 'Failed to delete backtests' });
    } finally {
      setConfirmDelete(null);
    }
  };

  // Sort helper — applied per-group
  const sortRows = (rows: any[]) => {
    if (!sortKey) return rows;
    return [...rows].sort((a, b) => {
      let av: any, bv: any;
      if (sortKey === 'symbol')       { av = a.symbol ?? ''; bv = b.symbol ?? ''; }
      else if (sortKey === 'model')   { av = a.model_type ?? ''; bv = b.model_type ?? ''; }
      else if (sortKey === 'period')  { av = a.start_date ?? ''; bv = b.start_date ?? ''; }
      else if (sortKey === 'return')  { av = a.total_return ?? -Infinity; bv = b.total_return ?? -Infinity; }
      else if (sortKey === 'sharpe')  { av = a.sharpe ?? -Infinity; bv = b.sharpe ?? -Infinity; }
      else if (sortKey === 'maxdd')   { av = a.max_drawdown ?? -Infinity; bv = b.max_drawdown ?? -Infinity; }
      else if (sortKey === 'winrate') { av = a.win_rate ?? -Infinity; bv = b.win_rate ?? -Infinity; }
      else if (sortKey === 'trades')  { av = a.trades_count ?? -Infinity; bv = b.trades_count ?? -Infinity; }
      else return 0;
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
  };

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const SortIcon = ({ col }: { col: string }) => {
    if (sortKey !== col) return <span className="ml-0.5 opacity-20">↕</span>;
    return <span className="ml-0.5 text-[var(--primary)]">{sortDir === 'asc' ? '▲' : '▼'}</span>;
  };

  // Modal chart indicators
  const btModalChartIndicators = useMemo(() => {
    if (!btModalIndicatorData || !btModalOhlcv || (btModalOhlcv as any[]).length === 0) return [];
    const validTimes = new Set((btModalOhlcv as any[]).map((d: any) => d.date));
    const colors = ['#6366f1', '#eab308', '#ec4899', '#06b6d4'];
    const firstPoint = (btModalIndicatorData as any[])[0];
    if (!firstPoint) return [];
    const keys = Object.keys(firstPoint).filter(k =>
      (k.toLowerCase().includes('sma') || k.toLowerCase().includes('ema') ||
       k.toLowerCase().includes('bb') || k.toLowerCase().includes('kama') ||
       k.toLowerCase().includes('vwkama')) && k !== 'date'
    );
    return keys.slice(0, 4).map((key, idx) => ({
      name: key.toUpperCase(),
      color: colors[idx % colors.length],
      data: (btModalIndicatorData as any[])
        .filter((d: any) => validTimes.has(d.date) && d[key] != null)
        .map((d: any) => ({ time: d.date, value: d[key] })),
    })).filter(s => s.data.length > 0);
  }, [btModalIndicatorData, btModalOhlcv]);

  // Modal chart trade markers
  const btModalTradeMarkers = useMemo(() => {
    if (!btDetail?.trade_log || !btModalOhlcv || (btModalOhlcv as any[]).length === 0) return [];
    const validTimes = new Set((btModalOhlcv as any[]).map((d: any) => d.date));
    const markers: any[] = [];
    btDetail.trade_log.forEach((trade: any) => {
      if (validTimes.has(trade.entry_date))
        markers.push({ time: trade.entry_date, side: 'BUY', price: trade.entry_price, reason: trade.action });
      if (validTimes.has(trade.exit_date))
        markers.push({ time: trade.exit_date, side: 'SELL', price: trade.exit_price, reason: trade.exit_reason });
    });
    return markers;
  }, [btDetail?.trade_log, btModalOhlcv]);

  const handleRun = () => {
    runBacktest.mutate(
      {
        model_type: modelType,
        knn_name: knnName || undefined,
        lstm_name: lstmName || undefined,
        stock_id: stockId ? parseInt(stockId) : undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        initial_capital: parseFloat(initialCapital),
        stoploss_pct: parseFloat(stoplossPct) || 5.0,
        target_pct: targetPct ? parseFloat(targetPct) : undefined,
        min_confidence: parseFloat(minConfidence) || 0.50,
      },
      {
        onSuccess: () => addNotification({ type: 'success', message: 'Backtest complete — table updated' }),
        onError: () => addNotification({ type: 'error', message: 'Backtest failed' }),
      }
    );
  };

  const handleBulkRun = async () => {
    if (!stocks || stocks.length === 0) return;
    shouldStopRef.current = false;  // reset before starting
    setIsBulkRunning(true);
    setBulkProgress({ current: 0, total: stocks.length, symbol: '' });
    addNotification({ type: 'info', message: `Starting bulk backtest for ${stocks.length} stocks` });
    let successCount = 0;
    for (let i = 0; i < stocks.length; i++) {
      const stock = stocks[i];
      if (shouldStopRef.current) break;  // reads current value — not a stale closure
      setBulkProgress(prev => ({ ...prev!, current: i + 1, symbol: stock.symbol }));
      try {
        await runBacktest.mutateAsync({
          model_type: modelType,
          knn_name: knnName || undefined,
          lstm_name: lstmName || undefined,
          stock_id: stock.id,
          start_date: startDate || undefined,
          end_date: endDate || undefined,
          initial_capital: parseFloat(initialCapital),
          stoploss_pct: parseFloat(stoplossPct) || 5.0,
          target_pct: targetPct ? parseFloat(targetPct) : undefined,
          min_confidence: parseFloat(minConfidence) || 0.50,
        });
        successCount++;
      } catch (err) {
        console.error(`Failed backtest for ${stock.symbol}`, err);
      }
    }
    const wasStopped = shouldStopRef.current;
    setIsBulkRunning(false);
    shouldStopRef.current = false;
    setBulkProgress(null);
    addNotification({
      type: wasStopped || successCount < stocks.length ? 'warning' : 'success',
      message: wasStopped
        ? `Bulk backtest stopped after ${successCount} stocks.`
        : `Bulk backtest complete: ${successCount}/${stocks.length} succeeded`,
    });
  };

  const handleStopBulk = () => { shouldStopRef.current = true; };  // write ref immediately — no re-render needed

  const modelColorMap: Record<string, any> = { ensemble: 'blue', knn: 'gray', lstm: 'purple', rl: 'yellow' };

  return (
    <div className="space-y-6">
      <PageHeader title="Backtest" description="Test strategies against historical data" />

      {/* ── Run Configuration (Compact Horizontal) ── */}
      <Card title="Backtest Configuration" data-guide-id="backtest-config">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[180px]">
            <Select value={modelType} onChange={setModelType} label="Model Type" options={[
              { value: 'ensemble', label: 'Ensemble' },
              { value: 'knn', label: 'KNN Only' },
              { value: 'lstm', label: 'LSTM Only' },
              { value: 'rl', label: 'RL Model' },
            ]} />
          </div>

          {['ensemble', 'knn'].includes(modelType) && (
            <div className="flex-1 min-w-[150px]">
               <Input value={knnName} onChange={setKnnName} label="KNN Name" placeholder="Latest" />
            </div>
          )}
          {['ensemble', 'lstm'].includes(modelType) && (
            <div className="flex-1 min-w-[150px]">
               <Input value={lstmName} onChange={setLstmName} label="LSTM Name" placeholder="Latest" />
            </div>
          )}

          <div className="flex-[1.5] min-w-[200px]">
            <SearchableSelect value={stockId} onChange={setStockId} options={stockOptions} label="Stock" placeholder="Select..." />
          </div>
          
          <div className="w-36">
            <Input value={startDate} onChange={setStartDate} label="Start" type="date" />
          </div>
          <div className="w-36">
            <Input value={endDate} onChange={setEndDate} label="End" type="date" />
          </div>
          <div className="w-28">
            <Input value={initialCapital} onChange={setInitialCapital} label="Capital" type="number" />
          </div>
          <div className="w-24">
            <Input value={stoplossPct} onChange={setStoplossPct} label="Stoploss %" type="number" placeholder="5.0" />
          </div>
          <div className="w-24">
            <Input value={targetPct} onChange={setTargetPct} label="Target %" type="number" placeholder="None" />
          </div>
          <div className="w-24">
            <Input value={minConfidence} onChange={setMinConfidence} label="Min Conf" type="number" placeholder="0.50" />
          </div>

          <div className="flex gap-2 pb-0.5">
            <Button
              onClick={handleRun}
              loading={runBacktest.isPending && !isBulkRunning}
              disabled={!stockId || isBulkRunning}
              data-guide-id="run-backtest-btn"
              className="px-4"
              title="Run Single Backtest"
            >
              <BarChart3 size={14} className="mr-1.5" /> Run
            </Button>

            <Button
              onClick={handleBulkRun}
              loading={isBulkRunning && !shouldStopRef.current}
              disabled={!stocks || stocks.length === 0 || isBulkRunning}
              variant="secondary"
              className="px-4 border-dashed border"
              title="Backtest All Stocks"
            >
              <Play size={14} className="mr-1.5" /> Bulk
            </Button>
            
            {isBulkRunning && (
              <Button onClick={handleStopBulk} variant="danger" className="px-3" title="Stop Bulk Run">
                <Square size={14} fill="currentColor" />
              </Button>
            )}
          </div>
        </div>

        {isBulkRunning && bulkProgress && (
          <div className="mt-4 p-2 rounded-[var(--radius-sm)] bg-[var(--bg-input)] border border-[var(--primary)]/20">
            <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--primary)] mb-1">
              <span>Processing: {bulkProgress.symbol} ({bulkProgress.current}/{bulkProgress.total})</span>
            </div>
            <div className="w-full h-1 bg-[var(--bg-card)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--primary)] transition-all duration-300"
                style={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
      </Card>

      {/* ── History: grouped by run date ── */}
      <Card
        title={
          <span className="flex items-center gap-2">
            Backtest History
            <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[var(--primary-subtle)] text-[var(--primary)]">
              {filteredBacktests.length}
            </span>
          </span>
        }
        action={
          <div className="flex items-center gap-3 flex-wrap">
            {selectedIds.size > 0 && (
              <Button
                variant="danger"
                size="sm"
                onClick={() => setConfirmDelete({ ids: Array.from(selectedIds), label: `${selectedIds.size} selected` })}
              >
                <Trash2 size={12} /> Delete {selectedIds.size}
              </Button>
            )}
            <Checkbox label="Show 0 trades" checked={showZeroTrades} onChange={setShowZeroTrades} />
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] pointer-events-none" />
              <input
                type="text"
                value={btSearch}
                onChange={e => setBtSearch(e.target.value)}
                placeholder="Symbol…"
                className="h-8 w-32 pl-7 pr-6 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] text-xs text-[var(--text)] placeholder:text-[var(--text-dim)] outline-none focus:border-[var(--primary)] transition-colors"
              />
              {btSearch && (
                <button onClick={() => setBtSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--text-dim)] hover:text-[var(--text)] transition-colors">
                  <X size={11} />
                </button>
              )}
            </div>
            <select
              value={btModelFilter}
              onChange={e => setBtModelFilter(e.target.value)}
              className="h-8 px-2.5 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] text-xs text-[var(--text)] outline-none focus:border-[var(--primary)] cursor-pointer appearance-none transition-colors"
              style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 12 12'%3E%3Cpath fill='%237a8ba8' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 8px center', paddingRight: '1.5rem' }}
            >
              <option value="">All Models</option>
              <option value="ensemble">Ensemble</option>
              <option value="knn">KNN</option>
              <option value="lstm">LSTM</option>
              <option value="rl">RL</option>
            </select>
          </div>
        }
      >
        {groupedBacktests.length === 0 ? (
          <EmptyState
            icon={<BarChart3 size={24} />}
            title={backtests && (backtests as any[]).length > 0 ? 'No results match filters' : 'No backtests yet'}
            description={backtests && (backtests as any[]).length > 0 ? 'Try clearing filters.' : 'Run a backtest above to see results here.'}
          />
        ) : (
          <div className="space-y-4">
            {groupedBacktests.map(([runDay, rows]) => {
              const isCollapsed = collapsedGroups.has(runDay);
              const allGroupSelected = rows.every(r => selectedIds.has(r.id));
              const someGroupSelected = rows.some(r => selectedIds.has(r.id));
              // Period range summary for this group
              const periods = Array.from(new Set(rows.map((r: any) => `${r.start_date?.slice(0,7)} → ${r.end_date?.slice(0,7)}`))).join(', ');
              const runLabel = runDay === 'Unknown' ? 'Unknown run date' : (() => {
                const d = new Date(runDay);
                return d.toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });
              })();

              return (
                <div key={runDay} className="border border-[var(--border)] rounded-[var(--radius)] overflow-hidden">
                  {/* Group header */}
                  <div className="flex items-center gap-3 px-4 py-3 bg-[var(--bg-hover)] border-b border-[var(--border)]">
                    <input
                      type="checkbox"
                      checked={allGroupSelected}
                      ref={el => { if (el) el.indeterminate = someGroupSelected && !allGroupSelected; }}
                      onChange={() => toggleGroup(runDay, rows)}
                      className="cursor-pointer accent-[var(--primary)] w-3.5 h-3.5"
                    />
                    <button
                      onClick={() => setCollapsedGroups(prev => { const n = new Set(prev); n.has(runDay) ? n.delete(runDay) : n.add(runDay); return n; })}
                      className="flex items-center gap-2 flex-1 text-left cursor-pointer"
                    >
                      {isCollapsed ? <ChevronRight size={14} className="text-[var(--text-dim)] flex-shrink-0" /> : <ChevronDown size={14} className="text-[var(--text-dim)] flex-shrink-0" />}
                      <span className="text-xs font-bold text-[var(--text)]">{runLabel}</span>
                      <span className="text-[10px] text-[var(--text-dim)] font-mono">{periods}</span>
                      <span className="ml-1 inline-flex items-center justify-center px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[var(--primary-subtle)] text-[var(--primary)]">
                        {rows.length} run{rows.length !== 1 ? 's' : ''}
                      </span>
                    </button>
                    <button
                      onClick={() => setConfirmDelete({ ids: rows.map((r: any) => r.id), label: `all ${rows.length} runs from ${runLabel}` })}
                      className="ml-auto flex items-center gap-1.5 text-[10px] font-semibold text-rose-400 hover:text-rose-300 transition-colors px-2 py-1 rounded hover:bg-rose-500/10 cursor-pointer"
                    >
                      <Trash2 size={11} /> Delete group
                    </button>
                  </div>

                  {/* Rows */}
                  {!isCollapsed && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[var(--border)]">
                            <th className="py-2 px-3 w-8"></th>
                            {([
                              { label: 'Symbol',  key: 'symbol',  right: false },
                              { label: 'Model',   key: 'model',   right: false },
                              { label: 'Period',  key: 'period',  right: false },
                              { label: 'Return',  key: 'return',  right: true  },
                              { label: 'Sharpe',  key: 'sharpe',  right: true  },
                              { label: 'Max DD',  key: 'maxdd',   right: true  },
                              { label: 'Win%',    key: 'winrate', right: true  },
                              { label: 'Trades',  key: 'trades',  right: true  },
                            ] as const).map(col => (
                              <th
                                key={col.key}
                                onClick={() => handleSort(col.key)}
                                className={`py-2 px-3 text-[10px] font-medium uppercase tracking-wider cursor-pointer select-none transition-colors
                                  ${col.right ? 'text-right' : 'text-left'}
                                  ${sortKey === col.key ? 'text-[var(--primary)]' : 'text-[var(--text-dim)] hover:text-[var(--text)]'}`}
                              >
                                {col.label}<SortIcon col={col.key} />
                              </th>
                            ))}
                            <th className="py-2 px-3 w-8"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {sortRows(rows).map((row: any) => (
                            <tr
                              key={row.id}
                              className={`border-b border-[var(--border)]/30 hover:bg-[var(--table-row-hover)] transition-colors cursor-pointer ${
                                selectedIds.has(row.id) ? 'bg-[var(--primary-subtle)]/40' : ''
                              }`}
                            >
                              <td className="py-2.5 px-3" onClick={e => { e.stopPropagation(); toggleSelect(row.id); }}>
                                <input type="checkbox" checked={selectedIds.has(row.id)} onChange={() => toggleSelect(row.id)} className="cursor-pointer accent-[var(--primary)] w-3.5 h-3.5" />
                              </td>
                              <td className="py-2.5 px-3 font-bold text-[var(--text)] tracking-tight" onClick={() => setBtDetailId(row.id)}>
                                {row.symbol || `#${row.stock_id}`}
                              </td>
                              <td className="py-2.5 px-3" onClick={() => setBtDetailId(row.id)}>
                                <Badge color={modelColorMap[row.model_type] ?? 'gray'}>{row.model_type}</Badge>
                              </td>
                              <td className="py-2.5 px-3 text-xs text-[var(--text-muted)] font-mono tabular-nums" onClick={() => setBtDetailId(row.id)}>
                                {row.start_date?.slice(0,10)} → {row.end_date?.slice(0,10)}
                              </td>
                              <td className="py-2.5 px-3 text-right" onClick={() => setBtDetailId(row.id)}>
                                {row.total_return != null ? (
                                  <span className={`font-bold tabular-nums text-xs font-mono ${row.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {row.total_return >= 0 ? '+' : ''}{(row.total_return * 100).toFixed(1)}%
                                  </span>
                                ) : <span className="text-[var(--text-dim)] text-xs">—</span>}
                              </td>
                              <td className="py-2.5 px-3 text-right text-xs tabular-nums font-mono" onClick={() => setBtDetailId(row.id)}>{row.sharpe?.toFixed(2) ?? '—'}</td>
                              <td className="py-2.5 px-3 text-right" onClick={() => setBtDetailId(row.id)}>
                                {row.max_drawdown != null ? <span className="text-rose-400 text-xs font-mono tabular-nums">{(row.max_drawdown * 100).toFixed(1)}%</span> : <span className="text-[var(--text-dim)] text-xs">—</span>}
                              </td>
                              <td className="py-2.5 px-3 text-right" onClick={() => setBtDetailId(row.id)}>
                                {row.win_rate != null ? <span className="text-sky-400 text-xs font-mono tabular-nums">{(row.win_rate * 100).toFixed(0)}%</span> : <span className="text-[var(--text-dim)] text-xs">—</span>}
                              </td>
                              <td className="py-2.5 px-3 text-right text-xs font-mono tabular-nums" onClick={() => setBtDetailId(row.id)}>{row.trades_count ?? '—'}</td>
                              <td className="py-2.5 px-3 text-right" onClick={e => e.stopPropagation()}>
                                <button
                                  onClick={() => setConfirmDelete({ ids: [row.id], label: `${row.symbol || '#'+row.stock_id} (${row.model_type})` })}
                                  className="p-1.5 rounded text-[var(--text-dim)] hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer"
                                  title="Delete"
                                >
                                  <Trash2 size={12} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {/* ── Confirm Delete Modal ── */}
      {confirmDelete && (
        <Modal
          open={true}
          onClose={() => setConfirmDelete(null)}
          title="Confirm Delete"
          variant="danger"
          size="sm"
          footer={
            <>
              <Button variant="secondary" onClick={() => setConfirmDelete(null)}>Cancel</Button>
              <Button variant="danger" loading={deleteBatch.isPending} onClick={handleDeleteConfirmed}>
                <Trash2 size={13} /> Delete
              </Button>
            </>
          }
        >
          <p className="text-sm text-[var(--text-muted)]">
            Delete <span className="font-semibold text-[var(--text)]">{confirmDelete.label}</span>? This cannot be undone.
          </p>
        </Modal>
      )}

      {/* ── Backtest Detail Modal ── */}
      {btDetailId != null && (() => {
        const listRecord = (backtests as any[])?.find((b: any) => b.id === btDetailId);
        const detail = btDetailData as any;
        const tradeLog: any[] = detail?.trade_log ?? [];
        const symbol = listRecord?.symbol || detail?.symbol || `#${detail?.stock_id ?? btDetailId}`;
        const mdlType = listRecord?.model_type || detail?.model_type || '—';
        const isLoading = btDetailLoading || btModalOhlcvLoading;

        return (
          <Modal
            open={btDetailId != null}
            onClose={() => { setBtDetailId(null); setBtModalTab('chart'); }}
            title={symbol}
            description={`${listRecord?.start_date ?? ''} → ${listRecord?.end_date ?? ''}  ·  ${listRecord?.interval?.toUpperCase() ?? ''}`}
            size="2xl"
            icon={<BarChart3 size={18} className="text-[var(--primary)]" />}
          >
            <div className="space-y-5">
              {/* Meta badges */}
              <div className="flex items-center gap-2 flex-wrap">
                <Badge color={modelColorMap[mdlType] ?? 'gray'}>{mdlType}</Badge>
                {listRecord?.interval && <Badge color="gray" variant="outline">{listRecord.interval.toUpperCase()}</Badge>}
                {detail?.trades_count != null && (
                  <span className="text-xs text-[var(--text-muted)] font-semibold">{detail.trades_count} trades</span>
                )}
              </div>

              {/* Stat strip */}
              {isLoading ? (
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="h-16 bg-[var(--bg-hover)] rounded-[var(--radius-sm)] animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                  <StatCard size="sm" label="Return"
                    value={detail?.total_return != null ? `${detail.total_return >= 0 ? '+' : ''}${(detail.total_return * 100).toFixed(1)}%` : '—'}
                    color={detail?.total_return >= 0 ? 'var(--success)' : 'var(--danger)'}
                  />
                  <StatCard size="sm" label="Sharpe" value={detail?.sharpe != null ? detail.sharpe.toFixed(2) : '—'} color="var(--primary)" />
                  <StatCard size="sm" label="Max DD"
                    value={detail?.max_drawdown != null ? `${(detail.max_drawdown * 100).toFixed(1)}%` : '—'}
                    color="var(--danger)"
                  />
                  <StatCard size="sm" label="Win Rate" value={detail?.win_rate != null ? `${(detail.win_rate * 100).toFixed(0)}%` : '—'} color="var(--info)" />
                  <StatCard size="sm" label="Prof. Factor" value={detail?.profit_factor != null ? detail.profit_factor.toFixed(2) : '—'} color="var(--warning)" />
                  <StatCard size="sm" label="Trades" value={detail?.trades_count ?? '—'} color="var(--text-muted)" />
                </div>
              )}

              {/* Tabs */}
              <div className="flex gap-0 border-b border-[var(--border)]">
                {([['chart', 'Price Chart'], ['trades', 'Trade Log']] as const).map(([tab, label]) => (
                  <button
                    key={tab}
                    onClick={() => setBtModalTab(tab)}
                    className={`px-5 py-2.5 text-sm font-medium relative transition-colors cursor-pointer ${
                      btModalTab === tab ? 'text-[var(--primary)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    {label}
                    {btModalTab === tab && <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[var(--primary)] rounded-t-full" />}
                  </button>
                ))}
              </div>

              {/* Chart tab */}
              {btModalTab === 'chart' && (
                <div>
                  {isLoading ? (
                    <div className="h-[380px] bg-[var(--bg-hover)] rounded-[var(--radius-sm)] animate-pulse flex items-center justify-center">
                      <span className="text-xs text-[var(--text-dim)] uppercase tracking-wider font-semibold">Loading chart…</span>
                    </div>
                  ) : btModalOhlcv && (btModalOhlcv as any[]).length > 0 ? (
                    <div className="-mx-1">
                      <LightweightCandleChart
                        ohlcv={(btModalOhlcv as any[]).map((d: any) => ({
                          time: d.date, open: d.open, high: d.high, low: d.low, close: d.close,
                        }))}
                        indicators={btModalChartIndicators}
                        trades={btModalTradeMarkers}
                        height={380}
                      />
                    </div>
                  ) : (
                    <EmptyState icon={<BarChart3 size={24} />} title="No chart data" description="OHLCV data unavailable for this period." />
                  )}
                  {btModalTradeMarkers.length > 0 && !isLoading && (
                    <div className="flex items-center gap-4 mt-3 px-1">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                        <span className="text-[11px] text-[var(--text-muted)] font-medium">BUY entry</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-rose-400 inline-block" />
                        <span className="text-[11px] text-[var(--text-muted)] font-medium">SELL exit</span>
                      </div>
                      <span className="text-[11px] text-[var(--text-dim)]">{btModalTradeMarkers.length} markers</span>
                    </div>
                  )}
                </div>
              )}

              {/* Trade log tab */}
              {btModalTab === 'trades' && (
                <div>
                  {isLoading ? (
                    <div className="space-y-2">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="h-10 bg-[var(--bg-hover)] animate-pulse rounded" />
                      ))}
                    </div>
                  ) : tradeLog.length > 0 ? (
                    <div className="max-h-[380px] overflow-y-auto">
                      <Table
                        compact
                        data={tradeLog}
                        columns={[
                          { key: 'entry_date', label: 'Entry Date', mono: true, sortable: true, render: (t: any) => <span className="text-xs">{t.entry_date}</span> },
                          { key: 'exit_date', label: 'Exit Date', mono: true, sortable: true, render: (t: any) => <span className="text-xs">{t.exit_date}</span> },
                          {
                            key: 'entry_price', label: 'Entry ₹', align: 'right', sortable: true,
                            render: (t: any) => <span className="tabular-nums text-xs font-[var(--font-mono)] text-emerald-400">₹{t.entry_price?.toFixed(1) ?? '—'}</span>,
                          },
                          {
                            key: 'exit_price', label: 'Exit ₹', align: 'right', sortable: true,
                            render: (t: any) => <span className="tabular-nums text-xs font-[var(--font-mono)] text-rose-400">₹{t.exit_price?.toFixed(1) ?? '—'}</span>,
                          },
                          { key: 'quantity', label: 'Qty', align: 'right', mono: true, render: (t: any) => <span className="text-xs">{t.quantity ?? '—'}</span> },
                          {
                            key: 'pnl_pct', label: 'PnL %', align: 'right', sortable: true,
                            render: (t: any) => t.pnl_pct != null ? (
                              <span className={`font-bold tabular-nums text-xs font-[var(--font-mono)] ${t.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(2)}%
                              </span>
                            ) : '—',
                          },
                          {
                            key: 'exit_reason', label: 'Exit Reason',
                            render: (t: any) => <span className="text-xs text-[var(--text-muted)] capitalize">{t.exit_reason?.replace(/_/g, ' ') ?? '—'}</span>,
                          },
                        ]}
                      />
                    </div>
                  ) : (
                    <EmptyState icon={<BarChart3 size={20} />} title="No trades" description="No trades were executed in this backtest." />
                  )}
                </div>
              )}
            </div>
          </Modal>
        );
      })()}
    </div>
  );
}
