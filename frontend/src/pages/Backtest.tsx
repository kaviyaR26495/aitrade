import { useState, useEffect, useMemo } from 'react';
import { Card, Button, Select, SearchableSelect, Input, Badge, StatCard, EmptyState, PageHeader, Table, Modal } from '../components/ui';
import { useUniverseStocks, useBacktestResults, useBacktestDetail, useRunBacktest, useOhlcv, useIndicators } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import LightweightCandleChart from '../components/LightweightCandleChart';
import { Square, BarChart3, Play, Search, X } from 'lucide-react';

export default function Backtest() {
  const { data: stocks } = useUniverseStocks();
  const { data: backtests } = useBacktestResults();
  const runBacktest = useRunBacktest();
  const { addNotification } = useAppStore();

  // ── Run form state ──
  const [modelType, setModelType] = useState('ensemble');
  const [knnName, setKnnName] = useState('');
  const [lstmName, setLstmName] = useState('');
  const [stockId, setStockId] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [isBulkRunning, setIsBulkRunning] = useState(false);
  const [shouldStopBulk, setShouldStopBulk] = useState(false);
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; symbol: string } | null>(null);

  // ── Table filters ──
  const [btSearch, setBtSearch] = useState('');
  const [btModelFilter, setBtModelFilter] = useState('');

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

  const filteredBacktests = useMemo(() => {
    if (!backtests) return [];
    return (backtests as any[]).filter(bt => {
      const matchSearch = !btSearch || (bt.symbol ?? '').toLowerCase().includes(btSearch.toLowerCase());
      const matchModel = !btModelFilter || bt.model_type === btModelFilter;
      return matchSearch && matchModel;
    });
  }, [backtests, btSearch, btModelFilter]);

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
      },
      {
        onSuccess: () => addNotification({ type: 'success', message: 'Backtest complete — table updated' }),
        onError: () => addNotification({ type: 'error', message: 'Backtest failed' }),
      }
    );
  };

  const handleBulkRun = async () => {
    if (!stocks || stocks.length === 0) return;
    setIsBulkRunning(true);
    setShouldStopBulk(false);
    setBulkProgress({ current: 0, total: stocks.length, symbol: '' });
    addNotification({ type: 'info', message: `Starting bulk backtest for ${stocks.length} stocks` });
    let successCount = 0;
    for (let i = 0; i < stocks.length; i++) {
      const stock = stocks[i];
      if (shouldStopBulk) break;
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
        });
        successCount++;
      } catch (err) {
        console.error(`Failed backtest for ${stock.symbol}`, err);
      }
    }
    const wasStopped = shouldStopBulk;
    setIsBulkRunning(false);
    setShouldStopBulk(false);
    setBulkProgress(null);
    addNotification({
      type: wasStopped || successCount < stocks.length ? 'warning' : 'success',
      message: wasStopped
        ? `Bulk backtest stopped. Processed ${successCount} stocks.`
        : `Bulk backtest complete: ${successCount}/${stocks.length} succeeded`,
    });
  };

  const handleStopBulk = () => setShouldStopBulk(true);

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
              loading={isBulkRunning && !shouldStopBulk}
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

      {/* ── Historical Backtests table ── */}
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
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] pointer-events-none" />
              <input
                type="text"
                value={btSearch}
                onChange={e => setBtSearch(e.target.value)}
                placeholder="Symbol…"
                className="h-8 w-36 pl-7 pr-7 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] text-xs text-[var(--text)] placeholder:text-[var(--text-dim)] outline-none focus:border-[var(--primary)] transition-colors"
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
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 12 12'%3E%3Cpath fill='%237a8ba8' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 8px center',
                paddingRight: '1.5rem',
              }}
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
        <Table
          data={filteredBacktests}
          onRowClick={(row: any) => setBtDetailId(row.id)}
          emptyState={
            <EmptyState
              icon={<BarChart3 size={24} />}
              title={backtests && (backtests as any[]).length > 0 ? 'No results match filters' : 'No backtests yet'}
              description={backtests && (backtests as any[]).length > 0 ? 'Try clearing the search or model filter.' : 'Run a backtest above to see results here.'}
            />
          }
          columns={[
            {
              key: 'symbol', label: 'Symbol', sortable: true,
              render: (row: any) => <span className="font-bold text-[var(--text)] tracking-tight">{row.symbol || `#${row.stock_id}`}</span>,
            },
            {
              key: 'model_type', label: 'Model', sortable: true,
              render: (row: any) => <Badge color={modelColorMap[row.model_type] ?? 'gray'}>{row.model_type}</Badge>,
            },
            {
              key: 'interval', label: 'Interval',
              render: (row: any) => <span className="text-[var(--text-muted)] text-xs uppercase">{row.interval ?? '—'}</span>,
            },
            {
              key: 'start_date', label: 'Start', sortable: true, mono: true,
              render: (row: any) => <span className="text-xs tabular-nums">{row.start_date ?? '—'}</span>,
            },
            {
              key: 'end_date', label: 'End', sortable: true, mono: true,
              render: (row: any) => <span className="text-xs tabular-nums">{row.end_date ?? '—'}</span>,
            },
            {
              key: 'total_return', label: 'Return', align: 'right', sortable: true,
              render: (row: any) => row.total_return != null ? (
                <span className={`font-bold tabular-nums font-[var(--font-mono)] ${row.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {row.total_return >= 0 ? '+' : ''}{(row.total_return * 100).toFixed(1)}%
                </span>
              ) : <span className="text-[var(--text-dim)]">—</span>,
            },
            {
              key: 'sharpe', label: 'Sharpe', align: 'right', sortable: true, mono: true,
              render: (row: any) => <span className="text-xs">{row.sharpe != null ? row.sharpe.toFixed(2) : '—'}</span>,
            },
            {
              key: 'max_drawdown', label: 'Max DD', align: 'right', sortable: true,
              render: (row: any) => row.max_drawdown != null ? (
                <span className="text-rose-400 font-mono text-xs tabular-nums">{(row.max_drawdown * 100).toFixed(1)}%</span>
              ) : <span className="text-[var(--text-dim)]">—</span>,
            },
            {
              key: 'win_rate', label: 'Win %', align: 'right', sortable: true,
              render: (row: any) => row.win_rate != null ? (
                <span className="text-sky-400 font-mono text-xs tabular-nums">{(row.win_rate * 100).toFixed(0)}%</span>
              ) : <span className="text-[var(--text-dim)]">—</span>,
            },
            {
              key: 'profit_factor', label: 'PF', align: 'right', sortable: true, mono: true,
              render: (row: any) => <span className="text-xs">{row.profit_factor != null ? row.profit_factor.toFixed(2) : '—'}</span>,
            },
            {
              key: 'trades_count', label: 'Trades', align: 'right', sortable: true, mono: true,
              render: (row: any) => <span className="text-xs">{row.trades_count ?? '—'}</span>,
            },
          ]}
        />
      </Card>

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
