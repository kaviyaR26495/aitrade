import { useState, useEffect, useMemo } from 'react';
import { Card, Button, Select, SearchableSelect, Input, Badge, StatCard, EmptyState, PageHeader, ListItem } from '../components/ui';
import { useUniverseStocks, useBacktestResults, useRunBacktest, useOhlcv, useIndicators, useTradePatterns } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import LightweightAreaChart, { type ChartPoint } from '../components/LightweightAreaChart';
import LightweightCandleChart from '../components/LightweightCandleChart';
import { Square, BarChart3, Play, Sparkles } from 'lucide-react';

export default function Backtest() {
  const { data: stocks } = useUniverseStocks();
  const { data: backtests } = useBacktestResults();
  const runBacktest = useRunBacktest();
  const { addNotification } = useAppStore();

  const [modelType, setModelType] = useState('ensemble');
  const [knnName, setKnnName] = useState('');
  const [lstmName, setLstmName] = useState('');
  const [stockId, setStockId] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [selectedResult, setSelectedResult] = useState<any>(null);
  const [isBulkRunning, setIsBulkRunning] = useState(false);
  const [shouldStopBulk, setShouldStopBulk] = useState(false);
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; symbol: string } | null>(null);
  const [bulkResults, setBulkResults] = useState<any[]>([]);
  const [selectedTrade, setSelectedTrade] = useState<{ backtestId: number; idx: number } | null>(null);

  const { data: matchedPatterns, isLoading: isLoadingPatterns } = useTradePatterns(
    selectedTrade?.backtestId ?? null,
    selectedTrade?.idx ?? null
  );

  // Default dates: Start 1 year ago, End today
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
        onSuccess: (res: any) => {
          addNotification({ type: 'success', message: 'Backtest complete' });
          setSelectedResult(res.data);
        },
        onError: () => addNotification({ type: 'error', message: 'Backtest failed' }),
      }
    );
  };

  const handleBulkRun = async () => {
    if (!stocks || stocks.length === 0) return;
    
    setIsBulkRunning(true);
    setShouldStopBulk(false);
    setBulkResults([]);
    setBulkProgress({ current: 0, total: stocks.length, symbol: '' });
    addNotification({ type: 'info', message: `Starting bulk backtest for ${stocks.length} stocks` });
    
    let successCount = 0;
    for (let i = 0; i < stocks.length; i++) {
      const stock = stocks[i];
      if (shouldStopBulk) break;

      setBulkProgress(prev => ({ ...prev!, current: i + 1, symbol: stock.symbol }));

      try {
        const res = await runBacktest.mutateAsync({
          model_type: modelType,
          knn_name: knnName || undefined,
          lstm_name: lstmName || undefined,
          stock_id: stock.id,
          start_date: startDate || undefined,
          end_date: endDate || undefined,
          initial_capital: parseFloat(initialCapital),
        });
        
        const resultItem = { ...res.data, symbol: stock.symbol };
        setBulkResults(prev => [resultItem, ...prev]);
        successCount++;
      } catch (err) {
        console.error(`Failed backtest for ${stock.symbol}`, err);
      }
    }
    
    const wasStopped = shouldStopBulk;
    setIsBulkRunning(false);
    setShouldStopBulk(false);
    setBulkProgress(null);

    if (wasStopped) {
      addNotification({ type: 'warning', message: `Bulk backtest stopped. Processed ${successCount} stocks.` });
    } else {
      addNotification({ 
        type: successCount === stocks.length ? 'success' : 'warning', 
        message: `Bulk backtest complete: ${successCount}/${stocks.length} succeeded` 
      });
    }
  };

  const handleStopBulk = () => {
    setShouldStopBulk(true);
  };

  useEffect(() => {
    setSelectedTrade(null);
  }, [selectedResult]);

  const { data: ohlcvData } = useOhlcv(
    selectedResult?.stock_id,
    selectedResult?.interval || 'day',
    selectedResult?.start_date || startDate,
    selectedResult?.end_date || endDate
  );

  const { data: indicatorData } = useIndicators(
    selectedResult?.stock_id,
    selectedResult?.interval || 'day',
    selectedResult?.start_date || startDate,
    selectedResult?.end_date || endDate
  );

  const chartIndicators = useMemo(() => {
    if (!indicatorData || indicatorData.length === 0 || !ohlcvData || ohlcvData.length === 0) return [];
    
    // Extract common indicators that are useful on the chart
    const indicators: any[] = [];
    const colors = ['#6366f1', '#eab308', '#ec4899', '#06b6d4'];
    
    // Get valid times from OHLCV data to ensure alignment
    const validTimes = new Set(ohlcvData.map((d: any) => d.date));
    
    // Filter indicators for ones that have data points in our timeframe
    const firstPoint = indicatorData[0];
    if (!firstPoint) return [];

    const availableKeys = Object.keys(firstPoint).filter(k => 
      (k.toLowerCase().includes('sma') || k.toLowerCase().includes('ema') || 
       k.toLowerCase().includes('bb') || k.toLowerCase().includes('kama') || 
       k.toLowerCase().includes('vwkama')) && k !== 'date'
    );

    availableKeys.slice(0, 4).forEach((key, idx) => {
      const series = indicatorData
        .filter((d: any) => validTimes.has(d.date) && d[key] != null)
        .map((d: any) => ({ time: d.date, value: d[key] }));
      
      if (series.length > 0) {
        indicators.push({
          name: key.toUpperCase(),
          color: colors[idx % colors.length],
          data: series
        });
      }
    });

    return indicators;
  }, [indicatorData, ohlcvData]);

  const tradeMarkers = useMemo(() => {
    if (!selectedResult?.trade_log || !ohlcvData || ohlcvData.length === 0) return [];
    
    const validTimes = new Set(ohlcvData.map((d: any) => d.date));
    const markers: any[] = [];
    
    selectedResult.trade_log.forEach((trade: any) => {
      // Add entry marker if it exists in OHLCV range
      if (validTimes.has(trade.entry_date)) {
        markers.push({
          time: trade.entry_date,
          side: 'BUY',
          price: trade.entry_price,
          reason: trade.action
        });
      }
      // Add exit marker if it exists in OHLCV range
      if (validTimes.has(trade.exit_date)) {
        markers.push({
          time: trade.exit_date,
          side: 'SELL',
          price: trade.exit_price,
          reason: trade.exit_reason
        });
      }
    });
    return markers;
  }, [selectedResult, ohlcvData]);

  const equityCurve = selectedResult?.equity_curve ?? [];

  return (
    <div className="space-y-8">
      <PageHeader title="Backtest" description="Test strategies against historical data" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="space-y-4">
          <Card title="Backtest Configuration" data-guide-id="backtest-config">
            <div className="space-y-5">
              <Select value={modelType} onChange={setModelType} label="Model Type" options={[
                { value: 'ensemble', label: 'Ensemble (KNN+LSTM)' },
                { value: 'knn', label: 'KNN Only' },
                { value: 'lstm', label: 'LSTM Only' },
                { value: 'rl', label: 'RL Model' },
              ]} />
              
              {['ensemble', 'knn'].includes(modelType) && (
                <Input value={knnName} onChange={setKnnName} label="KNN Model Name" placeholder="e.g. knn_9 (leave empty for latest)" />
              )}
              {['ensemble', 'lstm'].includes(modelType) && (
                <Input value={lstmName} onChange={setLstmName} label="LSTM Model Name" placeholder="e.g. lstm_9 (leave empty for latest)" />
              )}

              <SearchableSelect value={stockId} onChange={setStockId} options={stockOptions} label="Stock" placeholder="Search stocks..." />
              <div className="grid grid-cols-2 gap-4">
                <Input value={startDate} onChange={setStartDate} label="Start Date" type="date" />
                <Input value={endDate} onChange={setEndDate} label="End Date" type="date" />
              </div>
              <Input value={initialCapital} onChange={setInitialCapital} label="Initial Capital (₹)" type="number" />

              {isBulkRunning && bulkProgress && (
                <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)] border border-[var(--primary)]/20 animate-pulse">
                  <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--primary)] mb-1">
                    <span>Processing: {bulkProgress.symbol}</span>
                    <span>{bulkProgress.current} / {bulkProgress.total}</span>
                  </div>
                  <div className="w-full h-1 bg-[var(--bg-card)] rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-[var(--primary)] transition-all duration-300"
                      style={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <Button 
                onClick={handleRun} 
                loading={runBacktest.isPending && !isBulkRunning} 
                disabled={!stockId || isBulkRunning} 
                data-guide-id="run-backtest-btn" 
                className="w-full"
              >
                <BarChart3 size={14} /> Run Single Backtest
              </Button>
              
              <div className="flex gap-2">
                <Button 
                  onClick={handleBulkRun} 
                  loading={isBulkRunning && !shouldStopBulk} 
                  disabled={!stocks || stocks.length === 0 || isBulkRunning} 
                  variant="outline"
                  className="flex-1 border-dashed"
                >
                  <Play size={14} /> Backtest All
                </Button>

                {isBulkRunning && (
                  <Button 
                    onClick={handleStopBulk} 
                    variant="danger"
                    className="px-4"
                    title="Stop Bulk Run"
                  >
                    <Square size={14} fill="currentColor" />
                  </Button>
                )}
              </div>
            </div>
          </Card>

          {bulkResults.length > 0 && (
            <Card 
              title={`Recent Runs (${bulkResults.length})`}
              className="border-blue-500/30"
            >
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1 text-xs">
                {bulkResults.map((res: any, idx: number) => (
                  <ListItem
                    key={`${res.id}-${idx}`}
                    onClick={() => setSelectedResult(res)}
                    className={`!py-2 !px-3 border ${selectedResult?.id === res.id ? 'bg-[var(--primary-subtle)]/30 border-[var(--primary)]' : 'border-transparent'}`}
                    left={
                      <div className="flex flex-col">
                        <span className="font-bold text-[var(--text)]">{res.symbol}</span>
                        <span className="text-[9px] text-[var(--text-dim)] uppercase">{res.model_type}</span>
                      </div>
                    }
                    right={
                      <div className="text-right">
                        <div className={`font-mono font-bold ${res.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {(res.total_return * 100).toFixed(1)}%
                        </div>
                        <div className="text-[9px] text-[var(--text-dim)]">
                          {res.trades_count} trades
                        </div>
                      </div>
                    }
                  />
                ))}
              </div>
            </Card>
          )}
        </div>

        <div className="lg:col-span-2 space-y-4">
        {selectedResult ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
              <StatCard label="Total Return" value={`${(selectedResult.total_return * 100)?.toFixed(1)}%`}
                color={selectedResult.total_return >= 0 ? 'var(--success)' : 'var(--danger)'} />
              <StatCard label="Sharpe Ratio" value={selectedResult.sharpe_ratio?.toFixed(2) ?? '—'} color="var(--primary)" />
              <StatCard label="Max Drawdown" value={`${(selectedResult.max_drawdown * 100)?.toFixed(1)}%`} color="var(--danger)" />
              <StatCard label="Win Rate" value={`${(selectedResult.win_rate * 100)?.toFixed(0)}%`} color="var(--info)" />
            </div>

            {ohlcvData && (
              <Card title="Price Analysis & Trade Activity">
                <div className="-mx-2 mt-2">
                  <LightweightCandleChart
                    ohlcv={ohlcvData.map((d: any) => ({
                      time: d.date,
                      open: d.open,
                      high: d.high,
                      low: d.low,
                      close: d.close
                    }))}
                    indicators={chartIndicators}
                    trades={tradeMarkers}
                    height={500}
                  />
                </div>
              </Card>
            )}

            <Card title="Trade Logs">
              <div className="flex flex-col xl:flex-row gap-6">
                <div className={`flex-1 transition-all ${selectedTrade ? 'xl:w-2/3' : 'w-full'}`}>
              {selectedResult.trade_log && selectedResult.trade_log.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-[var(--border-light)] text-[var(--text-dim)] uppercase tracking-wider font-semibold">
                        <th className="py-2">Entry</th>
                        <th className="py-2">Exit</th>
                        <th className="py-2">Price In/Out</th>
                        <th className="py-2">Qty</th>
                        <th className="py-2 text-right">PnL%</th>
                        <th className="py-2 text-right">Reason</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-light)]/50">
                      {selectedResult.trade_log.map((trade: any, idx: number) => (
                        <tr 
                          key={idx} 
                          className={`hover:bg-[var(--bg-hover)]/30 transition-colors cursor-pointer ${selectedTrade?.idx === idx ? 'bg-[var(--primary)]/10 ring-1 ring-inset ring-[var(--primary)]/30' : ''}`}
                          onClick={() => setSelectedTrade({ backtestId: selectedResult.id, idx })}
                        >
                          <td className="py-2 tabular-nums">{trade.entry_date}</td>
                          <td className="py-2 tabular-nums">{trade.exit_date}</td>
                          <td className="py-2 tabular-nums">
                            <div className="font-medium text-[var(--text)]">₹{trade.entry_price.toFixed(1)}</div>
                            <div className="text-[var(--text-dim)]">₹{trade.exit_price.toFixed(1)}</div>
                          </td>
                          <td className="py-2 tabular-nums">{trade.quantity}</td>
                          <td className={`py-2 text-right tabular-nums font-bold ${trade.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                          </td>
                          <td className="py-2 text-right font-medium text-[var(--text-dim)] capitalize">
                            {trade.exit_reason?.replace('_', ' ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState icon={<BarChart3 size={24} />} title="No trades logged" description="No trades were executed during this backtest." />
              )}
                </div>

                {selectedTrade && (
                  <div className="xl:w-1/3 bg-[var(--bg-input)]/30 rounded-lg p-4 border border-[var(--border-light)]/50 animate-in fade-in slide-in-from-right-4 duration-300">
                    <div className="flex items-center gap-2 mb-4">
                      <Sparkles size={16} className="text-[var(--primary)]" />
                      <h4 className="font-bold text-sm tracking-tight">Golden Pattern Matches</h4>
                    </div>
                    
                    {isLoadingPatterns ? (
                      <div className="space-y-3">
                        <div className="h-10 bg-[var(--border-light)]/20 animate-pulse rounded" />
                        <div className="h-10 bg-[var(--border-light)]/20 animate-pulse rounded" />
                      </div>
                    ) : (matchedPatterns?.patterns && matchedPatterns.patterns.length > 0) ? (
                      <div className="space-y-3">
                        <p className="text-[10px] text-[var(--text-dim)] uppercase leading-relaxed font-bold">
                          The KNN model found these historical patterns most similar to this trade:
                        </p>
                        <div className="space-y-2">
                          {matchedPatterns.patterns.map((p: any) => (
                            <div key={p.id} className="p-2 rounded border border-[var(--border-light)]/50 bg-[var(--bg-card)]/50 flex justify-between items-center text-[11px]">
                              <div>
                                <div className="font-bold">{p.date}</div>
                                <div className="text-[9px] text-[var(--text-dim)] uppercase">Return: {(p.pnl_pct * 100).toFixed(1)}%</div>
                              </div>
                              <Badge color={p.label === 'BUY' ? 'emerald' : 'rose'}>{p.label}</Badge>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-6 text-[var(--text-dim)] italic text-xs">
                        No specific golden patterns matched. Trade likely driven by LSTM or ensemble logic.
                      </div>
                    )}
                    <button 
                      onClick={() => setSelectedTrade(null)}
                      className="mt-4 w-full py-2 text-[10px] uppercase tracking-wider font-extrabold text-[var(--text-dim)] hover:text-[var(--text)] transition-colors border-t border-[var(--border-light)]/30"
                    >
                      Close Analysis
                    </button>
                  </div>
                )}
              </div>
            </Card>
          </>
        ) : (
          <Card>
            <EmptyState
              icon={<BarChart3 size={36} />}
              title="No backtest results"
              description="Configure and run a backtest to see results."
            />
          </Card>
        )}
      </div>
      </div>

      {backtests && backtests.length > 0 && (
        <Card title="Previous Backtests">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {backtests.slice(0, 12).map((bt: any) => (
              <div 
                key={bt.id}
                onClick={() => setSelectedResult(bt)}
                className={`p-3 rounded-[var(--radius-sm)] border cursor-pointer transition-all hover:shadow-md ${
                  selectedResult?.id === bt.id 
                    ? 'border-[var(--primary)] bg-[var(--primary-subtle)]/10 shadow-sm' 
                    : 'border-[var(--border-light)] hover:border-[var(--text-dim)]/50'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex flex-col">
                    <span className="font-bold text-sm tracking-tight">{bt.symbol || `ID: ${bt.stock_id}`}</span>
                    <span className="text-[10px] uppercase text-[var(--text-dim)] font-semibold">{bt.model_type}</span>
                  </div>
                  <div className={`text-sm font-black tabular-nums ${bt.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {(bt.total_return * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="flex justify-between items-center text-[10px]">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[var(--text-dim)]">DRAWDOWN</span>
                    <span className="font-mono text-rose-300">{(bt.max_drawdown * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex flex-col gap-0.5 text-center">
                    <span className="text-[var(--text-dim)]">WIN RATE</span>
                    <span className="font-mono text-blue-300">{(bt.win_rate * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex flex-col gap-0.5 text-right">
                    <span className="text-[var(--text-dim)]">DATE</span>
                    <span className="font-mono">{new Date(bt.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
