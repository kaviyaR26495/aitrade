import { useState } from 'react';
import { Card, Button, Select, SearchableSelect, Input, Badge, StatCard, EmptyState, PageHeader, ListItem } from '../components/ui';
import { useUniverseStocks, useBacktestResults, useRunBacktest } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Area, AreaChart } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_AXIS_PROPS } from '../components/ChartTheme';
import { BarChart3 } from 'lucide-react';

export default function Backtest() {
  const { data: stocks } = useUniverseStocks();
  const { data: backtests } = useBacktestResults();
  const runBacktest = useRunBacktest();
  const { addNotification } = useAppStore();

  const [modelType, setModelType] = useState('ensemble');
  const [stockId, setStockId] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [selectedResult, setSelectedResult] = useState<any>(null);

  const stockOptions = [
    { value: '', label: 'Select stock...' },
    ...(stocks?.map((s: any) => ({ value: String(s.id), label: s.symbol })) ?? []),
  ];

  const handleRun = () => {
    runBacktest.mutate(
      {
        model_type: modelType,
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

  const equityCurve = selectedResult?.equity_curve ?? [];

  return (
    <div className="space-y-8">
      <PageHeader title="Backtest" description="Test strategies against historical data" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card title="Backtest Configuration" data-guide-id="backtest-config">
          <div className="space-y-5">
            <Select value={modelType} onChange={setModelType} label="Model Type" options={[
              { value: 'ensemble', label: 'Ensemble (KNN+LSTM)' },
              { value: 'knn', label: 'KNN Only' },
              { value: 'lstm', label: 'LSTM Only' },
              { value: 'rl', label: 'RL Model' },
            ]} />
            <SearchableSelect value={stockId} onChange={setStockId} options={stockOptions} label="Stock" placeholder="Search stocks..." />
            <div className="grid grid-cols-2 gap-4">
              <Input value={startDate} onChange={setStartDate} label="Start Date" type="date" />
              <Input value={endDate} onChange={setEndDate} label="End Date" type="date" />
            </div>
            <Input value={initialCapital} onChange={setInitialCapital} label="Initial Capital (₹)" type="number" />

            <Button onClick={handleRun} loading={runBacktest.isPending} disabled={!stockId} data-guide-id="run-backtest-btn" className="w-full">
              <BarChart3 size={14} /> Run Backtest
            </Button>
          </div>
        </Card>

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

              <Card title="Equity Curve">
                {equityCurve.length > 0 ? (
                  <div className="-mx-2 mt-2">
                    <ResponsiveContainer width="100%" height={340}>
                      <AreaChart data={equityCurve}>
                        <defs>
                          <linearGradient id="btEquityGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.15} />
                            <stop offset="100%" stopColor="var(--primary)" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="date" {...CHART_AXIS_PROPS} />
                        <YAxis {...CHART_AXIS_PROPS} tickFormatter={(v) => `₹${(v/1000).toFixed(0)}K`} />
                        <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                        <Legend />
                        <Area type="monotone" dataKey="equity" stroke="var(--primary)" strokeWidth={2} fill="url(#btEquityGrad)" name="Strategy" />
                        {equityCurve[0]?.benchmark && (
                          <Area type="monotone" dataKey="benchmark" stroke="var(--text-muted)" strokeWidth={1} fill="none" strokeDasharray="5 5" name="Buy & Hold" />
                        )}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState icon={<BarChart3 size={24} />} title="No equity curve data" description="Equity curve will appear after backtest completes." />
                )}
              </Card>

              <Card title="Trade Summary">
                <div className="grid grid-cols-3 gap-5 text-center">
                  <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
                    <div className="text-lg font-bold tabular-nums">{selectedResult.total_trades ?? 0}</div>
                    <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium">Total Trades</div>
                  </div>
                  <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
                    <div className="text-lg font-bold text-emerald-400 tabular-nums">{selectedResult.winning_trades ?? 0}</div>
                    <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium">Winning</div>
                  </div>
                  <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
                    <div className="text-lg font-bold text-rose-400 tabular-nums">{selectedResult.losing_trades ?? 0}</div>
                    <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium">Losing</div>
                  </div>
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
        <Card title="Backtest History">
          <div className="space-y-2">
            {backtests.map((bt: any) => (
              <ListItem
                key={bt.id}
                onClick={() => setSelectedResult(bt)}
                left={
                  <>
                    <Badge color={bt.model_type === 'ensemble' ? 'blue' : 'gray'}>{bt.model_type}</Badge>
                    <span className="font-medium">Stock #{bt.stock_id}</span>
                  </>
                }
                right={
                  <div className="flex items-center gap-3 text-xs">
                    <span className={`font-medium tabular-nums ${bt.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {(bt.total_return * 100).toFixed(1)}%
                    </span>
                    <span className="text-[var(--text-dim)] tabular-nums">Sharpe: {bt.sharpe_ratio?.toFixed(2)}</span>
                  </div>
                }
              />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
