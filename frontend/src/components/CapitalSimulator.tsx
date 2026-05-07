import { useState, useEffect, useMemo } from 'react';
import { Card, Button, Input, PageHeader, Select, Table, Badge } from './ui';
import { useRunCompoundBacktest, useCompoundBacktests, useCompoundBacktestDetail, useDeleteCompoundBacktest } from '../hooks/useApi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_AXIS_PROPS } from './ChartTheme';
import { useAppStore } from '../store/appStore';
import { Play, Trash2, ChevronDown, ChevronUp } from 'lucide-react';

export default function CapitalSimulator() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [stoplossPct, setStoplossPct] = useState('5.0');
  const [targetPct, setTargetPct] = useState('10.0');
  const [buyRegimeId, setBuyRegimeId] = useState<string>('');
  const [sellRegimeId, setSellRegimeId] = useState<string>('');
  const [maxPositions, setMaxPositions] = useState('10');
  
  const [results, setResults] = useState<any>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [showEquity, setShowEquity] = useState(true);
  const [showTradeLog, setShowTradeLog] = useState(false);
  const [showDailyTx, setShowDailyTx] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  
  const compoundBacktest = useRunCompoundBacktest();
  const historyQuery = useCompoundBacktests();
  const detailQuery = useCompoundBacktestDetail(selectedId);
  const deleteMutation = useDeleteCompoundBacktest();
  const { addNotification } = useAppStore();

  const handleRun = () => {
    setSelectedId(null); // Clear loaded id
    compoundBacktest.mutate(
      {
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        initial_capital: Number(initialCapital),
        stoploss_pct: Number(stoplossPct),
        target_pct: Number(targetPct),
        buy_regime_id: buyRegimeId ? Number(buyRegimeId) : undefined,
        sell_regime_id: sellRegimeId ? Number(sellRegimeId) : undefined,
        max_positions_per_day: Number(maxPositions)
      },
      {
        onSuccess: (data) => {
          setResults(data);
          addNotification({ type: 'success', message: 'Simulation complete' });
        },
        onError: () => {
          addNotification({ type: 'error', message: 'Simulation failed' });
        }
      }
    );
  };
  
  const activeResults = results || detailQuery.data;

  // When loading historical, fill the form fields
  useEffect(() => {
    if (detailQuery.data?.parameters) {
      const p = detailQuery.data.parameters;
      if (p.start_date) setStartDate(p.start_date);
      if (p.end_date) setEndDate(p.end_date);
      if (p.initial_capital) setInitialCapital(String(p.initial_capital));
      if (p.stoploss_pct) setStoplossPct(String(p.stoploss_pct));
      if (p.target_pct) setTargetPct(String(p.target_pct));
      if (p.buy_regime_id !== undefined && p.buy_regime_id !== null) setBuyRegimeId(String(p.buy_regime_id));
      if (p.sell_regime_id !== undefined && p.sell_regime_id !== null) setSellRegimeId(String(p.sell_regime_id));
      if (p.max_positions_per_day) setMaxPositions(String(p.max_positions_per_day));
    }
  }, [detailQuery.data]);

  const profitPct = useMemo(() => {
    if (!activeResults) return 0;
    const ic = activeResults.initial_capital || Number(initialCapital) || 1;
    return (activeResults.profit_booked / ic) * 100;
  }, [activeResults, initialCapital]);

  const holdingsValue = useMemo(() => {
    if (!activeResults?.trade_log) return 0;
    return activeResults.trade_log
      .filter((t: any) => t.exit_reason === 'end_of_backtest')
      .reduce((sum: number, t: any) => sum + t.quantity * t.exit_price, 0);
  }, [activeResults]);

  const dailyTxs = useMemo(() => {
    if (!activeResults?.trade_log) return [];
    type Tx = { date: string; type: 'BUY' | 'SELL'; symbol: string; price: number; qty: number; value: number; pnl?: number; reason?: string };
    const txs: Tx[] = [];
    for (const trade of activeResults.trade_log) {
      txs.push({ date: trade.entry_date, type: 'BUY', symbol: trade.symbol, price: trade.entry_price, qty: trade.quantity, value: trade.entry_price * trade.quantity });
      txs.push({ date: trade.exit_date, type: 'SELL', symbol: trade.symbol, price: trade.exit_price, qty: trade.quantity, value: trade.exit_price * trade.quantity, pnl: trade.pnl, reason: trade.exit_reason });
    }
    return txs.sort((a, b) => a.date.localeCompare(b.date) || (a.type === 'BUY' ? -1 : 1));
  }, [activeResults]);

  const equityData = activeResults?.equity_curve
    ? activeResults.equity_curve.map((p: any) => ({ date: p.date, value: Math.round(p.value) }))
    : [];
  const initialCap = activeResults?.initial_capital || Number(initialCapital) || 0;

  const columns = [
    { key: 'entry_date', label: 'Entry Date' },
    { key: 'exit_date', label: 'Exit Date' },
    { key: 'symbol', label: 'Symbol' },
    { key: 'entry_price', label: 'Entry Price', align: 'right' as const },
    { key: 'exit_price', label: 'Exit Price', align: 'right' as const },
    { key: 'quantity', label: 'Qty', align: 'right' as const },
    { key: 'exit_reason', label: 'Reason', align: 'center' as const, render: (s: any) => <Badge color="gray">{s.exit_reason}</Badge> },
    { 
      key: 'pnl', 
      label: 'PnL (₹)', 
      align: 'right' as const,
      render: (s: any) => <span className={s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>{s.pnl.toFixed(2)}</span>
    },
    { 
      key: 'pnl_pct', 
      label: 'PnL %', 
      align: 'right' as const,
      render: (s: any) => <span className={s.pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>{s.pnl_pct.toFixed(2)}%</span>
    },
  ];

  const txColumns = [
    { key: 'date', label: 'Date' },
    { key: 'type', label: 'Type', render: (t: any) => <Badge color={t.type === 'BUY' ? 'green' : 'red'}>{t.type}</Badge> },
    { key: 'symbol', label: 'Symbol' },
    { key: 'price', label: 'Price (₹)', align: 'right' as const, render: (t: any) => `₹${t.price.toFixed(2)}` },
    { key: 'qty', label: 'Qty', align: 'right' as const },
    { key: 'value', label: 'Value (₹)', align: 'right' as const, render: (t: any) => `₹${t.value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` },
    { key: 'pnl', label: 'P&L (₹)', align: 'right' as const, render: (t: any) => t.pnl !== undefined ? <span className={t.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>{t.pnl.toFixed(2)}</span> : <span className="text-[var(--text-muted)]">—</span> },
    { key: 'reason', label: 'Reason', align: 'center' as const, render: (t: any) => t.reason ? <Badge color="gray">{t.reason.replace(/_/g, ' ')}</Badge> : <span className="text-[var(--text-muted)]">—</span> },
  ];

  const historyColumns = [
    { key: 'id', label: 'ID' },
    { key: 'created_at', label: 'Date Run', render: (r: any) => new Date(r.created_at).toLocaleString() },
    { key: 'start_date', label: 'Start' },
    { key: 'end_date', label: 'End' },
    { key: 'initial_capital', label: 'Initial (₹)', align: 'right' as const, render: (r: any) => r.initial_capital.toLocaleString() },
    { key: 'final_capital', label: 'Final (₹)', align: 'right' as const, render: (r: any) => r.final_capital.toLocaleString() },
    { key: 'profit_booked', label: 'Profit (₹)', align: 'right' as const, render: (r: any) => <span className={r.profit_booked >= 0 ? 'text-emerald-400' : 'text-red-400'}>{r.profit_booked.toLocaleString()}</span> },
    { key: 'profit_pct', label: 'Profit %', align: 'right' as const, render: (r: any) => { const pct = r.initial_capital > 0 ? (r.profit_booked / r.initial_capital * 100) : 0; return <span className={pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>{pct.toFixed(2)}%</span>; } },
    { key: 'total_trades', label: 'Trades', align: 'center' as const },
    { 
      key: 'actions', 
      label: '', 
      align: 'right' as const,
      render: (r: any) => (
        <div className="flex justify-end gap-2">
          <Button size="sm" onClick={() => { setResults(null); setSelectedId(r.id); }}>Load</Button>
          <Button size="sm" variant="ghost" className="text-red-400 hover:text-red-300 hover:bg-red-500/10" onClick={() => deleteMutation.mutate(r.id)}>
            <Trash2 size={14} />
          </Button>
        </div>
      ) 
    }
  ];

  const regimeOptions = [
    { value: '', label: 'Any Regime' },
    { value: '0', label: '0: Bull+LowVol' },
    { value: '1', label: '1: Bull+HighVol' },
    { value: '2', label: '2: Bear+LowVol' },
    { value: '3', label: '3: Bear+HighVol' },
    { value: '4', label: '4: Neutral+LowVol' },
    { value: '5', label: '5: Neutral+HighVol' },
  ];

  return (
    <div className="space-y-6">
      <Card title="Simulation Parameters">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-36">
            <Input value={startDate} onChange={setStartDate} label="Start" type="date" />
          </div>
          <div className="w-36">
            <Input value={endDate} onChange={setEndDate} label="End" type="date" />
          </div>
          <div className="w-32">
            <Input value={initialCapital} onChange={setInitialCapital} label="Initial Capital (₹)" type="number" />
          </div>
          <div className="w-24">
            <Input value={stoplossPct} onChange={setStoplossPct} label="SL %" type="number" step="0.1" />
          </div>
          <div className="w-24">
            <Input value={targetPct} onChange={setTargetPct} label="Target %" type="number" step="0.1" />
          </div>
          <div className="w-32">
            <Select value={buyRegimeId} onChange={setBuyRegimeId} label="Buy Regime" options={regimeOptions} />
          </div>
          <div className="w-32">
            <Select value={sellRegimeId} onChange={setSellRegimeId} label="Sell Regime" options={regimeOptions} />
          </div>
          <div className="w-24">
            <Input value={maxPositions} onChange={setMaxPositions} label="Max Buys/Day" type="number" />
          </div>
          <Button onClick={handleRun} loading={compoundBacktest.isPending} icon={<Play size={16} />}>
            Run Simulator
          </Button>
        </div>
      </Card>

      {activeResults && (
        <>
          {activeResults.coverage_warning && (
            <div className="flex items-start gap-3 rounded-[var(--radius)] border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              <span className="mt-0.5 text-amber-400 shrink-0">⚠</span>
              <span>{activeResults.coverage_warning}</span>
            </div>
          )}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-5">
            <Card className="p-5 flex flex-col items-center justify-center">
              <span className="text-[var(--text-dim)] uppercase text-xs font-bold tracking-wider">Final Account Val</span>
              <span className="text-3xl font-bold font-mono mt-2 text-[var(--primary)]">₹{activeResults.final_capital.toLocaleString('en-IN')}</span>
            </Card>
            <Card className="p-5 flex flex-col items-center justify-center">
              <span className="text-[var(--text-dim)] uppercase text-xs font-bold tracking-wider">Total Profit Booked</span>
              <span className={`text-3xl font-bold font-mono mt-2 ${activeResults.profit_booked >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                ₹{activeResults.profit_booked.toLocaleString('en-IN')}
              </span>
            </Card>
            <Card className="p-5 flex flex-col items-center justify-center">
              <span className="text-[var(--text-dim)] uppercase text-xs font-bold tracking-wider">Profit %</span>
              <span className={`text-3xl font-bold font-mono mt-2 ${profitPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {profitPct.toFixed(2)}%
              </span>
            </Card>
            <Card className="p-5 flex flex-col items-center justify-center">
              <span className="text-[var(--text-dim)] uppercase text-xs font-bold tracking-wider">Holdings Value (End)</span>
              <span className="text-3xl font-bold font-mono mt-2 text-amber-400">₹{holdingsValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
            </Card>
            <Card className="p-5 flex flex-col items-center justify-center">
              <span className="text-[var(--text-dim)] uppercase text-xs font-bold tracking-wider">Total Trades</span>
              <span className="text-3xl font-bold font-mono mt-2">{activeResults.total_trades}</span>
            </Card>
          </div>

          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
            <button
              className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-[var(--text)] hover:bg-[var(--surface-hover)] transition-colors rounded-[var(--radius)]"
              onClick={() => setShowEquity(v => !v)}
            >
              <span>Equity Curve</span>
              {showEquity ? <ChevronUp size={16} className="text-[var(--text-dim)]" /> : <ChevronDown size={16} className="text-[var(--text-dim)]" />}
            </button>
            {showEquity && (
              <div className="h-[320px] border-t border-[var(--border)] pt-2 pr-4">
                {equityData.length === 0 ? (
                  <div className="flex h-full items-center justify-center text-sm text-[var(--text-dim)]">No equity data.</div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={equityData} margin={{ top: 10, right: 8, left: 8, bottom: 0 }}>
                      <defs>
                        <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="date" {...CHART_AXIS_PROPS} tick={{ ...CHART_AXIS_PROPS.tick, fontSize: 11 }} />
                      <YAxis {...CHART_AXIS_PROPS} tickFormatter={(v: number) => `₹${(v/1000).toFixed(0)}k`} width={56} />
                      <Tooltip
                        contentStyle={CHART_TOOLTIP_STYLE}
                        formatter={(v: number) => [`₹${v.toLocaleString('en-IN')}`, 'Equity']}
                      />
                      {initialCap > 0 && (
                        <ReferenceLine y={initialCap} stroke="#64748b" strokeDasharray="4 3" label={{ value: 'Start', fill: '#64748b', fontSize: 10, position: 'insideTopRight' }} />
                      )}
                      <Area type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} fill="url(#equityGrad)" dot={false} activeDot={{ r: 4 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}
          </div>

          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
            <button
              className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-[var(--text)] hover:bg-[var(--surface-hover)] transition-colors rounded-[var(--radius)]"
              onClick={() => setShowTradeLog(v => !v)}
            >
              <span>Trade Log <span className="text-[var(--text-dim)] font-normal">({(activeResults.trade_log || []).length} trades)</span></span>
              {showTradeLog ? <ChevronUp size={16} className="text-[var(--text-dim)]" /> : <ChevronDown size={16} className="text-[var(--text-dim)]" />}
            </button>
            {showTradeLog && (
              <div className="border-t border-[var(--border)]">
                <Table data={activeResults.trade_log || []} columns={columns} compact />
              </div>
            )}
          </div>

          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
            <button
              className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-[var(--text)] hover:bg-[var(--surface-hover)] transition-colors rounded-[var(--radius)]"
              onClick={() => setShowDailyTx(v => !v)}
            >
              <span>Daily Transactions <span className="text-[var(--text-dim)] font-normal">({dailyTxs.length} entries)</span></span>
              {showDailyTx ? <ChevronUp size={16} className="text-[var(--text-dim)]" /> : <ChevronDown size={16} className="text-[var(--text-dim)]" />}
            </button>
            {showDailyTx && (
              <div className="border-t border-[var(--border)]">
                {dailyTxs.length === 0 ? (
                  <div className="p-8 text-center text-sm text-[var(--text-dim)]">No transactions found.</div>
                ) : (
                  <Table data={dailyTxs} columns={txColumns as any} compact />
                )}
              </div>
            )}
          </div>
        </>
      )}

      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
        <button
          className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-[var(--text)] hover:bg-[var(--surface-hover)] transition-colors rounded-[var(--radius)]"
          onClick={() => setShowHistory(v => !v)}
        >
          <span>Simulation History</span>
          {showHistory ? <ChevronUp size={16} className="text-[var(--text-dim)]" /> : <ChevronDown size={16} className="text-[var(--text-dim)]" />}
        </button>
        {showHistory && (
          <div className="border-t border-[var(--border)]">
            {historyQuery.isLoading ? (
              <div className="p-8 text-center text-sm text-[var(--text-dim)]">Loading history...</div>
            ) : historyQuery.data?.length === 0 ? (
              <div className="p-8 text-center text-sm text-[var(--text-dim)]">No compound simulations run yet.</div>
            ) : (
              <Table data={historyQuery.data || []} columns={historyColumns as any} compact />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
