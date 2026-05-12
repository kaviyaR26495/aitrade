import { useMemo, useState } from 'react';
import { Card, Button, Badge, StatCard, Select, EmptyState, PageHeader, Table, Modal, Checkbox, Tooltip, SkeletonTable, type TableColumn } from '../components/ui';
import { usePredictions, useOrders, useRunPredictions, usePlaceOrder, useUniverseStocks, useOhlcv, useIndicators, useBatches, useDeleteBatch, usePredictionJob, useCancelPredictionJob, useForwardLook } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Crosshair, Play, Shield, ShieldAlert, Trash2, XCircle, Loader2, Filter } from 'lucide-react';
import { useEffect } from 'react';
import LightweightCandleChart, { type IndicatorSeries } from '../components/LightweightCandleChart';

const REGIME_HELP: Record<number, string> = {
  0: 'Bull trend, low volatility',
  1: 'Bull trend, high volatility',
  2: 'Bear trend, low volatility',
  3: 'Bear trend, high volatility',
  4: 'Neutral trend, low volatility',
  5: 'Neutral trend, high volatility',
};

type SignalFilter = 'ALL' | 'BUY' | 'SELL' | 'HOLD';
type RegimeFilter = 'ALL' | 'UNKNOWN' | '0' | '1' | '2' | '3' | '4' | '5';
type ConfidenceOperator = '>' | '>=' | '<' | '<=' | '=' | '!=';

export default function LiveTrading() {
  const { addNotification } = useAppStore();
  const [interval, setInterval] = useState('day');
  const minConfidence = 0;
  const [agreementOnly, setAgreementOnly] = useState(true);
  const [targetDate, setTargetDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [forwardLookTarget, setForwardLookTarget] = useState<{stockId: number, symbol: string, date: string} | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [signalFilter, setSignalFilter] = useState<SignalFilter>('ALL');
  const [regimeFilter, setRegimeFilter] = useState<RegimeFilter>('ALL');
  const [confidenceOperator, setConfidenceOperator] = useState<ConfidenceOperator>('>=');
  const [confidenceValue, setConfidenceValue] = useState('65');
  const [symbolQuery, setSymbolQuery] = useState('');

  const { data: batches } = useBatches(interval);
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(null);
  const deleteBatch = useDeleteBatch();

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const { data: job } = usePredictionJob(activeJobId);
  const cancelJob = useCancelPredictionJob();

  useEffect(() => {
    if (job?.batch_id && !selectedBatchId) {
      setSelectedBatchId(job.batch_id);
    }

    if (job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled') {
       if (job.status === 'completed') {
         addNotification({ type: 'success', message: 'Prediction session completed successfully' });
       } else if (job.status === 'cancelled') {
         addNotification({ type: 'warning', message: 'Prediction run stopped by user' });
       } else if (job.status === 'failed') {
         addNotification({ type: 'error', message: `Prediction failed: ${job.error}` });
       }
       setIsSubmitting(false);  // re-enable button once job finishes
       setTimeout(() => setActiveJobId(null), 3000);
    }
  }, [job, addNotification, selectedBatchId]);

  const { data: predictions, isLoading } = usePredictions({
    interval,
    batch_id: selectedBatchId || undefined,
    min_confidence: minConfidence,
    agreement_only: agreementOnly,
  }, {
    enabled: !!selectedBatchId || (batches !== undefined && batches.length === 0),
    refetchInterval: !!activeJobId && job?.status === 'running' ? 2000 : false
  });

  const { data: universeStocks } = useUniverseStocks();
  const { data: orders } = useOrders(20);
  const runPredictions = useRunPredictions();
  const placeOrder = usePlaceOrder();

  // Auto-select latest batch on first load or when a new one is created
  useEffect(() => {
    if (batches && batches.length > 0 && !selectedBatchId) {
      setSelectedBatchId(batches[0].batch_id);
    }
  }, [batches, selectedBatchId]);

  const [orderConfirm, setOrderConfirm] = useState<any>(null);
  const [selectedStock, setSelectedStock] = useState<{ id: number; symbol: string; date: string; action: string } | null>(null);

  // Stock selection for targeted prediction runs
  const [showStockSelector, setShowStockSelector] = useState(false);
  const [selectedStockIds, setSelectedStockIds] = useState<Set<number>>(new Set());
  const [stockSearchQuery, setStockSearchQuery] = useState('');

  const filteredPredictions = useMemo(() => {
    const all = predictions ?? [];
    const q = symbolQuery.trim().toLowerCase();
    const parsedConfidence = Number(confidenceValue);
    const hasConfidenceFilter = confidenceValue.trim().length > 0 && Number.isFinite(parsedConfidence);
    const threshold = hasConfidenceFilter
      ? Math.max(0, Math.min(1, parsedConfidence > 1 ? parsedConfidence / 100 : parsedConfidence))
      : 0;

    return all.filter((p: any) => {
      const action = String(p.action ?? '').toUpperCase();
      const regimeId = Number.isInteger(p.regime_id) ? String(Number(p.regime_id)) : 'UNKNOWN';
      const confidence = Number(p.confidence ?? 0);
      const matchesSignal = signalFilter === 'ALL' || action === signalFilter;
      const matchesRegime = regimeFilter === 'ALL' || regimeId === regimeFilter;
      const matchesConfidence = !hasConfidenceFilter || (
        confidenceOperator === '>' ? confidence > threshold :
        confidenceOperator === '>=' ? confidence >= threshold :
        confidenceOperator === '<' ? confidence < threshold :
        confidenceOperator === '<=' ? confidence <= threshold :
        confidenceOperator === '=' ? confidence === threshold :
        confidence !== threshold
      );
      const matchesSymbol = !q || String(p.symbol ?? '').toLowerCase().includes(q);
      return matchesSignal && matchesRegime && matchesConfidence && matchesSymbol;
    });
  }, [predictions, signalFilter, regimeFilter, confidenceOperator, confidenceValue, symbolQuery]);

  const buyPreds = filteredPredictions.filter((p: any) => String(p.action).toUpperCase() === 'BUY');
  const sellPreds = filteredPredictions.filter((p: any) => String(p.action).toUpperCase() === 'SELL');
  const holdPreds = filteredPredictions.filter((p: any) => String(p.action).toUpperCase() === 'HOLD');

  const predictionColumns: TableColumn<any>[] = [
    {
      key: 'symbol',
      label: 'Symbol',
      tooltip: 'Stock ticker symbol',
      render: (p) => <span className="font-medium">{p.symbol}</span>
    },
    {
      key: 'action',
      label: 'Signal',
      tooltip: 'Final ensemble recommendation (BUY, SELL, or HOLD)',
      render: (p) => {
        const action = String(p.action).toUpperCase();
        return <Badge color={action === 'BUY' ? 'green' : action === 'SELL' ? 'red' : 'gray'}>{action}</Badge>;
      }
    },
    {
      key: 'confidence',
      label: 'Confidence',
      tooltip: 'Aggregate weighted confidence level across all active models',
      align: 'center',
      render: (p) => (
        <span className={`font-medium tabular-nums ${p.confidence >= 0.8 ? 'text-emerald-400' : p.confidence >= 0.65 ? 'text-amber-400' : 'text-[var(--text-muted)]'}`}>
          {(p.confidence * 100).toFixed(0)}%
        </span>
      )
    },
    {
      key: 'knn',
      label: 'KNN',
      tooltip: 'K-Nearest Neighbors pattern matching prediction and confidence',
      align: 'center',
      render: (p) => {
        const action = String(p.knn_action || 'HOLD').toUpperCase();
        return (
          <span className="text-xs">
            <Badge color={action === 'BUY' ? 'green' : action === 'SELL' ? 'red' : 'gray'}>{action}</Badge>
            <span className="ml-1 text-[var(--text-dim)] tabular-nums">{(p.knn_confidence * 100).toFixed(0)}%</span>
          </span>
        );
      }
    },
    {
      key: 'lstm',
      label: 'LSTM',
      tooltip: 'Long Short-Term Memory neural network prediction and confidence',
      align: 'center',
      render: (p) => {
        const action = String(p.lstm_action || 'HOLD').toUpperCase();
        return (
          <span className="text-xs">
            <Badge color={action === 'BUY' ? 'green' : action === 'SELL' ? 'red' : 'gray'}>{action}</Badge>
            <span className="ml-1 text-[var(--text-dim)] tabular-nums">{(p.lstm_confidence * 100).toFixed(0)}%</span>
          </span>
        );
      }
    },
    {
      key: 'agreement',
      label: 'Agreement',
      tooltip: 'Indicates if both models agree on the signal direction',
      align: 'center',
      render: (p) => (p.agreement ? <Badge color="green">✓</Badge> : <Badge color="yellow">⚠</Badge>)
    },
    {
      key: 'regime',
      label: 'Regime',
      tooltip: 'Detected market regime ID at prediction time',
      align: 'center',
      mono: true,
      render: (p) => {
        const regimeId = Number.isInteger(p.regime_id) ? Number(p.regime_id) : null;
        const regimeCode = regimeId === null ? 'R?' : `R${regimeId}`;
        const regimeText = regimeId === null
          ? 'Regime unavailable for this prediction'
          : (REGIME_HELP[regimeId] ?? 'Unknown regime mapping');

        return (
        <Tooltip content={`${regimeCode}: ${regimeText}`} side="bottom">
          <span className="text-xs text-[var(--text-dim)] cursor-help border-b border-dotted border-[var(--border)]">
            R{p.regime_id ?? '—'}
          </span>
        </Tooltip>
        );
      }
    },
    {
      key: 'pattern',
      label: '',
      align: 'right',
      stopPropagation: true,
      render: (p) => (
        <div
          onClick={() => setSelectedStock({ id: p.stock_id, symbol: p.symbol, date: p.date, action: p.action })}
          className="text-[var(--text-dim)] hover:text-[var(--primary)] transition-colors cursor-pointer"
        >
          <Crosshair size={14} />
        </div>
      )
    },
    {
      key: 'outcome',
      label: 'Outcome (5D)',
      tooltip: 'Actual market movement for the 5 days following this prediction',
      align: 'center',
      stopPropagation: true,
      render: (p) => {
        const predDate = new Date(p.date + 'T00:00:00');
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - 5);
        const hasOutcome = predDate < cutoff;
        return (
          <Tooltip content={hasOutcome ? 'View 5-day outcome' : 'Outcome available after 5 trading days'} side="left">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => setForwardLookTarget({ stockId: p.stock_id, symbol: p.symbol, date: p.date })}
              disabled={!hasOutcome}
              className="h-7 w-7 p-0 text-[var(--text-dim)] hover:text-[var(--primary)] hover:bg-[var(--primary-dim)] disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <Play size={12} className="rotate-90" />
            </Button>
          </Tooltip>
        );
      }
    }
  ];

  const handleRunPredictions = () => {
    if (isSubmitting || !!activeJobId) return;  // guard against rapid clicks
    setIsSubmitting(true);
    const stockIds = selectedStockIds.size > 0
      ? Array.from(selectedStockIds)
      : universeStocks?.map((s: any) => s.id);
    runPredictions.mutate(
      { interval, agreement_required: agreementOnly, stock_ids: stockIds, target_date: targetDate },
      {
        onSuccess: (res: any) => {
          setActiveJobId(res.job_id);
          if (res.batch_id) setSelectedBatchId(res.batch_id);
          addNotification({ type: 'info', message: 'Started background prediction process' });
        },
        onError: () => {
          addNotification({ type: 'error', message: 'Failed to start prediction run' });
          setIsSubmitting(false);  // re-enable on error so user can retry
        },
        onSettled: () => {
          // isSubmitting stays true while job is active; cleared when job ends
        }
      }
    );
  };

  const confirmOrder = () => {
    if (!orderConfirm) return;
    placeOrder.mutate(
      {
        stock_id: orderConfirm.stock_id,
        ensemble_prediction_id: orderConfirm.id,
        transaction_type: orderConfirm.action,
        quantity: 10,
      },
      {
        onSuccess: () => {
          addNotification({ type: 'success', message: `Order placed for ${orderConfirm.symbol}` });
          setOrderConfirm(null);
        },
        onError: () => addNotification({ type: 'error', message: 'Order failed' }),
      }
    );
  };

  return (
    <div className="space-y-8">
      <PageHeader title="Live Trading" description="Generate predictions and place orders">
        <div className="flex gap-2 items-center">
          {batches && batches.length > 0 && (
            <div className="flex items-center gap-1 bg-[var(--bg-input)] p-1 rounded-[var(--radius-sm)] border border-[var(--border)] shadow-sm">
              <Select 
                value={selectedBatchId || ''} 
                onChange={setSelectedBatchId} 
                options={batches.map((b: any) => ({
                  value: b.batch_id,
                  label: `${b.target_date ? new Date(b.target_date + 'T00:00:00').toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) + ' · ' : ''}${new Date(b.run_at).toLocaleString('en-IN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' })} (${b.stock_count} stocks)`
                }))} 
                className="w-52 border-none bg-transparent hover:bg-[var(--bg-hover)] transition-colors text-xs font-medium" 
              />
              <Tooltip content="Delete Session" side="bottom">
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => {
                    if (selectedBatchId && confirm('Delete this prediction session?')) {
                      deleteBatch.mutate(selectedBatchId, {
                        onSuccess: () => setSelectedBatchId(null)
                      });
                    }
                  }}
                  loading={deleteBatch.isPending}
                  className="h-7 w-7 p-0 text-[var(--text-dim)] hover:text-red-400 hover:bg-red-500/10"
                >
                  <Trash2 size={14} />
                </Button>
              </Tooltip>
            </div>
          )}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius-sm)] shadow-sm">
            <span className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider">Date</span>
            <input 
              type="date" 
              value={targetDate} 
              onChange={(e) => setTargetDate(e.target.value)}
              className="bg-transparent border-none p-0 text-xs font-medium focus:ring-0 cursor-pointer w-[110px]"
            />
          </div>

          <div className="py-1.5 px-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)] shadow-sm border border-[var(--border)]">
            <Checkbox checked={agreementOnly} onChange={setAgreementOnly} label="Agreement Only" />
          </div>
          
          {activeJobId && (
            <div className="flex items-center gap-3 px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--primary-dim)] rounded-[var(--radius-sm)] shadow-sm animate-in fade-in slide-in-from-right-4">
              <div className="flex flex-col gap-0.5 min-w-[120px]">
                <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-dim)]">
                  <span>Progress</span>
                  <span>{job ? Math.max(job.progress, 1) : 0}%</span>
                </div>
                <div className="w-full bg-[var(--bg-card)] h-1.5 rounded-full overflow-hidden border border-[var(--border)]">
                  <div 
                    className="h-full bg-[var(--primary)] transition-all duration-500 ease-out" 
                    style={{ width: `${job ? Math.max(job.progress, 2) : 2}%` }} 
                  />
                </div>
              </div>
              
              <div className="h-4 w-px bg-[var(--border)] mx-1" />
              
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => {
                   if (confirm('Stop the current prediction run?')) {
                     cancelJob.mutate(activeJobId);
                   }
                }}
                className="h-7 px-2 text-xs text-red-400 hover:bg-red-500/10 gap-1.5"
                loading={cancelJob.isPending}
              >
                {cancelJob.isPending ? <Loader2 className="animate-spin" size={12} /> : <XCircle size={12} />}
                Stop
              </Button>
            </div>
          )}

          <Tooltip content="Select specific stocks to predict (default: all universe stocks)" side="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowStockSelector(true)}
              disabled={isSubmitting || !!activeJobId}
              icon={<Filter size={14} />}
              className="h-9 gap-1.5 border border-[var(--border)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]"
            >
              {selectedStockIds.size > 0 ? (
                <span className="text-[var(--primary)] font-semibold">{selectedStockIds.size} selected</span>
              ) : (
                <span className="text-[var(--text-dim)]">All stocks</span>
              )}
            </Button>
          </Tooltip>

          <Button 
            variant="primary" 
            onClick={handleRunPredictions} 
            icon={<Play size={16} />}
            loading={isSubmitting || (!!activeJobId && job?.status === 'running')}
            disabled={isSubmitting || !!activeJobId}
          >
            {isSubmitting || activeJobId ? 'Processing...' : 'Run Predictions'}
          </Button>
        </div>
      </PageHeader>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard label="BUY Signals" value={buyPreds.length} color="var(--success)" />
        <StatCard label="SELL Signals" value={sellPreds.length} color="var(--danger)" />
        <StatCard label="HOLD Signals" value={holdPreds.length} color="var(--warning)" />
        <StatCard label="Total Predictions" value={filteredPredictions.length} color="var(--primary)" />
        <StatCard label="Confidence Rule" value={`${confidenceOperator} ${confidenceValue || '—'}%`} color="var(--info)" />
      </div>

      <Card title="Signal Filters" className="p-4">
        <div className="flex flex-col lg:flex-row lg:items-center gap-3">
          <div className="flex items-center gap-2">
            {(['ALL', 'BUY', 'SELL', 'HOLD'] as SignalFilter[]).map((signal) => {
              const isActive = signalFilter === signal;
              return (
                <Button
                  key={signal}
                  size="sm"
                  variant={isActive ? 'primary' : 'ghost'}
                  onClick={() => setSignalFilter(signal)}
                  className={isActive ? '' : 'text-[var(--text-dim)]'}
                >
                  {signal}
                </Button>
              );
            })}
          </div>

          <div className="flex items-center gap-2 w-full lg:w-auto">
            <span className="text-xs text-[var(--text-dim)] font-medium uppercase tracking-wider">Confidence</span>
            <Select
              value={confidenceOperator}
              onChange={(v) => setConfidenceOperator(v as ConfidenceOperator)}
              options={[
                { value: '>', label: '>' },
                { value: '>=', label: '>=' },
                { value: '<', label: '<' },
                { value: '<=', label: '<=' },
                { value: '=', label: '=' },
                { value: '!=', label: '!=' },
              ]}
              className="w-[90px]"
            />
            <input
              type="number"
              min="0"
              max="100"
              step="1"
              value={confidenceValue}
              onChange={(e) => setConfidenceValue(e.target.value)}
              placeholder="e.g. 65"
              className="h-9 w-[120px] rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
              aria-label="Confidence threshold percent"
            />
            <span className="text-xs text-[var(--text-dim)]">%</span>
          </div>

          <div className="flex items-center gap-2 w-full lg:w-auto">
            <span className="text-xs text-[var(--text-dim)] font-medium uppercase tracking-wider">Regime</span>
            <Select
              value={regimeFilter}
              onChange={(v) => setRegimeFilter(v as RegimeFilter)}
              options={[
                { value: 'ALL', label: 'All Regimes' },
                { value: '0', label: 'R0 - Bull trend, low volatility' },
                { value: '1', label: 'R1 - Bull trend, high volatility' },
                { value: '2', label: 'R2 - Bear trend, low volatility' },
                { value: '3', label: 'R3 - Bear trend, high volatility' },
                { value: '4', label: 'R4 - Neutral trend, low volatility' },
                { value: '5', label: 'R5 - Neutral trend, high volatility' },
                { value: 'UNKNOWN', label: 'Unknown Regime' },
              ]}
              className="w-full lg:w-[280px]"
            />
          </div>

          <div className="lg:ml-auto w-full lg:w-[280px]">
            <input
              type="text"
              value={symbolQuery}
              onChange={(e) => setSymbolQuery(e.target.value)}
              placeholder="Search stock symbol..."
              className="w-full h-9 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
            />
          </div>
        </div>
      </Card>

      <Card title="Today's Predictions" noPadding data-guide-id="predictions-table" action={
        <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">
          {isLoading ? 'Loading...' : `${filteredPredictions.length} results`}
        </span>
      }>
        {isLoading ? (
          <div className="px-6 py-5"><SkeletonTable rows={4} /></div>
        ) : (
          <Table<any>
            columns={predictionColumns}
            data={filteredPredictions}
            isLoading={isLoading}
            onRowClick={(p) => setSelectedStock({ id: p.stock_id, symbol: p.symbol, date: p.date, action: p.action })}
            emptyState={
              <EmptyState
                icon={<Crosshair size={32} />}
                title="No predictions available"
                description='Click "Run Predictions" to generate ensemble signals.'
              />
            }
          />
        )}
      </Card>

      <Card title="Recent Orders">
        {orders && orders.length > 0 ? (
          <div className="space-y-2">
            {orders.map((o: any) => (
              <div key={o.id} className="flex items-center justify-between py-3 px-4 rounded-[var(--radius-sm)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] transition-colors text-sm">
                <div className="flex items-center gap-2.5">
                  <Badge color={o.transaction_type === 'BUY' ? 'green' : 'red'}>{o.transaction_type}</Badge>
                  <span className="font-medium">Stock #{o.stock_id}</span>
                  <span className="text-xs text-[var(--text-dim)] tabular-nums">Qty: {o.quantity}</span>
                </div>
                <div className="flex items-center gap-2.5">
                  <Badge color={o.status === 'placed' ? 'blue' : o.status === 'complete' ? 'green' : 'gray'}>
                    {o.status}
                  </Badge>
                  <span className="text-xs text-[var(--text-dim)]">{o.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState icon={<Shield size={24} />} title="No recent orders" description="Place orders from predictions above." />
        )}
      </Card>

      {/* Order Confirmation Modal */}
      {orderConfirm && (
        <Modal
          open={!!orderConfirm}
          onClose={() => setOrderConfirm(null)}
          title={`Execute ${orderConfirm.action} order?`}
          footer={
            <div className="flex gap-3">
              <Button variant="secondary" onClick={() => setOrderConfirm(null)} className="flex-1">Cancel</Button>
              <Button onClick={confirmOrder} loading={placeOrder.isPending} className="flex-1">Confirm Order</Button>
            </div>
          }
        >
          <div className="space-y-4">
            <div className="flex justify-between items-center p-4 bg-[var(--bg-card)] rounded-xl border border-[var(--border)]">
              <div>
                <div className="text-sm text-[var(--text-dim)]">Symbol</div>
                <div className="text-xl font-bold">{orderConfirm.symbol}</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-[var(--text-dim)]">Type</div>
                <Badge color={orderConfirm.action === 'BUY' ? 'green' : 'red'}>{orderConfirm.action}</Badge>
              </div>
            </div>
            <div className="space-y-2 pt-2 border-t border-[var(--border)]">
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Confidence</span>
                <span className="font-mono">{(orderConfirm.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Agreement</span>
                {orderConfirm.agreement ? <Badge color="green">✓</Badge> : <Badge color="yellow">⚠</Badge>}
              </div>
            </div>
            <p className="text-[11px] text-[var(--text-muted)] italic">
              This will place a live market order for 10 shares of {orderConfirm.symbol} on NSE.
            </p>
          </div>
        </Modal>
      )}

      {/* Pattern Visualization Modal */}
      {selectedStock && (
        <PredictionPatternModal
          stock={selectedStock}
          onClose={() => setSelectedStock(null)}
          interval={interval}
        />
      )}
      {forwardLookTarget && (
        <Modal 
          open={!!forwardLookTarget} 
          onClose={() => setForwardLookTarget(null)}
          title={`Outcome Performance: ${forwardLookTarget?.symbol}`}
          size="xl"
        >
          <ForwardLookContent target={forwardLookTarget} interval={interval} />
        </Modal>
      )}

      {/* Stock Selector Modal */}
      <Modal
        open={showStockSelector}
        onClose={() => { setShowStockSelector(false); setStockSearchQuery(''); }}
        title="Select Stocks for Prediction"
        description="Choose specific stocks to predict. Leave all unchecked to predict the entire universe."
        size="lg"
        icon={<Filter size={20} className="text-[var(--primary)]" />}
        footer={
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-[var(--text-dim)]">
              {selectedStockIds.size > 0
                ? `${selectedStockIds.size} of ${universeStocks?.length ?? 0} stocks selected`
                : `All ${universeStocks?.length ?? 0} stocks will be predicted`}
            </span>
            <div className="flex gap-2">
              <Button variant="secondary" size="sm" onClick={() => setSelectedStockIds(new Set())}>
                Clear All
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setSelectedStockIds(new Set(universeStocks?.map((s: any) => s.id) ?? []))}
              >
                Select All
              </Button>
              <Button variant="primary" size="sm" onClick={() => { setShowStockSelector(false); setStockSearchQuery(''); }}>
                Done
              </Button>
            </div>
          </div>
        }
      >
        <div className="space-y-3">
          <input
            type="text"
            value={stockSearchQuery}
            onChange={(e) => setStockSearchQuery(e.target.value)}
            placeholder="Search symbol..."
            className="w-full h-9 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-input)] px-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/40"
          />
          <div className="max-h-[380px] overflow-y-auto space-y-1 pr-1">
            {(universeStocks ?? [])
              .filter((s: any) => !stockSearchQuery || String(s.symbol ?? '').toLowerCase().includes(stockSearchQuery.toLowerCase()))
              .map((s: any) => (
                <label
                  key={s.id}
                  className="flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)] hover:bg-[var(--bg-hover)] cursor-pointer transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selectedStockIds.has(s.id)}
                    onChange={(e) => {
                      setSelectedStockIds(prev => {
                        const next = new Set(prev);
                        if (e.target.checked) next.add(s.id); else next.delete(s.id);
                        return next;
                      });
                    }}
                    className="w-4 h-4 rounded border-[var(--border)] accent-[var(--primary)] cursor-pointer"
                  />
                  <span className="text-sm font-medium">{s.symbol}</span>
                  {s.sector && <span className="text-xs text-[var(--text-dim)] ml-auto">{s.sector}</span>}
                </label>
              ))}
          </div>
        </div>
      </Modal>
    </div>
  );
}

function ForwardLookContent({ target, interval }: { target: {stockId: number, symbol: string, date: string}, interval: string }) {
  const { data, isLoading } = useForwardLook({
    stock_id: target.stockId,
    after_date: target.date,
    interval
  });

  if (isLoading) return <div className="p-20 flex justify-center"><Loader2 className="animate-spin text-[var(--primary)] text-primary" /></div>;
  if (!data || data.length === 0) return <div className="p-20 text-center text-[var(--text-dim)]">No historical data found for the following 5 days.</div>;

  const chartData = data?.map((d: any) => ({
    time: d.date,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close
  })) ?? [];

  return (
    <div className="space-y-4">
      <div className="bg-[var(--bg-card)] p-4 rounded-[var(--radius-md)] border border-[var(--border)] h-[400px]">
        <LightweightCandleChart 
          ohlcv={chartData} 
          indicators={[]} 
        />
      </div>
      <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider text-center">
        Showing next 5 candles after {target.date}
      </div>
    </div>
  );
}

function PredictionPatternModal({ stock, onClose, interval }: { stock: any; onClose: () => void; interval: string }) {
  // Use a 90-day window before prediction for context, but continue until today
  const predictionDate = stock.date;
  const startDt = new Date(predictionDate);
  startDt.setDate(startDt.getDate() - 90);
  const startDate = startDt.toISOString().split('T')[0];
  const endDate = new Date().toISOString().split('T')[0];

  const { data: ohlcv, isLoading: loadingOhlcv } = useOhlcv(stock.id, interval, startDate, endDate);
  const { data: indicators, isLoading: loadingInd } = useIndicators(stock.id, interval, startDate, endDate);

  const chartIndicators: IndicatorSeries[] = [];
  if (indicators) {
    const sma50 = indicators.map((d: any) => ({ time: d.date, value: d.sma_50 })).filter((d: any) => d.value);
    const sma200 = indicators.map((d: any) => ({ time: d.date, value: d.sma_200 })).filter((d: any) => d.value);
    
    if (sma50.length) chartIndicators.push({ name: 'SMA 50', color: '#3b82f6', data: sma50 });
    if (sma200.length) chartIndicators.push({ name: 'SMA 200', color: '#eab308', data: sma200 });
  }

  const chartOhlcv = ohlcv?.map((d: any) => ({
    time: d.date,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close
  })) ?? [];

  return (
    <Modal
      open={true}
      onClose={onClose}
      title={`${stock.symbol} Pattern Analysis`}
      size="2xl"
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex gap-6">
            <div className="flex flex-col">
              <span className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Signal Date</span>
              <div className="text-lg font-mono font-bold leading-none mt-1">{stock.date}</div>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Signal</span>
              <div className="mt-1"><Badge color={stock.action === 'BUY' ? 'green' : stock.action === 'SELL' ? 'red' : 'gray'}>{stock.action}</Badge></div>
            </div>
          </div>
          <div className="text-right text-[10px] text-[var(--text-muted)] max-w-[200px] leading-tight">
            Visualizing the recent price action and technical indicators that formed this signal.
          </div>
        </div>

        <div className="relative bg-[#0b0b14] rounded-2xl border border-[var(--border)] overflow-hidden min-h-[500px] flex items-center justify-center">
          {loadingOhlcv || loadingInd ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-[var(--text-dim)] animate-pulse">Reconstructing pattern data...</span>
            </div>
          ) : chartOhlcv.length > 0 ? (
            <LightweightCandleChart
              ohlcv={chartOhlcv}
              indicators={chartIndicators}
              height={500}
              verticalLineDate={predictionDate}
            />
          ) : (
            <div className="text-[var(--text-dim)] flex flex-col items-center gap-2">
              <ShieldAlert size={32} strokeWidth={1} />
              <span className="text-xs tracking-tight">Pattern data unavailable for this instrument.</span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Card className="p-3 bg-[var(--bg-card)]/30 border-[var(--border)]/50">
            <span className="text-[10px] text-[var(--text-dim)] uppercase font-bold">Trend Alignment</span>
            <p className="text-[11px] mt-1 leading-relaxed text-[var(--text-muted)]">
              Evaluation of SMA crossovers and ADX strength to confirm regime validity.
            </p>
          </Card>
          <Card className="p-3 bg-[var(--bg-card)]/30 border-[var(--border)]/50">
            <span className="text-[10px] text-[var(--text-dim)] uppercase font-bold">Momentum Divergence</span>
            <p className="text-[11px] mt-1 leading-relaxed text-[var(--text-muted)]">
              Multi-timeframe RSI and MACD integration for breakout confirmation.
            </p>
          </Card>
          <Card className="p-3 bg-[var(--bg-card)]/30 border-[var(--border)]/50">
            <span className="text-[10px] text-[var(--text-dim)] uppercase font-bold">Volume Surge</span>
            <p className="text-[11px] mt-1 leading-relaxed text-[var(--text-muted)]">
              Relative volume spikes compared to the moving average for signal strength.
            </p>
          </Card>
        </div>
      </div>
    </Modal>
  );
}
