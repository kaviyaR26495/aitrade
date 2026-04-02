import { useState } from 'react';
import { Card, Button, Badge, StatCard, Select, EmptyState, PageHeader, Table, Modal, Checkbox, SkeletonTable, type TableColumn } from '../components/ui';
import { usePredictions, useOrders, useRunPredictions, usePlaceOrder } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Crosshair, Play, Shield, ShieldAlert } from 'lucide-react';

export default function LiveTrading() {
  const [interval, setInterval] = useState('day');
  const [minConfidence, setMinConfidence] = useState(0.65);
  const [agreementOnly, setAgreementOnly] = useState(true);

  const { data: predictions, isLoading } = usePredictions({
    interval,
    min_confidence: minConfidence,
    agreement_only: agreementOnly,
  });
  const { data: orders } = useOrders(20);
  const runPredictions = useRunPredictions();
  const placeOrder = usePlaceOrder();
  const { addNotification } = useAppStore();

  const [orderConfirm, setOrderConfirm] = useState<any>(null);

  const buyPreds = predictions?.filter((p: any) => p.action === 'BUY') ?? [];
  const sellPreds = predictions?.filter((p: any) => p.action === 'SELL') ?? [];

  const predictionColumns: TableColumn<any>[] = [
    { key: 'symbol', label: 'Symbol', render: (p) => <span className="font-medium">{p.symbol}</span> },
    { key: 'action', label: 'Action', render: (p) => (
      <Badge color={p.action === 'BUY' ? 'green' : p.action === 'SELL' ? 'red' : 'gray'}>{p.action}</Badge>
    )},
    { key: 'confidence', label: 'Confidence', align: 'center', render: (p) => (
      <span className={`font-medium tabular-nums ${p.confidence >= 0.8 ? 'text-emerald-400' : p.confidence >= 0.65 ? 'text-amber-400' : 'text-[var(--text-muted)]'}`}>
        {(p.confidence * 100).toFixed(0)}%
      </span>
    )},
    { key: 'knn', label: 'KNN', align: 'center', render: (p) => (
      <span className="text-xs">
        <Badge color={p.knn_action === 'BUY' ? 'green' : p.knn_action === 'SELL' ? 'red' : 'gray'}>{p.knn_action}</Badge>
        <span className="ml-1 text-[var(--text-dim)] tabular-nums">{(p.knn_confidence * 100).toFixed(0)}%</span>
      </span>
    )},
    { key: 'lstm', label: 'LSTM', align: 'center', render: (p) => (
      <span className="text-xs">
        <Badge color={p.lstm_action === 'BUY' ? 'green' : p.lstm_action === 'SELL' ? 'red' : 'gray'}>{p.lstm_action}</Badge>
        <span className="ml-1 text-[var(--text-dim)] tabular-nums">{(p.lstm_confidence * 100).toFixed(0)}%</span>
      </span>
    )},
    { key: 'agreement', label: 'Agreement', align: 'center', render: (p) => (
      p.agreement ? <Badge color="green">✓</Badge> : <Badge color="yellow">⚠</Badge>
    )},
    { key: 'regime', label: 'Regime', align: 'center', mono: true, render: (p) => (
      <span className="text-xs text-[var(--text-dim)]">R{p.regime_id ?? '—'}</span>
    )},
    { key: 'actions', label: 'Action', align: 'right', render: (p) => (
      p.action !== 'HOLD' ? (
        <Button size="sm" variant="secondary" onClick={() => handlePlaceOrder(p)}>Order</Button>
      ) : null
    )},
  ];

  const handleRunPredictions = () => {
    runPredictions.mutate(
      { interval, agreement_required: agreementOnly },
      {
        onSuccess: (res: any) => {
          addNotification({ type: 'success', message: `${res.data.predictions_made} predictions generated` });
        },
        onError: () => addNotification({ type: 'error', message: 'Prediction run failed' }),
      }
    );
  };

  const handlePlaceOrder = (pred: any) => {
    setOrderConfirm(pred);
  };

  const confirmOrder = () => {
    if (!orderConfirm) return;
    placeOrder.mutate(
      {
        stock_id: orderConfirm.stock_id,
        ensemble_prediction_id: orderConfirm.id,
        transaction_type: orderConfirm.action,
        quantity: 1,
        variety: 'regular',
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
          <Select value={interval} onChange={setInterval} options={[
            { value: 'day', label: 'Daily' },
            { value: 'week', label: 'Weekly' },
          ]} className="w-28" />
          <div className="py-1.5 px-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)]" data-guide-id="agreement-filter">
            <Checkbox checked={agreementOnly} onChange={setAgreementOnly} label="Agreement Only" />
          </div>
          <Button onClick={handleRunPredictions} loading={runPredictions.isPending} data-guide-id="run-predictions-btn">
            <Play size={14} /> Run Predictions
          </Button>
        </div>
      </PageHeader>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard label="BUY Signals" value={buyPreds.length} color="var(--success)" />
        <StatCard label="SELL Signals" value={sellPreds.length} color="var(--danger)" />
        <StatCard label="Total Predictions" value={predictions?.length ?? 0} color="var(--primary)" />
        <StatCard label="Min Confidence" value={`${(minConfidence * 100).toFixed(0)}%`} color="var(--info)" />
      </div>

      <Card title="Today's Predictions" noPadding data-guide-id="predictions-table" action={
        <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">
          {isLoading ? 'Loading...' : `${predictions?.length ?? 0} results`}
        </span>
      }>
        {isLoading ? (
          <div className="px-6 py-5"><SkeletonTable rows={4} /></div>
        ) : (
          <Table<any>
            columns={predictionColumns}
            data={predictions ?? []}
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

      <Modal
        open={!!orderConfirm}
        onClose={() => setOrderConfirm(null)}
        title="Confirm Order"
        icon={<ShieldAlert size={20} className="text-[var(--primary)]" />}
        footer={
          <>
            <Button variant="secondary" onClick={() => setOrderConfirm(null)} className="flex-1">Cancel</Button>
            <Button onClick={confirmOrder} loading={placeOrder.isPending} className="flex-1">
              Confirm {orderConfirm?.action}
            </Button>
          </>
        }
      >
        {orderConfirm && (
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">Symbol</span>
              <span className="font-medium">{orderConfirm.symbol}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">Action</span>
              <Badge color={orderConfirm.action === 'BUY' ? 'green' : 'red'}>{orderConfirm.action}</Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">Ensemble Confidence</span>
              <span className="tabular-nums">{(orderConfirm.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">KNN</span>
              <span className="tabular-nums">{orderConfirm.knn_action} ({(orderConfirm.knn_confidence * 100).toFixed(0)}%)</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">LSTM</span>
              <span className="tabular-nums">{orderConfirm.lstm_action} ({(orderConfirm.lstm_confidence * 100).toFixed(0)}%)</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--text-muted)]">Agreement</span>
              {orderConfirm.agreement ? <Badge color="green">✓</Badge> : <Badge color="yellow">⚠</Badge>}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
