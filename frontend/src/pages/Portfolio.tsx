import { Card, Button, Badge, StatCard, EmptyState, PageHeader, Table, SkeletonTable, type TableColumn } from '../components/ui';
import { useHoldings, usePositions, usePortfolioSnapshot, useReconcilePortfolio } from '../hooks/useApi';
import { useAppStore } from '../store/appStore';
import { Briefcase, RefreshCw, RotateCcw, DollarSign } from 'lucide-react';

export default function Portfolio() {
  const { data: holdingsData, isLoading: loadingHoldings, refetch: refetchHoldings } = useHoldings();
  const { data: positionsData, isLoading: loadingPositions, refetch: refetchPositions } = usePositions();
  const { data: snapshot, isLoading: loadingSnapshot } = usePortfolioSnapshot();
  const reconcileMutation = useReconcilePortfolio();
  const { addNotification } = useAppStore();

  const holdings = holdingsData?.holdings ?? [];
  const positions = positionsData?.positions ?? [];

  const totalHoldingValue = holdings.reduce((sum: number, h: any) =>
    sum + (h.last_price ?? 0) * (h.quantity ?? 0), 0
  );

  const totalPnl = holdings.reduce((sum: number, h: any) =>
    sum + (h.pnl ?? 0), 0
  );

  const holdingColumns: TableColumn<any>[] = [
    { key: 'tradingsymbol', label: 'Symbol', render: (h) => <span className="font-medium">{h.tradingsymbol}</span> },
    { key: 'quantity', label: 'Qty', align: 'right', mono: true },
    { key: 'average_price', label: 'Avg Price', align: 'right', mono: true, render: (h) => `₹${h.average_price?.toFixed(2)}` },
    { key: 'last_price', label: 'LTP', align: 'right', mono: true, render: (h) => `₹${h.last_price?.toFixed(2)}` },
    { key: 'pnl', label: 'P&L', align: 'right', render: (h) => (
      <span className={`font-medium tabular-nums ${(h.pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
        ₹{h.pnl?.toFixed(2)}
      </span>
    )},
    { key: 'pnl_pct', label: 'P&L %', align: 'right', render: (h) => {
      const pct = h.average_price ? ((h.last_price - h.average_price) / h.average_price * 100) : 0;
      return <span className={`tabular-nums ${pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{pct.toFixed(1)}%</span>;
    }},
  ];

  const positionColumns: TableColumn<any>[] = [
    { key: 'tradingsymbol', label: 'Symbol', render: (p) => <span className="font-medium">{p.tradingsymbol}</span> },
    { key: 'product', label: 'Type', align: 'center', render: (p) => <Badge color={p.product === 'CNC' ? 'blue' : 'gray'}>{p.product}</Badge> },
    { key: 'quantity', label: 'Qty', align: 'right', mono: true },
    { key: 'buy_price', label: 'Buy Price', align: 'right', mono: true, render: (p) => `₹${p.buy_price?.toFixed(2)}` },
    { key: 'sell_price', label: 'Sell Price', align: 'right', mono: true, render: (p) => `₹${p.sell_price?.toFixed(2)}` },
    { key: 'pnl', label: 'P&L', align: 'right', render: (p) => (
      <span className={`font-medium tabular-nums ${(p.pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
        ₹{p.pnl?.toFixed(2)}
      </span>
    )},
  ];

  const handleRefresh = async () => {
    try {
      await Promise.all([refetchHoldings(), refetchPositions()]);
      addNotification({ type: 'success', message: 'Portfolio refreshed' });
    } catch {
      addNotification({ type: 'error', message: 'Failed to refresh portfolio' });
    }
  };

  const handleReconcile = () => {
    reconcileMutation.mutate(undefined, {
      onSuccess: (res: any) => {
        addNotification({ type: 'success', message: `Reconciled — ₹${res.total_equity?.toLocaleString('en-IN', { maximumFractionDigits: 0 })} total equity` });
      },
      onError: (e: any) => {
        addNotification({ type: 'error', message: e?.response?.data?.detail ?? 'Reconciliation failed' });
      },
    });
  };

  return (
    <div className="space-y-8">
      <PageHeader title="Portfolio" description="Holdings, positions, and P&L overview">
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={handleReconcile} loading={reconcileMutation.isPending}>
            <RotateCcw size={14} /> Reconcile
          </Button>
          <Button variant="secondary" size="sm" onClick={handleRefresh} loading={loadingHoldings || loadingPositions}>
            <RefreshCw size={14} className={(loadingHoldings || loadingPositions) ? 'animate-spin' : ''} /> Refresh
          </Button>
        </div>
      </PageHeader>

      {/* Reconciliation snapshot strip */}
      {!loadingSnapshot && snapshot && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 px-1">
          <StatCard label="Cash Available" value={`₹${snapshot.cash_available?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`} icon={<DollarSign size={16} />} color="var(--success)" />
          <StatCard label="Holdings Value" value={`₹${snapshot.holdings_value?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`} icon={<Briefcase size={16} />} color="var(--primary)" />
          <StatCard label="Total Equity" value={`₹${snapshot.total_equity?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`} color="var(--text)" />
          <StatCard label="Unrealised P&L" value={`₹${snapshot.unrealized_pnl?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`} color={snapshot.unrealized_pnl >= 0 ? 'var(--success)' : 'var(--danger)'} />
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
        <StatCard label="Holdings" value={holdings.length} icon={<Briefcase size={18} />} color="var(--primary)" />
        <StatCard label="Positions" value={positions.length} color="var(--info)" />
        <StatCard label="Total Value" value={`₹${totalHoldingValue.toLocaleString('en-IN')}`} color="var(--text)" />
        <StatCard
          label="Total P&L"
          value={`₹${totalPnl.toLocaleString('en-IN')}`}
          color={totalPnl >= 0 ? 'var(--success)' : 'var(--danger)'}
        />
      </div>

      <Card title="Holdings" noPadding data-guide-id="holdings-table">
        {loadingHoldings ? (
          <div className="px-6 py-5"><SkeletonTable rows={3} /></div>
        ) : (
          <Table<any>
            columns={holdingColumns}
            data={holdings}
            emptyState={<EmptyState icon={<Briefcase size={28} />} title="No holdings" description="Start trading to build your portfolio." />}
          />
        )}
      </Card>

      <Card title="Positions" noPadding>
        {loadingPositions ? (
          <div className="px-6 py-5"><SkeletonTable rows={3} /></div>
        ) : (
          <Table<any>
            columns={positionColumns}
            data={positions}
            emptyState={<EmptyState icon={<Briefcase size={24} />} title="No open positions" description="Positions will appear when you have active trades." />}
          />
        )}
      </Card>
    </div>
  );
}
