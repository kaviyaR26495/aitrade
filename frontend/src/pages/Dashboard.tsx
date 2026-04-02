import { Card, StatCard, Badge, EmptyState, PageHeader, ListItem } from '../components/ui';
import { usePredictions, useHoldings, useRlModels, useOrders } from '../hooks/useApi';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_LABEL_STYLE, CHART_AXIS_PROPS, CHART_GRID_PROPS } from '../components/ChartTheme';
import { TrendingUp, Brain, Target, AlertTriangle, ArrowRight } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const REGIME_LABELS: Record<number, { label: string; color: string }> = {
  0: { label: 'Bull+LV', color: 'green' },
  1: { label: 'Bull+HV', color: 'yellow' },
  2: { label: 'Bear+LV', color: 'red' },
  3: { label: 'Bear+HV', color: 'red' },
  4: { label: 'Neut+LV', color: 'gray' },
  5: { label: 'Neut+HV', color: 'yellow' },
};

export default function Dashboard() {
  const { data: predictions } = usePredictions();
  const { data: holdings } = useHoldings();
  const { data: rlModels } = useRlModels();
  const { data: orders } = useOrders(10);

  const buySignals = predictions?.filter((p: any) => p.action === 'BUY') ?? [];
  const sellSignals = predictions?.filter((p: any) => p.action === 'SELL') ?? [];
  const holdingsData = holdings?.holdings ?? [];
  const totalModels = rlModels?.length ?? 0;

  const equityCurve = Array.from({ length: 30 }, (_, i) => ({
    day: i + 1,
    value: 100000 + Math.random() * 10000 * Math.sin(i / 5) + i * 200,
  }));

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description={new Date().toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
      >
        <NavLink
          to="/trading"
          className="inline-flex items-center gap-2 h-10 px-5 rounded-[var(--radius-sm)] bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white text-sm font-semibold transition-all duration-200 shadow-sm hover:shadow-[0_0_20px_var(--primary-glow)] active:scale-[0.97]"
        >
          Go to Trading <ArrowRight size={14} />
        </NavLink>
      </PageHeader>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard label="Buy Signals" value={buySignals.length} icon={<TrendingUp size={18} />} color="var(--success)" data-guide-id="buy-signals-card" />
        <StatCard label="Active Holdings" value={holdingsData.length} icon={<Target size={18} />} color="var(--info)" />
        <StatCard label="Trained Models" value={totalModels} icon={<Brain size={18} />} color="var(--primary)" />
        <StatCard label="Sell Signals" value={sellSignals.length} icon={<AlertTriangle size={18} />} color="var(--danger)" data-guide-id="sell-signals-card" />
      </div>

      <Card title="Portfolio Equity Curve" data-guide-id="equity-curve" action={
        <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">Last 30 Days</span>
      }>
        <div className="-mx-2 mt-2">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={equityCurve}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="var(--primary)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" {...CHART_AXIS_PROPS} />
              <YAxis {...CHART_AXIS_PROPS} tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}K`} />
              <Tooltip
                contentStyle={CHART_TOOLTIP_STYLE}
                labelStyle={CHART_LABEL_STYLE}
                // @ts-ignore — Recharts formatter type mismatch
                formatter={(v: number) => [`₹${v.toFixed(0)}`, 'Value']}
              />
              <Area type="monotone" dataKey="value" stroke="var(--primary)" strokeWidth={2} fill="url(#equityGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card title="Today's Predictions" data-guide-id="predictions-card" action={
          predictions && predictions.length > 0 ? (
            <NavLink to="/trading" className="text-[11px] text-[var(--primary)] hover:text-[var(--primary-hover)] font-medium">View All →</NavLink>
          ) : null
        }>
          {predictions && predictions.length > 0 ? (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {predictions.slice(0, 10).map((p: any) => (
                <ListItem
                  key={p.id}
                  left={
                    <>
                      <Badge color={p.action === 'BUY' ? 'green' : p.action === 'SELL' ? 'red' : 'gray'}>{p.action}</Badge>
                      <span className="font-medium">{p.symbol}</span>
                    </>
                  }
                  right={
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-[var(--text-muted)] font-medium tabular-nums">{(p.confidence * 100).toFixed(0)}%</span>
                      {p.agreement && <Badge color="green">✓</Badge>}
                      {p.regime_id != null && (
                        <Badge color={REGIME_LABELS[p.regime_id]?.color as any ?? 'gray'}>{REGIME_LABELS[p.regime_id]?.label ?? `R${p.regime_id}`}</Badge>
                      )}
                    </div>
                  }
                />
              ))}
            </div>
          ) : (
            <EmptyState icon={<TrendingUp size={28} />} title="No predictions yet" description="Run predictions from the Trading page to see signals here." />
          )}
        </Card>

        <Card title="Recent Orders" data-guide-id="recent-orders-card" action={
          orders && orders.length > 0 ? (
            <NavLink to="/portfolio" className="text-[11px] text-[var(--primary)] hover:text-[var(--primary-hover)] font-medium">Portfolio →</NavLink>
          ) : null
        }>
          {orders && orders.length > 0 ? (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {orders.slice(0, 10).map((o: any) => (
                <ListItem
                  key={o.id}
                  left={
                    <>
                      <Badge color={o.transaction_type === 'BUY' ? 'green' : 'red'}>{o.transaction_type}</Badge>
                      <span className="font-medium">Stock #{o.stock_id}</span>
                    </>
                  }
                  right={
                    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                      <span className="tabular-nums">Qty: {o.quantity}</span>
                      <Badge color={o.status === 'placed' ? 'blue' : o.status === 'complete' ? 'green' : 'gray'}>{o.status}</Badge>
                    </div>
                  }
                />
              ))}
            </div>
          ) : (
            <EmptyState icon={<Target size={28} />} title="No orders yet" description="Place orders from the Live Trading page." />
          )}
        </Card>
      </div>
    </div>
  );
}
