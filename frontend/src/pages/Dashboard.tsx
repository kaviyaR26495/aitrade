import { Card, StatCard, Badge, EmptyState, PageHeader, ListItem } from '../components/ui';
import { useSignals, useHoldings, useRlModels, useOrders } from '../hooks/useApi';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_LABEL_STYLE, CHART_AXIS_PROPS, CHART_GRID_PROPS } from '../components/ChartTheme';
import { TrendingUp, Brain, Target, Zap, ArrowRight } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const STATUS_COLORS: Record<string, 'green' | 'blue' | 'yellow' | 'red' | 'gray'> = {
  pending: 'yellow',
  active: 'blue',
  target_hit: 'green',
  sl_hit: 'red',
  expired: 'gray',
};

export default function Dashboard() {
  const { data: signals } = useSignals();
  const { data: holdings } = useHoldings();
  const { data: rlModels } = useRlModels();
  const { data: orders } = useOrders(10);

  const activeSignals = signals?.filter((s: any) => s.status === 'active') ?? [];
  const pendingSignals = signals?.filter((s: any) => s.status === 'pending') ?? [];
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
        <StatCard label="Active Signals" value={activeSignals.length} icon={<Zap size={18} />} color="var(--success)" data-guide-id="active-signals-card" />
        <StatCard label="Active Holdings" value={holdingsData.length} icon={<Target size={18} />} color="var(--info)" />
        <StatCard label="Trained Models" value={totalModels} icon={<Brain size={18} />} color="var(--primary)" />
        <StatCard label="Pending Signals" value={pendingSignals.length} icon={<TrendingUp size={18} />} color="var(--warning)" data-guide-id="pending-signals-card" />
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
        <Card title="Today's Signals" data-guide-id="signals-card" action={
          signals && signals.length > 0 ? (
            <NavLink to="/trading" className="text-[11px] text-[var(--primary)] hover:text-[var(--primary-hover)] font-medium">View All →</NavLink>
          ) : null
        }>
          {signals && signals.length > 0 ? (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {signals.slice(0, 10).map((s: any) => (
                <ListItem
                  key={s.id}
                  left={
                    <>
                      <Badge color={STATUS_COLORS[s.status] ?? 'gray'}>{s.status?.replace('_', ' ')}</Badge>
                      <span className="font-medium">{s.symbol}</span>
                    </>
                  }
                  right={
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-[var(--text-muted)] tabular-nums">₹{Number(s.entry_price).toFixed(0)} → ₹{Number(s.target_price).toFixed(0)}</span>
                      <span className="text-[var(--text-muted)] font-medium tabular-nums">PoP {(Number(s.pop_score) * 100).toFixed(0)}%</span>
                      <span className="text-[var(--text-muted)] tabular-nums">R:R {Number(s.initial_rr_ratio).toFixed(1)}</span>
                    </div>
                  }
                />
              ))}
            </div>
          ) : (
            <EmptyState icon={<Zap size={28} />} title="No signals yet" description="Generate signals from the Trading page to see them here." />
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
