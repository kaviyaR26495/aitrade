import { useState, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Card, Button, Select, SearchableSelect, Badge, StatCard, EmptyState, PageHeader } from '../components/ui';
import { useUniverseStocks, useRegimeSummary, useRegime } from '../hooks/useApi';
import { classifyRegime } from '../services/api';
import { useAppStore } from '../store/appStore';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_AXIS_PROPS } from '../components/ChartTheme';
import LightweightRegimeChart from '../components/LightweightRegimeChart';
import { Layers } from 'lucide-react';

const REGIME_COLORS: Record<number, string> = {
  0: '#22c55e',
  1: '#a3e635',
  2: '#ef4444',
  3: '#991b1b',
  4: '#6b7280',
  5: '#eab308',
};

const REGIME_LABELS: Record<number, string> = {
  0: 'Bull+LowVol',
  1: 'Bull+HighVol',
  2: 'Bear+LowVol',
  3: 'Bear+HighVol',
  4: 'Neutral+LowVol',
  5: 'Neutral+HighVol',
};

export default function RegimeAnalysis() {
  const queryClient = useQueryClient();
  const { data: stocks } = useUniverseStocks();
  const { addNotification } = useAppStore();
  
  const [stockId, setStockId] = useState(() => localStorage.getItem('regime_stockId') || '');
  const [interval, setInterval] = useState(() => localStorage.getItem('regime_interval') || 'day');
  const [classifying, setClassifying] = useState(false);

  useEffect(() => {
    localStorage.setItem('regime_stockId', stockId);
  }, [stockId]);

  useEffect(() => {
    localStorage.setItem('regime_interval', interval);
  }, [interval]);

  const sid = stockId ? parseInt(stockId) : 0;
  const { data: summary } = useRegimeSummary(sid, interval);
  const { data: regimeData } = useRegime(sid, interval);

  const stockOptions = [
    { value: '', label: 'Select stock...' },
    ...(stocks?.map((s: any) => ({ value: String(s.id), label: s.symbol })) ?? []),
  ];

  const handleClassify = async () => {
    if (!sid) return;
    setClassifying(true);
    try {
      await classifyRegime(sid, interval);
      queryClient.invalidateQueries({ queryKey: ['regime'] });
      queryClient.invalidateQueries({ queryKey: ['regime-summary'] });
      addNotification({ type: 'success', message: 'Regime classification complete' });
    } catch {
      addNotification({ type: 'error', message: 'Classification failed' });
    } finally {
      setClassifying(false);
    }
  };

  const pieData = summary?.regime_breakdown
    ? Object.entries(summary.regime_breakdown).map(([rid, data]: [string, any]) => ({
        name: REGIME_LABELS[parseInt(rid)] ?? `Regime ${rid}`,
        value: data.count ?? 0,
        fill: REGIME_COLORS[parseInt(rid)] ?? '#6b7280',
      }))
    : [];

  const QUALITY_COLORS: Record<string, string> = {
    high: 'var(--success)',
    medium: 'var(--warning)',
    low: 'var(--danger)',
  };

  const qualityData = summary?.quality_tiers
    ? Object.entries(summary.quality_tiers).map(([tier, count]) => ({
        tier,
        count: count as number,
        fill: QUALITY_COLORS[tier] || 'var(--primary)',
      }))
    : [];

  const timelineData: any[] = regimeData?.map((r: any) => ({
    date: r.date,
    close: r.close ?? 0,
    regime_id: r.regime_id,
    quality: r.quality_score,
  })) ?? [];

  return (
    <div className="space-y-8">
      <PageHeader title="Regime Analysis" description="Market regime classification and quality metrics">
        <SearchableSelect value={stockId} onChange={setStockId} options={stockOptions} className="w-48" placeholder="Search stocks..." />
        <Select
          value={interval}
          onChange={setInterval}
          options={[{ value: 'day', label: 'Daily' }, { value: 'week', label: 'Weekly' }]}
          className="w-28"
        />
        <Button size="sm" onClick={handleClassify} loading={classifying} disabled={!sid} data-guide-id="regime-classify-btn">
          <Layers size={14} /> Classify
        </Button>
      </PageHeader>

      {!sid ? (
        <Card>
          <EmptyState icon={<Layers size={32} />} title="Select a stock" description="Choose a stock from the dropdown to view regime analysis." />
        </Card>
      ) : (
        <>
          {summary && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
              <StatCard label="Total Candles" value={summary.total_candles ?? 0} color="var(--primary)" />
              <StatCard label="Quality ≥ 0.8" value={summary.quality_tiers?.high ?? 0} color="var(--success)" />
              <StatCard label="Transitions" value={summary.total_transitions ?? 0} color="var(--warning)" />
              <StatCard label="Avg Quality" value={summary.avg_quality?.toFixed(2) ?? '—'} color="var(--info)" />
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Regime Distribution" data-guide-id="regime-pie-chart">
              {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      innerRadius={55}
                      dataKey="value"
                      label={({ name, percent }) => `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`}
                      labelLine={false}
                      fontSize={10}
                      strokeWidth={2}
                      stroke="var(--bg)"
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState icon={<Layers size={24} />} title="No regime data" description="Run classification first to see distribution." />
              )}
            </Card>

            <Card title="Data Quality Distribution">
              {qualityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={qualityData}>
                    <XAxis dataKey="tier" {...CHART_AXIS_PROPS} />
                    <YAxis {...CHART_AXIS_PROPS} />
                    <Tooltip contentStyle={CHART_TOOLTIP_STYLE} cursor={{fill: 'var(--bg-hover)'}} />
                    <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                      {qualityData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState icon={<Layers size={24} />} title="No quality data" description="Quality scores appear after regime classification." />
              )}
            </Card>
          </div>

          <Card title="Regime Timeline" data-guide-id="regime-timeline">
            {timelineData.length > 0 ? (
              <div className="-mx-2 mt-2">
                <LightweightRegimeChart
                  data={timelineData.map((d: any) => ({
                    time: d.date,
                    value: d.close ?? 0,
                    regime_id: d.regime_id,
                  }))}
                  height={340}
                />
              </div>
            ) : (
              <EmptyState icon={<Layers size={24} />} title="No timeline data" description="Run classification to see regime timeline." />
            )}
          </Card>

          <Card title="Regime Legend">
            <div className="flex flex-wrap gap-3">
              {Object.entries(REGIME_LABELS).map(([id, label]) => (
                <Badge key={id} color="gray" variant="outline" size="md" dot>
                  <span className="w-2 h-2 rounded-full inline-block mr-1" style={{ backgroundColor: REGIME_COLORS[parseInt(id)] }} />
                  {label}
                </Badge>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
