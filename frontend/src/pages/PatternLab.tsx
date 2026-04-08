import { useState } from 'react';
import { Card, Badge, EmptyState, PageHeader, Modal, Tabs } from '../components/ui';
import { Search, Cpu, Layers, BarChart2, X } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import api from '../services/api';
import LightweightCandleChart, { type OhlcvPoint, type TradeMarker } from '../components/LightweightCandleChart';

/* ── helpers ──────────────────────────────────────────────────────── */
const getBadgeColor = (label: number) =>
  label === 1 ? 'green' : label === -1 ? 'red' : 'gray';

const getLabelTxt = (label: number) =>
  label === 1 ? 'BUY' : label === -1 ? 'SELL' : 'HOLD';

/* ── Pattern Detail Modal ─────────────────────────────────────────── */
interface PatternDetailProps {
  pattern: any;
  onClose: () => void;
}

function PatternDetailModal({ pattern, onClose }: PatternDetailProps) {
  const [tab, setTab] = useState('chart');

  const { data: ohlcvData, isLoading: ohlcvLoading } = useQuery({
    queryKey: ['pattern_ohlcv', pattern.id],
    queryFn: () => api.get(`/models/patterns/${pattern.id}/ohlcv`).then(r => r.data),
    enabled: !!pattern,
  });

  const candles: OhlcvPoint[] = ohlcvData?.candles ?? [];
  const patternDate: string = ohlcvData?.pattern_date ?? '';

  const tradeMarkers: TradeMarker[] =
    patternDate && pattern.label !== 0
      ? [{
          time: patternDate,
          side: pattern.label === 1 ? 'BUY' : 'SELL',
          price: candles.find(c => c.time === patternDate)?.close ?? 0,
          reason: `${getLabelTxt(pattern.label)} · conf ${(pattern.confidence * 100).toFixed(1)}%`,
        }]
      : [];

  const intervalStr: string =
    typeof pattern.interval === 'object' && pattern.interval !== null
      ? String((pattern.interval as any).value ?? pattern.interval)
      : String(pattern.interval ?? 'day');

  const modalTabs = [
    { id: 'chart', label: 'Candlestick', icon: <BarChart2 size={13} /> },
    { id: 'knn',   label: 'KNN Inspector', icon: <Cpu size={13} /> },
    { id: 'lstm',  label: 'LSTM Attention', icon: <Layers size={13} /> },
  ];

  return (
    <Modal
      open
      onClose={onClose}
      title={`Pattern #${pattern.id}`}
      description={`Stock #${pattern.stock_id} · ${intervalStr} · ${pattern.date}`}
      icon={<BarChart2 size={18} className="text-[var(--color-primary)]" />}
      size="xl"
    >
      {/* ── Meta info grid ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        {[
          { label: 'Action',      value: <Badge color={getBadgeColor(pattern.label) as any}>{getLabelTxt(pattern.label)}</Badge> },
          { label: 'P&L',         value: <span className={pattern.pnl_percent > 0 ? 'text-[var(--color-green)]' : 'text-[var(--color-red)]'}>{pattern.pnl_percent?.toFixed(2)}%</span> },
          { label: 'Confidence',  value: `${(pattern.confidence * 100).toFixed(1)}%` },
          { label: 'Regime',      value: `#${pattern.regime_id ?? '—'}` },
          { label: 'RL Model',    value: `#${pattern.rl_model_id}` },
          { label: 'Stock ID',    value: `#${pattern.stock_id}` },
          { label: 'Date',        value: String(pattern.date) },
          { label: 'ATR %',       value: pattern.atr_at_capture != null ? `${(pattern.atr_at_capture * 100).toFixed(2)}%` : '—' },
        ].map(({ label, value }) => (
          <div key={label} className="p-3 rounded-[var(--radius-sm)] bg-[var(--bg-input)] border border-[var(--border-color)]">
            <div className="text-[10px] uppercase tracking-wider text-[var(--text-dim)] mb-1 font-medium">{label}</div>
            <div className="text-sm font-semibold">{value}</div>
          </div>
        ))}
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────── */}
      <Tabs tabs={modalTabs} activeTab={tab} onTabChange={setTab} variant="pills" className="mb-4" />

      {/* ── Chart tab ──────────────────────────────────────────────── */}
      {tab === 'chart' && (
        ohlcvLoading ? (
          <div className="h-64 flex items-center justify-center text-[var(--text-muted)] animate-pulse text-sm">
            Loading candles…
          </div>
        ) : candles.length === 0 ? (
          <div className="h-48">
            <EmptyState
              icon={<BarChart2 size={28} />}
              title="No OHLCV data"
              description="No candle data found for this stock/interval in the DB."
            />
          </div>
        ) : (
          <div className="rounded-[var(--radius-sm)] overflow-hidden border border-[var(--border-color)]">
            <LightweightCandleChart
              ohlcv={candles}
              trades={tradeMarkers}
              height={320}
            />
          </div>
        )
      )}

      {/* ── KNN tab ────────────────────────────────────────────────── */}
      {tab === 'knn' && (
        <div className="space-y-3">
          <div className="flex justify-between items-center pb-2 border-b border-[var(--border-color)]">
            <span className="font-semibold text-sm">Target Pattern ID: #{pattern.id}</span>
            <Badge color={getBadgeColor(pattern.label) as any}>{getLabelTxt(pattern.label)}</Badge>
          </div>
          <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">Nearest Neighbors</h4>
          <ul className="space-y-2">
            {[1, 2, 3].map((i) => (
              <li key={i} className="flex justify-between items-center p-3 rounded-[var(--radius-sm)] bg-[var(--bg-panel)] border border-[var(--border-color)]">
                <div>
                  <p className="font-medium text-sm">Neighbor #{pattern.id + i * 10}</p>
                  <p className="text-xs text-[var(--text-muted)]">Similarity: {(0.95 - i * 0.05).toFixed(2)}</p>
                </div>
                <Badge color={getBadgeColor(pattern.label) as any}>{getLabelTxt(pattern.label)}</Badge>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── LSTM tab ───────────────────────────────────────────────── */}
      {tab === 'lstm' && (
        <div className="space-y-3">
          <div className="flex justify-between items-center pb-2 border-b border-[var(--border-color)]">
            <span className="font-semibold text-sm">Sequence Analysis for #{pattern.id}</span>
          </div>
          <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">Attention Weights</h4>
          <div className="space-y-3">
            {['Price Momentum', 'Volume Profile', 'Volatility', 'RSI', 'MACD'].map((feature, i) => (
              <div key={feature}>
                <div className="flex justify-between text-xs mb-1 text-[var(--text-muted)]">
                  <span>{feature}</span>
                  <span>{(0.85 - i * 0.13).toFixed(2)}</span>
                </div>
                <div className="h-2 w-full bg-[var(--bg-panel)] rounded overflow-hidden">
                  <div className="h-full bg-[var(--color-primary)]" style={{ width: `${(0.85 - i * 0.13) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Close ──────────────────────────────────────────────────── */}
      <div className="flex justify-end mt-5 pt-4 border-t border-[var(--border-color)]">
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 px-4 py-2 rounded-[var(--radius-sm)] bg-[var(--bg-input)] border border-[var(--border-color)] text-sm font-medium hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
        >
          <X size={14} /> Close
        </button>
      </div>
    </Modal>
  );
}

/* ── Main page ────────────────────────────────────────────────────── */
export default function PatternLab() {
  const [selectedPattern, setSelectedPattern] = useState<any>(null);
  const [labelFilter, setLabelFilter] = useState<number | null>(null);

  const { data: patterns = [], isLoading } = useQuery<any[]>({
    queryKey: ['golden_patterns'],
    queryFn: () => api.get('/models/patterns').then(r => r.data),
  });

  const buyCount  = patterns.filter(p => p.label === 1).length;
  const sellCount = patterns.filter(p => p.label === -1).length;
  const holdCount = patterns.filter(p => p.label === 0).length;

  const displayed = labelFilter === null
    ? patterns
    : patterns.filter(p => p.label === labelFilter);

  return (
    <div className="space-y-8">
      <PageHeader title="Pattern Lab" description="Explore golden patterns from RL replay memory" />

      {/* ── Statistics (click to filter) ───────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'BUY',  color: 'green' as const, count: buyCount,  val: 1 },
          { label: 'SELL', color: 'red'   as const, count: sellCount, val: -1 },
          { label: 'HOLD', color: 'gray'  as const, count: holdCount, val: 0 },
        ].map(({ label, color, count, val }) => (
          <button
            key={label}
            onClick={() => setLabelFilter(prev => prev === val ? null : val)}
            className={`p-4 rounded-[var(--radius)] text-center border transition-colors cursor-pointer ${
              labelFilter === val
                ? 'bg-[var(--bg-active)] border-[var(--color-primary)]'
                : 'bg-[var(--bg-input)] border-[var(--border-color)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            <Badge color={color}>{label}</Badge>
            <div className="text-2xl font-bold mt-1.5 tabular-nums">{count || '—'}</div>
            <div className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </button>
        ))}
      </div>

      {/* ── Pattern table ───────────────────────────────────────────── */}
      <Card
        title="Golden Pattern Explorer"
        action={
          labelFilter !== null ? (
            <button
              onClick={() => setLabelFilter(null)}
              className="text-xs text-[var(--text-muted)] flex items-center gap-1 hover:text-[var(--text)] transition-colors cursor-pointer"
            >
              <X size={12} /> Clear filter
            </button>
          ) : undefined
        }
      >
        {isLoading ? (
          <div className="p-8 text-center text-[var(--text-muted)] animate-pulse">Loading patterns…</div>
        ) : displayed.length === 0 ? (
          <EmptyState
            icon={<Search size={36} />}
            title="No patterns found"
            description="Train an RL model then distill it to extract golden patterns."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b border-[var(--border-color)]">
                  <th className="py-3 px-4 font-medium">ID</th>
                  <th className="py-3 px-4 font-medium">RL Model</th>
                  <th className="py-3 px-4 font-medium">Stock</th>
                  <th className="py-3 px-4 font-medium">Date</th>
                  <th className="py-3 px-4 font-medium">Interval</th>
                  <th className="py-3 px-4 font-medium">Regime</th>
                  <th className="py-3 px-4 font-medium">Action</th>
                  <th className="py-3 px-4 font-medium text-right">P&L (%)</th>
                  <th className="py-3 px-4 font-medium text-right">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-color)]">
                {displayed.map((p) => (
                  <tr
                    key={p.id}
                    className="cursor-pointer hover:bg-[var(--bg-hover)] transition-colors"
                    onClick={() => setSelectedPattern(p)}
                    title="Click to inspect pattern"
                  >
                    <td className="py-3 px-4 font-mono text-[var(--text-muted)]">#{p.id}</td>
                    <td className="py-3 px-4 font-medium">#{p.rl_model_id}</td>
                    <td className="py-3 px-4">#{p.stock_id}</td>
                    <td className="py-3 px-4 text-[var(--text-muted)]">{p.date}</td>
                    <td className="py-3 px-4 text-[var(--text-muted)]">
                      {typeof p.interval === 'object' && p.interval !== null
                        ? String((p.interval as any).value ?? p.interval)
                        : String(p.interval ?? '—')}
                    </td>
                    <td className="py-3 px-4">Regime {p.regime_id ?? '—'}</td>
                    <td className="py-3 px-4">
                      <Badge color={getBadgeColor(p.label) as any}>{getLabelTxt(p.label)}</Badge>
                    </td>
                    <td className={`py-3 px-4 text-right tabular-nums ${p.pnl_percent > 0 ? 'text-[var(--color-green)]' : 'text-[var(--color-red)]'}`}>
                      {p.pnl_percent?.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right tabular-nums">{(p.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* ── Pattern Detail Modal (popup) ─────────────────────────────── */}
      {selectedPattern && (
        <PatternDetailModal
          pattern={selectedPattern}
          onClose={() => setSelectedPattern(null)}
        />
      )}
    </div>
  );
}
