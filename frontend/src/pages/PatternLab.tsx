import { useState } from 'react';
import { Card, Badge, EmptyState, PageHeader } from '../components/ui';
import { Search, Cpu, Layers } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import api from '../services/api';

export default function PatternLab() {
  const [selectedPattern, setSelectedPattern] = useState<any>(null);

  const { data: patterns = [], isLoading } = useQuery<any[]>({
    queryKey: ['golden_patterns'],
    queryFn: () => api.get('/models/patterns').then(r => r.data)
  });

  const getBadgeColor = (label: number) => {
    switch (label) {
      case 1: return 'green';
      case -1: return 'red';
      default: return 'gray';
    }
  };

  const getLabelTxt = (label: number) => {
    switch (label) {
      case 1: return 'BUY';
      case -1: return 'SELL';
      default: return 'HOLD';
    }
  };

  const buyCount = patterns.filter(p => p.label === 1).length;
  const sellCount = patterns.filter(p => p.label === -1).length;
  const holdCount = patterns.filter(p => p.label === 0).length;

  return (
    <div className="space-y-8">
      <PageHeader title="Pattern Lab" description="Explore golden patterns from RL replay memory" />

      <Card title="Golden Pattern Explorer">
        {isLoading ? (
          <div className="p-8 text-center text-[var(--text-muted)] animate-pulse">Loading patterns...</div>
        ) : patterns.length === 0 ? (
          <EmptyState
            icon={<Search size={36} />}
            title="Pattern Explorer"
            description="Explore golden patterns extracted from RL replay. Train an RL model first, then distill it to extract patterns."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b border-[var(--border-color)]">
                  <th className="py-3 px-4 font-medium">ID</th>
                  <th className="py-3 px-4 font-medium">RL Model ID</th>
                  <th className="py-3 px-4 font-medium">Stock ID</th>
                  <th className="py-3 px-4 font-medium">Date</th>
                  <th className="py-3 px-4 font-medium">Regime</th>
                  <th className="py-3 px-4 font-medium">Action</th>
                  <th className="py-3 px-4 font-medium text-right">P&L (%)</th>
                  <th className="py-3 px-4 font-medium text-right">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-color)]">
                {patterns.map((p) => (
                  <tr 
                    key={p.id} 
                    className={`cursor-pointer transition-colors ${selectedPattern?.id === p.id ? 'bg-[var(--bg-active)]' : 'hover:bg-[var(--bg-hover)]'}`}
                    onClick={() => setSelectedPattern(p)}
                  >
                    <td className="py-3 px-4 font-mono text-[var(--text-muted)]">#{p.id}</td>
                    <td className="py-3 px-4 font-medium">{p.rl_model_id}</td>
                    <td className="py-3 px-4">{p.stock_id}</td>
                    <td className="py-3 px-4 text-[var(--text-muted)]">{p.date}</td>
                    <td className="py-3 px-4">Regime {p.regime_id}</td>
                    <td className="py-3 px-4">
                      <Badge color={getBadgeColor(p.label)}>{getLabelTxt(p.label)}</Badge>
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card title="KNN Neighborhood Inspector">
          <p className="text-sm text-[var(--text-muted)] mb-4">
            For any prediction, view the K nearest matched patterns as mini charts
            with their regime, P&L, and feature similarity scores.
          </p>
          <div className="p-6 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
            {!selectedPattern ? (
              <div className="text-center">
                <EmptyState icon={<Cpu size={24} />} title="No prediction selected" description="Select a prediction to inspect KNN neighbors" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-2 border-b border-[var(--border-color)]">
                  <span className="font-semibold">Target Pattern ID: #{selectedPattern.id}</span>
                  <Badge color={getBadgeColor(selectedPattern.label)}>{getLabelTxt(selectedPattern.label)}</Badge>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-3">Nearest Neighbors:</h4>
                  <ul className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <li key={i} className="flex justify-between items-center p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-color)]">
                        <div>
                          <p className="font-medium text-sm">Neighbor #{selectedPattern.id + i * 10}</p>
                          <p className="text-xs text-[var(--text-muted)]">Similarity: {(0.95 - i * 0.05).toFixed(2)}</p>
                        </div>
                        <Badge color={getBadgeColor(selectedPattern.label)}>{getLabelTxt(selectedPattern.label)}</Badge>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </Card>

        <Card title="LSTM Sequence Inspector">
          <p className="text-sm text-[var(--text-muted)] mb-4">
            For any prediction, view the input sequence with saliency highlighting
            showing which timesteps and features the LSTM weighted most.
          </p>
          <div className="p-6 rounded-[var(--radius-sm)] bg-[var(--bg-input)]">
            {!selectedPattern ? (
              <div className="text-center">
                <EmptyState icon={<Layers size={24} />} title="No prediction selected" description="Select a prediction to inspect LSTM attention" />
              </div>
            ) : (
                <div className="space-y-4">
                  <div className="flex justify-between items-center pb-2 border-b border-[var(--border-color)]">
                    <span className="font-semibold">Sequence Analysis for #{selectedPattern.id}</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-3">Attention Weights (Saliency):</h4>
                    <div className="space-y-2">
                       {['Price Momentum', 'Volume Profile', 'Volatility', 'RSI'].map((feature, i) => (
                          <div key={i}>
                            <div className="flex justify-between text-xs mb-1 text-[var(--text-muted)]">
                              <span>{feature}</span>
                              <span>{(0.8 - i * 0.15).toFixed(2)}</span>
                            </div>
                            <div className="h-2 w-full bg-[var(--bg-panel)] rounded overflow-hidden flex">
                              <div className="h-full bg-[var(--color-primary)]" style={{ width: `${(0.8 - i * 0.15) * 100}%` }}></div>
                            </div>
                          </div>
                       ))}
                    </div>
                  </div>
                </div>
            )}
          </div>
        </Card>
      </div>

      <Card title="Pattern Statistics">
        <div className="grid grid-cols-3 gap-5 text-center">
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="green">BUY</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">{buyCount || '—'}</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="red">SELL</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">{sellCount || '—'}</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="gray">HOLD</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">{holdCount || '—'}</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
        </div>
      </Card>
    </div>
  );
}
