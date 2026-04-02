import { Card, Badge, EmptyState, PageHeader } from '../components/ui';
import { Search, Cpu, Layers } from 'lucide-react';

export default function PatternLab() {
  return (
    <div className="space-y-8">
      <PageHeader title="Pattern Lab" description="Explore golden patterns from RL replay memory" />

      <Card title="Golden Pattern Explorer">
        <EmptyState
          icon={<Search size={36} />}
          title="Pattern Explorer"
          description="Explore golden patterns extracted from RL replay. Train an RL model first, then distill it to extract patterns. Each pattern shows the market conditions where the RL agent took profitable actions."
        />
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card title="KNN Neighborhood Inspector">
          <p className="text-sm text-[var(--text-muted)] mb-4">
            For any prediction, view the K nearest matched patterns as mini charts
            with their regime, P&L, and feature similarity scores.
          </p>
          <div className="p-6 rounded-[var(--radius-sm)] bg-[var(--bg-input)] text-center">
            <EmptyState icon={<Cpu size={24} />} title="No prediction selected" description="Select a prediction to inspect KNN neighbors" />
          </div>
        </Card>

        <Card title="LSTM Sequence Inspector">
          <p className="text-sm text-[var(--text-muted)] mb-4">
            For any prediction, view the input sequence with saliency highlighting
            showing which timesteps and features the LSTM weighted most.
          </p>
          <div className="p-6 rounded-[var(--radius-sm)] bg-[var(--bg-input)] text-center">
            <EmptyState icon={<Layers size={24} />} title="No prediction selected" description="Select a prediction to inspect LSTM attention" />
          </div>
        </Card>
      </div>

      <Card title="Pattern Statistics">
        <div className="grid grid-cols-3 gap-5 text-center">
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="green">BUY</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">—</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="red">SELL</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">—</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
          <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
            <Badge color="gray">HOLD</Badge>
            <div className="text-2xl font-bold mt-2 tabular-nums">—</div>
            <div className="text-[11px] text-[var(--text-dim)] uppercase tracking-wider font-medium mt-0.5">patterns</div>
          </div>
        </div>
      </Card>
    </div>
  );
}
