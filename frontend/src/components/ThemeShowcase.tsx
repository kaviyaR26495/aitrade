import { useAppStore } from '../store/appStore';
import type { ThemeId } from '../store/appStore';
import { Check, Monitor, Moon, Palette, Sparkles } from 'lucide-react';
import { Card, PageHeader, Badge, Button } from './ui';

const THEMES: { id: ThemeId; name: string; description: string; icon: typeof Moon; colors: string[] }[] = [
  {
    id: 'deep-ocean',
    name: 'Deep Ocean',
    description: 'Dark navy with indigo accents. Classic, focused.',
    icon: Moon,
    colors: ['#060b16', '#0c1322', '#6366f1', '#06b6d4'],
  },
  {
    id: 'midnight',
    name: 'Midnight Purple',
    description: 'Rich purple tones with warm pink gradients.',
    icon: Sparkles,
    colors: ['#0b0a14', '#13112a', '#8b5cf6', '#ec4899'],
  },
  {
    id: 'carbon',
    name: 'Carbon',
    description: 'Neutral dark gray. Clean and minimal.',
    icon: Monitor,
    colors: ['#0e0e0e', '#1a1a1a', '#3b82f6', '#10b981'],
  },
  {
    id: 'emerald',
    name: 'Emerald Night',
    description: 'Deep green dark theme. Calm and natural.',
    icon: Palette,
    colors: ['#06120e', '#0c1e16', '#10b981', '#06b6d4'],
  },
];

export default function ThemeShowcase() {
  const { theme, setTheme } = useAppStore();

  return (
    <div className="max-w-5xl space-y-8">
      <PageHeader
        title="Appearance"
        description="Choose a theme to personalize your workspace"
      />

      {/* Theme Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {THEMES.map((t) => {
          const isActive = theme === t.id;
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setTheme(t.id)}
              className={`
                relative text-left p-5 rounded-[var(--radius-lg)] border-2 transition-all duration-200
                hover:shadow-[var(--shadow-md)] cursor-pointer group
                ${isActive
                  ? 'border-[var(--primary)] bg-[var(--primary-subtle)] shadow-[0_0_20px_var(--primary-glow)]'
                  : 'border-[var(--border)] bg-[var(--bg-card)] hover:border-[var(--text-dim)]'
                }
              `}
            >
              {/* Active checkmark */}
              {isActive && (
                <div className="absolute top-3 right-3 w-6 h-6 rounded-full bg-[var(--primary)] flex items-center justify-center">
                  <Check size={14} className="text-white" />
                </div>
              )}

              <div className="flex items-start gap-3">
                {/* Color preview */}
                <div className="flex-shrink-0 w-12 h-12 rounded-lg overflow-hidden grid grid-cols-2 grid-rows-2 shadow-inner border border-white/5">
                  {t.colors.map((c, i) => (
                    <div key={i} style={{ backgroundColor: c }} />
                  ))}
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <Icon size={14} className={isActive ? 'text-[var(--primary)]' : 'text-[var(--text-muted)]'} />
                    <span className={`text-sm font-semibold ${isActive ? 'text-[var(--primary)]' : 'text-[var(--text)]'}`}>
                      {t.name}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">{t.description}</p>
                </div>
              </div>

              {/* Mini preview bar */}
              <div className="mt-3 flex gap-1.5 h-1 rounded-full overflow-hidden">
                {t.colors.map((c, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
            </button>
          );
        })}
      </div>

      {/* Live Preview Section */}
      <Card title="Live Preview">
        <p className="text-[13px] text-[var(--text-muted)] mb-6">
          This section updates live as you switch themes. What you see here is what the entire app looks like.
        </p>

        {/* Sample stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius)] p-4">
            <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-semibold mb-2">Primary</div>
            <div className="text-lg font-bold" style={{ color: 'var(--primary)' }}>Active</div>
          </div>
          <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius)] p-4">
            <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-semibold mb-2">Success</div>
            <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>+12.4%</div>
          </div>
          <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius)] p-4">
            <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-semibold mb-2">Warning</div>
            <div className="text-lg font-bold" style={{ color: 'var(--warning)' }}>3 alerts</div>
          </div>
          <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-[var(--radius)] p-4">
            <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-semibold mb-2">Danger</div>
            <div className="text-lg font-bold" style={{ color: 'var(--danger)' }}>-2.1%</div>
          </div>
        </div>

        {/* Sample badges */}
        <div className="flex flex-wrap gap-2 mb-6">
          <Badge color="green">BUY</Badge>
          <Badge color="red">SELL</Badge>
          <Badge color="yellow">PENDING</Badge>
          <Badge color="blue">INFO</Badge>
          <Badge color="gray">HOLD</Badge>
          <Badge color="purple">ML</Badge>
        </div>

        {/* Sample buttons */}
        <div className="flex flex-wrap gap-3 mb-6">
          <Button variant="primary" size="sm">Primary</Button>
          <Button variant="secondary" size="sm">Secondary</Button>
          <Button variant="danger" size="sm">Danger</Button>
          <Button variant="ghost" size="sm">Ghost</Button>
        </div>

        {/* Sample surface stack */}
        <div className="grid grid-cols-5 gap-3">
          {[
            { label: 'Page BG', css: 'var(--bg)' },
            { label: 'Card', css: 'var(--bg-card)' },
            { label: 'Input', css: 'var(--bg-input)' },
            { label: 'Hover', css: 'var(--bg-hover)' },
            { label: 'Active', css: 'var(--bg-active)' },
          ].map((s) => (
            <div
              key={s.label}
              className="py-3 px-2 rounded-lg border border-[var(--border)] text-center"
              style={{ backgroundColor: s.css }}
            >
              <div className="text-[11px] font-medium text-[var(--text-muted)]">{s.label}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Color Palette */}
      <Card title="Color Palette">
        <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
          {[
            { name: 'Primary', var: '--primary' },
            { name: 'Primary Hover', var: '--primary-hover' },
            { name: 'Success', var: '--success' },
            { name: 'Danger', var: '--danger' },
            { name: 'Warning', var: '--warning' },
            { name: 'Info', var: '--info' },
          ].map((c) => (
            <div key={c.name} className="text-center">
              <div
                className="w-full h-12 rounded-lg border border-[var(--border)] mb-2"
                style={{ backgroundColor: `var(${c.var})` }}
              />
              <div className="text-xs font-medium text-[var(--text)]">{c.name}</div>
              <div className="text-[10px] text-[var(--text-dim)] font-mono">{c.var}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

export { ThemeShowcase };