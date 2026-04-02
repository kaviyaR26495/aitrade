/**
 * Shared Recharts theme configuration.
 * Centralizes tooltip, axis, and grid styling so every chart is consistent.
 */

export const CHART_TOOLTIP_STYLE: React.CSSProperties = {
  background: 'var(--chart-tooltip-bg)',
  border: '1px solid var(--chart-tooltip-border)',
  borderRadius: 'var(--radius-sm)',
  boxShadow: 'var(--shadow-lg)',
  fontSize: 12,
};

export const CHART_LABEL_STYLE: React.CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 11,
};

export const CHART_AXIS_PROPS = {
  stroke: 'var(--chart-axis)',
  fontSize: 11,
  axisLine: false,
  tickLine: false,
} as const;

export const CHART_GRID_PROPS = {
  strokeDasharray: '3 3',
  stroke: 'var(--chart-grid)',
} as const;
