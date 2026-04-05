/**
 * Canvas-based area chart using TradingView's lightweight-charts library.
 *
 * Replaces SVG-based Recharts AreaChart for financial time-series data.
 * lightweight-charts renders to a <canvas> element which handles 750+ data
 * points without the DOM bottleneck caused by SVG nodes.
 *
 * Props
 * -----
 * data         – Array of { time: string (YYYY-MM-DD), value: number }
 * secondSeries – Optional second line (e.g. benchmark). Same shape as data.
 * secondLabel  – Legend label for the second series (default: "Benchmark")
 * height       – Chart height in px (default: 340)
 * color        – Main series colour (default: CSS var(--primary) resolved at mount)
 * valueFormatter – Tick formatter for the right price axis (default: identity)
 */

import { useEffect, useRef } from 'react';
import {
  createChart,
  ColorType,
  AreaSeries,
  type IChartApi,
  type ISeriesApi,
} from 'lightweight-charts';

export interface ChartPoint {
  time: string;  // YYYY-MM-DD
  value: number;
}

interface Props {
  data: ChartPoint[];
  secondSeries?: ChartPoint[];
  secondLabel?: string;
  height?: number;
  color?: string;
  secondColor?: string;
  valueFormatter?: (v: number) => string;
  className?: string;
}

// Resolve a CSS variable to a hex colour string at runtime.
function resolveCssVar(varName: string, fallback: string): string {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim();
  return raw || fallback;
}

export default function LightweightAreaChart({
  data,
  secondSeries,
  secondLabel = 'Benchmark',
  height = 340,
  color,
  secondColor,
  valueFormatter,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const secondSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const bg     = resolveCssVar('--bg-card', '#1c1c2e');
    const text   = resolveCssVar('--text', '#e2e8f0');
    const muted  = resolveCssVar('--text-muted', '#64748b');
    const grid   = resolveCssVar('--border', '#2a2a3e');
    const main   = color  ?? resolveCssVar('--primary', '#6366f1');
    const second = secondColor ?? resolveCssVar('--text-muted', '#64748b');

    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: muted,
        fontSize: 11,
        fontFamily: 'inherit',
      },
      grid: {
        vertLines: { color: grid, style: 1 },
        horzLines: { color: grid, style: 1 },
      },
      rightPriceScale: {
        borderColor: grid,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: grid,
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        vertLine: { color: muted },
        horzLine: { color: muted },
      },
    });

    // Apply custom price formatter if supplied
    if (valueFormatter) {
      chart.applyOptions({
        localization: { priceFormatter: valueFormatter },
      });
    }

    const mainSeries = chart.addSeries(AreaSeries, {
      lineColor: main,
      topColor: `${main}26`,   // 15% opacity
      bottomColor: `${main}00`,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    mainSeries.setData(data as any);
    mainSeriesRef.current = mainSeries;

    if (secondSeries && secondSeries.length > 0) {
      const s2 = chart.addSeries(AreaSeries, {
        lineColor: second,
        topColor: `${second}00`,
        bottomColor: `${second}00`,
        lineWidth: 1,
        lineStyle: 1,  // dashed
        priceLineVisible: false,
        lastValueVisible: false,
        title: secondLabel,
      });
      s2.setData(secondSeries as any);
      secondSeriesRef.current = s2;
    }

    chart.timeScale().fitContent();
    chartRef.current = chart;

    // Responsive resize
    const ro = new ResizeObserver(() => {
      if (chartRef.current && el) {
        chartRef.current.applyOptions({ width: el.clientWidth });
      }
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      mainSeriesRef.current = null;
      secondSeriesRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update data without rebuilding the chart instance
  useEffect(() => {
    if (mainSeriesRef.current && data.length > 0) {
      mainSeriesRef.current.setData(data as any);
      chartRef.current?.timeScale().fitContent();
    }
  }, [data]);

  useEffect(() => {
    if (secondSeriesRef.current && secondSeries && secondSeries.length > 0) {
      secondSeriesRef.current.setData(secondSeries as any);
    }
  }, [secondSeries]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ width: '100%', height: `${height}px` }}
    />
  );
}
