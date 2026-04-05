/**
 * Canvas-based price chart with regime-transition markers.
 *
 * Replaces the SVG Recharts AreaChart + ReferenceArea combination used in
 * RegimeAnalysis.tsx.  lightweight-charts renders ~750 daily candles to a
 * single <canvas> element, eliminating the per-point SVG DOM nodes that
 * caused the browser to freeze.
 *
 * Regime coloring is achieved via Series.setMarkers():
 *   - A colored square marker is placed at every regime-change point.
 *   - A coloured baseline histogram at the bottom provides a continuous
 *     regime band similar to the old ReferenceArea blocks.
 */

import { useEffect, useRef } from 'react';
import {
  createChart,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';

export interface RegimePoint {
  time: string;   // YYYY-MM-DD
  value: number;  // close price
  regime_id: number;
}

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

function resolveCssVar(varName: string, fallback: string): string {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim();
  return raw || fallback;
}

interface Props {
  data: RegimePoint[];
  height?: number;
  className?: string;
}

export default function LightweightRegimeChart({ data, height = 340, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const priceSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const regimeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const muted = resolveCssVar('--text-muted', '#64748b');
    const grid  = resolveCssVar('--border', '#2a2a3e');
    const text  = resolveCssVar('--text', '#e2e8f0');

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
        scaleMargins: { top: 0.1, bottom: 0.15 },
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

    // ── Main price series ────────────────────────────────────────────
    const priceSeries = chart.addAreaSeries({
      lineColor: text,
      topColor: `${muted}18`,
      bottomColor: `${muted}00`,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    priceSeriesRef.current = priceSeries;

    // ── Regime histogram — thin coloured band at the bottom ──────────
    const regimeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'regime',
    });
    chart.priceScale('regime').applyOptions({
      scaleMargins: { top: 0.95, bottom: 0 },
    });
    regimeSeriesRef.current = regimeSeries;

    _applyData(priceSeries, regimeSeries, data);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      chartRef.current?.applyOptions({ width: el.clientWidth });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      priceSeriesRef.current = null;
      regimeSeriesRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (priceSeriesRef.current && regimeSeriesRef.current && data.length > 0) {
      _applyData(priceSeriesRef.current, regimeSeriesRef.current, data);
      chartRef.current?.timeScale().fitContent();
    }
  }, [data]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ width: '100%', height: `${height}px` }}
    />
  );
}

function _applyData(
  priceSeries: ISeriesApi<'Area'>,
  regimeSeries: ISeriesApi<'Histogram'>,
  data: RegimePoint[],
) {
  if (data.length === 0) return;

  const priceData = data.map((d) => ({ time: d.time as Time, value: d.value }));
  priceSeries.setData(priceData);

  // Regime histogram: constant height = 1 per bar, colour = regime colour
  const regimeData = data.map((d) => ({
    time: d.time as Time,
    value: 1,
    color: `${REGIME_COLORS[d.regime_id] ?? '#6b7280'}80`, // 50% alpha
  }));
  regimeSeries.setData(regimeData);

  // Regime-change markers on the price series
  const markers: SeriesMarker<Time>[] = [];
  for (let i = 1; i < data.length; i++) {
    if (data[i].regime_id !== data[i - 1].regime_id) {
      markers.push({
        time: data[i].time as Time,
        position: 'aboveBar',
        color: REGIME_COLORS[data[i].regime_id] ?? '#6b7280',
        shape: 'circle',
        size: 1,
        text: REGIME_LABELS[data[i].regime_id] ?? '',
      });
    }
  }
  priceSeries.setMarkers(markers);
}
