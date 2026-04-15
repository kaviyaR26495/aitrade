import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  createSeriesMarkers,
  ColorType,
  CandlestickSeries,
  LineSeries,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';

export interface OhlcvPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface IndicatorSeries {
  name: string;
  color: string;
  data: { time: string; value: number }[];
}

export interface TradeMarker {
  time: string;
  side: 'BUY' | 'SELL';
  price: number;
  reason?: string;
}

interface Props {
  ohlcv: OhlcvPoint[];
  indicators?: IndicatorSeries[];
  trades?: TradeMarker[];
  height?: number;
  className?: string;
  verticalLineDate?: string;
}

function resolveCssVar(varName: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim();
  return raw || fallback;
}

export default function LightweightCandleChart({
  ohlcv,
  indicators = [],
  trades = [],
  height = 500,
  className,
  verticalLineDate,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const vertLineRef = useRef<HTMLDivElement>(null);
  const [legendData, setLegendData] = useState<any>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const bg = resolveCssVar('--bg-card', '#11111d');
    const text = resolveCssVar('--text', '#f8fafc');
    const muted = resolveCssVar('--text-muted', '#64748b');
    const grid = resolveCssVar('--border', '#1e293b');
    const up = '#22c55e';
    const down = '#ef4444';

    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: muted,
        fontSize: 12,
        fontFamily: "'Inter', sans-serif",
      },
      grid: {
        vertLines: { color: grid, style: 1 },
        horzLines: { color: grid, style: 1 },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: muted,
          width: 1,
          style: 3,
          labelBackgroundColor: '#334155',
        },
        horzLine: {
          color: muted,
          width: 1,
          style: 3,
          labelBackgroundColor: '#334155',
        },
      },
      timeScale: {
        borderColor: grid,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 2,
        fixLeftEdge: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: up,
      downColor: down,
      borderVisible: false,
      wickUpColor: up,
      wickDownColor: down,
    });

    candleSeries.setData(ohlcv as any);

    const lineSeriesList: { name: string; series: ISeriesApi<'Line'> }[] = [];

    // Add Indicators
    indicators.forEach((ind) => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: ind.color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: ind.name,
      });
      lineSeries.setData(ind.data as any);
      lineSeriesList.push({ name: ind.name, series: lineSeries });
    });

    // Add Trade Markers
    if (trades.length > 0) {
      const markers: SeriesMarker<Time>[] = trades.map((t) => ({
        time: t.time as Time,
        position: t.side === 'BUY' ? 'belowBar' : 'aboveBar',
        color: t.side === 'BUY' ? up : down,
        shape: t.side === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: `${t.side} @ ${t.price}`,
        size: 2,
      }));
      createSeriesMarkers(candleSeries, markers);
    }

    // Subscribe to crosshair movement for legend
    chart.subscribeCrosshairMove((param) => {
      if (
        param.point === undefined ||
        !param.time ||
        param.point.x < 0 ||
        param.point.x > el.clientWidth ||
        param.point.y < 0 ||
        param.point.y > height
      ) {
        setLegendData(null);
      } else {
        const data = param.seriesData.get(candleSeries) as any;
        if (data) {
          const indicatorsAtPoint: Record<string, number> = {};
          lineSeriesList.forEach((item) => {
            const val = param.seriesData.get(item.series) as any;
            if (val) indicatorsAtPoint[item.name] = val.value ?? val;
          });

          setLegendData({
            time: param.time,
            ohlc: data,
            indicators: indicatorsAtPoint,
          });
        }
      }
    });

    chart.timeScale().fitContent();
    chartRef.current = chart;

    const updateVertLine = () => {
      if (verticalLineDate && vertLineRef.current && chart) {
        const x = chart.timeScale().timeToCoordinate(verticalLineDate as Time);
        if (x !== null && x > 0 && x < el.clientWidth) {
          vertLineRef.current.style.display = 'block';
          vertLineRef.current.style.left = `${x}px`;
        } else {
          vertLineRef.current.style.display = 'none';
        }
      }
    };

    const handleResize = () => {
      if (chart && el) {
        chart.applyOptions({ width: el.clientWidth });
        updateVertLine();
      }
    };
    window.addEventListener('resize', handleResize);

    if (verticalLineDate) {
      chart.timeScale().subscribeVisibleTimeRangeChange(updateVertLine);
      chart.timeScale().subscribeVisibleLogicalRangeChange(updateVertLine);
      setTimeout(updateVertLine, 50);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (verticalLineDate) {
        chart.timeScale().unsubscribeVisibleTimeRangeChange(updateVertLine);
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(updateVertLine);
      }
      chart.remove();
    };
  }, [ohlcv, indicators, trades, height, verticalLineDate]);

  return (
    <div ref={containerRef} className={`relative ${className}`} style={{ width: '100%', minHeight: height }}>
      {verticalLineDate && (
        <div 
          ref={vertLineRef}
          className="absolute top-0 bottom-0 w-[1px] bg-[var(--primary)] z-0 hidden pointer-events-none"
          style={{ mixBlendMode: 'screen', opacity: 0.5, borderLeft: '1px dashed var(--primary)' }}
        >
          <div className="absolute top-0 left-2 text-[10px] text-[var(--primary)] font-mono font-bold whitespace-nowrap bg-[var(--bg-card)]/50 px-1 rounded">Prediction</div>
        </div>
      )}
      {legendData && (
        <div className="absolute top-4 left-4 z-10 p-3 bg-[var(--bg-card)]/80 backdrop-blur-md rounded-lg border border-[var(--border)] text-[10px] space-y-1 pointer-events-none shadow-xl">
          <div className="flex gap-3 font-mono">
            <span className="text-[var(--text-muted)] font-bold uppercase">{legendData.time.toString()}</span>
            <div className="flex gap-2">
              <span>O: <span className="text-[var(--text)]">{legendData.ohlc.open.toFixed(1)}</span></span>
              <span>H: <span className="text-[var(--text)]">{legendData.ohlc.high.toFixed(1)}</span></span>
              <span>L: <span className="text-[var(--text)]">{legendData.ohlc.low.toFixed(1)}</span></span>
              <span>C: <span className="text-[var(--text)]">{legendData.ohlc.close.toFixed(1)}</span></span>
            </div>
          </div>
          {Object.keys(legendData.indicators).length > 0 && (
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 pt-1 border-t border-[var(--border)]">
              {Object.entries(legendData.indicators).map(([name, val]: [string, any]) => (
                <div key={name} className="flex gap-1 items-center">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: indicators.find(i => i.name === name)?.color }} />
                  <span className="text-[var(--text-muted)] uppercase font-semibold">{name}:</span>
                  <span className="text-[var(--text)] font-mono">{val.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
