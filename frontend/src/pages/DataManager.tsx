import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Card, Button, Input, Select, Badge, EmptyState, PageHeader, ListItem, Skeleton } from '../components/ui';
import { useStocks, useUniverseStocks, useSyncData, useOhlcv } from '../hooks/useApi';
import { syncData, syncStockList, syncHolidays } from '../services/api';
import { useAppStore } from '../store/appStore';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { CHART_TOOLTIP_STYLE, CHART_LABEL_STYLE, CHART_AXIS_PROPS } from '../components/ChartTheme';
import { RefreshCw, Download, BarChart3, Database } from 'lucide-react';

export default function DataManager() {
  const { data: allStocks, isLoading, refetch } = useStocks();
  const { data: universeStocks } = useUniverseStocks();
  const syncMutation = useSyncData();
  const { addNotification } = useAppStore();
  const [search, setSearch] = useState('');
  const [selectedStock, setSelectedStock] = useState<number | null>(null);
  const [interval, setInterval] = useState('day');
  const [showAll, setShowAll] = useState(false);
  const [syncingStockId, setSyncingStockId] = useState<number | null>(null);
  const [syncAllProgress, setSyncAllProgress] = useState<{ current: number; total: number } | null>(null);
  const [isSyncingAll, setIsSyncingAll] = useState(false);
  const qc = useQueryClient();

  const { data: ohlcv, isLoading: ohlcvLoading } = useOhlcv(selectedStock ?? 0, interval);

  const stocks = showAll ? allStocks : universeStocks;

  const filteredStocks = stocks?.filter((s: any) =>
    s.symbol?.toLowerCase().includes(search.toLowerCase()) ||
    s.name?.toLowerCase().includes(search.toLowerCase())
  ) ?? [];

  const handleSyncAll = async () => {
    if (isSyncingAll || !stocks?.length) return;
    setIsSyncingAll(true);

    const ids = stocks.map((s: any) => s.id);
    setSyncAllProgress({ current: 0, total: ids.length });

    let synced = 0, upToDate = 0, failed = 0;

    for (let i = 0; i < ids.length; i++) {
      const sid = ids[i];
      setSyncingStockId(sid);
      setSyncAllProgress({ current: i + 1, total: ids.length });
      try {
        const res = await syncData([sid], interval);
        const result = res.data?.[0];
        if (!result || result.ohlcv_synced === -1) {
          failed++;
        } else if (result.ohlcv_synced === 0) {
          upToDate++;
        } else {
          synced++;
          // Invalidate cache so chart refreshes if this stock is currently selected
          qc.invalidateQueries({ queryKey: ['ohlcv', sid, interval] });
        }
      } catch {
        failed++;
      }
    }

    setSyncingStockId(null);
    setSyncAllProgress(null);
    setIsSyncingAll(false);

    addNotification({
      type: failed > 0 ? 'warning' : 'success',
      message: `Sync complete — ${synced} updated, ${upToDate} already up to date${failed > 0 ? `, ${failed} failed` : ''}`,
    });
  };

  const handleSyncStock = (stockId: number) => {
    setSelectedStock(stockId); // auto-select so chart is visible
    setSyncingStockId(stockId);
    syncMutation.mutate(
      { stockIds: [stockId], interval },
      {
        onSuccess: (data) => {
          setSyncingStockId(null);
          const result = data?.data?.[0];
          if (!result || result.ohlcv_synced === -1) {
            const sym = result?.symbol && result.symbol !== '?' ? result.symbol + ': ' : '';
            addNotification({ type: 'error', message: `${sym}Sync failed — check Zerodha auth, then run Sync Stock List` });
          } else if (result.ohlcv_synced === 0) {
            addNotification({ type: 'info', message: `${result.symbol}: Already up to date — no new candles to fetch` });
          } else {
            const indMsg = result.indicators_computed > 0
              ? `, ${result.indicators_computed} indicators computed`
              : ' (indicators pending — will compute on next sync)';
            addNotification({ type: 'success', message: `${result.symbol}: ${result.ohlcv_synced} candles synced${indMsg}` });
          }
        },
        onError: () => {
          setSyncingStockId(null);
          addNotification({ type: 'error', message: 'Sync failed' });
        },
      }
    );
  };

  const handleSyncStockList = async () => {
    try {
      const res = await syncStockList();
      const count = res.data?.stocks_populated ?? 0;
      if (count === 0) {
        addNotification({ type: 'warning', message: 'Stock list sync returned 0 stocks — check Zerodha authentication' });
      } else {
        addNotification({ type: 'success', message: `Stock list updated: ${count} NSE instruments synced` });
      }
      refetch();
    } catch {
      addNotification({ type: 'error', message: 'Failed to sync stock list — authenticate Zerodha first' });
    }
  };

  const handleSyncHolidays = async () => {
    try {
      await syncHolidays();
      addNotification({ type: 'success', message: 'Holidays synced from NSE' });
    } catch {
      addNotification({ type: 'error', message: 'Failed to sync holidays' });
    }
  };

  const chartData = ohlcv?.map((d: any) => ({
    date: d.date,
    close: d.close,
    volume: d.volume,
  })) ?? [];

  return (
    <div className="space-y-8">
      <PageHeader title="Data Manager" description="Manage stock data and sync from NSE">
        <Button variant="secondary" size="sm" onClick={handleSyncStockList} data-guide-id="sync-stocks-btn">
          <Download size={14} /> Sync Stock List
        </Button>
        <Button variant="secondary" size="sm" onClick={handleSyncHolidays} data-guide-id="sync-holidays-btn">
          Sync Holidays
        </Button>
        <Button size="sm" onClick={handleSyncAll} disabled={isSyncingAll} data-guide-id="sync-all-btn">
          <RefreshCw size={14} className={`mr-1 ${syncAllProgress ? 'animate-spin' : ''}`} />
          {syncAllProgress ? `Syncing ${syncAllProgress.current}/${syncAllProgress.total}…` : 'Sync All'}
        </Button>
      </PageHeader>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card title="Stocks" className="lg:col-span-1">
          <div className="space-y-3">
            <Input value={search} onChange={setSearch} placeholder="Search stocks..." data-guide-id="stock-search" />
            <div className="flex items-center gap-2">
              <Select
                value={interval}
                onChange={setInterval}
                options={[
                  { value: 'day', label: 'Daily' },
                  { value: 'week', label: 'Weekly' },
                ]}
                className="flex-1"
                data-guide-id="interval-select-data"
              />
              <Button
                variant={showAll ? 'secondary' : 'primary'}
                size="sm"
                onClick={() => setShowAll(prev => !prev)}
              >
                {showAll ? 'All' : 'Universe'}
              </Button>
            </div>
            <div className="max-h-[480px] overflow-y-auto space-y-1.5">
              {isLoading ? (
                <Skeleton lines={5} className="h-10" />
              ) : filteredStocks.length === 0 ? (
                <EmptyState icon={<Database size={24} />} title="No stocks found" description="Sync stock list first to populate." />
              ) : (
                filteredStocks.slice(0, 100).map((s: any) => {
                  const isSyncing = syncingStockId === s.id;
                  return (
                    <ListItem
                      key={s.id}
                      onClick={() => setSelectedStock(s.id)}
                      active={selectedStock === s.id}
                      className={isSyncing ? 'ring-1 ring-[var(--primary)]/60 !bg-[var(--primary-subtle)]' : ''}
                      left={
                        <>
                          <span className="font-medium">{s.symbol}</span>
                          {s.name && <span className="text-xs text-[var(--text-dim)] truncate">{s.name}</span>}
                        </>
                      }
                      right={
                        <>
                          {s.is_active && <Badge color="green">Active</Badge>}
                          {isSyncing ? (
                            <span className="p-1.5 text-[var(--primary)]">
                              <RefreshCw size={12} className="animate-spin" />
                            </span>
                          ) : (
                            <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); handleSyncStock(s.id); }}>
                              <RefreshCw size={12} />
                            </Button>
                          )}
                        </>
                      }
                    />
                  );
                })
              )}
            </div>
          </div>
        </Card>

        <Card title={selectedStock ? `OHLCV Chart` : 'Select a stock'} className="lg:col-span-2" data-guide-id="ohlcv-chart" action={
          chartData.length > 0 ? (
            <span className="text-[11px] text-[var(--text-dim)] font-medium uppercase tracking-wider">{chartData.length} candles · {interval}</span>
          ) : null
        }>
          {ohlcvLoading ? (
            <div className="pt-2">
              <Skeleton lines={6} className="h-10" />
            </div>
          ) : chartData.length > 0 ? (
            <div className="-mx-2 mt-2">
              <ResponsiveContainer width="100%" height={360}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="closeGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="var(--primary)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" {...CHART_AXIS_PROPS} tickFormatter={(v) => v?.slice(5)} />
                  <YAxis {...CHART_AXIS_PROPS} />
                  <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelStyle={CHART_LABEL_STYLE} />
                  <Area type="monotone" dataKey="close" stroke="var(--primary)" strokeWidth={1.5} fill="url(#closeGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              icon={<BarChart3 size={28} />}
              title={selectedStock ? 'No data available' : 'Select a stock'}
              description={selectedStock ? 'Sync this stock to load OHLCV data.' : 'Choose a stock from the list to view its chart.'}
            />
          )}
        </Card>
      </div>
    </div>
  );
}
