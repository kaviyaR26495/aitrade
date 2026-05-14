import { useState, useEffect } from 'react';
import { Card, Button, Input, Select, SearchableSelect, PageHeader, Checkbox, Skeleton, Badge } from '../components/ui';
import { useConfig, useUpdateSetting, useUniverse, useSetUniverse, useStocks, useAuthStatus } from '../hooks/useApi';
import {
  getLoginUrl,
  zerodhaCallback,
  syncStockList,
  syncHolidays,
  listStocks,
  syncData,
  getChatProviders,
  getChatStatus,
  updateConfigBatch,
  listUniverseStocks,
  saveAutoLoginConfig,
  getAutoLoginStatus,
  triggerMorningTrade,
} from '../services/api';
import { useAppStore } from '../store/appStore';
import { ExternalLink, MessageCircle, Shield, ShieldCheck, ShieldAlert, Database, Server, Cpu, RefreshCw, Trash2, Zap, Clock, KeyRound, Eye, EyeOff, CheckCircle2, AlertCircle, Smartphone } from 'lucide-react';
import SystemMaintenance from '../components/SystemMaintenance';
import { ZerodhaAutoLogin } from '../plugins/ZerodhaAutoLogin';
import { Capacitor } from '@capacitor/core';

const SETTING_FIELDS = [
  { key: 'KITE_API_KEY', label: 'Kite API Key', type: 'password', guideId: 'zerodha-api-key' },
  { key: 'KITE_API_SECRET', label: 'Kite API Secret', type: 'password', guideId: 'zerodha-api-secret' },
  { key: 'ZERODHA_IP', label: 'Zerodha Whitelisted IP', type: 'text' },
  { key: 'STOPLOSS_PCT', label: 'Stop Loss %', type: 'number', guideId: 'stoploss-input' },
  { key: 'BUY_LIMIT', label: 'Buy Limit (₹)', type: 'number', guideId: 'buy-limit-input' },
  { key: 'SEQ_LEN', label: 'Sequence Length', type: 'number' },
  { key: 'PROFIT_HORIZON', label: 'Profit Horizon (days)', type: 'number' },
  { key: 'MIN_CONFIDENCE', label: 'Min Confidence', type: 'number', guideId: 'confidence-input' },
  { key: 'QUALITY_THRESHOLD', label: 'Quality Threshold', type: 'number', guideId: 'quality-threshold-input' },
];

export default function Settings() {
  const { data: config, isLoading } = useConfig();
  const updateSetting = useUpdateSetting();
  const { addNotification } = useAppStore();
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState<Set<string>>(new Set());
  const [loginUrl, setLoginUrl] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [populateLoading, setPopulateLoading] = useState(false);
  const [showLoginSteps, setShowLoginSteps] = useState(false);
  const { data: authStatus, refetch: refetchAuth, isFetching: isAuthFetching } = useAuthStatus();
  const isAuthenticated = authStatus?.authenticated ?? false;

  // Auto-Login state
  const [autoLoginUserId, setAutoLoginUserId] = useState('');
  const [autoLoginPassword, setAutoLoginPassword] = useState('');
  const [autoLoginTotpSecret, setAutoLoginTotpSecret] = useState('');
  const [autoLoginEnabled, setAutoLoginEnabled] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showTotp, setShowTotp] = useState(false);
  const [autoLoginSaving, setAutoLoginSaving] = useState(false);
  const [autoLoginConfigured, setAutoLoginConfigured] = useState(false);
  const [autoLoginTestLoading, setAutoLoginTestLoading] = useState(false);
  const [autoLoginStatus, setAutoLoginStatus] = useState<{ configured: boolean; enabled: boolean; user_id: string } | null>(null);

  // Stock Universe state
  const { data: universe } = useUniverse();
  const { data: allStocks } = useStocks();
  const setUniverseMutation = useSetUniverse();
  const [universeCategory, setUniverseCategory] = useState('nifty_50');
  const [customSymbols, setCustomSymbols] = useState<string[]>([]);
  const [customInput, setCustomInput] = useState('');

  useEffect(() => {
    if (universe) {
      setUniverseCategory(universe.category ?? 'nifty_50');
      setCustomSymbols(universe.custom_symbols ?? []);
    }
  }, [universe]);

  const { setPipelineUniverse } = useAppStore();

  const handleSaveUniverse = () => {
    setUniverseMutation.mutate(
      { category: universeCategory, customSymbols },
      {
        onSuccess: async () => {
          addNotification({ type: 'success', message: 'Stock universe updated' });
          // Sync global store with the newly resolved universe symbols
          try {
            const res = await listUniverseStocks();
            const symbols = (res.data ?? []).map((s: any) => s.symbol);
            setPipelineUniverse(symbols);
          } catch (e) {
            console.error('Failed to sync pipeline universe after save', e);
          }
        },
        onError: () => addNotification({ type: 'error', message: 'Failed to update universe' }),
      }
    );
  };

  const addCustomSymbol = (symbol: string) => {
    const sym = symbol.trim().toUpperCase();
    if (sym && !customSymbols.includes(sym)) {
      setCustomSymbols(prev => [...prev, sym]);
    }
    setCustomInput('');
  };

  const removeCustomSymbol = (symbol: string) => {
    setCustomSymbols(prev => prev.filter(s => s !== symbol));
  };

  const allStockOptions = (allStocks ?? []).map((s: any) => ({ value: s.symbol, label: s.symbol }));

  // Chat LLM config state
  const [chatProvider, setChatProvider] = useState('openai');
  const [chatModel, setChatModel] = useState('gpt-4.1-nano');
  const [chatApiKey, setChatApiKey] = useState('');
  const [chatOllamaUrl, setChatOllamaUrl] = useState('http://localhost:11434');
  const [chatProviders, setChatProviders] = useState<Record<string, { models: string[] }>>({});
  const [chatConfigured, setChatConfigured] = useState(false);
  const [chatSaving, setChatSaving] = useState(false);

  useEffect(() => {
    if (config) {
      setFormValues(config);
      setDirty(new Set());
    }
  }, [config]);

  useEffect(() => {
    Promise.all([getChatProviders(), getChatStatus()]).then(([prov, status]) => {
      setChatProviders(prov.data);
      setChatProvider(status.data.provider || 'openai');
      setChatModel(status.data.model || 'gpt-4.1-nano');
      setChatOllamaUrl(status.data.ollama_base_url || 'http://localhost:11434');
      setChatConfigured(status.data.configured);
    }).catch(() => {});
  }, []);

  const handleChange = (key: string, value: string) => {
    setFormValues((prev) => ({ ...prev, [key]: value }));
    setDirty((prev) => new Set([...prev, key]));
  };

  const handleSave = (key: string) => {
    updateSetting.mutate(
      { key, value: formValues[key] ?? '' },
      {
        onSuccess: () => {
          addNotification({ type: 'success', message: `${key} updated` });
          setDirty((prev) => {
            const next = new Set(prev);
            next.delete(key);
            return next;
          });
        },
        onError: () => addNotification({ type: 'error', message: 'Save failed' }),
      }
    );
  };

  const handleSaveAll = () => {
    dirty.forEach((key) => handleSave(key));
  };

  const handleZerodhaLogin = async () => {
    try {
      setIsLoggingIn(true);
      const res = await getLoginUrl();
      setLoginUrl(res.data.login_url ?? '');
      window.open(res.data.login_url, '_blank');
    } catch {
      addNotification({ type: 'error', message: 'Failed to get login URL' });
    } finally {
      setIsLoggingIn(false);
    }
  };

  // Auto-Login handlers
  useEffect(() => {
    const loadStatus = async () => {
      try {
        if (Capacitor.isNativePlatform()) {
          // On Android APK: native plugin has the ground-truth alarm state
          const nativeStatus = await ZerodhaAutoLogin.getStatus();
          setAutoLoginConfigured(nativeStatus.configured);
          setAutoLoginEnabled(nativeStatus.enabled);
          setAutoLoginUserId(nativeStatus.userId || '');
          setAutoLoginStatus({ configured: nativeStatus.configured, enabled: nativeStatus.enabled, user_id: nativeStatus.userId });
        } else {
          // On web: use the backend API
          const res = await getAutoLoginStatus();
          setAutoLoginStatus(res.data);
          setAutoLoginEnabled(res.data.enabled);
          setAutoLoginUserId(res.data.user_id || '');
          setAutoLoginConfigured(res.data.configured);
        }
      } catch {
        // Non-fatal — user just hasn't set it up yet
      }
    };
    loadStatus();
  }, []);

  const handleSaveAutoLogin = async () => {
    if (!autoLoginUserId || !autoLoginPassword || !autoLoginTotpSecret) {
      addNotification({ type: 'error', message: 'All three fields (User ID, Password, TOTP Secret) are required' });
      return;
    }
    setAutoLoginSaving(true);
    try {
      // Layer 1: Save to backend DB (for server-side Celery auth check + Telegram alerts)
      await saveAutoLoginConfig({
        zerodha_user_id: autoLoginUserId,
        zerodha_password: autoLoginPassword,
        zerodha_totp_secret: autoLoginTotpSecret,
        enabled: autoLoginEnabled,
      });

      // Layer 2: Save to device's EncryptedSharedPreferences + schedule AlarmManager alarm
      // This only works on the Android APK — on web it's a no-op
      if (Capacitor.isNativePlatform()) {
        await ZerodhaAutoLogin.saveCredentials({
          userId: autoLoginUserId,
          password: autoLoginPassword,
          totpSecret: autoLoginTotpSecret,
          enabled: autoLoginEnabled,
        });
      }

      setAutoLoginConfigured(true);
      setAutoLoginPassword('');
      setAutoLoginTotpSecret('');

      const msg = Capacitor.isNativePlatform()
        ? autoLoginEnabled
          ? '✅ Auto-login enabled — alarm set for 8:00 AM daily. Credentials stored on device.'
          : '💾 Config saved (disabled). Toggle Enable to activate the daily alarm.'
        : autoLoginEnabled
          ? '✅ Config saved to server. Install the APK on your phone to activate the 8 AM alarm.'
          : '💾 Config saved to server (disabled).';

      addNotification({ type: 'success', message: msg });
    } catch (e: any) {
      addNotification({ type: 'error', message: e?.response?.data?.detail ?? 'Failed to save auto-login config' });
    } finally {
      setAutoLoginSaving(false);
    }
  };

  const handleTestAutoLogin = async () => {
    setAutoLoginTestLoading(true);
    try {
      if (Capacitor.isNativePlatform()) {
        // On Android APK: fire the native login service right now (full WebView flow)
        const result = await ZerodhaAutoLogin.testLoginNow();
        addNotification({ type: 'success', message: `📱 ${result.message} — watch for the notification` });
      } else {
        // On web: trigger the backend trade sequence (requires Zerodha already connected)
        const res = await triggerMorningTrade();
        addNotification({ type: 'success', message: `✅ Trade sequence queued: task ${res.data.task_id}` });
      }
    } catch (e: any) {
      addNotification({ type: 'error', message: e?.response?.data?.detail ?? 'Test failed — ensure Zerodha is connected first' });
    } finally {
      setAutoLoginTestLoading(false);
    }
  };

  const handlePopulateFromZerodha = async () => {
    setPopulateLoading(true);
    try {
      const stocksRes = await syncStockList();
      const holidaysRes = await syncHolidays();
      const listRes = await listStocks();
      const firstBatchIds = (listRes.data ?? []).slice(0, 20).map((s: any) => s.id);

      if (firstBatchIds.length > 0) {
        await syncData(firstBatchIds, 'day');
      }

      addNotification({
        type: 'success',
        message:
          `Population done: stocks=${stocksRes.data?.stocks_populated ?? 0}, holidays=${holidaysRes.data?.holidays_synced ?? 0}, sample_sync=${firstBatchIds.length}`,
      });
    } catch (e: any) {
      addNotification({
        type: 'error',
        message: e?.response?.data?.detail ?? 'Populate failed. Authenticate Zerodha first.',
      });
    } finally {
      setPopulateLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <PageHeader title="Settings" description="Configure broker, models, and application parameters">
        {dirty.size > 0 && (
          <Button size="sm" onClick={handleSaveAll}>
            Save All ({dirty.size} changes)
          </Button>
        )}
      </PageHeader>

      {/* Zerodha Auth */}
      <Card title="Zerodha Authentication">
        <div className="space-y-5">
          {/* Status Banner */}
          <div className={`flex flex-wrap items-center justify-between gap-4 p-4 rounded-[var(--radius)] ${isAuthenticated ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-rose-500/10 border border-rose-500/30'}`}>
            <div className="flex items-center gap-3">
              {isAuthenticated
                ? <ShieldCheck size={20} className="text-emerald-400 flex-shrink-0" />
                : <ShieldAlert size={20} className="text-rose-400 flex-shrink-0" />
              }
              <div>
                <p className={`text-sm font-semibold ${isAuthenticated ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {isAuthenticated ? 'Connected to Zerodha' : 'Not connected — login required'}
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  {isAuthenticated
                    ? 'Token is active. Valid for 24 hours from the time of login.'
                    : 'Complete the steps below to authenticate your Kite account.'}
                </p>
              </div>
            </div>
            <div className="flex gap-2 flex-shrink-0">
              <Button variant="secondary" size="sm" onClick={() => refetchAuth()} loading={isAuthFetching}>
                <RefreshCw size={13} className={isAuthFetching ? "animate-spin" : ""} /> Refresh
              </Button>
              {isAuthenticated && (
                <Button variant="secondary" size="sm" onClick={() => setShowLoginSteps((v) => !v)}>
                  {showLoginSteps ? 'Hide' : 'Re-authenticate'}
                </Button>
              )}
            </div>
          </div>

          {/* Login Steps — always shown when not authenticated, toggled when authenticated */}
          {(!isAuthenticated || showLoginSteps) && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-4 p-5 rounded-[var(--radius)] bg-[var(--bg-input)]">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-[var(--primary-subtle)] flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-bold text-[var(--primary)]">1</span>
                  </div>
                  <div>
                    <p className="text-sm font-semibold">Open Zerodha Login</p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      Sign in and copy the redirect URL or request_token.
                    </p>
                  </div>
                </div>
                <Button onClick={handleZerodhaLogin} disabled={isLoggingIn} loading={isLoggingIn} data-guide-id="zerodha-login-btn" className="flex-shrink-0">
                  <ExternalLink size={14} /> {isLoggingIn ? "Logging in..." : "Login with Kite"}
                </Button>
              </div>

              {loginUrl && (
                <div className="p-4 rounded-[var(--radius)] bg-[var(--bg-input)] text-xs text-[var(--text-muted)] break-all font-mono leading-relaxed">
                  {loginUrl}
                </div>
              )}
            </>
          )}

            <div className="p-5 rounded-[var(--radius)] bg-[var(--bg-input)] space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-[var(--primary-subtle)] flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-[var(--primary)]">2</span>
                </div>
                <div>
                  <p className="text-sm font-semibold">Populate Database</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">
                    Syncs stock list, holidays, and sample OHLCV for first 20 active stocks.
                  </p>
                </div>
              </div>
              <Button onClick={handlePopulateFromZerodha} loading={populateLoading} data-guide-id="populate-btn">
                <Database size={14} /> Populate From Zerodha
              </Button>
            </div>
        </div>
      </Card>

      {/* Auto-Login Configuration */}
      <Card title="Automated Daily Login (8 AM)">
        <div className="space-y-5">
          {/* Status Banner */}
          <div className={`flex flex-wrap items-center justify-between gap-3 p-4 rounded-[var(--radius)] ${autoLoginConfigured && autoLoginEnabled ? 'bg-emerald-500/10 border border-emerald-500/30' : autoLoginConfigured ? 'bg-amber-500/10 border border-amber-500/30' : 'bg-[var(--bg-input)] border border-[var(--border)]'}`}>
            <div className="flex items-center gap-3">
              {autoLoginConfigured && autoLoginEnabled
                ? <CheckCircle2 size={18} className="text-emerald-400 flex-shrink-0" />
                : autoLoginConfigured
                ? <Clock size={18} className="text-amber-400 flex-shrink-0" />
                : <AlertCircle size={18} className="text-[var(--text-muted)] flex-shrink-0" />
              }
              <div>
                <p className={`text-sm font-semibold ${autoLoginConfigured && autoLoginEnabled ? 'text-emerald-300' : autoLoginConfigured ? 'text-amber-300' : 'text-[var(--text-muted)]'}`}>
                  {autoLoginConfigured && autoLoginEnabled
                    ? '✅ Auto-login active — Zerodha will connect at 8:00 AM daily'
                    : autoLoginConfigured
                    ? '⚡ Configured but disabled — toggle to activate'
                    : '🔒 Not configured — set up below for hands-free daily login'
                  }
                </p>
                {autoLoginStatus?.user_id && (
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">User: {autoLoginStatus.user_id}</p>
                )}
              </div>
            </div>
            {autoLoginConfigured && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--text-muted)]">Enabled</span>
                <button
                  type="button"
                  onClick={() => setAutoLoginEnabled(v => !v)}
                  className={`relative w-10 h-5 rounded-full transition-colors ${autoLoginEnabled ? 'bg-emerald-500' : 'bg-[var(--border)]'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${autoLoginEnabled ? 'translate-x-5' : ''}`} />
                </button>
              </div>
            )}
          </div>

          {/* Platform context banner */}
          <div className={`flex items-center gap-3 p-3 rounded-[var(--radius)] text-xs ${Capacitor.isNativePlatform() ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300' : 'bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)]'}`}>
            <Smartphone size={14} className="flex-shrink-0" />
            {Capacitor.isNativePlatform()
              ? '📱 Running on Android APK — the 8 AM alarm will fire even when the app is closed.'
              : '🌐 Running in browser — save config here, then open the app on your phone to activate the daily alarm.'
            }
          </div>

          {/* How it works */}
          <div className="p-4 rounded-[var(--radius)] bg-blue-500/5 border border-blue-500/20">
            <div className="flex items-start gap-3">
              <Zap size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-semibold text-blue-300 mb-1">How automated login works</p>
                <ol className="text-xs text-[var(--text-muted)] space-y-1 list-decimal list-inside">
                  <li>At 8:00 AM, the Android app wakes up automatically (even if you're asleep)</li>
                  <li>It opens Zerodha's login page in the background</li>
                  <li>Fills in your User ID, Password, and computes the TOTP code automatically</li>
                  <li>Sends the token to the server — trading begins</li>
                  <li>You get a notification: "✅ Connected to Zerodha"</li>
                </ol>
              </div>
            </div>
          </div>

          {/* TOTP Secret help */}
          <div className="p-4 rounded-[var(--radius)] bg-amber-500/5 border border-amber-500/20">
            <div className="flex items-start gap-3">
              <KeyRound size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-semibold text-amber-300 mb-1">Where to find your TOTP Secret Key</p>
                <p className="text-xs text-[var(--text-muted)]">
                  Go to <strong className="text-amber-200">Zerodha Console → My Profile → Security → Two-factor authentication → Setup TOTP</strong>.
                  The "Secret key" is the 32-character code shown (e.g. <code className="font-mono bg-black/30 px-1 rounded">JBSWY3DPEHPK3PXP...</code>).
                  This is NOT the 6-digit code — it's the seed used to generate it. Store this once; it doesn't change.
                </p>
              </div>
            </div>
          </div>

          {/* Credential fields */}
          <div className="grid grid-cols-1 gap-4">
            <Input
              value={autoLoginUserId}
              onChange={setAutoLoginUserId}
              label="Zerodha User ID"
              placeholder="e.g. AB1234"
              id="auto-login-user-id"
            />
            <div className="relative">
              <Input
                value={autoLoginPassword}
                onChange={setAutoLoginPassword}
                label="Zerodha Password"
                type={showPassword ? 'text' : 'password'}
                placeholder={autoLoginConfigured ? '••••••• (saved — enter new to replace)' : 'Your Zerodha login password'}
                id="auto-login-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(v => !v)}
                className="absolute right-3 top-8 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
              >
                {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <div className="relative">
              <Input
                value={autoLoginTotpSecret}
                onChange={setAutoLoginTotpSecret}
                label="TOTP Secret Key (base32)"
                type={showTotp ? 'text' : 'password'}
                placeholder={autoLoginConfigured ? '••••••• (saved — enter new to replace)' : 'e.g. JBSWY3DPEHPK3PXP...'}
                id="auto-login-totp-secret"
              />
              <button
                type="button"
                onClick={() => setShowTotp(v => !v)}
                className="absolute right-3 top-8 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
              >
                {showTotp ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          {/* Enable toggle for new setup */}
          {!autoLoginConfigured && (
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setAutoLoginEnabled(v => !v)}
                className={`relative w-10 h-5 rounded-full transition-colors ${autoLoginEnabled ? 'bg-emerald-500' : 'bg-[var(--border)]'}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${autoLoginEnabled ? 'translate-x-5' : ''}`} />
              </button>
              <span className="text-sm text-[var(--text-muted)]">Enable auto-login at 8:00 AM daily</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleSaveAutoLogin} loading={autoLoginSaving} id="save-auto-login-btn">
              <KeyRound size={14} /> {autoLoginConfigured ? 'Update Auto-Login Config' : 'Save & Enable Auto-Login'}
            </Button>
            {autoLoginConfigured && (
              <Button variant="secondary" onClick={handleTestAutoLogin} loading={autoLoginTestLoading} id="test-auto-login-btn">
                <Zap size={14} /> Test Morning Trade Now
              </Button>
            )}
          </div>
          <p className="text-xs text-[var(--text-muted)]">
            Credentials are encrypted and stored securely. The Android APK also caches them locally using Android Keystore (hardware-backed encryption).
          </p>
        </div>
      </Card>

      {/* Stock Universe */}
      <Card title="Stock Universe">
        <p className="text-sm text-[var(--text-muted)] mb-5">
          Choose which stocks to operate on across all pages. Pick a preset index or build a custom list.
        </p>
        <div className="space-y-5">
          <div className="flex items-end gap-4">
            <Select
              value={universeCategory}
              onChange={setUniverseCategory}
              label="Index / Category"
              options={[
                { value: 'nifty_50', label: 'Nifty 50' },
                { value: 'nifty_100', label: 'Nifty 100' },
                { value: 'nifty_500', label: 'Nifty 500 (all active)' },
                { value: 'custom', label: 'Custom only' },
              ]}
              className="flex-1"
            />
            {universe && (
              <Badge color="blue">{universe.resolved_count} stocks</Badge>
            )}
          </div>

          <div>
            <label className="block text-xs font-medium text-[var(--text-muted)] mb-2">Add Custom Stocks</label>
            <div className="flex gap-2">
              <SearchableSelect
                value={customInput}
                onChange={(v) => { addCustomSymbol(v); }}
                options={allStockOptions.filter((o: { value: string }) => !customSymbols.includes(o.value))}
                placeholder="Search & add stock..."
                className="flex-1"
              />
            </div>
          </div>

          {customSymbols.length > 0 && (
            <div>
              <label className="block text-xs font-medium text-[var(--text-muted)] mb-2">
                Custom Stocks ({customSymbols.length})
              </label>
              <div className="flex flex-wrap gap-2">
                {customSymbols.map(sym => (
                  <span
                    key={sym}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-[var(--primary-subtle)] text-[var(--primary)]"
                  >
                    {sym}
                    <button
                      type="button"
                      onClick={() => removeCustomSymbol(sym)}
                      className="hover:text-[var(--danger)] transition-colors cursor-pointer"
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          <Button onClick={handleSaveUniverse} loading={setUniverseMutation.isPending}>
            Save Universe
          </Button>
        </div>
      </Card>

      {/* Settings Form */}
      <Card title="Application Settings">
        {isLoading ? (
          <div className="space-y-4">
          <Skeleton lines={4} className="h-12" />
          </div>
        ) : (
          <div className="space-y-5">
            {SETTING_FIELDS.map(({ key, label, type, guideId }) => (
              <div key={key} className="flex items-end gap-4" {...(guideId ? { 'data-guide-id': guideId } : {})}>
                <Input
                  value={formValues[key] ?? ''}
                  onChange={(v) => handleChange(key, v)}
                  label={label}
                  type={type}
                  className="flex-1"
                />
                {dirty.has(key) && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleSave(key)}
                    loading={updateSetting.isPending}
                  >
                    Save
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Feature Configuration */}
      <Card title="Feature Configuration">
        <p className="text-sm text-[var(--text-muted)] mb-5">
          Select which indicator groups to include in model training.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {['TGRB', 'RSI', 'SRSI', 'KAMA', 'VWKAMA', 'OBV', 'MACD', 'SMA', 'Bollinger', 'ADX'].map((group) => (
            <div key={group} className="py-3 px-4 rounded-[var(--radius)] bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] transition-colors border border-transparent hover:border-[var(--border)]">
              <Checkbox checked={true} onChange={() => {}} label={group} />
            </div>
          ))}
        </div>
      </Card>

      <SystemMaintenance />

      {/* Chat Assistant */}
      <Card title="Chat Assistant" data-guide-id="chat-config-section">
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <span>Configure the LLM provider for the AITrade chatbot assistant.</span>
            {chatConfigured
              ? <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> Configured</span>
              : <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400"><span className="w-1.5 h-1.5 rounded-full bg-amber-400" /> Not configured</span>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Select
                value={chatProvider}
                onChange={(v) => {
                  setChatProvider(v);
                  const models = chatProviders[v]?.models ?? [];
                  if (models.length > 0) setChatModel(models[0]);
                }}
                label="Provider"
                options={Object.keys(chatProviders).map((p) => ({ value: p, label: p.charAt(0).toUpperCase() + p.slice(1) }))}
              />
            </div>
            <div>
              <Select
                value={chatModel}
                onChange={setChatModel}
                label="Model"
                options={(chatProviders[chatProvider]?.models ?? []).map((m) => ({ value: m, label: m }))}
              />
            </div>
          </div>
          {chatProvider !== 'ollama' ? (
            <Input
              value={chatApiKey}
              onChange={setChatApiKey}
              label="API Key"
              type="password"
              placeholder={chatConfigured ? '••••••• (saved — enter new to replace)' : 'Enter API key'}
            />
          ) : (
            <Input
              value={chatOllamaUrl}
              onChange={setChatOllamaUrl}
              label="Ollama Base URL"
              placeholder="http://localhost:11434"
            />
          )}
          <Button
            loading={chatSaving}
            onClick={async () => {
              setChatSaving(true);
              try {
                const settings = [
                  { key: 'llm_provider', value: chatProvider },
                  { key: 'llm_model', value: chatModel },
                  ...(chatApiKey ? [{ key: `llm_api_key_${chatProvider}`, value: chatApiKey }] : []),
                  { key: 'ollama_base_url', value: chatOllamaUrl },
                ];
                await updateConfigBatch(settings);
                setChatConfigured(true);
                setChatApiKey('');
                addNotification({ type: 'success', message: 'Chat assistant configured' });
              } catch {
                addNotification({ type: 'error', message: 'Failed to save chat config' });
              } finally {
                setChatSaving(false);
              }
            }}
          >
            <MessageCircle size={14} /> Save Chat Config
          </Button>
        </div>
      </Card>

      {/* System Info */}
      <Card title="System Info">
        <div className="grid grid-cols-2 gap-4 text-sm">
          {[
            { icon: <Database size={18} />, label: 'Database', value: 'MySQL 8' },
            { icon: <Server size={18} />, label: 'Backend', value: 'FastAPI + Python 3.11' },
            { icon: <Cpu size={18} />, label: 'ML Stack', value: 'SB3 + PyTorch + scikit-learn' },
            { icon: <Shield size={18} />, label: 'Pipeline', value: 'Regime → RL → KNN+LSTM' },
          ].map(({ icon, label, value }) => (
            <div key={label} className="flex items-center gap-4 p-4 rounded-[var(--radius)] bg-[var(--bg-input)]">
              <div className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--primary-subtle)] text-[var(--primary)]">{icon}</div>
              <div>
                <span className="text-xs text-[var(--text-muted)] font-medium">{label}</span>
                <p className="font-semibold text-sm mt-0.5">{value}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
