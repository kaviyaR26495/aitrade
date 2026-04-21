import { create } from 'zustand';

export type ThemeId = 'deep-ocean' | 'midnight' | 'carbon' | 'emerald';

// ── API Console log ────────────────────────────────────────────────────
export interface ApiLogEntry {
  id: string;
  timestamp: number;       // unix ms
  method: string;
  url: string;
  status: number | null;   // null = in-flight, -1 = network error
  duration: number | null; // ms
  requestBody?: string;
  responseBody?: string;
}

const LOG_STORAGE_KEY = 'aitrade-api-log';
const LOG_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

function loadLog(): ApiLogEntry[] {
  try {
    const raw = localStorage.getItem(LOG_STORAGE_KEY);
    if (!raw) return [];
    const all: ApiLogEntry[] = JSON.parse(raw);
    const cutoff = Date.now() - LOG_TTL_MS;
    return all.filter((e) => e.timestamp >= cutoff);
  } catch { return []; }
}

function persistLog(entries: ApiLogEntry[]) {
  try {
    const cutoff = Date.now() - LOG_TTL_MS;
    // Keep at most 500 entries to guard against quota exhaustion
    const trimmed = entries.filter((e) => e.timestamp >= cutoff).slice(-500);
    localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(trimmed));
  } catch { /* storage quota — ignore */ }
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
}

export interface StockSelectorSelection {
  manualSelected: Set<string>;
  activePresetCategories: Set<string>;
  excludedSymbols: Set<string>;
}

const STOCK_SELECTOR_DRAFT_KEY = 'aitrade-stock-selector-draft';
const PIPELINE_UNIVERSE_STORAGE_KEY = 'aitrade-pipeline-universe';

function saveStockSelectorDraft(selection: StockSelectorSelection) {
  try {
    const serialized = {
      manualSelected: Array.from(selection.manualSelected),
      activePresetCategories: Array.from(selection.activePresetCategories),
      excludedSymbols: Array.from(selection.excludedSymbols),
    };
    localStorage.setItem(STOCK_SELECTOR_DRAFT_KEY, JSON.stringify(serialized));
  } catch (e) {
    console.error('Failed to save stock selector draft', e);
  }
}

function loadStockSelectorDraft(): StockSelectorSelection {
  try {
    const raw = localStorage.getItem(STOCK_SELECTOR_DRAFT_KEY);
    if (!raw) return createEmptyStockSelectorSelection();
    const parsed = JSON.parse(raw);
    return {
      manualSelected: new Set(parsed.manualSelected || []),
      activePresetCategories: new Set(parsed.activePresetCategories || []),
      excludedSymbols: new Set(parsed.excludedSymbols || []),
    };
  } catch {
    return createEmptyStockSelectorSelection();
  }
}

function savePipelineUniverse(symbols: string[]) {
  try {
    localStorage.setItem(PIPELINE_UNIVERSE_STORAGE_KEY, JSON.stringify(symbols));
  } catch (e) {
    console.error('Failed to save pipeline universe', e);
  }
}

function loadPipelineUniverse(): string[] {
  try {
    const raw = localStorage.getItem(PIPELINE_UNIVERSE_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

const ACTIVE_PIPELINE_JOB_KEY = 'aitrade-active-pipeline-job';

function saveActivePipelineJobId(id: string | null) {
  try {
    if (id) {
      localStorage.setItem(ACTIVE_PIPELINE_JOB_KEY, id);
    } else {
      localStorage.removeItem(ACTIVE_PIPELINE_JOB_KEY);
    }
  } catch (e) {
    console.error('Failed to save active pipeline job ID', e);
  }
}

function loadActivePipelineJobId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_PIPELINE_JOB_KEY);
  } catch {
    return null;
  }
}

function createEmptyStockSelectorSelection(): StockSelectorSelection {
  return {
    manualSelected: new Set(),
    activePresetCategories: new Set(),
    excludedSymbols: new Set(),
  };
}

// ── Notifications ──────────────────────────────────────────────────────
interface AppState {
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  theme: ThemeId;
  setTheme: (t: ThemeId) => void;

  loading: boolean;
  setLoading: (v: boolean) => void;

  notifications: Notification[];
  addNotification: (n: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;

  // API Console
  apiLog: ApiLogEntry[];
  apiConsoleOpen: boolean;
  addApiLog: (entry: ApiLogEntry) => void;
  updateApiLog: (id: string, patch: Partial<ApiLogEntry>) => void;
  clearApiLog: () => void;
  toggleApiConsole: () => void;

  // Training Console
  trainingConsoleOpen: boolean;
  trainingConsoleModelId: number | null;
  toggleTrainingConsole: () => void;
  openTrainingConsole: (modelId: number) => void;
  setTrainingConsoleModelId: (id: number | null) => void;

  // Stock selector working draft — persists while navigating the SPA
  stockSelectorSelection: StockSelectorSelection;
  setStockSelectorSelection: (
    next: StockSelectorSelection | ((prev: StockSelectorSelection) => StockSelectorSelection),
  ) => void;
  resetStockSelectorSelection: () => void;

  // Pipeline Universe — stock symbols confirmed for One-Click Training Pipeline
  pipelineUniverse: string[];
  setPipelineUniverse: (symbols: string[]) => void;
  activePipelineJobId: string | null;
  setActivePipelineJobId: (id: string | null) => void;

  isZerodhaAuthenticated: boolean;
  setIsZerodhaAuthenticated: (v: boolean) => void;
  isAuthRefreshing: boolean;

  // Retrain alert — shown when models are stale (> 30 days since last CT run)
  retrainAlert: boolean;
  retrainDaysSince: number | null;
  retrainHasModels: boolean;
  setRetrainAlert: (v: boolean) => void;
  checkRetrainStatus: () => Promise<void>;

  // Initialization
  isInitialized: boolean;
  initialize: () => Promise<void>;
  error: string | null;
}

function getStoredTheme(): ThemeId {
  try {
    const saved = localStorage.getItem('aitrade-theme');
    if (saved && ['deep-ocean', 'midnight', 'carbon', 'emerald'].includes(saved)) {
      return saved as ThemeId;
    }
  } catch { /* ignore */ }
  return 'deep-ocean';
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  theme: getStoredTheme(),
  setTheme: (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('aitrade-theme', theme);
    set({ theme });
  },

  loading: false,
  setLoading: (loading) => set({ loading }),

  notifications: [],
  addNotification: (n) =>
    set((s) => ({
      notifications: [...s.notifications, { ...n, id: crypto.randomUUID() }],
    })),
  removeNotification: (id) =>
    set((s) => ({
      notifications: s.notifications.filter((n) => n.id !== id),
    })),

  // API Console
  apiLog: loadLog(),
  apiConsoleOpen: false,
  addApiLog: (entry) =>
    set((s) => {
      const updated = [...s.apiLog, entry];
      persistLog(updated);
      return { apiLog: updated };
    }),
  updateApiLog: (id, patch) =>
    set((s) => {
      const updated = s.apiLog.map((e) => (e.id === id ? { ...e, ...patch } : e));
      persistLog(updated);
      return { apiLog: updated };
    }),
  clearApiLog: () => {
    try { localStorage.removeItem(LOG_STORAGE_KEY); } catch { /* ignore */ }
    set({ apiLog: [] });
  },
  toggleApiConsole: () => set((s) => ({ apiConsoleOpen: !s.apiConsoleOpen })),

  // Training Console
  trainingConsoleOpen: false,
  trainingConsoleModelId: null,
  toggleTrainingConsole: () => set((s) => ({ trainingConsoleOpen: !s.trainingConsoleOpen })),
  openTrainingConsole: (modelId) => set({ trainingConsoleOpen: true, trainingConsoleModelId: modelId }),
  setTrainingConsoleModelId: (id) => set({ trainingConsoleModelId: id }),

  // Pipeline Universe
  pipelineUniverse: loadPipelineUniverse(),
  setPipelineUniverse: (symbols) => {
    savePipelineUniverse(symbols);
    set({ pipelineUniverse: symbols });
  },
  activePipelineJobId: loadActivePipelineJobId(),
  setActivePipelineJobId: (id) => {
    saveActivePipelineJobId(id);
    set({ activePipelineJobId: id });
  },

  // Stock Selector working draft
  stockSelectorSelection: loadStockSelectorDraft(),
  setStockSelectorSelection: (next) =>
    set((state) => {
      const nextVal = typeof next === 'function'
        ? next(state.stockSelectorSelection)
        : next;
      saveStockSelectorDraft(nextVal);
      return { stockSelectorSelection: nextVal };
    }),
  resetStockSelectorSelection: () => {
    try { localStorage.removeItem(STOCK_SELECTOR_DRAFT_KEY); } catch { /* ignore */ }
    set({ stockSelectorSelection: createEmptyStockSelectorSelection() });
  },

  isZerodhaAuthenticated: false,
  setIsZerodhaAuthenticated: (isZerodhaAuthenticated) => set({ isZerodhaAuthenticated }),
  isAuthRefreshing: false,

  retrainAlert: false,
  retrainDaysSince: null,
  retrainHasModels: true,
  setRetrainAlert: (retrainAlert) => set({ retrainAlert }),
  checkRetrainStatus: async () => {
    try {
      const { getRetrainStatus } = await import('../services/api');
      const res = await getRetrainStatus();
      if (res.data.needs_retrain) {
        set({
          retrainAlert: true,
          retrainDaysSince: res.data.days_since_retrain,
          retrainHasModels: res.data.has_models ?? true,
        });
      } else {
        set({ retrainAlert: false, retrainDaysSince: null });
      }
    } catch {
      // non-critical — ignore
    }
  },

  isInitialized: true,
  error: null,
  initialize: async () => {
    // health check + auth check — never blocks app render
    try {
      const { getHealth, getAuthStatus, getLoginUrl } = await import('../services/api');
      
      // 1. Backend Health Check
      const healthRes = await getHealth();
      if (healthRes.status !== 200) throw new Error(`Health check returned ${healthRes.status}`);
      set({ error: null });

      // 2. Zerodha Auth Check + Silent Refresh
      const authRes = await getAuthStatus();
      const authenticated = authRes.data.authenticated;
      set({ isZerodhaAuthenticated: authenticated });

      // 3. Check if models are stale and a retrain is recommended
      try {
        const { getRetrainStatus } = await import('../services/api');
        const retrainRes = await getRetrainStatus();
        if (retrainRes.data.needs_retrain) {
          set({ 
            retrainAlert: true,
            retrainDaysSince: retrainRes.data.days_since_retrain,
            retrainHasModels: retrainRes.data.has_models ?? true,
          });
        }
      } catch {
        // non-critical — ignore
      }

      if (!authenticated && !window.location.pathname.startsWith('/token')) {
        // Attempt Silent Refresh (skip if we're on the OAuth callback route)
        const REFRESH_KEY = 'aitrade-last-refresh';
        const now = Date.now();
        const lastRefresh = sessionStorage.getItem(REFRESH_KEY);
        
        // Prevent infinite loops: only try auto-refreshing once every 5 minutes in this tab
        if (!lastRefresh || (now - parseInt(lastRefresh)) > 300000) {
          sessionStorage.setItem(REFRESH_KEY, now.toString());
          console.log('Zerodha session expired. Attempting silent refresh...');
          
          try {
            const loginRes = await getLoginUrl();
            const loginUrl = loginRes.data.login_url;
            if (loginUrl) {
              // Append state=currentURL so backend knows to redirect back
              const redirectUrl = `${loginUrl}&state=${encodeURIComponent(window.location.href)}`;
              set({ isAuthRefreshing: true });
              window.location.assign(redirectUrl);
            }
          } catch (err) {
            console.error('Failed to trigger auto-refresh:', err);
          }
        }
      }
    } catch (err) {
      console.error('Initialization failed:', err);
      set({ error: 'Backend is not reachable or session could not be refreshed.' });
    }
  },
}));
