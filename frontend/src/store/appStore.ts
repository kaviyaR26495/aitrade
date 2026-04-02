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
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
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
}));
