import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Database, Layers, BrainCircuit, Search,
  BarChart3, Crosshair, Briefcase, Settings, ChevronLeft, ChevronRight,
  TrendingUp, Palette, ListChecks, History, LogOut, LogIn, ChevronUp, ChevronDown, ShieldCheck, User
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { usePipelineStatus } from '../hooks/useApi';

const PIPELINE_STAGE_LABELS = [
  'Data Sync', 'CQL Pre-train', 'BC Warmup',
  'PPO Fine-tune', 'Meta-Model Train', 'Backtest', 'Ready',
];

function getPipelineLabel(status?: { status: string; current_stage: number } | null): string {
  if (!status) return 'System Idle — Ready';
  if (status.status === 'queued') return 'Queued…';
  if (status.status === 'running') {
    const label = PIPELINE_STAGE_LABELS[status.current_stage] ?? `Stage ${status.current_stage + 1}`;
    return `Training: ${label}…`;
  }
  if (status.status === 'completed') return 'Training Complete';
  if (status.status === 'failed') return 'Pipeline Failed';
  return 'System Idle — Ready';
}

const NAV_SECTIONS = [
  {
    label: 'Overview',
    items: [
      { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard', id: 'dashboard' },
      { to: '/data', icon: Database, label: 'Data Manager', id: 'data' },
      { to: '/stocks', icon: ListChecks, label: 'Stock Selector', id: 'stocks' },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { to: '/regime', icon: Layers, label: 'Regime Analysis', id: 'regime' },
      { to: '/models', icon: BrainCircuit, label: 'Model Studio', id: 'models' },
      { to: '/patterns', icon: Search, label: 'Pattern Lab', id: 'patterns' },
    ],
  },
  {
    label: 'Execution',
    items: [
      { to: '/backtest', icon: BarChart3, label: 'Backtest', id: 'backtest' },
      { to: '/backfill', icon: History, label: 'Backfill Predictions', id: 'backtest' },
      { to: '/trading', icon: Crosshair, label: 'Live Trading', id: 'trading' },
      { to: '/portfolio', icon: Briefcase, label: 'Portfolio', id: 'portfolio' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/settings', icon: Settings, label: 'Settings', id: 'settings' },
      { to: '/theme', icon: Palette, label: 'Appearance', id: 'theme' },
    ],
  },
];

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar, activePipelineJobId } = useAppStore();
  const [rolesOpen, setRolesOpen] = useState(false);
  const { data: pipelineStatus } = usePipelineStatus(activePipelineJobId);

  const isActive = pipelineStatus?.status === 'running' || pipelineStatus?.status === 'queued';
  const isFailed = pipelineStatus?.status === 'failed';
  const dotClass = isActive
    ? 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.5)] animate-pulse'
    : isFailed
      ? 'bg-red-400 shadow-[0_0_6px_rgba(239,68,68,0.5)]'
      : 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.5)]';
  const pipelineLabel = getPipelineLabel(pipelineStatus ?? null);

  const userRole = localStorage.getItem('aitrade-current-role');
  const userPermissionsStr = localStorage.getItem('aitrade-current-permissions');
  const userPermissions = userPermissionsStr ? JSON.parse(userPermissionsStr) : [];
  const hasFullAccess = userRole === 'super_admin' || userPermissions.includes('*');

  const filteredNavSections = NAV_SECTIONS.map(section => ({
    ...section,
    items: section.items.filter(item => hasFullAccess || userPermissions.includes(item.id))
  })).filter(section => section.items.length > 0);

  return (
    <aside
      className={`
        flex flex-col h-screen bg-[var(--sidebar-bg)]
        border-r border-[var(--sidebar-border)]
        transition-[width] duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] flex-shrink-0
        ${sidebarOpen ? 'w-[var(--sidebar-w)]' : 'w-[var(--sidebar-collapsed)]'}
      `}
    >
      {/* Brand */}
      <div className="flex items-center justify-between px-5 h-16 border-b border-[var(--sidebar-border)] flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex-shrink-0 w-9 h-9 rounded-lg bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-end)] flex items-center justify-center shadow-sm">
            <TrendingUp size={17} className="text-white" />
          </div>
          {sidebarOpen && (
            <span className="text-base font-bold gradient-text truncate leading-none">Nueroalgo.in</span>
          )}
        </div>
        <button
          onClick={toggleSidebar}
          className="flex-shrink-0 p-1.5 rounded-md hover:bg-[var(--sidebar-item-hover)] text-[var(--text-dim)] hover:text-[var(--text-muted)] transition-colors"
        >
          {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-3">
        {filteredNavSections.map((section) => (
          <div key={section.label} className="mb-1">
            {sidebarOpen && (
              <div className="flex items-center gap-2 px-2 pt-4 pb-1.5">
                <span className="w-1 h-1 rounded-full bg-[var(--border)]" />
                <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-[var(--text-dim)]">
                  {section.label}
                </span>
              </div>
            )}
            {!sidebarOpen && section.label !== 'Overview' && (
              <div className="mx-2 my-3 h-px bg-[var(--border-light)]" />
            )}
            <div className="space-y-0.5">
              {section.items.map(({ to, icon: Icon, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) => `
                    flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                    text-[13px] font-medium transition-all duration-150 group relative
                    ${isActive
                      ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                      : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                    }
                  `}
                >
                  {({ isActive }) => (
                    <>
                      <Icon size={18} className={`flex-shrink-0 transition-opacity ${isActive ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                      {sidebarOpen && <span className="truncate">{label}</span>}
                      {!sidebarOpen && (
                        <div className="
                          absolute left-full ml-3 px-3 py-1.5 rounded-md
                          bg-[var(--bg-elevated)] border border-[var(--border)]
                          text-xs text-[var(--text)] whitespace-nowrap
                          opacity-0 pointer-events-none group-hover:opacity-100
                          transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50
                        ">
                          {label}
                        </div>
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}

                {/* User Management Section */}
        <div className="mb-1">
          {sidebarOpen && (
            <div className="flex items-center gap-2 px-2 pt-4 pb-1.5">
              <span className="w-1 h-1 rounded-full bg-[var(--border)]" />
              <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-[var(--text-dim)]">
                User Management
              </span>
            </div>
          )}
          {!sidebarOpen && (
            <div className="mx-2 my-3 h-px bg-[var(--border-light)]" />
          )}
          <div className="space-y-0.5">
          <div className="space-y-0.5">
            {/* Roles link - super_admin only */}
            {userRole === 'super_admin' && (
              <NavLink
                to="/admin?tab=roles"
                className={({ isActive }) => `
                  flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                  text-[13px] font-medium transition-all duration-150 group relative
                  ${(isActive && location.search.includes('tab=roles'))
                    ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                  }
                `}
              >
                <ShieldCheck size={18} className={`flex-shrink-0 transition-opacity ${location.search.includes('tab=roles') ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                {sidebarOpen && <span className="truncate">Roles</span>}
                {!sidebarOpen && (
                  <div className="absolute left-full ml-3 px-3 py-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border)] text-xs text-[var(--text)] whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50">
                    Roles
                  </div>
                )}
              </NavLink>
            )}

            {/* User link - super_admin or admin */}
            {(userRole === 'super_admin' || userRole === 'admin') && (
              <NavLink
                to="/admin?tab=user"
                className={({ isActive }) => `
                  flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                  text-[13px] font-medium transition-all duration-150 group relative
                  ${(isActive && location.search.includes('tab=user'))
                    ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                  }
                `}
              >
                <User size={18} className={`flex-shrink-0 transition-opacity ${location.search.includes('tab=user') ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                {sidebarOpen && <span className="truncate">{userRole === 'admin' ? 'Create User' : 'User'}</span>}
                {!sidebarOpen && (
                  <div className="absolute left-full ml-3 px-3 py-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border)] text-xs text-[var(--text)] whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50">
                    {userRole === 'admin' ? 'Create User' : 'User'}
                  </div>
                )}
              </NavLink>
            )}

            {/* My Profile link - available to everyone */}
            <NavLink
              to="/admin?tab=profile"
              className={({ isActive }) => `
                flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                text-[13px] font-medium transition-all duration-150 group relative
                ${(isActive && location.search.includes('tab=profile'))
                  ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                }
              `}
            >
              <ShieldCheck size={18} className={`flex-shrink-0 transition-opacity ${location.search.includes('tab=profile') ? '' : 'opacity-60 group-hover:opacity-100'}`} />
              {sidebarOpen && <span className="truncate">My Profile</span>}
              {!sidebarOpen && (
                <div className="absolute left-full ml-3 px-3 py-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border)] text-xs text-[var(--text)] whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50">
                  My Profile
                </div>
              )}
            </NavLink>
          </div>
          </div>
        </div>

      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-[var(--sidebar-border)] flex-shrink-0">
        {sidebarOpen ? (
          <div className="px-3 py-2.5 rounded-lg bg-[var(--primary-subtle)] border border-[var(--primary-glow)]">
            <div className="flex items-center gap-2 mb-0.5">
              <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
              <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Pipeline</span>
            </div>
            <div className="text-[11px] text-[var(--text-muted)] font-medium mb-3">{pipelineLabel}</div>
            
            <button
              onClick={() => {
                localStorage.removeItem('aitrade-logged-in');
                localStorage.removeItem('aitrade-current-user');
                window.location.href = '/';
              }}
              className="w-full flex items-center justify-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-500 rounded-md transition-colors text-xs font-semibold"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className={`w-2 h-2 rounded-full ${dotClass}`} title={pipelineLabel} />
            <button
              onClick={() => {
                localStorage.removeItem('aitrade-logged-in');
                localStorage.removeItem('aitrade-current-user');
                window.location.href = '/';
              }}
              className="text-red-400 hover:text-red-500 transition-colors"
              title="Sign Out"
            >
              <LogOut size={18} />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
