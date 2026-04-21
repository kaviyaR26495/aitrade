import { Outlet, useLocation } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import Sidebar from './Sidebar';
import ChatBot from './ChatBot';
import ApiConsole from './ApiConsole';
import TrainingConsole from './TrainingConsole';
import { useAppStore } from '../store/appStore';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info, RefreshCw } from 'lucide-react';

const NOTIFICATION_STYLES = {
  success: { bg: 'bg-emerald-500/10 border-emerald-500/30', icon: CheckCircle, iconColor: 'text-emerald-400' },
  error:   { bg: 'bg-rose-500/10 border-rose-500/30', icon: AlertCircle, iconColor: 'text-rose-400' },
  warning: { bg: 'bg-amber-500/10 border-amber-500/30', icon: AlertTriangle, iconColor: 'text-amber-400' },
  info:    { bg: 'bg-indigo-500/10 border-indigo-500/30', icon: Info, iconColor: 'text-indigo-400' },
};

export default function Layout() {
  const { notifications, removeNotification, retrainAlert, retrainDaysSince, retrainHasModels, setRetrainAlert, addNotification, checkRetrainStatus } = useAppStore();
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  const [retrainLoading, setRetrainLoading] = useState(false);

  // Re-check retrain status on mount so that models trained after the initial
  // app load are detected and the banner is dismissed automatically.
  useEffect(() => {
    checkRetrainStatus();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Scroll to top on route change
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [location.pathname]);

  // Auto-dismiss toasts after 5 seconds
  useEffect(() => {
    if (notifications.length === 0) return;
    const latest = notifications[notifications.length - 1];
    const timer = window.setTimeout(() => removeNotification(latest.id), 5000);
    return () => clearTimeout(timer);
  }, [notifications, removeNotification]);

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg)]">
      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Retrain alert banner — shown when models are stale (> 30 days) */}
        {retrainAlert && (
          <div className="flex items-center gap-3 px-4 py-2.5 bg-amber-500/10 border-b border-amber-500/30 text-[13px] text-[var(--text)] shrink-0">
            <AlertTriangle size={15} className="flex-shrink-0 text-amber-400" />
            <span className="flex-1">
              {retrainDaysSince === null
                ? (retrainHasModels
                    ? "Model ensemble not yet built — individual models exist but have not been assembled. Run auto-retrain to configure the ensemble."
                    : "No trained models found — run auto-retrain to train and assemble the trading engine.")
                : `Model ensemble may be stale — it has been ${retrainDaysSince} day${retrainDaysSince === 1 ? '' : 's'} since the last auto-retrain.`}
              <span className="ml-1 text-[var(--text-muted)]">Training on fresh patterns improves signal quality as market regimes evolve.</span>
            </span>
            <button
              disabled={retrainLoading}
              onClick={async () => {
                setRetrainLoading(true);
                try {
                  const { triggerAutoRetrain } = await import('../services/api');
                  await triggerAutoRetrain(2);
                  setRetrainAlert(false);
                  addNotification({ type: 'success', message: 'Auto-retrain started in background. New models will appear in Model Studio when complete.' });
                } catch {
                  addNotification({ type: 'error', message: 'Failed to start auto-retrain — check server logs.' });
                } finally {
                  setRetrainLoading(false);
                }
              }}
              className="flex items-center gap-1.5 px-3 py-1 rounded border border-amber-500/40 bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 transition-colors disabled:opacity-50"
            >
              <RefreshCw size={12} className={retrainLoading ? 'animate-spin' : ''} />
              {retrainLoading ? 'Starting…' : 'Start Auto-retrain'}
            </button>
            <button
              onClick={() => setRetrainAlert(false)}
              className="flex-shrink-0 p-0.5 rounded hover:bg-white/5 text-[var(--text-muted)]"
              aria-label="Dismiss retrain alert"
            >
              <X size={13} />
            </button>
          </div>
        )}

        <main ref={mainRef} className="flex-1 overflow-y-auto">
          <div className="max-w-[1400px] mx-auto px-6 py-7 lg:px-8 lg:py-8 page-enter relative z-[1]">
            <Outlet />
          </div>
        </main>

        <ApiConsole />
        <TrainingConsole />
      </div>

      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2.5 w-[340px] pointer-events-none">
        {notifications.map((n) => {
          const style = NOTIFICATION_STYLES[n.type as keyof typeof NOTIFICATION_STYLES] ?? NOTIFICATION_STYLES.info;
          const IconComp = style.icon;
          return (
            <div
              key={n.id}
              className={`
                pointer-events-auto flex items-start gap-2.5 px-3.5 py-2.5
                rounded-[var(--radius)] border backdrop-blur-md
                shadow-[var(--shadow-lg)] text-[13px]
                animate-[fade-in_0.2s_ease-out] ${style.bg}
              `}
            >
              <IconComp size={15} className={`flex-shrink-0 mt-0.5 ${style.iconColor}`} />
              <span className="flex-1 text-[var(--text)] leading-relaxed">{n.message}</span>
              <button onClick={() => removeNotification(n.id)} className="flex-shrink-0 p-0.5 rounded hover:bg-white/5 text-[var(--text-muted)]">
                <X size={13} />
              </button>
            </div>
          );
        })}
      </div>

      <ChatBot />
    </div>
  );
}