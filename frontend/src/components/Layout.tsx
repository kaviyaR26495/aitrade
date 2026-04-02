import { Outlet, useLocation } from 'react-router-dom';
import { useEffect, useRef } from 'react';
import Sidebar from './Sidebar';
import ChatBot from './ChatBot';
import ApiConsole from './ApiConsole';
import TrainingConsole from './TrainingConsole';
import { useAppStore } from '../store/appStore';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';

const NOTIFICATION_STYLES = {
  success: { bg: 'bg-emerald-500/10 border-emerald-500/30', icon: CheckCircle, iconColor: 'text-emerald-400' },
  error:   { bg: 'bg-rose-500/10 border-rose-500/30', icon: AlertCircle, iconColor: 'text-rose-400' },
  warning: { bg: 'bg-amber-500/10 border-amber-500/30', icon: AlertTriangle, iconColor: 'text-amber-400' },
  info:    { bg: 'bg-indigo-500/10 border-indigo-500/30', icon: Info, iconColor: 'text-indigo-400' },
};

export default function Layout() {
  const { notifications, removeNotification } = useAppStore();
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);

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