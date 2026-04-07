import React, { useState } from 'react';
import { Trash2, AlertTriangle, Database } from 'lucide-react';
import { Card, Button } from './ui';
import { useAppStore } from '../store/appStore';

const SystemMaintenance: React.FC = () => {
  const [loading, setLoading] = useState<string | null>(null);
  const { addNotification } = useAppStore();

  const handleCleanup = async (type: 'logs' | 'models') => {
    setLoading(type);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(`/api/config/cleanup/${type}`, {
        method: 'DELETE',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const data = await response.json();
      
      if (data.status === 'queued') {
        addNotification({ type: 'success', message: `Queued ${data.queued} ${type === 'logs' ? 'log file(s)' : 'model director(ies)'} for deletion.` });
      } else if (data.status === 'nothing to delete') {
        addNotification({ type: 'info', message: `Nothing to delete.` });
      } else {
        addNotification({ type: 'info', message: data.status });
      }
    } catch (error: any) {
      clearTimeout(timeoutId);
      const msg = error?.name === 'AbortError' ? 'Cleanup timed out.' : 'Failed to perform cleanup.';
      console.error('Cleanup failed:', error);
      addNotification({ type: 'error', message: msg });
    } finally {
      setLoading(null);
    }
  };

  return (
    <Card 
      title={<div className="flex items-center gap-2"><Database size={14} /> System Maintenance</div>}
      className="border-red-900/20"
    >
      <div className="space-y-6">
        <p className="text-sm text-[var(--text-muted)]">
          Remove temporary artifacts and logs to save disk space and keep the workspace clean.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-[var(--radius)] bg-[var(--bg-input)] border border-[var(--border-light)]">
            <h4 className="text-sm font-semibold mb-1 flex items-center gap-2">
              <Trash2 size={14} className="text-red-400" />
              Log Cleanup
            </h4>
            <p className="text-xs text-[var(--text-muted)] mb-4">
              Deletes all training session log files in the backend logs directory.
            </p>
            <Button 
                variant="secondary"
                className="w-full text-xs"
                onClick={() => handleCleanup('logs')}
                loading={loading === 'logs'}
            >
              Clear Training Logs
            </Button>
          </div>

          <div className="p-4 rounded-[var(--radius)] bg-[var(--bg-input)] border border-[var(--border-light)]">
            <h4 className="text-sm font-semibold mb-1 flex items-center gap-2">
              <AlertTriangle size={14} className="text-orange-400" />
              Prune Models
            </h4>
            <p className="text-xs text-[var(--text-muted)] mb-4">
              Keeps only the 3 most recent versions of each distilled model (KNN/LSTM).
            </p>
            <Button 
                variant="secondary"
                className="w-full text-xs"
                onClick={() => handleCleanup('models')}
                loading={loading === 'models'}
            >
              Prune Old Models
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default SystemMaintenance;
