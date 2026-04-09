import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Loader2, CheckCircle2, XCircle, ArrowRight } from 'lucide-react';
import { zerodhaCallback } from '../services/api';
import { useAppStore } from '../store/appStore';

export default function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { addNotification, setIsZerodhaAuthenticated } = useAppStore();
  
  const [status, setStatus] = useState<'pending' | 'success' | 'error'>('pending');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const requestToken = searchParams.get('request_token');
  const state = searchParams.get('state');

  useEffect(() => {
    async function handleCallback() {
      if (!requestToken) {
        setStatus('error');
        setErrorMsg('No request token found in URL.');
        return;
      }

      try {
        await zerodhaCallback(requestToken, state || undefined);
        setStatus('success');
        setIsZerodhaAuthenticated(true);
        addNotification({
          type: 'success',
          message: 'Zerodha connected successfully!',
        });

        // Small delay to show success state before redirecting
        setTimeout(() => {
          if (state) {
            window.location.href = state;
          } else {
            navigate('/');
          }
        }, 1500);
      } catch (err: any) {
        console.error('Auth callback failed:', err);
        setStatus('error');
        setErrorMsg(err.response?.data?.detail || 'Authentication failed. Please try again.');
        addNotification({
          type: 'error',
          message: 'Failed to connect to Zerodha.',
        });
      }
    }

    handleCallback();
  }, [requestToken, state, navigate, addNotification, setIsZerodhaAuthenticated]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary/20 rounded-full blur-[128px] pointer-events-none" />
      
      <div className="w-full max-w-md relative">
        <div className="p-8 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl space-y-8 text-center">
          
          {status === 'pending' && (
            <div className="space-y-6">
              <div className="relative">
                <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse" />
                <Loader2 className="w-16 h-16 text-primary animate-spin mx-auto relative" />
              </div>
              <div className="space-y-2">
                <h1 className="text-2xl font-bold tracking-tight text-white">Authenticating...</h1>
                <p className="text-white/60">Finalizing your Zerodha connection.</p>
              </div>
            </div>
          )}

          {status === 'success' && (
            <div className="space-y-6 animate-in fade-in zoom-in duration-500">
              <div className="relative">
                <div className="absolute inset-0 bg-emerald-500/20 rounded-full blur-xl" />
                <CheckCircle2 className="w-16 h-16 text-emerald-500 mx-auto relative" />
              </div>
              <div className="space-y-2">
                <h1 className="text-2xl font-bold tracking-tight text-white">Success!</h1>
                <p className="text-white/60">Your Zerodha account is now connected.</p>
              </div>
              <div className="flex items-center justify-center gap-2 text-sm text-primary animate-pulse">
                Redirecting you back <ArrowRight className="w-4 h-4" />
              </div>
            </div>
          )}

          {status === 'error' && (
            <div className="space-y-6 animate-in fade-in zoom-in duration-500">
              <div className="relative">
                <div className="absolute inset-0 bg-red-500/20 rounded-full blur-xl" />
                <XCircle className="w-16 h-16 text-red-500 mx-auto relative" />
              </div>
              <div className="space-y-2">
                <h1 className="text-2xl font-bold tracking-tight text-white">Authentication Failed</h1>
                <p className="text-red-400/80">{errorMsg}</p>
              </div>
              <button
                onClick={() => navigate('/settings')}
                className="w-full py-3 px-4 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium transition-all"
              >
                Go Back to Settings
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
