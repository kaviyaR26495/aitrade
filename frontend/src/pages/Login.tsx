import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { Eye, EyeOff } from 'lucide-react';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  
  const navigate = useNavigate();
  const { addNotification } = useAppStore();

  useEffect(() => {
    // Clear old data to start fresh as requested by user
    if (!localStorage.getItem('aitrade-v2-migrated')) {
      localStorage.removeItem('aitrade-users');
      localStorage.removeItem('aitrade-admins');
      localStorage.setItem('aitrade-v2-migrated', 'true');
    }
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }

    const storedUsersStr = localStorage.getItem('aitrade-users');
    let storedUsers = storedUsersStr ? JSON.parse(storedUsersStr) : [];
    
    // FIRST TIME SETUP: If no users exist, the first person to login becomes Super Admin
    if (storedUsers.length === 0) {
      const superAdminUser = {
        username,
        password,
        role: 'super_admin',
        permissions: ['*']
      };
      storedUsers = [superAdminUser];
      localStorage.setItem('aitrade-users', JSON.stringify(storedUsers));
      
      localStorage.setItem('aitrade-logged-in', 'true');
      localStorage.setItem('aitrade-current-user', username);
      localStorage.setItem('aitrade-current-role', 'super_admin');
      localStorage.setItem('aitrade-current-permissions', JSON.stringify(['*']));
      
      addNotification({ type: 'success', message: `Welcome! You are now the Super Admin.` });
      navigate('/dashboard');
      return;
    }

    // NORMAL LOGIN
    const userExists = storedUsers.find((u: any) => u.username === username && u.password === password);
    if (userExists) {
      localStorage.setItem('aitrade-logged-in', 'true');
      localStorage.setItem('aitrade-current-user', username);
      localStorage.setItem('aitrade-current-role', userExists.role || 'user');
      localStorage.setItem('aitrade-current-permissions', JSON.stringify(userExists.permissions || []));
      addNotification({ type: 'success', message: `Welcome ${username}!` });
      navigate('/dashboard');
    } else {
      setError('Invalid credentials');
      addNotification({ type: 'error', message: 'Invalid credentials' });
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-[var(--bg)] relative overflow-hidden">
      
      {/* Decorative Background Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-[var(--primary)]/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-blue-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="w-full max-w-md p-8 bg-[var(--bg-elevated)] rounded-xl border border-[var(--border)] shadow-[var(--shadow-lg)] relative z-10">
        
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-end)] flex items-center justify-center shadow-lg">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
                <polyline points="16 7 22 7 22 13"></polyline>
              </svg>
            </div>
          </div>
          <h2 className="text-2xl font-bold text-[var(--text)] mb-2">
            Nueroalgo.in Portal
          </h2>
          <p className="text-sm text-[var(--text-muted)]">
            Enter your credentials to access the trading platform
          </p>
        </div>

        <form onSubmit={handleLogin} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1.5">Username / Phone Number</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2.5 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary)] transition-all"
              placeholder="Enter username or phone number"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1.5">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2.5 pr-12 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary)] transition-all"
                placeholder="Enter password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>
          
          {error && (
            <div className="bg-red-500/10 text-red-400 text-sm font-medium p-3 rounded-lg border border-red-500/20 text-center animate-[fade-in_0.2s_ease-out]">
              {error}
            </div>
          )}
          
          <div className="pt-2">
            <button
              type="submit"
              className="w-full text-white font-medium py-2.5 rounded-lg transition-all shadow-md bg-blue-600 hover:bg-blue-500 shadow-blue-500/20"
            >
              Sign In
            </button>
          </div>
        </form>

      </div>
    </div>
  );
}
