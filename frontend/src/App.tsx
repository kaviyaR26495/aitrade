import { Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import Layout from './components/Layout';
import { ErrorBoundary } from './components/ErrorBoundary';
import Dashboard from './pages/Dashboard';
import DataManager from './pages/DataManager';
import RegimeAnalysis from './pages/RegimeAnalysis';
import ModelStudio from './pages/ModelStudio';
import PatternLab from './pages/PatternLab';
import Backtest from './pages/Backtest';
import LiveTrading from './pages/LiveTrading';
import Portfolio from './pages/Portfolio';
import Settings from './pages/Settings';
import ThemeShowcase from './components/ThemeShowcase';
import StockSelector from './pages/StockSelector';
import AuthCallback from './pages/AuthCallback';
import BackfillPredictions from './pages/BackfillPredictions';
import Login from './pages/Login';
import AdminDashboard from './pages/AdminDashboard';
import { useAppStore } from './store/appStore';

export default function App() {
  const { initialize } = useAppStore();

  useEffect(() => {
    // Background health check — never blocks rendering
    initialize();
  }, []);

  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/token" element={<AuthCallback />} />
        <Route path="/" element={<Login />} />
        <Route element={<Layout />}>
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/data" element={<DataManager />} />
          <Route path="/stocks" element={<StockSelector />} />
          <Route path="/regime" element={<RegimeAnalysis />} />
          <Route path="/models" element={<ModelStudio />} />
          <Route path="/patterns" element={<PatternLab />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/backfill" element={<BackfillPredictions />} />
          <Route path="/trading" element={<LiveTrading />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/theme" element={<ThemeShowcase />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}
