import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import MainLayout from './components/Layout/MainLayout';
import ProtectedRoute from './components/Auth/ProtectedRoute';
import { AuthProvider } from './contexts/AuthContext';

// Lazy load pages for faster initial load
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const MetricsPage = lazy(() => import('./pages/MetricsPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));

// Minimal loading fallback
const PageLoader = () => (
  <div className="flex h-screen items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
  </div>
);

const ProtectedAppLayout = () => (
  <MainLayout>
    <Outlet />
  </MainLayout>
);

const App = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<ProtectedRoute />}>
              <Route element={<ProtectedAppLayout />}>
                <Route path="/" element={<UploadPage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/upload" element={<Navigate to="/" replace />} />
                <Route path="/metrics" element={<MetricsPage />} />
                <Route path="/history" element={<HistoryPage />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
