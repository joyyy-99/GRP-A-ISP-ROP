import { NavLink } from 'react-router-dom';
import { LayoutDashboard, UploadCloud, BarChart3, History, X } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

const NAV_ITEMS = [
  { name: 'Upload & Analyze', to: '/', icon: UploadCloud },
  { name: 'Insights Dashboard', to: '/dashboard', icon: LayoutDashboard },
  { name: 'Metrics', to: '/metrics', icon: BarChart3 },
  { name: 'History', to: '/history', icon: History },
];

const baseLinkClasses =
  'group flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary';

const inactiveClasses =
  'text-slate-500 hover:bg-background hover:text-primary-dark';

const activeClasses =
  'bg-primary/10 text-primary-dark border-r-4 border-primary-dark shadow-subtle';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  const { profile, user } = useAuth();
  const displayName = profile?.fullName ?? user?.email ?? 'User';
  const displayRole = profile?.role ?? null;

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-50 flex w-64 transform flex-col bg-surface shadow-lg transition-transform duration-300 lg:translate-x-0 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}
    >
      <div className="border-b border-slate-100 px-6 py-6">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <span className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-primary-dark">
              ROP Detection
            </span>
            <h1 className="text-2xl font-bold text-slate-900">
              AI-Powered Diagnosis
            </h1>
            <p className="text-xs text-slate-500">
              Precision screening for retinopathy of prematurity
            </p>
          </div>
          <button
            type="button"
            className="mt-1 inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition hover:bg-background hover:text-primary lg:hidden"
            aria-label="Close navigation"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-4 py-6">
        {NAV_ITEMS.map(({ name, to, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            onClick={onClose}
            className={({ isActive }) =>
              [
                baseLinkClasses,
                isActive ? activeClasses : inactiveClasses,
              ].join(' ')
            }
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-background text-primary-dark transition group-hover:scale-105">
              <Icon className="h-5 w-5" strokeWidth={1.75} />
            </span>
            <span>{name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-slate-100 px-6 py-6">
        <div className="rounded-xl bg-background px-4 py-3 shadow-subtle">
          <p className="text-sm font-semibold text-slate-900">{displayName}</p>
          {displayRole ? (
            <p className="text-xs text-slate-500 capitalize">{displayRole}</p>
          ) : null}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
