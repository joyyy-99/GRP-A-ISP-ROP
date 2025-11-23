import { Bell, Info, LogOut, Menu, Settings, UserRound } from 'lucide-react';
import type { ReactNode } from 'react';
import { useSidebar } from './MainLayout';
import { useAuth } from '../../hooks/useAuth';

interface HeaderProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

const iconButtonClasses =
  'relative inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition duration-200 hover:bg-primary/10 hover:text-primary-dark focus-visible:outline-primary';

const Header = ({ title, subtitle, actions }: HeaderProps) => {
  const sidebar = useSidebar();
  const { user, profile, signOut } = useAuth();

  const displayName = profile?.fullName ?? user?.email ?? 'User';
  const displayRole = profile?.role ?? null;

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Sign out error:', error);
    }
  };

  return (
    <header className="sticky top-0 z-30 bg-white/95 px-6 py-5 shadow-subtle backdrop-blur supports-[backdrop-filter]:backdrop-blur">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-1 items-center gap-4">
          {sidebar ? (
            <button
              type="button"
              className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition duration-200 hover:bg-primary/10 hover:text-primary-dark focus-visible:outline-primary lg:hidden"
              aria-label="Toggle navigation"
              onClick={sidebar.toggleSidebar}
            >
              <Menu className="h-5 w-5" />
            </button>
          ) : null}
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-slate-900 lg:text-3xl">
              {title}
            </h2>
            {subtitle ? (
              <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {actions}
          <button
            type="button"
            className={iconButtonClasses}
            aria-label="Notifications"
          >
            <Bell className="h-5 w-5" strokeWidth={1.75} />
            <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-danger" />
          </button>
          <button
            type="button"
            className={iconButtonClasses}
            aria-label="Information"
          >
            <Info className="h-5 w-5" strokeWidth={1.75} />
          </button>
          <button
            type="button"
            className={iconButtonClasses}
            aria-label="Settings"
          >
            <Settings className="h-5 w-5" strokeWidth={1.75} />
          </button>
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-background px-3 py-1 text-sm text-slate-600">
            <UserRound className="h-4 w-4 text-primary" />
            <span className="max-w-[10rem] truncate font-medium">
              {displayName}
            </span>
            {displayRole ? (
              <span className="hidden capitalize text-xs text-slate-400 sm:inline">
                {displayRole}
              </span>
            ) : null}
            <button
              type="button"
              onClick={handleSignOut}
              className="ml-2 inline-flex items-center gap-1 rounded-full bg-white px-3 py-1 text-xs font-semibold text-danger transition duration-200 hover:bg-danger/10 focus-visible:outline-danger"
            >
              <LogOut className="h-3.5 w-3.5" />
              Sign out
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
