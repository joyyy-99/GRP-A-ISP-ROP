import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import Sidebar from './Sidebar';

interface MainLayoutProps {
  children: ReactNode;
}

interface SidebarContextValue {
  isOpen: boolean;
  toggleSidebar: () => void;
  closeSidebar: () => void;
}

const SidebarContext = createContext<SidebarContextValue | undefined>(undefined);

export const useSidebar = () => {
  return useContext(SidebarContext);
};

const MainLayout = ({ children }: MainLayoutProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const value = useMemo<SidebarContextValue>(
    () => ({
      isOpen,
      toggleSidebar: () => setIsOpen((prev) => !prev),
      closeSidebar: () => setIsOpen(false),
    }),
    [isOpen]
  );

  return (
    <SidebarContext.Provider value={value}>
      <div className="relative flex min-h-screen bg-background text-slate-900">
        {isOpen ? (
          <div
            className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-sm lg:hidden"
            onClick={() => setIsOpen(false)}
            aria-hidden="true"
          />
        ) : null}
        <Sidebar isOpen={isOpen} onClose={() => setIsOpen(false)} />
        <main className="flex-1 overflow-y-auto pb-8 lg:ml-64">{children}</main>
      </div>
    </SidebarContext.Provider>
  );
};

export default MainLayout;
