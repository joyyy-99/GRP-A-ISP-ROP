import { createContext, useContext } from 'react';
import type { Session, User } from '@supabase/supabase-js';

export interface UserProfile {
  fullName: string | null;
  role: string | null;
}

export interface AuthContextValue {
  user: User | null;
  session: Session | null;
  profile: UserProfile | null;
  isLoading: boolean;
  signIn: (opts: { email: string; password: string }) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signUp: (opts: { email: string; password: string; fullName?: string; role?: string }) => Promise<{
    requiresEmailConfirmation: boolean;
  }>;
  signOut: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
