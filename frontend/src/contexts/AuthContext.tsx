import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type { Session, User } from '@supabase/supabase-js';
import { supabase, isSupabaseConfigured } from '../services/supabaseClient';
import {
  AuthContext,
  type AuthContextValue,
  type UserProfile,
} from '../hooks/useAuth';

const PROFILE_TIMEOUT_MS = 15000;

const buildProfileFromUser = (user: User | null): UserProfile | null => {
  if (!user) {
    return null;
  }

  const metadata = user.user_metadata as Record<string, unknown> | undefined;
  const fallbackName =
    (metadata?.full_name as string | undefined) ??
    user.email ??
    null;
  const fallbackRole = (metadata?.role as string | undefined) ?? null;

  return {
    fullName: fallbackName,
    role: fallbackRole,
  };
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Track if initial auth check is complete to prevent re-renders
  const initializedRef = useRef(false);

  const loadProfile = async (nextUser: User | null) => {
    if (!isSupabaseConfigured || !nextUser) {
      setProfile(buildProfileFromUser(nextUser));
      return;
    }

    const fallback = buildProfileFromUser(nextUser);
    try {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;
      const profilePromise = supabase
        .from('profiles')
        .select('full_name, role')
        .eq('id', nextUser.id)
        .maybeSingle();

      const timeoutPromise = new Promise<never>((_, reject) =>
        {
          timeoutId = setTimeout(
            () => reject(new Error('Profile fetch timed out')),
            PROFILE_TIMEOUT_MS
          );
        }
      );

      const { data, error } = (await Promise.race([
        profilePromise,
        timeoutPromise,
      ])) as Awaited<typeof profilePromise>;

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      if (error) {
        console.warn('Unable to fetch profile:', error.message);
        setProfile(fallback);
        return;
      }

      setProfile({
        fullName: data?.full_name ?? fallback?.fullName ?? null,
        role: data?.role ?? fallback?.role ?? null,
      });
    } catch (err) {
      console.warn('Unexpected profile fetch error:', err);
      setProfile(fallback);
    }
  };

  useEffect(() => {
    // Only run initialization once
    if (initializedRef.current) {
      return;
    }

    if (!isSupabaseConfigured) {
      initializedRef.current = true;
      setIsLoading(false);
      return;
    }

    let mounted = true;

    const initialiseSession = async () => {
      try {
        const { data, error } = await supabase.auth.getSession();
        if (!mounted) {
          return;
        }

        if (error) {
          console.error('Supabase getSession error:', error.message);
        }

        const nextSession = data?.session ?? null;
        const nextUser = nextSession?.user ?? null;
        setSession(nextSession);
        setUser(nextUser);
        if (nextUser) {
          await loadProfile(nextUser);
        } else {
          setProfile(null);
        }
      } catch (err) {
        if (!mounted) {
          return;
        }
        console.error('Failed to initialise Supabase session:', err);
        setSession(null);
        setUser(null);
        setProfile(null);
      } finally {
        if (mounted) {
          initializedRef.current = true;
          setIsLoading(false);
        }
      }
    };

    void initialiseSession();

    // DISABLED: Auth state change listener causes re-renders on tab focus
    // Auth changes will only be detected on page refresh
    // This is acceptable for a demo/showcase scenario

    return () => {
      mounted = false;
    };
  }, []);

  const signIn = useCallback(async ({ email, password }: { email: string; password: string }) => {
    setIsLoading(true);
    try {
      const { data, error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) {
        throw new Error(error.message);
      }
      // Manually update state since we disabled the auth listener
      if (data.session) {
        setSession(data.session);
        setUser(data.user);
        await loadProfile(data.user);
      }
    } catch (err) {
      throw err instanceof Error ? err : new Error('Unable to sign in.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const signInWithGoogle = useCallback(async () => {
    setIsLoading(true);
    let didRedirect = false;
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo:
            typeof window !== 'undefined' ? window.location.origin : undefined,
        },
      });

      if (error) {
        throw new Error(error.message);
      }

      if (data?.url) {
        didRedirect = true;
        window.location.assign(data.url);
        return;
      }

      throw new Error('Unable to start Google sign-in. Please try again.');
    } catch (err) {
      throw err instanceof Error ? err : new Error('Google sign-in failed.');
    } finally {
      if (!didRedirect) {
        setIsLoading(false);
      }
    }
  }, []);

  const signUp = useCallback(async ({
    email,
    password,
    fullName,
    role,
  }: {
    email: string;
    password: string;
    fullName?: string;
    role?: string;
  }) => {
    setIsLoading(true);
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName ?? null,
            role: role ?? null,
          },
        },
      });

      if (error) {
        throw new Error(error.message);
      }

      if (data.user) {
        try {
          await supabase
            .from('profiles')
            .upsert(
              { id: data.user.id, full_name: fullName ?? null, role: role ?? null },
              { onConflict: 'id' }
            );
        } catch (profileError) {
          console.warn('Unable to upsert profile:', profileError);
        }
      }

      return {
        requiresEmailConfirmation: !data.session,
      };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const signOut = useCallback(async () => {
    setIsLoading(true);
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        throw new Error(error.message);
      }
      setProfile(null);
      setUser(null);
      setSession(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      session,
      profile,
      isLoading,
      signIn,
      signInWithGoogle,
      signUp,
      signOut,
    }),
    [isLoading, profile, session, signIn, signInWithGoogle, signOut, signUp, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
