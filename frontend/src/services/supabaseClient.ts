import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  // eslint-disable-next-line no-console
  console.warn(
    'Supabase environment variables are missing. Authentication and persistence features will not work until VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are provided.'
  );
}

export const supabase = createClient(
  supabaseUrl ?? '',
  supabaseAnonKey ?? '',
  {
    auth: {
      persistSession: true,
      // DISABLED: autoRefreshToken causes page reload on tab visibility change
      autoRefreshToken: false,
      detectSessionInUrl: true,
      storageKey: 'rop-auth-token',
      storage: typeof window !== 'undefined' ? window.localStorage : undefined,
    },
    // Disable realtime subscriptions which can cause reconnection issues
    realtime: {
      params: {
        eventsPerSecond: 0,
      },
    },
  }
);

export const isSupabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);
