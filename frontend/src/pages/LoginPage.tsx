import { useState, type FormEvent } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import type { Location } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

type AuthMode = 'signin' | 'signup';

const LoginPage = () => {
  const { signIn, signInWithGoogle, signUp, user, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation() as { state?: { from?: Location } };

  const [mode, setMode] = useState<AuthMode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const from = location.state?.from?.pathname ?? '/';

  if (user && !isLoading) {
    return <Navigate to={from} replace />;
  }

  const toggleMode = () => {
    setMode((prev) => (prev === 'signin' ? 'signup' : 'signin'));
    setError(null);
    setInfo(null);
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setInfo(null);
    setIsSubmitting(true);
    try {
      await signInWithGoogle();
    } catch (err) {
      setIsSubmitting(false);
      setError(err instanceof Error ? err.message : 'Google sign-in failed.');
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setInfo(null);
    setIsSubmitting(true);

    try {
      if (mode === 'signin') {
        await signIn({ email, password });
        navigate(from, { replace: true });
      } else {
        const result = await signUp({ email, password, fullName, role });
        if (result.requiresEmailConfirmation) {
          setInfo(
            'Account created! Please confirm your email address before signing in.'
          );
          setMode('signin');
        } else {
          navigate(from, { replace: true });
        }
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : `Unable to ${mode === 'signin' ? 'sign in' : 'sign up'}.`
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-md rounded-3xl bg-white p-10 shadow-subtle">
        <h1 className="text-3xl font-semibold text-slate-900">
          {mode === 'signin' ? 'Sign in' : 'Create your account'}
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          {mode === 'signin'
            ? 'Use your clinician credentials to access the ROP detection dashboard.'
            : 'Register to start uploading retinal images and reviewing AI predictions.'}
        </p>

        <form onSubmit={handleSubmit} className="mt-8 space-y-4">
          {mode === 'signup' ? (
            <>
              <div className="space-y-2">
                <label htmlFor="fullName" className="text-sm font-medium text-slate-600">
                  Full name
                </label>
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Dr. Jane Doe"
                  required
                  className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-3 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="role" className="text-sm font-medium text-slate-600">
                  Role
                </label>
                <select
                  id="role"
                  value={role}
                  onChange={(event) => setRole(event.target.value)}
                  required
                  className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-3 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                >
                  <option value="">Select your role</option>
                  <option value="clinician">Clinician</option>
                  <option value="researcher">Researcher</option>
                  <option value="admin">Administrator</option>
                </select>
              </div>
            </>
          ) : null}

          <div className="space-y-2">
            <label htmlFor="email" className="text-sm font-medium text-slate-600">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-3 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="password" className="text-sm font-medium text-slate-600">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-3 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>
          {error ? (
            <div className="rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
              {error}
            </div>
          ) : null}
          {info ? (
            <div className="rounded-2xl border border-primary/20 bg-primary/10 px-4 py-3 text-sm text-primary">
              {info}
            </div>
          ) : null}
          <button
            type="submit"
            disabled={isSubmitting || isLoading}
            className="w-full rounded-full bg-primary py-3 text-sm font-semibold text-white shadow-card transition duration-200 hover:bg-primary-dark focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSubmitting
              ? mode === 'signin'
                ? 'Signing in...'
                : 'Creating account...'
              : mode === 'signin'
              ? 'Sign in'
              : 'Sign up'}
          </button>
        </form>

        <div className="mt-6">
          <div className="flex items-center gap-4">
            <div className="h-px flex-1 bg-slate-200" />
            <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
              or
            </span>
            <div className="h-px flex-1 bg-slate-200" />
          </div>
          <button
            type="button"
            onClick={handleGoogleSignIn}
            disabled={isSubmitting || isLoading}
            className="mt-6 flex w-full items-center justify-center gap-3 rounded-full border border-slate-200 bg-white py-3 text-sm font-semibold text-slate-700 transition duration-200 hover:border-primary hover:text-primary focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-60"
          >
            <span aria-hidden="true" className="text-base font-bold text-primary">
              G
            </span>
            Continue with Google
          </button>
        </div>

        <div className="mt-6 text-center text-sm text-slate-500">
          {mode === 'signin' ? (
            <>
              Need an account?{' '}
              <button
                type="button"
                onClick={toggleMode}
                className="font-semibold text-primary hover:underline focus-visible:outline-none"
              >
                Sign up
              </button>
            </>
          ) : (
            <>
              Already registered?{' '}
              <button
                type="button"
                onClick={toggleMode}
                className="font-semibold text-primary hover:underline focus-visible:outline-none"
              >
                Back to sign in
              </button>
            </>
          )}
        </div>

        <div className="mt-4 text-center text-xs text-slate-400">
          <Link to="https://supabase.com/docs/guides/auth" target="_blank" rel="noreferrer">
            Help with confirmation emails
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
