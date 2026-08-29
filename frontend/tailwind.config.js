/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4A90E2',
          light: '#6AA5EB',
          dark: '#2E75C8',
        },
        background: '#F0F4F8',
        success: {
          DEFAULT: '#4CAF50',
          light: '#6BCF6F',
        },
        warning: {
          DEFAULT: '#FFC107',
          light: '#FFD54F',
        },
        danger: {
          DEFAULT: '#F44336',
          light: '#FF6F61',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          muted: '#E2E8F0',
        },
        slate: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5F6',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Helvetica Neue"',
          'Arial',
          '"Noto Sans"',
          'sans-serif',
        ],
      },
      boxShadow: {
        card: '0 10px 30px rgba(74, 144, 226, 0.12)',
        subtle: '0 4px 12px rgba(15, 23, 42, 0.08)',
      },
      borderRadius: {
        xl: '1rem',
      },
      transitionDuration: {
        250: '250ms',
      },
    },
  },
  plugins: [],
}
