import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Note: StrictMode removed to prevent double-mounting issues during development
// which can cause the app to reload when switching browser tabs
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <App />
);
