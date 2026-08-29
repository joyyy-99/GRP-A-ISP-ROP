import { Link } from 'react-router-dom';
import type { PredictionResult } from '../../types';
import { CLASS_COLORS, CLASS_NAMES } from '../../types';

interface RecentPredictionsProps {
  items: PredictionResult[];
}

const formatRelativeTime = (isoDate: string) => {
  const target = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - target.getTime();

  const minutes = Math.floor(diffMs / (1000 * 60));
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
};

const RecentPredictions = ({ items }: RecentPredictionsProps) => {
  return (
    <section className="rounded-2xl bg-white p-6 shadow-subtle card-hover">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
            Latest Activity
          </p>
          <h3 className="mt-1 text-lg font-semibold text-slate-900">
            Recent Predictions
          </h3>
        </div>
        <Link
          to="/history"
          className="text-sm font-semibold text-primary hover:underline focus-visible:outline-primary"
        >
          View History →
        </Link>
      </div>

      {items.length === 0 ? (
        <div className="mt-6 rounded-xl border border-dashed border-slate-200 bg-background px-6 py-12 text-center text-sm text-slate-500">
          No predictions yet. Upload an image to see results here.
        </div>
      ) : (
        <div className="mt-6 space-y-3">
          {items.map((item) => {
            const confidenceEntry =
              item.confidences.find((c) => c.classId === item.predictedClass) ??
              item.confidences[0];
            const confidence = confidenceEntry
              ? (confidenceEntry.confidence * 100).toFixed(1)
              : '0.0';
            const badgeColor = CLASS_COLORS[item.predictedClass];

            return (
              <article
                key={item.id}
                className="flex items-center justify-between rounded-xl border border-slate-100 bg-background px-4 py-3 transition duration-200 hover:border-primary/40 hover:shadow-subtle"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-900">
                    {item.filename}
                  </p>
                  <p className="text-xs text-slate-500">
                    {formatRelativeTime(item.createdAt)}
                  </p>
                </div>
                <div className="flex items-center gap-3 pl-4">
                  <span
                    className="rounded-full px-3 py-1 text-xs font-semibold text-white shadow-sm"
                    style={{ backgroundColor: badgeColor }}
                  >
                    {CLASS_NAMES[item.predictedClass]}
                  </span>
                  <span className="text-sm font-semibold text-slate-700">
                    {confidence}%
                  </span>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
};

export default RecentPredictions;
