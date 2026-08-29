import { Link } from 'react-router-dom';
import type { ConfusionMatrixData } from '../../types';

interface ConfusionMatrixPreviewProps {
  data: ConfusionMatrixData;
}

const getCellColor = (value: number) => {
  const intensity = Math.min(1, Math.max(0, value));
  const alpha = 0.15 + intensity * 0.75;
  return `rgba(74, 144, 226, ${alpha.toFixed(2)})`;
};

const ConfusionMatrixPreview = ({ data }: ConfusionMatrixPreviewProps) => {
  return (
    <section className="rounded-2xl bg-white p-6 shadow-subtle transition duration-250 card-hover">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
            Model Overview
          </p>
          <h3 className="mt-1 text-lg font-semibold text-slate-900">
            Confusion Matrix
          </h3>
        </div>
        <Link
          to="/metrics"
          className="text-sm font-semibold text-primary hover:underline focus-visible:outline-primary"
        >
          View Details →
        </Link>
      </div>

      <div className="mt-6 space-y-2">
        <div className="grid grid-cols-4 gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          <span />
          {data.labels.map((label) => (
            <span key={label} className="text-center">
              {label}
            </span>
          ))}
        </div>

        {data.normalized.map((row, rowIndex) => (
          <div
            key={data.labels[rowIndex]}
            className="grid grid-cols-4 items-center gap-2 text-sm"
          >
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              {data.labels[rowIndex]}
            </span>
            {row.map((value, colIndex) => {
              const color = getCellColor(value);
              const formatted = `${(value * 100).toFixed(1)}%`;
              const isHigh = value >= 0.5;
              const textColor = isHigh ? 'text-white' : 'text-slate-700';

              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`flex h-16 items-center justify-center rounded-xl font-semibold ${textColor}`}
                  style={{ backgroundColor: color }}
                >
                  {formatted}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
};

export default ConfusionMatrixPreview;
