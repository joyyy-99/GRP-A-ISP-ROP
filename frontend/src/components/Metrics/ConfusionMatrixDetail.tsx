import { useState } from 'react';
import type { ConfusionMatrixData } from '../../types';

interface ConfusionMatrixDetailProps {
  data: ConfusionMatrixData;
  accuracy: number;
}

type MatrixView = 'normalized' | 'raw';

const getCellBackground = (value: number) => {
  const clamped = Math.min(1, Math.max(0, value));
  const alpha = 0.15 + clamped * 0.85;
  return `rgba(74, 144, 226, ${alpha.toFixed(2)})`;
};

const ConfusionMatrixDetail = ({
  data,
  accuracy,
}: ConfusionMatrixDetailProps) => {
  const [view, setView] = useState<MatrixView>('normalized');

  const renderCell = (rowIdx: number, colIdx: number) => {
    const rawValue = data.raw[rowIdx][colIdx];
    const normalizedValue = data.normalized[rowIdx][colIdx];
    const displayValue =
      view === 'normalized'
        ? `${(normalizedValue * 100).toFixed(1)}%`
        : rawValue.toLocaleString();

    const textColor = normalizedValue >= 0.5 ? 'text-white' : 'text-slate-700';

    return (
      <div
        key={`${rowIdx}-${colIdx}`}
        className={`flex h-24 items-center justify-center rounded-2xl text-lg font-semibold ${textColor}`}
        style={{ backgroundColor: getCellBackground(normalizedValue) }}
        title={`Predicted ${data.labels[colIdx]} | Actual ${data.labels[rowIdx]}\nRaw: ${rawValue} | Normalized: ${(normalizedValue * 100).toFixed(2)}%`}
      >
        {displayValue}
      </div>
    );
  };

  return (
    <section className="rounded-3xl bg-white p-8 shadow-subtle">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
            Test Performance
          </p>
          <h3 className="text-2xl font-semibold text-slate-900">
            Confusion Matrix
          </h3>
          <p className="text-sm text-slate-500">
            Overall test accuracy {(accuracy * 100).toFixed(2)}%
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-background p-1 text-xs font-semibold text-slate-600">
          <button
            type="button"
            onClick={() => setView('normalized')}
            className={`rounded-full px-4 py-2 transition ${
              view === 'normalized'
                ? 'bg-white text-primary shadow-subtle'
                : 'hover:text-primary'
            }`}
          >
            Normalized
          </button>
          <button
            type="button"
            onClick={() => setView('raw')}
            className={`rounded-full px-4 py-2 transition ${
              view === 'raw'
                ? 'bg-white text-primary shadow-subtle'
                : 'hover:text-primary'
            }`}
          >
            Raw Counts
          </button>
        </div>
      </div>

      <div className="mt-8 overflow-x-auto">
        <div className="inline-grid grid-cols-[auto_repeat(3,minmax(120px,1fr))] gap-3">
          <div />
          {data.labels.map((label) => (
            <div
              key={`col-${label}`}
              className="text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-400"
            >
              Predicted {label}
            </div>
          ))}

          {data.labels.map((label, rowIdx) => (
            <div key={`row-${label}`} className="contents">
              <div className="flex items-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Actual {label}
              </div>
              {data.labels.map((_, colIdx) => renderCell(rowIdx, colIdx))}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-primary/20" />
          Lower match probability
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-primary/80" />
          Higher match probability
        </span>
      </div>
    </section>
  );
};

export default ConfusionMatrixDetail;
