import { useMemo, useState } from 'react';
import { Download, Eye, Trash2 } from 'lucide-react';
import { CLASS_COLORS, CLASS_IDS, CLASS_NAMES } from '../../types';
import type { ClassId, PredictionResult } from '../../types';

interface PredictionHistoryProps {
  data: PredictionResult[];
  onDelete: (id: string) => void;
  onView?: (prediction: PredictionResult) => void;
}

type SortDirection = 'asc' | 'desc';
type ClassFilter = ClassId | 'all';

const formatDateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });

const formatConfidence = (prediction: PredictionResult) => {
  const entry =
    prediction.confidences.find(
      (item) => item.classId === prediction.predictedClass
    ) ?? prediction.confidences[0];
  return entry ? `${(entry.confidence * 100).toFixed(1)}%` : '0.0%';
};

const buildCsvContent = (rows: PredictionResult[]) => {
  const headers = [
    'Timestamp',
    'Image Name',
    'Predicted Class',
    'Confidence',
    'Confidence Label',
  ];

  const lines = rows.map((row) => [
    new Date(row.createdAt).toISOString(),
    `"${row.filename.replace(/"/g, '""')}"`,
    `"${CLASS_NAMES[row.predictedClass].replace(/"/g, '""')}"`,
    formatConfidence(row),
    row.confidenceLabel,
  ]);

  return [headers.join(','), ...lines.map((line) => line.join(','))].join('\n');
};

const PredictionHistory = ({
  data,
  onDelete,
  onView,
}: PredictionHistoryProps) => {
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [classFilter, setClassFilter] = useState<ClassFilter>('all');

  const filtered = useMemo(() => {
    const rows =
      classFilter === 'all'
        ? data
        : data.filter((item) => item.predictedClass === classFilter);

    return [...rows].sort((a, b) => {
      const aTime = new Date(a.createdAt).getTime();
      const bTime = new Date(b.createdAt).getTime();

      return sortDirection === 'desc' ? bTime - aTime : aTime - bTime;
    });
  }, [classFilter, data, sortDirection]);

  const handleExport = () => {
    if (filtered.length === 0) return;

    const blob = new Blob([buildCsvContent(filtered)], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'rop_prediction_history.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="rounded-3xl bg-white p-6 shadow-subtle">
      <div className="flex flex-col gap-4 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
            Study Log
          </p>
          <h3 className="mt-1 text-2xl font-semibold text-slate-900">
            Prediction History
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={classFilter}
            onChange={(event) =>
              setClassFilter(
                event.target.value === 'all'
                  ? 'all'
                  : (Number(event.target.value) as ClassId)
              )
            }
            className="rounded-full border border-slate-200 bg-background px-4 py-2 text-sm font-medium text-slate-700 focus-visible:outline-primary"
          >
            <option value="all">All Classes</option>
            {CLASS_IDS.map((id) => (
              <option key={id} value={id}>
                {CLASS_NAMES[id]}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleExport}
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition duration-200 hover:border-primary hover:text-primary focus-visible:outline-primary"
            disabled={filtered.length === 0}
          >
            <Download className="h-4 w-4" />
            Export to CSV
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-200 bg-background px-8 py-12 text-center text-sm text-slate-500">
          No predictions in history
        </div>
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-background text-slate-500">
              <tr>
                <th className="px-6 py-4 text-left font-semibold uppercase tracking-[0.2em]">
                  Timestamp
                </th>
                <th className="px-6 py-4 text-left font-semibold uppercase tracking-[0.2em]">
                  Image Name
                </th>
                <th className="px-6 py-4 text-left font-semibold uppercase tracking-[0.2em]">
                  Predicted Class
                </th>
                <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                  Confidence
                </th>
                <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white text-slate-700">
              {filtered.map((row) => (
                <tr key={row.id} className="hover:bg-background/60">
                  <td className="px-6 py-4">
                    <button
                      type="button"
                      onClick={() =>
                        setSortDirection((prev) =>
                          prev === 'desc' ? 'asc' : 'desc'
                        )
                      }
                      className="font-medium text-slate-700 underline-offset-4 hover:underline focus-visible:outline-primary"
                    >
                      {formatDateTime(row.createdAt)}
                    </button>
                  </td>
                  <td className="max-w-xs px-6 py-4">
                    <span className="block truncate font-medium text-slate-800">
                      {row.filename}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className="inline-flex rounded-full px-3 py-1 text-xs font-semibold text-white"
                      style={{
                        backgroundColor: CLASS_COLORS[row.predictedClass],
                      }}
                    >
                      {CLASS_NAMES[row.predictedClass]}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right font-semibold">
                    {formatConfidence(row)}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => onView?.(row)}
                        className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-700 transition duration-200 hover:border-primary hover:text-primary focus-visible:outline-primary"
                      >
                        <Eye className="h-4 w-4" />
                        View
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(row.id)}
                        className="inline-flex items-center gap-1 rounded-full border border-danger/30 bg-danger/10 px-3 py-1 text-xs font-semibold text-danger transition duration-200 hover:bg-danger/20 focus-visible:outline-danger"
                      >
                        <Trash2 className="h-4 w-4" />
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default PredictionHistory;
