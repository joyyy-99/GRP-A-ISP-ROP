import type { ClassPerformance } from '../../types';

interface PerformanceTableProps {
  rows: ClassPerformance[];
}

const formatValue = (value: number) => value.toFixed(2);

const PerformanceTable = ({ rows }: PerformanceTableProps) => {
  return (
    <section className="rounded-3xl bg-white p-6 shadow-subtle">
      <div className="mb-4">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
          Class Breakdown
        </p>
        <h3 className="mt-1 text-xl font-semibold text-slate-900">
          Per-Class Performance
        </h3>
      </div>
      <div className="overflow-hidden rounded-2xl border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-background text-slate-500">
            <tr>
              <th className="px-6 py-4 text-left font-semibold uppercase tracking-[0.2em]">
                Class
              </th>
              <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                Precision
              </th>
              <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                Recall
              </th>
              <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                F1-Score
              </th>
              <th className="px-6 py-4 text-right font-semibold uppercase tracking-[0.2em]">
                Support
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white text-slate-700">
            {rows.map((row, index) => (
              <tr
                key={row.classId}
                className={index % 2 === 1 ? 'bg-background/60' : 'bg-white'}
              >
                <td className="px-6 py-4 font-semibold text-slate-800">
                  {row.className}
                </td>
                <td className="px-6 py-4 text-right font-medium">
                  {formatValue(row.precision)}
                </td>
                <td className="px-6 py-4 text-right font-medium">
                  {formatValue(row.recall)}
                </td>
                <td className="px-6 py-4 text-right font-medium">
                  {formatValue(row.f1Score)}
                </td>
                <td className="px-6 py-4 text-right font-medium">
                  {row.support.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default PerformanceTable;
