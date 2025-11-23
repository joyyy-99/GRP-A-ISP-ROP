import { useCallback, useEffect, useState } from 'react';
import Header from '../components/Layout/Header';
import PredictionHistoryTable from '../components/History/PredictionHistory';
import PredictionResultCard from '../components/Upload/PredictionResult';
import { deletePrediction, getPredictionHistory } from '../services/api';
import type { PredictionResult } from '../types';
import { CLASS_NAMES } from '../types';

const buildReport = (result: PredictionResult) => {
  const lines = [
    'ROP Detection Report',
    `Generated: ${new Date(result.createdAt).toLocaleString()}`,
    `Filename: ${result.filename}`,
    `Diagnosis: ${CLASS_NAMES[result.predictedClass]} (${result.confidenceLabel} confidence)`,
    '',
    'Confidence Distribution:',
    ...result.confidences.map(
      (entry) =>
        `${CLASS_NAMES[entry.classId]}: ${(entry.confidence * 100).toFixed(1)}%`
    ),
  ];

  const blob = new Blob([lines.join('\n')], {
    type: 'text/plain;charset=utf-8;',
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `rop_prediction_${result.id}.txt`;
  anchor.click();
  URL.revokeObjectURL(url);
};

const HistoryPage = () => {
  const [history, setHistory] = useState<PredictionResult[]>([]);
  const [selected, setSelected] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getPredictionHistory();
      setHistory(data);
      if (selected) {
        const refreshed = data.find((item) => item.id === selected.id) ?? null;
        setSelected(refreshed);
      }
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load prediction history.'
      );
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleDelete = async (id: string) => {
    try {
      await deletePrediction(id);
      if (selected?.id === id) {
        setSelected(null);
      }
      await loadHistory();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to delete the selected prediction.'
      );
    }
  };

  const handleView = (prediction: PredictionResult) => {
    setSelected(prediction);
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Prediction History"
        subtitle="Track and manage recent ROP screening results."
      />

      <div className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6">
        {error ? (
          <div className="mb-6 rounded-2xl border border-danger/40 bg-danger/10 px-6 py-4 text-sm text-danger">
            {error}
          </div>
        ) : null}

        {loading && !history.length ? (
          <div className="h-80 animate-pulse rounded-3xl bg-slate-200/60" />
        ) : null}

        <PredictionHistoryTable
          data={history}
          onDelete={handleDelete}
          onView={handleView}
        />

        {selected ? (
          <div className="mt-8">
            <PredictionResultCard
              result={selected}
              isLoading={false}
              onUploadNew={() => setSelected(null)}
              onDownloadReport={buildReport}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default HistoryPage;
