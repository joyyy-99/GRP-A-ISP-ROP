import { useEffect, useState } from 'react';
import Header from '../components/Layout/Header';
import ConfusionMatrixDetail from '../components/Metrics/ConfusionMatrixDetail';
import PerformanceTable from '../components/Metrics/PerformanceTable';
import { getMetrics } from '../services/api';
import type { ModelMetricsResponse } from '../types';
import { TEST_SET_DISTRIBUTION } from '../services/modelData';

const MetricsPage = () => {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const response = await getMetrics();
        setMetrics(response);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Unable to load detailed metrics.'
        );
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  return (
    <div className="min-h-screen">
      <Header
        title="Model Metrics"
        subtitle="Deep dive into evaluation metrics and class-wise performance."
      />

      <div className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6">
        {error ? (
          <div className="mb-8 rounded-2xl border border-danger/40 bg-danger/10 px-6 py-4 text-sm text-danger">
            {error}
          </div>
        ) : null}

        {loading && !metrics ? (
          <div className="space-y-6">
            <div className="h-96 animate-pulse rounded-3xl bg-slate-200/60" />
            <div className="h-64 animate-pulse rounded-3xl bg-slate-200/60" />
          </div>
        ) : null}

        {metrics ? (
          <div className="space-y-8">
            <ConfusionMatrixDetail
              data={metrics.confusionMatrix}
              accuracy={metrics.metrics.accuracy}
            />
            <div className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
              <PerformanceTable rows={metrics.classPerformance} />

              <section className="rounded-3xl bg-white p-6 shadow-subtle">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Model Information
                </p>
                <h3 className="mt-1 text-xl font-semibold text-slate-900">
                  RETFound Training Summary
                </h3>

                <dl className="mt-6 space-y-4 text-sm text-slate-600">
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Architecture</dt>
                    <dd>{metrics.modelInfo.architecture}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Training Epochs</dt>
                    <dd>{metrics.modelInfo.trainingEpochs}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Best Epoch</dt>
                    <dd>Epoch {metrics.modelInfo.bestEpoch} / 100</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Loss Function</dt>
                    <dd>{metrics.modelInfo.lossFunction}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Learning Rate</dt>
                    <dd>{metrics.modelInfo.learningRate.toExponential()}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="font-semibold text-slate-700">Batch Size</dt>
                    <dd>{metrics.modelInfo.batchSize}</dd>
                  </div>
                </dl>

                <div className="mt-6 rounded-2xl bg-background px-5 py-4 text-sm text-slate-600">
                  <p className="font-semibold text-slate-700">Test Set Distribution</p>
                  <ul className="mt-3 space-y-2">
                    <li>No ROP: {TEST_SET_DISTRIBUTION.byClass[0].toLocaleString()} images</li>
                    <li>Pre-Plus: {TEST_SET_DISTRIBUTION.byClass[1].toLocaleString()} images</li>
                    <li>Plus: {TEST_SET_DISTRIBUTION.byClass[2].toLocaleString()} images</li>
                  </ul>
                  <p className="mt-3 text-xs text-slate-500">
                    Total test images: {TEST_SET_DISTRIBUTION.total.toLocaleString()}
                  </p>
                </div>

                <a
                  href={metrics.modelInfo.checkpointPath}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-6 inline-flex items-center justify-center rounded-full bg-primary px-5 py-2 text-sm font-semibold text-white shadow-card transition duration-200 hover:bg-primary-dark focus-visible:outline-primary"
                >
                  View Modal Checkpoint
                </a>
              </section>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default MetricsPage;
