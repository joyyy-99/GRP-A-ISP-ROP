import { useEffect, useState } from 'react';
import Header from '../components/Layout/Header';
import MetricCard from '../components/Dashboard/MetricCard';
import ConfusionMatrixPreview from '../components/Dashboard/ConfusionMatrixPreview';
import RecentPredictions from '../components/Dashboard/RecentPredictions';
import { getRecentPredictions } from '../services/api';
import type { ModelMetricsResponse, PredictionResult } from '../types';
import {
  CLASS_PERFORMANCE,
  CONFUSION_MATRIX,
  MODEL_INFO,
} from '../services/modelData';

const formatPercent = (value: number, decimals = 2) =>
  `${(value * 100).toFixed(decimals)}%`;
const formatDecimal = (value: number, decimals = 4) =>
  value.toFixed(decimals);

const DashboardPage = () => {
  const [metrics] = useState<ModelMetricsResponse | null>({
    metrics: {
      accuracy: 0.7559,
      precision: 0.7732,
      recall: 0.7389,
      f1Score: 0.7416,
      rocAuc: 0.9033,
      kappa: 0.6256,
      hammingLoss: 0.1627,
      jaccardScore: 0.5954,
      averagePrecision: 0.825,
    },
    confusionMatrix: CONFUSION_MATRIX,
    classPerformance: CLASS_PERFORMANCE,
    modelInfo: MODEL_INFO,
  });
  const [recent, setRecent] = useState<PredictionResult[]>([]);
  const [loading] = useState(false);
  const [error] = useState<string | null>(null);

  useEffect(() => {
    const loadRecent = async () => {
      try {
        const data = await getRecentPredictions();
        setRecent(data);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Failed to load recent predictions:', err);
      }
    };

    void loadRecent();
  }, []);

  const metricsCards = metrics
    ? [
        {
          title: 'Test Accuracy',
          value: formatPercent(metrics.metrics.accuracy),
          variant: 'primary' as const,
          subtitle: 'Best epoch 26 / 100',
        },
        {
          title: 'F1 Score',
          value: formatDecimal(metrics.metrics.f1Score),
          variant: 'success' as const,
        },
        {
          title: 'ROC AUC',
          value: formatDecimal(metrics.metrics.rocAuc),
          variant: 'success' as const,
        },
        {
          title: "Cohen's Kappa",
          value: formatDecimal(metrics.metrics.kappa, 4),
          variant: 'neutral' as const,
        },
      ]
    : [];

  const secondaryCards = metrics
    ? [
        {
          title: 'Precision',
          value: formatDecimal(metrics.metrics.precision),
          subtitle: `Average Precision ${formatDecimal(
            metrics.metrics.averagePrecision
          )}`,
        },
        {
          title: 'Recall',
          value: formatDecimal(metrics.metrics.recall),
        },
        {
          title: 'Hamming Loss',
          value: formatDecimal(metrics.metrics.hammingLoss, 4),
        },
        {
          title: 'Jaccard Score',
          value: formatDecimal(metrics.metrics.jaccardScore, 4),
        },
      ]
    : [];

  return (
    <div className="min-h-screen">
      <Header
        title="Insights Dashboard"
        subtitle="Review model performance trends and quality signals for recent runs."
        actions={
          <span className="rounded-full bg-background px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Live Modal API
          </span>
        }
      />

      <div className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6">
        {error ? (
          <div className="mb-8 rounded-2xl border border-danger/40 bg-danger/10 px-6 py-4 text-sm text-danger">
            {error}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-4">
          {loading && !metricsCards.length
            ? Array.from({ length: 4 }).map((_, idx) => (
                <div
                  key={`skeleton-${idx}`}
                  className="h-36 animate-pulse rounded-2xl bg-slate-200/60"
                />
              ))
            : metricsCards.map(({ title, value, variant, subtitle }) => (
                <MetricCard
                  key={title}
                  title={title}
                  value={value}
                  variant={variant}
                  subtitle={subtitle}
                />
              ))}
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-4">
          {secondaryCards.map(({ title, value, subtitle }) => (
            <MetricCard
              key={title}
              title={title}
              value={value}
              subtitle={subtitle}
              variant="neutral"
            />
          ))}
        </div>

        <div className="mt-10 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          {metrics ? (
            <ConfusionMatrixPreview data={metrics.confusionMatrix} />
          ) : (
            <div className="h-80 animate-pulse rounded-2xl bg-slate-200/60" />
          )}
          <RecentPredictions items={recent} />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
