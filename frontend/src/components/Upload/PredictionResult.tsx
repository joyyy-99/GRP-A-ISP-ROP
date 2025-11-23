import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FileDown,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import {
  CLASS_COLORS,
  CLASS_DESCRIPTIONS,
  CLASS_IDS,
  CLASS_NAMES,
} from '../../types';
import type { PredictionResult } from '../../types';
import {
  getStoragePublicUrl,
  gradcamBucketName,
} from '../../services/predictions';
import { isSupabaseConfigured, supabase } from '../../services/supabaseClient';

interface PredictionResultProps {
  result: PredictionResult | null;
  isLoading: boolean;
  onUploadNew?: () => void;
  onDownloadReport?: (result: PredictionResult) => void;
  onGradcamStateChange?: (state: {
    show: boolean;
    url: string | null;
    loading: boolean;
    error: string | null;
    available: boolean;
  }) => void;
  hideActions?: boolean;
  hideGradcam?: boolean;
  actionSlot?: ReactNode;
  titleOverride?: string;
  subtitleOverride?: string;
}

const iconMap: Record<number, typeof CheckCircle2> = {
  0: CheckCircle2,
  1: AlertTriangle,
  2: XCircle,
};

const formatConfidenceLabel = (label: PredictionResult['confidenceLabel']) => {
  if (label === 'High') return 'High confidence';
  if (label === 'Medium') return 'Moderate confidence';
  return 'Low confidence';
};

const PredictionResultCard = ({
  result,
  isLoading,
  onUploadNew,
  onDownloadReport,
  onGradcamStateChange,
  hideActions = false,
  hideGradcam = false,
  actionSlot,
  titleOverride,
  subtitleOverride,
}: PredictionResultProps) => {
  const [showGradcam, setShowGradcam] = useState(false);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [gradcamError, setGradcamError] = useState<string | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);

  const gradcamAvailable = useMemo(() => {
    if (!result) return false;
    if (typeof result.gradcamAvailable === 'boolean') {
      return result.gradcamAvailable;
    }
    return Boolean(
      result.gradcamPath || result.gradcamUrl || result.gradcamBase64
    );
  }, [result]);

  useEffect(() => {
    if (!result) {
      setGradcamUrl(null);
      setGradcamError(null);
      setGradcamLoading(false);
      setShowGradcam(false);
      return;
    }

    const immediateUrl =
      result.gradcamUrl ||
      (result.gradcamBase64
        ? `data:image/png;base64,${result.gradcamBase64}`
        : null);

    setGradcamUrl(immediateUrl);
    setGradcamError(null);
    setGradcamLoading(false);
    setShowGradcam(Boolean(immediateUrl));
  }, [result]);

  useEffect(() => {
    onGradcamStateChange?.({
      show: showGradcam,
      url: gradcamUrl,
      loading: gradcamLoading,
      error: gradcamError,
      available: gradcamAvailable,
    });
  }, [
    showGradcam,
    gradcamUrl,
    gradcamLoading,
    gradcamError,
    gradcamAvailable,
    onGradcamStateChange,
  ]);

  const handleToggleGradcam = async () => {
    if (!result) return;

    if (showGradcam) {
      setShowGradcam(false);
      return;
    }

    setGradcamError(null);

    if (gradcamUrl) {
      setShowGradcam(true);
      return;
    }

    if (result.gradcamBase64) {
      setGradcamUrl(`data:image/png;base64,${result.gradcamBase64}`);
      setShowGradcam(true);
      return;
    }

    if (result.gradcamPath && isSupabaseConfigured) {
      try {
        setGradcamLoading(true);
        const { data, error } = await supabase.storage
          .from(gradcamBucketName)
          .createSignedUrl(result.gradcamPath, 60 * 10);
        if (error || !data?.signedUrl) {
          throw new Error(error?.message ?? 'Signed URL not available');
        }
        setGradcamUrl(data.signedUrl);
        setShowGradcam(true);
      } catch (error) {
        setGradcamError(
          error instanceof Error
            ? error.message
            : 'Unable to fetch Grad-CAM overlay right now.'
        );
      } finally {
        setGradcamLoading(false);
      }
      return;
    }

    if (result.gradcamPath && !isSupabaseConfigured) {
      const publicUrl = getStoragePublicUrl(gradcamBucketName, result.gradcamPath);
      if (publicUrl) {
        setGradcamUrl(publicUrl);
        setShowGradcam(true);
        return;
      }
    }

    setGradcamError('Grad-CAM overlay is not yet available for this scan.');
  };

  const showLowConfidence = useMemo(() => {
    if (!result) return false;
    const topConfidence =
      result.topConfidence ??
      result.confidences.find(
        (item) => item.classId === result.predictedClass
      )?.confidence ??
      0;
    return topConfidence < 0.6;
  }, [result]);

  const Icon =
    result && iconMap[result.predictedClass]
      ? iconMap[result.predictedClass]
      : CheckCircle2;

  const title =
    titleOverride ??
    (result ? result.filename : 'Upload a retinal image to begin analysis');

  const subtitle =
    subtitleOverride ??
    (result
      ? `${CLASS_NAMES[result.predictedClass]} • ${new Date(
          result.createdAt
        ).toLocaleString()}`
      : '');

  if (!result) {
    return (
      <div className="flex min-h-[480px] flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-white/60 p-8 text-center shadow-subtle">
        {isLoading ? (
          <>
            <RefreshCw className="mb-6 h-10 w-10 animate-spin text-primary" />
            <p className="text-base font-semibold text-slate-700">
              Analyzing image...
            </p>
            <p className="mt-2 text-sm text-slate-500">
              The model is processing the retinal scan to detect ROP indicators.
            </p>
          </>
        ) : (
          <>
            <CheckCircle2 className="mb-6 h-10 w-10 text-primary" />
            <p className="text-base font-semibold text-slate-700">
              Upload images to view predictions.
            </p>
            <p className="mt-2 text-sm text-slate-500">
              You can upload up to 20 fundus images belonging to the same patient. We&apos;ll analyze each scan and generate a combined summary.
            </p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 rounded-3xl border border-slate-200 bg-white p-8 shadow-subtle">
      <div className="flex items-center gap-3">
        <span
          className={`flex h-12 w-12 items-center justify-center rounded-full text-white shadow-card transition duration-200 ${
            result.predictedClass === 0
              ? 'bg-success'
              : result.predictedClass === 1
              ? 'bg-warning'
              : 'bg-danger'
          }`}
        >
          <Icon className="h-6 w-6" />
        </span>
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
            {title}
          </p>
          <h3 className="text-2xl font-semibold text-slate-900">{subtitle}</h3>
        </div>
      </div>

      <div className="space-y-3 rounded-2xl border border-slate-200 bg-background/40 p-4">
        <h4 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
          Model Assessment
        </h4>
        <p className="text-lg font-semibold text-slate-900">
          {CLASS_NAMES[result.predictedClass]}
        </p>
        <p className="text-sm text-slate-600">
          {CLASS_DESCRIPTIONS[result.predictedClass]}
        </p>
        {result.prediction ? (
          <p className="mt-1 text-sm font-semibold text-primary">
            Model output: {result.prediction}
          </p>
        ) : null}
        {showLowConfidence ? (
          <p className="mt-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
            Confidence is relatively low. Please ensure the image clearly shows the retina for an accurate result.
          </p>
        ) : null}
        <div className="rounded-xl bg-background px-4 py-3 text-sm font-semibold text-slate-600">
          {formatConfidenceLabel(result.confidenceLabel)}
          {typeof result.latencyMs === 'number' ? (
            <span className="ml-2 text-xs font-normal text-slate-500">
              Response time{' '}
              {result.latencyMs < 1000
                ? `${result.latencyMs.toFixed(0)} ms`
                : `${(result.latencyMs / 1000).toFixed(2)} s`}
            </span>
          ) : null}
        </div>
        {(result.originalFileSize || result.processedFileSize) && (
          <div className="grid gap-4 rounded-xl border border-slate-200 bg-white/70 px-4 py-3 text-xs text-slate-600 sm:grid-cols-2">
            {typeof result.originalFileSize === 'number' ? (
              <div>
                <span className="font-semibold text-slate-700">Original file:</span>{' '}
                {(result.originalFileSize / (1024 * 1024)).toFixed(2)} MB
              </div>
            ) : null}
            {typeof result.processedFileSize === 'number' ? (
              <div>
                <span className="font-semibold text-slate-700">Processed size:</span>{' '}
                {(result.processedFileSize / (1024 * 1024)).toFixed(2)} MB
                {result.wasDownsampled ? ' (downsampled)' : ''}
              </div>
            ) : null}
          </div>
        )}
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
          Confidence Distribution
        </h4>
        <div className="space-y-4">
          {CLASS_IDS.map((classId) => {
            const confidenceEntry = result.confidences.find(
              (item) => item.classId === classId
            );
            const confidenceValue = confidenceEntry
              ? (confidenceEntry.confidence * 100).toFixed(1)
              : '0.0';
            const widthPercentage = confidenceEntry
              ? `${(confidenceEntry.confidence * 100).toFixed(1)}%`
              : '0%';

            return (
              <div key={classId} className="space-y-2">
                <div className="flex items-center justify-between text-sm text-slate-600">
                  <span className="font-medium text-slate-700">
                    {CLASS_NAMES[classId]}
                  </span>
                  <span>{confidenceValue}%</span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: widthPercentage,
                      backgroundColor: CLASS_COLORS[classId],
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {hideGradcam ? null : (
        <div className="space-y-3 rounded-2xl border border-slate-200 bg-background/40 p-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h4 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Grad-CAM Interpretation
              </h4>
              <p className="text-xs text-slate-500">
                Highlights retinal regions that most influenced this prediction.
              </p>
            </div>
            <button
              type="button"
              onClick={handleToggleGradcam}
              disabled={
                gradcamLoading ||
                (!gradcamAvailable &&
                  !gradcamUrl &&
                  !result.gradcamBase64 &&
                  !result.gradcamPath)
              }
              className="inline-flex items-center justify-center rounded-full border border-primary/60 bg-white px-4 py-2 text-xs font-semibold text-primary transition duration-200 hover:bg-primary hover:text-white focus-visible:outline-primary disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400"
            >
              {showGradcam ? 'Hide Grad-CAM' : 'Show Grad-CAM'}
            </button>
          </div>
          {gradcamError ? (
            <div className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-xs text-danger">
              {gradcamError}
            </div>
          ) : null}
          {gradcamLoading ? (
            <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white/70 px-4 py-3 text-xs font-medium text-slate-600">
              <RefreshCw className="h-4 w-4 animate-spin text-primary" />
              Generating interpretability overlay...
            </div>
          ) : null}
          {showGradcam && gradcamUrl ? (
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr),minmax(0,0.9fr)]">
              <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-subtle">
                <img
                  src={gradcamUrl}
                  alt={`Grad-CAM overlay for ${result.filename}`}
                  className="w-full object-contain"
                />
              </div>
              <div className="flex flex-col gap-2 rounded-2xl border border-slate-200 bg-white/80 p-4 text-xs text-slate-600">
                <p className="font-semibold text-slate-700">How to read this overlay</p>
                <ul className="space-y-1">
                  <li>
                    • Bright regions mark retinal tissue that drove the {CLASS_NAMES[result.predictedClass]} prediction.
                  </li>
                  <li>
                    • Compare the highlight with the raw fundus image from the upload grid to confirm vessel or ridge changes.
                  </li>
                  <li>• Use alongside clinical notes before escalating care decisions.</li>
                </ul>
              </div>
            </div>
          ) : null}
          {!gradcamAvailable && !gradcamLoading && !gradcamUrl ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-3 text-xs text-slate-500">
              Grad-CAM overlay will appear once it has been generated and stored for this image.
            </div>
          ) : null}
        </div>
      )}

      {hideActions ? (
        actionSlot ? <div>{actionSlot}</div> : null
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => onUploadNew?.()}
            className="inline-flex items-center justify-center rounded-full bg-primary px-6 py-3 text-sm font-semibold text-white shadow-card transition duration-200 hover:bg-primary-dark focus-visible:outline-primary"
          >
            Upload New Image
          </button>
          <button
            type="button"
            onClick={() => result && onDownloadReport?.(result)}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition duration-200 hover:border-primary hover:text-primary focus-visible:outline-primary"
          >
            <FileDown className="h-4 w-4" />
            Download Report
          </button>
        </div>
      )}
    </div>
  );
};

export default PredictionResultCard;


