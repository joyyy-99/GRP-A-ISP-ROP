import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Check, CheckCircle2, Loader2 } from 'lucide-react';
import Header from '../components/Layout/Header';
import ImageDropzone from '../components/Upload/ImageDropzone';
import { predictImageBatch } from '../services/api';
import type { BatchPredictionResult, PredictionResult } from '../types';
import { CLASS_COLORS, CLASS_NAMES } from '../types';

const MAX_BATCH_IMAGES = 20;

const UploadErrorCard = ({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) => (
  <div className="flex flex-col gap-4 rounded-3xl border-2 border-danger bg-danger/10 p-8 text-danger shadow-lg animate-[fadeIn_0.4s_ease-in]">
    <div className="flex items-center gap-3 text-danger">
      <AlertTriangle className="h-10 w-10 animate-[fadeIn_0.6s_ease-in]" strokeWidth={2} />
      <div>
        <h3 className="text-xl font-bold">Upload Issue</h3>
        <p className="mt-1 text-sm font-medium">{message}</p>
      </div>
    </div>
    <button
      type="button"
      onClick={onRetry}
      className="self-start rounded-full border-2 border-danger bg-white px-6 py-3 text-sm font-bold text-danger transition duration-200 hover:bg-danger hover:text-white focus-visible:outline-danger shadow-md"
    >
      Start a new batch
    </button>
  </div>
);

const BatchSummaryCard = memo(({
  summary,
  totalImages,
  onReset,
  onDownloadReport,
}: {
  summary: BatchPredictionResult['summary'];
  totalImages: number;
  onReset: () => void;
  onDownloadReport: () => void;
}) => {
  const severityColor =
    summary.aggregateClass === 2
      ? 'text-danger'
      : summary.aggregateClass === 1
      ? 'text-warning'
      : 'text-success';

  return (
    <div className="space-y-6 rounded-3xl border border-slate-200 bg-white p-8 shadow-subtle">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">
            Patient Summary
          </p>
          <h2 className="text-2xl font-semibold text-slate-900">
            {summary.patientName?.trim()
              ? summary.patientName
              : summary.patientId?.trim()
              ? `Patient ${summary.patientId}`
              : 'Unnamed patient'}
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Batch analyzed on {new Date(summary.createdAt).toLocaleString()}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-background px-4 py-3 text-center">
          <p className="text-2xl font-semibold text-slate-900">{totalImages}</p>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Images</p>
        </div>
      </div>

      <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-background/40 p-4">
        <div className="flex items-center gap-3">
          <CheckCircle2
            className={`h-10 w-10 ${
              summary.aggregateClass === 2
                ? 'text-danger'
                : summary.aggregateClass === 1
                ? 'text-warning'
                : 'text-success'
            }`}
          />
          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-slate-400">Model verdict</p>
            <h3 className={`text-xl font-semibold ${severityColor}`}>
              {CLASS_NAMES[summary.aggregateClass]}
            </h3>
            <p className="text-xs text-slate-500">
              Confidence: {(summary.aggregateConfidence * 100).toFixed(1)}% (
              {summary.aggregateLabel})
            </p>
          </div>
        </div>
        <p className="text-sm text-slate-600">{summary.notes}</p>
      </div>

      <div className="space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">
          Distribution across uploaded images
        </h4>
        <div className="space-y-3">
          {summary.distribution.map((entry) => {
            const percentage = totalImages > 0 ? (entry.count / totalImages) * 100 : 0;
            return (
              <div key={entry.classId} className="space-y-1.5">
                <div className="flex items-center justify-between text-sm text-slate-600">
                  <span className="font-medium text-slate-700">
                    {CLASS_NAMES[entry.classId]}
                  </span>
                  <span>
                    {entry.count} / {totalImages} ({percentage.toFixed(0)}%)
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: `${percentage.toFixed(1)}%`,
                      backgroundColor: CLASS_COLORS[entry.classId],
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onDownloadReport}
          className="inline-flex items-center justify-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-semibold text-white transition duration-200 hover:bg-primary/90 focus-visible:outline-primary shadow-md"
        >
          Download Batch Report (PDF)
        </button>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition duration-200 hover:border-primary hover:text-primary focus-visible:outline-primary"
        >
          <Check className="h-4 w-4" />
          Start new batch
        </button>
      </div>
    </div>
  );
});

BatchSummaryCard.displayName = 'BatchSummaryCard';

const UploadPage = () => {
  const [patientId, setPatientId] = useState('');
  const [patientName, setPatientName] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [filePreviews, setFilePreviews] = useState<string[]>([]);
  const [batchResult, setBatchResult] = useState<BatchPredictionResult | null>(null);
  const [perImageResults, setPerImageResults] = useState<(PredictionResult | null)[]>([]);
  const [perImageErrors, setPerImageErrors] = useState<Map<number, string>>(new Map());
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const [dropzoneError, setDropzoneError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStarted, setAnalysisStarted] = useState(false);

  // Use ref to track previews for cleanup
  const filePreviewsRef = useRef<string[]>([]);

  useEffect(() => {
    filePreviewsRef.current = filePreviews;
  }, [filePreviews]);

  useEffect(() => {
    // Cleanup function to revoke object URLs when component unmounts
    return () => {
      filePreviewsRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const resetBatch = useCallback(() => {
    setSelectedFiles([]);
    filePreviews.forEach((url) => URL.revokeObjectURL(url));
    setFilePreviews([]);
    setBatchResult(null);
    setPerImageResults([]);
    setPerImageErrors(new Map());
    setProgress({ completed: 0, total: 0 });
    setError(null);
    setDropzoneError(null);
    setIsAnalyzing(false);
    setAnalysisStarted(false);
  }, [filePreviews]);

  const handleDownloadBatchReport = useCallback(async () => {
    if (!batchResult) return;

    const { jsPDF } = await import('jspdf');
    const doc = new jsPDF();

    // Helper to convert File to base64
    const fileToBase64 = (file: File): Promise<string> => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    };

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const contentWidth = pageWidth - 2 * margin;
    let yPosition = margin;

    // Helper to add new page if needed
    const checkPageBreak = (neededSpace: number) => {
      if (yPosition + neededSpace > pageHeight - margin) {
        doc.addPage();
        yPosition = margin;
        return true;
      }
      return false;
    };

    // Title
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('ROP Detection Batch Report', margin, yPosition);
    yPosition += 12;

    // Patient Info
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition);
    yPosition += 6;

    if (batchResult.summary.patientName) {
      doc.text(`Patient: ${batchResult.summary.patientName}`, margin, yPosition);
      yPosition += 6;
    }

    if (batchResult.summary.patientId) {
      doc.text(`Patient ID: ${batchResult.summary.patientId}`, margin, yPosition);
      yPosition += 6;
    }

    doc.text(`Total Images: ${batchResult.predictions.length}`, margin, yPosition);
    yPosition += 10;

    // Summary Section
    checkPageBreak(40);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Batch Summary', margin, yPosition);
    yPosition += 8;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Aggregate Diagnosis: ${CLASS_NAMES[batchResult.summary.aggregateClass]}`, margin, yPosition);
    yPosition += 6;
    doc.text(`Confidence: ${batchResult.summary.aggregateLabel} (${(batchResult.summary.aggregateConfidence * 100).toFixed(1)}%)`, margin, yPosition);
    yPosition += 6;

    if (batchResult.summary.notes) {
      const splitNotes = doc.splitTextToSize(batchResult.summary.notes, contentWidth);
      doc.text(splitNotes, margin, yPosition);
      yPosition += splitNotes.length * 5 + 5;
    }

    // Distribution
    yPosition += 5;
    checkPageBreak(30);
    doc.setFont('helvetica', 'bold');
    doc.text('Class Distribution:', margin, yPosition);
    yPosition += 6;
    doc.setFont('helvetica', 'normal');

    batchResult.summary.distribution.forEach((dist) => {
      if (dist.count > 0) {
        doc.text(
          `  ${CLASS_NAMES[dist.classId]}: ${dist.count} image${dist.count > 1 ? 's' : ''} (avg ${(dist.averageConfidence * 100).toFixed(1)}%)`,
          margin,
          yPosition
        );
        yPosition += 5;
      }
    });

    // Individual Results
    yPosition += 10;
    checkPageBreak(20);
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Individual Image Results', margin, yPosition);
    yPosition += 10;

    for (let i = 0; i < batchResult.predictions.length; i++) {
      const prediction = batchResult.predictions[i];
      const originalFile = selectedFiles.find(f => f.name === prediction.filename);

      checkPageBreak(140);

      // Image header
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.text(`Image ${i + 1}: ${prediction.filename}`, margin, yPosition);
      yPosition += 7;

      // Add original image
      if (originalFile) {
        try {
          const imageBase64 = await fileToBase64(originalFile);
          const imgWidth = 70;
          const imgHeight = 70;

          doc.setFont('helvetica', 'italic');
          doc.setFontSize(9);
          doc.text('Original Image:', margin + 5, yPosition);
          yPosition += 5;

          doc.addImage(imageBase64, 'JPEG', margin + 5, yPosition, imgWidth, imgHeight);

          // Add Grad-CAM next to original if available
          if (prediction.gradcamBase64) {
            try {
              const gradcamData = prediction.gradcamBase64.startsWith('data:')
                ? prediction.gradcamBase64
                : `data:image/png;base64,${prediction.gradcamBase64}`;

              doc.text('Grad-CAM Heatmap:', margin + imgWidth + 15, yPosition - 5);
              doc.addImage(gradcamData, 'PNG', margin + imgWidth + 15, yPosition, imgWidth, imgHeight);
            } catch (error) {
              console.error('Error adding Grad-CAM to PDF:', error);
            }
          }

          yPosition += imgHeight + 5;
        } catch (error) {
          console.error('Error adding original image to PDF:', error);
          doc.setFont('helvetica', 'normal');
          doc.setFontSize(9);
          doc.text('(Original image unavailable)', margin + 10, yPosition);
          yPosition += 5;
        }
      }

      yPosition += 3;

      // Prediction details
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.text(`Diagnosis: ${CLASS_NAMES[prediction.predictedClass]}`, margin + 5, yPosition);
      yPosition += 5;
      doc.text(`Confidence: ${prediction.confidenceLabel} (${((prediction.topConfidence || 0) * 100).toFixed(1)}%)`, margin + 5, yPosition);
      yPosition += 5;

      // Confidence breakdown
      doc.setFont('helvetica', 'italic');
      doc.setFontSize(9);
      doc.text('Confidence Distribution:', margin + 5, yPosition);
      yPosition += 5;

      prediction.confidences.forEach((conf) => {
        doc.text(
          `  ${CLASS_NAMES[conf.classId]}: ${(conf.confidence * 100).toFixed(1)}%`,
          margin + 10,
          yPosition
        );
        yPosition += 4;
      });

      yPosition += 5;

      // Draw separator line
      if (i < batchResult.predictions.length - 1) {
        doc.setDrawColor(200, 200, 200);
        doc.line(margin, yPosition, pageWidth - margin, yPosition);
        yPosition += 8;
      }
    }

    // Save the PDF
    const filename = batchResult.summary.patientId
      ? `ROP_Batch_Report_${batchResult.summary.patientId}_${new Date().toISOString().split('T')[0]}.pdf`
      : `ROP_Batch_Report_${new Date().toISOString().split('T')[0]}.pdf`;

    doc.save(filename);
  }, [batchResult, selectedFiles]);

  const analyzeBatch = useCallback(
    async (files: File[]) => {
      if (!files.length) return;

      setIsAnalyzing(true);
      setError(null);
      setBatchResult(null);
      setPerImageResults(Array(files.length).fill(null));
      setPerImageErrors(new Map());
      setProgress({ completed: 0, total: files.length });

      try {
        const result = await predictImageBatch(files, {
          patientId: patientId.trim() || undefined,
          patientName: patientName.trim() || undefined,
          onProgress: ({ completed, total, fileIndex, result: partialResult }) => {
            setProgress({ completed, total });
            if (typeof fileIndex === 'number' && partialResult) {
              setPerImageResults((prev) => {
                const next = prev.slice();
                next[fileIndex] = partialResult;
                return next;
              });
            }
          },
        });

        setBatchResult(result);

        // Build a map to reconstruct all results including successful ones
        const allResults: (PredictionResult | null)[] = Array(files.length).fill(null);
        result.predictions.forEach((prediction) => {
          const index = files.findIndex(f => f.name === prediction.filename);
          if (index !== -1) {
            allResults[index] = prediction;
          }
        });

        setPerImageResults(allResults);

        // Track errors
        if (result.errors && result.errors.length > 0) {
          const errorMap = new Map<number, string>();
          result.errors.forEach((err) => {
            errorMap.set(err.fileIndex, err.error);
          });
          setPerImageErrors(errorMap);
        }
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Unable to analyze this batch right now. Please retry.'
        );
      } finally {
        setIsAnalyzing(false);
      }
    },
    [patientId, patientName]
  );

  const handleFilesSelected = useCallback(
    (files: File[]) => {
      const limitedFiles = files.slice(0, MAX_BATCH_IMAGES);
      filePreviews.forEach((url) => URL.revokeObjectURL(url));
      const previews = limitedFiles.map((file) => URL.createObjectURL(file));
      setDropzoneError(null);
      setSelectedFiles(limitedFiles);
      setFilePreviews(previews);
      setAnalysisStarted(true);
      void analyzeBatch(limitedFiles);
    },
    [analyzeBatch, filePreviews]
  );

  const processingIndex = useMemo(() => {
    if (!isAnalyzing) {
      return -1;
    }
    return perImageResults.findIndex((item) => !item);
  }, [isAnalyzing, perImageResults]);

  const fileItems = useMemo(
    () =>
      selectedFiles.map((file, index) => {
        const result = perImageResults[index];
        const errorMessage = perImageErrors.get(index);
        const completed = Boolean(result);
        const failed = Boolean(errorMessage);

        let status: 'completed' | 'processing' | 'queued' | 'pending' | 'error';
        if (failed) {
          status = 'error';
        } else if (completed) {
          status = 'completed';
        } else if (isAnalyzing && index === processingIndex) {
          status = 'processing';
        } else if (analysisStarted) {
          status = 'queued';
        } else {
          status = 'pending';
        }

        return {
          index,
          file,
          previewUrl: filePreviews[index] ?? null,
          status,
          result: result ?? null,
          error: errorMessage ?? null,
        };
      }),
    [analysisStarted, filePreviews, isAnalyzing, perImageResults, perImageErrors, processingIndex, selectedFiles]
  );

  return (
    <div className="min-h-screen">
      <Header
        title="Upload & Analyze"
        subtitle="Upload retinal image batches, generate ROP predictions, and review Grad-CAM overlays."
      />

      <div className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6">
        {error ? <UploadErrorCard message={error} onRetry={resetBatch} /> : null}

        <div className="grid gap-8 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-subtle">
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Patient Details
              </h3>
              <p className="mt-2 text-xs text-slate-500">
                Optional identifiers help track batch outcomes across visits.
              </p>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <label className="space-y-1 text-xs font-medium text-slate-600">
                  <span>Patient name</span>
                  <input
                    type="text"
                    value={patientName}
                    onChange={(event) => setPatientName(event.target.value)}
                    placeholder="e.g. Baby A. Smith"
                    className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-2.5 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                  />
                </label>
                <label className="space-y-1 text-xs font-medium text-slate-600">
                  <span>Hospital / MRN ID</span>
                  <input
                    type="text"
                    value={patientId}
                    onChange={(event) => setPatientId(event.target.value)}
                    placeholder="Optional patient ID"
                    className="w-full rounded-2xl border border-slate-200 bg-background px-4 py-2.5 text-sm text-slate-800 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                  />
                </label>
              </div>
            </div>

            <div className="rounded-3xl border border-dashed border-slate-200 bg-white/60 p-6 shadow-subtle">
              <ImageDropzone
                onFilesSelected={handleFilesSelected}
                error={dropzoneError}
                onError={setDropzoneError}
                disabled={isAnalyzing}
                maxFiles={MAX_BATCH_IMAGES}
                allowMultiple
              />
            </div>

            {fileItems.length ? (
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-subtle">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Uploaded Images
                  </h3>
                  <div className="flex items-center gap-3 text-xs">
                    <p className="text-slate-500">
                      {progress.completed}/{progress.total} processed
                    </p>
                    {perImageErrors.size > 0 && (
                      <p className="font-semibold text-danger">
                        {perImageErrors.size} failed
                      </p>
                    )}
                  </div>
                </div>
                <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {fileItems.map((item) => (
                    <div
                      key={`${item.file.name}-${item.index}-preview`}
                      className="rounded-2xl border border-slate-200 bg-background p-3"
                    >
                      <div className="relative h-36 w-full overflow-hidden rounded-xl bg-slate-100">
                        {item.previewUrl ? (
                          <img
                            src={item.previewUrl}
                            alt={item.file.name}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center text-slate-400">
                            No preview
                          </div>
                        )}
                        <span
                          className={`absolute left-3 top-3 rounded-full px-3 py-1 text-xs font-semibold text-white ${
                            item.status === 'error'
                              ? 'bg-danger'
                              : item.status === 'completed'
                              ? 'bg-success'
                              : item.status === 'processing'
                              ? 'bg-primary'
                              : item.status === 'queued'
                              ? 'bg-warning'
                              : 'bg-slate-400'
                          }`}
                        >
                          {item.status === 'error'
                            ? 'Failed'
                            : item.status === 'completed'
                            ? 'Complete'
                            : item.status === 'processing'
                            ? 'Analyzing'
                            : item.status === 'queued'
                            ? 'Queued'
                            : 'Pending'}
                        </span>
                      </div>
                      <div className="mt-3 space-y-1 text-xs">
                        <p className="truncate font-semibold text-slate-700">
                          {item.file.name}
                        </p>
                        <p className="text-slate-500">
                          {(item.file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                        {item.error && (
                          <div className="mt-2 rounded-lg border border-danger/30 bg-danger/10 p-2">
                            <p className="text-xs font-medium text-danger leading-relaxed">
                              {item.error}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="space-y-6">
            {isAnalyzing ? (
              <div className="flex items-center gap-3 rounded-3xl border border-primary/30 bg-primary/5 px-5 py-4 text-sm text-primary shadow-subtle">
                <Loader2 className="h-5 w-5 animate-spin" />
                Analyzing retinal images... ({progress.completed}/{progress.total})
              </div>
            ) : null}

            {batchResult ? (
              <BatchSummaryCard
                summary={batchResult.summary}
                totalImages={selectedFiles.length}
                onReset={resetBatch}
                onDownloadReport={handleDownloadBatchReport}
              />
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
