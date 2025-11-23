import axios from 'axios';
import type {
  ClassId,
  ModelMetricsResponse,
  PredictionConfidence,
  PredictionResult,
} from '../types';
import { CLASS_NAMES, CLASS_COLORS, CLASS_IDS } from '../types';
import {
  CLASS_PERFORMANCE,
  CONFUSION_MATRIX,
  MODEL_INFO,
  MODEL_METRICS,
  createPredictionId,
  getConfidenceLabel,
} from './modelData';
import {
  fetchRecentPredictions,
  fetchPredictionHistory,
  removePredictionRecord,
  clearPredictionHistory,
} from './predictions';
import { supabase } from './supabaseClient';

import type {
  BatchPredictionResult,
  BatchPredictionSummary,
  BatchPredictionDistributionEntry,
} from '../types';

const API_TIMEOUT = 60_000; // Reduced timeout for faster failure detection

const API_ENDPOINTS = {
  predict: 'https://joy-awino--retfound-rop-predict-image.modal.run',
  metrics: 'https://joy-awino--retfound-rop-get-metrics.modal.run',
  confusionMatrix: 'https://joy-awino--retfound-rop-get-confusion-matrix.modal.run',
};

// Optimized image settings for faster upload
const MAX_IMAGE_DIMENSION = 1024; // Reduced from 1600 for faster processing
const MAX_IMAGE_BYTES = 2 * 1024 * 1024; // Reduced from 6MB to 2MB

const readBlobAsDataUrl = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () =>
      reject(new Error('Unable to read the selected image. Please try again.'));
    reader.readAsDataURL(blob);
  });

const downsampleImageIfNeeded = async (
  file: File
): Promise<{ blob: Blob; wasDownsampled: boolean }> => {
  if (typeof window === 'undefined' || !window.HTMLCanvasElement) {
    return { blob: file, wasDownsampled: false };
  }

  return new Promise((resolve) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const { width, height } = image;
      const largestDimension = Math.max(width, height);
      const needsResize = largestDimension > MAX_IMAGE_DIMENSION;
      const needsCompress = file.size > MAX_IMAGE_BYTES;

      if (!needsResize && !needsCompress) {
        resolve({ blob: file, wasDownsampled: false });
        return;
      }

      const scale = needsResize ? MAX_IMAGE_DIMENSION / largestDimension : 1;
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const context = canvas.getContext('2d', { alpha: false });

      if (!context) {
        resolve({ blob: file, wasDownsampled: false });
        return;
      }

      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            resolve({ blob: file, wasDownsampled: false });
            return;
          }
          resolve({
            blob: new File([blob], file.name.replace(/\.[^.]+$/, '.jpg'), {
              type: 'image/jpeg',
              lastModified: Date.now(),
            }),
            wasDownsampled: true,
          });
        },
        'image/jpeg',
        0.85 // Reduced from 0.92 for faster upload while maintaining quality
      );
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve({ blob: file, wasDownsampled: false });
    };

    image.src = objectUrl;
  });
};

const prepareImageForUpload = async (file: File) => {
  const { blob, wasDownsampled } = await downsampleImageIfNeeded(file);
  const dataUrl = await readBlobAsDataUrl(blob);
  const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
  return {
    base64,
    wasDownsampled,
    originalFileSize: file.size,
    processedFileSize: blob.size,
  };
};

const sendPredictionRequest = async (payload: {
  image: string;
  file_name: string;
  user_id?: string;
  metadata?: Record<string, unknown>;
}) => {
  const requestStart =
    typeof performance !== 'undefined' ? performance.now() : Date.now();

  try {
    const { data } = await axios.post(API_ENDPOINTS.predict, payload, {
      timeout: API_TIMEOUT,
    });

    const requestEnd =
      typeof performance !== 'undefined' ? performance.now() : Date.now();

    if (data?.detail && typeof data.detail === 'string') {
      throw new Error(data.detail);
    }

    if (data?.error) {
      throw new Error(
        typeof data.error === 'string'
          ? data.error
          : 'Unable to analyze the image. Please try again.'
      );
    }

    return { data, latencyMs: requestEnd - requestStart };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const responseData = error.response?.data as any;
      const detail =
        responseData?.detail || responseData?.error || responseData?.message;

      if (detail && typeof detail === 'string') {
        throw new Error(detail);
      }

      throw new Error(
        error.message || 'Unable to analyze the image. Please try again.'
      );
    }
    throw error;
  }
};

type ApiPredictionConfidence =
  | number
  | {
      classId?: number;
      label?: number;
      value?: number;
      score?: number;
      confidence?: number;
    };

const normalizeConfidences = (
  confidences: ApiPredictionConfidence[]
): PredictionConfidence[] => {
  if (!Array.isArray(confidences) || confidences.length === 0) {
    return [
      { classId: 0, confidence: 1 },
      { classId: 1, confidence: 0 },
      { classId: 2, confidence: 0 },
    ];
  }

  const mapped = confidences.map((entry, index) => {
    if (typeof entry === 'number') {
      return { classId: (index as ClassId) ?? 0, confidence: entry };
    }

    const classId =
      (entry.classId ?? entry.label ?? (index as ClassId)) as ClassId;
    const confidence = entry.confidence ?? entry.value ?? entry.score ?? 0;

    return {
      classId,
      confidence,
    };
  });

  const total = mapped.reduce((sum, item) => sum + item.confidence, 0);
  if (total <= 0) {
    return mapped.map((item, index) => ({
      classId: item.classId,
      confidence: index === 0 ? 1 : 0,
    }));
  }

  return mapped
    .map((item) => ({
      ...item,
      confidence: item.confidence / total,
    }))
    .sort((a, b) => b.confidence - a.confidence);
};


const classifyPredictionResponse = (
  payload: any,
  extras: Partial<PredictionResult> = {}
): PredictionResult => {
  const rawConfidences = Array.isArray(payload?.confidences)
    ? payload.confidences
    : payload?.probabilities
    ? Object.entries(payload.probabilities).map(([key, value], index) => ({
        classId: index as ClassId,
        confidence: Number(value),
        label: key,
      }))
    : [];

  const confidences = normalizeConfidences(rawConfidences);
  const topConfidence = payload?.topConfidence ?? confidences[0]?.confidence ?? 0;
  const predictedClass = (payload?.predictedClass ?? payload?.predicted_class ?? confidences[0]?.classId ?? 0) as ClassId;

  return {
    id: payload?.id ?? createPredictionId(),
    filename:
      payload?.filename ??
      payload?.file_name ??
      payload?.original_filename ??
      'uploaded-image.png',
    predictedClass,
    confidences,
    createdAt: payload?.createdAt ?? payload?.created_at ?? new Date().toISOString(),
    confidenceLabel: payload?.confidenceLabel ?? getConfidenceLabel(topConfidence),
    topConfidence,
    storagePath: payload?.storagePath ?? payload?.storage_path ?? undefined,
    gradcamPath: payload?.gradcamPath ?? payload?.gradcam_path ?? null,
    gradcamUrl: payload?.gradcamUrl ?? payload?.gradcam_url ?? null,
    gradcamBase64: payload?.gradcamBase64 ?? payload?.gradcam_base64 ?? null,
    gradcamAvailable:
      payload?.gradcamAvailable ??
      payload?.gradcam_available ??
      Boolean(
        (payload?.gradcamPath ?? payload?.gradcam_path) ||
          (payload?.gradcamUrl ?? payload?.gradcam_url) ||
          (payload?.gradcamBase64 ?? payload?.gradcam_base64)
      ),
    ...extras,
  };
};
const normalizeModelMetricsResponse = (payload: any): ModelMetricsResponse => {
  if (
    payload?.metrics &&
    payload?.confusionMatrix &&
    payload?.classPerformance &&
    payload?.modelInfo
  ) {
    return payload as ModelMetricsResponse;
  }

  const metrics = payload?.metrics ?? {
    accuracy: payload?.accuracy ?? MODEL_METRICS.accuracy,
    precision: payload?.precision ?? MODEL_METRICS.precision,
    recall: payload?.recall ?? MODEL_METRICS.recall,
    f1Score: payload?.f1Score ?? payload?.f1 ?? MODEL_METRICS.f1Score,
    rocAuc: payload?.rocAuc ?? payload?.roc_auc ?? MODEL_METRICS.rocAuc,
    kappa: payload?.kappa ?? MODEL_METRICS.kappa,
    hammingLoss: payload?.hammingLoss ?? payload?.hamming_loss ?? MODEL_METRICS.hammingLoss,
    jaccardScore: payload?.jaccardScore ?? payload?.jaccard ?? MODEL_METRICS.jaccardScore,
    averagePrecision:
      payload?.averagePrecision ?? payload?.avg_precision ?? MODEL_METRICS.averagePrecision,
  };

  const confusionMatrix =
    payload?.confusionMatrix ??
    payload?.confusion_matrix ??
    payload?.matrix ??
    CONFUSION_MATRIX;

  const classPerformance =
    payload?.classPerformance ??
    payload?.class_performance ??
    payload?.perClass ??
    CLASS_PERFORMANCE;

  const modelInfo =
    payload?.modelInfo ??
    payload?.model_info ??
    {
      architecture: payload?.architecture ?? MODEL_INFO.architecture,
      trainingEpochs:
        payload?.trainingEpochs ?? payload?.training_epochs ?? MODEL_INFO.trainingEpochs,
      bestEpoch: payload?.bestEpoch ?? MODEL_INFO.bestEpoch,
      lossFunction:
        payload?.lossFunction ?? payload?.loss_function ?? MODEL_INFO.lossFunction,
      learningRate:
        payload?.learningRate ?? payload?.learning_rate ?? MODEL_INFO.learningRate,
      batchSize: payload?.batchSize ?? payload?.batch_size ?? MODEL_INFO.batchSize,
      checkpointPath:
        payload?.checkpointPath ??
        payload?.checkpoint_path ??
        MODEL_INFO.checkpointPath,
    };

  return {
    metrics,
    confusionMatrix,
    classPerformance,
    modelInfo,
  };
};


export const predictImage = async (file: File): Promise<PredictionResult> => {
  const prepared = await prepareImageForUpload(file);

  let userId: string | undefined;
  try {
    const { data: sessionData } = await supabase.auth.getSession();
    userId = sessionData.session?.user?.id ?? undefined;
  } catch (sessionError) {
    console.warn('Unable to fetch Supabase session:', sessionError);
  }

  const { data, latencyMs } = await sendPredictionRequest({
    image: prepared.base64,
    file_name: file.name,
    user_id: userId,
    metadata: {
      original_size: prepared.originalFileSize,
      processed_size: prepared.processedFileSize,
      was_downsampled: prepared.wasDownsampled,
    },
  });

  return classifyPredictionResponse(data, {
    latencyMs,
    originalFileSize: prepared.originalFileSize,
    processedFileSize: prepared.processedFileSize,
    wasDownsampled: prepared.wasDownsampled,
  });
};

const severityOrder: ClassId[] = [2, 1, 0];

const aggregateBatchResults = (
  predictions: PredictionResult[],
  patientId?: string,
  patientName?: string
): BatchPredictionSummary => {
  const distributionMap = new Map<ClassId, { count: number; totalConfidence: number }>();

  predictions.forEach((prediction) => {
    const classId = prediction.predictedClass;
    const entry = distributionMap.get(classId) ?? { count: 0, totalConfidence: 0 };
    const confidence =
      prediction.topConfidence ??
      prediction.confidences.find((item) => item.classId === classId)?.confidence ??
      0;
    entry.count += 1;
    entry.totalConfidence += confidence;
    distributionMap.set(classId, entry);
  });

  const distribution: BatchPredictionDistributionEntry[] = CLASS_IDS.map((classId) => {
    const entry = distributionMap.get(classId);
    if (!entry || entry.count === 0) {
      return { classId, count: 0, averageConfidence: 0 };
    }
    return {
      classId,
      count: entry.count,
      averageConfidence: entry.totalConfidence / entry.count,
    };
  });

  const { classId: aggregateClass, averageConfidence } = distribution
    .filter((item) => item.count > 0)
    .sort((a, b) => {
      if (b.count !== a.count) {
        return b.count - a.count;
      }
      const severityScoreA = severityOrder.indexOf(a.classId);
      const severityScoreB = severityOrder.indexOf(b.classId);
      return severityScoreB - severityScoreA;
    })[0] ??
    // default to no ROP if everything else fails
    { classId: 0 as ClassId, averageConfidence: 0 };

  const notes = (() => {
    if (aggregateClass === 2) {
      return 'One or more images show severe ROP. Escalate to urgent intervention.';
    }
    if (aggregateClass === 1) {
      return 'Signs of early ROP detected. Schedule timely follow-up for this patient.';
    }
    return 'No ROP detected across the uploaded images. Continue routine monitoring.';
  })();

  return {
    patientId,
    patientName,
    createdAt: new Date().toISOString(),
    aggregateClass,
    aggregateConfidence: averageConfidence,
    aggregateLabel: getConfidenceLabel(averageConfidence),
    distribution,
    notes,
  };
};

export const predictImageBatch = async (
  files: File[],
  options?: {
    patientId?: string;
    patientName?: string;
    onProgress?: (payload: {
      completed: number;
      total: number;
      fileIndex: number;
      result?: PredictionResult;
    }) => void;
  }
): Promise<BatchPredictionResult> => {
  const total = files.length;
  if (!total) {
    throw new Error('Please select at least one image to analyze.');
  }
  if (total > 20) {
    throw new Error('Batch prediction supports up to 20 images at a time.');
  }

  let userId: string | undefined;
  try {
    const { data: sessionData } = await supabase.auth.getSession();
    userId = sessionData.session?.user?.id ?? undefined;
  } catch (sessionError) {
    console.warn('Unable to fetch Supabase session:', sessionError);
  }

  const preparedData = await Promise.all(
    files.map(async (file, index) => ({
      index,
      file,
      prepared: await prepareImageForUpload(file),
    }))
  );

  const predictions: (PredictionResult | undefined)[] = new Array(total);
  const errors: Map<number, { filename: string; error: string }> = new Map();
  const concurrencyLimit = Math.min(6, total); // Increased from 4 to 6 for faster batch processing
  let nextTask = 0;
  let completedCount = 0;

  const worker = async (): Promise<void> => {
    while (true) {
      const currentTask = nextTask;
      if (currentTask >= preparedData.length) {
        return;
      }
      nextTask += 1;

      const { file, prepared, index } = preparedData[currentTask];
      try {
        const { data, latencyMs } = await sendPredictionRequest({
          image: prepared.base64,
          file_name: file.name,
          user_id: userId,
          metadata: {
            original_size: prepared.originalFileSize,
            processed_size: prepared.processedFileSize,
            was_downsampled: prepared.wasDownsampled,
            batch_index: index,
            batch_total: total,
            patient_id: options?.patientId ?? null,
            patient_name: options?.patientName ?? null,
          },
        });

        const prediction = classifyPredictionResponse(data, {
          latencyMs,
          originalFileSize: prepared.originalFileSize,
          processedFileSize: prepared.processedFileSize,
          wasDownsampled: prepared.wasDownsampled,
        });

        predictions[index] = prediction;
        completedCount += 1;
        options?.onProgress?.({
          completed: completedCount,
          total,
          fileIndex: index,
          result: prediction,
        });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';
        errors.set(index, { filename: file.name, error: errorMessage });
        completedCount += 1;
        options?.onProgress?.({
          completed: completedCount,
          total,
          fileIndex: index,
        });
      }
    }
  };

  await Promise.all(
    Array.from({ length: concurrencyLimit }).map(() => worker())
  );

  const filledPredictions = predictions.filter(
    (item): item is PredictionResult => Boolean(item)
  );

  // If ALL images failed, throw an error
  if (filledPredictions.length === 0 && errors.size > 0) {
    const firstError = Array.from(errors.values())[0];
    throw new Error(firstError.error);
  }

  const summary = aggregateBatchResults(
    filledPredictions,
    options?.patientId,
    options?.patientName
  );

  const errorArray = Array.from(errors.entries()).map(([fileIndex, { filename, error }]) => ({
    fileIndex,
    filename,
    error,
  }));

  return {
    summary,
    predictions: filledPredictions,
    errors: errorArray.length > 0 ? errorArray : undefined,
  };
};
export const getMetrics = async (): Promise<ModelMetricsResponse> => {
  try {
    const { data } = await axios.get(API_ENDPOINTS.metrics, { timeout: API_TIMEOUT });
    return normalizeModelMetricsResponse(data);
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? error.message
        : 'Unable to load model metrics at this time.'
    );
  }
};

export const getConfusionMatrix = async () => {
  try {
    const { data } = await axios.get(API_ENDPOINTS.confusionMatrix, {
      timeout: API_TIMEOUT,
    });
    return data as typeof CONFUSION_MATRIX;
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? error.message
        : 'Unable to load confusion matrix.'
    );
  }
};

export const getPredictionHistory = async (): Promise<PredictionResult[]> => {
  return fetchPredictionHistory();
};

export const getRecentPredictions = async (
  limit = 5
): Promise<PredictionResult[]> => {
  return fetchRecentPredictions(limit);
};

export const deletePrediction = async (id: string): Promise<void> => {
  await removePredictionRecord(id);
};

export const clearHistory = async (): Promise<void> => {
  await clearPredictionHistory();
};

export const getPredictionBadgeStyles = (classId: ClassId) => ({
  label: CLASS_NAMES[classId],
  color: CLASS_COLORS[classId],
});



