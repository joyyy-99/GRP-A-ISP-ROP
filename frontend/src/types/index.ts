export type ClassId = 0 | 1 | 2;

export interface MetricData {
  accuracy: number;
  f1Score: number;
  rocAuc: number;
  kappa: number;
  precision: number;
  recall: number;
  hammingLoss: number;
  jaccardScore: number;
  averagePrecision: number;
}

export interface ConfusionMatrixData {
  normalized: number[][];
  raw: number[][];
  labels: string[];
}

export interface ClassPerformance {
  classId: ClassId;
  className: string;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface PredictionConfidence {
  classId: ClassId;
  confidence: number;
}

export interface PredictionResult {
  id: string;
  filename: string;
  predictedClass: ClassId;
  confidences: PredictionConfidence[];
  createdAt: string;
  confidenceLabel: 'High' | 'Medium' | 'Low';
  prediction?: string;
  storagePath?: string;
  gradcamPath?: string | null;
  gradcamUrl?: string | null;
  gradcamBase64?: string | null;
  gradcamAvailable?: boolean;
  topConfidence?: number;
  latencyMs?: number;
  originalFileSize?: number;
  processedFileSize?: number;
  wasDownsampled?: boolean;
}

export interface BatchPredictionDistributionEntry {
  classId: ClassId;
  count: number;
  averageConfidence: number;
}

export interface BatchPredictionSummary {
  patientId?: string;
  patientName?: string;
  createdAt: string;
  aggregateClass: ClassId;
  aggregateConfidence: number;
  aggregateLabel: 'High' | 'Medium' | 'Low';
  distribution: BatchPredictionDistributionEntry[];
  notes?: string;
}

export interface BatchPredictionError {
  fileIndex: number;
  filename: string;
  error: string;
}

export interface BatchPredictionResult {
  summary: BatchPredictionSummary;
  predictions: PredictionResult[];
  errors?: BatchPredictionError[];
}

export interface ModelInfo {
  architecture: string;
  trainingEpochs: number;
  bestEpoch: number;
  lossFunction: string;
  learningRate: number;
  batchSize: number;
  checkpointPath: string;
}

export const CLASS_NAMES: Record<ClassId, string> = {
  0: 'No ROP',
  1: 'Early ROP',
  2: 'Severe ROP',
};

export const CLASS_IDS: ClassId[] = [0, 1, 2];

export const CLASS_COLORS: Record<ClassId, string> = {
  0: '#4CAF50',
  1: '#FFC107',
  2: '#F44336',
};

export const CLASS_DESCRIPTIONS: Record<ClassId, string> = {
  0: 'No signs of retinopathy of prematurity detected.',
  1: 'Early ROP changes present - monitor closely and schedule follow-up.',
  2: 'Severe ROP detected - requires urgent ophthalmologic intervention.',
};

export interface ModelMetricsResponse {
  metrics: MetricData;
  confusionMatrix: ConfusionMatrixData;
  classPerformance: ClassPerformance[];
  modelInfo: ModelInfo;
}

export interface PredictionHistoryRecord extends PredictionResult {}
