import type {
  ClassId,
  ClassPerformance,
  ConfusionMatrixData,
  MetricData,
  ModelInfo,
  PredictionConfidence,
  PredictionResult,
} from '../types';
import { CLASS_COLORS, CLASS_IDS, CLASS_NAMES } from '../types';

export const MODEL_METRICS: MetricData = {
  accuracy: 0.7559,
  f1Score: 0.7416,
  rocAuc: 0.9033,
  kappa: 0.6256,
  precision: 0.7732,
  recall: 0.7389,
  hammingLoss: 0.1627,
  jaccardScore: 0.5954,
  averagePrecision: 0.825,
};

export const CONFUSION_MATRIX: ConfusionMatrixData = {
  labels: ['No ROP', 'Early ROP', 'Severe ROP'],
  normalized: [
    [0.91007, 0.08993, 0],
    [0.36676, 0.62464, 0.0086],
    [0.1793, 0.13863, 0.68207],
  ],
  raw: [
    [506, 50, 0],
    [128, 218, 3],
    [97, 75, 369],
  ],
};

export const CLASS_PERFORMANCE: ClassPerformance[] = [
  {
    classId: 0,
    className: CLASS_NAMES[0],
    precision: 0.7214,
    recall: 0.8381,
    f1Score: 0.7754,
    support: 556,
  },
  {
    classId: 1,
    className: CLASS_NAMES[1],
    precision: 0.5672,
    recall: 0.7135,
    f1Score: 0.632,
    support: 349,
  },
  {
    classId: 2,
    className: CLASS_NAMES[2],
    precision: 0.9834,
    recall: 0.6562,
    f1Score: 0.7871,
    support: 541,
  },
];

export const TEST_SET_DISTRIBUTION = {
  total: 556 + 349 + 541,
  byClass: {
    0: 556,
    1: 349,
    2: 541,
  } as Record<ClassId, number>,
};

export const MODEL_INFO: ModelInfo = {
  architecture: 'RETFound (DINOv2 ViT)',
  trainingEpochs: 100,
  bestEpoch: 26,
  lossFunction: 'Balanced Focal Loss',
  learningRate: 5e-6,
  batchSize: 12,
  checkpointPath:
    'https://modal.com/storage/joy-awino/main/rop-checkpoints/rop_advanced5/20251118_114953',
};

const CLASS_RULES: Array<{
  classId: ClassId;
  probability: number;
  confidenceRange: [number, number];
}> = [
  { classId: 0, probability: 0.4, confidenceRange: [0.75, 0.95] },
  { classId: 2, probability: 0.7, confidenceRange: [0.6, 0.85] }, // cumulative
  { classId: 1, probability: 1, confidenceRange: [0.65, 0.85] },
];

const OTHER_CONFIDENCE_RANGE: [number, number] = [0.05, 0.25];

const randomInRange = (min: number, max: number) =>
  Math.random() * (max - min) + min;

const pickClassByProbability = (): {
  classId: ClassId;
  confidenceRange: [number, number];
} => {
  const roll = Math.random();
  const rule = CLASS_RULES.find((item) => roll <= item.probability);
  return rule ?? CLASS_RULES[CLASS_RULES.length - 1];
};

const normalizeConfidences = (confidences: PredictionConfidence[]) => {
  const total = confidences.reduce((sum, item) => sum + item.confidence, 0);
  return confidences.map((item) => ({
    ...item,
    confidence: item.confidence / total,
  }));
};

export const getConfidenceLabel = (
  value: number
): 'High' | 'Medium' | 'Low' => {
  if (value >= 0.8) return 'High';
  if (value >= 0.65) return 'Medium';
  return 'Low';
};

const makeConfidenceMap = ({
  targetClass,
  targetConfidence,
}: {
  targetClass: ClassId;
  targetConfidence: number;
}): PredictionConfidence[] => {
  const confidences: PredictionConfidence[] = CLASS_IDS.map((classId) =>
    classId === targetClass
      ? { classId, confidence: targetConfidence }
      : {
          classId,
          confidence: randomInRange(
            OTHER_CONFIDENCE_RANGE[0],
            OTHER_CONFIDENCE_RANGE[1]
          ),
        }
  );

  return normalizeConfidences(confidences).sort(
    (a, b) => b.confidence - a.confidence
  );
};

export const createPredictionId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `pred-${Math.random().toString(36).slice(2, 10)}`;
};

export const generateMockPrediction = async (
  filename: string
): Promise<PredictionResult> => {
  const { classId, confidenceRange } = pickClassByProbability();
  const targetConfidence = randomInRange(
    confidenceRange[0],
    confidenceRange[1]
  );
  const confidences = makeConfidenceMap({
    targetClass: classId,
    targetConfidence,
  });

  const topConfidence = confidences[0]?.confidence ?? targetConfidence;

  return new Promise<PredictionResult>((resolve) => {
    const delay = 2000 + Math.random() * 600;
    setTimeout(() => {
      resolve({
        id: createPredictionId(),
        filename,
        predictedClass: classId,
        confidences,
        createdAt: new Date().toISOString(),
        confidenceLabel: getConfidenceLabel(topConfidence),
      });
    }, delay);
  });
};

export const getClassColor = (classId: ClassId) => CLASS_COLORS[classId];
