import type { PredictionConfidence, PredictionResult } from '../types';
import { CLASS_IDS } from '../types';
import { supabase, isSupabaseConfigured } from './supabaseClient';
import { getConfidenceLabel } from './modelData';

const RETINA_BUCKET = 'retina-uploads';
const GRADCAM_BUCKET = 'gradcam-overlays';

const ensureAuth = async () => {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
  }
  const { data, error } = await supabase.auth.getSession();
  if (error) {
    throw new Error(error.message);
  }
  if (!data.session?.user) {
    throw new Error('User session not found.');
  }
  return data.session;
};

const uploadFileToBucket = async ({
  file,
  bucket,
  userId,
}: {
  file: File;
  bucket: string;
  userId: string;
}) => {
  const extension = file.name.split('.').pop() ?? 'jpg';
  const sanitizedName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
  const path = `${userId}/${Date.now()}-${sanitizedName}`;

  const { error } = await supabase.storage
    .from(bucket)
    .upload(path, file, {
      contentType: file.type || `image/${extension}`,
      upsert: false,
    });

  if (error) {
    throw new Error(`Failed to upload file: ${error.message}`);
  }

  return path;
};

const mapRecordToPrediction = (record: any): PredictionResult => {
  const rawConfidences: PredictionConfidence[] = Array.isArray(record.confidences)
    ? (record.confidences as PredictionConfidence[])
    : record.confidences?.map?.((item: any) => ({
        classId: item.classId ?? item.class_id,
        confidence: item.confidence ?? item.value ?? 0,
      })) ?? [];

  const sortedConfidences: PredictionConfidence[] = rawConfidences.length
    ? rawConfidences.slice().sort(
        (a: PredictionConfidence, b: PredictionConfidence) => b.confidence - a.confidence
      )
    : CLASS_IDS.map((classId) => ({ classId, confidence: classId === 0 ? 1 : 0 }));

  const topConfidence = sortedConfidences[0]?.confidence ?? 0;
  const confidenceLabel = record.confidence_label ?? getConfidenceLabel(topConfidence);
  const gradcamPath = record.gradcam_path ?? null;
  const gradcamUrl =
    gradcamPath && isSupabaseConfigured
      ? getStoragePublicUrl(GRADCAM_BUCKET, gradcamPath)
      : null;

  return {
    id: record.id,
    filename: record.original_filename,
    predictedClass: record.predicted_class,
    confidences: sortedConfidences,
    createdAt: record.created_at,
    confidenceLabel,
    storagePath: record.storage_path ?? undefined,
    gradcamPath,
    gradcamUrl,
    gradcamBase64: null,
    gradcamAvailable: Boolean(gradcamPath ?? gradcamUrl),
    topConfidence,
  };
};

export const savePredictionRecord = async (
  file: File,
  prediction: PredictionResult
): Promise<PredictionResult> => {
  const session = await ensureAuth();
  const user = session.user;

  let storagePath: string | undefined;
  try {
    storagePath = await uploadFileToBucket({
      file,
      bucket: RETINA_BUCKET,
      userId: user.id,
    });
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Retina image upload failed:', error);
  }

  const { data, error } = await supabase
    .from('predictions')
    .insert({
      user_id: user.id,
      original_filename: prediction.filename,
      storage_path: storagePath ?? null,
      predicted_class: prediction.predictedClass,
      confidences: prediction.confidences,
      confidence_label: prediction.confidenceLabel,
      top_confidence: prediction.topConfidence ?? prediction.confidences[0]?.confidence ?? 0,
      created_at: prediction.createdAt,
      gradcam_path: prediction.gradcamPath ?? null,
    })
    .select('*')
    .single();

  if (error) {
    throw new Error(`Failed to persist prediction: ${error.message}`);
  }

  return mapRecordToPrediction(data);
};

export const fetchRecentPredictions = async (limit = 5): Promise<PredictionResult[]> => {
  const session = await ensureAuth();
  const { data, error } = await supabase
    .from('predictions')
    .select('*')
    .eq('user_id', session.user.id)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) {
    throw new Error(`Unable to load recent predictions: ${error.message}`);
  }

  return (data ?? []).map(mapRecordToPrediction);
};

export const fetchPredictionHistory = async (): Promise<PredictionResult[]> => {
  const session = await ensureAuth();
  const { data, error } = await supabase
    .from('predictions')
    .select('*')
    .eq('user_id', session.user.id)
    .order('created_at', { ascending: false });

  if (error) {
    throw new Error(`Unable to load prediction history: ${error.message}`);
  }

  return (data ?? []).map(mapRecordToPrediction);
};

export const removePredictionRecord = async (id: string) => {
  const session = await ensureAuth();
  const { error } = await supabase
    .from('predictions')
    .delete()
    .eq('id', id)
    .eq('user_id', session.user.id);

  if (error) {
    throw new Error(`Unable to delete prediction: ${error.message}`);
  }
};

export const clearPredictionHistory = async () => {
  const session = await ensureAuth();
  const { error } = await supabase
    .from('predictions')
    .delete()
    .eq('user_id', session.user.id);

  if (error) {
    throw new Error(`Unable to clear prediction history: ${error.message}`);
  }
};

export const getStoragePublicUrl = (bucket: string, path: string) => {
  const { data, error } = supabase.storage.from(bucket).getPublicUrl(path);
  if (error) {
    // eslint-disable-next-line no-console
    console.warn('Supabase public URL lookup failed:', error.message);
    return null;
  }
  return data?.publicUrl ?? null;
};

export const retinaBucketName = RETINA_BUCKET;
export const gradcamBucketName = GRADCAM_BUCKET;
