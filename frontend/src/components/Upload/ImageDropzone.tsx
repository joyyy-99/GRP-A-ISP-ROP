import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, ImageIcon, AlertCircle } from 'lucide-react';

interface ImageDropzoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  error?: string | null;
  onError?: (message: string | null) => void;
  maxFiles?: number;
  allowMultiple?: boolean;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const DEFAULT_MAX_FILES = 20;

const ImageDropzone = ({
  onFilesSelected,
  disabled = false,
  error,
  onError,
  maxFiles = DEFAULT_MAX_FILES,
  allowMultiple = true,
}: ImageDropzoneProps) => {
  const handleDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (!acceptedFiles.length) {
        onError?.('Unable to read the selected file. Please try again.');
        return;
      }

      if (acceptedFiles.length > maxFiles) {
        onError?.(
          `Too many files selected. Please upload up to ${maxFiles} images per batch.`
        );
        return;
      }

      onError?.(null);
      onFilesSelected(acceptedFiles.slice(0, maxFiles));
    },
    [maxFiles, onError, onFilesSelected]
  );

  const handleRejected = useCallback(() => {
    onError?.(
      'Unsupported file. Please upload a JPEG or PNG image under 10MB.'
    );
  }, [onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    onDropRejected: handleRejected,
    multiple: allowMultiple,
    disabled,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png'],
    },
    maxSize: MAX_FILE_SIZE,
  });

  const stateClasses = (() => {
    if (disabled) return 'cursor-not-allowed opacity-60';
    if (error) return 'border-danger bg-danger/5';
    if (isDragActive) return 'border-primary bg-primary/5';
    return 'border-slate-200 bg-background hover:border-primary/60';
  })();

  const Icon = isDragActive ? ImageIcon : UploadCloud;

  return (
    <div
      {...getRootProps({
        className: `flex flex-col items-center justify-center rounded-3xl border-2 border-dashed px-8 py-16 text-center transition duration-200 focus-visible:outline-primary ${stateClasses}`,
      })}
    >
      <input {...getInputProps()} aria-label="Upload retinal image" />
      <span className="flex h-16 w-16 items-center justify-center rounded-full bg-white shadow-subtle">
        <Icon
          className={`h-10 w-10 ${
            error ? 'text-danger' : 'text-primary-dark'
          }`}
          strokeWidth={1.75}
        />
      </span>
      <p className="mt-6 text-lg font-semibold text-slate-900">
        Drag &amp; Drop {allowMultiple ? 'Images' : 'Image'}
      </p>
      <p className="mt-2 text-sm text-slate-500">or click to browse files</p>
      <p className="mt-4 text-xs uppercase tracking-[0.2em] text-slate-400">
        Supported: JPG, PNG - Max 10MB
      </p>
      {allowMultiple ? (
        <p className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-400">
          Upload up to {maxFiles} images per batch
        </p>
      ) : null}

      {error ? (
        <div className="mt-6 flex items-center gap-2 rounded-xl border border-danger/30 bg-white px-4 py-3 text-sm text-danger">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      ) : null}
    </div>
  );
};

export default ImageDropzone;
