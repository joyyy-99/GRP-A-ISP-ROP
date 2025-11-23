# ROP Severity Classification System

An AI-powered screening tool for Retinopathy of Prematurity (ROP) severity classification using RETFound vision transformer architecture.

## Overview

This system uses **RETFound**, a foundation model pre-trained on 1.6 million retinal images, fine-tuned for 3-class ROP severity classification:

- **No ROP** - Normal retina, no disease
- **Early ROP** - Mild to moderate ROP
- **Severe ROP** - Treatment-requiring ROP 

### Key Features

- **High Performance**: 75.6% accuracy, 90.3% ROC-AUC
- **Batch Processing**: Upload and analyze up to 20 images at once
- **Interpretability**: Grad-CAM visualizations highlight regions driving predictions
- **PDF Reports**: Generate clinical reports with predictions and overlays
- **Real-time Inference**: GPU-accelerated predictions (~200-500ms per image)

## Demo

### Upload & Analyze
Upload retinal fundus images for instant severity classification with confidence scores.

### Grad-CAM Visualization
See which retinal regions the model focuses on for each prediction.

### Batch Reports
Generate PDF reports with patient summaries and individual image analyses.

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Modal account (for backend deployment)
- Supabase account (for authentication & storage)

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
# Edit .env with your Supabase credentials:
# VITE_SUPABASE_URL=https://<project>.supabase.co
# VITE_SUPABASE_ANON_KEY=<your-anon-key>

# Run development server
npm run dev

# Or run production build (recommended for demos)
npm run demo
```

### Backend Deployment (Modal)

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Create required secrets
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create supabase-service \
  SUPABASE_URL=<url> \
  SUPABASE_SERVICE_ROLE_KEY=<key> \
  SUPABASE_RETINA_BUCKET=retina-uploads \
  SUPABASE_GRADCAM_BUCKET=gradcam-overlays

# Deploy
modal deploy modal_retfound.py
```

## Project Structure

```
GRP-A-ISP-ROP/
├── frontend/                    # React dashboard
│   ├── src/
│   │   ├── components/          # UI components
│   │   │   ├── Auth/            # Authentication
│   │   │   ├── Dashboard/       # Metrics display
│   │   │   ├── History/         # Prediction history
│   │   │   ├── Layout/          # Navigation
│   │   │   ├── Metrics/         # Performance tables
│   │   │   └── Upload/          # Image upload
│   │   ├── contexts/            # React contexts
│   │   ├── pages/               # Route pages
│   │   ├── services/            # API clients
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── splits/
│       └── split_by_patient.json   # Train/val/test splits
├── modal_retfound.py            # Backend inference & training
├── requirements.txt             # Python dependencies
├── TECHNICAL_REPORT.md          # Detailed technical documentation
└── README.md                    # This file
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 75.59% |
| F1 Score | 74.16% |
| ROC AUC | 90.33% |
| Precision | 77.32% |
| Recall | 73.89% |
| Cohen's Kappa | 0.6256 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-image` | POST | Image prediction with Grad-CAM |
| `/get-metrics` | GET | Model performance metrics |
| `/get-confusion-matrix` | GET | Confusion matrix data |

### Prediction Request

```json
POST /predict-image
{
  "image": "base64_encoded_image",
  "file_name": "retina_001.jpg",
  "user_id": "optional-user-id"
}
```

### Prediction Response

```json
{
  "id": "prediction-uuid",
  "predictedClass": 1,
  "prediction": "Early ROP",
  "confidence": 0.847,
  "confidenceLabel": "High",
  "probabilities": {
    "No ROP": 0.112,
    "Early ROP": 0.847,
    "Severe ROP": 0.041
  },
  "gradcamBase64": "base64_overlay_image",
  "model": "RETFound-ROP"
}
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| ML Model | RETFound (DINOv2 ViT) |
| ML Framework | PyTorch 2.5.1 |
| Backend | Modal (Serverless GPU) |
| API | FastAPI |
| Frontend | React 19 + TypeScript |
| Styling | TailwindCSS |
| Database | Supabase (PostgreSQL) |
| Authentication | Supabase Auth |

## Supabase Setup

### 1. Create Tables

```sql
-- User profiles
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  role TEXT DEFAULT 'clinician',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prediction records
CREATE TABLE predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  original_filename TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  predicted_class SMALLINT NOT NULL,
  confidences JSONB NOT NULL,
  confidence_label TEXT NOT NULL,
  top_confidence NUMERIC NOT NULL,
  gradcam_path TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2. Create Storage Buckets

- `retina-uploads` - For uploaded retinal images
- `gradcam-overlays` - For Grad-CAM visualization overlays

### 3. Enable Row Level Security

```sql
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can manage own predictions"
  ON predictions FOR ALL USING (auth.uid() = user_id);
```

## Environment Variables

### Frontend (.env)

```ini
VITE_MODAL_API_URL=https://your-modal-endpoint.modal.run
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon-key>
```

### Modal Secrets

```ini
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
SUPABASE_RETINA_BUCKET=retina-uploads
SUPABASE_GRADCAM_BUCKET=gradcam-overlays
HF_TOKEN=<huggingface-token>
```

## Development

### Running Locally

```bash
# Frontend
cd frontend
npm run dev          # Development with HMR
npm run demo         # Production build (stable, no reloads)

# Backend (requires Modal)
modal serve modal_retfound.py   # Local development
modal deploy modal_retfound.py  # Production deployment
```

### Building for Production

```bash
cd frontend
npm run build
npm run preview
```

## Documentation

- **[TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md)** - Comprehensive technical documentation including model architecture, training strategy, and evaluation metrics.

## License

This project was developed for academic purposes.

## Acknowledgments

- **RETFound** - Foundation model for retinal imaging
- **Modal** - Serverless GPU infrastructure
- **Supabase** - Backend-as-a-service platform
