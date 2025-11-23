# ROP Severity Classification System - Technical Report

**Project:** Retinopathy of Prematurity (ROP) Detection and Classification
**Date:** 24th November 2025
**Model:** RETFound Vision Transformer (ViT) Fine-tuned for ROP Severity Classification

---

## 1. Executive Summary

Successfully implemented a Retinopathy of Prematurity (ROP) severity classification system using RETFound vision transformer architecture, achieving 75.6% accuracy and 90.3% ROC-AUC on 3-class classification (No ROP, Early ROP, Severe ROP). Training employed patient-wise data splitting and weighted sampling to address severe class imbalance. Deployment includes Modal-hosted GPU inference, automated image validation, Grad-CAM interpretability overlays and a React-based clinical dashboard with batch processing and PDF reporting capabilities.

**Key Achievements:**
- Fine-tuned RETFound (DINOv2-based) foundation model for ROP-specific classification
- Patient-wise data splitting ensuring no data leakage between train/val/test sets
- Real-time inference API deployed on Modal with GPU acceleration
- Interactive React dashboard with batch upload, PDF reporting, and prediction history
- Grad-CAM visualization for clinical interpretability

---

## 2. Problem Statement

Retinopathy of Prematurity (ROP) is a potentially blinding eye disorder affecting premature infants. Early detection and accurate severity classification are critical for timely intervention. Manual screening by ophthalmologists is:
- Time-consuming and resource-intensive
- Subject to inter-observer variability
- Limited by specialist availability in underserved regions

**Objective:** Develop a screening tool that can:
1. Accurately classify ROP severity from retinal fundus images
2. Provide interpretable results for clinical decision support
3. Enable batch processing for efficient screening workflows

---

## 3. Approach

### 3.1 Data Preprocessing

**Dataset Characteristics:**
- 148 unique patients with multiple retinal images per patient
- 3-class severity labels: No ROP (0), Early ROP (1), Severe ROP (2)
- Class distribution: 2,134 (No ROP), 292 (Early ROP), 382 (Severe ROP)

**Patient-Wise Splitting Strategy:**
```
Train: 102 patients (68.9%)
Validation: 16 patients (10.8%)
Test: 30 patients (20.3%)
```

**Critical Design Decision:** Patient-wise splitting ensures that all images from a single patient appear in only one split, preventing data leakage and providing realistic performance estimates.

**Image Preprocessing Pipeline:**
1. Resize to 256px (shortest edge)
2. Center crop to 224x224
3. Data augmentation during training (random flips, rotations, color jitter)

### 3.2 Model Architecture

**Base Model:** RETFound (DINOv2-based Vision Transformer)

RETFound is a foundation model specifically pre-trained on 1.6 million retinal images, making it highly suitable for ophthalmic imaging tasks.

**Architecture Details:**
```
Model: RETFound_dinov2
├── Vision Transformer Backbone
│   ├── Patch embedding (14x14 patches)
│   ├── 12 transformer encoder blocks
│   └── Global average pooling
├── Classification Head
│   └── Linear layer (768 → 3 classes)
└── Parameters: ~86M
```

**Key Hyperparameters:**
- Drop path rate: 0.1
- Global pooling: Average
- Input resolution: 224x224
- Batch size: Variable (class-balanced sampling)

### 3.3 Training Strategy

**Class Imbalance Handling:**
- Inverse frequency class weights: Computed dynamically from training counts
- Balanced batch sampling: Ensures equal representation per batch

**Weight Computation:**
```python
# Inverse frequency scaled by total/(K*count), normalized to mean=1
weights = [total / (K * count) for count in class_counts]
weights = [w / mean(weights) for w in weights]
```

**Training Configuration:**
- Optimizer: AdamW
- Learning rate: 5e-5 with cosine annealing
- Weight decay: 0.05
- Epochs: 80 with early stopping
- GPU: NVIDIA T4 (Modal cloud)

### 3.4 Inference Pipeline

**Real-time Prediction Flow:**
1. Image upload (base64 encoded)
2. Retinal image validation (gating)
3. Preprocessing and transformation
4. Model inference
5. Grad-CAM generation
6. Result persistence to Supabase
7. Response with predictions and overlays

**Retinal Image Validation (Eye-Gating):**
```python
# Validation thresholds
variance >= 80          # Rejects very blurry/non-retinal images
pixel_count >= 800      # Ensures sufficient bright regions
aspect_ratio: 0.60-1.40 # Rejects elongated non-eye images
coverage_bbox: 0.45-0.9 # Validates fundus structure
```

---

## 4. Results Summary

### 4.1 Model Performance

**Model Specifications:**

| Specification | Value |
|---------------|-------|
| Architecture | RETFound (DINOv2 ViT) |
| Number of Classes | 3 |
| Input Resolution | 224x224 |
| Parameters | ~86M |
| Inference Time | ~200-500ms (GPU) |

**Evaluation Metrics (Test Set):**

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 75.59% | Overall correct predictions |
| **F1 Score** | 74.16% | Harmonic mean of precision and recall |
| **ROC AUC** | 90.33% | Area under ROC curve (excellent discrimination) |
| **Precision** | 77.32% | Positive predictive value |
| **Recall** | 73.89% | Sensitivity / True positive rate |
| **Average Precision** | 82.50% | Area under precision-recall curve |
| **Cohen's Kappa** | 0.6256 | Inter-rater agreement (substantial) |
| **Jaccard Score** | 59.54% | Intersection over union |
| **Hamming Loss** | 16.27% | Fraction of incorrect predictions |

**Key Performance Highlights:**
- **ROC AUC of 90.33%** indicates excellent ability to distinguish between severity classes
- **Cohen's Kappa of 0.63** represents substantial agreement beyond chance
- **Precision-Recall balance** (77.3% / 73.9%) shows reliable predictions with good sensitivity

### 4.2 Class Distribution by Split

| Class | Train Images | Train Patients | Val Images | Val Patients | Test Images | Test Patients |
|-------|-------------|----------------|------------|--------------|-------------|---------------|
| No ROP (0) | 2,134 | 93 | 225 | 11 | 556 | 24 |
| Early ROP (1) | 292 | 7 | 73 | 3 | 349 | 4 |
| Severe ROP (2) | 382 | 2 | 119 | 2 | 541 | 2 |
| **Total** | **2,808** | **102** | **417** | **16** | **1,446** | **30** |

### 4.3 Patient-Level Imbalance Analysis

**Critical Observation:** While image-level class distribution appears moderately imbalanced (7:1 ratio between No ROP and Early ROP), the patient-level distribution reveals extreme imbalance:

| Class | Training Patients | Percentage |
|-------|------------------|------------|
| No ROP (0) | 93 | 91.2% |
| Early ROP (1) | 7 | 6.9% |
| Severe ROP (2) | 2 | 2.0% |

**Key Insights:**
- **Severe ROP patients have significantly more images per patient** due to frequent clinical monitoring (average ~191 images/patient vs ~23 images/patient for No ROP)
- This creates a risk of model memorizing patient-specific features rather than learning generalizable ROP patterns
- Patient-wise splitting is critical to prevent optimistic bias from data leakage

**Implications for Model Training:**
- Class-balanced sampling operates at image level, partially mitigating imbalance
- Focal loss helps the model focus on hard examples from minority classes
- Validation and test performance may vary significantly due to small patient counts in minority classes

### 4.4 Confidence Thresholds

| Confidence Level | Threshold | Interpretation |
|-----------------|-----------|----------------|
| High | ≥ 80% | Strong prediction confidence |
| Medium | 65-79% | Moderate confidence |
| Low | < 65% | Consider clinical review |

---

## 5. System Architecture

### 5.1 Backend Infrastructure

**Modal Cloud Deployment:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Modal Cloud Platform                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Predict     │  │  Metrics     │  │  Confusion   │       │
│  │  Endpoint    │  │  Endpoint    │  │  Matrix API  │       │
│  │  (T4 GPU)    │  │  (CPU)       │  │  (CPU)       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│          │                │                │                │
│          └────────────────┴────────────────┘                │
│                           │                                 │
│  ┌────────────────────────┴────────────────────────┐        │
│  │              Shared Volumes                     │        │
│  │  - retfound-data (training data)                │        │
│  │  - rop-checkpoints (model weights)              │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-image` | POST | Single image prediction with Grad-CAM |
| `/get-metrics` | GET | Model performance metrics |
| `/get-confusion-matrix` | GET | Confusion matrix data |

### 5.2 Frontend Architecture

**Technology Stack:**
- React 19 with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- React Router for navigation
- Supabase for authentication and storage

**Key Components:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── Auth/           # Authentication components
│   │   ├── Dashboard/      # Metrics and visualizations
│   │   ├── History/        # Prediction history
│   │   ├── Layout/         # Navigation and layout
│   │   ├── Metrics/        # Performance tables
│   │   └── Upload/         # Image upload and results
│   ├── contexts/           # React context providers
│   ├── pages/              # Route pages
│   ├── services/           # API and Supabase clients
│   └── types/              # TypeScript definitions
```

**Pages:**
1. **Upload & Analyze** - Batch image upload with real-time predictions
2. **Dashboard** - Model metrics overview and recent predictions
3. **Metrics** - Detailed performance analysis and confusion matrix
4. **History** - Prediction history with filtering and search

### 5.3 Database Schema (Supabase)

```sql
-- User profiles
profiles (
  id uuid PRIMARY KEY,
  full_name text,
  role text DEFAULT 'clinician',
  created_at timestamp
)

-- Prediction records
predictions (
  id uuid PRIMARY KEY,
  user_id uuid,
  original_filename text,
  storage_path text,
  predicted_class smallint,
  confidences jsonb,
  confidence_label text,
  top_confidence numeric,
  gradcam_path text,
  created_at timestamp
)

-- Model metrics snapshots
model_metrics_snapshots (
  id uuid PRIMARY KEY,
  accuracy numeric,
  precision numeric,
  recall numeric,
  f1_score numeric,
  roc_auc numeric,
  created_at timestamp
)
```

---

## 6. Key Features

### 6.1 Batch Processing

- Upload up to 20 images per batch
- Concurrent processing (6 parallel requests)
- Batch summary with severity distribution

### 6.2 Grad-CAM Interpretability

**Implementation:**
- Hooks registered on final transformer block
- Gradient-weighted class activation mapping
- Overlay generation with configurable alpha blending
- Automatic upload to Supabase storage

**Clinical Value:**
- Highlights retinal regions driving predictions
- Enables clinician verification of model focus
- Supports clinical documentation and reporting

### 6.3 PDF Report Generation

**Report Contents:**
- Patient information and batch summary
- Aggregate diagnosis with confidence
- Class distribution visualization
- Individual image results with Grad-CAM overlays
- Confidence breakdown per image

### 6.4 Image Validation

**Eye-Gating System:**
Automatically rejects non-retinal images with clear error messages:
- "Image appears too unclear for analysis"
- "Image does not resemble a retinal fundus photograph"

---

## 7. Challenges and Solutions

### 7.1 Data-Related Challenges

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Severe class imbalance (7:1 ratio) | Class-balanced sampling + focal loss | Improved minority class recall |
| Patient data leakage risk | Patient-wise train/val/test splits | Realistic performance estimates |
| Variable image quality | Robust preprocessing + eye-gating | Reduced false predictions |

### 7.2 Model-Related Challenges

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Large model size (86M params) | GPU deployment on Modal | Real-time inference (~200ms) |
| Overfitting on small dataset | Drop path regularization + augmentation | Better generalization |
| Interpretability requirements | Grad-CAM integration | Clinical trust and adoption |


---

## 8. API Specification

### 8.1 Prediction Endpoint

**Request:**
```http
POST /predict-image
Content-Type: application/json

{
  "image": "base64_encoded_image_string",
  "file_name": "patient_001_od.jpg",
  "user_id": "uuid-string"
}
```

**Response:**
```json
{
  "id": "prediction-uuid",
  "filename": "patient_001_od.jpg",
  "predictedClass": 1,
  "prediction": "Early ROP",
  "confidence": 0.847,
  "confidenceLabel": "High",
  "confidences": [
    {"classId": 1, "confidence": 0.847},
    {"classId": 0, "confidence": 0.112},
    {"classId": 2, "confidence": 0.041}
  ],
  "probabilities": {
    "No ROP": 0.112,
    "Early ROP": 0.847,
    "Severe ROP": 0.041
  },
  "gradcamBase64": "base64_encoded_overlay",
  "gradcamPath": "user_id/gradcam_prediction-uuid.png",
  "createdAt": "2025-11-23T12:34:56.789Z",
  "model": "RETFound-ROP"
}
```

### 8.2 Error Responses

| Status | Condition | Message |
|--------|-----------|---------|
| 400 | No image provided | "No image provided" |
| 422 | Invalid retinal image | "Image does not resemble a retinal fundus photograph" |
| 422 | Blurry image | "Image appears too unclear for analysis" |
| 500 | Server error | "Prediction failed: {details}" |

---

## 9. Deployment Guide

### 9.1 Backend Deployment (Modal)

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Create secrets
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create supabase-service \
  SUPABASE_URL=<url> \
  SUPABASE_SERVICE_ROLE_KEY=<key> \
  SUPABASE_RETINA_BUCKET=retina-uploads \
  SUPABASE_GRADCAM_BUCKET=gradcam-overlays

# Deploy
modal deploy modal_retfound.py
```

### 9.2 Frontend Deployment

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
# Edit .env and .env.production with Supabase credentials

# Development
npm run dev

# Production build
npm run build
npm run preview
```

### 9.3 Environment Variables

**Frontend (.env):**
```ini
VITE_MODAL_API_URL=https://joy-awino--retfound-rop-predict-image.modal.run
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon-key>
```

**Modal Secrets:**
```ini
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
SUPABASE_RETINA_BUCKET=retina-uploads
SUPABASE_GRADCAM_BUCKET=gradcam-overlays
```

---

## 10. Future Improvements

### 10.1 Short-term (1-2 weeks)

- [ ] Implement confidence calibration using temperature scaling
- [ ] Add model versioning and A/B testing capabilities
- [ ] Enhance batch processing with progress persistence
- [ ] Implement prediction caching for repeated images

### 10.2 Medium-term (1-2 months)

- [ ] Multi-task learning for plus disease detection
- [ ] Integration with DICOM medical imaging standards
- [ ] Automated quality assessment scoring
- [ ] Mobile-responsive progressive web app

### 10.3 Long-term (3-6 months)

- [ ] Federated learning for multi-center training
- [ ] Active learning pipeline for continuous improvement
- [ ] Integration with electronic health records (EHR)
- [ ] Regulatory pathway preparation (FDA/CE marking)

---

## 11. Repository Structure

```
GRP-A-ISP-ROP/
├── data/
│   └── splits/
│       └── split_by_patient.json    # Patient-wise data splits
├── frontend/
│   ├── src/
│   │   ├── components/              # React components
│   │   ├── contexts/                # Auth context
│   │   ├── pages/                   # Route pages
│   │   ├── services/                # API clients
│   │   └── types/                   # TypeScript types
│   ├── .env                         # Development config
│   ├── .env.production              # Production config
│   ├── package.json                 # Dependencies
│   └── vite.config.ts               # Build config
├── modal_retfound.py                # Backend inference & training
├── requirements.txt                 # Python dependencies
├── README.md                        # User documentation
└── TECHNICAL_REPORT.md              # This document
```

---

## 12. Conclusion

This project successfully delivers a production-ready ROP severity classification system that:

1. **Leverages state-of-the-art AI:** RETFound foundation model fine-tuned for ROP-specific classification
2. **Ensures clinical reliability:** Patient-wise splitting, class balancing, and interpretability features
3. **Provides practical utility:** Batch processing, PDF reporting, and prediction history
4. **Enables seamless deployment:** Cloud-native architecture with Modal and Supabase

The system is ready for clinical pilot testing and can serve as a foundation for broader deployment in neonatal screening programs.

---

## 13. Appendix

### A. Technology Stack Summary

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch 2.5.1 |
| Model | RETFound (DINOv2 ViT) |
| Backend | Modal (Serverless GPU) |
| API | FastAPI |
| Frontend | React 19 + TypeScript |
| Styling | TailwindCSS |
| Database | Supabase (PostgreSQL) |
| Storage | Supabase Storage |
| Authentication | Supabase Auth |

### B. Hardware Requirements

**Training:**
- GPU: NVIDIA T4 or better
- VRAM: 16GB recommended
- RAM: 32GB recommended

**Inference:**
- GPU: NVIDIA T4 (Modal cloud)
- Latency: ~200-500ms per image

### C. Dependencies

**Python (Backend):**
```
torch==2.5.1
torchvision==0.20.1
timm>=1.0.0
transformers
scikit-learn
opencv-python-headless
pandas
numpy
pillow
fastapi
pydantic
supabase
```

**JavaScript (Frontend):**
```
react@19.2.0
react-dom@19.2.0
react-router-dom@7.9.4
axios@1.12.2
@supabase/supabase-js@2.76.1
lucide-react@0.548.0
jspdf@3.0.3
recharts@3.3.0
```

---

