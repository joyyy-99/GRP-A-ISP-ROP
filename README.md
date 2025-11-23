[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/F63P1L7A)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20100702&assignment_repo_type=AssignmentRepo)
# ROP Severity Classification Pipeline

This repository provides a complete pipeline for preprocessing, training, evaluating, and inferring Retinopathy of Prematurity (ROP) severity from retinal fundus images. It includes scripts for data preparation, model training, evaluation, and metric extraction, leveraging PyTorch and EfficientNet architectures.

## Project Structure

```
GRP-A-ISP-ROP/
│
├── data/
│   ├── metadata/
│   │   ├── parsed.csv
│   │   └── parsed_severity.csv
│   └── splits/
│       └── split_by_patient.json
│
├── scripts/
│   ├── make_splits.py
│   ├── parse_and_merge.py
│   ├── sanity_loader.py
│   ├── train_baseline.py
│   ├── infer_batch.py
│   ├── eval_severity.py
│   ├── eval_confusion.py
│   └── derive_labels.py
│
├── src/
│   └── rop_pipeline/
│       ├── __init__.py
│       ├── dataset.py
│       ├── model.py
│       ├── parse_filenames.py
│       ├── splits.py
│       ├── transforms.py
│       └── validators.py
│
├── extract_metrics.py
├── modal_train.py
├── modal_retfound.py
├── requirements.txt
└── README.md
```

## Main Components

### Data Preparation

- **scripts/parse_and_merge.py**: Parses image filenames and merges with patient info to create a metadata CSV.
- **scripts/make_splits.py**: Splits patients into train/val/test sets and saves as JSON.
- **scripts/derive_labels.py**: Derives severity labels from diagnosis and plus-form columns.

### Dataset and Transforms

- **src/rop_pipeline/dataset.py**: PyTorch `Dataset` for loading images and labels from metadata.
- **src/rop_pipeline/transforms.py**: Image augmentation and normalization routines.

### Model

- **src/rop_pipeline/model.py**: EfficientNet-B0 classifier with a custom head for ROP severity prediction.

### Training and Evaluation

- **scripts/train_baseline.py**: Main training script supporting patient-wise splits, class imbalance, and progressive unfreezing.
- **scripts/eval_severity.py**: Evaluates model on a split, computes metrics and confusion matrix.
- **scripts/eval_confusion.py**: Similar to above, focused on validation set.
- **scripts/infer_batch.py**: Batch inference for all images of a patient, with aggregation.

### Utilities

- **src/rop_pipeline/parse_filenames.py**: Extracts metadata from image filenames.
- **src/rop_pipeline/splits.py**: Utilities for loading and applying patient splits.
- **src/rop_pipeline/validators.py**: Heuristics for checking image validity (e.g., fundus detection).

### Modal Integration

- **modal_train.py**: Runs training on Modal cloud with GPU and volume support.
- **modal_retfound.py**: Modal script for RETFound-based training (multi-GPU).
- **extract_metrics.py**: Extracts metrics from RETFound evaluation, parses logs and outputs, and visualizes training curves.

### Quick Start

1. **Install dependencies**:
	```
	pip install -r requirements.txt
	```

2. **Prepare metadata**:
	```
	python scripts/parse_and_merge.py --images_root <images_dir> --patient_info <info.csv/xlsx> --out_csv data/metadata/parsed.csv
	python scripts/derive_labels.py
	```

3. **Create splits**:
	```
	python scripts/make_splits.py
	```

4. **Train model**:
	```
	python scripts/train_baseline.py --csv data/metadata/parsed_severity.csv --split_json data/splits/split_by_patient.json
	```

5. **Evaluate**:
	```
	python scripts/eval_severity.py --csv data/metadata/parsed_severity.csv --split_json data/splits/split_by_patient.json --checkpoint <path_to_best.pt>
	```

6. **Extract metrics and visualize**:
	```
	modal run extract_metrics.py --action evaluate
	modal run extract_metrics.py --action visualize
	```

## Requirements

See `requirements.txt` for Python dependencies (PyTorch, torchvision, timm, pandas, numpy, scikit-learn, etc.).

## Notes

- All scripts assume images and metadata are organized as described above.
- Modal scripts require a Modal account and setup for cloud execution.
- For more details, see comments in each script and module.

## Supabase (Auth & Persistence) Setup

The new React dashboard relies on Supabase for authentication, Postgres storage, and file/object hosting. Follow these steps before enabling the frontend auth flow:

1. **Create a Supabase project**  
   - Visit [supabase.com](https://supabase.com), create a project, and note the *Project URL* and *Anon API Key*.  
   - Add them to `frontend/.env` and `frontend/.env.production`:
     ```ini
     VITE_SUPABASE_URL=https://<your-project>.supabase.co
     VITE_SUPABASE_ANON_KEY=<anon-public-key>
     ```

2. **Provision storage buckets**  
   - In the Supabase dashboard, create two buckets (both public access can be left off if you plan to serve files via signed URLs):
     - `retina-uploads` – raw images uploaded by clinicians.
     - `gradcam-overlays` – Grad-CAM heatmap overlays generated by the Modal backend.

3. **Create database tables**  
   Execute the SQL below in the Supabase SQL editor to create the base schema.
   ```sql
   -- Users are managed by Supabase Auth; profiles holds app-specific fields.
   create table if not exists profiles (
     id uuid primary key references auth.users(id) on delete cascade,
     full_name text,
     role text default 'clinician',
     created_at timestamp with time zone default now()
   );

   -- Stores individual prediction events and metadata.
   create table if not exists predictions (
    id uuid primary key default gen_random_uuid(),
     user_id uuid references auth.users(id) on delete cascade,
     original_filename text not null,
     storage_path text not null,
     pred icted_class smallint not null,
     confidences jsonb not null,
     confidence_label text not null,
     top_confidence numeric not null,
     gradcam_path text,
     created_at timestamp with time zone default now()
   );

   -- Optional: snapshot of model metrics served on the dashboard.
   create table if not exists model_metrics_snapshots (
     id uuid primary key default gen_random_uuid(),
     accuracy numeric,
     precision numeric,
     recall numeric,
     f1_score numeric,
     roc_auc numeric,
     kappa numeric,
     hamming_loss numeric,
     jaccard_score numeric,
     average_precision numeric,
     source text,
     created_at timestamp with time zone default now()
   );
   ```

4. **Configure Row Level Security (RLS)**  
   Enable RLS on `profiles`, `predictions`, and `model_metrics_snapshots`, then add policies such as:
   ```sql
   alter table profiles enable row level security;
   alter table predictions enable row level security;
   alter table model_metrics_snapshots enable row level security;

   create policy "Profiles are readable by owner"
     on profiles for select
     using (auth.uid() = id);

   create policy "Profiles are editable by owner"
     on profiles for insert with check (auth.uid() = id)
     using (auth.uid() = id);

   create policy "Users can manage their predictions"
     on predictions
     using (auth.uid() = user_id)
     with check (auth.uid() = user_id);

   create policy "Metrics readable by authenticated users"
     on model_metrics_snapshots for select
     using (auth.role() = 'authenticated');
   ```

5. **Frontend configuration**  
   The dashboard uses `frontend/src/services/supabaseClient.ts` to initialise the Supabase client. Ensure the environment variables are set before running `npm run dev` or building the app.

6. **Clinician registration flow**  
- **Modal deployment variables**  
- **Automatic retina validation**  
  The prediction endpoint returns a 422 error when an upload looks non-retinal (very low variance), so the UI can prompt the clinician to try a clearer image.
  When deploying `modal_retfound.py`, create a Modal secret (for example `supabase-service`) that exposes:
  ```
  SUPABASE_URL=<your project url>
  SUPABASE_SERVICE_ROLE_KEY=<service role key>
  SUPABASE_RETINA_BUCKET=retina-uploads
  SUPABASE_GRADCAM_BUCKET=gradcam-overlays
  ```
  Attach this secret to the `ROPPredictor` class so inference containers can write to Supabase.
   Users can self-register from the `/login` screen (which now supports both sign-in and sign-up). If email confirmation is enabled in Supabase, new clinicians must click the verification link before their first sign-in. Once confirmed, a corresponding row is inserted or updated in the `profiles` table automatically.

This backbone will power authentication, prediction persistence, and interpretability assets once the remaining steps in the roadmap are implemented.
