#!/usr/bin/env python3
"""
Batch inference script for ROP severity classification.
Processes all images for a single patient with eye-gating and patient-level aggregation.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image

from src.rop_pipeline.dataset import ROPDataset
from src.rop_pipeline.model import EffNetClassifier
from src.rop_pipeline.validators import assert_fundus_or_raise
from src.rop_pipeline.transforms import build_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Infer ROP severity for a patient's images")
    
    parser.add_argument("--csv", type=str, default="data/metadata/parsed.csv",
                       help="Path to CSV metadata file")
    parser.add_argument("--ckpt", type=str, default="experiments/runs/baseline/best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--label_col", type=str, default="dg",
                       help="Label column name")
    parser.add_argument("--patient_id", type=str, required=True,
                       help="3-digit patient ID")
    parser.add_argument("--out_json", type=str, required=True,
                       help="Output JSON file path")
    
    return parser.parse_args()


def load_model(ckpt_path: str, num_classes: int, device: torch.device) -> EffNetClassifier:
    """Load model from checkpoint."""
    model = EffNetClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_num_classes_from_csv(csv_path: str, label_col: str) -> int:
    """Get number of classes from CSV file."""
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    return len(df[label_col].unique())


def process_patient_images(csv_path: str, patient_id: str, label_col: str, 
                          model: EffNetClassifier, device: torch.device) -> Dict[str, Any]:
    """Process all images for a patient and return predictions."""
    
    # Load and filter data for the patient
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    
    # Normalize patient_id and filter
    normalized_patient_id = str(patient_id).zfill(3)
    patient_df = df[df["patient_id"].astype(str).str.zfill(3) == normalized_patient_id].copy()
    
    if len(patient_df) == 0:
        raise ValueError(f"No images found for patient {normalized_patient_id}")
    
    # Get transforms (same as evaluation)
    transforms = build_transforms(train=False)
    
    # Get class mapping from the full dataset
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    image_preds = []
    all_probs = []
    
    print(f"Processing {len(patient_df)} images for patient {normalized_patient_id}...")
    
    for idx, row in patient_df.iterrows():
        try:
            # Load and validate image
            img_path = row["filepath"]
            img = Image.open(img_path).convert("RGB")
            
            # Eye-gate the image
            assert_fundus_or_raise(img)
            
            # Apply transforms
            img_tensor = transforms(img).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                pred_class_idx = torch.argmax(probs, dim=1).item()
                pred_class = idx_to_class[pred_class_idx]
                
                # Store probabilities for aggregation
                probs_np = probs.cpu().numpy().flatten()
                all_probs.append(probs_np)
                
                # Store per-image prediction
                image_pred = {
                    "filepath": img_path,
                    "predicted_class": pred_class,
                    "predicted_class_idx": pred_class_idx,
                    "probabilities": probs_np.tolist(),
                    "max_probability": float(probs_np.max())
                }
                image_preds.append(image_pred)
                
        except Exception as e:
            if "not a retinal fundus photo" in str(e):
                raise ValueError(f"Eye-gating failed for {img_path}: {e}")
            else:
                raise RuntimeError(f"Error processing {img_path}: {e}")
    
    # Calculate patient-level aggregate by averaging probabilities
    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        patient_pred_idx = np.argmax(avg_probs)
        patient_pred = idx_to_class[patient_pred_idx]
        
        patient_prediction = {
            "predicted_class": patient_pred,
            "predicted_class_idx": int(patient_pred_idx),
            "averaged_probabilities": avg_probs.tolist(),
            "max_probability": float(avg_probs.max()),
            "num_images": len(all_probs)
        }
    else:
        patient_prediction = {
            "predicted_class": None,
            "predicted_class_idx": None,
            "averaged_probabilities": None,
            "max_probability": None,
            "num_images": 0
        }
    
    return {
        "patient_id": normalized_patient_id,
        "image_preds": image_preds,
        "patient_pred": patient_prediction
    }


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get number of classes
    num_classes = get_num_classes_from_csv(args.csv, args.label_col)
    print(f"Number of classes: {num_classes}")
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, num_classes, device)
    
    # Process patient images
    try:
        results = process_patient_images(
            args.csv, 
            args.patient_id, 
            args.label_col, 
            model, 
            device
        )
        
        # Save results
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        num_images = results["patient_pred"]["num_images"]
        patient_class = results["patient_pred"]["predicted_class"]
        max_prob = results["patient_pred"]["max_probability"]
        
        print(f"\nInference completed!")
        print(f"Patient {args.patient_id}: {num_images} images processed")
        print(f"Patient-level prediction: Class {patient_class} (confidence: {max_prob:.4f})")
        print(f"Results saved to: {args.out_json}")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
