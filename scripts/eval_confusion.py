#!/usr/bin/env python3
"""
Evaluation script for ROP severity classification.
Runs inference on validation set and provides comprehensive metrics including confusion matrix.
"""

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.rop_pipeline.dataset import ROPDataset
from src.rop_pipeline.model import EffNetClassifier
from src.rop_pipeline.splits import load_patient_split, split_dataframe_by_patients


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ROP severity classifier on validation set")
    
    parser.add_argument("--csv", type=str, default="data/metadata/parsed_severity.csv",
                       help="Path to CSV metadata file")
    parser.add_argument("--split_json", type=str, default="data/splits/split_by_patient.json",
                       help="Path to patient split JSON file")
    parser.add_argument("--label_col", type=str, default="severity_bin",
                       help="Label column name")
    parser.add_argument("--ckpt", type=str, default="experiments/runs/severity3_288/best.pt",
                       help="Path to model checkpoint")
    
    return parser.parse_args()


def get_num_classes_from_csv(csv_path: str, label_col: str) -> int:
    """Get number of classes from CSV file."""
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    return len(df[label_col].unique())


def load_model(ckpt_path: str, num_classes: int, device: torch.device) -> EffNetClassifier:
    """Load model from checkpoint."""
    model = EffNetClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(model: EffNetClassifier, val_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Run inference on validation set and return predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running inference on validation set...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    return np.array(all_preds), np.array(all_labels), all_probs


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive metrics."""
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics as dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }


def print_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Print confusion matrix in a readable text grid format."""
    print("\nConfusion Matrix:")
    print("=" * (len(class_names) * 8 + 10))
    
    # Header
    header = "True\\Pred".ljust(10)
    for name in class_names:
        header += f"{name}".rjust(8)
    print(header)
    print("-" * (len(class_names) * 8 + 10))
    
    # Rows
    for i, true_class in enumerate(class_names):
        row = f"{true_class}".ljust(10)
        for j in range(len(class_names)):
            row += f"{cm[i, j]}".rjust(8)
        print(row)
    
    print("=" * (len(class_names) * 8 + 10))


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print comprehensive metrics summary."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 60)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<15} "
              f"{class_metrics['precision']:<10.4f} "
              f"{class_metrics['recall']:<10.4f} "
              f"{class_metrics['f1']:<10.4f} "
              f"{class_metrics['support']:<10}")
    
    print("-" * 60)
    
    # Print confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print_confusion_matrix(cm, metrics['class_names'])


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.csv)
    
    # Load patient splits
    split = load_patient_split(args.split_json)
    train_df, val_df, test_df = split_dataframe_by_patients(df, split)
    
    print(f"Validation patients: {len(split['val'])}, samples: {len(val_df)}")
    
    # Get number of classes and class names
    num_classes = get_num_classes_from_csv(args.csv, args.label_col)
    class_names = [str(i) for i in range(num_classes)]
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, num_classes, device)
    
    # Create validation dataset and dataloader
    val_dataset = ROPDataset(args.csv, train=False, label_col=args.label_col, patient_ids=split['val'])
    
    # DataLoader settings (same as training script)
    num_workers = 0 if platform.system() == "Windows" else 4
    pin_memory = (device.type == "cuda")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Run inference
    y_pred, y_true, y_probs = run_inference(model, val_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print results
    print_metrics_summary(metrics)
    
    # Save results
    output_path = Path("reports/val_confusion.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add additional info to metrics
    metrics['checkpoint'] = args.ckpt
    metrics['csv_file'] = args.csv
    metrics['label_column'] = args.label_col
    metrics['num_validation_samples'] = len(y_true)
    metrics['num_validation_patients'] = len(split['val'])
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

