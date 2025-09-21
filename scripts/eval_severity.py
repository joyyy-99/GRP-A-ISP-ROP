#!/usr/bin/env python3
"""
Evaluation script for ROP severity classification.
Runs inference on specified split and provides comprehensive metrics including confusion matrix.
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
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.rop_pipeline.dataset import ROPDataset
from src.rop_pipeline.model import EffNetClassifier
from src.rop_pipeline.splits import load_patient_split, split_dataframe_by_patients


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ROP severity classifier")
    
    parser.add_argument("--csv", type=str, default="/workspace/data/parsed_severity.csv",
                       help="Path to CSV metadata file")
    parser.add_argument("--split_json", type=str, default="/workspace/data/split_by_patient.json",
                       help="Path to patient split JSON file")
    parser.add_argument("--label_col", type=str, default="severity_bin",
                       help="Label column name")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--images_root", type=str, required=True,
                       help="Path to images directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Which split to evaluate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
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


def run_inference(model: EffNetClassifier, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Run inference on dataset and return predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"Running inference on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
    
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


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


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
    
    # Select the appropriate split
    if args.split == "train":
        split_df = train_df
        split_patients = split['train']
    elif args.split == "val":
        split_df = val_df
        split_patients = split['val']
    else:  # test
        split_df = test_df
        split_patients = split['test']
    
    print(f"{args.split.capitalize()} patients: {len(split_patients)}, samples: {len(split_df)}")
    
    # Get number of classes and class names
    num_classes = get_num_classes_from_csv(args.csv, args.label_col)
    class_names = [str(i) for i in range(num_classes)]
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, num_classes, device)
    
    # Create dataset and dataloader
    dataset = ROPDataset(args.csv, train=False, label_col=args.label_col, 
                        patient_ids=split_patients, images_root=args.images_root)
    
    # DataLoader settings
    num_workers = 0 if platform.system() == "Windows" else 4
    pin_memory = (device.type == "cuda")
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Run inference
    y_pred, y_true, y_probs = run_inference(model, data_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print results
    print_metrics_summary(metrics)
    
    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    report_path = checkpoint_dir / "report.json"
    cm_path = checkpoint_dir / "confusion_matrix.png"
    
    # Add additional info to metrics
    metrics['checkpoint'] = args.checkpoint
    metrics['csv_file'] = args.csv
    metrics['label_column'] = args.label_col
    metrics['split'] = args.split
    metrics['num_samples'] = len(y_true)
    metrics['num_patients'] = len(split_patients)
    
    # Save JSON report
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, str(cm_path))
    
    print(f"\nResults saved to:")
    print(f"  Report: {report_path}")
    print(f"  Confusion matrix: {cm_path}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
