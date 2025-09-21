#!/usr/bin/env python3
"""
Training script for ROP severity classification using EfficientNet-B0.
Implements patient-wise split, class imbalance handling, and progressive unfreezing.
"""

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

from src.rop_pipeline.dataset import ROPDataset
from src.rop_pipeline.model import EffNetClassifier, freeze_backbone
from src.rop_pipeline.splits import load_patient_split, split_dataframe_by_patients
from src.rop_pipeline.validators import is_probable_fundus


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ROP severity classifier")
    
    parser.add_argument("--csv", type=str, default="data/metadata/parsed_severity.csv",
                       help="Path to CSV metadata file")
    parser.add_argument("--split_json", type=str, default="data/splits/split_by_patient.json",
                       help="Path to patient split JSON file")
    parser.add_argument("--label_col", type=str, default="severity_bin",
                       help="Label column name")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=12,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--freeze_warmup", type=int, default=4,
                       help="Number of epochs to freeze backbone")
    parser.add_argument("--outdir", type=str, default="experiments/runs/baseline",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--images_root", type=str, 
                       default=r"C:\Users\joyaw\OneDrive\Desktop\archive\images_stack_without_captions\images_stack_without_captions",
                       help="Path to images directory")
    
    return parser.parse_args()


def create_weighted_sampler(df: pd.DataFrame, label_col: str) -> WeightedRandomSampler:
    """Create WeightedRandomSampler based on inverse class frequency."""
    # Get class counts
    class_counts = df[label_col].value_counts().sort_index()
    
    # Calculate inverse frequencies
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Create sample weights for each row
    sample_weights = [class_weights[class_counts.index.get_loc(label)] for label in df[label_col]]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )


def validate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Validate model and return metrics."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'val_loss': val_loss,
        'val_f1_macro': val_f1_macro
    }


def eye_gate_batch(images: torch.Tensor, device: torch.device) -> None:
    """Optional eye-gating: validate a single image from batch."""
    try:
        # Take first image from batch and convert to PIL for validation
        img_tensor = images[0].cpu()
        # Denormalize (assuming ImageNet normalization)
        img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to numpy array
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Quick validation (lightweight)
        if not is_probable_fundus(img_array):
            print("Warning: Potential non-fundus image detected in batch")
    except Exception:
        # Silently continue - don't let validation slow down training
        pass


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Eye-gating disabled during training (kept for inference only)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.csv)
    
    # Load patient splits
    split = load_patient_split(args.split_json)
    train_df, val_df, test_df = split_dataframe_by_patients(df, split)
    
    # Start-of-run summary
    print("\n" + "="*60)
    print("TRAINING SETUP SUMMARY")
    print("="*60)
    print(f"Train patients: {len(split['train'])}, samples: {len(train_df)}")
    print(f"Val patients: {len(split['val'])}, samples: {len(val_df)}")
    print(f"Test patients: {len(split['test'])}, samples: {len(test_df)}")
    print(f"Total patients: {len(split['train']) + len(split['val']) + len(split['test'])}")
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    
    # Log class distribution
    print("\nClass distribution:")
    train_class_counts = train_df[args.label_col].value_counts().sort_index()
    val_class_counts = val_df[args.label_col].value_counts().sort_index()
    print("Train classes:", train_class_counts.to_dict())
    print("Val classes:", val_class_counts.to_dict())
    print("="*60)
    
    # Create datasets
    train_dataset = ROPDataset(args.csv, train=True, label_col=args.label_col, patient_ids=split['train'], images_root=args.images_root)
    val_dataset = ROPDataset(args.csv, train=False, label_col=args.label_col, patient_ids=split['val'], images_root=args.images_root)
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_df, args.label_col)
    
    # Create data loaders
    # Use num_workers=0 on Windows for safety, otherwise use 4
    num_workers = 0 if platform.system() == "Windows" else 4
    pin_memory = (device.type == "cuda")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create model
    num_classes = len(train_dataset.class_to_idx)
    model = EffNetClassifier(num_classes=num_classes).to(device)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Create separate param groups for head and backbone
    head_params = list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())
    
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': 1e-5}
    ], weight_decay=1e-4)
    
    # Scheduler (applied after unfreezing)
    scheduler = None
    
    # Training loop
    best_val_f1 = 0.0
    metrics_history = []
    early_stop_patience = 6
    epochs_without_improvement = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Freeze warmup: {args.freeze_warmup} epochs")
    print(f"Early stopping patience: {early_stop_patience} epochs")
    print(f"Model classes: {num_classes}")
    
    for epoch in range(args.epochs):
        # Freeze/unfreeze backbone
        if epoch < args.freeze_warmup:
            freeze_backbone(model, freeze=True)
            print(f"Epoch {epoch+1}: Backbone frozen")
        else:
            freeze_backbone(model, freeze=False)
            if epoch == args.freeze_warmup:
                print(f"Epoch {epoch+1}: Backbone unfrozen")
                # Rebuild optimizer with both param groups active
                head_params = list(model.head.parameters())
                backbone_params = list(model.backbone.parameters())
                optimizer = optim.AdamW([
                    {'params': head_params, 'lr': args.lr},
                    {'params': backbone_params, 'lr': 1e-5}
                ], weight_decay=1e-4)
                
                # Initialize scheduler after unfreezing
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.freeze_warmup
                )
                print(f"  -> Optimizer rebuilt with head LR={args.lr}, backbone LR=1e-5")
                print(f"  -> CosineAnnealingLR scheduler initialized (T_max={args.epochs - args.freeze_warmup})")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = validate_model(model, val_loader, device)
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_f1_macro': val_metrics['val_f1_macro']
        }
        metrics_history.append(epoch_metrics)
        
        # Step scheduler if active
        if scheduler is not None:
            scheduler.step()
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            print(f"Epoch {epoch+1:2d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_f1_macro={val_metrics['val_f1_macro']:.4f}, "
                  f"LRs={[f'{lr:.2e}' for lr in current_lrs]}")
        else:
            print(f"Epoch {epoch+1:2d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_f1_macro={val_metrics['val_f1_macro']:.4f}")
        
        # Save best model and check early stopping
        if val_metrics['val_f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1_macro']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), outdir / "best.pt")
            print(f"  -> New best model saved (F1: {best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epochs")
            
        # Early stopping check
        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
            print(f"Best validation F1 macro: {best_val_f1:.4f}")
            break
    
    # Save final metrics
    final_metrics = {
        'best_val_f1_macro': best_val_f1,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'freeze_warmup': args.freeze_warmup,
        'num_classes': num_classes,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'history': metrics_history
    }
    
    with open(outdir / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation F1 macro: {best_val_f1:.4f}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
