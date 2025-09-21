#!/usr/bin/env python3
"""
Script to derive severity labels from ROP diagnosis data.
Creates severity_bin column based on dg (diagnosis) and pf (plus-form) values.
"""

import pandas as pd
from pathlib import Path


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that required columns exist in the DataFrame."""
    required_columns = ['patient_id', 'dg', 'pf']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def derive_severity_bin(row: pd.Series) -> int:
    """
    Derive severity_bin based on dg and pf values.
    
    Rules:
    0 (Normal/Other): dg in {0, 9, 10, 11, 12, 13}
    1 (Early ROP): dg in {1, 2} or (dg == 3 and pf in {0, 1})
    2 (Severe / Type-1 ROP): dg in {4, 5, 6, 7, 8} or pf == 2
    """
    dg = row['dg']
    pf = row['pf']
    
    # 0 (Normal/Other): dg in {0, 9, 10, 11, 12, 13}
    if dg in {0, 9, 10, 11, 12, 13}:
        return 0
    
    # 2 (Severe / Type-1 ROP): dg in {4, 5, 6, 7, 8} or pf == 2
    elif dg in {4, 5, 6, 7, 8} or pf == 2:
        return 2
    
    # 1 (Early ROP): dg in {1, 2} or (dg == 3 and pf in {0, 1})
    elif dg in {1, 2} or (dg == 3 and pf in {0, 1}):
        return 1
    
    # Default case (should not happen with valid data)
    else:
        print(f"Warning: Unhandled case - dg={dg}, pf={pf}")
        return 0


def main():
    """Main function to derive severity labels."""
    # Input and output paths
    input_path = "data/metadata/parsed.csv"
    output_path = "data/metadata/parsed_severity.csv"
    
    print("Loading data...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")
    
    print(f"Loaded {len(df)} rows from {input_path}")
    
    # Validate required columns
    try:
        validate_columns(df)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Normalize patient_id to 3-digit strings
    print("Normalizing patient_id to 3-digit strings...")
    df['patient_id'] = df['patient_id'].astype(str).str.zfill(3)
    
    # Derive severity_bin column
    print("Deriving severity_bin labels...")
    df['severity_bin'] = df.apply(derive_severity_bin, axis=1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Print class counts and summary
    print("\n" + "="*50)
    print("SEVERITY LABEL DERIVATION SUMMARY")
    print("="*50)
    
    severity_counts = df['severity_bin'].value_counts().sort_index()
    severity_labels = {
        0: "Normal/Other",
        1: "Early ROP", 
        2: "Severe / Type-1 ROP"
    }
    
    print(f"Total rows processed: {len(df)}")
    print("\nSeverity class distribution:")
    for severity_bin, count in severity_counts.items():
        label = severity_labels.get(severity_bin, f"Unknown ({severity_bin})")
        percentage = (count / len(df)) * 100
        print(f"  {severity_bin} ({label}): {count:,} rows ({percentage:.1f}%)")
    
    # Check for any unhandled cases
    unique_dg_values = sorted(df['dg'].unique())
    unique_pf_values = sorted(df['pf'].unique())
    print(f"\nUnique dg values: {unique_dg_values}")
    print(f"Unique pf values: {unique_pf_values}")
    
    print(f"\nResults saved to: {output_path}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())

