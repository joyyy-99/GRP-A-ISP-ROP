import json
import pandas as pd
from typing import Dict, List, Tuple


def load_patient_split(json_path: str) -> Dict[str, List[str]]:
    """
    Load patient split from JSON file and return standardized format.
    
    Args:
        json_path: Path to JSON file containing patient splits
        
    Returns:
        Dictionary with keys "train", "val", "test" containing lists of 3-digit patient IDs
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Map the JSON keys to standardized keys
    split_mapping = {
        "train_patients": "train",
        "val_patients": "val", 
        "test_patients": "test"
    }
    
    result = {}
    for json_key, standard_key in split_mapping.items():
        if json_key in data:
            # Ensure all patient IDs are 3-digit strings
            patient_ids = [str(pid).zfill(3) for pid in data[json_key]]
            result[standard_key] = patient_ids
        else:
            result[standard_key] = []
    
    return result


def split_dataframe_by_patients(df: pd.DataFrame, split: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by patient IDs according to the provided split.
    
    Args:
        df: DataFrame with 'patient_id' column
        split: Dictionary with "train", "val", "test" keys containing lists of patient IDs
        
    Returns:
        Tuple of (train_df, val_df, test_df) filtered by patient IDs
    """
    # Ensure patient_id column exists
    if 'patient_id' not in df.columns:
        raise ValueError("DataFrame must contain 'patient_id' column")
    
    # Normalize patient_id to 3-digit strings
    df = df.copy()
    df['patient_id'] = df['patient_id'].astype(str).str.zfill(3)
    
    # Filter dataframes by patient IDs
    train_df = df[df['patient_id'].isin(split['train'])].copy()
    val_df = df[df['patient_id'].isin(split['val'])].copy()
    test_df = df[df['patient_id'].isin(split['test'])].copy()
    
    return train_df, val_df, test_df


# Export functions
__all__ = ["load_patient_split", "split_dataframe_by_patients"]
