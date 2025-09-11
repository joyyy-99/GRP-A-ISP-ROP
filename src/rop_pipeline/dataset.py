import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.rop_pipeline.transforms import build_transforms

class ROPDataset(Dataset):
    """Uses data/metadata/parsed.csv; labels from 'dg' column by default."""
    def __init__(self, csv_path: str, train: bool = True, label_col: str = "dg", patient_ids: list[str] | None = None):
        self.df = pd.read_csv(csv_path)
        if "error" in self.df.columns:
            self.df = self.df[self.df["error"].isna()].copy()
        self.df = self.df.dropna(subset=[label_col]).copy()
        self.df[label_col] = self.df[label_col].astype(int)
        
        # Normalize patient_id if present
        if "patient_id" in self.df.columns:
            self.df["patient_id"] = self.df["patient_id"].astype(str).str.zfill(3)
        
        # Filter by patient_ids if provided
        if patient_ids is not None:
            # Normalize provided patient_ids to 3-digit strings
            normalized_patient_ids = [str(pid).zfill(3) for pid in patient_ids]
            self.df = self.df[self.df["patient_id"].isin(normalized_patient_ids)].copy()
            if len(self.df) == 0:
                raise ValueError("No rows after patient filter")

        classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.label_col = label_col
        self.transforms = build_transforms(train=train)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transforms(img)
        y = self.class_to_idx[int(row[self.label_col])]
        meta = {
            "patient_id": row.get("patient_id"),
            "device": row.get("device"),
            "series": row.get("series"),
            "filepath": row["filepath"],
        }
        return img, y, meta
