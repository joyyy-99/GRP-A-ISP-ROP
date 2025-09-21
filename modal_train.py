#!/usr/bin/env python3
"""
Modal wrapper for ROP training.
Runs your existing scripts/train_baseline.py on a GPU using Modal volumes:

  rop-data         -> /workspace/data
    - scripts/ ...
    - src/ ...
    - parsed_severity.csv                
    - split_by_patient.json
    

  rop-checkpoints  -> /workspace/checkpoints
    - baseline_YYYYMMDD_HHMMSS/ (best.pt, metrics.json, classes.json, config.json)


"""

import os
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

import modal

# ---- Modal image with required deps (headless OpenCV avoids libGL errors) ----
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "pandas",
        "numpy",
        "scikit-learn",
        "opencv-python-headless",  
        "tqdm",
    )
)

# ---- Modal app ----
app = modal.App("rop-training", image=image)

# ---- Volumes (created if missing) ----
ROP_DATA = modal.Volume.from_name("rop-data", create_if_missing=True)
ROP_CKPT = modal.Volume.from_name("rop-checkpoints", create_if_missing=True)


@app.function(
    gpu="T4",                        # change to "A10G" or "A100" if available
    timeout=60 * 60 * 6,             # 6 hours
    volumes={
        "/workspace/data": ROP_DATA,
        "/workspace/checkpoints": ROP_CKPT,
    },
)
def train(
    csv_name: str = "parsed_severity.csv",            
    split_name: str = "split_by_patient.json",
    label_col: str = "severity_bin",                             
    epochs: int = 40,
    freeze_warmup: int = 5,
    outdir: str | None = None,
):
    """
    Runs scripts/train_baseline.py inside the container.
    """

    # Resolve in-volume paths
    csv_path = f"/workspace/data/{csv_name}"
    split_json = f"/workspace/data/{split_name}"

    # Timestamp in EAT (Africa/Nairobi)
    eat = ZoneInfo("Africa/Nairobi")
    ts = datetime.now(eat).strftime("%Y%m%d_%H%M%S")

    # Output folder in checkpoints volume
    if outdir is None:
        outdir = f"/workspace/checkpoints/baseline_{ts}"
    os.makedirs(outdir, exist_ok=True)

    # Ensure Python can import `src/...` when running from /workspace/data
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/data"

    # Build and run the training command as a module for clean imports
    cmd = [
        "python",
        "-m",
        "scripts.train_baseline",
        "--csv",        csv_path,
        "--split_json", split_json,
        "--label_col",  label_col,
        "--epochs",     str(epochs),
        "--freeze_warmup", str(freeze_warmup),
        "--outdir",     outdir,
        "--images_root", "/workspace/data/images_stack_without_captions/images_stack_without_captions",
    ]

    print("RUN:", " ".join(cmd), "cwd=/workspace/data", flush=True)
    subprocess.run(cmd, check=True, cwd="/workspace/data", env=env)
    print("DONE. Artifacts saved to:", outdir, flush=True)


if __name__ == "__main__":
    print("Use: modal run modal_train.py::train")


