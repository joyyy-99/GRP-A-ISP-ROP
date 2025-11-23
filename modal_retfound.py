#!/usr/bin/env python3
"""
RETFound training script for cleaned ROP dataset
- Adds Class-Balanced Focal Loss (toggle with USE_FOCAL=1)
- Allows dynamic class weights via TRAIN_COUNTS env
- Balanced Batch Sampler (toggle with USE_SAMPLER=1)
- Web endpoints for predictions and metrics

This version reads local env vars in the @app.local_entrypoint (main)
and passes them as explicit arguments to the remote train_retfound function.

[PATCH v13] CORRECTS:
1.  Fixed insertion point for sampler block - now inserts BEFORE dataset_train line
2.  Added web endpoints for inference API
3.  Fixed class indentation for ROPPredictor
4.  Fixed numpy concatenate dimension mismatch by flattening border arrays
"""

import os
import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path
import re
import io
import base64
import uuid
import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import modal
import numpy as np
import cv2

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# --------------------------------------------------------------------------------------
# Build the runtime image
# --------------------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        [
            "timm>=1.0.0",
            "einops",
            "transformers",
            "scikit-learn",
            "opencv-python-headless",
            "pandas",
            "tqdm",
            "huggingface_hub",
            "numpy",
            "pillow",
            "scipy",
            "matplotlib",
            "seaborn",
            "pycm",
            "tensorboard",
            "fastapi",
            "pydantic",
            "supabase",
        ]
    )
    .run_commands(
        ["git clone https://github.com/rmaphoh/RETFound.git /workspace/RETFound"]
    )
    .env({"HF_HOME": "/workspace/hf_cache", "CUDA_VISIBLE_DEVICES": "0"})
)

app = modal.App("retfound-rop", image=image)

DATA_VOL = modal.Volume.from_name("retfound-data", create_if_missing=False)
CKPT_VOL = modal.Volume.from_name("rop-checkpoints", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")
SUPABASE_SECRET = modal.Secret.from_name("supabase-service")

# Default counts (can be overridden with TRAIN_COUNTS="2134,292,382")
# Milder weights for better minority class learning
DEFAULT_TRAIN_COUNTS = (2134, 292, 382)


def _compute_weights_from_counts(counts):
    # inverse frequency scaled by total/(K*count) then normalized near mean=1
    total = float(sum(counts))
    k = float(len(counts))
    raw = [total / (k * float(c)) for c in counts]
    mean = sum(raw) / k
    return [round(w / mean, 3) for w in raw]


# --------------------------------------------------------------------------------------
# WEB API - Prediction Model Class
# --------------------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    volumes={"/workspace/checkpoints": CKPT_VOL,},
    secrets=[HF_SECRET, SUPABASE_SECRET],
    scaledown_window=300,
    image=image,
)
class ROPPredictor:
    """Model inference class for ROP predictions"""

    class _GradCAMWrapper:
        """Lightweight Grad-CAM helper that attaches hooks to the target layer."""

        def __init__(self, model, target_layer):
            import torch

            self.model = model
            self.target_layer = target_layer
            self.activations = None
            self.gradients = None
            self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
            try:
                self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
            except AttributeError:
                # Fallback for older PyTorch versions
                self._backward_handle = target_layer.register_backward_hook(self._backward_hook)  # type: ignore[assignment]

            # Ensure model gradients are enabled during CAM computation
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None

        def _forward_hook(self, _module, _inputs, output):
            self.activations = output

        def _backward_hook(self, _module, _grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            self.gradients = grad

        def close(self):
            if hasattr(self, "_forward_handle") and self._forward_handle:
                self._forward_handle.remove()
            if hasattr(self, "_backward_handle") and self._backward_handle:
                self._backward_handle.remove()

        def generate(self, input_tensor, target_index: Optional[int], spatial_size: Tuple[int, int]):
            import torch

            if target_index is not None and target_index < 0:
                target_index = None

            with torch.enable_grad():
                logits = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                if target_index is None or target_index >= probabilities.shape[1]:
                    target_index = int(torch.argmax(probabilities, dim=1).item())

                score = logits[:, target_index]
                self.model.zero_grad(set_to_none=True)
                if input_tensor.grad is not None:
                    input_tensor.grad.zero_()
                score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            heatmap = self._build_heatmap(self.activations, self.gradients, spatial_size)

            return heatmap

        @staticmethod
        def _build_heatmap(activations, gradients, spatial_size: Tuple[int, int]):
            import torch

            if isinstance(activations, tuple):
                activations = activations[0]
            if isinstance(gradients, tuple):
                gradients = gradients[0]

            if activations.ndim == 4:
                # CNN-style activations: (B, C, H, W)
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=spatial_size,
                    mode="bilinear",
                    align_corners=False,
                )
                cam = cam[0, 0]
            elif activations.ndim == 3:
                # ViT-style activations: (B, tokens, dim)
                activations = activations[0]
                gradients = gradients[0]

                tokens = activations.shape[0]
                side = int(math.sqrt(tokens))

                # If tokens don't form a square, assume first token is class token and remove it
                if side * side != tokens:
                    print(f"[Grad-CAM] Removing class token: {tokens} -> {tokens - 1} tokens")
                    activations = activations[1:]
                    gradients = gradients[1:]
                    tokens = activations.shape[0]
                    side = int(math.sqrt(tokens))

                    if side * side != tokens:
                        raise ValueError(f"Cannot reshape {tokens} tokens into a square feature map even after removing class token.")

                weights = gradients.mean(dim=0)
                cam = torch.matmul(activations, weights)
                cam = cam.reshape(1, 1, side, side)
                cam = torch.relu(cam)
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=spatial_size,
                    mode="bilinear",
                    align_corners=False,
                )
                cam = cam[0, 0]
            else:
                raise ValueError(f"Unsupported activation dimensions for Grad-CAM: {activations.shape}")

            cam = cam.detach()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            return cam.cpu().numpy()

    def _find_gradcam_target_layer(self):
        """Heuristic to select a Grad-CAM compatible layer from the model."""
        model = getattr(self, "model", None)
        if model is None:
            return None

        blocks = getattr(model, "blocks", None)
        if isinstance(blocks, (list, tuple)) and blocks:
            return blocks[-1]

        if hasattr(model, "blocks") and getattr(model, "blocks", None):  # type: ignore[attr-defined]
            try:
                return model.blocks[-1]  # type: ignore[index]
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"Grad-CAM target from model.blocks failed: {exc}")

        for attr_name in ("layer4", "layers", "encoder", "stages"):
            try:
                candidate = getattr(model, attr_name, None)
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            if candidate is None:
                continue

            if isinstance(candidate, (list, tuple)) and candidate:
                return candidate[-1]
            if hasattr(candidate, "__len__") and len(candidate) > 0:  # type: ignore[arg-type]
                try:
                    return candidate[-1]  # type: ignore[index]
                except Exception:  # pylint: disable=broad-exception-caught
                    return candidate
            return candidate

        return None
    
    @modal.enter()
    def load_model(self):
        """Load the trained model when container starts"""
        import torch
        import sys

        sys.path.insert(0, '/workspace/RETFound')

        try:
            from models_vit import RETFound_dinov2  # type: ignore
        except ImportError:
            print("Warning: Could not import from models_vit, trying alternative import")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "models_vit",
                "/workspace/RETFound/models_vit.py",
            )
            models_vit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models_vit)
            RETFound_dinov2 = getattr(models_vit, "RETFound_dinov2", None)
            if RETFound_dinov2 is None:
                raise AttributeError("models_vit.RETFound_dinov2 not available")

        checkpoint_base = Path("/workspace/checkpoints/rop_advanced5")
        if not checkpoint_base.exists():
            raise RuntimeError(f"Checkpoint directory not found: {checkpoint_base}")

        checkpoint_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
        if not checkpoint_dirs:
            raise RuntimeError(f"No checkpoint directories found in {checkpoint_base}")

        latest_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint from: {latest_dir}")

        checkpoint_names = ["checkpoint-best.pth", "checkpoint.pth", "model_best.pth"]
        checkpoint_path = None
        for name in checkpoint_names:
            potential_path = latest_dir / name
            if potential_path.exists():
                checkpoint_path = potential_path
                break

        if not checkpoint_path:
            raise RuntimeError(f"No checkpoint file found in {latest_dir}")

        print(f"Loading model from: {checkpoint_path}")

        from types import SimpleNamespace

        model_args = SimpleNamespace(model='RETFound_dinov2', nb_classes=3)
        self.model = RETFound_dinov2(
            model_args,
            num_classes=3,
            drop_path_rate=0.1,
            global_pool='avg',
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        self.model.cuda()

        self.gradcam_target_layer = self._find_gradcam_target_layer()
        if self.gradcam_target_layer is None:
            print("Grad-CAM target layer could not be resolved; overlays will be unavailable.")

        from torchvision import transforms

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.class_names = ["No ROP", "Early ROP", "Severe ROP"]
        self.retina_bucket = os.environ.get("SUPABASE_RETINA_BUCKET", "retina-uploads")
        self.gradcam_bucket = os.environ.get("SUPABASE_GRADCAM_BUCKET", "gradcam-overlays")
        self.supabase_client = None

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_url and supabase_key:
            try:
                from supabase import create_client

                self.supabase_client = create_client(supabase_url, supabase_key)
                print("Supabase client initialised for persistence.")
            except Exception as exc:
                print(f"Failed to initialise Supabase client: {exc}")
                self.supabase_client = None
        else:
            print("Supabase credentials not provided; predictions will not be persisted.")

        print("Model loaded successfully.")
        print(f"  Classes: {self.class_names}")

    @modal.method()
    def predict(self, image_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        """Run prediction on image bytes"""
        import torch
        from PIL import Image

        metadata = metadata or {}

        clarity_error = (
            "Image appears to show a retina but is too unclear for analysis. "
            "Please upload a sharper fundus photograph."
        )
        structure_error = (
            "Uploaded image does not resemble a retinal fundus photograph. "
            "Please provide a clear retinal fundus image."
        )

        def raise_clarity() -> None:
            raise ValueError(clarity_error)

        def raise_structure() -> None:
            raise ValueError(structure_error)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            gray = np.asarray(image.convert('L'), dtype=np.float32)
            variance = float(gray.var())
            mean_intensity = float(gray.mean())

            if variance < 80 or mean_intensity < 35:
                raise_clarity()

            height_px, width_px = gray.shape
            mask_threshold = max(mean_intensity * 0.8, 95.0)
            mask = gray > mask_threshold
            ys, xs = np.where(mask)
            if len(xs) < 800:
                raise_structure()

            height = int(ys.max() - ys.min() + 1)
            width = int(xs.max() - xs.min() + 1)
            if height <= 0 or width <= 0:
                raise_structure()

            aspect_ratio = width / height
            coverage_bbox = mask.sum() / (height * width)
            coverage_full = mask.sum() / (height_px * width_px)

            if not (
                0.60 <= aspect_ratio <= 1.40
                and 0.45 <= coverage_bbox <= 0.9
                and 0.2 <= coverage_full <= 0.75
            ):
                raise_structure()

            border_thickness = max(int(min(height_px, width_px) * 0.12), 24)
            if border_thickness * 2 >= min(height_px, width_px):
                raise_clarity()

            center_region = gray[
                border_thickness : height_px - border_thickness,
                border_thickness : width_px - border_thickness,
            ]

            # DEBUG: v13 fix for concatenation - all arrays are flattened to 1D
            print(f"[DEBUG v13] Preparing border regions - image shape: {gray.shape}, border_thickness: {border_thickness}")
            top_border = gray[:border_thickness, :].flatten()
            bottom_border = gray[height_px - border_thickness :, :].flatten()
            left_border = gray[border_thickness : height_px - border_thickness, :border_thickness].flatten()
            right_border = gray[border_thickness : height_px - border_thickness, width_px - border_thickness :].flatten()

            print(f"[DEBUG v13] Border shapes - top: {top_border.shape}, bottom: {bottom_border.shape}, left: {left_border.shape}, right: {right_border.shape}")

            border_region = np.concatenate(
                [top_border, bottom_border, left_border, right_border],
                axis=0
            )
            print(f"[DEBUG v13] Concatenated border_region shape: {border_region.shape}")
            if border_region.size == 0 or center_region.size == 0:
                raise_structure()

            if float(border_region.mean()) > float(center_region.mean()) * 0.95:
                raise_clarity()

            input_tensor = self.transform(image).unsqueeze(0).cuda()

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            probs_dict = {
                self.class_names[i]: float(probabilities[0][i])
                for i in range(len(self.class_names))
            }

            confidences = [
                {"classId": i, "confidence": float(probabilities[0][i])}
                for i in range(len(self.class_names))
            ]
            confidences.sort(key=lambda item: item["confidence"], reverse=True)

            predicted_class = int(predicted.item())
            confidence_value = float(confidence.item())
            prediction_id = metadata.get("prediction_id") or str(uuid.uuid4())
            filename = metadata.get("file_name") or f"retina-{prediction_id}.png"
            created_at = datetime.utcnow().isoformat()

            if confidence_value < 0.4:
                raise ValueError(
                    "Model is not confident in this prediction; please upload a clearer retinal image."
                )

            result: Dict[str, Any] = {
                "id": prediction_id,
                "filename": filename,
                "predictedClass": predicted_class,
                "confidences": confidences,
                "createdAt": created_at,
                "confidenceLabel": self._confidence_label(confidence_value),
                "prediction": self.class_names[predicted_class],
                "confidence": confidence_value,
                "probabilities": probs_dict,
                "model": "RETFound-ROP",
            }

            gradcam_payload: Dict[str, Any] = {}
            try:
                gradcam_payload = self.generate_gradcam(
                    image=image,
                    prediction_id=prediction_id,
                    predicted_class=predicted_class,
                    user_id=metadata.get("user_id"),
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"Grad-CAM overlay generation error: {exc}")

            if gradcam_payload:
                result.update(gradcam_payload)

            result = self._persist_prediction(
                image_bytes=image_bytes,
                metadata=metadata,
                result=result,
            )

            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            raise

    def _confidence_label(self, value: float) -> str:
        if value >= 0.8:
            return "High"
        if value >= 0.65:
            return "Medium"
        return "Low"

    def _extract_public_url(self, response: Any) -> Optional[str]:
        """Best-effort extraction of a Supabase public URL from varied response shapes."""
        if response is None:
            return None

        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, dict):
                return (
                    data.get("publicUrl")
                    or data.get("public_url")
                    or data.get("publicURL")
                )
            return (
                response.get("publicUrl")
                or response.get("public_url")
                or response.get("publicURL")
            )

        for attr in ("public_url", "publicUrl", "publicURL", "url"):
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, str):
                    return value
        return None

    def generate_gradcam(
        self,
        *,
        image: "PILImage",
        prediction_id: str,
        predicted_class: int,
        user_id: Optional[str] = None,
        alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """Produce a Grad-CAM overlay and upload it to Supabase if available."""
        if getattr(self, "model", None) is None:
            return {}

        if getattr(self, "transform", None) is None:
            print("Grad-CAM skipped: transform not initialised.")
            return {}

        if getattr(self, "gradcam_target_layer", None) is None:
            print("Grad-CAM skipped: target layer not resolved.")
            return {}

        import torch
        from PIL import Image as PilImage  # Local alias to avoid confusion

        if isinstance(image, PilImage.Image):
            original = image.copy()
        else:
            try:
                original = PilImage.open(io.BytesIO(image)).convert("RGB")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"Grad-CAM skipped: unable to load image payload ({exc}).")
                return {}

        try:
            tensor = self.transform(original)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Grad-CAM skipped: preprocessing failed ({exc}).")
            return {}

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_tensor = tensor.unsqueeze(0).to(device).detach()
        input_tensor.requires_grad_(True)

        gradcam_wrapper = self._GradCAMWrapper(self.model, self.gradcam_target_layer)
        try:
            heatmap = gradcam_wrapper.generate(
                input_tensor,
                predicted_class,
                (input_tensor.shape[-2], input_tensor.shape[-1]),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Grad-CAM generation failed: {exc}")
            return {}
        finally:
            gradcam_wrapper.close()

        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            alpha = 0.45

        resized_heatmap = cv2.resize(
            heatmap,
            (original.width, original.height),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap_uint8 = np.uint8(resized_heatmap * 255)
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_rgb = np.array(original.convert("RGB"))
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(color_map, alpha, original_bgr, 1.0 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_image = PilImage.fromarray(overlay_rgb)

        buffer = io.BytesIO()
        overlay_image.save(buffer, format="PNG")
        overlay_bytes = buffer.getvalue()

        gradcam_path = None
        gradcam_url = None
        gradcam_base64: Optional[str] = base64.b64encode(overlay_bytes).decode("utf-8")

        if self.supabase_client and user_id:
            filename = f"{prediction_id}-gradcam.png"
            storage_path = f"{user_id}/{filename}"
            try:
                self.supabase_client.storage.from_(self.gradcam_bucket).upload(
                    storage_path,
                    overlay_bytes,
                    {"content-type": "image/png", "upsert": "true"},
                )
                gradcam_path = storage_path
                url_response = self.supabase_client.storage.from_(self.gradcam_bucket).get_public_url(storage_path)
                gradcam_url = self._extract_public_url(url_response)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"Supabase Grad-CAM upload failed: {exc}")
        else:
            if not self.supabase_client:
                print("Grad-CAM upload skipped: Supabase client unavailable.")
            if not user_id:
                print("Grad-CAM upload skipped: missing user_id for storage path.")

        return {
            "gradcamPath": gradcam_path,
            "gradcamUrl": gradcam_url,
            "gradcamBase64": gradcam_base64,
            "gradcamAvailable": bool(gradcam_path or gradcam_base64),
        }

    def _persist_prediction(
        self,
        *,
        image_bytes: bytes,
        metadata: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.supabase_client is None:
            return result

        user_id = metadata.get("user_id")
        if not user_id:
            print("Supabase persistence skipped: missing user_id")
            return result

        filename = result.get("filename") or f"retina-{result['id']}.png"
        storage_path = f"{user_id}/{filename}"

        try:
            self.supabase_client.storage.from_(self.retina_bucket).upload(
                storage_path,
                image_bytes,
                {"content-type": "image/png"},
            )
            result["storagePath"] = storage_path
        except Exception as exc:
            print(f"Supabase upload failed: {exc}")
            result["storagePath"] = None
            storage_path = None

        db_payload = {
            "id": result["id"],
            "user_id": user_id,
            "original_filename": filename,
            "storage_path": storage_path,
            "predicted_class": result["predictedClass"],
            "confidences": result["confidences"],
            "confidence_label": result["confidenceLabel"],
            "top_confidence": result["confidences"][0]["confidence"] if result["confidences"] else 0,
            "gradcam_path": result.get("gradcamPath"),
        }

        try:
            response = self.supabase_client.table("predictions").insert(db_payload).execute()
            if getattr(response, "data", None):
                saved = response.data[0]
                result["id"] = saved.get("id", result["id"])
                if saved.get("gradcam_path"):
                    result["gradcamPath"] = saved.get("gradcam_path")
        except Exception as exc:
            print(f"Supabase insert failed: {exc}")

        return result


# --------------------------------------------------------------------------------------
# WEB ENDPOINTS
# --------------------------------------------------------------------------------------

@app.function(image=image)
@modal.asgi_app()
def predict_image():
    """
    Main prediction endpoint - FastAPI app
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (for development)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    class PredictionRequest(BaseModel):
        image: str
        file_name: Optional[str] = None
        user_id: Optional[str] = None
    
    @web_app.post("/")
    async def predict(request: PredictionRequest):
        """
        Expects JSON: 
        {
            "image": "base64_encoded_image_string"
        }
        
        Returns:
        {
            "prediction": "class_name",
            "confidence": 0.95,
            "probabilities": {"class_0": 0.95, "class_1": 0.03, "class_2": 0.02},
            "timestamp": "2025-10-26T12:34:56",
            "model": "RETFound-ROP"
        }
        """
        try:
            # Extract and decode image
            image_base64 = request.image
            if not image_base64:
                raise HTTPException(status_code=400, detail="No image provided")

            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]

            image_bytes = base64.b64decode(image_base64)

            predictor = ROPPredictor()
            metadata = {
                k: v
                for k, v in {
                    'file_name': request.file_name,
                    'user_id': request.user_id,
                }.items() if v
            }
            result = predictor.predict.remote(image_bytes, metadata)

            return result

        except ValueError as ve:
            raise HTTPException(
                status_code=422,
                detail=str(ve),
            ) from ve
        except Exception as e:
            message = str(e)
            if 'low variance' in message.lower():
                raise HTTPException(
                    status_code=422,
                    detail="Image appears not to be a retinal fundus photo. Please upload a clear retinal image.",
                )
            raise HTTPException(status_code=500, detail=f"Prediction failed: {message}")
    
    return web_app


@app.function(
    volumes={"/workspace/checkpoints": CKPT_VOL},
    image=image,
)
@modal.asgi_app()
def get_metrics():
    """
    Get model performance metrics - FastAPI app
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI()
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    async def metrics():
        """Returns metrics from the latest checkpoint"""
        try:
            import pandas as pd
            
            # Find latest checkpoint
            checkpoint_base = Path("/workspace/checkpoints/rop_advanced5")
            if not checkpoint_base.exists():
                raise HTTPException(status_code=404, detail="No checkpoints found")
            
            checkpoint_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                raise HTTPException(status_code=404, detail="No checkpoint directories found")
            
            latest_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
            
            # Look for metrics file
            metrics_files = [
                "metrics_test.csv",
                "metrics_val.csv", 
                "log.txt"
            ]
            
            for metrics_file in metrics_files:
                metrics_path = latest_dir / metrics_file
                if metrics_path.exists():
                    if metrics_file.endswith('.csv'):
                        df = pd.read_csv(metrics_path)
                        return {
                            "metrics": df.to_dict(orient='records'),
                            "source": str(metrics_file),
                            "checkpoint": latest_dir.name,
                        }
            
            # Try loading from config.json if available
            config_path = latest_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return {
                    "config": config,
                    "checkpoint": latest_dir.name,
                    "note": "Detailed metrics file not found"
                }
            
            raise HTTPException(status_code=404, detail="No metrics files found")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")
    
    return web_app


@app.function(
    volumes={"/workspace/checkpoints": CKPT_VOL},
    image=image,
)
@modal.asgi_app()
def get_confusion_matrix():
    """
    Get confusion matrix data - FastAPI app
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI()
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    async def confusion_matrix():
        """Returns confusion matrix from the latest checkpoint"""
        try:
            # Find latest checkpoint
            checkpoint_base = Path("/workspace/checkpoints/rop_advanced5")
            if not checkpoint_base.exists():
                raise HTTPException(status_code=404, detail="No checkpoints found")
            
            checkpoint_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                raise HTTPException(status_code=404, detail="No checkpoint directories found")
            
            latest_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
            
            # Look for confusion matrix image
            cm_image = latest_dir / "confusion_matrix_test.jpg"
            if cm_image.exists():
                # Return base64 encoded image
                with open(cm_image, 'rb') as f:
                    img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                return {
                    "confusion_matrix_image": f"data:image/jpeg;base64,{img_base64}",
                    "checkpoint": latest_dir.name,
                }
            
            # If no image, try to return raw data if available
            raise HTTPException(
                status_code=404, 
                detail=f"Confusion matrix image not found in {latest_dir.name}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load confusion matrix: {str(e)}")
    
    return web_app


@app.function(image=image)
@modal.asgi_app()
def health_check():
    """Simple health check endpoint - FastAPI app"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI()
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    async def health():
        return {
            "status": "healthy",
            "service": "retfound-rop-api",
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    return web_app


# --------------------------------------------------------------------------------------
# Patch 1: replace criterion with (a) weighted CE or (b) class-balanced focal loss
# This function is unchanged.
# --------------------------------------------------------------------------------------
@app.function(image=image)
def patch_retfound_for_class_weights(use_focal: bool, train_counts_csv: str):
    """Modify main_finetune.py to use weighted CrossEntropy or Class-Balanced Focal Loss."""
    main_file = "/workspace/RETFound/main_finetune.py"

    # If train_counts_csv is empty, skip patching (use default CE loss)
    if not train_counts_csv or train_counts_csv.strip() == "":
        print("TRAIN_COUNTS empty - skipping class weights patch, using default CE loss")
        return True

    # parse counts
    try:
        counts = tuple(int(x.strip()) for x in train_counts_csv.split(","))
        if len(counts) != 3:
            counts = DEFAULT_TRAIN_COUNTS
    except Exception:
        counts = DEFAULT_TRAIN_COUNTS

    weights = _compute_weights_from_counts(counts)  # normalized around 1.0

    with open(main_file, "r") as f:
        lines = f.readlines()

    patched = False
    for i, line in enumerate(lines):
        if "criterion =" in line and ("LabelSmoothing" in line or "CrossEntropy" in line):
            indent = len(line) - len(line.lstrip())
            spaces = " " * indent

            header = [
                f"{spaces}# ======= Auto-patched: Class-balanced loss (no env) =======\n",
                f"{spaces}import torch\n",
                f"{spaces}import torch.nn as nn\n",
                f"{spaces}import torch.nn.functional as F\n",
                f"{spaces}\n",
                f"{spaces}class BalancedFocalLoss(nn.Module):\n",
                f"{spaces}    def __init__(self, gamma=2.0, alpha=None):\n",
                f"{spaces}        super().__init__()\n",
                f"{spaces}        self.gamma = gamma\n",
                f"{spaces}        self.alpha = alpha\n",
                f"{spaces}\n",
                f"{spaces}    def forward(self, inputs, targets):\n",
                f"{spaces}        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)\n",
                f"{spaces}        pt = torch.exp(-ce)\n",
                f"{spaces}        loss = ((1 - pt) ** self.gamma) * ce\n",
                f"{spaces}        return loss.mean()\n",
                f"{spaces}\n",
                f"{spaces}class_weights = torch.tensor({weights}, dtype=torch.float32, device=args.device)\n",
            ]

            if use_focal:
                body = [
                    f"{spaces}criterion = BalancedFocalLoss(gamma=2.0, alpha=class_weights)\n",
                    f"{spaces}print(f'Using Balanced Focal Loss with class weights: {{class_weights.detach().cpu().tolist()}}')\n",
                ]
            else:
                body = [
                    f"{spaces}criterion = nn.CrossEntropyLoss(weight=class_weights)\n",
                    f"{spaces}print(f'Using Weighted CrossEntropy with class weights: {{class_weights.detach().cpu().tolist()}}')\n",
                ]

            tail = [f"{spaces}# ======= End auto-patch =======\n"]

            lines[i : i + 1] = header + body + tail
            patched = True
            break

    if not patched:
        print("Warning: Could not locate criterion to patch in main_finetune.py")
        return False

    with open(main_file, "w") as f:
        f.writelines(lines)

    print(
        f"Patched RETFound criterion. FOCAL={use_focal}  counts={counts}  weights={weights}"
    )
    return True


# --------------------------------------------------------------------------------------
# Patch 2: Balanced Batch Sampler for training DataLoader
# FIXED v12: Insert BEFORE dataset_train instead of AFTER to avoid return statement issues
# --------------------------------------------------------------------------------------
@app.function(image=image)
def patch_retfound_for_balanced_sampler(enable_sampler: bool):
    """
    Modify main_finetune.py to use WeightedRandomSampler on the TRAIN loader.
    If enable_sampler=False, do nothing (leave original dataloader).
    """
    if not enable_sampler:
        print("Sampler disabled by caller; skipping sampler patch.")
        return True

    main_file = "/workspace/RETFound/main_finetune.py"
    with open(main_file, "r") as f:
        txt = f.read()

    # 1) Ensure imports at the top of the file
    if "from torch.utils.data import WeightedRandomSampler" not in txt:
        # Find the first import statement
        import_match = re.search(r'^import ', txt, re.MULTILINE)
        if import_match:
            insert_pos = import_match.start()
            txt = txt[:insert_pos] + "from torch.utils.data import WeightedRandomSampler\n" + txt[insert_pos:]
        else:
            txt = "from torch.utils.data import WeightedRandomSampler\n" + txt

    # 2) Find where DataLoader for train is created and modify it
    # Strategy: Find the data_loader_train = DataLoader(...) block and replace it entirely
    
    # First, let's create a wrapper and function that will be inserted to build the sampler
    sampler_function = """
class WeightedRandomSamplerWrapper:
    \"\"\"Wrapper for WeightedRandomSampler that adds set_epoch method for compatibility\"\"\"
    def __init__(self, sampler):
        self.sampler = sampler
        
    def __iter__(self):
        return iter(self.sampler)
    
    def __len__(self):
        return len(self.sampler)
    
    def set_epoch(self, epoch):
        # WeightedRandomSampler doesn't need set_epoch, but we provide it for compatibility
        pass

def _build_balanced_sampler(dataset):
    \"\"\"Build WeightedRandomSampler for balanced batching\"\"\"
    try:
        _labels = None
        if hasattr(dataset, 'targets') and len(dataset.targets) > 0:
            _labels = [int(x) for x in dataset.targets]
            print(f"SAMPLER: Found {len(_labels)} labels in dataset.targets")
        elif hasattr(dataset, 'samples') and len(dataset.samples) > 0:
            _labels = [int(x[1]) for x in dataset.samples]
            print(f"SAMPLER: Found {len(_labels)} labels in dataset.samples")
        
        if _labels:
            import torch
            from torch.utils.data import WeightedRandomSampler
            labels_tensor = torch.tensor(_labels, dtype=torch.long)
            counts = torch.bincount(labels_tensor)
            class_weights = 1.0 / torch.clamp(counts.float(), min=1.0)
            sample_weights = class_weights[labels_tensor]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            print(f'SAMPLER SUCCESS: Using WeightedRandomSampler with counts={counts.tolist()}')
            return WeightedRandomSamplerWrapper(sampler)
        else:
            print("SAMPLER WARNING: Could not find labels")
            return None
    except Exception as e:
        print(f'SAMPLER ERROR: {e}')
        return None

"""

    # Insert the helper function before the main function
    main_func_match = re.search(r'def main\(', txt)
    if main_func_match:
        txt = txt[:main_func_match.start()] + sampler_function + txt[main_func_match.start():]
    else:
        print("Warning: Could not find main() function")
        return False

    # 3) Now modify the DataLoader creation
    # Find and replace data_loader_train - preserve indentation
    dl_train_pattern = r'(\s*)(data_loader_train\s*=\s*torch\.utils\.data\.DataLoader\s*\([^)]*dataset_train[^)]*\))'
    
    def replace_train_loader(match):
        indent = match.group(1)  # Capture the leading whitespace
        return f"""{indent}# Auto-patched: Use balanced sampler
{indent}train_sampler = _build_balanced_sampler(dataset_train)
{indent}data_loader_train = torch.utils.data.DataLoader(
{indent}    dataset_train,
{indent}    sampler=train_sampler,
{indent}    batch_size=args.batch_size,
{indent}    num_workers=args.num_workers,
{indent}    pin_memory=args.pin_mem,
{indent}    drop_last=True,
{indent}    shuffle=False  # sampler handles shuffling
{indent})"""
    
    txt = re.sub(dl_train_pattern, replace_train_loader, txt, count=1, flags=re.DOTALL)
    
    # Also ensure val loader doesn't shuffle - preserve indentation
    dl_val_pattern = r'(\s*)(data_loader_val\s*=\s*torch\.utils\.data\.DataLoader\s*\([^)]*dataset_val[^)]*\))'
    
    if re.search(dl_val_pattern, txt, re.DOTALL):
        def replace_val_loader(match):
            indent = match.group(1)  # Capture the leading whitespace
            return f"""{indent}data_loader_val = torch.utils.data.DataLoader(
{indent}    dataset_val,
{indent}    sampler=sampler_val,
{indent}    batch_size=args.batch_size,
{indent}    num_workers=args.num_workers,
{indent}    pin_memory=args.pin_mem,
{indent}    drop_last=False,
{indent}    shuffle=False
{indent})"""
        
        txt = re.sub(dl_val_pattern, replace_val_loader, txt, count=1, flags=re.DOTALL)
        print("Successfully patched both train and val DataLoaders.")
    else:
        print("Successfully patched train DataLoader (val loader not found).")

    with open(main_file, "w") as f:
        f.write(txt)

    return True


# --------------------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
    volumes={
        "/workspace/data": DATA_VOL,
        "/workspace/checkpoints": CKPT_VOL,
    },
    secrets=[HF_SECRET],
)
def train_retfound(
    use_focal: bool = False,
    use_sampler: bool = False,
    train_counts_csv: str = "2134,292,382",
):
    """
    Train RETFound on cleaned dataset.
    Args are passed from the local_entrypoint.
    """

    os.makedirs("/workspace/hf_cache", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not available")

    from huggingface_hub import login
    login(token=hf_token)

    # Subprocess env for RETFound run
    run_env = {
        **os.environ,
        "HF_TOKEN": hf_token,
        "HF_HOME": "/workspace/hf_cache",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    data_path = "/workspace/data/retfound"
    if not os.path.exists(data_path):
        raise RuntimeError(f"Data path {data_path} does not exist")

    print("Applying patch 1: Class Weights...")
    ok1 = patch_retfound_for_class_weights.local(use_focal, train_counts_csv)
    print(f"Patch 1 complete (Result: {ok1}).")
    
    print("Applying patch 2: Balanced Sampler...")
    ok2 = patch_retfound_for_balanced_sampler.local(use_sampler)
    print(f"Patch 2 complete (Result: {ok2}).")

    if not ok1:
        print("Warning: Loss patch failed, continuing with original loss...")
    
    if not ok2 and use_sampler:
        print("Warning: Sampler patch failed, continuing without sampler...")
    
    print("All patches applied. Starting training...")

    task_name = f"rop_advanced5_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=48766",
        "main_finetune.py",
        "--model",
        "RETFound_dinov2",
        "--model_arch",
        "retfound_dinov2",
        "--finetune",
        "RETFound_dinov2_meh",
        "--savemodel",
        "--global_pool",
        "--batch_size",
        "16",
        "--epochs",
        "80",  
        "--nb_classes",
        "3",
        "--data_path",
        data_path,
        "--input_size",
        "224",
        "--task",
        task_name,
        "--adaptation",
        "finetune",
        "--blr",
        "3e-5",  # Increased from 5e-6 to 3e-5
        "--lr",
        "3e-5",  # Increased from 5e-6 to 3e-5
        "--min_lr",
        "5e-7",  # Raised to maintain higher floor
        "--warmup_epochs",
        "12",  # Back to 12 to match shorter schedule
        # Regularization
        "--weight_decay",
        "0.04",  # Slightly reduced to preserve features
        "--drop_path",
        "0.12",
        # Keep minority-harming heavy augs off
        "--mixup",
        "0.0",
        "--cutmix",
        "0.0",
        "--reprob",
        "0.0",
        "--world_size",
        "1",
    ]

    print("\n" + "=" * 70)
    print("ROP TRAINING")
    print("=" * 70)
    print(f"Task: {task_name}")
    print("\nDataset:")
    print("  Train: C0=2134, C1=292, C2=382")
    print("  Val:   C0=225,  C1=73,  C2=119")
    print("  Test:  C0=556,  C1=349, C2=541")
    
    # Parse counts for logging
    if train_counts_csv and train_counts_csv.strip():
        try:
            log_counts = tuple(int(x.strip()) for x in train_counts_csv.split(","))
            if len(log_counts) != 3:
                log_counts = DEFAULT_TRAIN_COUNTS
        except Exception:
            log_counts = DEFAULT_TRAIN_COUNTS
        
        cw = _compute_weights_from_counts(log_counts)
        print("\nClass Weights (auto-balanced from counts):")
        print(f"  -> {cw}  (normalized)")
    else:
        print("\nClass Weights: NONE (using uniform weights)")
        cw = [1.0, 1.0, 1.0]
    
    print(f"Loss: {'Balanced Focal' if use_focal else 'Weighted CE' if (train_counts_csv and train_counts_csv.strip()) else 'Standard CE'}")
    print(f"Sampler: {'ON' if use_sampler else 'OFF'}")
    print(f"Train-counts CSV: {train_counts_csv if train_counts_csv else 'NONE (uniform)'}")
    print(f"\nHyperparameters:")
    print(f"  Epochs: 80")
    print(f"  Warmup: 12")
    print(f"  LR: 3e-5")
    print(f"  Min LR: 5e-7")
    print(f"  Weight Decay: 0.04")
    print(f"  Batch Size: 16")
    print("=" * 70 + "\n")

    try:
        result = subprocess.run(
            cmd,
            cwd="/workspace/RETFound",
            env=run_env,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Training failed")
        raise

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_dir = Path("/workspace/checkpoints/rop_advanced5") / timestamp
    dest_dir.mkdir(parents=True, exist_ok=True)

    retfound_out_dir = Path("/workspace/RETFound/output_dir") / task_name
    if retfound_out_dir.exists():
        print(f"\nSaving to {dest_dir}")
        for item in retfound_out_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_dir / item.name)
                print(f"  Copied {item.name}")

    config = {
        "version": "clean_class0",
        "task_name": task_name,
        "timestamp": timestamp,
        "gpu": "1x A100-40GB",
        "loss": "focal" if use_focal else "weighted_ce",
        "sampler": "on" if use_sampler else "off",
        "class_weights": str(_compute_weights_from_counts(log_counts)),
        "dataset_notes": "Class 0 cleaned",
        "train_counts": train_counts_csv,
        "imbalance_ratio": "Class 0/Class 1 ≈ 7.3x",
    }
    with open(dest_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Saved to: {dest_dir}")
    CKPT_VOL.commit()
    return str(dest_dir)


# --------------------------------------------------------------------------------------
# Verify data
# --------------------------------------------------------------------------------------
@app.function(image=image, volumes={"/workspace/data": DATA_VOL})
def verify_data():
    """Verify data structure"""
    data_path = "/workspace/data/retfound"

    if not os.path.exists(data_path):
        return False

    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            return False

        print(f"\n{split.upper()}:")
        for class_name in ["class_0", "class_1", "class_2"]:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                count = len(
                    [
                        f
                        for f in os.listdir(class_path)
                        if f.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                        )
                    ]
                )
                print(f"  {class_name}: {count} images")
            else:
                return False

    return True


# --------------------------------------------------------------------------------------
# Local entrypoint
# --------------------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Run training on dataset"""

    print("Verifying data...")
    if not verify_data.remote():
        print("\nData verification failed")
        return

    # Read from your LOCAL environment here
    use_focal_val = os.environ.get("USE_FOCAL", "0") == "1"
    use_sampler_val = os.environ.get("USE_SAMPLER", "0") == "1"
    train_counts_csv_val = os.environ.get("TRAIN_COUNTS", "2134,292,382")  # Milder weights

    print(f"\n--- Local settings detected ---")
    print(f"USE_FOCAL (raw: {os.environ.get('USE_FOCAL')}) -> {use_focal_val}")
    print(f"USE_SAMPLER (raw: {os.environ.get('USE_SAMPLER')}) -> {use_sampler_val}")
    print(f"TRAIN_COUNTS (raw: {os.environ.get('TRAIN_COUNTS')}) -> {train_counts_csv_val}")
    print("---------------------------------\n")

    print("Starting training...")
    # Pass them as arguments to the remote function
    output_dir = train_retfound.remote(
        use_focal=use_focal_val,
        use_sampler=use_sampler_val,
        train_counts_csv=train_counts_csv_val,
    )

    print(f"\nComplete! Artifacts: {output_dir}")
    print("\nUpdate extract_metrics.py checkpoint path to 'rop_advanced5'")
