"""
Brain MRI tumor classifier — ResNet-50 transfer learning.

Classes: glioma, meningioma, no_tumor, pituitary
Dataset: Kaggle Brain MRI Images for Brain Tumor Detection
         (Navoneel Chakrabarty, ~3060 images, 4 classes)

Model lifecycle:
  1. Train on Colab: ml_training/train_brain_mri.ipynb
  2. Notebook pushes model to HuggingFace Hub
  3. On first call to load(), downloads from HF and caches locally

Until the model is available, is_ready() returns False and the router
returns a 503 with a clear hint.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from backend.config import get_settings

log = structlog.get_logger()

# 4-class model trained on masoudnickparvar/brain-tumor-mri-dataset (/data/Training).
# Classes in alphabetical order as assigned by ImageFolder.
LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
MODEL_VERSION = "0.1.0"

LABEL_DESCRIPTIONS = {
    "glioma": (
        "Glioma — tumour arising from glial cells (astrocytes, oligodendrocytes). "
        "Ranges from low-grade (slow-growing) to high-grade (glioblastoma, aggressive)."
    ),
    "meningioma": (
        "Meningioma — tumour of the meninges (brain / spinal cord lining). "
        "Usually benign and slow-growing; rarely malignant."
    ),
    "notumor": "No tumour detected in this MRI image.",
    "pituitary": (
        "Pituitary adenoma — benign tumour of the pituitary gland. "
        "Can affect hormone production and vision depending on size and location."
    ),
}

_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

HF_REPO = "rAhuL45647/brain-mri-resnet50"   # push target from Colab notebook


def _build_model(num_classes: int = len(LABELS)) -> nn.Module:
    """ResNet-50 with Sequential(Dropout → Linear) head — matches Colab training."""
    import torchvision.models as models
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


class BrainMRIClassifier:
    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cpu")
        self._local_weights: Path = (
            get_settings().models_dir / "brain_mri_resnet50_v0.1.0.pt"
        )

    def load(self) -> None:
        """Try local weights first, then HuggingFace Hub."""
        if self._local_weights.exists():
            self._load_local()
        else:
            self._load_from_hub()

    def _load_local(self) -> None:
        try:
            model = _build_model()
            state = torch.load(self._local_weights, map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            log.info("brain_mri.model.loaded_local", path=str(self._local_weights))
        except Exception as e:
            log.error("brain_mri.model.load_local_failed", error=str(e))

    def _load_from_hub(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
            settings = get_settings()
            if not settings.huggingface_token:
                log.warning(
                    "brain_mri.model.hf_token_missing",
                    hint="Set HUGGINGFACE_TOKEN in .env to download from HF Hub.",
                )
                return
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename="brain_mri_resnet50_v0.1.0.pt",
                token=settings.huggingface_token,
                local_dir=str(settings.models_dir),
            )
            self._local_weights = Path(path)
            self._load_local()
        except Exception as e:
            log.warning(
                "brain_mri.model.hf_download_failed",
                error=str(e),
                hint=(
                    f"Train the model via ml_training/train_brain_mri.ipynb, "
                    f"push to HF Hub ({HF_REPO}), then restart the backend."
                ),
            )

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, file_bytes: bytes) -> tuple[list[dict], int]:
        """Returns (predictions_sorted_desc, latency_ms)."""
        start = time.perf_counter()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predictions = [
            {
                "label": label,
                "probability": float(prob),
                "description": LABEL_DESCRIPTIONS[label],
            }
            for label, prob in sorted(
                zip(LABELS, probs), key=lambda x: x[1], reverse=True
            )
        ]
        latency_ms = int((time.perf_counter() - start) * 1000)
        return predictions, latency_ms


classifier = BrainMRIClassifier()
