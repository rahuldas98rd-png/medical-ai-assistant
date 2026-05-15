"""
Chest X-ray multi-label pathology classifier.

Backbone: DenseNet-121 from TorchXRayVision (MIT), pretrained on the union
of NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest. Outputs probabilities
for 18 pathologies.

Why this approach (not custom training)? On these datasets, a fine-tuned
DenseNet-121 already matches published benchmarks; replicating that with
limited Colab time would take days and add no real value. The library is
explicitly designed for clinical research and the models are referenced
in peer-reviewed papers. Good enough as a starting point.

Model size: ~30 MB, downloaded on first load to ~/.torchxrayvision/.
CPU inference: ~3-5s on i5-12400.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
import skimage.io
import structlog
import torch
import torchxrayvision as xrv

from backend.core.exceptions import InvalidInputError, ModelNotLoadedError

log = structlog.get_logger()

MODEL_NAME = "densenet121-res224-all"   # trained on ALL the union datasets
MODEL_VERSION = "0.1.0"

# Probability thresholds — illustrative, NOT clinically calibrated.
TOP_FINDING_THRESHOLD = 0.5
CONFIDENCE_LOW = 0.3
CONFIDENCE_HIGH = 0.7

# Plain-language descriptions shown alongside each prediction.
PATHOLOGY_DESCRIPTIONS: dict[str, str] = {
    "Atelectasis": "Partial collapse of lung tissue.",
    "Consolidation": "Lung tissue filled with fluid or solid material.",
    "Infiltration": "Substance accumulated in lung tissue.",
    "Pneumothorax": "Air in the pleural space — collapsed lung.",
    "Edema": "Fluid accumulation in lung tissue.",
    "Emphysema": "Damaged air sacs in lungs (often from smoking).",
    "Fibrosis": "Scarring of lung tissue.",
    "Effusion": "Fluid in the pleural space around the lungs.",
    "Pneumonia": "Lung infection causing inflammation.",
    "Pleural_Thickening": "Thickening of lung's outer lining.",
    "Cardiomegaly": "Enlarged heart silhouette.",
    "Nodule": "Small round growth (<3 cm) in lung.",
    "Mass": "Larger growth (≥3 cm) in lung tissue.",
    "Hernia": "Diaphragmatic hernia.",
    "Lung Lesion": "Abnormal tissue in lung.",
    "Fracture": "Bone fracture (typically rib).",
    "Lung Opacity": "Hazy areas in the lungs of unspecified cause.",
    "Enlarged Cardiomediastinum": "Enlarged mediastinum / heart area.",
}


class ChestXRayClassifier:
    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._device = torch.device("cpu")
        self._pathologies: list[str] = []
        self._transform: Optional[xrv.datasets.XRayResizer] = None

    def load(self) -> None:
        """Load pretrained DenseNet-121. ~30 MB download on first run."""
        try:
            self._model = xrv.models.DenseNet(weights=MODEL_NAME)
            self._model.eval()
            self._model.to(self._device)
            self._pathologies = list(self._model.pathologies)
            self._transform = xrv.datasets.XRayResizer(224)
            log.info(
                "chest_xray.model.loaded",
                model=MODEL_NAME,
                pathologies=len([p for p in self._pathologies if p]),
            )
        except Exception as e:
            log.error("chest_xray.model.load_failed", error=str(e))
            raise

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, image_bytes: bytes) -> tuple[list[dict], int]:
        """
        Run inference. Returns (predictions_list, latency_ms).
        Each prediction is {"name": str, "probability": float}.
        """
        if not self.is_ready():
            raise ModelNotLoadedError("Chest X-ray model not loaded.")

        start = time.perf_counter()

        # ---- Load & normalize image ----
        try:
            img = skimage.io.imread(io.BytesIO(image_bytes))
        except Exception as e:
            raise InvalidInputError(f"Could not decode image: {e}")

        # Handle RGBA: drop alpha channel
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]

        # Convert RGB → grayscale (X-rays are single-channel)
        if img.ndim == 3:
            img = img.mean(axis=2)

        # Detect bit depth so we normalize correctly to [-1024, 1024]
        normalize_to = 65535 if img.dtype == np.uint16 or img.max() > 255 else 255
        img = xrv.datasets.normalize(img, normalize_to)

        # Add channel dim: HxW → 1xHxW
        img = img[None, ...].astype(np.float32)

        # Resize / center-crop to 224x224 (the model's expected input)
        img = self._transform(img)

        # ---- Inference ----
        tensor = torch.from_numpy(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            outputs = self._model(tensor)
            probs = outputs[0].cpu().numpy()

        # ---- Build response ----
        predictions = [
            {"name": p, "probability": float(prob)}
            for p, prob in zip(self._pathologies, probs)
            if p  # some pathology slots are empty in unified vocab
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return predictions, latency_ms


classifier = ChestXRayClassifier()