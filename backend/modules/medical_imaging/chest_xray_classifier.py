"""
Chest X-ray multi-label pathology classifier with Grad-CAM + view confidence.

Adds to v0.1.0:
  - DICOM input support (via dicom_handler)
  - Grad-CAM heatmaps showing WHERE the model focused for each pathology
  - View-confidence heuristic that flags out-of-distribution inputs
    (lateral X-rays, non-X-ray images) based on prediction-spread statistics
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional

import numpy as np
import skimage.io
import structlog
import torch
import torchxrayvision as xrv
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from backend.core.exceptions import InvalidInputError, ModelNotLoadedError
from backend.modules.medical_imaging.dicom_handler import (
    is_dicom,
    read_dicom_to_array,
)

log = structlog.get_logger()

MODEL_NAME = "densenet121-res224-all"
MODEL_VERSION = "0.2.0"

TOP_FINDING_THRESHOLD = 0.5
CONFIDENCE_LOW = 0.3
CONFIDENCE_HIGH = 0.7

# Generate Grad-CAM for at most this many top findings (cap latency).
MAX_HEATMAPS = 3

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

    # ---- Lifecycle ----
    def load(self) -> None:
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

    # ---- Preprocessing (shared by predict + heatmap) ----
    def _decode_to_2d(
        self, file_bytes: bytes, content_type: str, filename: str
    ) -> np.ndarray:
        """Decode any supported input to a 2D grayscale numpy array."""
        if is_dicom(content_type, file_bytes, filename):
            return read_dicom_to_array(file_bytes)

        try:
            img = skimage.io.imread(io.BytesIO(file_bytes))
        except Exception as e:
            raise InvalidInputError(f"Could not decode image: {e}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.ndim == 3:
            img = img.mean(axis=2)
        return img.astype(np.float32)

    def _to_model_tensor(
        self, img_2d: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Take a 2D image, return (model_input_tensor, display_image_2d_224).
        The display_image is the 224×224 grayscale array used for heatmap overlays.
        """
        normalize_to = 65535 if img_2d.dtype == np.uint16 or img_2d.max() > 255 else 255
        img_norm = xrv.datasets.normalize(img_2d, normalize_to)
        img_norm = img_norm[None, ...].astype(np.float32)  # 1xHxW
        img_resized = self._transform(img_norm)  # 1x224x224

        tensor = torch.from_numpy(img_resized).unsqueeze(0).to(self._device)
        display = img_resized[0]  # 224x224 for overlay
        return tensor, display

    # ---- Inference ----
    def predict(
        self,
        file_bytes: bytes,
        content_type: str = "",
        filename: str = "",
        generate_heatmaps: bool = True,
    ) -> tuple[list[dict], dict, int, str]:
        """
        Returns (predictions, view_confidence, latency_ms, input_format).

        Each prediction may include a 'heatmap_base64' field for the top
        findings (up to MAX_HEATMAPS) if generate_heatmaps is True.
        """
        if not self.is_ready():
            raise ModelNotLoadedError("Chest X-ray model not loaded.")

        start = time.perf_counter()
        input_format = "dicom" if is_dicom(content_type, file_bytes, filename) else "image"

        img_2d = self._decode_to_2d(file_bytes, content_type, filename)
        tensor, display_2d = self._to_model_tensor(img_2d)

        with torch.no_grad():
            outputs = self._model(tensor)
            probs = outputs[0].cpu().numpy()

        predictions = [
            {"name": p, "probability": float(prob)}
            for p, prob in zip(self._pathologies, probs)
            if p
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        view_conf = self._assess_view_confidence(predictions)

        # Heatmaps for top findings (only if view looks frontal — no point
        # generating heatmaps for an OOD input)
        if generate_heatmaps and view_conf["likely_frontal_view"]:
            top = [p for p in predictions if p["probability"] >= TOP_FINDING_THRESHOLD][:MAX_HEATMAPS]
            for pred in top:
                try:
                    pred["heatmap_base64"] = self._generate_heatmap(
                        tensor, display_2d, pred["name"]
                    )
                except Exception as e:
                    log.warning(
                        "chest_xray.heatmap_failed",
                        pathology=pred["name"],
                        error=str(e),
                    )

        latency_ms = int((time.perf_counter() - start) * 1000)
        return predictions, view_conf, latency_ms, input_format

    # ---- Grad-CAM ----
    def _generate_heatmap(
        self, tensor: torch.Tensor, display_2d: np.ndarray, pathology: str
    ) -> str:
        """Generate a Grad-CAM overlay for `pathology`, return base64 PNG."""
        if pathology not in self._pathologies:
            raise ValueError(f"Unknown pathology: {pathology}")
        pathology_idx = self._pathologies.index(pathology)

        # Target layer: last dense block (highest-resolution semantic features)
        target_layer = self._model.features.denseblock4

        cam = GradCAM(model=self._model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pathology_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

        # Normalize display image to [0, 1] and convert to RGB for overlay
        disp = display_2d - display_2d.min()
        disp_max = disp.max()
        if disp_max > 0:
            disp = disp / disp_max
        disp_rgb = np.stack([disp, disp, disp], axis=-1).astype(np.float32)

        overlay = show_cam_on_image(disp_rgb, grayscale_cam, use_rgb=True)

        # Encode PNG → base64
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ---- View confidence heuristic ----
    @staticmethod
    def _assess_view_confidence(predictions: list[dict]) -> dict:
        """
        On a frontal X-ray the model gives well-separated probabilities.
        On a lateral X-ray (or non-X-ray, or otherwise OOD input), nearly
        everything clusters around 0.5. We use that as a soft signal.
        """
        probs = np.array([p["probability"] for p in predictions])
        spread = float(probs.std())
        uncertain_count = int(np.sum((probs >= 0.4) & (probs <= 0.6)))

        # Calibrated against the lateral failure case we hit during testing.
        likely_frontal = spread > 0.12 and uncertain_count < 8

        warning: Optional[str] = None
        if not likely_frontal:
            warning = (
                "Prediction probabilities are tightly clustered, which often "
                "indicates the input isn't a frontal (PA/AP) chest X-ray. "
                "Results may be unreliable. Try uploading a frontal-view X-ray."
            )

        return {
            "spread": spread,
            "uncertain_count": uncertain_count,
            "likely_frontal_view": likely_frontal,
            "warning": warning,
        }


classifier = ChestXRayClassifier()