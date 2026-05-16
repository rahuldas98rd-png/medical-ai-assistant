"""Brain MRI tumor classification module."""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.brain_mri.classifier import classifier
from backend.modules.brain_mri.router import router as mri_router


class BrainMRIModule(BaseModule):
    name = "brain_mri"
    version = "0.1.0"
    description = (
        "Brain MRI tumor classification: glioma, meningioma, pituitary adenoma, or no tumour. "
        "ResNet-50 trained on the Kaggle Brain MRI dataset (~3060 images). "
        "Model requires Colab training — see ml_training/train_brain_mri.ipynb."
    )
    tags = ["imaging"]

    def get_router(self) -> APIRouter:
        return mri_router

    def on_startup(self) -> None:
        try:
            classifier.load()
            self._ready = classifier.is_ready()
        except Exception:
            self._ready = False

    def health_check(self) -> dict:
        info = super().health_check()
        info["model_ready"] = classifier.is_ready()
        info["train_instructions"] = "Run ml_training/train_brain_mri.ipynb on Google Colab"
        return info
