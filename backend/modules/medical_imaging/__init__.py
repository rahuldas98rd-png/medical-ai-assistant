"""Medical imaging module — X-ray / MRI / CT classification."""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.medical_imaging.router import router
from backend.modules.medical_imaging.service import service


class MedicalImagingModule(BaseModule):
    name = "medical_imaging"
    version = "0.1.0"
    description = (
        "Medical image classification. Currently supports chest X-ray "
        "(18 pathologies via DenseNet-121 pretrained on NIH / CheXpert / "
        "MIMIC-CXR / PadChest). MRI/CT and Grad-CAM heatmaps planned for v0.2.0."
    )
    tags = ["imaging"]

    def get_router(self) -> APIRouter:
        return router

    def on_startup(self) -> None:
        try:
            service.load()
            self._ready = service.is_ready()
        except Exception:
            # Don't crash the whole app if the model can't be loaded
            # (e.g. no internet on first run). Module just reports unready.
            self._ready = False

    def health_check(self) -> dict:
        info = super().health_check()
        info["chest_xray_model_loaded"] = service.is_ready()
        return info