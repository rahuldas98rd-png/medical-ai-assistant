"""Prescription OCR module — image → structured medicines / dosages / instructions."""

from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.prescription_ocr.router import router
from backend.modules.prescription_ocr.service import service


class PrescriptionOCRModule(BaseModule):
    name = "prescription_ocr"
    version = "0.1.0"
    description = (
        "Extract structured information (patient, doctor, medicines, dosages, "
        "frequencies) from prescription images via Tesseract OCR + rule-based "
        "entity recognition."
    )
    tags = ["ocr"]

    def get_router(self) -> APIRouter:
        return router

    def on_startup(self) -> None:
        service.load()
        self._ready = service.is_ready()

    def health_check(self) -> dict:
        info = super().health_check()
        info["tesseract_available"] = service.engine.is_available()
        info["medicine_dictionary_size"] = len(service._medicine_dict)
        return info