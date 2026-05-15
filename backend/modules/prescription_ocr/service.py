"""Service orchestrator: image bytes → preprocessed → OCR → structured."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import structlog

from backend.core.exceptions import InvalidInputError, ModelNotLoadedError
from backend.database import PredictionLog, get_session
from backend.modules.prescription_ocr.entity_extractor import extract_structured
from backend.modules.prescription_ocr.ocr_engines import TesseractEngine
from backend.modules.prescription_ocr.preprocessing import preprocess_for_ocr
from backend.modules.prescription_ocr.schemas.prescription import PrescriptionResponse

from backend.modules.prescription_ocr.pdf_handler import (
    extract_embedded_text,
    is_pdf,
    render_pages_to_png_bytes,
)

log = structlog.get_logger()

MODULE_VERSION = "0.1.0"
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class PrescriptionOCRService:
    def __init__(self) -> None:
        self.engine = TesseractEngine()
        self._medicine_dict: set[str] = set()
        self._dict_path: Path = Path(__file__).parent / "data" / "medicine_dictionary.txt"

    def load(self) -> None:
        if self._dict_path.exists():
            with open(self._dict_path, "r", encoding="utf-8") as f:
                self._medicine_dict = {
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.startswith("#")
                }
            log.info("prescription_ocr.dict.loaded", count=len(self._medicine_dict))
        else:
            log.warning("prescription_ocr.dict.missing", path=str(self._dict_path))

    def is_ready(self) -> bool:
        # We consider the service ready if Tesseract is present.
        # An empty medicine dict still works (falls back to MED_FORM_PREFIX).
        return self.engine.is_available()

    def process(self, file_bytes: bytes, content_type: str = "") -> PrescriptionResponse:
        if not self.is_ready():
            raise ModelNotLoadedError(
                "Tesseract OCR is not available on this system. Install from "
                "https://github.com/UB-Mannheim/tesseract/wiki and set "
                "TESSERACT_CMD in .env if it's not on the default path."
            )

        if len(file_bytes) > MAX_FILE_BYTES:
            raise InvalidInputError(
                f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit."
            )

        start = time.perf_counter()
        raw_text = ""
        strategy = ""

        # ---- PDF branch ----
        if is_pdf(content_type, file_bytes):
            try:
                embedded = extract_embedded_text(file_bytes)
            except ValueError as e:
                raise InvalidInputError(str(e))
            except Exception as e:
                raise InvalidInputError(f"Could not read PDF: {e}")

            if len(embedded.strip()) > 50:
                raw_text = embedded
                strategy = "pdf_embedded_text"
            else:
                # Scanned PDF — render and OCR each page
                try:
                    page_images = render_pages_to_png_bytes(file_bytes)
                except Exception as e:
                    raise InvalidInputError(f"Could not render PDF pages: {e}")
                page_texts: list[str] = []
                for img_bytes in page_images:
                    processed = preprocess_for_ocr(img_bytes, aggressive=False)
                    page_texts.append(self.engine.extract_text(processed))
                raw_text = "\n".join(page_texts)
                strategy = f"pdf_ocr ({len(page_images)} pages)"
        # ---- Image branch (unchanged) ----
        else:
            try:
                processed = preprocess_for_ocr(file_bytes, aggressive=False)
            except InvalidInputError:
                raise
            except Exception as e:
                raise InvalidInputError(f"Image preprocessing failed: {e}")
            raw_text = self.engine.extract_text(processed)
            strategy = "minimal"
            if len(raw_text.strip()) < 30:
                log.info("prescription_ocr.retrying", reason="low_text_yield")
                try:
                    processed = preprocess_for_ocr(file_bytes, aggressive=True)
                    aggressive_text = self.engine.extract_text(processed)
                    if len(aggressive_text.strip()) > len(raw_text.strip()):
                        raw_text = aggressive_text
                        strategy = "aggressive"
                except Exception as e:
                    log.warning("prescription_ocr.aggressive_failed", error=str(e))

        extraction = extract_structured(raw_text, self._medicine_dict)
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._audit_log(file_bytes, extraction, latency_ms)

        return PrescriptionResponse(
            extraction=extraction,
            raw_text=raw_text,
            ocr_engine=f"tesseract ({strategy})",
            processing_time_ms=latency_ms,
        )

    def _audit_log(self, image_bytes, extraction, latency_ms) -> None:
        """Hash inputs — never persist raw image bytes."""
        try:
            with get_session() as s:
                s.add(
                    PredictionLog(
                        module_name="prescription_ocr",
                        module_version=MODULE_VERSION,
                        input_hash=hashlib.sha256(image_bytes).hexdigest(),
                        prediction={
                            "medicines_found": len(extraction.medicines),
                            "has_warnings": bool(extraction.confidence_warnings),
                        },
                        confidence=None,
                        latency_ms=latency_ms,
                    )
                )
        except Exception as e:
            log.error("prescription_ocr.audit_log_failed", error=str(e))


service = PrescriptionOCRService()