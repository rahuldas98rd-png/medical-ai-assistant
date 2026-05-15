"""
OCR engine wrappers.

Phase 2 v0.1.0 uses Tesseract only. v0.2.0 will add an EasyOCR fallback
for handwritten prescriptions and pick the better result per-region.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytesseract
import structlog

log = structlog.get_logger()


def _configure_tesseract_path() -> None:
    """Find tesseract.exe on Windows. No-op on Linux/Mac (it's on PATH)."""
    # 1. Explicit env var wins
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and Path(env_path).exists():
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    # 2. Common Windows install locations
    for candidate in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return

    # 3. Otherwise assume it's on PATH (Linux/Mac, or user added it manually).


_configure_tesseract_path()


class TesseractEngine:
    def __init__(self) -> None:
        self._available = self._probe()

    def _probe(self) -> bool:
        try:
            v = pytesseract.get_tesseract_version()
            log.info("tesseract.detected", version=str(v))
            return True
        except Exception as e:
            log.warning("tesseract.unavailable", error=str(e))
            return False

    def is_available(self) -> bool:
        return self._available

    def extract_text(self, image: np.ndarray, psm: int = 3) -> str:
        """
        Run OCR on a preprocessed image array.

        PSM 3 (default) = automatic page segmentation. Best for prescriptions
        which have multiple zones (header, patient block, Rx list, footer).
        PSM 6 (uniform block) is sometimes useful as a fallback.
        """
        if not self._available:
            raise RuntimeError("Tesseract binary not found. See README setup.")
        config = f"--oem 3 --psm {psm}"
        return pytesseract.image_to_string(image, config=config)