"""
Image preprocessing for OCR.

Strategy: TWO paths. Default is minimal — just ensure decent resolution and
let Tesseract's own pipeline handle the rest. This works far better than
aggressive preprocessing on clean digital images. Aggressive mode (denoise +
binarize + deskew) is reserved for phone photos with uneven lighting.

The service tries minimal first; if Tesseract returns almost nothing, it
retries with aggressive.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.core.exceptions import InvalidInputError

MIN_LONG_EDGE = 1500  # upscale below this — Tesseract prefers ~300 DPI


def preprocess_for_ocr(image_bytes: bytes, aggressive: bool = False) -> np.ndarray:
    """Return a numpy array ready for Tesseract."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidInputError("Could not decode image (corrupt or unsupported format).")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if image is small — Tesseract works best around 300 DPI equivalent
    h, w = gray.shape
    if max(h, w) < MIN_LONG_EDGE:
        scale = MIN_LONG_EDGE / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if not aggressive:
        # Clean digital images: minimal touch. Tesseract handles the rest.
        return gray

    # ---- Aggressive path (phone photos, uneven lighting, skew) ----
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Otsu's binarization — better than adaptive for documents with
    # bimodal histograms (clearly dark text on light background).
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return _deskew(binary)


def _deskew(binary: np.ndarray) -> np.ndarray:
    """
    Estimate skew from text pixel orientation and rotate to correct.
    Operates on a BINARY image (text=0, background=255). Doing this on
    grayscale is unreliable because most pixels are 'non-zero' and the
    angle estimate becomes meaningless — a real bug in the previous version.
    """
    inverted = cv2.bitwise_not(binary)  # text now = 255
    coords = np.column_stack(np.where(inverted == 255))
    if len(coords) < 100:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return binary

    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )