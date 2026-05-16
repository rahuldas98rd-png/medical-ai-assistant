"""
DICOM (.dcm) reading for chest X-ray module.

DICOM is the standard medical imaging format. Real X-rays from PACS or
clinical workflows arrive as DICOM, not PNG. This module reads DICOM bytes
into a numpy array compatible with our existing preprocessing pipeline.

Handled concerns:
  - Magic byte detection (DICOMs don't have a reliable file extension)
  - VOI LUT (window/level) — DICOMs encode display windowing separately
    from pixel data; without this, images look washed-out
  - MONOCHROME1 inversion — some DICOMs store inverted intensities
    (high values = dark) compared to MONOCHROME2 (high values = light)
  - 12/16-bit pixel data — common in radiology, must be normalized
"""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def is_dicom(content_type: str, file_bytes: bytes, filename: str = "") -> bool:
    """Detect DICOM by content-type, magic bytes, or extension."""
    if content_type in ("application/dicom", "application/x-dicom"):
        return True
    # Standard DICOM signature: 'DICM' at byte offset 128
    if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
        return True
    if filename.lower().endswith((".dcm", ".dicom")):
        return True
    return False


def read_dicom_to_array(file_bytes: bytes) -> np.ndarray:
    """
    Decode a DICOM byte stream to a 2D grayscale numpy array.

    Applies window/level if encoded in the file, and corrects MONOCHROME1
    inversion. Returns float32 array; caller normalizes for the model.
    """
    ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

    if not hasattr(ds, "pixel_array"):
        raise ValueError("DICOM file has no pixel data.")

    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply Value-of-Interest LUT if present (windowing).
    # Falls back silently if the file lacks the required tags.
    try:
        pixel_array = apply_voi_lut(pixel_array, ds).astype(np.float32)
    except Exception:
        pass

    # MONOCHROME1: high values = dark. Invert to match MONOCHROME2 convention.
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # If it's a color/multi-frame DICOM (rare for chest X-ray), take the
    # first frame and average channels.
    if pixel_array.ndim == 3:
        if pixel_array.shape[0] in (1, 3, 4) and pixel_array.shape[-1] != 3:
            pixel_array = pixel_array[0]  # first frame
        if pixel_array.ndim == 3:
            pixel_array = pixel_array.mean(axis=-1)

    return pixel_array