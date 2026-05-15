"""
PDF prescription handling.

Two-tier strategy:
  1. If the PDF has embedded text (digital PDF), extract it directly.
     This is faster AND more accurate than running OCR over rendered pages.
  2. If embedded text is empty/short (scanned PDF), render each page to a
     PNG byte stream so the existing image-OCR pipeline can process it.

Uses pypdfium2 (Apache-2.0, no system binaries) — important for Windows.
"""

from __future__ import annotations

import io

import pypdfium2 as pdfium

MAX_PAGES = 5  # safety cap — refuse PDFs longer than this in v0.1.0
RENDER_SCALE = 2.0  # 2x scale of native; ~200 DPI equivalent — good for OCR


def is_pdf(content_type: str, file_bytes: bytes) -> bool:
    """Detect PDF by content-type OR magic bytes."""
    return (
        content_type == "application/pdf"
        or file_bytes[:5] == b"%PDF-"
    )


def extract_embedded_text(pdf_bytes: bytes) -> str:
    """Pull selectable text from a digital PDF. Empty string if none."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    if len(pdf) > MAX_PAGES:
        raise ValueError(
            f"PDF has {len(pdf)} pages; max supported is {MAX_PAGES}."
        )
    parts: list[str] = []
    for page in pdf:
        textpage = page.get_textpage()
        try:
            parts.append(textpage.get_text_range())
        finally:
            textpage.close()
        page.close()
    pdf.close()
    return "\n".join(parts)


def render_pages_to_png_bytes(pdf_bytes: bytes) -> list[bytes]:
    """Render each PDF page as a PNG byte stream (for OCR fallback)."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    if len(pdf) > MAX_PAGES:
        raise ValueError(
            f"PDF has {len(pdf)} pages; max supported is {MAX_PAGES}."
        )
    images: list[bytes] = []
    for page in pdf:
        bitmap = page.render(scale=RENDER_SCALE)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        images.append(buf.getvalue())
        bitmap.close()
        page.close()
    pdf.close()
    return images