"""Pydantic schemas for prescription OCR module."""

from typing import Optional

from pydantic import BaseModel, Field


class Medicine(BaseModel):
    name: str = Field(description="Medicine name (e.g., 'Metformin').")
    dosage: Optional[str] = Field(None, description="Dose with unit, e.g., '500 mg'.")
    frequency: Optional[str] = Field(
        None, description="How often, e.g., 'BD (twice daily)'."
    )
    duration: Optional[str] = Field(None, description="How long, e.g., '30 days'.")
    raw_line: str = Field(description="The OCR line this was parsed from.")


class PrescriptionExtraction(BaseModel):
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    doctor_name: Optional[str] = None
    prescription_date: Optional[str] = None
    medicines: list[Medicine] = Field(default_factory=list)
    general_instructions: list[str] = Field(default_factory=list)
    confidence_warnings: list[str] = Field(
        default_factory=list,
        description="Things that should make the user double-check (low text "
                    "volume, no medicines matched, etc).",
    )


class PrescriptionResponse(BaseModel):
    extraction: PrescriptionExtraction
    raw_text: str = Field(description="Full unprocessed OCR output, for reference.")
    ocr_engine: str
    processing_time_ms: int
    disclaimer: str = (
        "OCR can make mistakes, especially with handwritten prescriptions or "
        "poor image quality. Always verify extracted information against the "
        "original prescription. This tool is NOT a substitute for professional "
        "pharmacist review."
    )