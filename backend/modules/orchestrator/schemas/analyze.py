"""Schemas for the orchestrator multi-module analysis endpoint."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AssessmentStatus(str, Enum):
    success = "success"
    flagged = "flagged"          # keywords matched but not enough inputs to run model
    skipped = "skipped"          # no relevant keywords found
    error = "error"


class RiskSummary(BaseModel):
    label: str                   # "low" / "moderate" / "high"
    score: float
    description: str


class ConditionAssessment(BaseModel):
    condition: str               # internal key e.g. "diabetes"
    display_name: str
    status: AssessmentStatus
    matched_keywords: list[str]  # terms from symptom text that triggered this
    risk: Optional[RiskSummary] = None
    top_contributors: list[dict] = []
    recommendations: list[str] = []
    detail_page: str             # Streamlit page name for deep-dive


class ImageFinding(BaseModel):
    label: str
    probability: float
    description: str = ""


class ImageAssessment(BaseModel):
    image_type: str              # "chest_xray" / "brain_mri" / "prescription"
    status: AssessmentStatus
    top_finding: Optional[ImageFinding] = None
    all_findings: list[ImageFinding] = []
    extra: dict[str, Any] = {}  # OCR text for prescriptions, view info for X-rays


class OrchestratorReport(BaseModel):
    condition_assessments: list[ConditionAssessment]
    image_assessment: Optional[ImageAssessment] = None
    overall_summary: str
    key_recommendations: list[str]
    inputs_used: dict[str, Any]  # echo back what was provided
    processing_time_ms: int
    disclaimer: str = (
        "This multi-module screening report is for general educational purposes only. "
        "Results are estimates based on population-level statistical models and do NOT "
        "constitute a medical diagnosis. Always consult a qualified healthcare professional "
        "for personal medical evaluation and treatment decisions."
    )
