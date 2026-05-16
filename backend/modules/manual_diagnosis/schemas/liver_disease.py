"""Request / response schemas for liver disease risk prediction."""

from typing import Literal

from pydantic import BaseModel, Field


class LiverDiseaseRiskRequest(BaseModel):
    """
    Inputs match the Indian Liver Patient Dataset (UCI) — a standard benchmark
    for liver disease classification using liver function test (LFT) values.
    """

    age: int = Field(ge=1, le=120, description="Age in years.", examples=[45])
    gender: Literal[0, 1] = Field(
        description="Gender: 1 = male, 0 = female.",
        examples=[1],
    )
    total_bilirubin: float = Field(
        ge=0.1, le=75.0,
        description="Total Bilirubin (mg/dL). Normal: 0.2–1.2. Elevated: >1.2.",
        examples=[0.7],
    )
    direct_bilirubin: float = Field(
        ge=0.0, le=20.0,
        description="Direct (conjugated) Bilirubin (mg/dL). Normal: 0.0–0.3.",
        examples=[0.1],
    )
    alkaline_phosphotase: float = Field(
        ge=20, le=2500,
        description="Alkaline Phosphotase (IU/L). Normal: 44–147.",
        examples=[187.0],
    )
    alamine_aminotransferase: float = Field(
        ge=1, le=2000,
        description="ALT / SGPT (IU/L) — liver cell damage marker. Normal: 7–56.",
        examples=[16.0],
    )
    aspartate_aminotransferase: float = Field(
        ge=1, le=5000,
        description="AST / SGOT (IU/L) — liver / heart cell damage marker. Normal: 10–40.",
        examples=[18.0],
    )
    total_proteins: float = Field(
        ge=2.0, le=12.0,
        description="Total proteins (g/dL). Normal: 6.3–8.2.",
        examples=[6.8],
    )
    albumin: float = Field(
        ge=0.5, le=6.0,
        description="Albumin (g/dL) — main blood protein made by the liver. Normal: 3.5–5.0.",
        examples=[3.3],
    )
    albumin_globulin_ratio: float = Field(
        ge=0.1, le=3.0,
        description="Albumin/Globulin ratio. Normal: 1.0–2.5. Low ratio may indicate liver disease.",
        examples=[0.9],
    )


class RiskLevel(BaseModel):
    label: Literal["low", "moderate", "high"]
    score: float = Field(ge=0, le=1)
    description: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float


class LiverDiseaseRiskResponse(BaseModel):
    risk: RiskLevel
    top_contributors: list[FeatureContribution]
    recommendations: list[str]
    model_version: str
    disclaimer: str = (
        "This is an educational liver disease risk estimate derived from "
        "population-level ML patterns on the Indian Liver Patient Dataset. "
        "It is NOT a diagnosis. Liver disease diagnosis requires a full clinical "
        "evaluation including ultrasound, liver biopsy (if indicated), and assessment "
        "by a gastroenterologist or hepatologist."
    )
