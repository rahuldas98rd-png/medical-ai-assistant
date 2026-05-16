"""Request / response schemas for heart disease risk prediction."""

from typing import Literal

from pydantic import BaseModel, Field


class HeartDiseaseRiskRequest(BaseModel):
    """
    Inputs match the Cleveland Heart Disease dataset (UCI) — the standard
    benchmark for ML-based coronary artery disease risk scoring.
    """

    age: int = Field(ge=1, le=120, description="Age in years.", examples=[54])
    sex: Literal[0, 1] = Field(
        description="Biological sex: 1 = male, 0 = female.",
        examples=[1],
    )
    chest_pain_type: Literal[0, 1, 2, 3] = Field(
        description=(
            "Chest pain type: 0 = typical angina, 1 = atypical angina, "
            "2 = non-anginal pain, 3 = asymptomatic."
        ),
        examples=[2],
    )
    resting_bp: float = Field(
        ge=60, le=250,
        description="Resting blood pressure on admission (mm Hg). Normal: 60–80 diastolic.",
        examples=[130.0],
    )
    cholesterol: float = Field(
        ge=100, le=600,
        description="Serum cholesterol (mg/dL). Desirable: <200. High: ≥240.",
        examples=[250.0],
    )
    fasting_blood_sugar_gt120: Literal[0, 1] = Field(
        description="Fasting blood sugar > 120 mg/dL: 1 = true, 0 = false.",
        examples=[0],
    )
    resting_ecg: Literal[0, 1, 2] = Field(
        description=(
            "Resting ECG results: 0 = normal, 1 = ST-T wave abnormality, "
            "2 = left ventricular hypertrophy."
        ),
        examples=[1],
    )
    max_heart_rate: float = Field(
        ge=60, le=220,
        description="Maximum heart rate achieved during exercise stress test (bpm).",
        examples=[160.0],
    )
    exercise_angina: Literal[0, 1] = Field(
        description="Exercise-induced angina: 1 = yes, 0 = no.",
        examples=[0],
    )
    st_depression: float = Field(
        ge=0.0, le=10.0,
        description="ST depression induced by exercise relative to rest (mm).",
        examples=[1.5],
    )
    st_slope: Literal[0, 1, 2] = Field(
        description=(
            "Slope of the peak exercise ST segment: 0 = upsloping (normal), "
            "1 = flat, 2 = downsloping (worst)."
        ),
        examples=[1],
    )
    num_major_vessels: Literal[0, 1, 2, 3] = Field(
        description="Number of major vessels (0–3) colored by fluoroscopy.",
        examples=[0],
    )
    thalassemia: Literal[0, 1, 2] = Field(
        description=(
            "Thalassemia: 0 = normal, 1 = fixed defect, 2 = reversible defect."
        ),
        examples=[1],
    )


class RiskLevel(BaseModel):
    label: Literal["low", "moderate", "high"]
    score: float = Field(ge=0, le=1)
    description: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float


class HeartDiseaseRiskResponse(BaseModel):
    risk: RiskLevel
    top_contributors: list[FeatureContribution]
    recommendations: list[str]
    model_version: str
    disclaimer: str = (
        "This is an educational heart disease risk estimate derived from "
        "population-level ML patterns on the Cleveland Heart Disease dataset. "
        "It is NOT a cardiac diagnosis. Coronary artery disease diagnosis requires "
        "ECG, stress tests, imaging, and evaluation by a cardiologist."
    )
