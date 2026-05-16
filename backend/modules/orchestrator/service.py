"""
Orchestrator service — routes inputs to relevant module services and
synthesizes a unified report.

Design decisions:
  - Calls module service singletons directly (no HTTP round-trip).
  - Tabular models run only when enough inputs are provided; otherwise the
    condition is flagged for deeper assessment on its dedicated page.
  - Image routing is explicit (user selects type) to avoid unreliable
    auto-detection across X-ray / MRI / prescription domains.
  - Each module call is wrapped in try/except so one failure never breaks
    the whole report.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import structlog

from backend.modules.orchestrator.schemas.analyze import (
    AssessmentStatus,
    ConditionAssessment,
    ImageAssessment,
    ImageFinding,
    OrchestratorReport,
    RiskSummary,
)

log = structlog.get_logger()

# ── Symptom → condition keyword map ──────────────────────────────────────────

CONDITION_META: dict[str, dict] = {
    "diabetes": {
        "display": "Diabetes Risk",
        "page": "1_Diabetes_Risk",
        "keywords": [
            "diabetes", "diabetic", "blood sugar", "glucose", "insulin",
            "thirst", "frequent urination", "polyuria", "polydipsia",
            "hba1c", "a1c", "prediabetes", "hyperglycemia", "sweet urine",
        ],
        "needs": ["glucose", "bmi", "age"],
    },
    "hypertension": {
        "display": "Hypertension Risk",
        "page": "4_Hypertension_Risk",
        "keywords": [
            "blood pressure", "hypertension", "high bp", "systolic",
            "diastolic", "hypertensive", "headache", "dizziness",
            "pounding head", "blurred vision", "nosebleed",
        ],
        "needs": ["systolic_bp", "age"],
    },
    "heart_disease": {
        "display": "Heart Disease Risk",
        "page": "5_Heart_Disease_Risk",
        "keywords": [
            "chest pain", "heart", "cardiac", "angina", "palpitation",
            "shortness of breath", "breathless", "cholesterol", "coronary",
            "ecg", "ekg", "heart attack", "tachycardia", "arrhythmia",
            "left arm pain", "jaw pain",
        ],
        "needs": [],  # too many clinical inputs — always flag only
    },
    "liver_disease": {
        "display": "Liver Disease Risk",
        "page": "6_Liver_Disease_Risk",
        "keywords": [
            "liver", "jaundice", "yellow skin", "yellow eyes", "hepatitis",
            "bilirubin", "abdominal pain", "right side pain", "cirrhosis",
            "fatty liver", "alt", "ast", "nausea", "dark urine",
        ],
        "needs": [],  # requires specific lab values — always flag only
    },
}


def _screen_symptoms(text: str) -> dict[str, list[str]]:
    """Return {condition: [matched keywords]} for every condition with ≥1 match."""
    lower = text.lower()
    hits: dict[str, list[str]] = {}
    for cond, meta in CONDITION_META.items():
        matched = [kw for kw in meta["keywords"] if kw in lower]
        if matched:
            hits[cond] = matched
    return hits


def _has_inputs(needed: list[str], inputs: dict[str, Any]) -> bool:
    return all(inputs.get(k) is not None for k in needed)


# ── Tabular model runners ─────────────────────────────────────────────────────

def _run_diabetes(inputs: dict) -> tuple[RiskSummary, list[dict], list[str]]:
    from backend.modules.manual_diagnosis.diabetes_service import service
    from backend.modules.manual_diagnosis.schemas.diabetes import DiabetesRiskRequest

    req = DiabetesRiskRequest(
        pregnancies=inputs.get("pregnancies", 0),
        glucose=inputs["glucose"],
        blood_pressure=inputs.get("systolic_bp", 80.0),
        skin_thickness=20.0,
        insulin=0.0,
        bmi=inputs["bmi"],
        diabetes_pedigree=0.3,
        age=inputs["age"],
    )
    resp = service.predict(req)
    risk = RiskSummary(
        label=resp.risk.label,
        score=resp.risk.score,
        description=resp.risk.description,
    )
    contributors = [c.model_dump() for c in resp.top_contributors]
    return risk, contributors, resp.recommendations


def _run_hypertension(inputs: dict) -> tuple[RiskSummary, list[dict], list[str]]:
    from backend.modules.manual_diagnosis.hypertension_service import service
    from backend.modules.manual_diagnosis.schemas.hypertension import HypertensionRiskRequest

    bmi = inputs.get("bmi") or 26.0
    req = HypertensionRiskRequest(
        age=inputs["age"],
        systolic_bp=inputs["systolic_bp"],
        ldl_cholesterol=3.5,
        adiposity=float(bmi),
        family_history=False,
        type_a_behavior=50,
        obesity_index=float(bmi),
        alcohol_units_week=0.0,
        tobacco_kg_lifetime=0.0,
    )
    resp = service.predict(req)
    risk = RiskSummary(
        label=resp.risk.label,
        score=resp.risk.score,
        description=resp.risk.description,
    )
    contributors = [c.model_dump() for c in resp.top_contributors]
    return risk, contributors, resp.recommendations


# ── Image runners ─────────────────────────────────────────────────────────────

def _run_chest_xray(image_bytes: bytes) -> ImageAssessment:
    from backend.modules.medical_imaging.service import service

    resp = service.process(image_bytes, content_type="image/jpeg", filename="upload.jpg")
    top = resp.top_findings[0] if resp.top_findings else resp.predictions[0]
    return ImageAssessment(
        image_type="chest_xray",
        status=AssessmentStatus.success,
        top_finding=ImageFinding(
            label=top.name,
            probability=top.probability,
            description=top.description,
        ),
        all_findings=[
            ImageFinding(label=p.name, probability=p.probability, description=p.description)
            for p in resp.predictions[:6]
        ],
        extra={
            "view_frontal": resp.view_confidence.likely_frontal_view,
            "processing_time_ms": resp.processing_time_ms,
        },
    )


def _run_brain_mri(image_bytes: bytes) -> ImageAssessment:
    from backend.modules.brain_mri.classifier import classifier

    preds, latency_ms = classifier.predict(image_bytes)
    top = preds[0]
    return ImageAssessment(
        image_type="brain_mri",
        status=AssessmentStatus.success,
        top_finding=ImageFinding(
            label=top["label"],
            probability=top["probability"],
            description=top["description"],
        ),
        all_findings=[
            ImageFinding(label=p["label"], probability=p["probability"])
            for p in preds
        ],
        extra={"processing_time_ms": latency_ms},
    )


def _run_prescription(image_bytes: bytes, content_type: str) -> ImageAssessment:
    from backend.modules.prescription_ocr.service import service

    resp = service.process(image_bytes, content_type=content_type)
    medicine_names = [m.name for m in resp.medicines[:5]]
    return ImageAssessment(
        image_type="prescription",
        status=AssessmentStatus.success,
        top_finding=ImageFinding(
            label=f"{len(resp.medicines)} medicine(s) extracted",
            probability=1.0,
        ),
        extra={
            "medicines": [m.model_dump() for m in resp.medicines],
            "raw_text_preview": resp.raw_text[:300] if resp.raw_text else "",
            "medicine_names": medicine_names,
        },
    )


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesize(
    assessments: list[ConditionAssessment],
    image: Optional[ImageAssessment],
    symptoms: str,
) -> tuple[str, list[str]]:
    """Build overall summary text and deduplicated key recommendations."""
    parts: list[str] = []
    recs: list[str] = []

    if image:
        itype = image.image_type.replace("_", " ").title()
        if image.status == AssessmentStatus.success and image.top_finding:
            tf = image.top_finding
            parts.append(
                f"{itype} analysis: top finding is '{tf.label}' "
                f"(confidence {tf.probability:.0%})."
            )
        if image.image_type == "prescription" and image.extra.get("medicine_names"):
            parts.append(
                f"Prescription: {', '.join(image.extra['medicine_names'])} identified."
            )

    run = [a for a in assessments if a.status == AssessmentStatus.success]
    flagged = [a for a in assessments if a.status == AssessmentStatus.flagged]

    for a in run:
        risk_label = a.risk.label if a.risk else "unknown"
        parts.append(
            f"{a.display_name}: estimated {risk_label} risk "
            f"(score {a.risk.score:.0%})." if a.risk else f"{a.display_name}: assessed."
        )
        recs.extend(a.recommendations[:2])

    if flagged:
        names = ", ".join(a.display_name for a in flagged)
        parts.append(
            f"Symptoms also suggest {names} — visit the dedicated pages "
            "for a detailed assessment with full clinical inputs."
        )

    if not parts:
        parts.append(
            "No specific conditions were triggered from the provided symptoms and inputs. "
            "Provide additional clinical measurements for a more complete assessment."
        )

    # Deduplicate recommendations while preserving order
    seen: set[str] = set()
    unique_recs: list[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    unique_recs.append(
        "Consult a qualified healthcare professional for clinical evaluation — "
        "this report is for educational screening only."
    )

    return " ".join(parts), unique_recs


# ── Main orchestrator ─────────────────────────────────────────────────────────

class OrchestratorService:
    def analyze(
        self,
        symptoms: str,
        age: Optional[int],
        gender: Optional[str],
        bmi: Optional[float],
        glucose: Optional[float],
        systolic_bp: Optional[float],
        image_bytes: Optional[bytes],
        image_type: Optional[str],
        content_type: str = "",
    ) -> OrchestratorReport:
        t0 = time.perf_counter()

        inputs = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "glucose": glucose,
            "systolic_bp": systolic_bp,
        }

        # 1. Screen symptoms
        keyword_hits = _screen_symptoms(symptoms)

        # 2. Build condition assessments
        assessments: list[ConditionAssessment] = []

        for cond, meta in CONDITION_META.items():
            matched = keyword_hits.get(cond, [])
            if not matched:
                continue

            needed = meta["needs"]
            ca = ConditionAssessment(
                condition=cond,
                display_name=meta["display"],
                matched_keywords=matched,
                detail_page=meta["page"],
                status=AssessmentStatus.flagged,
            )

            if needed and _has_inputs(needed, inputs):
                try:
                    if cond == "diabetes":
                        risk, contrib, recs = _run_diabetes(inputs)
                    elif cond == "hypertension":
                        risk, contrib, recs = _run_hypertension(inputs)
                    else:
                        raise NotImplementedError
                    ca.status = AssessmentStatus.success
                    ca.risk = risk
                    ca.top_contributors = contrib
                    ca.recommendations = recs
                except Exception as e:
                    log.error("orchestrator.model_failed", condition=cond, error=str(e))
                    ca.status = AssessmentStatus.error
                    ca.recommendations = [f"Assessment failed — visit {meta['display']} page directly."]

            assessments.append(ca)

        # 3. Image analysis
        image_assessment: Optional[ImageAssessment] = None
        if image_bytes and image_type:
            try:
                if image_type == "chest_xray":
                    image_assessment = _run_chest_xray(image_bytes)
                elif image_type == "brain_mri":
                    image_assessment = _run_brain_mri(image_bytes)
                elif image_type == "prescription":
                    image_assessment = _run_prescription(image_bytes, content_type)
                else:
                    log.warning("orchestrator.unknown_image_type", image_type=image_type)
            except Exception as e:
                log.error("orchestrator.image_failed", image_type=image_type, error=str(e))
                image_assessment = ImageAssessment(
                    image_type=image_type,
                    status=AssessmentStatus.error,
                    extra={"error": str(e)},
                )

        # 4. Synthesize
        summary, key_recs = _synthesize(assessments, image_assessment, symptoms)

        inputs_used = {k: v for k, v in inputs.items() if v is not None}
        if image_type:
            inputs_used["image_type"] = image_type

        return OrchestratorReport(
            condition_assessments=assessments,
            image_assessment=image_assessment,
            overall_summary=summary,
            key_recommendations=key_recs,
            inputs_used=inputs_used,
            processing_time_ms=int((time.perf_counter() - t0) * 1000),
        )


service = OrchestratorService()
