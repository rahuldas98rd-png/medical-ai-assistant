"""
Tests for the orchestrator module.

Covers:
  - Symptom keyword routing (_screen_symptoms)
  - Condition triggering logic (flagged vs success vs no-match)
  - Image type validation (valid/invalid types)
  - Edge cases: SQL injection, XSS, empty-ish symptoms, adversarial inputs
  - Integration: /analyze endpoint returns correct structure (no models needed
    when no relevant keywords + no image)
"""

import pytest
from pydantic import ValidationError

from backend.modules.orchestrator.service import (
    CONDITION_META,
    OrchestratorService,
    _has_inputs,
    _screen_symptoms,
    _synthesize,
)
from backend.modules.orchestrator.schemas.analyze import (
    AssessmentStatus,
    ConditionAssessment,
    OrchestratorReport,
)


# ── _screen_symptoms ──────────────────────────────────────────────────────────

class TestScreenSymptoms:
    def test_diabetes_keyword_match(self):
        hits = _screen_symptoms("I have excessive thirst and frequent urination")
        assert "diabetes" in hits
        assert "thirst" in hits["diabetes"] or "frequent urination" in hits["diabetes"]

    def test_hypertension_keyword_match(self):
        hits = _screen_symptoms("My blood pressure is very high and I have headache")
        assert "hypertension" in hits

    def test_heart_disease_keyword_match(self):
        hits = _screen_symptoms("I have chest pain and shortness of breath")
        assert "heart_disease" in hits

    def test_liver_keyword_match(self):
        hits = _screen_symptoms("I have jaundice and my eyes are yellow")
        assert "liver_disease" in hits

    def test_no_match_on_unrelated_text(self):
        hits = _screen_symptoms("I have a sprained ankle from playing soccer")
        assert len(hits) == 0

    def test_multiple_conditions_from_one_text(self):
        hits = _screen_symptoms(
            "I have high blood pressure and chest pain with glucose of 200"
        )
        assert "hypertension" in hits
        assert "heart_disease" in hits
        assert "diabetes" in hits

    def test_case_insensitive_matching(self):
        hits = _screen_symptoms("FREQUENT URINATION and HIGH GLUCOSE levels")
        assert "diabetes" in hits

    def test_partial_keyword_substring(self):
        # "polyuria" is a keyword — should match when present
        hits = _screen_symptoms("Doctor noted polyuria and polydipsia")
        assert "diabetes" in hits

    def test_empty_string_returns_no_hits(self):
        hits = _screen_symptoms("          ")
        assert len(hits) == 0


# ── _has_inputs ───────────────────────────────────────────────────────────────

class TestHasInputs:
    def test_all_present(self):
        assert _has_inputs(["glucose", "bmi", "age"], {"glucose": 120.0, "bmi": 28.5, "age": 40})

    def test_one_missing(self):
        assert not _has_inputs(["glucose", "bmi", "age"], {"glucose": 120.0, "age": 40})

    def test_none_value_counts_as_missing(self):
        assert not _has_inputs(["glucose"], {"glucose": None})

    def test_empty_needs_always_true(self):
        assert _has_inputs([], {})

    def test_heart_disease_always_flagged(self):
        # heart_disease.needs == [] so _has_inputs returns True, but condition
        # is never in the 'needs and _has_inputs' branch because there's no model runner
        assert CONDITION_META["heart_disease"]["needs"] == []
        assert CONDITION_META["liver_disease"]["needs"] == []


# ── _synthesize ───────────────────────────────────────────────────────────────

class TestSynthesize:
    def _make_assessment(self, condition="diabetes", status=AssessmentStatus.flagged):
        return ConditionAssessment(
            condition=condition,
            display_name="Diabetes Risk",
            status=status,
            matched_keywords=["glucose"],
            detail_page="Diabetes_Risk",
        )

    def test_no_assessments_returns_fallback(self):
        summary, recs = _synthesize([], None, "some text")
        assert "No specific conditions" in summary

    def test_flagged_assessments_in_summary(self):
        a = self._make_assessment(status=AssessmentStatus.flagged)
        summary, recs = _synthesize([a], None, "glucose spike")
        assert "Diabetes Risk" in summary

    def test_consult_rec_always_appended(self):
        _, recs = _synthesize([], None, "random text")
        assert any("healthcare professional" in r for r in recs)

    def test_recommendations_deduplicated(self):
        from backend.modules.orchestrator.schemas.analyze import RiskSummary
        a1 = ConditionAssessment(
            condition="diabetes", display_name="Diabetes Risk",
            status=AssessmentStatus.success,
            matched_keywords=["glucose"], detail_page="Diabetes_Risk",
            risk=RiskSummary(label="high", score=0.8, description="high"),
            recommendations=["See a doctor.", "Exercise daily."],
        )
        a2 = ConditionAssessment(
            condition="hypertension", display_name="Hypertension Risk",
            status=AssessmentStatus.success,
            matched_keywords=["blood pressure"], detail_page="Hypertension_Risk",
            risk=RiskSummary(label="moderate", score=0.4, description="moderate"),
            recommendations=["See a doctor.", "Reduce salt."],
        )
        _, recs = _synthesize([a1, a2], None, "text")
        assert recs.count("See a doctor.") == 1


# ── OrchestratorService.analyze — unit (no models loaded) ────────────────────

class TestOrchestratorServiceUnit:
    def setup_method(self):
        self.svc = OrchestratorService()

    def test_no_symptoms_returns_empty_assessments(self):
        # Provide a neutral text that matches no keywords
        report = self.svc.analyze(
            symptoms="I have a common cold with runny nose.",
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        assert isinstance(report, OrchestratorReport)
        assert len(report.condition_assessments) == 0
        assert report.image_assessment is None

    def test_diabetes_keywords_flagged_without_inputs(self):
        report = self.svc.analyze(
            symptoms="I have frequent urination and excessive thirst.",
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        diabetes_ca = next(
            (a for a in report.condition_assessments if a.condition == "diabetes"), None
        )
        assert diabetes_ca is not None
        assert diabetes_ca.status == AssessmentStatus.flagged

    def test_hypertension_flagged_without_systolic_bp(self):
        report = self.svc.analyze(
            symptoms="I have high blood pressure and dizziness.",
            age=45, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        htn = next(
            (a for a in report.condition_assessments if a.condition == "hypertension"), None
        )
        assert htn is not None
        assert htn.status == AssessmentStatus.flagged

    def test_heart_disease_always_flagged(self):
        # heart_disease has needs=[] but no model runner → always flagged
        report = self.svc.analyze(
            symptoms="I have chest pain and cardiac palpitation.",
            age=50, gender="male", bmi=28.0, glucose=120.0, systolic_bp=140.0,
            image_bytes=None, image_type=None,
        )
        hd = next(
            (a for a in report.condition_assessments if a.condition == "heart_disease"), None
        )
        assert hd is not None
        assert hd.status == AssessmentStatus.flagged

    def test_report_has_disclaimer(self):
        report = self.svc.analyze(
            symptoms="I have a runny nose and sneezing.",
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        assert len(report.disclaimer) > 10

    def test_processing_time_ms_is_non_negative(self):
        report = self.svc.analyze(
            symptoms="General fatigue and tiredness.",
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        assert report.processing_time_ms >= 0

    def test_inputs_used_echoed(self):
        report = self.svc.analyze(
            symptoms="I have frequent urination and excessive thirst.",
            age=40, gender="male", bmi=28.5, glucose=150.0, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        assert report.inputs_used.get("age") == 40
        assert report.inputs_used.get("glucose") == 150.0
        assert "systolic_bp" not in report.inputs_used

    def test_unknown_image_type_no_crash(self):
        # Service wraps image runner in try/except; unknown type logs warning
        report = self.svc.analyze(
            symptoms="I have chest pain.",
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=b"fake-image-bytes",
            image_type="unknown_type",
        )
        # Should return a report without crashing (image_assessment may be None)
        assert isinstance(report, OrchestratorReport)


# ── Edge-case / adversarial inputs (service-level) ───────────────────────────

class TestOrchestratorEdgeCases:
    def setup_method(self):
        self.svc = OrchestratorService()

    def _analyze(self, symptoms, **kwargs):
        defaults = dict(
            age=None, gender=None, bmi=None, glucose=None, systolic_bp=None,
            image_bytes=None, image_type=None,
        )
        defaults.update(kwargs)
        return self.svc.analyze(symptoms=symptoms, **defaults)

    def test_sql_injection_string_no_crash(self):
        """SQL injection in symptom text must not raise or corrupt state."""
        report = self._analyze("'; DROP TABLE predictions; --")
        assert isinstance(report, OrchestratorReport)

    def test_xss_string_no_crash(self):
        """XSS payload in symptom text must be handled safely."""
        report = self._analyze("<script>alert('xss')</script> blood pressure issues")
        assert isinstance(report, OrchestratorReport)

    def test_very_long_symptoms_no_crash(self):
        """2000-character symptom string (max allowed by router) must not crash."""
        report = self._analyze("I have frequent urination. " * 80)
        assert isinstance(report, OrchestratorReport)

    def test_unicode_symptoms_no_crash(self):
        report = self._analyze("我有高血糖 and glucose problems 🩺")
        assert isinstance(report, OrchestratorReport)

    def test_newlines_in_symptoms_no_crash(self):
        report = self._analyze("chest pain\nshortness of breath\ndizziness")
        assert isinstance(report, OrchestratorReport)

    def test_gender_invalid_value_ignored(self):
        """Invalid gender values should be cleaned to None without error."""
        report = self._analyze(
            "blood pressure headache", gender="attack_payload",
            age=40, systolic_bp=140.0,
        )
        assert isinstance(report, OrchestratorReport)

    def test_matched_keywords_list_populated(self):
        report = self._analyze("I have diabetes and insulin problems with hyperglycemia.")
        diabetes_ca = next(
            (a for a in report.condition_assessments if a.condition == "diabetes"), None
        )
        assert diabetes_ca is not None
        assert len(diabetes_ca.matched_keywords) >= 1


# ── Integration: HTTP endpoint ────────────────────────────────────────────────

@pytest.mark.integration
class TestOrchestratorEndpoint:
    def test_analyze_basic_post(self, client):
        resp = client.post(
            "/api/v1/orchestrator/analyze",
            data={"symptoms": "I have frequent urination and excessive thirst"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "condition_assessments" in data
        assert "overall_summary" in data
        assert "disclaimer" in data

    def test_analyze_too_short_symptoms_rejected(self, client):
        resp = client.post(
            "/api/v1/orchestrator/analyze",
            data={"symptoms": "sick"},
        )
        assert resp.status_code == 422

    def test_analyze_image_without_type_rejected(self, client):
        image_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header
        resp = client.post(
            "/api/v1/orchestrator/analyze",
            data={"symptoms": "I have chest pain and blood pressure issues"},
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_analyze_invalid_image_type_rejected(self, client):
        resp = client.post(
            "/api/v1/orchestrator/analyze",
            data={
                "symptoms": "I have chest pain and blood pressure issues",
                "image_type": "cat_photo",
            },
        )
        assert resp.status_code == 422

    def test_status_endpoint(self, client):
        resp = client.get("/api/v1/orchestrator/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "sub_modules" in data
        assert "rate_limit" in data
        assert "diabetes" in data["sub_modules"]
