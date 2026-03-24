"""
tests/test_pipeline.py
========================
Integration tests for the Clinical Trial Intelligence Platform.

Tests cover:
  - Data generation
  - NER extraction
  - AE detection and classification
  - Patient-trial matching
  - Protocol deviation detection
  - API endpoints
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nlp.medical_ner import MedicalNER, MedicalEntity
from src.adverse_events.detector import AEDetector
from src.adverse_events.classifier import AEClassifier
from src.patient.matcher import PatientTrialMatcher
from src.protocol.analyzer import ProtocolAnalyzer


# ═══ Fixtures ════════════════════════════════════════════════════════

@pytest.fixture
def ner():
    return MedicalNER()


@pytest.fixture
def ae_detector():
    return AEDetector()


@pytest.fixture
def ae_classifier():
    return AEClassifier()


@pytest.fixture
def matcher():
    return PatientTrialMatcher()


@pytest.fixture
def protocol_analyzer():
    return ProtocolAnalyzer()


@pytest.fixture
def sample_narrative():
    return (
        "Patient developed Grade 3 neutropenia on Day 14 of pembrolizumab therapy. "
        "ANC nadir was 800/µL. The event was assessed as probably related to docetaxel. "
        "Patient was hospitalized for 3 days and received G-CSF support."
    )


@pytest.fixture
def sample_patient():
    return {
        "patient_id": "PAT-001",
        "age": 62,
        "sex": "Male",
        "primary_diagnosis": "NSCLC",
        "ecog_ps": 1,
        "anc": 2000,
        "platelets": 150000,
        "comorbidities": "hypertension, diabetes",
    }


@pytest.fixture
def sample_visits():
    return pd.DataFrame([
        {"patient_id": "PAT-001", "trial_id": "NCT-2025-ONC-0451",
         "visit_number": i, "visit_date": f"2025-{i:02d}-15"}
        for i in range(1, 8)
    ])


# ═══ NER Tests ═══════════════════════════════════════════════════════

class TestMedicalNER:

    def test_extract_drugs(self, ner):
        result = ner.extract("Patient received pembrolizumab 200mg IV.")
        drugs = result.drugs
        assert len(drugs) >= 1
        assert any("pembrolizumab" in d.text.lower() for d in drugs)

    def test_extract_conditions(self, ner):
        result = ner.extract("Diagnosed with neutropenia and nausea.")
        conditions = result.conditions
        assert len(conditions) >= 1

    def test_extract_severity(self, ner):
        result = ner.extract("Grade 3 neutropenia observed.")
        severities = result.severities
        assert len(severities) >= 1
        assert any("3" in s.text for s in severities)

    def test_extract_causality(self, ner):
        result = ner.extract("The event was probably related to the study drug.")
        causalities = result.causalities
        assert len(causalities) >= 1

    def test_extract_lab_values(self, ner):
        result = ner.extract("ANC was 800/µL and ALT elevated to 150 U/L.")
        labs = result.lab_tests
        assert len(labs) >= 1

    def test_empty_text(self, ner):
        result = ner.extract("")
        assert len(result.entities) == 0

    def test_batch_extract(self, ner):
        texts = ["pembrolizumab therapy", "Grade 2 nausea"]
        results = ner.extract_batch(texts)
        assert len(results) == 2


# ═══ AE Detection Tests ═════════════════════════════════════════════

class TestAEDetector:

    def test_detect_from_narrative(self, ae_detector, sample_narrative):
        results = ae_detector.detect_from_narrative(sample_narrative)
        assert len(results) >= 1
        terms = [r.ae_term.lower() for r in results]
        assert any("neutropenia" in t for t in terms)

    def test_detect_severity(self, ae_detector, sample_narrative):
        results = ae_detector.detect_from_narrative(sample_narrative)
        assert any(r.severity for r in results)

    def test_detect_causality(self, ae_detector, sample_narrative):
        results = ae_detector.detect_from_narrative(sample_narrative)
        assert any(r.causality for r in results)

    def test_empty_narrative(self, ae_detector):
        results = ae_detector.detect_from_narrative("")
        assert len(results) == 0

    def test_safety_signal_detection(self, ae_detector):
        ae_df = pd.DataFrame({
            "preferred_term": ["nausea"] * 30 + ["nausea"] * 10 +
                              ["headache"] * 15 + ["headache"] * 14,
            "arm": ["Treatment"] * 30 + ["Control"] * 10 +
                   ["Treatment"] * 15 + ["Control"] * 14,
        })
        signals = ae_detector.safety_signal_detection(ae_df)
        assert len(signals) >= 1
        # Nausea should be flagged (30 vs 10)
        nausea_sig = [s for s in signals if s.ae_term == "nausea"]
        assert len(nausea_sig) >= 1
        assert nausea_sig[0].prr > 1.0


# ═══ AE Classification Tests ════════════════════════════════════════

class TestAEClassifier:

    def test_classify_basic(self, ae_classifier):
        clf = ae_classifier.classify("nausea", "Patient had moderate nausea.")
        assert clf.preferred_term == "nausea"
        assert clf.system_organ_class == "Gastrointestinal disorders"
        assert clf.confidence > 0

    def test_classify_severity(self, ae_classifier):
        clf = ae_classifier.classify("neutropenia",
                                     "Grade 3 neutropenia, ANC 800/µL")
        assert clf.severity_grade == 3
        assert clf.severity_label == "Severe"

    def test_classify_seriousness(self, ae_classifier):
        clf = ae_classifier.classify("neutropenia",
                                     "Patient was hospitalized due to febrile neutropenia")
        assert clf.is_serious is True
        assert "Requires hospitalization" in clf.seriousness_criteria

    def test_classify_causality(self, ae_classifier):
        clf = ae_classifier.classify("rash",
                                     "Rash was probable, likely related to pembrolizumab")
        assert clf.causality == "Probable"


# ═══ Patient Matching Tests ══════════════════════════════════════════

class TestPatientMatcher:

    def test_match_eligible_patient(self, matcher, sample_patient):
        result = matcher.match_patient(sample_patient, "NCT-2025-ONC-0451")
        assert result.patient_id == "PAT-001"
        assert result.score > 0

    def test_match_ineligible_by_age(self, matcher):
        patient = {
            "patient_id": "PAT-002",
            "age": 15,  # Below minimum age
            "primary_diagnosis": "NSCLC",
            "ecog_ps": 1,
            "anc": 2000,
            "platelets": 150000,
        }
        result = matcher.match_patient(patient, "NCT-2025-ONC-0451")
        assert result.eligible is False
        assert any("age" in r.lower() or "INC-01" in r for r in result.reasons)

    def test_match_all_trials(self, matcher, sample_patient):
        results = matcher.match_all_trials(sample_patient)
        assert len(results) > 0
        # Results should be sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_match_cohort(self, matcher):
        patients_df = pd.DataFrame([
            {"patient_id": "P1", "age": 55, "primary_diagnosis": "NSCLC",
             "ecog_ps": 0, "anc": 3000, "platelets": 200000, "comorbidities": ""},
            {"patient_id": "P2", "age": 16, "primary_diagnosis": "NSCLC",
             "ecog_ps": 0, "anc": 3000, "platelets": 200000, "comorbidities": ""},
        ])
        results = matcher.match_cohort(patients_df, "NCT-2025-ONC-0451")
        assert len(results) == 2

    def test_unknown_trial(self, matcher, sample_patient):
        result = matcher.match_patient(sample_patient, "NCT-FAKE")
        assert result.eligible is False


# ═══ Protocol Analyzer Tests ═════════════════════════════════════════

class TestProtocolAnalyzer:

    def test_detect_missed_visits(self, protocol_analyzer):
        visits = pd.DataFrame([
            {"patient_id": "P1", "trial_id": "NCT-2025-ONC-0451",
             "visit_number": i, "visit_date": f"2025-{i:02d}-15"}
            for i in range(1, 5)  # Only 4 of 12 visits
        ])
        devs = protocol_analyzer.detect_deviations(visits, "NCT-2025-ONC-0451")
        assert len(devs) >= 1
        assert any(d.deviation_type == "Missed Visit" for d in devs)

    def test_compliance_score(self, protocol_analyzer, sample_visits):
        score = protocol_analyzer.compliance_score(
            "PAT-001", sample_visits, "NCT-2025-ONC-0451"
        )
        assert 0 <= score.overall_score <= 1
        assert score.patient_id == "PAT-001"

    def test_study_metrics(self, protocol_analyzer, sample_visits):
        metrics = protocol_analyzer.study_metrics(
            sample_visits, "NCT-2025-ONC-0451"
        )
        assert metrics.enrollment_count >= 1
        assert metrics.trial_id == "NCT-2025-ONC-0451"

    def test_empty_visits(self, protocol_analyzer):
        visits = pd.DataFrame(columns=["patient_id", "trial_id", "visit_number"])
        devs = protocol_analyzer.detect_deviations(visits, "NCT-2025-ONC-0451")
        assert len(devs) == 0


# ═══ API Tests ═══════════════════════════════════════════════════════

class TestAPI:
    """Test FastAPI endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        try:
            from fastapi.testclient import TestClient
            from src.api.main import app
            self.client = TestClient(app)
            self.available = True
        except ImportError:
            self.available = False

    def test_health(self):
        if not self.available:
            pytest.skip("FastAPI not installed")
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_list_trials(self):
        if not self.available:
            pytest.skip("FastAPI not installed")
        resp = self.client.get("/trials")
        assert resp.status_code == 200
        trials = resp.json()
        assert len(trials) >= 1

    def test_ae_detect(self):
        if not self.available:
            pytest.skip("FastAPI not installed")
        resp = self.client.post("/ae/detect", json={
            "text": "Patient developed Grade 3 neutropenia after pembrolizumab."
        })
        assert resp.status_code == 200
        assert resp.json()["ae_count"] >= 1

    def test_ae_classify(self):
        if not self.available:
            pytest.skip("FastAPI not installed")
        resp = self.client.post("/ae/classify", json={
            "ae_term": "nausea",
            "narrative": "Moderate nausea, probably related to docetaxel",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["preferred_term"] == "nausea"
        assert data["system_organ_class"] == "Gastrointestinal disorders"

    def test_patient_match(self):
        if not self.available:
            pytest.skip("FastAPI not installed")
        resp = self.client.post("/patient/match", json={
            "patient": {
                "patient_id": "TEST-01", "age": 55,
                "primary_diagnosis": "NSCLC", "ecog_ps": 0,
                "anc": 3000, "platelets": 200000, "comorbidities": "",
            },
            "trial_id": "NCT-2025-ONC-0451",
        })
        assert resp.status_code == 200
        assert "eligible" in resp.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
