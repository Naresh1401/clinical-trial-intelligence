"""
src/api/main.py
=================
FastAPI backend for Clinical Trial Intelligence Platform.

Endpoints:
  POST /ae/detect              — Detect AEs from narrative text
  POST /ae/classify            — Classify an AE term
  POST /ae/signals             — Disproportionality signal detection
  POST /patient/match          — Match patient → trial eligibility
  POST /patient/match-cohort   — Batch match cohort → trial
  POST /protocol/deviations    — Detect protocol deviations
  POST /protocol/compliance    — Patient compliance score
  GET  /trials                 — List available trials
  GET  /health                 — Health check
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nlp.medical_ner import MedicalNER
from src.adverse_events.detector import AEDetector
from src.adverse_events.classifier import AEClassifier
from src.patient.matcher import PatientTrialMatcher, DEFAULT_PROTOCOLS
from src.protocol.analyzer import ProtocolAnalyzer

# ── App setup ──────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Trial Intelligence API",
    description="NLP-powered clinical trial analysis: AE detection, patient matching, protocol compliance",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize services ─────────────────────────────────────────────

ner = MedicalNER()
ae_detector = AEDetector()
ae_classifier = AEClassifier()
patient_matcher = PatientTrialMatcher()
protocol_analyzer = ProtocolAnalyzer()

logger.info("All services initialized")


# ── Request / Response models ────────────────────────────────────────

class NarrativeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Clinical narrative text")

class AEClassifyRequest(BaseModel):
    ae_term: str = Field(..., min_length=1, max_length=500)
    narrative: str = ""
    drug: str = ""

class AESignalRequest(BaseModel):
    ae_data: List[Dict[str, Any]] = Field(..., description="List of AE records with 'ae_term' and 'arm' fields")

class PatientMatchRequest(BaseModel):
    patient: Dict[str, Any] = Field(..., description="Patient record dict")
    trial_id: str

class CohortMatchRequest(BaseModel):
    patients: List[Dict[str, Any]]
    trial_id: str

class DeviationRequest(BaseModel):
    visits: List[Dict[str, Any]]
    trial_id: str

class ComplianceRequest(BaseModel):
    patient_id: str
    visits: List[Dict[str, Any]]
    trial_id: str


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


# ── Trial Info ───────────────────────────────────────────────────────

@app.get("/trials")
def list_trials():
    """List all available trial protocols."""
    return [
        {
            "trial_id": p.trial_id,
            "title": p.title,
            "phase": p.phase,
            "therapeutic_area": p.therapeutic_area,
            "criteria_count": len(p.criteria),
        }
        for p in DEFAULT_PROTOCOLS
    ]


# ── AE Detection ────────────────────────────────────────────────────

@app.post("/ae/detect")
def detect_adverse_events(req: NarrativeRequest):
    """Detect adverse events from clinical narrative text."""
    results = ae_detector.detect_from_narrative(req.text)
    return {
        "narrative_length": len(req.text),
        "ae_count": len(results),
        "adverse_events": [
            {
                "ae_term": ae.ae_term,
                "severity": ae.severity,
                "causality": ae.causality,
                "confidence": ae.confidence,
                "drugs_mentioned": ae.drugs,
            }
            for ae in results
        ],
    }


@app.post("/ae/classify")
def classify_adverse_event(req: AEClassifyRequest):
    """Classify an AE by severity, causality, seriousness, expectedness."""
    clf = ae_classifier.classify(req.ae_term, req.narrative, req.drug)
    return {
        "ae_term": clf.ae_term,
        "preferred_term": clf.preferred_term,
        "system_organ_class": clf.system_organ_class,
        "severity_grade": clf.severity_grade,
        "severity_label": clf.severity_label,
        "causality": clf.causality,
        "is_serious": clf.is_serious,
        "seriousness_criteria": clf.seriousness_criteria,
        "is_expected": clf.is_expected,
        "confidence": clf.confidence,
    }


@app.post("/ae/signals")
def detect_safety_signals(req: AESignalRequest):
    """Run disproportionality analysis for safety signal detection."""
    if not req.ae_data:
        raise HTTPException(status_code=400, detail="ae_data cannot be empty")
    ae_df = pd.DataFrame(req.ae_data)
    if "ae_term" not in ae_df.columns or "arm" not in ae_df.columns:
        raise HTTPException(status_code=400,
                            detail="ae_data records must have 'ae_term' and 'arm' fields")
    signals = ae_detector.safety_signal_detection(ae_df)
    return {
        "total_ae_records": len(ae_df),
        "signals_detected": len(signals),
        "signals": [
            {
                "ae_term": s.ae_term,
                "prr": s.prr,
                "ror": s.ror,
                "chi_squared": s.chi_squared,
                "is_signal": s.is_signal,
                "treatment_count": s.treatment_count,
                "control_count": s.control_count,
            }
            for s in signals
        ],
    }


# ── Patient Matching ────────────────────────────────────────────────

@app.post("/patient/match")
def match_patient(req: PatientMatchRequest):
    """Match a single patient against a trial's eligibility criteria."""
    result = patient_matcher.match_patient(req.patient, req.trial_id)
    return {
        "patient_id": result.patient_id,
        "trial_id": result.trial_id,
        "eligible": result.eligible,
        "score": result.score,
        "criteria_met": result.criteria_met,
        "criteria_failed": result.criteria_failed,
        "reasons": result.reasons,
    }


@app.post("/patient/match-cohort")
def match_cohort(req: CohortMatchRequest):
    """Match a patient cohort against a trial."""
    if not req.patients:
        raise HTTPException(status_code=400, detail="patients list cannot be empty")
    df = pd.DataFrame(req.patients)
    results = patient_matcher.match_cohort(df, req.trial_id)
    return {
        "trial_id": req.trial_id,
        "total_screened": len(results),
        "eligible": int(results["eligible"].sum()),
        "eligibility_rate": round(results["eligible"].mean() * 100, 1),
        "results": results.to_dict(orient="records"),
    }


# ── Protocol Compliance ─────────────────────────────────────────────

@app.post("/protocol/deviations")
def detect_deviations(req: DeviationRequest):
    """Detect protocol deviations from visit data."""
    if not req.visits:
        raise HTTPException(status_code=400, detail="visits list cannot be empty")
    visits_df = pd.DataFrame(req.visits)
    deviations = protocol_analyzer.detect_deviations(visits_df, req.trial_id)
    return {
        "trial_id": req.trial_id,
        "deviation_count": len(deviations),
        "deviations": [
            {
                "patient_id": d.patient_id,
                "type": d.deviation_type,
                "severity": d.severity,
                "description": d.description,
            }
            for d in deviations
        ],
    }


@app.post("/protocol/compliance")
def patient_compliance(req: ComplianceRequest):
    """Calculate protocol compliance score for a patient."""
    visits_df = pd.DataFrame(req.visits)
    score = protocol_analyzer.compliance_score(
        req.patient_id, visits_df, req.trial_id,
    )
    return {
        "patient_id": score.patient_id,
        "trial_id": score.trial_id,
        "overall_score": score.overall_score,
        "visit_compliance": score.visit_compliance,
        "visit_window_compliance": score.visit_window_compliance,
        "lab_compliance": score.lab_compliance,
        "deviation_count": len(score.deviations),
    }


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
