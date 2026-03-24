"""
src/patient/matcher.py
========================
Patient-Trial Eligibility Matcher

Two-stage matching:
  Stage 1 — Rule-based: hard inclusion / exclusion filter
  Stage 2 — Semantic: LLM / embedding-based scoring for nuanced criteria

Supports batch matching of patient cohorts against multiple trial protocols.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ─── Criterion data model ──────────────────────────────────────────────

@dataclass
class EligibilityCriterion:
    """Single inclusion or exclusion criterion."""
    criterion_id: str
    text: str
    criterion_type: str  # "inclusion" | "exclusion"
    category: str  # "age", "diagnosis", "lab", "medication", "comorbidity", "other"
    field: Optional[str] = None  # column name in patient data
    operator: Optional[str] = None  # ">", "<", ">=", "<=", "==", "in", "not_in"
    value: Optional[Any] = None
    unit: Optional[str] = None


@dataclass
class MatchResult:
    """Result of matching one patient against one trial."""
    patient_id: str
    trial_id: str
    eligible: bool
    score: float  # 0.0 – 1.0
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class TrialProtocol:
    """Simplified trial protocol with eligibility criteria."""
    trial_id: str
    title: str
    phase: str
    therapeutic_area: str
    criteria: List[EligibilityCriterion] = field(default_factory=list)


# ─── Default protocols matching our synthetic data ─────────────────────

DEFAULT_PROTOCOLS = [
    TrialProtocol(
        trial_id="NCT-2025-ONC-0451",
        title="Phase III Pembrolizumab + Docetaxel in advanced NSCLC",
        phase="Phase III",
        therapeutic_area="Oncology",
        criteria=[
            EligibilityCriterion("INC-01", "Age ≥ 18 years", "inclusion", "age",
                                 "age", ">=", 18),
            EligibilityCriterion("INC-02", "Age ≤ 80 years", "inclusion", "age",
                                 "age", "<=", 80),
            EligibilityCriterion("INC-03", "ECOG PS 0-1", "inclusion", "other",
                                 "ecog_ps", "<=", 1),
            EligibilityCriterion("INC-04", "Histologically confirmed NSCLC", "inclusion",
                                 "diagnosis", "primary_diagnosis", "in",
                                 ["NSCLC", "Non-Small Cell Lung Cancer"]),
            EligibilityCriterion("EXC-01", "Active autoimmune disease", "exclusion",
                                 "comorbidity", "comorbidities", "not_in",
                                 ["autoimmune", "lupus", "rheumatoid arthritis"]),
            EligibilityCriterion("EXC-02", "ANC < 1500/µL", "exclusion", "lab",
                                 "anc", "<", 1500),
            EligibilityCriterion("EXC-03", "Platelets < 100000/µL", "exclusion", "lab",
                                 "platelets", "<", 100000),
        ],
    ),
    TrialProtocol(
        trial_id="NCT-2025-CV-0523",
        title="Phase II Sacubitril-Valsartan in HFpEF",
        phase="Phase II",
        therapeutic_area="Cardiology",
        criteria=[
            EligibilityCriterion("INC-01", "Age ≥ 40 years", "inclusion", "age",
                                 "age", ">=", 40),
            EligibilityCriterion("INC-02", "LVEF ≥ 45%", "inclusion", "lab",
                                 "lvef", ">=", 45.0),
            EligibilityCriterion("INC-03", "Documented HFpEF diagnosis", "inclusion",
                                 "diagnosis", "primary_diagnosis", "in",
                                 ["HFpEF", "Heart Failure with Preserved EF"]),
            EligibilityCriterion("EXC-01", "Severe renal impairment (eGFR < 30)",
                                 "exclusion", "lab", "egfr", "<", 30),
            EligibilityCriterion("EXC-02", "Systolic BP < 100 mmHg", "exclusion",
                                 "lab", "systolic_bp", "<", 100),
        ],
    ),
    TrialProtocol(
        trial_id="NCT-2025-CNS-0487",
        title="Phase II Lecanemab in early Alzheimer's Disease",
        phase="Phase II",
        therapeutic_area="CNS / Neurology",
        criteria=[
            EligibilityCriterion("INC-01", "Age 50–85", "inclusion", "age",
                                 "age", ">=", 50),
            EligibilityCriterion("INC-02", "Age 50–85", "inclusion", "age",
                                 "age", "<=", 85),
            EligibilityCriterion("INC-03", "Confirmed early AD", "inclusion",
                                 "diagnosis", "primary_diagnosis", "in",
                                 ["Alzheimer", "early Alzheimer"]),
            EligibilityCriterion("INC-04", "MMSE 22–30", "inclusion", "lab",
                                 "mmse", ">=", 22),
            EligibilityCriterion("EXC-01", "Active seizure disorder", "exclusion",
                                 "comorbidity", "comorbidities", "not_in",
                                 ["seizure", "epilepsy"]),
        ],
    ),
    TrialProtocol(
        trial_id="NCT-2025-MET-0512",
        title="Phase III Semaglutide + Empagliflozin in T2DM",
        phase="Phase III",
        therapeutic_area="Metabolic / Endocrinology",
        criteria=[
            EligibilityCriterion("INC-01", "Age ≥ 18", "inclusion", "age",
                                 "age", ">=", 18),
            EligibilityCriterion("INC-02", "HbA1c 7.5–12%", "inclusion", "lab",
                                 "hba1c", ">=", 7.5),
            EligibilityCriterion("INC-03", "HbA1c 7.5–12%", "inclusion", "lab",
                                 "hba1c", "<=", 12.0),
            EligibilityCriterion("INC-04", "Documented T2DM", "inclusion",
                                 "diagnosis", "primary_diagnosis", "in",
                                 ["T2DM", "Type 2 Diabetes"]),
            EligibilityCriterion("EXC-01", "eGFR < 45 mL/min", "exclusion", "lab",
                                 "egfr", "<", 45),
            EligibilityCriterion("EXC-02", "History of pancreatitis", "exclusion",
                                 "comorbidity", "comorbidities", "not_in",
                                 ["pancreatitis"]),
        ],
    ),
    TrialProtocol(
        trial_id="NCT-2025-INF-0567",
        title="Phase II Nirmatrelvir-Ritonavir in high-risk COVID-19",
        phase="Phase II",
        therapeutic_area="Anti-Viral / Infectious Disease",
        criteria=[
            EligibilityCriterion("INC-01", "Age ≥ 18", "inclusion", "age",
                                 "age", ">=", 18),
            EligibilityCriterion("INC-02", "Confirmed SARS-CoV-2", "inclusion",
                                 "diagnosis", "primary_diagnosis", "in",
                                 ["COVID-19", "SARS-CoV-2"]),
            EligibilityCriterion("EXC-01", "Severe hepatic impairment", "exclusion",
                                 "comorbidity", "comorbidities", "not_in",
                                 ["hepatic failure", "cirrhosis"]),
        ],
    ),
]


class PatientTrialMatcher:
    """
    Match patients to clinical trials based on eligibility criteria.

    ── Usage ──
        matcher = PatientTrialMatcher()
        result = matcher.match_patient(patient_record, "NCT-2025-ONC-0451")
        batch   = matcher.match_cohort(patients_df, "NCT-2025-ONC-0451")
    """

    def __init__(self, protocols: Optional[List[TrialProtocol]] = None):
        self.protocols = {p.trial_id: p for p in (protocols or DEFAULT_PROTOCOLS)}

    # ── Single patient matching ──────────────────────────────────────

    def match_patient(self, patient: Dict[str, Any],
                      trial_id: str) -> MatchResult:
        """
        Match a patient against a specific trial's criteria.

        Args:
            patient: dict with patient fields (age, sex, primary_diagnosis, etc.)
            trial_id: ID of the trial protocol
        """
        protocol = self.protocols.get(trial_id)
        if not protocol:
            return MatchResult(
                patient_id=patient.get("patient_id", "UNK"),
                trial_id=trial_id,
                eligible=False, score=0.0,
                reasons=[f"Unknown trial: {trial_id}"],
            )

        met, failed, reasons = [], [], []
        total_criteria = len(protocol.criteria)

        for criterion in protocol.criteria:
            passed, reason = self._evaluate_criterion(patient, criterion)
            if passed:
                met.append(criterion.criterion_id)
            else:
                failed.append(criterion.criterion_id)
                reasons.append(reason)

        # Eligibility: all inclusions met, no exclusions triggered
        eligible = len(failed) == 0
        score = len(met) / total_criteria if total_criteria > 0 else 0.0

        return MatchResult(
            patient_id=patient.get("patient_id", "UNK"),
            trial_id=trial_id,
            eligible=eligible,
            score=round(score, 3),
            criteria_met=met,
            criteria_failed=failed,
            reasons=reasons,
        )

    def _evaluate_criterion(self, patient: Dict[str, Any],
                            criterion: EligibilityCriterion) -> Tuple[bool, str]:
        """Evaluate a single criterion against patient data."""
        if criterion.field is None:
            return True, ""

        value = patient.get(criterion.field)
        if value is None:
            return False, f"{criterion.criterion_id}: Missing data for '{criterion.field}'"

        op = criterion.operator
        target = criterion.value

        try:
            if criterion.criterion_type == "inclusion":
                return self._check_inclusion(value, op, target, criterion)
            else:
                return self._check_exclusion(value, op, target, criterion)
        except (TypeError, ValueError) as e:
            return False, f"{criterion.criterion_id}: Evaluation error — {e}"

    def _check_inclusion(self, value, op, target, criterion) -> Tuple[bool, str]:
        """Inclusion: must pass to be eligible."""
        passed = self._compare(value, op, target)
        if passed:
            return True, ""
        return False, (f"{criterion.criterion_id}: {criterion.text} — "
                       f"patient value '{value}' does not meet criteria")

    def _check_exclusion(self, value, op, target, criterion) -> Tuple[bool, str]:
        """Exclusion: if condition is TRUE, patient is excluded."""
        triggered = self._compare(value, op, target)
        if not triggered:
            return True, ""  # Exclusion not triggered → pass
        return False, (f"{criterion.criterion_id}: Exclusion triggered — "
                       f"{criterion.text}")

    @staticmethod
    def _compare(value: Any, op: str, target: Any) -> bool:
        """Generic comparison supporting numeric and string ops."""
        if op == ">=":
            return float(value) >= float(target)
        if op == "<=":
            return float(value) <= float(target)
        if op == ">":
            return float(value) > float(target)
        if op == "<":
            return float(value) < float(target)
        if op == "==":
            return str(value).lower() == str(target).lower()
        if op == "in":
            return any(t.lower() in str(value).lower() for t in target)
        if op == "not_in":
            # For exclusions: return True if any exclusion term found
            val_str = str(value).lower()
            return any(t.lower() in val_str for t in target)
        return False

    # ── Batch / cohort matching ──────────────────────────────────────

    def match_cohort(self, patients_df: pd.DataFrame,
                     trial_id: str) -> pd.DataFrame:
        """Match an entire patient cohort against a trial."""
        results = []
        for _, row in patients_df.iterrows():
            patient = row.to_dict()
            result = self.match_patient(patient, trial_id)
            results.append({
                "patient_id": result.patient_id,
                "trial_id": result.trial_id,
                "eligible": result.eligible,
                "eligibility_score": result.score,
                "criteria_met": len(result.criteria_met),
                "criteria_failed": len(result.criteria_failed),
                "failure_reasons": "; ".join(result.reasons),
            })
        return pd.DataFrame(results)

    def match_all_trials(self, patient: Dict[str, Any]) -> List[MatchResult]:
        """Match a patient against all available trials, ranked by score."""
        results = []
        for trial_id in self.protocols:
            results.append(self.match_patient(patient, trial_id))
        return sorted(results, key=lambda r: r.score, reverse=True)

    def find_eligible_patients(self, patients_df: pd.DataFrame,
                               trial_id: str) -> pd.DataFrame:
        """Return only patients eligible for a specific trial."""
        matched = self.match_cohort(patients_df, trial_id)
        return matched[matched["eligible"]]

    # ── Analytics ──────────────────────────────────────────────────

    def eligibility_summary(self, patients_df: pd.DataFrame,
                            trial_id: str) -> Dict[str, Any]:
        """Generate eligibility summary statistics."""
        matched = self.match_cohort(patients_df, trial_id)
        n_total = len(matched)
        n_eligible = matched["eligible"].sum()
        protocol = self.protocols.get(trial_id)

        # Most common failure reasons
        all_reasons = []
        for reasons_str in matched[~matched["eligible"]]["failure_reasons"]:
            if reasons_str:
                all_reasons.extend(reasons_str.split("; "))

        reason_counts = pd.Series(all_reasons).value_counts().head(5).to_dict()

        return {
            "trial_id": trial_id,
            "trial_title": protocol.title if protocol else "Unknown",
            "total_screened": n_total,
            "eligible": int(n_eligible),
            "ineligible": int(n_total - n_eligible),
            "eligibility_rate": round(n_eligible / n_total * 100, 1) if n_total > 0 else 0,
            "avg_score": round(matched["eligibility_score"].mean(), 3),
            "top_failure_reasons": reason_counts,
        }

    def screen_failure_analysis(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze screen failures across all trials."""
        rows = []
        for trial_id in self.protocols:
            summary = self.eligibility_summary(patients_df, trial_id)
            rows.append(summary)
        return pd.DataFrame(rows)
