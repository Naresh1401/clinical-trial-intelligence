"""
src/adverse_events/classifier.py
==================================
Classify adverse events by severity, causality, seriousness, and expectedness.

Two strategies:
  1. Rule-based — fast, zero-cost, uses NER + pattern matching
  2. LLM-based — higher accuracy, handles ambiguous narratives

Also provides MedDRA-style coding: maps AE terms → Preferred Term → SOC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger


# ── MedDRA-style mapping (subset for demo) ──────────────────────────────
MEDDRA_SOC_MAP: Dict[str, str] = {
    # Preferred Term → System Organ Class
    "nausea": "Gastrointestinal disorders",
    "vomiting": "Gastrointestinal disorders",
    "diarrhoea": "Gastrointestinal disorders",
    "diarrhea": "Gastrointestinal disorders",
    "constipation": "Gastrointestinal disorders",
    "pancreatitis": "Gastrointestinal disorders",
    "abdominal pain": "Gastrointestinal disorders",
    "neutropenia": "Blood and lymphatic system disorders",
    "anaemia": "Blood and lymphatic system disorders",
    "anemia": "Blood and lymphatic system disorders",
    "thrombocytopenia": "Blood and lymphatic system disorders",
    "fatigue": "General disorders",
    "asthenia": "General disorders",
    "infusion related reaction": "General disorders",
    "injection site reaction": "General disorders",
    "pyrexia": "General disorders",
    "rash": "Skin and subcutaneous tissue disorders",
    "pruritus": "Skin and subcutaneous tissue disorders",
    "dermatitis": "Skin and subcutaneous tissue disorders",
    "headache": "Nervous system disorders",
    "dizziness": "Nervous system disorders",
    "peripheral neuropathy": "Nervous system disorders",
    "confusional state": "Nervous system disorders",
    "ARIA edema": "Nervous system disorders",
    "ARIA hemorrhage": "Nervous system disorders",
    "syncope": "Nervous system disorders",
    "hypotension": "Vascular disorders",
    "hypertension": "Vascular disorders",
    "bradycardia": "Cardiac disorders",
    "tachycardia": "Cardiac disorders",
    "electrocardiogram QT prolonged": "Cardiac disorders",
    "pneumonitis": "Respiratory, thoracic and mediastinal disorders",
    "dyspnoea": "Respiratory, thoracic and mediastinal disorders",
    "cough": "Respiratory, thoracic and mediastinal disorders",
    "hepatotoxicity": "Hepatobiliary disorders",
    "transaminases increased": "Hepatobiliary disorders",
    "acute kidney injury": "Renal and urinary disorders",
    "hypoglycaemia": "Metabolism and nutrition disorders",
    "hyperkalaemia": "Metabolism and nutrition disorders",
    "decreased appetite": "Metabolism and nutrition disorders",
    "volume depletion": "Metabolism and nutrition disorders",
    "diabetic ketoacidosis": "Metabolism and nutrition disorders",
    "hyperglycaemia": "Metabolism and nutrition disorders",
    "urinary tract infection": "Infections and infestations",
    "insomnia": "Psychiatric disorders",
    "depression": "Psychiatric disorders",
    "anxiety": "Psychiatric disorders",
    "fall": "Injury, poisoning and procedural complications",
}

# Severity grading rules
SEVERITY_KEYWORDS = {
    1: ["mild", "grade 1", "slight", "minor"],
    2: ["moderate", "grade 2"],
    3: ["severe", "grade 3", "marked", "significant"],
    4: ["life-threatening", "life threatening", "grade 4", "disabling"],
    5: ["death", "fatal", "grade 5"],
}

# Causality assessment rules
CAUSALITY_KEYWORDS = {
    "Not Related": ["not related", "unrelated", "no relationship", "coincidental"],
    "Unlikely": ["unlikely", "doubtful", "improbable", "remote"],
    "Possible": ["possible", "may be related", "cannot be ruled out"],
    "Probable": ["probable", "likely", "plausible", "consistent with"],
    "Definite": ["definite", "certain", "clearly related", "re-challenge positive"],
}


@dataclass
class AEClassification:
    """Complete classification of an adverse event."""
    ae_term: str
    preferred_term: str
    system_organ_class: str
    severity_grade: int
    severity_label: str
    causality: str
    is_serious: bool
    seriousness_criteria: List[str]
    is_expected: bool
    confidence: float


class AEClassifier:
    """
    Classify adverse events from narratives.

    Assigns:
      - MedDRA coding (PT → SOC)
      - CTCAE severity grading (1-5)
      - Causality assessment
      - Seriousness determination
      - Expectedness (based on known AE profile)
    """

    def __init__(self, known_aes: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            known_aes: Dict mapping drug → list of expected AE terms
                       Used for expectedness classification.
        """
        self.known_aes = known_aes or {}

    def classify(self, ae_term: str, narrative: str = "",
                 drug: str = "") -> AEClassification:
        """Classify a single adverse event."""
        pt = self._map_to_preferred_term(ae_term)
        soc = MEDDRA_SOC_MAP.get(pt, "General disorders")
        grade, severity_label = self._assess_severity(narrative)
        causality = self._assess_causality(narrative)
        is_serious, criteria = self._assess_seriousness(narrative, grade)
        is_expected = self._assess_expectedness(ae_term, drug)

        confidence = 0.6
        if narrative:
            confidence += 0.2
        if grade > 0:
            confidence += 0.1
        if causality != "Unknown":
            confidence += 0.1

        return AEClassification(
            ae_term=ae_term,
            preferred_term=pt,
            system_organ_class=soc,
            severity_grade=grade,
            severity_label=severity_label,
            causality=causality,
            is_serious=is_serious,
            seriousness_criteria=criteria,
            is_expected=is_expected,
            confidence=min(confidence, 1.0),
        )

    def classify_dataframe(self, df: pd.DataFrame,
                           ae_col: str = "ae_term",
                           narrative_col: str = "narrative",
                           drug_col: str = "arm") -> pd.DataFrame:
        """Classify all AEs in a dataframe. Adds classification columns."""
        results = []
        for _, row in df.iterrows():
            ae = row.get(ae_col, "")
            narr = row.get(narrative_col, "")
            drug = row.get(drug_col, "")
            clf = self.classify(ae, narr, drug)
            results.append({
                "classified_pt": clf.preferred_term,
                "classified_soc": clf.system_organ_class,
                "classified_grade": clf.severity_grade,
                "classified_causality": clf.causality,
                "classified_serious": clf.is_serious,
                "classified_expected": clf.is_expected,
                "classification_confidence": clf.confidence,
            })
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(results)], axis=1)

    def _map_to_preferred_term(self, ae_term: str) -> str:
        """Map raw AE term to MedDRA preferred term."""
        normalized = ae_term.lower().strip()
        if normalized in MEDDRA_SOC_MAP:
            return normalized
        # Fuzzy matching on substrings
        for pt in MEDDRA_SOC_MAP:
            if pt in normalized or normalized in pt:
                return pt
        return normalized

    def _assess_severity(self, narrative: str) -> Tuple[int, str]:
        """Extract severity grade from narrative text."""
        text = narrative.lower()
        for grade in [5, 4, 3, 2, 1]:
            for kw in SEVERITY_KEYWORDS[grade]:
                if kw in text:
                    labels = {1: "Mild", 2: "Moderate", 3: "Severe",
                              4: "Life-threatening", 5: "Fatal"}
                    return grade, labels[grade]
        return 0, "Unknown"

    def _assess_causality(self, narrative: str) -> str:
        """Extract causality from narrative text."""
        text = narrative.lower()
        for assessment, keywords in CAUSALITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return assessment
        return "Unknown"

    def _assess_seriousness(self, narrative: str, grade: int) -> Tuple[bool, List[str]]:
        """Determine if AE is serious (SAE)."""
        text = narrative.lower()
        criteria = []

        sae_checks = {
            "Results in death": ["death", "fatal", "deceased", "died"],
            "Life-threatening": ["life-threatening", "life threatening"],
            "Requires hospitalization": ["hospitalized", "hospitalised", "hospitalization",
                                         "hospital admission", "admitted to"],
            "Prolongs hospitalization": ["prolonged hospitalization", "extended stay"],
            "Results in disability": ["disability", "incapacity", "permanent damage"],
            "Congenital anomaly": ["congenital", "birth defect"],
            "Medically important": ["medically important", "medically significant",
                                    "ICU", "intensive care", "emergency"],
        }

        for criterion, keywords in sae_checks.items():
            if any(kw in text for kw in keywords):
                criteria.append(criterion)

        is_serious = len(criteria) > 0 or grade >= 4
        return is_serious, criteria

    def _assess_expectedness(self, ae_term: str, drug: str) -> bool:
        """Check if AE is expected (listed in known safety profile)."""
        if not drug or not self.known_aes:
            return True  # Default to expected if no reference
        known = self.known_aes.get(drug, [])
        return ae_term.lower() in [k.lower() for k in known]

    def generate_ae_table(self, ae_df: pd.DataFrame,
                          group_by: str = "system_organ_class") -> pd.DataFrame:
        """
        Generate a clinical-style AE summary table (Table 14.3.1 format).

        Columns: SOC / PT, Treatment N(%), Control N(%), Total N(%)
        """
        if "arm" not in ae_df.columns:
            return ae_df

        groups = ae_df.groupby([group_by, "arm"]).size().unstack(fill_value=0)
        total_treat = len(ae_df[ae_df["arm"] == "Treatment"])
        total_ctrl = len(ae_df[ae_df["arm"] == "Control"])

        result = groups.copy()
        if "Treatment" in result.columns:
            result["Treatment_pct"] = (result["Treatment"] / total_treat * 100).round(1)
        if "Control" in result.columns:
            result["Control_pct"] = (result["Control"] / total_ctrl * 100).round(1)
        result["Total"] = result.get("Treatment", 0) + result.get("Control", 0)

        return result.sort_values("Total", ascending=False)
