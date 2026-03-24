"""
src/adverse_events/detector.py
===============================
Adverse Event detection and signal analysis from clinical narratives.

Capabilities:
  - Extract AE signals from free-text narratives
  - Detect potential safety signals (disproportionality analysis)
  - Flag narratives with high-severity or unexpected AEs
  - Aggregate AE frequencies by SOC, PT, severity, causality

Uses NLP-based entity extraction + statistical signal detection (PRR, ROR).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from src.nlp.medical_ner import MedicalNER, NERResult


@dataclass
class DetectedAE:
    """A single adverse event extracted from a narrative."""
    ae_term: str
    severity: Optional[str] = None
    causality: Optional[str] = None
    drugs: List[str] = field(default_factory=list)
    lab_abnormalities: List[str] = field(default_factory=list)
    is_serious: bool = False
    confidence: float = 0.0
    source_text: str = ""


@dataclass
class SafetySignal:
    """A detected safety signal from disproportionality analysis."""
    ae_term: str
    drug: str
    prr: float           # Proportional Reporting Ratio
    prr_ci_lower: float
    prr_ci_upper: float
    ror: float            # Reporting Odds Ratio
    ror_ci_lower: float
    ror_ci_upper: float
    chi2: float
    p_value: float
    is_signal: bool       # PRR >= 2, chi2 >= 4, N >= 3
    n_cases: int
    total_ae: int


class AEDetector:
    """
    Detects and classifies adverse events from clinical narratives.

    Pipeline:
      1. NER extraction → drugs, conditions, severity, causality
      2. AE term matching and classification
      3. Seriousness assessment (SAE criteria detection)
      4. Signal detection using disproportionality analysis
    """

    SAE_KEYWORDS = [
        "death", "fatal", "life-threatening", "life threatening",
        "hospitalization", "hospitalisation", "hospitalized", "hospitalised",
        "disability", "incapacity", "congenital", "birth defect",
        "medically important", "medically significant",
        "ICU", "intensive care",
    ]

    HIGH_RISK_TERMS = [
        "anaphylaxis", "cardiac arrest", "stroke", "pulmonary embolism",
        "deep vein thrombosis", "DVT", "PE", "Stevens-Johnson",
        "toxic epidermal necrolysis", "TEN", "rhabdomyolysis",
        "tumor lysis syndrome", "DIC", "disseminated intravascular",
        "hepatic failure", "liver failure", "renal failure",
    ]

    def __init__(self):
        self.ner = MedicalNER()
        self._sae_pattern = re.compile(
            "|".join(re.escape(k) for k in self.SAE_KEYWORDS), re.IGNORECASE
        )
        self._high_risk_pattern = re.compile(
            "|".join(re.escape(k) for k in self.HIGH_RISK_TERMS), re.IGNORECASE
        )

    def detect_from_narrative(self, narrative: str) -> List[DetectedAE]:
        """Extract adverse events from a single clinical narrative."""
        ner_result = self.ner.extract(narrative)

        conditions = [e.text for e in ner_result.conditions]
        drugs = [e.text for e in ner_result.drugs]
        severities = [e.text for e in ner_result.severities]
        causalities = [e.text for e in ner_result.causalities]
        lab_tests = [e.text for e in ner_result.lab_values]

        # Detect seriousness
        is_serious = bool(self._sae_pattern.search(narrative))
        is_high_risk = bool(self._high_risk_pattern.search(narrative))

        detected = []
        for condition in conditions:
            severity = severities[0] if severities else None
            causality = causalities[0] if causalities else None

            # Confidence scoring
            confidence = 0.5
            if severity:
                confidence += 0.15
            if causality:
                confidence += 0.15
            if drugs:
                confidence += 0.10
            if is_serious:
                confidence += 0.10

            detected.append(DetectedAE(
                ae_term=condition,
                severity=severity,
                causality=causality,
                drugs=drugs,
                lab_abnormalities=lab_tests,
                is_serious=is_serious or is_high_risk,
                confidence=min(confidence, 1.0),
                source_text=narrative[:500],
            ))

        return detected

    def detect_batch(self, narratives: List[str]) -> pd.DataFrame:
        """Process multiple narratives and return structured AE dataframe."""
        all_aes = []
        for i, narr in enumerate(narratives):
            detected = self.detect_from_narrative(narr)
            for ae in detected:
                all_aes.append({
                    "narrative_idx": i,
                    "ae_term": ae.ae_term,
                    "severity": ae.severity,
                    "causality": ae.causality,
                    "drugs": "; ".join(ae.drugs),
                    "lab_abnormalities": "; ".join(ae.lab_abnormalities),
                    "is_serious": ae.is_serious,
                    "confidence": ae.confidence,
                })
        return pd.DataFrame(all_aes) if all_aes else pd.DataFrame()

    def safety_signal_detection(
        self,
        ae_df: pd.DataFrame,
        drug_col: str = "arm",
        ae_col: str = "preferred_term",
        treatment_value: str = "Treatment",
    ) -> List[SafetySignal]:
        """
        Disproportionality analysis to detect safety signals.

        Uses:
          - PRR (Proportional Reporting Ratio)
          - ROR (Reporting Odds Ratio)
          - Chi-squared test

        A signal is flagged when PRR >= 2.0, chi2 >= 4.0, N >= 3.
        """
        signals = []
        total = len(ae_df)
        all_terms = ae_df[ae_col].unique()

        for term in all_terms:
            # 2x2 contingency table
            #                | This AE | Other AEs |
            # Treatment      |    a    |     b     |
            # Control         |    c    |     d     |
            mask_term = ae_df[ae_col] == term
            mask_treat = ae_df[drug_col] == treatment_value

            a = int((mask_term & mask_treat).sum())
            b = int((~mask_term & mask_treat).sum())
            c = int((mask_term & ~mask_treat).sum())
            d = int((~mask_term & ~mask_treat).sum())

            if a < 1 or c < 1 or (a + b) < 1 or (c + d) < 1:
                continue

            # PRR
            prr = (a / (a + b)) / (c / (c + d)) if (c / (c + d)) > 0 else 0
            log_prr = np.log(prr) if prr > 0 else 0
            se_log_prr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d)) if all(
                x > 0 for x in [a, a+b, c, c+d]) else 0
            prr_ci_lower = np.exp(log_prr - 1.96 * se_log_prr) if se_log_prr > 0 else 0
            prr_ci_upper = np.exp(log_prr + 1.96 * se_log_prr) if se_log_prr > 0 else 0

            # ROR
            ror = (a * d) / (b * c) if (b * c) > 0 else 0
            log_ror = np.log(ror) if ror > 0 else 0
            se_log_ror = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(
                x > 0 for x in [a, b, c, d]) else 0
            ror_ci_lower = np.exp(log_ror - 1.96 * se_log_ror) if se_log_ror > 0 else 0
            ror_ci_upper = np.exp(log_ror + 1.96 * se_log_ror) if se_log_ror > 0 else 0

            # Chi-squared
            table = np.array([[a, b], [c, d]])
            chi2, p_val = 0.0, 1.0
            if table.min() >= 0 and table.sum() > 0:
                try:
                    chi2, p_val, _, _ = stats.chi2_contingency(table, correction=True)
                except ValueError:
                    pass

            is_signal = prr >= 2.0 and chi2 >= 4.0 and a >= 3

            signals.append(SafetySignal(
                ae_term=term,
                drug=treatment_value,
                prr=round(prr, 3),
                prr_ci_lower=round(prr_ci_lower, 3),
                prr_ci_upper=round(prr_ci_upper, 3),
                ror=round(ror, 3),
                ror_ci_lower=round(ror_ci_lower, 3),
                ror_ci_upper=round(ror_ci_upper, 3),
                chi2=round(chi2, 3),
                p_value=round(p_val, 6),
                is_signal=is_signal,
                n_cases=a,
                total_ae=total,
            ))

        # Sort by PRR descending
        signals.sort(key=lambda s: s.prr, reverse=True)
        return signals

    def ae_summary(self, ae_df: pd.DataFrame) -> Dict:
        """Generate a comprehensive AE summary across all trials."""
        total = len(ae_df)
        sae_count = int((ae_df["serious"] == "Yes").sum()) if "serious" in ae_df else 0

        by_soc = ae_df["system_organ_class"].value_counts().to_dict() if "system_organ_class" in ae_df else {}
        by_severity = ae_df["severity"].value_counts().to_dict() if "severity" in ae_df else {}
        by_causality = ae_df["causality"].value_counts().to_dict() if "causality" in ae_df else {}
        top_terms = ae_df["ae_term"].value_counts().head(15).to_dict() if "ae_term" in ae_df else {}

        return {
            "total_ae": total,
            "total_sae": sae_count,
            "sae_rate": round(sae_count / total, 4) if total else 0,
            "by_system_organ_class": by_soc,
            "by_severity": by_severity,
            "by_causality": by_causality,
            "top_ae_terms": top_terms,
            "unique_patients": ae_df["subject_id"].nunique() if "subject_id" in ae_df else 0,
        }
