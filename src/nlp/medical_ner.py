"""
src/nlp/medical_ner.py
=======================
Medical Named Entity Recognition for clinical trial text.

Extracts:
  - Drug names (interventions)
  - Medical conditions / diagnoses
  - Lab tests and values
  - Anatomical terms
  - Temporal expressions (Study Day, dates)
  - Severity / grade terms
  - Causality terms

Uses a hybrid approach:
  1. Rule-based extraction with clinical regex patterns
  2. Optional spaCy/transformer-based NER for production use
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class MedicalEntity:
    """A recognized medical entity from clinical text."""
    text: str
    label: str          # DRUG, CONDITION, LAB_TEST, LAB_VALUE, ANATOMY, TEMPORAL, SEVERITY, CAUSALITY
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None  # Normalized/canonical form


@dataclass
class NERResult:
    """Complete NER output for a document."""
    entities: List[MedicalEntity] = field(default_factory=list)
    raw_text: str = ""

    @property
    def drugs(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "DRUG"]

    @property
    def conditions(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "CONDITION"]

    @property
    def lab_tests(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "LAB_TEST"]

    @property
    def lab_values(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "LAB_VALUE"]

    @property
    def severities(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "SEVERITY"]

    @property
    def causalities(self) -> List[MedicalEntity]:
        return [e for e in self.entities if e.label == "CAUSALITY"]

    def to_dict(self) -> Dict:
        return {
            "entity_count": len(self.entities),
            "entities": [
                {"text": e.text, "label": e.label, "start": e.start,
                 "end": e.end, "confidence": e.confidence, "normalized": e.normalized}
                for e in self.entities
            ],
            "summary": {
                "drugs": [e.text for e in self.drugs],
                "conditions": [e.text for e in self.conditions],
                "lab_tests": [e.text for e in self.lab_tests],
                "severities": [e.text for e in self.severities],
                "causalities": [e.text for e in self.causalities],
            }
        }


# ── Drug lexicon (common clinical trial drugs + classes) ─────────────────
DRUG_NAMES = [
    # Oncology
    "pembrolizumab", "nivolumab", "atezolizumab", "carboplatin", "pemetrexed",
    "cisplatin", "paclitaxel", "docetaxel", "doxorubicin", "cyclophosphamide",
    "bevacizumab", "trastuzumab", "imatinib", "erlotinib", "osimertinib",
    "prednisone", "prednisolone", "dexamethasone", "G-CSF", "filgrastim",
    # Cardiology
    "dapagliflozin", "empagliflozin", "canagliflozin", "metoprolol", "carvedilol",
    "lisinopril", "enalapril", "valsartan", "sacubitril", "spironolactone",
    "digoxin", "amiodarone", "warfarin", "apixaban", "rivaroxaban",
    # CNS
    "lecanemab", "aducanumab", "donepezil", "memantine", "galantamine",
    "risperidone", "quetiapine", "sertraline", "escitalopram",
    # Metabolic
    "tirzepatide", "semaglutide", "liraglutide", "metformin", "insulin",
    "sitagliptin", "empagliflozin", "pioglitazone",
    # Anti-viral
    "remdesivir", "paxlovid", "nirmatrelvir", "molnupiravir", "baricitinib",
]

DRUG_CLASSES = [
    "checkpoint inhibitor", "PD-1 inhibitor", "PD-L1 inhibitor", "SGLT2 inhibitor",
    "GLP-1 receptor agonist", "anti-amyloid antibody", "monoclonal antibody",
    "tyrosine kinase inhibitor", "ACE inhibitor", "ARB", "beta-blocker",
    "corticosteroid", "chemotherapy", "immunotherapy", "antiviral",
]

# ── Condition patterns ───────────────────────────────────────────────────
CONDITION_PATTERNS = [
    r"(?:non-small cell lung cancer|NSCLC)",
    r"(?:heart failure|HFpEF|HFrEF|CHF)",
    r"(?:Alzheimer(?:'s)? disease|MCI|mild cognitive impairment)",
    r"(?:type [12] diabetes|T2DM|T1DM|diabetes mellitus)",
    r"(?:COVID-?19|SARS-CoV-2)",
    r"(?:neutropenia|anemia|anaemia|thrombocytopenia|pancytopenia)",
    r"(?:pneumonitis|interstitial lung disease)",
    r"(?:hepatotoxicity|drug-induced liver injury|DILI)",
    r"(?:nausea|vomiting|diarrhea|diarrhoea|constipation)",
    r"(?:fatigue|asthenia)",
    r"(?:rash|dermatitis|pruritus|urticaria)",
    r"(?:headache|dizziness|vertigo|syncope)",
    r"(?:hypotension|hypertension|tachycardia|bradycardia)",
    r"(?:peripheral neuropathy|neuropathy)",
    r"(?:hypoglycemia|hypoglycaemia|hyperglycemia|hyperglycaemia)",
    r"(?:acute kidney injury|AKI|renal failure|nephrotoxicity)",
    r"(?:pancreatitis|cholecystitis)",
    r"(?:QT prolongation|QTcF prolongation|arrhythmia)",
    r"(?:infusion.related reaction|infusion reaction|IRR)",
    r"(?:ARIA-?E|ARIA-?H|amyloid-related imaging abnormalit(?:y|ies))",
    r"(?:diabetic ketoacidosis|DKA)",
    r"(?:volume depletion|dehydration)",
    r"(?:injection site reaction)",
    r"(?:urinary tract infection|UTI)",
    r"(?:autoimmune disease|brain metastases)",
]

# ── Lab test patterns ────────────────────────────────────────────────────
LAB_TESTS = [
    "ANC", "WBC", "hemoglobin", "platelets", "ALT", "AST", "bilirubin",
    "creatinine", "eGFR", "BUN", "albumin", "HbA1c", "NT-proBNP", "BNP",
    "CRP", "lipase", "amylase", "potassium", "sodium", "LDH", "ALP",
    "creatinine clearance", "SpO2", "MMSE", "CDR-SB", "LVEF",
    "blood glucose", "fasting glucose", "QTcF", "INR",
]

LAB_VALUE_PATTERN = re.compile(
    r"(\b(?:" + "|".join(re.escape(t) for t in LAB_TESTS) + r")\b)"
    r"\s*(?:was|of|=|:)?\s*"
    r"([\d]+\.?[\d]*)\s*"
    r"((?:x10\^9/L|g/dL|U/L|mg/dL|mL/min|%|pg/mL|mg/L|mEq/L|ms|mmol/L))?",
    re.IGNORECASE,
)

# ── Severity patterns ────────────────────────────────────────────────────
SEVERITY_PATTERNS = [
    (r"Grade\s+[1-5](?:\s*\([^)]+\))?", "SEVERITY"),
    (r"(?:mild|moderate|severe|life[- ]threatening|fatal)", "SEVERITY"),
    (r"(?:CTCAE|NCI-CTCAE)\s+Grade\s+[1-5]", "SEVERITY"),
]

# ── Causality patterns ───────────────────────────────────────────────────
CAUSALITY_PATTERNS = [
    (r"(?:not related|unrelated)", "NOT_RELATED"),
    (r"(?:unlikely|unlikely related)", "UNLIKELY"),
    (r"(?:possible|possibly related)", "POSSIBLE"),
    (r"(?:probable|probably related)", "PROBABLE"),
    (r"(?:definite|definitely related|certain)", "DEFINITE"),
]

# ── Temporal patterns ────────────────────────────────────────────────────
TEMPORAL_PATTERN = re.compile(
    r"(?:Study Day\s+\d+|Day\s+\d+|Week\s+\d+|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s*\d{4})",
    re.IGNORECASE,
)


class MedicalNER:
    """
    Clinical text NER using hybrid rule-based + optional ML approach.

    Usage:
        ner = MedicalNER()
        result = ner.extract("Patient developed Grade 3 neutropenia on Day 15...")
        print(result.drugs, result.conditions, result.severities)
    """

    def __init__(self, use_transformer: bool = False):
        self._drug_pattern = re.compile(
            r"\b(" + "|".join(re.escape(d) for d in DRUG_NAMES) + r")\b",
            re.IGNORECASE,
        )
        self._drug_class_pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in DRUG_CLASSES) + r")\b",
            re.IGNORECASE,
        )
        self._condition_pattern = re.compile(
            "|".join(CONDITION_PATTERNS), re.IGNORECASE
        )
        self._lab_test_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in LAB_TESTS) + r")\b",
            re.IGNORECASE,
        )
        self._use_transformer = use_transformer
        self._transformer_model = None

    def extract(self, text: str) -> NERResult:
        """Extract all medical entities from clinical text."""
        entities: List[MedicalEntity] = []

        # 1. Drugs
        for m in self._drug_pattern.finditer(text):
            entities.append(MedicalEntity(
                text=m.group(), label="DRUG",
                start=m.start(), end=m.end(),
                normalized=m.group().lower()
            ))
        for m in self._drug_class_pattern.finditer(text):
            entities.append(MedicalEntity(
                text=m.group(), label="DRUG_CLASS",
                start=m.start(), end=m.end(),
                normalized=m.group().lower()
            ))

        # 2. Conditions
        for m in self._condition_pattern.finditer(text):
            entities.append(MedicalEntity(
                text=m.group(), label="CONDITION",
                start=m.start(), end=m.end(),
                normalized=m.group().lower()
            ))

        # 3. Lab tests + values
        for m in LAB_VALUE_PATTERN.finditer(text):
            entities.append(MedicalEntity(
                text=m.group(1), label="LAB_TEST",
                start=m.start(1), end=m.end(1),
                normalized=m.group(1).upper()
            ))
            if m.group(2):
                entities.append(MedicalEntity(
                    text=f"{m.group(2)} {m.group(3) or ''}".strip(),
                    label="LAB_VALUE",
                    start=m.start(2), end=m.end(3) if m.group(3) else m.end(2),
                    normalized=m.group(2)
                ))

        # Standalone lab test mentions
        for m in self._lab_test_pattern.finditer(text):
            if not any(e.start == m.start() and e.label == "LAB_TEST" for e in entities):
                entities.append(MedicalEntity(
                    text=m.group(), label="LAB_TEST",
                    start=m.start(), end=m.end(),
                    normalized=m.group().upper()
                ))

        # 4. Severity
        for pattern, _ in SEVERITY_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=m.group(), label="SEVERITY",
                    start=m.start(), end=m.end()
                ))

        # 5. Causality
        for pattern, norm in CAUSALITY_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=m.group(), label="CAUSALITY",
                    start=m.start(), end=m.end(),
                    normalized=norm
                ))

        # 6. Temporal
        for m in TEMPORAL_PATTERN.finditer(text):
            entities.append(MedicalEntity(
                text=m.group(), label="TEMPORAL",
                start=m.start(), end=m.end()
            ))

        # Deduplicate overlapping entities (keep longest span per position)
        entities = self._deduplicate(entities)

        return NERResult(entities=entities, raw_text=text)

    def extract_batch(self, texts: List[str]) -> List[NERResult]:
        """Extract entities from multiple texts."""
        return [self.extract(t) for t in texts]

    @staticmethod
    def _deduplicate(entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping the longest span."""
        if not entities:
            return entities
        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        result = [sorted_ents[0]]
        for ent in sorted_ents[1:]:
            prev = result[-1]
            if ent.start >= prev.end:
                result.append(ent)
            elif (ent.end - ent.start) > (prev.end - prev.start):
                result[-1] = ent
        return result
