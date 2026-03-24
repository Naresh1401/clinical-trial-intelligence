"""
generate_synthetic_data.py
===========================
Generate realistic synthetic clinical trial data for the CTI platform.

Produces:
  - patients.csv        — 2,000 enrolled patients across 5 trials
  - adverse_events.csv  — 8,000 AE reports with narratives
  - protocols.csv       — 5 trial protocols with eligibility criteria
  - lab_results.csv     — 25,000 lab measurements
  - visits.csv          — 15,000 scheduled/completed visits

All data is 100% synthetic. No real patient data. No PHI.
Follows CDISC CDASH / SDTM naming conventions where applicable.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── Configuration ────────────────────────────────────────────────────────
NUM_PATIENTS = 2_000
NUM_AE = 8_000
NUM_LABS = 25_000
NUM_VISITS = 15_000

TRIALS = [
    {
        "trial_id": "NCT-2025-ONC-0451",
        "title": "Phase III Randomized Study of Pembrolizumab Plus Chemotherapy vs Chemotherapy Alone in Advanced NSCLC",
        "phase": "Phase III",
        "therapeutic_area": "Oncology",
        "indication": "Non-Small Cell Lung Cancer (NSCLC)",
        "sponsor": "Medpace CRO (Sponsor: Apex Biopharma)",
        "status": "Enrolling",
        "target_enrollment": 450,
        "drug": "Pembrolizumab + Carboplatin/Pemetrexed",
        "comparator": "Carboplatin/Pemetrexed + Placebo",
        "primary_endpoint": "Progression-Free Survival (PFS)",
        "eligibility": {
            "min_age": 18, "max_age": 80,
            "gender": "All",
            "ecog_range": [0, 1],
            "conditions_required": ["NSCLC Stage IIIB/IV", "PD-L1 TPS >= 1%"],
            "conditions_excluded": ["autoimmune disease", "active brain metastases",
                                     "prior anti-PD-1 therapy", "HIV positive"],
            "lab_requirements": {"ANC": (">=", 1.5), "platelets": (">=", 100),
                                  "creatinine_clearance": (">=", 45)},
        },
    },
    {
        "trial_id": "NCT-2025-CAR-0782",
        "title": "Phase II Dose-Finding Study of Dapagliflozin in Heart Failure with Preserved Ejection Fraction",
        "phase": "Phase II",
        "therapeutic_area": "Cardiology",
        "indication": "Heart Failure with Preserved Ejection Fraction (HFpEF)",
        "sponsor": "Medpace CRO (Sponsor: Cardium Therapeutics)",
        "status": "Enrolling",
        "target_enrollment": 320,
        "drug": "Dapagliflozin 10mg",
        "comparator": "Placebo",
        "primary_endpoint": "Change in NT-proBNP at 12 weeks",
        "eligibility": {
            "min_age": 40, "max_age": 85,
            "gender": "All",
            "ecog_range": [0, 2],
            "conditions_required": ["Heart Failure NYHA II-III", "LVEF >= 45%"],
            "conditions_excluded": ["Type 1 Diabetes", "eGFR < 30",
                                     "active malignancy", "recent MI (< 3 months)"],
            "lab_requirements": {"NT-proBNP": (">=", 300), "eGFR": (">=", 30),
                                  "HbA1c": ("<=", 10)},
        },
    },
    {
        "trial_id": "NCT-2025-CNS-1134",
        "title": "Phase II Adaptive Platform Trial of Lecanemab in Early Alzheimer's Disease",
        "phase": "Phase II",
        "therapeutic_area": "CNS / Neurology",
        "indication": "Early Alzheimer's Disease (MCI due to AD)",
        "sponsor": "Medpace CRO (Sponsor: NeuraCure Inc.)",
        "status": "Enrolling",
        "target_enrollment": 280,
        "drug": "Lecanemab 10mg/kg IV Q2W",
        "comparator": "Placebo IV Q2W",
        "primary_endpoint": "Change in CDR-SB at 18 months",
        "eligibility": {
            "min_age": 55, "max_age": 85,
            "gender": "All",
            "ecog_range": [0, 1],
            "conditions_required": ["MCI or mild AD", "Amyloid PET positive or CSF A-beta42 low",
                                     "MMSE 22-30"],
            "conditions_excluded": ["vascular dementia", "Parkinson's disease",
                                     "major psychiatric disorder", "anticoagulant therapy"],
            "lab_requirements": {"MMSE": (">=", 22), "platelets": (">=", 100)},
        },
    },
    {
        "trial_id": "NCT-2025-MET-0298",
        "title": "Phase III Study of Tirzepatide vs Semaglutide in Type 2 Diabetes with Obesity",
        "phase": "Phase III",
        "therapeutic_area": "Metabolic / Endocrinology",
        "indication": "Type 2 Diabetes Mellitus with BMI >= 30",
        "sponsor": "Medpace CRO (Sponsor: MetaGen Pharma)",
        "status": "Active",
        "target_enrollment": 500,
        "drug": "Tirzepatide 15mg SC QW",
        "comparator": "Semaglutide 2.4mg SC QW",
        "primary_endpoint": "Change in HbA1c at 40 weeks",
        "eligibility": {
            "min_age": 18, "max_age": 75,
            "gender": "All",
            "ecog_range": [0, 1],
            "conditions_required": ["Type 2 Diabetes", "HbA1c 7.0-10.5%", "BMI >= 30"],
            "conditions_excluded": ["Type 1 Diabetes", "pancreatitis history",
                                     "medullary thyroid carcinoma", "MEN2 syndrome"],
            "lab_requirements": {"HbA1c": (">=", 7.0), "eGFR": (">=", 30)},
        },
    },
    {
        "trial_id": "NCT-2025-INF-0567",
        "title": "Phase II Study of Novel Broad-Spectrum Antiviral in Moderate COVID-19",
        "phase": "Phase II",
        "therapeutic_area": "Anti-Viral / Anti-Infective",
        "indication": "Moderate COVID-19 (hospitalized, not ventilated)",
        "sponsor": "Medpace CRO (Sponsor: ViralShield Bio)",
        "status": "Completed",
        "target_enrollment": 200,
        "drug": "VS-4891 800mg BID x 5 days",
        "comparator": "Standard of Care + Placebo",
        "primary_endpoint": "Time to sustained clinical recovery",
        "eligibility": {
            "min_age": 18, "max_age": 80,
            "gender": "All",
            "ecog_range": [0, 2],
            "conditions_required": ["SARS-CoV-2 positive", "Hospitalized",
                                     "SpO2 >= 94% on room air"],
            "conditions_excluded": ["mechanical ventilation", "ECMO",
                                     "severe hepatic impairment", "concurrent antiviral"],
            "lab_requirements": {"SpO2": (">=", 94), "ALT": ("<=", 5)},
        },
    },
]

# ── AE vocabularies (MedDRA-style) ──────────────────────────────────────
AE_TERMS = {
    "Oncology": [
        ("Nausea", "Gastrointestinal disorders", "nausea"),
        ("Fatigue", "General disorders", "fatigue"),
        ("Neutropenia", "Blood and lymphatic system disorders", "neutropenia"),
        ("Anemia", "Blood and lymphatic system disorders", "anaemia"),
        ("Rash", "Skin and subcutaneous tissue disorders", "rash"),
        ("Pneumonitis", "Respiratory disorders", "pneumonitis"),
        ("Diarrhea", "Gastrointestinal disorders", "diarrhoea"),
        ("Hepatotoxicity", "Hepatobiliary disorders", "hepatotoxicity"),
        ("Thrombocytopenia", "Blood and lymphatic system disorders", "thrombocytopenia"),
        ("Peripheral neuropathy", "Nervous system disorders", "peripheral neuropathy"),
    ],
    "Cardiology": [
        ("Hypotension", "Vascular disorders", "hypotension"),
        ("Dizziness", "Nervous system disorders", "dizziness"),
        ("Urinary tract infection", "Infections", "urinary tract infection"),
        ("Volume depletion", "Metabolism disorders", "volume depletion"),
        ("Diabetic ketoacidosis", "Metabolism disorders", "diabetic ketoacidosis"),
        ("Acute kidney injury", "Renal disorders", "acute kidney injury"),
        ("Bradycardia", "Cardiac disorders", "bradycardia"),
        ("Hyperkalemia", "Metabolism disorders", "hyperkalaemia"),
    ],
    "CNS / Neurology": [
        ("Headache", "Nervous system disorders", "headache"),
        ("ARIA-E (edema)", "Nervous system disorders", "ARIA edema"),
        ("ARIA-H (hemorrhage)", "Nervous system disorders", "ARIA hemorrhage"),
        ("Infusion-related reaction", "General disorders", "infusion related reaction"),
        ("Dizziness", "Nervous system disorders", "dizziness"),
        ("Confusion", "Psychiatric disorders", "confusional state"),
        ("Fall", "Injury", "fall"),
        ("Urinary tract infection", "Infections", "urinary tract infection"),
    ],
    "Metabolic / Endocrinology": [
        ("Nausea", "Gastrointestinal disorders", "nausea"),
        ("Diarrhea", "Gastrointestinal disorders", "diarrhoea"),
        ("Injection site reaction", "General disorders", "injection site reaction"),
        ("Hypoglycemia", "Metabolism disorders", "hypoglycaemia"),
        ("Pancreatitis", "Gastrointestinal disorders", "pancreatitis"),
        ("Decreased appetite", "Metabolism disorders", "decreased appetite"),
        ("Vomiting", "Gastrointestinal disorders", "vomiting"),
        ("Constipation", "Gastrointestinal disorders", "constipation"),
    ],
    "Anti-Viral / Anti-Infective": [
        ("Diarrhea", "Gastrointestinal disorders", "diarrhoea"),
        ("Elevated transaminases", "Hepatobiliary disorders", "transaminases increased"),
        ("Headache", "Nervous system disorders", "headache"),
        ("Nausea", "Gastrointestinal disorders", "nausea"),
        ("Rash", "Skin and subcutaneous tissue disorders", "rash"),
        ("QT prolongation", "Cardiac disorders", "electrocardiogram QT prolonged"),
        ("Insomnia", "Psychiatric disorders", "insomnia"),
        ("Fatigue", "General disorders", "fatigue"),
    ],
}

SEVERITY_GRADES = ["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)",
                   "Grade 4 (Life-threatening)", "Grade 5 (Death)"]
SEVERITY_WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]

CAUSALITY = ["Not Related", "Unlikely", "Possible", "Probable", "Definite"]
CAUSALITY_WEIGHTS = [0.15, 0.20, 0.30, 0.25, 0.10]

OUTCOMES = ["Recovered/Resolved", "Recovering/Resolving", "Not Recovered",
            "Recovered with Sequelae", "Fatal", "Unknown"]
OUTCOME_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.05, 0.05]

ACTIONS_TAKEN = ["None", "Dose Reduced", "Drug Interrupted", "Drug Withdrawn",
                 "Concomitant Medication", "Non-drug Therapy"]
ACTION_WEIGHTS = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]

SERIOUSNESS_CRITERIA = [
    "Results in death", "Life-threatening", "Requires hospitalization",
    "Prolongs hospitalization", "Results in disability", "Congenital anomaly",
    "Other medically important"
]

# ── Narrative templates ──────────────────────────────────────────────────
NARRATIVE_TEMPLATES = [
    "A {age}-year-old {sex} enrolled in {trial_id} ({indication}) on {drug} "
    "developed {ae_term} ({severity}) on Study Day {study_day}. "
    "The event was assessed as {causality} to study drug. "
    "{action_detail} The event {outcome_detail}.",

    "Subject {subject_id} ({age}y {sex}) experienced {ae_term} starting on "
    "{onset_date} (Study Day {study_day}). Severity: {severity}. "
    "The investigator assessed causality as {causality}. "
    "Action taken: {action}. Outcome: {outcome}. {extra_detail}",

    "On {onset_date}, the patient (Subject {subject_id}, {age}-year-old {sex}) "
    "reported {ae_term}. This was a {severity} event in the context of "
    "{indication} treatment with {drug}. "
    "Causality assessment: {causality}. {sae_detail} {outcome_detail}",

    "{ae_term} was reported for Subject {subject_id} ({sex}, {age} years) on "
    "Day {study_day} of treatment with {drug} for {indication}. "
    "The event was graded as {severity} and considered {causality} to the "
    "investigational product. {action_detail} Current status: {outcome}.",
]

EXTRA_DETAILS = {
    "Neutropenia": "ANC nadir was {:.1f} x10^9/L. G-CSF was administered.",
    "Pneumonitis": "CT scan showed bilateral ground-glass opacities. Prednisone 1mg/kg initiated.",
    "Hepatotoxicity": "ALT peaked at {:.0f} U/L ({}x ULN). Drug was held pending resolution.",
    "Hypoglycemia": "Blood glucose nadir was {:.0f} mg/dL. Treated with oral glucose.",
    "ARIA-E (edema)": "MRI showed vasogenic edema in the parietal lobe. Dosing suspended per protocol.",
    "Pancreatitis": "Lipase was {:.0f} U/L ({}x ULN). Patient hospitalized for observation.",
    "QT prolongation": "QTcF interval was {:.0f} ms. ECG monitoring intensified.",
    "Acute kidney injury": "Serum creatinine rose to {:.1f} mg/dL from baseline {:.1f} mg/dL.",
    "Diabetic ketoacidosis": "pH {:.2f}, bicarbonate {:.0f} mEq/L. ICU admission required.",
}


# ── Helper functions ─────────────────────────────────────────────────────
def _generate_subject_id(trial_idx: int) -> str:
    site = random.randint(100, 199)
    subj = random.randint(1001, 9999)
    return f"{site}-{subj}"


def _generate_narrative(patient: dict, ae: dict, trial: dict) -> str:
    template = random.choice(NARRATIVE_TEMPLATES)

    action_details = {
        "None": "No action was taken regarding study drug.",
        "Dose Reduced": "The dose of study drug was reduced.",
        "Drug Interrupted": "Study drug was temporarily interrupted.",
        "Drug Withdrawn": "Study drug was permanently discontinued.",
        "Concomitant Medication": "Concomitant medication was administered for symptom management.",
        "Non-drug Therapy": "Non-pharmacological intervention was provided.",
    }

    outcome_details = {
        "Recovered/Resolved": "resolved without sequelae.",
        "Recovering/Resolving": "is resolving at the time of this report.",
        "Not Recovered": "has not resolved as of the data cut-off date.",
        "Recovered with Sequelae": "resolved with residual sequelae.",
        "Fatal": "resulted in the patient's death.",
        "Unknown": "outcome is unknown.",
    }

    is_sae = ae["serious"] == "Yes"
    sae_detail = ""
    if is_sae:
        criteria = random.choice(SERIOUSNESS_CRITERIA[:4])
        sae_detail = f"This was classified as a Serious Adverse Event ({criteria})."

    extra = ""
    ae_term = ae["ae_term"]
    if ae_term in EXTRA_DETAILS:
        tmpl = EXTRA_DETAILS[ae_term]
        if ae_term == "Neutropenia":
            extra = tmpl.format(random.uniform(0.2, 1.4))
        elif ae_term in ("Hepatotoxicity", "Pancreatitis"):
            val = random.uniform(150, 800)
            uln_mult = val / 40
            extra = tmpl.format(val, f"{uln_mult:.1f}")
        elif ae_term == "Hypoglycemia":
            extra = tmpl.format(random.uniform(35, 65))
        elif ae_term == "QT prolongation":
            extra = tmpl.format(random.uniform(480, 550))
        elif ae_term == "Acute kidney injury":
            bl = random.uniform(0.8, 1.2)
            extra = tmpl.format(bl + random.uniform(0.5, 2.0), bl)
        elif ae_term == "Diabetic ketoacidosis":
            extra = tmpl.format(random.uniform(7.1, 7.25), random.uniform(10, 18))
        else:
            extra = tmpl

    narrative = template.format(
        age=patient["age"],
        sex=patient["sex"],
        trial_id=trial["trial_id"],
        indication=trial["indication"],
        drug=trial["drug"],
        ae_term=ae_term,
        severity=ae["severity"],
        study_day=ae["study_day"],
        causality=ae["causality"],
        action=ae["action_taken"],
        outcome=ae["outcome"],
        subject_id=patient["subject_id"],
        onset_date=ae["onset_date"],
        action_detail=action_details.get(ae["action_taken"], ""),
        outcome_detail=outcome_details.get(ae["outcome"], ""),
        sae_detail=sae_detail,
        extra_detail=extra,
    )
    return narrative


# ── Generators ───────────────────────────────────────────────────────────
def generate_patients() -> pd.DataFrame:
    """Generate synthetic patient demographics (DM domain in SDTM)."""
    records = []
    for trial in TRIALS:
        n = trial["target_enrollment"]
        elig = trial["eligibility"]
        for _ in range(n):
            age = random.randint(elig["min_age"], elig["max_age"])
            sex = random.choice(["Male", "Female"]) if elig["gender"] == "All" else elig["gender"]
            race = np.random.choice(
                ["White", "Black or African American", "Asian",
                 "American Indian", "Native Hawaiian", "Multiple", "Unknown"],
                p=[0.55, 0.18, 0.12, 0.03, 0.02, 0.05, 0.05]
            )
            ethnicity = np.random.choice(
                ["Not Hispanic or Latino", "Hispanic or Latino", "Unknown"],
                p=[0.75, 0.18, 0.07]
            )
            consent_date = fake.date_between(
                start_date=datetime(2024, 6, 1), end_date=datetime(2025, 12, 31)
            )
            arm = np.random.choice(["Treatment", "Control"], p=[0.6, 0.4])

            records.append({
                "subject_id": _generate_subject_id(TRIALS.index(trial)),
                "trial_id": trial["trial_id"],
                "therapeutic_area": trial["therapeutic_area"],
                "indication": trial["indication"],
                "age": age,
                "sex": sex,
                "race": race,
                "ethnicity": ethnicity,
                "country": np.random.choice(
                    ["USA", "Germany", "UK", "Japan", "Canada", "Australia", "France"],
                    p=[0.35, 0.15, 0.12, 0.10, 0.10, 0.10, 0.08]
                ),
                "site_id": f"SITE-{random.randint(100, 199):03d}",
                "arm": arm,
                "consent_date": consent_date,
                "randomization_date": consent_date + timedelta(days=random.randint(1, 14)),
                "status": np.random.choice(
                    ["Active", "Completed", "Discontinued", "Screen Failure"],
                    p=[0.50, 0.25, 0.15, 0.10]
                ),
                "ecog_ps": random.choice(elig["ecog_range"]),
                "bmi": round(random.uniform(18.5, 42.0), 1),
                "smoking_status": np.random.choice(
                    ["Never", "Former", "Current"], p=[0.40, 0.35, 0.25]
                ),
            })
    return pd.DataFrame(records)


def generate_adverse_events(patients: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic adverse event records (AE domain in SDTM)."""
    records = []
    active_patients = patients[patients["status"].isin(["Active", "Completed", "Discontinued"])]

    for _ in range(NUM_AE):
        pat = active_patients.sample(1).iloc[0]
        trial = next(t for t in TRIALS if t["trial_id"] == pat["trial_id"])
        ta = trial["therapeutic_area"]

        ae_info = random.choice(AE_TERMS[ta])
        ae_term, soc, pt = ae_info

        severity = np.random.choice(SEVERITY_GRADES, p=SEVERITY_WEIGHTS)
        grade_num = SEVERITY_GRADES.index(severity) + 1

        # Higher grades more likely to be serious
        is_serious = "Yes" if (grade_num >= 3 or random.random() < 0.15) else "No"

        # Causality correlates with treatment arm
        if pat["arm"] == "Treatment":
            cw = [0.10, 0.15, 0.30, 0.30, 0.15]
        else:
            cw = [0.30, 0.30, 0.25, 0.10, 0.05]
        causality = np.random.choice(CAUSALITY, p=cw)

        study_day = random.randint(1, 365)
        onset = pat["randomization_date"] + timedelta(days=study_day)
        duration = random.randint(1, 90) if grade_num < 5 else random.randint(1, 30)
        resolution = onset + timedelta(days=duration) if grade_num < 5 else None

        action = np.random.choice(ACTIONS_TAKEN, p=ACTION_WEIGHTS)
        outcome = np.random.choice(OUTCOMES, p=OUTCOME_WEIGHTS)
        if grade_num == 5:
            outcome = "Fatal"

        ae_record = {
            "ae_id": str(uuid.uuid4())[:8],
            "subject_id": pat["subject_id"],
            "trial_id": pat["trial_id"],
            "therapeutic_area": ta,
            "arm": pat["arm"],
            "ae_term": ae_term,
            "preferred_term": pt,
            "system_organ_class": soc,
            "severity": severity,
            "grade": grade_num,
            "serious": is_serious,
            "causality": causality,
            "onset_date": onset,
            "resolution_date": resolution,
            "study_day": study_day,
            "duration_days": duration if grade_num < 5 else None,
            "action_taken": action,
            "outcome": outcome,
        }

        narrative = _generate_narrative(
            {"age": pat["age"], "sex": pat["sex"], "subject_id": pat["subject_id"]},
            ae_record,
            trial,
        )
        ae_record["narrative"] = narrative
        records.append(ae_record)

    return pd.DataFrame(records)


def generate_lab_results(patients: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic lab results (LB domain in SDTM)."""
    LAB_TESTS = {
        "ANC":        {"unit": "x10^9/L", "normal": (1.5, 8.0),  "range": (0.1, 15.0)},
        "Hemoglobin":  {"unit": "g/dL",    "normal": (12.0, 17.0), "range": (5.0, 20.0)},
        "Platelets":   {"unit": "x10^9/L", "normal": (150, 400),  "range": (10, 600)},
        "WBC":         {"unit": "x10^9/L", "normal": (4.0, 11.0), "range": (0.5, 30.0)},
        "ALT":         {"unit": "U/L",     "normal": (7, 40),     "range": (3, 1000)},
        "AST":         {"unit": "U/L",     "normal": (8, 33),     "range": (3, 800)},
        "Creatinine":  {"unit": "mg/dL",   "normal": (0.6, 1.2),  "range": (0.3, 8.0)},
        "eGFR":        {"unit": "mL/min",  "normal": (60, 120),   "range": (5, 150)},
        "HbA1c":       {"unit": "%",       "normal": (4.0, 5.7),  "range": (3.5, 14.0)},
        "NT-proBNP":   {"unit": "pg/mL",   "normal": (0, 125),    "range": (5, 15000)},
        "CRP":         {"unit": "mg/L",    "normal": (0, 5),      "range": (0.1, 200)},
        "Albumin":     {"unit": "g/dL",    "normal": (3.5, 5.0),  "range": (1.5, 6.0)},
    }

    VISITS = ["Screening", "Baseline", "Week 2", "Week 4", "Week 8",
              "Week 12", "Week 24", "Week 36", "End of Treatment"]

    records = []
    active = patients[patients["status"] != "Screen Failure"]
    sampled = active.sample(min(len(active), NUM_LABS // (len(LAB_TESTS) * 3)))

    for _, pat in sampled.iterrows():
        num_visits = random.randint(3, len(VISITS))
        visit_list = VISITS[:num_visits]
        for visit in visit_list:
            for test_name, spec in LAB_TESTS.items():
                # 85% of values normal, 15% abnormal
                if random.random() < 0.85:
                    val = random.uniform(*spec["normal"])
                else:
                    val = random.uniform(*spec["range"])
                lo, hi = spec["normal"]
                flag = "Normal"
                if val < lo:
                    flag = "Low"
                elif val > hi:
                    flag = "High"

                records.append({
                    "subject_id": pat["subject_id"],
                    "trial_id": pat["trial_id"],
                    "visit": visit,
                    "test_name": test_name,
                    "result_value": round(val, 2),
                    "unit": spec["unit"],
                    "reference_low": lo,
                    "reference_high": hi,
                    "flag": flag,
                    "collection_date": pat["randomization_date"] + timedelta(
                        days=VISITS.index(visit) * random.randint(7, 21)
                    ),
                })
    return pd.DataFrame(records[:NUM_LABS])


def generate_visits(patients: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic visit schedule (SV domain in SDTM)."""
    VISIT_SCHEDULE = [
        ("Screening", -14), ("Baseline", 1), ("Week 2", 15), ("Week 4", 29),
        ("Week 8", 57), ("Week 12", 85), ("Week 24", 169), ("Week 36", 253),
        ("End of Treatment", 365), ("Follow-up 30d", 395),
    ]

    records = []
    active = patients[patients["status"] != "Screen Failure"]

    for _, pat in active.iterrows():
        n_visits = random.randint(3, len(VISIT_SCHEDULE))
        for visit_name, target_day in VISIT_SCHEDULE[:n_visits]:
            planned = pat["randomization_date"] + timedelta(days=target_day)
            actual_offset = random.randint(-3, 7)
            actual = planned + timedelta(days=actual_offset)

            # Determine if visit was completed
            completed = random.random() < 0.88
            deviation = abs(actual_offset) > 5
            deviation_reason = ""
            if deviation:
                deviation_reason = np.random.choice([
                    "Patient scheduling conflict",
                    "Site holiday / closure",
                    "Weather / travel issue",
                    "Patient illness (non-AE)",
                    "Administrative delay",
                ])

            records.append({
                "subject_id": pat["subject_id"],
                "trial_id": pat["trial_id"],
                "visit_name": visit_name,
                "visit_number": VISIT_SCHEDULE.index((visit_name, target_day)) + 1,
                "planned_date": planned,
                "actual_date": actual if completed else None,
                "target_day": target_day,
                "actual_study_day": target_day + actual_offset if completed else None,
                "status": "Completed" if completed else np.random.choice(
                    ["Missed", "Cancelled", "Rescheduled"], p=[0.5, 0.3, 0.2]
                ),
                "protocol_deviation": deviation,
                "deviation_reason": deviation_reason if deviation else "",
                "window_min": target_day - 3,
                "window_max": target_day + 5,
            })
    return pd.DataFrame(records[:NUM_VISITS])


def generate_protocols() -> pd.DataFrame:
    """Generate protocol metadata for all trials."""
    records = []
    for trial in TRIALS:
        elig = trial["eligibility"]
        records.append({
            "trial_id": trial["trial_id"],
            "title": trial["title"],
            "phase": trial["phase"],
            "therapeutic_area": trial["therapeutic_area"],
            "indication": trial["indication"],
            "sponsor": trial["sponsor"],
            "status": trial["status"],
            "target_enrollment": trial["target_enrollment"],
            "drug": trial["drug"],
            "comparator": trial["comparator"],
            "primary_endpoint": trial["primary_endpoint"],
            "min_age": elig["min_age"],
            "max_age": elig["max_age"],
            "gender": elig["gender"],
            "inclusion_criteria": "; ".join(elig["conditions_required"]),
            "exclusion_criteria": "; ".join(elig["conditions_excluded"]),
            "lab_requirements": str(elig["lab_requirements"]),
        })
    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Generating synthetic clinical trial data...\n")

    print("  Patients...")
    patients = generate_patients()
    patients.to_csv("data/patients.csv", index=False)
    print(f"    → {len(patients):,} patients across {patients['trial_id'].nunique()} trials")

    print("  Adverse Events...")
    aes = generate_adverse_events(patients)
    aes.to_csv("data/adverse_events.csv", index=False)
    print(f"    → {len(aes):,} AE reports")
    print(f"    → SAEs: {(aes['serious'] == 'Yes').sum():,} ({(aes['serious'] == 'Yes').mean():.1%})")

    print("  Lab Results...")
    labs = generate_lab_results(patients)
    labs.to_csv("data/lab_results.csv", index=False)
    print(f"    → {len(labs):,} lab measurements")

    print("  Visits...")
    visits = generate_visits(patients)
    visits.to_csv("data/visits.csv", index=False)
    print(f"    → {len(visits):,} visit records")

    print("  Protocols...")
    protocols = generate_protocols()
    protocols.to_csv("data/protocols.csv", index=False)
    print(f"    → {len(protocols)} trial protocols")

    print("\n✅ All data generated in data/ directory")
    print("\n── Summary ──")
    print(f"  Patients:  {len(patients):>8,}")
    print(f"  AEs:       {len(aes):>8,}")
    print(f"  Labs:      {len(labs):>8,}")
    print(f"  Visits:    {len(visits):>8,}")
    print(f"  Protocols: {len(protocols):>8}")
    print(f"  Total:     {len(patients)+len(aes)+len(labs)+len(visits)+len(protocols):>8,} rows")


if __name__ == "__main__":
    main()
