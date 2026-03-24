# 🧬 Clinical Trial Intelligence Platform

> **End-to-end AI/ML platform for clinical trial analytics** — adverse event detection, patient-trial matching, protocol compliance monitoring, and safety signal analysis.

Built for clinical research organizations (CROs) managing multi-therapeutic-area trials across Oncology, Cardiology, CNS/Neurology, Metabolic/Endocrinology, and Infectious Disease.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
│  Trial Overview │ AE Monitor │ Patient Match │ Protocol Comply  │
└────────────────────────┬────────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  /ae/detect  /ae/classify  /ae/signals  /patient/match          │
│  /protocol/deviations  /protocol/compliance  /trials            │
└──┬──────────┬──────────────┬────────────────┬───────────────────┘
   │          │              │                │
┌──▼───┐ ┌───▼────┐  ┌──────▼──────┐  ┌──────▼──────┐
│ NLP  │ │  AE    │  │  Patient    │  │  Protocol   │
│Engine│ │Detector│  │  Matcher    │  │  Analyzer   │
└──────┘ └────────┘  └─────────────┘  └─────────────┘
   │          │              │                │
   └──────────┴──────────────┴────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Synthetic Clinical │
              │    Trial Data       │
              │  (CDISC/SDTM-like)  │
              └─────────────────────┘
```

## Key Features

### 1. Adverse Event Detection & Classification
- **NER-based AE extraction** from free-text clinical narratives
- **CTCAE severity grading** (Grade 1–5) from context
- **Causality assessment** (Not Related → Definite)
- **MedDRA-style coding**: raw term → Preferred Term → System Organ Class
- **Seriousness determination** (SAE criteria: hospitalization, life-threatening, death, disability)
- **Safety signal detection** via disproportionality analysis (PRR, ROR, χ²)

### 2. Patient-Trial Eligibility Matching
- **Rule-based eligibility screening** against inclusion/exclusion criteria
- Supports numeric comparisons, diagnosis matching, comorbidity exclusion
- **Eligibility scoring** (0–100%) with detailed failure reasons
- **Batch cohort screening** across multiple trials
- **Screen failure analysis** — identifies most common exclusion reasons

### 3. Protocol Compliance Monitoring
- **Missed visit detection** against protocol-defined schedules
- **Visit window violation** tracking (±N day windows)
- **Lab compliance** — checks for missing required assessments
- **Patient-level compliance scoring** (composite of visit, window, lab adherence)
- **Study-level metrics** — enrollment, completion rate, deviation counts

### 4. Medical NLP Engine
- **Named Entity Recognition**: Drugs (60+), Conditions (25+ patterns), Lab Tests/Values, Severity grades, Causality, Temporal expressions
- **Clinical text embeddings** via sentence-transformers with TF-IDF fallback
- **Batch processing** for large-scale narrative analysis
- **Span deduplication** for overlapping entity resolution

### 5. Synthetic Clinical Trial Data
- **5 therapeutic areas** matching CRO focus: Oncology (NSCLC), Cardiology (HFpEF), CNS (Alzheimer's), Metabolic (T2DM), Anti-viral (COVID-19)
- **CDISC CDASH/SDTM-aligned** domain naming (DM, AE, LB, SV)
- **2,000 patients**, 8,000 AEs with narratives, 25,000 lab results, 15,000 visits
- Free-text AE narratives with clinical detail (ANC nadirs, CT findings, ECG data)
- Realistic severity/causality distributions per therapeutic area

## Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP / ML | spaCy, scikit-learn, sentence-transformers, transformers |
| Backend | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy, Faker |
| Testing | pytest |
| Infra | Makefile, Docker-ready |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Naresh1401/clinical-trial-intelligence.git
cd clinical-trial-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python data/generate_synthetic_data.py

# 4. Run tests
pytest tests/ -v

# 5. Start API server
uvicorn src.api.main:app --reload --port 8000

# 6. Launch dashboard (in another terminal)
streamlit run dashboard/app.py
```

Or use Make:
```bash
make all        # install + generate data + run tests
make api        # start FastAPI server
make dashboard  # start Streamlit dashboard
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/trials` | List available trial protocols |
| `POST` | `/ae/detect` | Extract AEs from narrative text |
| `POST` | `/ae/classify` | Classify AE (severity, causality, SOC) |
| `POST` | `/ae/signals` | Disproportionality safety signal analysis |
| `POST` | `/patient/match` | Match patient → trial eligibility |
| `POST` | `/patient/match-cohort` | Batch cohort matching |
| `POST` | `/protocol/deviations` | Detect protocol deviations |
| `POST` | `/protocol/compliance` | Patient compliance score |

### Example: AE Detection

```bash
curl -X POST http://localhost:8000/ae/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient developed Grade 3 neutropenia after pembrolizumab. ANC 800. Hospitalized."}'
```

```json
{
  "ae_count": 1,
  "adverse_events": [
    {
      "ae_term": "neutropenia",
      "severity": "Grade 3",
      "causality": "Unknown",
      "confidence": 0.85,
      "drugs_mentioned": ["pembrolizumab"]
    }
  ]
}
```

## Project Structure

```
clinical-trial-intelligence/
├── data/
│   └── generate_synthetic_data.py   # Synthetic data generator (CDISC-aligned)
├── src/
│   ├── nlp/
│   │   ├── medical_ner.py           # Medical Named Entity Recognition
│   │   └── embeddings.py            # Clinical text embeddings
│   ├── adverse_events/
│   │   ├── detector.py              # AE detection + safety signals (PRR/ROR)
│   │   └── classifier.py            # AE classification (severity/causality/SOC)
│   ├── patient/
│   │   └── matcher.py               # Patient-trial eligibility matching
│   ├── protocol/
│   │   └── analyzer.py              # Protocol compliance & deviation detection
│   └── api/
│       └── main.py                  # FastAPI REST endpoints
├── dashboard/
│   └── app.py                       # Streamlit multi-tab dashboard
├── tests/
│   └── test_pipeline.py             # Comprehensive test suite
├── requirements.txt
├── Makefile
└── README.md
```

## Clinical Domain Context

This platform addresses core workflows in clinical trial operations at CROs:

- **Safety Monitoring**: Automated AE detection from clinical narratives replaces manual medical coding, accelerating pharmacovigilance
- **Signal Detection**: PRR/ROR disproportionality analysis mirrors what's used in FDA's FAERS and EMA's EudraVigilance
- **Patient Recruitment**: Eligibility matching against I/E criteria reduces screen failure rates (industry avg: 15–25%)
- **Protocol Compliance**: Automated deviation detection catches issues that typically surface during SDV (source data verification) or during CRA monitoring visits
- **CDISC Standards**: Data structures follow CDASH (collection) and SDTM (submission) conventions used in FDA/EMA regulatory submissions

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_pipeline.py::TestAEDetector -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
