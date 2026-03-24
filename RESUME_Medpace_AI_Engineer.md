# NARESH SAMPANGI
**AI/ML Engineer — Healthcare & Clinical Trials**

ns.apj1729@gmail.com | [github.com/Naresh1401](https://github.com/Naresh1401) | LinkedIn

---

## SUMMARY

AI/ML Engineer with hands-on experience building end-to-end NLP, LLM, and machine learning systems for healthcare and clinical research. Proven ability to deploy production pipelines covering adverse event detection, patient-trial matching, medical NER, RAG architectures, and safety signal analytics. Deep familiarity with clinical trial workflows, CDISC/SDTM data standards, and pharmacovigilance concepts (PRR/ROR disproportionality analysis). Experienced with Python, FastAPI, LangChain, transformers, and cloud-native ML infrastructure.

---

## TECHNICAL SKILLS

| Category | Technologies |
|----------|-------------|
| **AI / ML** | scikit-learn, XGBoost, PyTorch, TensorFlow, sentence-transformers, QLoRA fine-tuning |
| **NLP / LLM** | spaCy, Hugging Face Transformers, OpenAI GPT-4, LangChain, LlamaIndex, RAG pipelines |
| **Healthcare** | CDISC (CDASH/SDTM), MedDRA coding, CTCAE grading, ICD-10, pharmacovigilance, HIPAA |
| **Backend** | Python, FastAPI, Flask, REST APIs, Pydantic, microservices |
| **Data** | pandas, NumPy, SQL, PostgreSQL, MongoDB, Pinecone, ChromaDB, FAISS |
| **MLOps / Cloud** | Docker, AWS (SageMaker, Lambda, S3), GCP, CI/CD, MLflow, Weights & Biases |
| **Visualization** | Streamlit, Plotly, Matplotlib |
| **Tools** | Git, pytest, Makefile, Jupyter, VS Code |

---

## RELEVANT PROJECTS

### Clinical Trial Intelligence Platform
*End-to-end AI platform for clinical trial analytics* | [GitHub](https://github.com/Naresh1401/clinical-trial-intelligence)

- Built **NLP-powered adverse event detection** from clinical narratives using custom medical NER (60+ drug patterns, 25+ condition patterns, CTCAE severity grading, causality assessment)
- Implemented **disproportionality safety signal detection** (PRR, ROR, χ² analysis) mirroring FDA FAERS methodology for automated pharmacovigilance
- Developed **patient-trial eligibility matcher** with rule-based I/E criteria evaluation, eligibility scoring, and batch cohort screening across 5 therapeutic areas
- Created **protocol compliance engine** detecting missed visits, visit window violations, and lab compliance deviations with patient-level scoring
- Built **MedDRA-style AE classification** — maps raw terms to Preferred Terms → System Organ Classes with seriousness determination (SAE criteria)
- Generated CDISC CDASH/SDTM-aligned synthetic data: 2,000 patients, 8,000 AEs, 25K labs across Oncology, Cardiology, CNS, Metabolic, and Infectious Disease trials
- Delivered as **FastAPI REST backend** (9 endpoints) + **Streamlit dashboard** (5 interactive tabs) with 30 passing integration tests
- **Stack**: Python, FastAPI, Streamlit, spaCy, scikit-learn, sentence-transformers, Plotly, Pydantic

### Pharma Patient Analytics Dashboard
*HIPAA-safe medication adherence analytics for pharmaceutical data* | [GitHub](https://github.com/Naresh1401/pharma-patient-analytics)

- Built medication adherence scoring (MPR/PDC calculations) with patient-level risk stratification
- Implemented Kaplan-Meier survival curves and treatment outcome prediction using logistic regression
- Created HIPAA-aware data generation with realistic pharmaceutical patterns (NDC codes, ICD-10 diagnoses)
- Delivered interactive Streamlit dashboard with demographic analysis, drug utilization metrics, and cohort comparison
- **Stack**: Python, Streamlit, scikit-learn, Plotly, pandas

### Enterprise RAG Pipeline
*Production-grade Retrieval Augmented Generation system* | [GitHub](https://github.com/Naresh1401/Enterprise-Rag-Pipeline)

- Architected multi-stage RAG pipeline with hybrid retrieval (dense + sparse), document chunking, and re-ranking
- Integrated vector stores (Pinecone/ChromaDB) with LangChain for scalable enterprise knowledge retrieval
- Implemented context-aware response generation with citation tracking and hallucination guardrails
- **Stack**: Python, LangChain, OpenAI, Pinecone, FastAPI

### LLM Fine-Tuning & Real-Time Inference
*QLoRA fine-tuning and streaming inference pipeline* | [GitHub](https://github.com/Naresh1401/LLM)

- Fine-tuned LLMs using QLoRA (4-bit quantization) for domain-specific tasks with parameter-efficient training
- Built real-time streaming inference with Bytewax for low-latency LLM serving
- Implemented RAG features, custom prompt engineering, and evaluation pipelines
- **Stack**: PyTorch, Hugging Face, QLoRA, Bytewax, PEFT

### Intelligent Document Processing
*AI-powered medical document extraction and classification* | [GitHub](https://github.com/Naresh1401/Intelligent-document-processing)

- Automated extraction of structured data from medical forms, insurance documents, and clinical records
- Combined OCR with NLP entity extraction for multi-format document processing
- **Stack**: Python, transformers, OCR, NLP

### LLM Safety Guardrails
*Safety and compliance framework for LLM deployments* | [GitHub](https://github.com/Naresh1401/LLM-safety-guardrails)

- Built input/output content filtering, prompt injection detection, and toxic content classification
- Implemented PII detection and redaction for HIPAA-compliant LLM interactions
- **Stack**: Python, transformers, regex-based filtering

### Agentic RAG Chatbot
*Multi-agent conversational AI with tool use* | [GitHub](https://github.com/Naresh1401/Agenticragchatbot)

- Developed multi-agent architecture with specialized agents for retrieval, reasoning, and tool execution
- Integrated external tool calling and dynamic agent orchestration
- **Stack**: Python, LangChain, OpenAI, agent frameworks

---

## EXPERIENCE

**AI/ML Engineer** — *[Company]* | *[Date range]*
- Developed and deployed NLP/ML models for document processing and text classification in production
- Built RESTful APIs serving ML model predictions with sub-200ms latency
- Designed data pipelines processing structured and unstructured clinical/healthcare data
- Collaborated with cross-functional teams to translate business requirements into ML solutions

---

## EDUCATION

**[Degree]** — *[University]* | *[Year]*

---

## WHY MEDPACE

- Direct experience building clinical trial AI systems (adverse event detection, safety signals, protocol compliance, patient matching) that align with Medpace's CRO operations across 5+ therapeutic areas
- Strong NLP/LLM foundation — can immediately contribute to automating clinical data processing, medical coding, and protocol analysis workflows
- Hands-on pharmacovigilance signal detection using industry-standard methods (PRR, ROR) applicable to Medpace's safety monitoring
- Demonstrated ability to ship full-stack AI applications (API + dashboard + tests) from concept to deployment
- Familiar with CDISC standards, MedDRA terminology, and CTCAE grading used in regulatory submissions
