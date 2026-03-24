"""
dashboard/app.py
==================
Streamlit Dashboard — Clinical Trial Intelligence Platform

Tabs:
  1. Trial Overview      — enrollment, completion, study metrics
  2. Adverse Events      — AE detection, safety signal heatmap
  3. Patient Matching    — eligibility screening & failure analysis
  4. Protocol Compliance — deviation tracking, compliance scores
  5. NLP Explorer        — Interactive NER and text analysis
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.medical_ner import MedicalNER
from src.adverse_events.detector import AEDetector
from src.adverse_events.classifier import AEClassifier
from src.patient.matcher import PatientTrialMatcher
from src.protocol.analyzer import ProtocolAnalyzer

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Trial Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = ROOT / "data"


# ── Data loading (cached) ───────────────────────────────────────────

@st.cache_data
def load_data():
    """Load all synthetic CSV datasets."""
    data = {}
    for name in ["patients", "adverse_events", "lab_results", "visits", "protocols"]:
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            data[name] = pd.DataFrame()
    return data


@st.cache_resource
def init_services():
    """Initialize NLP and analysis services."""
    return {
        "ner": MedicalNER(),
        "ae_detector": AEDetector(),
        "ae_classifier": AEClassifier(),
        "matcher": PatientTrialMatcher(),
        "protocol": ProtocolAnalyzer(),
    }


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("🧬 Clinical Trial Intelligence")
st.sidebar.markdown("---")

data = load_data()
services = init_services()

# Trial selector
trials = data["protocols"]["trial_id"].unique().tolist() if not data["protocols"].empty else []
selected_trial = st.sidebar.selectbox("Select Trial", trials, index=0 if trials else 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built for**: Medpace AI Engineer Demo\n\n"
    "**Stack**: Python · FastAPI · Streamlit · spaCy · scikit-learn"
)

# ── Tab layout ───────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Trial Overview",
    "⚠️ Adverse Events",
    "🧑‍⚕️ Patient Matching",
    "📋 Protocol Compliance",
    "🔬 NLP Explorer",
])


# ═══ TAB 1: Trial Overview ══════════════════════════════════════════

with tab1:
    st.header("Trial Overview")

    if data["protocols"].empty:
        st.warning("No data found. Run `python data/generate_synthetic_data.py` first.")
        st.stop()

    trial_info = data["protocols"][data["protocols"]["trial_id"] == selected_trial]
    if not trial_info.empty:
        row = trial_info.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Phase", row.get("phase", "N/A"))
        c2.metric("Therapeutic Area", row.get("therapeutic_area", "N/A"))
        c3.metric("Status", row.get("status", "Enrolling"))
        c4.metric("Duration", f"{row.get('duration_weeks', 'N/A')} weeks")

    # Enrollment by trial
    if not data["patients"].empty:
        enroll = data["patients"].groupby("trial_id").size().reset_index(name="patients")
        fig = px.bar(enroll, x="trial_id", y="patients", color="trial_id",
                     title="Enrollment by Trial", text_auto=True)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # Demographics
    if not data["patients"].empty:
        trial_pts = data["patients"][data["patients"]["trial_id"] == selected_trial]
        col1, col2 = st.columns(2)
        with col1:
            if "sex" in trial_pts.columns:
                fig = px.pie(trial_pts, names="sex", title="Sex Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "age" in trial_pts.columns:
                fig = px.histogram(trial_pts, x="age", nbins=20, title="Age Distribution",
                                   color_discrete_sequence=["#636EFA"])
                st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 2: Adverse Events ══════════════════════════════════════════

with tab2:
    st.header("Adverse Event Monitoring")

    ae_df = data["adverse_events"]
    if ae_df.empty:
        st.info("No AE data available.")
    else:
        trial_ae = ae_df[ae_df["trial_id"] == selected_trial]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total AEs", len(trial_ae))
        sae_count = trial_ae[trial_ae["severity"].isin(["Severe", "Life-threatening"])].shape[0] if "severity" in trial_ae.columns else 0
        c2.metric("Serious AEs", sae_count)
        c3.metric("Unique Terms", trial_ae["ae_term"].nunique() if "ae_term" in trial_ae.columns else 0)
        related = trial_ae[trial_ae["causality"].isin(["Probable", "Definite"])].shape[0] if "causality" in trial_ae.columns else 0
        c4.metric("Drug-Related", related)

        # AE frequency bar chart
        if "ae_term" in trial_ae.columns:
            top_ae = trial_ae["ae_term"].value_counts().head(15).reset_index()
            top_ae.columns = ["AE Term", "Count"]
            fig = px.bar(top_ae, x="Count", y="AE Term", orientation="h",
                         title="Top 15 Adverse Events", color="Count",
                         color_continuous_scale="Reds")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        # Severity distribution
        if "severity" in trial_ae.columns:
            sev = trial_ae["severity"].value_counts().reset_index()
            sev.columns = ["Severity", "Count"]
            fig = px.pie(sev, names="Severity", values="Count",
                         title="Severity Distribution",
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)

        # Safety signal detection
        st.subheader("Safety Signal Detection (PRR / ROR)")
        if "arm" in trial_ae.columns and "ae_term" in trial_ae.columns:
            signals = services["ae_detector"].safety_signal_detection(trial_ae)
            if signals:
                sig_df = pd.DataFrame([
                    {"AE Term": s.ae_term, "PRR": round(s.prr, 2),
                     "ROR": round(s.ror, 2), "χ²": round(s.chi_squared, 2),
                     "Signal": "⚠️ YES" if s.is_signal else "No",
                     "Treat N": s.treatment_count, "Ctrl N": s.control_count}
                    for s in signals
                ])
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals detected above threshold.")

        # Interactive AE Detection
        st.subheader("Interactive AE Detection from Narrative")
        sample_narrative = st.text_area(
            "Enter clinical narrative:",
            value="Patient developed Grade 3 neutropenia on Day 14 of pembrolizumab therapy. "
                  "ANC nadir 800/µL. Dose was held. Probable drug-related reaction.",
            height=100,
        )
        if st.button("Detect AEs", key="detect_ae"):
            detected = services["ae_detector"].detect_from_narrative(sample_narrative)
            if detected:
                for ae in detected:
                    st.success(f"**{ae.ae_term}** — Severity: {ae.severity}, "
                               f"Causality: {ae.causality}, Confidence: {ae.confidence:.0%}")
            else:
                st.info("No AEs detected in this text.")


# ═══ TAB 3: Patient Matching ═══════════════════════════════════════

with tab3:
    st.header("Patient-Trial Eligibility Matching")

    if data["patients"].empty:
        st.info("No patient data available.")
    else:
        trial_pts = data["patients"][data["patients"]["trial_id"] == selected_trial]

        if not trial_pts.empty:
            summary = services["matcher"].eligibility_summary(trial_pts, selected_trial)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Screened", summary["total_screened"])
            c2.metric("Eligible", summary["eligible"])
            c3.metric("Ineligible", summary["ineligible"])
            c4.metric("Eligibility Rate", f"{summary['eligibility_rate']}%")

            # Score distribution
            matched = services["matcher"].match_cohort(trial_pts, selected_trial)
            fig = px.histogram(matched, x="eligibility_score", nbins=20,
                               title="Eligibility Score Distribution",
                               color_discrete_sequence=["#00CC96"])
            st.plotly_chart(fig, use_container_width=True)

            # Top failure reasons
            if summary.get("top_failure_reasons"):
                st.subheader("Top Screen Failure Reasons")
                reasons_df = pd.DataFrame(
                    list(summary["top_failure_reasons"].items()),
                    columns=["Reason", "Count"]
                )
                fig = px.bar(reasons_df, x="Count", y="Reason", orientation="h",
                             color="Count", color_continuous_scale="Oranges")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        # Cross-trial analysis
        st.subheader("Cross-Trial Eligibility Analysis")
        all_pts = data["patients"]
        if not all_pts.empty:
            cross_df = services["matcher"].screen_failure_analysis(all_pts)
            if not cross_df.empty:
                fig = px.bar(cross_df, x="trial_id", y="eligibility_rate",
                             title="Eligibility Rate by Trial", text_auto=True,
                             color="eligibility_rate",
                             color_continuous_scale="Greens")
                fig.update_layout(xaxis_title="", yaxis_title="Eligibility Rate (%)")
                st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 4: Protocol Compliance ═════════════════════════════════════

with tab4:
    st.header("Protocol Compliance & Deviations")

    visits_df = data["visits"]
    if visits_df.empty:
        st.info("No visit data available.")
    else:
        # Study-level metrics
        metrics = services["protocol"].study_metrics(visits_df, selected_trial)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Enrolled", metrics.enrollment_count)
        c2.metric("Completion Rate", f"{metrics.completion_rate:.0%}")
        c3.metric("Total Deviations", metrics.deviation_count)
        c4.metric("Major Deviations", metrics.major_deviation_count)

        # Deviations
        deviations = services["protocol"].detect_deviations(visits_df, selected_trial)
        if deviations:
            dev_df = pd.DataFrame([
                {"Patient": d.patient_id, "Type": d.deviation_type,
                 "Severity": d.severity, "Description": d.description}
                for d in deviations
            ])
            st.subheader(f"Protocol Deviations ({len(deviations)} found)")

            # Deviation type breakdown
            type_counts = dev_df["Type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig = px.pie(type_counts, names="Type", values="Count",
                         title="Deviation Types")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(dev_df, use_container_width=True, hide_index=True)
        else:
            st.success("No protocol deviations detected.")

        # Cross-trial metrics
        st.subheader("Cross-Trial Compliance")
        all_metrics = services["protocol"].all_study_metrics(visits_df)
        if not all_metrics.empty:
            fig = px.bar(all_metrics, x="trial_id",
                         y=["completion_rate", "avg_compliance"],
                         barmode="group", title="Completion & Compliance by Trial")
            st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 5: NLP Explorer ════════════════════════════════════════════

with tab5:
    st.header("NLP Explorer — Medical Named Entity Recognition")

    default_text = (
        "The patient, a 67-year-old male on pembrolizumab 200mg IV q3w for "
        "non-small cell lung cancer, presented with Grade 2 nausea and Grade 3 "
        "neutropenia (ANC 900/µL) on Cycle 3 Day 1. ALT was elevated to 3x ULN. "
        "Pembrolizumab was held and G-CSF support initiated. The neutropenia was "
        "assessed as probably related to docetaxel."
    )

    input_text = st.text_area("Enter clinical text for NER extraction:",
                              value=default_text, height=150)

    if st.button("Extract Entities", key="ner_extract"):
        ner_result = services["ner"].extract(input_text)

        # Display entities by type
        entity_types = {}
        for ent in ner_result.entities:
            entity_types.setdefault(ent.label, []).append(ent)

        for label, entities in entity_types.items():
            with st.expander(f"{label} ({len(entities)} found)", expanded=True):
                for e in entities:
                    conf_bar = "🟢" if e.confidence > 0.8 else "🟡" if e.confidence > 0.6 else "🔴"
                    st.markdown(f"{conf_bar} **{e.text}** — confidence: {e.confidence:.0%}")

        # Summary table
        if ner_result.entities:
            ent_df = pd.DataFrame([
                {"Text": e.text, "Type": e.label, "Confidence": f"{e.confidence:.0%}",
                 "Position": f"{e.start}-{e.end}"}
                for e in ner_result.entities
            ])
            st.subheader("All Extracted Entities")
            st.dataframe(ent_df, use_container_width=True, hide_index=True)

    # Sample narratives from data
    if not data["adverse_events"].empty and "narrative" in data["adverse_events"].columns:
        st.subheader("Sample Narratives from Trial Data")
        trial_ae = data["adverse_events"][data["adverse_events"]["trial_id"] == selected_trial]
        samples = trial_ae[trial_ae["narrative"].notna()].head(5)
        for _, row in samples.iterrows():
            with st.expander(f"{row.get('ae_term', 'AE')} — {row.get('patient_id', '')}"):
                st.write(row["narrative"])
