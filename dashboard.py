# ============================================================
#  HOSPITAL MANAGEMENT SYSTEM — Streamlit Dashboard
#  Connects to: LengthOfStay.csv (real data) + FastAPI backend
#  Author : Ramit Gupta
# ============================================================
# INSTALL:
#   pip install streamlit pandas plotly requests scikit-learn
# RUN:
#   streamlit run dashboard.py
# Make sure FastAPI is running: uvicorn app:app --reload --port 8000
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first streamlit call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hospital Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    .metric-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high { color: #dc3545; font-weight: 600; }
    .risk-med  { color: #fd7e14; font-weight: 600; }
    .risk-low  { color: #198754; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #212529;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .api-status-ok   { color: #198754; font-size: 0.8rem; }
    .api-status-fail { color: #dc3545; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

import os
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
DATA_PATH   = "data/LengthOfStay.csv"

# ─────────────────────────────────────────────
# DATA LOADING — cached so it loads once
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Parse dates
    for col in ["vdate", "discharged"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Clean target
    df = df[df["lengthofstay"].notna() & (df["lengthofstay"] > 0)]
    df["lengthofstay"] = df["lengthofstay"].clip(upper=df["lengthofstay"].quantile(0.99))

    # Feature engineering for analytics
    if "vdate" in df.columns:
        df["admit_dayofweek"] = df["vdate"].dt.day_name()
        df["admit_month"]     = df["vdate"].dt.month
        df["admit_month_name"]= df["vdate"].dt.strftime("%b")
        df["admit_quarter"]   = df["vdate"].dt.quarter
        df["admit_season"]    = df["admit_month"].map(
            {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",
             5:"Spring",6:"Summer",7:"Summer",8:"Summer",
             9:"Fall",10:"Fall",11:"Fall"}
        )

    # Comorbidity flags
    condition_flags = [
        "dialysisrenalendstage","asthma","irondef","pneum",
        "substancedependence","psychologicaldisordermajor",
        "depress","psychother","fibrosisandother","malnutrition","hemo"
    ]
    existing_flags = [c for c in condition_flags if c in df.columns]
    for col in existing_flags:
        df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["comorbidity_score"] = df[existing_flags].sum(axis=1)

    # LOS categories
    df["los_category"] = pd.cut(
        df["lengthofstay"],
        bins=[0, 3, 7, 14, 100],
        labels=["Short (<3d)", "Medium (3–7d)", "Long (7–14d)", "Extended (14d+)"]
    )

    # Risk flag: high comorbidity + high rcount = high risk
    if "rcount" in df.columns:
        df["rcount"] = pd.to_numeric(df["rcount"], errors="coerce").fillna(0)
        df["risk_flag"] = "Low"
        df.loc[(df["comorbidity_score"] >= 3) | (df["rcount"] >= 3), "risk_flag"] = "Medium"
        df.loc[(df["comorbidity_score"] >= 5) & (df["rcount"] >= 2), "risk_flag"] = "High"

    return df


def check_api():
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🏥 Hospital Settings")

        total_beds = st.number_input(
            "Total Hospital Beds",
            min_value=50, max_value=5000,
            value=500, step=10
        )

        st.markdown("---")
        st.markdown("### Filters")

        # Facility filter
        if "facid" in df.columns:
            facilities = ["All"] + sorted(df["facid"].dropna().unique().tolist())
            selected_facility = st.selectbox("Facility", facilities)
        else:
            selected_facility = "All"

        # Gender filter
        if "gender" in df.columns:
            genders = ["All"] + sorted(df["gender"].dropna().unique().tolist())
            selected_gender = st.selectbox("Gender", genders)
        else:
            selected_gender = "All"

        # Season filter
        if "admit_season" in df.columns:
            seasons = ["All"] + sorted(df["admit_season"].dropna().unique().tolist())
            selected_season = st.selectbox("Season", seasons)
        else:
            selected_season = "All"

        # LOS range slider
        los_min = int(df["lengthofstay"].min())
        los_max = int(df["lengthofstay"].max())
        los_range = st.slider(
            "LOS Range (days)",
            min_value=los_min, max_value=los_max,
            value=(los_min, los_max)
        )

        st.markdown("---")
        api_ok = check_api()
        if api_ok:
            st.markdown('<p class="api-status-ok">✅ FastAPI connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="api-status-fail">⚠️ FastAPI offline — start uvicorn</p>', unsafe_allow_html=True)

        st.caption(f"Dataset: {len(df):,} patients")
        st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y')}")

    # Apply filters
    filtered = df.copy()
    if selected_facility != "All":
        filtered = filtered[filtered["facid"] == selected_facility]
    if selected_gender != "All":
        filtered = filtered[filtered["gender"] == selected_gender]
    if selected_season != "All" and "admit_season" in filtered.columns:
        filtered = filtered[filtered["admit_season"] == selected_season]
    filtered = filtered[
        (filtered["lengthofstay"] >= los_range[0]) &
        (filtered["lengthofstay"] <= los_range[1])
    ]

    return filtered, total_beds


# ─────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────

def tab_overview(df, total_beds):
    occupied    = len(df[df["lengthofstay"] >= 1]) // 10  # proxy for active patients
    occupied    = min(occupied, total_beds)
    occ_pct     = round(occupied / total_beds * 100, 1)
    avg_los     = round(df["lengthofstay"].mean(), 1)
    high_risk   = len(df[df["risk_flag"] == "High"]) if "risk_flag" in df.columns else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Patients",     f"{len(df):,}")
    col2.metric("Active (est.)",      f"{occupied:,}")
    col3.metric("Bed Occupancy",      f"{occ_pct}%",
                delta=f"{occ_pct - 70:.1f}% vs target")
    col4.metric("Avg LOS",            f"{avg_los}d")
    col5.metric("Total Beds",         f"{total_beds:,}")
    col6.metric("High Risk Patients", f"{high_risk:,}",
                delta=None if high_risk == 0 else "needs attention",
                delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">Monthly Admission Volume</div>', unsafe_allow_html=True)
        if "admit_month" in df.columns:
            monthly = df.groupby("admit_month").size().reset_index(name="admissions")
            monthly["month_name"] = pd.to_datetime(monthly["admit_month"], format="%m").dt.strftime("%b")
            fig = px.bar(
                monthly, x="month_name", y="admissions",
                color="admissions",
                color_continuous_scale=["#cfe2ff", "#0d6efd"],
                labels={"month_name": "Month", "admissions": "Admissions"}
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10, b=10), height=280,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">LOS Category Split</div>', unsafe_allow_html=True)
        if "los_category" in df.columns:
            cat_counts = df["los_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.pie(
                cat_counts, values="Count", names="Category",
                color_discrete_sequence=["#198754", "#0d6efd", "#fd7e14", "#dc3545"],
                hole=0.55
            )
            fig.update_layout(margin=dict(t=10, b=10), height=280,
                              legend=dict(font=dict(size=10)))
            fig.update_traces(textinfo="percent", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2 — LOS ANALYSIS
# ─────────────────────────────────────────────

def tab_los(df):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg LOS",    f"{df['lengthofstay'].mean():.1f}d")
    col2.metric("Median LOS", f"{df['lengthofstay'].median():.1f}d")
    col3.metric("Max LOS",    f"{df['lengthofstay'].max():.0f}d")
    col4.metric("Std Dev",    f"{df['lengthofstay'].std():.1f}d")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">LOS Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="lengthofstay", nbins=30,
            labels={"lengthofstay": "Length of Stay (days)", "count": "Patients"},
            color_discrete_sequence=["#0d6efd"]
        )
        fig.update_layout(margin=dict(t=10,b=10), height=260,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(gridcolor="#f0f0f0")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">LOS by Comorbidity Score</div>', unsafe_allow_html=True)
        comorbid_los = df.groupby("comorbidity_score")["lengthofstay"].mean().reset_index()
        fig = px.bar(
            comorbid_los, x="comorbidity_score", y="lengthofstay",
            labels={"comorbidity_score": "Comorbidity Score", "lengthofstay": "Avg LOS (days)"},
            color="lengthofstay",
            color_continuous_scale=["#cfe2ff", "#dc3545"]
        )
        fig.update_layout(margin=dict(t=10,b=10), height=260,
                          showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">LOS by Season</div>', unsafe_allow_html=True)
    if "admit_season" in df.columns:
        season_los = df.groupby("admit_season")["lengthofstay"].mean().reset_index()
        fig = px.bar(
            season_los, x="admit_season", y="lengthofstay",
            color="admit_season",
            color_discrete_sequence=["#198754","#fd7e14","#0d6efd","#6f42c1"],
            labels={"admit_season": "Season", "lengthofstay": "Avg LOS (days)"}
        )
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10), height=220,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3 — BED OCCUPANCY
# ─────────────────────────────────────────────

def tab_beds(df, total_beds):
    occupied  = min(len(df) // 10, total_beds)
    available = total_beds - occupied
    occ_pct   = round(occupied / total_beds * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Beds",      f"{total_beds:,}")
    col2.metric("Occupied",        f"{occupied:,}")
    col3.metric("Available",       f"{available:,}")
    col4.metric("Occupancy Rate",  f"{occ_pct}%",
                delta=f"{'Critical' if occ_pct>=90 else 'High Load' if occ_pct>=80 else 'Normal'}",
                delta_color="inverse" if occ_pct >= 80 else "normal")

    st.markdown("---")

    # Occupancy gauge
    st.markdown('<div class="section-header">Occupancy Gauge</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=occ_pct,
        delta={"reference": 80, "valueformat": ".1f"},
        title={"text": "Bed Occupancy %", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#dc3545" if occ_pct >= 90 else "#fd7e14" if occ_pct >= 80 else "#198754"},
            "steps": [
                {"range": [0, 70],  "color": "#d1e7dd"},
                {"range": [70, 85], "color": "#fff3cd"},
                {"range": [85, 100],"color": "#f8d7da"}
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 85}
        }
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Per-facility occupancy if facid available
    if "facid" in df.columns:
        st.markdown('<div class="section-header">Occupancy by Facility</div>', unsafe_allow_html=True)
        fac_counts = df.groupby("facid").size().reset_index(name="patients")
        fac_counts["beds_alloc"]  = total_beds // fac_counts["facid"].nunique()
        fac_counts["occupancy"]   = (fac_counts["patients"] / fac_counts["beds_alloc"] * 100).clip(upper=100).round(1)
        fac_counts["facid"]       = "Facility " + fac_counts["facid"].astype(str)

        fig = px.bar(
            fac_counts, x="facid", y="occupancy",
            color="occupancy",
            color_continuous_scale=["#198754","#fd7e14","#dc3545"],
            range_color=[0, 100],
            labels={"facid": "Facility", "occupancy": "Occupancy %"}
        )
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="85% threshold")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=10,b=10), height=250,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(gridcolor="#f0f0f0", range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 — DEMOGRAPHICS
# ─────────────────────────────────────────────

def tab_demographics(df):
    col_left, col_right = st.columns(2)

    with col_left:
        if "gender" in df.columns:
            st.markdown('<div class="section-header">Gender Distribution</div>', unsafe_allow_html=True)
            gender_counts = df["gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig = px.pie(gender_counts, values="Count", names="Gender",
                         color_discrete_sequence=["#0d6efd","#d4537e"],
                         hole=0.5)
            fig.update_layout(margin=dict(t=10,b=10), height=240)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">LOS by Gender</div>', unsafe_allow_html=True)
        if "gender" in df.columns:
            gender_los = df.groupby("gender")["lengthofstay"].mean().reset_index()
            fig = px.bar(gender_los, x="gender", y="lengthofstay",
                         color="gender",
                         color_discrete_sequence=["#0d6efd","#d4537e"],
                         labels={"gender":"Gender","lengthofstay":"Avg LOS (days)"})
            fig.update_layout(showlegend=False, margin=dict(t=10,b=10), height=240,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

    # Comorbidity profile
    st.markdown('<div class="section-header">Top Comorbidities in Patient Population</div>', unsafe_allow_html=True)
    condition_cols = [
        "dialysisrenalendstage","asthma","irondef","pneum",
        "substancedependence","psychologicaldisordermajor",
        "depress","psychother","fibrosisandother","malnutrition","hemo"
    ]
    existing = [c for c in condition_cols if c in df.columns]
    if existing:
        condition_rates = {c: df[c].mean() * 100 for c in existing}
        cond_df = pd.DataFrame({
            "Condition": list(condition_rates.keys()),
            "Prevalence %": list(condition_rates.values())
        }).sort_values("Prevalence %", ascending=True)

        fig = px.bar(cond_df, x="Prevalence %", y="Condition",
                     orientation="h",
                     color="Prevalence %",
                     color_continuous_scale=["#cfe2ff","#dc3545"])
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=10,b=10), height=320,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_xaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5 — RISK FLAGS
# ─────────────────────────────────────────────
def tab_risk(df):
    if "risk_flag" not in df.columns:
        st.info("Risk flags not available in filtered data.")
        return

    # ── KPIs ──
    col1, col2, col3 = st.columns(3)
    high  = len(df[df["risk_flag"] == "High"])
    med   = len(df[df["risk_flag"] == "Medium"])
    low   = len(df[df["risk_flag"] == "Low"])

    col1.metric("High Risk", high, delta="Immediate attention", delta_color="inverse")
    col2.metric("Medium Risk", med, delta="Monitor closely")
    col3.metric("Low Risk", low)

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    # ── TABLE (NO STYLER ERROR) ──
    with col_left:
        st.markdown("### High Risk Patient Sample")

        high_risk_df = df[df["risk_flag"] == "High"].head(15).copy()

        display_cols = ["rcount", "comorbidity_score", "lengthofstay", "gender", "risk_flag"]
        display_cols = [c for c in display_cols if c in high_risk_df.columns]

        st.dataframe(
            high_risk_df[display_cols],
            use_container_width=True,
            height=340
        )

    # ── PIE CHART ──
    with col_right:
        st.markdown("### Risk Distribution")

        risk_counts = df["risk_flag"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig = px.pie(
            risk_counts,
            values="Count",
            names="Risk Level",
            color="Risk Level",
            color_discrete_map={
                "High": "#dc3545",
                "Medium": "#fd7e14",
                "Low": "#198754"
            },
            hole=0.5
        )
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

    # ── BAR CHART ──
    st.markdown("### Avg LOS by Risk Level")

    risk_los = df.groupby("risk_flag")["lengthofstay"].mean().reset_index()

    fig = px.bar(
        risk_los,
        x="risk_flag",
        y="lengthofstay",
        color="risk_flag",
        color_discrete_map={
            "High": "#dc3545",
            "Medium": "#fd7e14",
            "Low": "#198754"
        },
        labels={
            "risk_flag": "Risk Level",
            "lengthofstay": "Avg LOS (days)"
        }
    )

    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 6 — TRENDS & SEASONALITY
# ─────────────────────────────────────────────

def tab_trends(df):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Admissions by Day of Week</div>', unsafe_allow_html=True)
        if "admit_dayofweek" in df.columns:
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_counts = df.groupby("admit_dayofweek").size().reset_index(name="admissions")
            dow_counts["admit_dayofweek"] = pd.Categorical(
                dow_counts["admit_dayofweek"], categories=dow_order, ordered=True
            )
            dow_counts = dow_counts.sort_values("admit_dayofweek")
            fig = px.bar(dow_counts, x="admit_dayofweek", y="admissions",
                         color="admissions",
                         color_continuous_scale=["#cfe2ff","#dc3545"],
                         labels={"admit_dayofweek":"Day","admissions":"Admissions"})
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10), height=260,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(gridcolor="#f0f0f0")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Admissions by Season</div>', unsafe_allow_html=True)
        if "admit_season" in df.columns:
            season_counts = df.groupby("admit_season").size().reset_index(name="admissions")
            fig = px.bar(season_counts, x="admit_season", y="admissions",
                         color="admit_season",
                         color_discrete_sequence=["#0d6efd","#198754","#fd7e14","#6f42c1"],
                         labels={"admit_season":"Season","admissions":"Admissions"})
            fig.update_layout(showlegend=False, margin=dict(t=10,b=10), height=260,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Monthly Admissions — Seasonal Trend</div>', unsafe_allow_html=True)
    if "admit_month" in df.columns:
        monthly = df.groupby(["admit_month","admit_month_name"]).size().reset_index(name="admissions")
        monthly = monthly.sort_values("admit_month")
        fig = px.line(monthly, x="admit_month_name", y="admissions",
                      markers=True,
                      labels={"admit_month_name":"Month","admissions":"Admissions"},
                      color_discrete_sequence=["#0d6efd"])
        fig.update_traces(line_width=2.5, marker_size=7)
        fig.update_layout(margin=dict(t=10,b=10), height=240,
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 7 — LIVE PREDICT (calls FastAPI)
# ─────────────────────────────────────────────

def tab_predict():
    api_ok = check_api()

    if not api_ok:
        st.error("⚠️ FastAPI server is not running. Start it with: `uvicorn app:app --reload --port 8000`")
        st.info("You can still use the form — start the API then click Predict again.")

    st.markdown('<div class="section-header">Predict Length of Stay for New Patient</div>', unsafe_allow_html=True)
    st.caption("Calls your live FastAPI /predict endpoint — same model trained in Phase 1")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender  = st.selectbox("Gender", ["M", "F"])
        bmi     = st.number_input("BMI", 10.0, 60.0, 27.5, 0.5)
        rcount  = st.number_input("Prior readmissions (rcount)", 0, 20, 1)
        facid   = st.number_input("Facility ID", 1, 10, 3)

    with col2:
        st.markdown("**Comorbidities**")
        asthma     = st.checkbox("Asthma")
        depress    = st.checkbox("Depression")
        pneum      = st.checkbox("Pneumonia")
        irondef    = st.checkbox("Iron Deficiency")
        malnutrition = st.checkbox("Malnutrition")
        hemo       = st.checkbox("Hematological condition")

    with col3:
        st.markdown("**Lab Values**")
        hematocrit   = st.number_input("Hematocrit",    20.0, 60.0, 36.5, 0.5)
        sodium       = st.number_input("Sodium",        120.0, 160.0, 138.0, 0.5)
        glucose      = st.number_input("Glucose",       50.0,  400.0, 105.0, 1.0)
        bloodureanitro = st.number_input("Blood Urea Nitrogen", 1.0, 100.0, 18.0, 0.5)
        creatinine   = st.number_input("Creatinine",    0.1,  15.0,  1.1,  0.1)
        pulse        = st.number_input("Pulse",         40.0, 150.0, 78.0, 1.0)
        respiration  = st.number_input("Respiration",   8.0,  40.0,  18.0, 0.5)
        neutrophils  = st.number_input("Neutrophils",   0.5,  20.0,  7.2,  0.1)

    def yn(v): return "Yes" if v else "No"

    payload = {
        "gender": gender, "bmi": bmi, "rcount": int(rcount),
        "facid": int(facid),
        "dialysisrenalendstage": "No", "asthma": yn(asthma),
        "irondef": yn(irondef), "pneum": yn(pneum),
        "substancedependence": "No", "psychologicaldisordermajor": "No",
        "depress": yn(depress), "psychother": "No",
        "fibrosisandother": "No", "malnutrition": yn(malnutrition),
        "hemo": yn(hemo),
        "hematocrit": hematocrit, "neutrophils": neutrophils,
        "sodium": sodium, "glucose": glucose,
        "bloodureanitro": bloodureanitro, "creatinine": creatinine,
        "pulse": pulse, "respiration": respiration,
        "secondarydiagnosisnonicd9": 0
    }

    if st.button("🔮 Predict LOS", type="primary", disabled=not api_ok):
        with st.spinner("Calling FastAPI..."):
            try:
                r = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
                result = r.json()

                los   = result.get("predicted_los_days", "—")
                range_ = result.get("confidence_range", "—")
                req_id = result.get("request_id", "—")
                reasons = result.get("top_reasons", [])

                st.success(f"✅ Prediction complete — Request ID: {req_id}")

                rc1, rc2 = st.columns(2)
                rc1.metric("Predicted LOS", f"{los} days")
                rc2.metric("Confidence Range", range_)

                if reasons:
                    st.markdown("**Top 3 Clinical Drivers (SHAP)**")
                    for r in reasons:
                        direction = "🔴" if r["impact"] == "increases stay" else "🟢"
                        st.markdown(
                            f"{direction} **{r['feature']}** — {r['impact']} "
                            f"(SHAP: {r['shap_value']})"
                        )

                with st.expander("Full API Response"):
                    st.json(result)

            except Exception as e:
                st.error(f"API call failed: {e}")

    # Batch section
    st.markdown("---")
    st.markdown('<div class="section-header">Recent Predictions Log</div>', unsafe_allow_html=True)
    if st.button("Fetch Last 10 Predictions from API"):
        try:
            r = requests.get(f"{FASTAPI_URL}/logs/recent?n=10", timeout=5)
            data = r.json()
            if data.get("recent"):
                log_df = pd.DataFrame(data["recent"])
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("No predictions logged yet. Make some predictions above first.")
        except Exception as e:
            st.error(f"Could not fetch logs: {e}")


# ─────────────────────────────────────────────
# TAB 8 — OPERATIONAL KPIs
# ─────────────────────────────────────────────

def tab_kpi(df, total_beds):
    occupied = min(len(df) // 10, total_beds)
    occ_pct  = round(occupied / total_beds * 100, 1)
    avg_los  = round(df["lengthofstay"].mean(), 1)
    readmit_rate = round(df[df["rcount"] > 0].shape[0] / len(df) * 100, 1) if "rcount" in df.columns else 0
    turnover = round(occupied / avg_los * 30 / total_beds, 2) if avg_los > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bed Turnover Rate", f"{turnover}/month",
                delta="per bed", delta_color="off")
    col2.metric("Avg LOS",           f"{avg_los}d",
                delta=f"{'Above' if avg_los > 5 else 'Below'} 5d benchmark",
                delta_color="inverse" if avg_los > 5 else "normal")
    col3.metric("Occupancy Rate",    f"{occ_pct}%",
                delta=f"{'Critical' if occ_pct>=90 else 'OK'}",
                delta_color="inverse" if occ_pct >= 85 else "normal")
    col4.metric("Readmission Rate",  f"{readmit_rate}%",
                delta="Target <10%" if readmit_rate > 10 else "On target",
                delta_color="inverse" if readmit_rate > 10 else "normal")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">LOS Distribution by Readmission Count</div>', unsafe_allow_html=True)
        if "rcount" in df.columns:
            rcount_los = df.groupby("rcount")["lengthofstay"].mean().reset_index().head(10)
            fig = px.bar(rcount_los, x="rcount", y="lengthofstay",
                         color="lengthofstay",
                         color_continuous_scale=["#cfe2ff","#dc3545"],
                         labels={"rcount":"Prior Readmissions","lengthofstay":"Avg LOS (days)"})
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10), height=260,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">High Risk vs LOS Scatter</div>', unsafe_allow_html=True)
        if "risk_flag" in df.columns:
            sample = df.sample(min(500, len(df)))
            fig = px.scatter(
                sample, x="comorbidity_score", y="lengthofstay",
                color="risk_flag",
                color_discrete_map={"High":"#dc3545","Medium":"#fd7e14","Low":"#198754"},
                labels={"comorbidity_score":"Comorbidity Score","lengthofstay":"LOS (days)"},
                opacity=0.6
            )
            fig.update_layout(margin=dict(t=10,b=10), height=260,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    st.markdown("## 🏥 Hospital Management System")
    st.caption("Real-time analytics powered by Microsoft Hospital Dataset + XGBoost ML Model")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `LengthOfStay.csv` inside a `data/` folder.")
        st.stop()

    filtered_df, total_beds = render_sidebar(df)

    if len(filtered_df) == 0:
        st.warning("No data matches current filters. Adjust filters in the sidebar.")
        st.stop()

    tabs = st.tabs([
        "📊 Overview",
        "🛏️ LOS Analysis",
        "🏥 Bed Occupancy",
        "👥 Demographics",
        "⚠️ Risk Flags",
        "📈 Trends",
        "🔮 Live Predict",
        "📋 KPIs"
    ])

    with tabs[0]: tab_overview(filtered_df, total_beds)
    with tabs[1]: tab_los(filtered_df)
    with tabs[2]: tab_beds(filtered_df, total_beds)
    with tabs[3]: tab_demographics(filtered_df)
    with tabs[4]: tab_risk(filtered_df)
    with tabs[5]: tab_trends(filtered_df)
    with tabs[6]: tab_predict()
    with tabs[7]: tab_kpi(filtered_df, total_beds)


if __name__ == "__main__":
    main()