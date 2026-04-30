# ============================================================
#  HOSPITAL BED OCCUPANCY — PHASE 2: FastAPI INFERENCE SERVER
#  Wraps the Phase 1 trained pipeline into a real REST API
#  Author : Ramit Gupta
# ============================================================
# FOLDER STRUCTURE:
#   project/
#   ├── data/
#   ├── models/            ← joblib file saved by Phase 1
#   ├── reports/
#   ├── logs/              ← auto-created, all predictions logged here
#   ├── hospital_pipeline_phase1.py
#   └── app.py             ← THIS FILE
# ============================================================
# RUN:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
# DOCS: http://localhost:8000/docs   ← auto-generated Swagger UI
# ============================================================

import os
import glob
import json
import uuid
import logging
import warnings
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import feature engineering from Phase 1 — same function, no duplication
from hospital_pipeline_phase1 import engineer_features, CLINICAL_THRESHOLD

warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING SETUP
# Every prediction is logged to a JSON file.
# This is how real systems track model behavior
# in production — not print statements.
# ─────────────────────────────────────────────

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# We load the model ONCE when the server starts,
# not on every request. This is how production
# APIs work — loading on every request would be
# 10–50x slower.
# ─────────────────────────────────────────────

def load_latest_model():
    model_files = glob.glob("models/*.joblib")
    if not model_files:
        raise FileNotFoundError(
            "No model found in models/. Run hospital_pipeline_phase1.py first."
        )
    latest = max(model_files, key=os.path.getctime)
    pipeline = joblib.load(latest)
    logger.info(f"Model loaded: {latest}")
    print(f"✅ Model loaded: {latest}")
    return pipeline, latest

PIPELINE, MODEL_PATH = load_latest_model()


# ─────────────────────────────────────────────
# SHAP EXPLAINER — loaded once at startup
# ─────────────────────────────────────────────

def build_explainer(pipeline):
    try:
        model    = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        print("✅ SHAP explainer ready")
        return explainer
    except Exception as e:
        print(f"⚠️  SHAP explainer not available: {e}")
        return None

EXPLAINER = build_explainer(PIPELINE)


# ─────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Hospital LOS Prediction API",
    description=(
        "Predicts patient Length of Stay (LOS) at admission time. "
        "Built on Microsoft Hospital Dataset. "
        "Returns prediction, confidence range, and SHAP explanations."
    ),
    version="1.0.0",
)

# Allow frontend / Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS (Pydantic)
# Pydantic validates every incoming request.
# If a field is wrong type or missing — FastAPI
# auto-returns a 422 error with a clear message.
# This is how real APIs prevent bad data.
# ─────────────────────────────────────────────

class PatientAdmission(BaseModel):
    # ── Patient demographics
    gender:  str = Field(..., example="M", description="M or F")
    bmi:     float = Field(..., example=27.5, description="BMI in kg/m2")

    # ── Readmission history
    rcount:  int = Field(..., example=1, description="Readmissions in last 180 days")

    # ── Comorbidity flags (Yes / No)
    dialysisrenalendstage:      str = Field("No", example="No")
    asthma:                     str = Field("No", example="No")
    irondef:                    str = Field("No", example="No")
    pneum:                      str = Field("No", example="No")
    substancedependence:        str = Field("No", example="No")
    psychologicaldisordermajor: str = Field("No", example="No")
    depress:                    str = Field("No", example="No")
    psychother:                 str = Field("No", example="No")
    fibrosisandother:           str = Field("No", example="No")
    malnutrition:               str = Field("No", example="No")
    hemo:                       str = Field("No", example="No")

    # ── Lab values
    hematocrit:   float = Field(..., example=36.5)
    neutrophils:  float = Field(..., example=7.2)
    sodium:       float = Field(..., example=138.0)
    glucose:      float = Field(..., example=105.0)
    bloodureanitro: float = Field(..., example=18.0)
    creatinine:   float = Field(..., example=1.1)
    pulse:        float = Field(..., example=78.0)
    respiration:  float = Field(..., example=18.0)

    # ── Other
    secondarydiagnosisnonicd9: int   = Field(0, example=0)
    facid:                     int   = Field(..., example=3, description="Facility ID")

    # ── Optional: admission date. Defaults to today if not provided.
    vdate: Optional[str] = Field(
        None, example="2025-04-30",
        description="Admission date (YYYY-MM-DD). Defaults to today."
    )

    @validator("gender")
    def gender_must_be_valid(cls, v):
        if v.upper() not in ("M", "F"):
            raise ValueError("gender must be M or F")
        return v.upper()

    @validator("bmi")
    def bmi_must_be_positive(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("BMI must be between 1 and 100")
        return v


class SHAPReason(BaseModel):
    feature:      str
    impact:       str   # "increases stay" or "decreases stay"
    shap_value:   float

class PredictionResponse(BaseModel):
    request_id:           str
    predicted_los_days:   float
    confidence_range:     str
    clinical_note:        str
    top_reasons:          list[SHAPReason]
    model_used:           str
    predicted_at:         str


# ─────────────────────────────────────────────
# CORE PREDICTION LOGIC
# ─────────────────────────────────────────────

def run_prediction(patient: PatientAdmission) -> PredictionResponse:
    request_id = str(uuid.uuid4())[:8]

    # Build input dataframe
    patient_dict = patient.dict()

    # Handle admission date
    if patient_dict.get("vdate"):
        patient_dict["vdate"] = pd.to_datetime(patient_dict["vdate"], errors="coerce")
    else:
        patient_dict["vdate"] = pd.Timestamp.today().normalize()

    df_input = pd.DataFrame([patient_dict])

    # Feature engineering — same function as Phase 1, no duplication
    df_input = engineer_features(df_input)

    # Predict
    raw_pred = PIPELINE.predict(df_input)[0]
    los_days = max(1.0, round(float(raw_pred), 1))

    # Confidence range
    low  = max(1, round(los_days - CLINICAL_THRESHOLD, 1))
    high = round(los_days + CLINICAL_THRESHOLD, 1)

    # ── SHAP explanation ──────────────────────
    top_reasons = []
    if EXPLAINER is not None:
        try:
            preproc       = PIPELINE.named_steps["preprocessor"]
            X_transformed = preproc.transform(df_input)

            shap_vals     = EXPLAINER.shap_values(X_transformed)[0]

            num_features  = list(preproc.transformers_[0][2])
            cat_features  = list(preproc.transformers_[1][2])
            feature_names = num_features + cat_features

            # Get top 3 features by absolute SHAP value
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
            for idx in top_idx:
                top_reasons.append(SHAPReason(
                    feature    = feature_names[idx],
                    impact     = "increases stay" if shap_vals[idx] > 0 else "decreases stay",
                    shap_value = round(float(shap_vals[idx]), 3)
                ))
        except Exception as e:
            logger.warning(f"SHAP failed for request {request_id}: {e}")

    # ── Log every prediction ──────────────────
    log_entry = {
        "request_id":         request_id,
        "predicted_los_days": los_days,
        "comorbidity_score":  int(df_input.get("comorbidity_score", [0])[0])
                              if "comorbidity_score" in df_input.columns else None,
        "facid":              patient.facid,
        "timestamp":          datetime.utcnow().isoformat()
    }
    logger.info(json.dumps(log_entry))

    return PredictionResponse(
        request_id         = request_id,
        predicted_los_days = los_days,
        confidence_range   = f"{low}–{high} days",
        clinical_note      = (
            f"Prediction is within ±{CLINICAL_THRESHOLD} days clinically acceptable range"
        ),
        top_reasons        = top_reasons,
        model_used         = os.path.basename(MODEL_PATH),
        predicted_at       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    )


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms API is running."""
    return {
        "status":  "running",
        "service": "Hospital LOS Prediction API",
        "version": "1.0.0",
        "docs":    "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check used by Docker / load balancers."""
    return {
        "status":      "healthy",
        "model_loaded": os.path.basename(MODEL_PATH),
        "shap_ready":  EXPLAINER is not None,
        "timestamp":   datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientAdmission):
    """
    Predict Length of Stay for a newly admitted patient.

    - Accepts patient demographics, comorbidity flags, and lab values
    - Returns predicted LOS in days + confidence range
    - Returns top 3 SHAP reasons explaining the prediction
    - Logs every prediction to logs/predictions.log
    """
    try:
        result = run_prediction(patient)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientAdmission]):
    """
    Batch prediction — send multiple patients at once.
    Returns a list of predictions. Useful for shift-start planning.
    Max 100 patients per request.
    """
    if len(patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 100 patients per request"
        )
    results = []
    for patient in patients:
        try:
            results.append(run_prediction(patient))
        except Exception as e:
            results.append({"error": str(e)})
    return {"total": len(results), "predictions": results}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the currently loaded model."""
    return {
        "model_file":         os.path.basename(MODEL_PATH),
        "model_path":         MODEL_PATH,
        "clinical_threshold": f"±{CLINICAL_THRESHOLD} days",
        "shap_available":     EXPLAINER is not None,
        "loaded_at":          datetime.utcnow().isoformat()
    }


@app.get("/logs/recent", tags=["Monitoring"])
def recent_logs(n: int = 10):
    """
    Returns the last N prediction log entries.
    This is how you monitor model behavior in production
    without opening a server — just call this endpoint.
    """
    log_path = "logs/predictions.log"
    if not os.path.exists(log_path):
        return {"logs": [], "message": "No predictions logged yet"}

    with open(log_path, "r") as f:
        lines = f.readlines()

    # Parse only JSON lines (skip plain text log lines)
    entries = []
    for line in reversed(lines):
        try:
            json_part = line.split(" | INFO | ")[-1].strip()
            entries.append(json.loads(json_part))
            if len(entries) >= n:
                break
        except Exception:
            continue

    return {"total_logged": len(lines), "recent": entries}
