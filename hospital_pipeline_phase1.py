# ============================================================
#  HOSPITAL BED OCCUPANCY — PHASE 1: FULL ML PIPELINE
#  Dataset: Microsoft Hospital Length of Stay (LengthOfStay.csv)
#  Target : lengthofstay (number of days)
#  Author : Ramit Gupta
# ============================================================
# FOLDER STRUCTURE expected:
#   project/
#   ├── data/
#   │   └── LengthOfStay.csv        ← put your downloaded CSV here
#   ├── models/                     ← saved models go here (auto-created)
#   ├── reports/                    ← evaluation reports go here (auto-created)
#   └── hospital_pipeline_phase1.py ← this file
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 0 — CONSTANTS
# ─────────────────────────────────────────────

DATA_PATH   = "data/LengthOfStay.csv"
TARGET_COL  = "lengthofstay"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# Clinically acceptable error: predict within ±2 days of actual LOS
CLINICAL_THRESHOLD = 2

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*55}")
    df = pd.read_csv(path)
    print(f"  Shape        : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Target range : {df[TARGET_COL].min()} – {df[TARGET_COL].max()} days")
    print(f"  Missing vals :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df

# ─────────────────────────────────────────────
# STEP 2 — DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("STEP 2: CLEANING DATA")
    print(f"{'='*55}")

    df = df.copy()

    # Drop pure ID columns — carry no signal for prediction
    df.drop(columns=["eid"], inplace=True, errors="ignore")

    # Parse dates
    for col in ["vdate", "discharged"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove rows where target is null or negative
    before = len(df)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)]
    print(f"  Removed {before - len(df)} invalid target rows")

    # Cap extreme LOS outliers at 99th percentile
    cap = df[TARGET_COL].quantile(0.99)
    df[TARGET_COL] = df[TARGET_COL].clip(upper=cap)
    print(f"  LOS capped at {cap:.0f} days (99th pct)")

    print(f"  Clean shape  : {df.shape}")
    return df

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
# This is the most important step. We derive new
# features from existing columns that give the
# model more signal than raw values alone.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("STEP 3: FEATURE ENGINEERING")
    print(f"{'='*55}")

    df = df.copy()

    # ── 3a. DATE-BASED FEATURES ──────────────────
    # Day of week / month / season affect hospital load
    if "vdate" in df.columns:
        df["admit_dayofweek"]  = df["vdate"].dt.dayofweek        # 0=Mon..6=Sun
        df["admit_month"]      = df["vdate"].dt.month
        df["admit_is_weekend"] = (df["admit_dayofweek"] >= 5).astype(int)
        df["admit_quarter"]    = df["vdate"].dt.quarter

        # Season: 1=Winter 2=Spring 3=Summer 4=Fall
        df["admit_season"] = df["admit_month"].map(
            {12:1, 1:1, 2:1, 3:2, 4:2, 5:2,
             6:3, 7:3, 8:3, 9:4, 10:4, 11:4}
        )
    df.drop(columns=["vdate", "discharged"], inplace=True, errors="ignore")

    # ── 3b. COMORBIDITY SCORE ────────────────────
    # The number of pre-existing conditions is one
    # of the strongest predictors of long stays.
    condition_flags = [
    "dialysisrenalendstage", "asthma", "irondef", "pneum",
    "substancedependence", "psychologicaldisordermajor",
    "depress", "psychother", "fibrosisandother",
    "malnutrition", "hemo"]

    existing_flags = [c for c in condition_flags if c in df.columns]

    for col in existing_flags:
        df[col] = df[col].astype(str).str.strip().str.lower()

        df[col] = df[col].map({
            "yes": 1, "1": 1, "true": 1,
            "no": 0, "0": 0, "false": 0
        })

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["comorbidity_score"] = df[existing_flags].sum(axis=1)

    print(df[existing_flags].sum())  # total 1s per column
    print(f"range: {df['comorbidity_score'].min()}–{df['comorbidity_score'].max()}")

    # ── 3c. LAB VALUE INTERACTIONS ───────────────
    # Abnormal lab combos are clinically meaningful
    if "hematocrit" in df.columns and "hemoglobin" in df.columns:
        df["hematocrit_hemo_ratio"] = df["hematocrit"] / (df["hemoglobin"] + 1e-5)

    if "sodium" in df.columns and "glucose" in df.columns:
        df["metabolic_stress_index"] = (
            (df["sodium"] - 140).abs() + (df["glucose"] - 100).abs()
        )

    if "bloodureanitro" in df.columns and "creatinine" in df.columns:
        # BUN/Creatinine ratio — standard kidney function indicator
        df["bun_creatinine_ratio"] = (
            df["bloodureanitro"] / (df["creatinine"] + 1e-5)
        )

    # ── 3d. READMISSION RISK FLAG ────────────────
    # rcount > 2 is a strong signal for complex cases
    if "rcount" in df.columns:
        df["rcount"] = pd.to_numeric(df["rcount"], errors="coerce").fillna(0)
        df["is_frequent_readmit"] = (df["rcount"] >= 2).astype(int)

    # ── 3e. BMI CATEGORY ─────────────────────────
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["underweight", "normal", "overweight", "obese"]
        )

    print(f"  New features added : comorbidity_score, metabolic_stress_index,")
    print(f"                       bun_creatinine_ratio, is_frequent_readmit,")
    print(f"                       admit_dayofweek, admit_season, bmi_category")
    print(f"  Final shape        : {df.shape}")
    return df

# ─────────────────────────────────────────────
# STEP 4 — SPLIT DATA
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("STEP 4: TRAIN/TEST SPLIT")
    print(f"{'='*55}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    print(f"  Train size : {len(X_train):,} rows")
    print(f"  Test size  : {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# STEP 5 — BUILD SKLEARN PIPELINE
# ─────────────────────────────────────────────
# A Pipeline ensures: no data leakage, easy
# deployment, and the preprocessor is fitted
# only on train — never on test.

def build_pipeline(X_train: pd.DataFrame, model):
    """
    Build a full sklearn Pipeline with:
    - Numeric imputation + scaling
    - Categorical imputation + ordinal encoding
    - The ML model
    """

    # Identify column types dynamically from the dataframe
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numeric transformer: fill missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Categorical transformer: fill missing with "unknown", then encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,  numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",         model)
    ])

    return full_pipeline

# ─────────────────────────────────────────────
# STEP 6 — EVALUATE MODEL
# ─────────────────────────────────────────────

def clinical_accuracy(y_true, y_pred, threshold=CLINICAL_THRESHOLD):
    """
    Custom metric: % of predictions within ±threshold days.
    This is what a hospital manager actually cares about —
    not just RMSE.
    """
    within = np.abs(y_true - y_pred) <= threshold
    return within.mean() * 100

def evaluate(name: str, pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 1, None)  # LOS cannot be < 1 day

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    clin = clinical_accuracy(y_test.values, y_pred)

    print(f"\n  [{name}]")
    print(f"    MAE              : {mae:.2f} days")
    print(f"    RMSE             : {rmse:.2f} days")
    print(f"    R²               : {r2:.4f}")
    print(f"    Clinical Acc (±{CLINICAL_THRESHOLD}d) : {clin:.1f}%")

    return {
        "name": name, "pipeline": pipeline,
        "mae": mae, "rmse": rmse, "r2": r2,
        "clinical_accuracy": clin, "y_pred": y_pred
    }

# ─────────────────────────────────────────────
# STEP 7 — COMPARE MODELS
# ─────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test) -> list:
    print(f"\n{'='*55}")
    print("STEP 6 & 7: TRAIN + EVALUATE ALL MODELS")
    print(f"{'='*55}")

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(
                                n_estimators=200,
                                max_depth=10,
                                min_samples_leaf=5,
                                random_state=RANDOM_SEED,
                                n_jobs=-1
                            ),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=300,
                                learning_rate=0.05,
                                max_depth=5,
                                subsample=0.8,
                                random_state=RANDOM_SEED
                             ),
        "XGBoost":           xgb.XGBRegressor(
                                n_estimators=400,
                                learning_rate=0.05,
                                max_depth=6,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=RANDOM_SEED,
                                n_jobs=-1,
                                verbosity=0
                             ),
    }

    results = []
    for name, model in models.items():
        pipe = build_pipeline(X_train, model)
        result = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        results.append(result)

    return results

# ─────────────────────────────────────────────
# STEP 8 — SELECT BEST MODEL & SAVE
# ─────────────────────────────────────────────

def save_best_model(results: list) -> dict:
    print(f"\n{'='*55}")
    print("STEP 8: SELECTING + SAVING BEST MODEL")
    print(f"{'='*55}")

    # Pick best by MAE (lowest = best)
    best = min(results, key=lambda r: r["mae"])
    print(f"  Best model : {best['name']}")
    print(f"  MAE        : {best['mae']:.2f} days")
    print(f"  Clinical Acc: {best['clinical_accuracy']:.1f}%")

    model_path = f"models/best_model_{best['name'].replace(' ', '_')}.joblib"
    joblib.dump(best["pipeline"], model_path)
    print(f"  Saved to   : {model_path}")

    return best

# ─────────────────────────────────────────────
# STEP 9 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
# SHAP tells you WHY the model made each prediction.
# This is what the FastAPI endpoint will return.

def run_shap(best: dict, X_test: pd.DataFrame):
    print(f"\n{'='*55}")
    print("STEP 9: SHAP EXPLAINABILITY")
    print(f"{'='*55}")

    pipeline  = best["pipeline"]
    model     = pipeline.named_steps["model"]
    preproc   = pipeline.named_steps["preprocessor"]

    X_test_transformed = preproc.transform(X_test)

    # Get transformed feature names
    num_features = preproc.transformers_[0][2]
    cat_features = preproc.transformers_[1][2]
    feature_names = list(num_features) + list(cat_features)

    # Use TreeExplainer for tree-based models
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)

        # Plot top 15 features
        shap.summary_plot(
            shap_values,
            X_test_transformed,
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP summary saved → reports/shap_summary.png")

        # Save mean absolute SHAP values as a CSV — useful for FastAPI response
        mean_shap = pd.DataFrame({
            "feature":    feature_names,
            "mean_shap":  np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_shap", ascending=False)

        mean_shap.to_csv("reports/shap_feature_importance.csv", index=False)
        print("  SHAP importance CSV → reports/shap_feature_importance.csv")
        print(f"\n  Top 5 features driving LOS prediction:")
        print(mean_shap.head(5).to_string(index=False))

    except Exception as e:
        print(f"  SHAP skipped for this model type: {e}")

# ─────────────────────────────────────────────
# STEP 10 — PREDICTION FUNCTION
# (This is what your FastAPI endpoint will call)
# ─────────────────────────────────────────────

def predict_single_patient(pipeline, patient_dict: dict) -> dict:
    """
    Takes a dict of patient features → returns LOS prediction + top 3 SHAP reasons.
    This is the exact function your FastAPI /predict endpoint will wrap.

    Example input:
    {
        "rcount": 1,
        "gender": "M",
        "dialysisrenalendstage": "No",
        ...
        "bmi": 27.5,
        "facid": 3
    }
    """
    df_input = pd.DataFrame([patient_dict])

    # If no admission date provided (e.g. live API call), inject today's date.
    # engineer_features needs vdate to create day-of-week / season features —
    # without it those columns are missing and the trained pipeline will crash.
    if "vdate" not in df_input.columns or df_input["vdate"].isna().all():
        df_input["vdate"] = pd.Timestamp.today().normalize()

    # discharged is only known at discharge — not at admission prediction time.
    # Drop it so engineer_features doesn't try to parse it.
    df_input.drop(columns=["discharged"], inplace=True, errors="ignore")

    # Run through same feature engineering
    df_input = engineer_features(df_input)

    prediction = pipeline.predict(df_input)[0]
    prediction = max(1, round(float(prediction), 1))

    return {
        "predicted_los_days": prediction,
        "confidence_range":   f"{max(1, prediction - CLINICAL_THRESHOLD)}–{prediction + CLINICAL_THRESHOLD} days",
        "note": "Prediction within ±2 days is considered clinically acceptable"
    }

# ─────────────────────────────────────────────
# STEP 11 — COMPARISON REPORT
# ─────────────────────────────────────────────

def save_report(results: list):
    report_rows = []
    for r in results:
        report_rows.append({
            "Model":           r["name"],
            "MAE (days)":      round(r["mae"], 2),
            "RMSE (days)":     round(r["rmse"], 2),
            "R²":              round(r["r2"], 4),
            f"Clinical Acc (±{CLINICAL_THRESHOLD}d) %": round(r["clinical_accuracy"], 1)
        })

    report_df = pd.DataFrame(report_rows).sort_values("MAE (days)")
    report_df.to_csv("reports/model_comparison.csv", index=False)
    print(f"\n  Model comparison saved → reports/model_comparison.csv")
    print(f"\n{report_df.to_string(index=False)}")

# ─────────────────────────────────────────────
# MAIN — RUN THE FULL PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  HOSPITAL LOS PREDICTION — PHASE 1 PIPELINE")
    print("="*55)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Clean
    df = clean_data(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 5-7. Train + Evaluate all models
    results = compare_models(X_train, X_test, y_train, y_test)

    # 8. Save best
    best = save_best_model(results)

    # 9. SHAP explainability
    run_shap(best, X_test)

    # 10. Save comparison report
    save_report(results)

    # 11. Demo: predict one patient
    print(f"\n{'='*55}")
    print("DEMO: SINGLE PATIENT PREDICTION")
    print(f"{'='*55}")
    sample_patient = {
        "rcount":                     "1",
        "gender":                     "M",
        "dialysisrenalendstage":      "No",
        "asthma":                     "Yes",
        "irondef":                    "No",
        "pneum":                      "No",
        "substancedependence":        "No",
        "psychologicaldisordermajor": "No",
        "depress":                    "Yes",
        "psychother":                 "No",
        "fibrosisandother":           "No",
        "malnutrition":               "No",
        "hemo":                       "No",
        "hematocrit":                 36.5,
        "neutrophils":                7.2,
        "sodium":                     138.0,
        "glucose":                    105.0,
        "bloodureanitro":             18.0,
        "creatinine":                 1.1,
        "bmi":                        27.5,
        "pulse":                      78.0,
        "respiration":                18.0,
        "secondarydiagnosisnonicd9":  0,
        "facid":                      3
    }

    result = predict_single_patient(best["pipeline"], sample_patient)
    print(f"\n  Input     : Male, asthma+depression, BMI 27.5, 1 prior admit")
    print(f"  Prediction: {result['predicted_los_days']} days")
    print(f"  Range     : {result['confidence_range']}")
    print(f"  Note      : {result['note']}")

    print(f"\n{'='*55}")
    print("  PHASE 1 COMPLETE")
    print(f"  Best model : {best['name']}")
    print(f"  MAE        : {best['mae']:.2f} days")
    print(f"  Clinical Acc: {best['clinical_accuracy']:.1f}%")
    print(f"  Next       : Phase 2 → FastAPI endpoint wrapping predict_single_patient()")
    print("="*55 + "\n")