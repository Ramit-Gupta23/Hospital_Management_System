# ============================================================
#  test_api.py — Manual API Tests
#  Run AFTER starting the server with: uvicorn app:app --reload
#  Then in a new terminal: python test_api.py
# ============================================================

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ── TEST 1: Health Check ──────────────────────────────────
print_section("TEST 1: HEALTH CHECK")
r = requests.get(f"{BASE_URL}/health")
print(json.dumps(r.json(), indent=2))

# ── TEST 2: Single Prediction ─────────────────────────────
print_section("TEST 2: SINGLE PATIENT PREDICTION")
patient = {
    "gender":                     "M",
    "bmi":                        27.5,
    "rcount":                     1,
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
    "pulse":                      78.0,
    "respiration":                18.0,
    "secondarydiagnosisnonicd9":  0,
    "facid":                      3
}
r = requests.post(f"{BASE_URL}/predict", json=patient)
print(json.dumps(r.json(), indent=2))

# ── TEST 3: High-Risk Patient ─────────────────────────────
print_section("TEST 3: HIGH RISK PATIENT")
high_risk = {
    **patient,
    "rcount":                     5,
    "dialysisrenalendstage":      "Yes",
    "pneum":                      "Yes",
    "malnutrition":               "Yes",
    "bmi":                        16.0,
    "creatinine":                 4.5,
    "glucose":                    220.0,
    "sodium":                     125.0,
}
r = requests.post(f"{BASE_URL}/predict", json=high_risk)
resp = r.json()
print(f"  Predicted LOS : {resp['predicted_los_days']} days")
print(f"  Range         : {resp['confidence_range']}")
print(f"  Top Reasons   :")
for reason in resp.get("top_reasons", []):
    print(f"    - {reason['feature']}: {reason['impact']} ({reason['shap_value']})")

# ── TEST 4: Batch Prediction ──────────────────────────────
print_section("TEST 4: BATCH PREDICTION (3 patients)")
batch = [patient, high_risk, {**patient, "gender": "F", "bmi": 22.0, "rcount": 0}]
r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
resp = r.json()
print(f"  Total predictions: {resp['total']}")
for i, pred in enumerate(resp["predictions"]):
    print(f"  Patient {i+1}: {pred['predicted_los_days']} days ({pred['confidence_range']})")

# ── TEST 5: Recent Logs ───────────────────────────────────
print_section("TEST 5: RECENT PREDICTION LOGS")
r = requests.get(f"{BASE_URL}/logs/recent?n=3")
print(json.dumps(r.json(), indent=2))

# ── TEST 6: Validation Error ──────────────────────────────
print_section("TEST 6: VALIDATION ERROR (bad gender)")
bad_patient = {**patient, "gender": "X"}
r = requests.post(f"{BASE_URL}/predict", json=bad_patient)
print(f"  Status: {r.status_code}")
print(f"  Error : {r.json()['detail'][0]['msg']}")

print("\n✅ All tests done. Check http://localhost:8000/docs for Swagger UI.")
