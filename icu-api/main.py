from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from uuid import uuid4

app = FastAPI(title="ICU Guard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VITAL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "vital"))

MODEL_PATH = os.path.join(BASE_DIR, "model", "xgb_deterioration_model.joblib")
DATA_PATH = os.path.join(VITAL_DIR, "hospital_deterioration_hourly_panel.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.json")
ALERTS_PATH = os.path.join(BASE_DIR, "alerts.json")

GRU_PRED_PATH = os.path.join(VITAL_DIR, "gru_hourly_predictions_from_npy.csv")
XGB_PRED_PATH = os.path.join(VITAL_DIR, "xgb_hourly_predictions_from_npy.csv")

HIGH_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.20
FUSION_ALPHA = 0.60

FEATURE_COLS = [
    "hour_from_admission",
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_device",
    "oxygen_flow",
    "mobility_score",
    "nurse_alert",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "sepsis_risk_score",
    "age",
    "gender",
    "comorbidity_index",
    "admission_type",
    "baseline_risk_score",
    "los_hours",
]

xgb_model = joblib.load(MODEL_PATH)
fusion_cache = None


class PredictionInput(BaseModel):
    patient_id: str = "ICU-2408"
    hour_from_admission: float = 24

    heart_rate: float
    respiratory_rate: float
    spo2_pct: float
    temperature_c: float
    systolic_bp: float
    diastolic_bp: float

    oxygen_device: str = "none"
    oxygen_flow: float = 0.0
    mobility_score: float = 2.0
    nurse_alert: float = 0.0
    wbc_count: float = 8.0
    lactate: float
    creatinine: float
    crp_level: float = 10.0
    hemoglobin: float = 12.0
    sepsis_risk_score: float = 0.2
    age: float
    gender: str = "F"
    comorbidity_index: float = 1.0
    admission_type: str = "Emergency"
    baseline_risk_score: float = 0.3
    los_hours: float = 24.0


class AlertAcknowledgeInput(BaseModel):
    reviewed_by: str = "Dr. Meriem"


def load_json_file(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_json_file(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_history():
    return load_json_file(HISTORY_PATH)


def save_history(records):
    save_json_file(HISTORY_PATH, records)


def load_alerts():
    return load_json_file(ALERTS_PATH)


def save_alerts(records):
    save_json_file(ALERTS_PATH, records)


def categorize_risk(score: float):
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def build_clinical_message(risk_category: str, risk_score: float):
    if risk_category == "High":
        return {
            "headline": "Immediate clinical review recommended",
            "summary": f"Predicted deterioration risk is high ({risk_score:.2f}). Escalate bedside assessment and review within the next 12 hours.",
            "actions": [
                "Review airway, breathing, and circulation immediately.",
                "Repeat full vital signs and confirm SpO2 trend.",
                "Review lactate, creatinine, and blood pressure urgently.",
                "Notify the responsible ICU physician if clinically indicated.",
            ],
        }
    if risk_category == "Medium":
        return {
            "headline": "Close monitoring recommended",
            "summary": f"Predicted deterioration risk is moderate ({risk_score:.2f}). Increase surveillance and reassess clinical status.",
            "actions": [
                "Repeat vitals and reassess respiratory status.",
                "Review fluid status, perfusion, and recent labs.",
                "Monitor for progression over the next few hours.",
                "Escalate if BP, SpO2, RR, lactate, or mental status worsens.",
            ],
        }
    return {
        "headline": "Low immediate risk",
        "summary": f"Predicted deterioration risk is currently low ({risk_score:.2f}). Continue standard monitoring and reassess if the condition changes.",
        "actions": [
            "Continue routine observation.",
            "Repeat assessment if symptoms or vitals worsen.",
            "Document current status and trend.",
        ],
    }


def normalize_patient_id(value):
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(value).strip()


def normalize_hour(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def load_fusion_dataset():
    global fusion_cache
    if fusion_cache is not None:
        return fusion_cache.copy()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not os.path.exists(XGB_PRED_PATH):
        raise FileNotFoundError(f"XGB prediction file not found: {XGB_PRED_PATH}")
    if not os.path.exists(GRU_PRED_PATH):
        raise FileNotFoundError(f"GRU prediction file not found: {GRU_PRED_PATH}")

    base_df = pd.read_csv(DATA_PATH).copy()
    xgb = pd.read_csv(XGB_PRED_PATH).copy()
    gru = pd.read_csv(GRU_PRED_PATH).copy()

    xgb = xgb.rename(columns={"risk_score": "xgb_score"})
    gru = gru.rename(columns={"risk_score": "gru_score"})

    xgb = xgb[["patient_id", "hour_from_admission", "true_label", "xgb_score"]].copy()
    gru = gru[["patient_id", "hour_from_admission", "true_label", "gru_score"]].copy()

    xgb["patient_id_key"] = xgb["patient_id"].map(normalize_patient_id)
    xgb["hour_key"] = xgb["hour_from_admission"].map(normalize_hour)

    gru["patient_id_key"] = gru["patient_id"].map(normalize_patient_id)
    gru["hour_key"] = gru["hour_from_admission"].map(normalize_hour)

    base_df["patient_id_key"] = base_df["patient_id"].map(normalize_patient_id)
    base_df["hour_key"] = base_df["hour_from_admission"].map(normalize_hour)

    pred = pd.merge(
        xgb,
        gru[["patient_id_key", "hour_key", "gru_score"]],
        on=["patient_id_key", "hour_key"],
        how="inner",
    )

    pred["fusion_score"] = FUSION_ALPHA * pred["xgb_score"] + (1.0 - FUSION_ALPHA) * pred["gru_score"]
    pred["risk_score"] = pred["fusion_score"]
    pred["risk_category"] = pred["risk_score"].apply(lambda x: categorize_risk(float(x)))
    pred["model_mode"] = "Fusion"
    pred["model_used"] = f"Fusion (XGB {FUSION_ALPHA:.2f} + GRU {1.0 - FUSION_ALPHA:.2f})"

    merged = pd.merge(
        base_df,
        pred[
            [
                "patient_id_key",
                "hour_key",
                "true_label",
                "xgb_score",
                "gru_score",
                "fusion_score",
                "risk_score",
                "risk_category",
                "model_mode",
                "model_used",
            ]
        ],
        on=["patient_id_key", "hour_key"],
        how="inner",
    )

    if "deterioration_next_12h" in merged.columns:
        merged["true_label"] = merged["true_label"].where(
            merged["true_label"].notna(),
            merged["deterioration_next_12h"],
        )

    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    merged["risk_rank"] = merged["risk_category"].map(risk_order).fillna(3)

    merged = merged.sort_values(
        ["risk_rank", "risk_score", "true_label", "hour_from_admission"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    fusion_cache = merged.copy()
    return fusion_cache.copy()


def find_exact_case(patient_id: str, hour_from_admission: float):
    df = load_fusion_dataset()
    pid = str(patient_id)
    hr = float(hour_from_admission)

    matched = df[
        (df["patient_id"].astype(str) == pid) &
        (df["hour_from_admission"].astype(float) == hr)
    ]

    if matched.empty:
        return None

    return matched.iloc[0]


def build_result_and_persist(
    patient_id: str,
    hour_from_admission: float,
    risk_score: float,
    xgb_score=None,
    gru_score=None,
    fusion_score=None,
    model_used="xgb_deterioration_model.joblib",
    model_mode="XGBoost only",
    top_factors=None,
    source="manual_form",
    note=None,
):
    risk_category = categorize_risk(risk_score)
    prediction = 1 if risk_category in ["High", "Medium"] else 0
    created_at = datetime.now().isoformat(timespec="seconds")
    clinical_message = build_clinical_message(risk_category, float(risk_score))

    result = {
        "patient_id": str(patient_id),
        "prediction": prediction,
        "risk_category": risk_category,
        "risk_score": round(float(risk_score), 4),
        "confidence": round(float(risk_score), 4),
        "xgb_score": round(float(xgb_score), 4) if xgb_score is not None and pd.notna(xgb_score) else None,
        "gru_score": round(float(gru_score), 4) if gru_score is not None and pd.notna(gru_score) else None,
        "fusion_score": round(float(fusion_score), 4) if fusion_score is not None and pd.notna(fusion_score) else None,
        "top_factors": top_factors or [
            "Respiratory Rate",
            "Blood Pressure",
            "Lactate",
            "SpO2",
            "Creatinine",
        ],
        "model_used": model_used,
        "model_mode": model_mode,
        "source": source,
        "created_at": created_at,
        "hour_from_admission": float(hour_from_admission),
        "clinical_message": clinical_message,
        "note": note,
        "thresholds": {
            "high": HIGH_THRESHOLD,
            "medium": MEDIUM_THRESHOLD,
            "fusion_alpha_xgb": FUSION_ALPHA,
            "fusion_alpha_gru": round(1.0 - FUSION_ALPHA, 2),
        },
    }

    history = load_history()
    history.insert(0, result)
    history = history[:100]
    save_history(history)

    if risk_category in ["High", "Medium"]:
        alerts = load_alerts()
        new_alert = {
            "alert_id": str(uuid4()),
            "patient_id": str(patient_id),
            "risk_category": risk_category,
            "risk_score": round(float(risk_score), 4),
            "confidence": round(float(risk_score), 4),
            "xgb_score": round(float(xgb_score), 4) if xgb_score is not None and pd.notna(xgb_score) else None,
            "gru_score": round(float(gru_score), 4) if gru_score is not None and pd.notna(gru_score) else None,
            "fusion_score": round(float(fusion_score), 4) if fusion_score is not None and pd.notna(fusion_score) else None,
            "status": "Immediate review required" if risk_category == "High" else "Clinical review recommended",
            "created_at": created_at,
            "acknowledged": False,
            "acknowledged_at": None,
            "reviewed_by": None,
            "model_used": model_used,
            "model_mode": model_mode,
            "clinical_message": clinical_message,
            "note": note,
        }
        alerts.insert(0, new_alert)
        alerts = alerts[:100]
        save_alerts(alerts)

        result["alert_created"] = True
        result["alert_status"] = new_alert["status"]
    else:
        result["alert_created"] = False
        result["alert_status"] = None

    return result


@app.get("/")
def root():
    return {"message": "ICU Guard AI API is running"}


@app.get("/model-status")
def model_status():
    df = load_fusion_dataset()
    return {
        "xgb_loaded": True,
        "fusion_enabled": True,
        "fusion_rows": len(df),
        "fusion_alpha_xgb": FUSION_ALPHA,
        "fusion_alpha_gru": round(1.0 - FUSION_ALPHA, 2),
        "data_path": DATA_PATH,
        "xgb_predictions_path": XGB_PRED_PATH,
        "gru_predictions_path": GRU_PRED_PATH,
    }


@app.get("/history")
def get_history():
    records = load_history()
    return {"count": len(records), "items": records}


@app.get("/alerts")
def get_alerts():
    alerts = load_alerts()
    active_alerts = [a for a in alerts if not a.get("acknowledged", False)]
    return {"count": len(active_alerts), "items": active_alerts}


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, payload: AlertAcknowledgeInput):
    alerts = load_alerts()
    found = False

    for alert in alerts:
        if alert.get("alert_id") == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = datetime.now().isoformat(timespec="seconds")
            alert["reviewed_by"] = payload.reviewed_by
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Alert not found")

    save_alerts(alerts)
    return {"message": "Alert acknowledged", "alert_id": alert_id}


@app.delete("/alerts")
def clear_alerts():
    save_alerts([])
    return {"message": "All alerts cleared"}


@app.get("/demo-cases")
def get_demo_cases(
    limit: int = Query(default=50, ge=1, le=200),
    risk: str = Query(default="all")
):
    df = load_fusion_dataset()

    risk_normalized = risk.lower()
    if risk_normalized in ["high", "medium", "low"]:
        df = df[df["risk_category"].str.lower() == risk_normalized]

    items = []
    for _, row in df.head(limit).iterrows():
        items.append({
            "patient_id": str(row["patient_id"]),
            "hour_from_admission": float(row["hour_from_admission"]),
            "age": float(row["age"]),
            "heart_rate": float(row["heart_rate"]),
            "respiratory_rate": float(row["respiratory_rate"]),
            "spo2_pct": float(row["spo2_pct"]),
            "temperature_c": float(row["temperature_c"]),
            "systolic_bp": float(row["systolic_bp"]),
            "diastolic_bp": float(row["diastolic_bp"]),
            "oxygen_device": str(row["oxygen_device"]),
            "oxygen_flow": float(row["oxygen_flow"]),
            "mobility_score": float(row["mobility_score"]),
            "nurse_alert": float(row["nurse_alert"]),
            "wbc_count": float(row["wbc_count"]),
            "lactate": float(row["lactate"]),
            "creatinine": float(row["creatinine"]),
            "crp_level": float(row["crp_level"]),
            "hemoglobin": float(row["hemoglobin"]),
            "sepsis_risk_score": float(row["sepsis_risk_score"]),
            "gender": str(row["gender"]),
            "comorbidity_index": float(row["comorbidity_index"]),
            "admission_type": str(row["admission_type"]),
            "baseline_risk_score": float(row["baseline_risk_score"]),
            "los_hours": float(row["los_hours"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "xgb_score": round(float(row["xgb_score"]), 4),
            "gru_score": round(float(row["gru_score"]), 4),
            "fusion_score": round(float(row["fusion_score"]), 4),
            "risk_category": str(row["risk_category"]),
            "true_label": int(row["true_label"]),
            "model_mode": str(row["model_mode"]),
            "model_used": str(row["model_used"]),
        })

    return {"count": len(items), "items": items}


@app.get("/predict-demo")
def predict_demo_case(
    patient_id: str = Query(...),
    hour_from_admission: float = Query(...)
):
    row = find_exact_case(patient_id, hour_from_admission)

    if row is None:
        raise HTTPException(status_code=404, detail="Demo case not found")

    result = build_result_and_persist(
        patient_id=str(row["patient_id"]),
        hour_from_admission=float(row["hour_from_admission"]),
        risk_score=float(row["risk_score"]),
        xgb_score=float(row["xgb_score"]),
        gru_score=float(row["gru_score"]),
        fusion_score=float(row["fusion_score"]),
        model_used=str(row["model_used"]),
        model_mode=str(row["model_mode"]),
        source="training_data_demo_exact_24h",
        note="Exact 24h sequence from dataset was used.",
    )

    result["input_snapshot"] = {
        "age": float(row["age"]),
        "heart_rate": float(row["heart_rate"]),
        "respiratory_rate": float(row["respiratory_rate"]),
        "spo2_pct": float(row["spo2_pct"]),
        "temperature_c": float(row["temperature_c"]),
        "systolic_bp": float(row["systolic_bp"]),
        "diastolic_bp": float(row["diastolic_bp"]),
        "oxygen_device": str(row["oxygen_device"]),
        "oxygen_flow": float(row["oxygen_flow"]),
        "mobility_score": float(row["mobility_score"]),
        "nurse_alert": float(row["nurse_alert"]),
        "wbc_count": float(row["wbc_count"]),
        "lactate": float(row["lactate"]),
        "creatinine": float(row["creatinine"]),
        "crp_level": float(row["crp_level"]),
        "hemoglobin": float(row["hemoglobin"]),
        "sepsis_risk_score": float(row["sepsis_risk_score"]),
        "gender": str(row["gender"]),
        "comorbidity_index": float(row["comorbidity_index"]),
        "admission_type": str(row["admission_type"]),
        "baseline_risk_score": float(row["baseline_risk_score"]),
        "los_hours": float(row["los_hours"]),
    }
    result["true_label"] = int(row["true_label"])

    return result


@app.post("/predict")
def predict(data: PredictionInput):
    # 1) Try exact 24h sequence first
    exact_row = find_exact_case(data.patient_id, data.hour_from_admission)

    if exact_row is not None:
        result = build_result_and_persist(
            patient_id=str(exact_row["patient_id"]),
            hour_from_admission=float(exact_row["hour_from_admission"]),
            risk_score=float(exact_row["risk_score"]),
            xgb_score=float(exact_row["xgb_score"]),
            gru_score=float(exact_row["gru_score"]),
            fusion_score=float(exact_row["fusion_score"]),
            model_used=str(exact_row["model_used"]),
            model_mode="Fusion (exact 24h sequence)",
            source="manual_form_exact_dataset_match",
            note="Exact 24h historical sequence was found in dataset and used for GRU + Fusion.",
        )

        result["input_snapshot"] = {
            "age": float(exact_row["age"]),
            "heart_rate": float(exact_row["heart_rate"]),
            "respiratory_rate": float(exact_row["respiratory_rate"]),
            "spo2_pct": float(exact_row["spo2_pct"]),
            "temperature_c": float(exact_row["temperature_c"]),
            "systolic_bp": float(exact_row["systolic_bp"]),
            "diastolic_bp": float(exact_row["diastolic_bp"]),
            "oxygen_device": str(exact_row["oxygen_device"]),
            "oxygen_flow": float(exact_row["oxygen_flow"]),
            "mobility_score": float(exact_row["mobility_score"]),
            "nurse_alert": float(exact_row["nurse_alert"]),
            "wbc_count": float(exact_row["wbc_count"]),
            "lactate": float(exact_row["lactate"]),
            "creatinine": float(exact_row["creatinine"]),
            "crp_level": float(exact_row["crp_level"]),
            "hemoglobin": float(exact_row["hemoglobin"]),
            "sepsis_risk_score": float(exact_row["sepsis_risk_score"]),
            "gender": str(exact_row["gender"]),
            "comorbidity_index": float(exact_row["comorbidity_index"]),
            "admission_type": str(exact_row["admission_type"]),
            "baseline_risk_score": float(exact_row["baseline_risk_score"]),
            "los_hours": float(exact_row["los_hours"]),
        }
        result["true_label"] = int(exact_row["true_label"])
        return result

    # 2) Otherwise fallback to XGB only on current snapshot
    row = {
        "hour_from_admission": data.hour_from_admission,
        "heart_rate": data.heart_rate,
        "respiratory_rate": data.respiratory_rate,
        "spo2_pct": data.spo2_pct,
        "temperature_c": data.temperature_c,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "oxygen_device": data.oxygen_device,
        "oxygen_flow": data.oxygen_flow,
        "mobility_score": data.mobility_score,
        "nurse_alert": data.nurse_alert,
        "wbc_count": data.wbc_count,
        "lactate": data.lactate,
        "creatinine": data.creatinine,
        "crp_level": data.crp_level,
        "hemoglobin": data.hemoglobin,
        "sepsis_risk_score": data.sepsis_risk_score,
        "age": data.age,
        "gender": data.gender,
        "comorbidity_index": data.comorbidity_index,
        "admission_type": data.admission_type,
        "baseline_risk_score": data.baseline_risk_score,
        "los_hours": data.los_hours,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS)
    xgb_score = float(xgb_model.predict_proba(df)[0][1])

    return build_result_and_persist(
        patient_id=data.patient_id,
        hour_from_admission=data.hour_from_admission,
        risk_score=xgb_score,
        xgb_score=xgb_score,
        gru_score=None,
        fusion_score=None,
        model_used="xgb_deterioration_model.joblib",
        model_mode="XGBoost only (no exact 24h sequence found)",
        source="manual_form_snapshot_only",
        note="Exact 24h sequence was not available for this patient/hour, so GRU/Fusion were not used.",
    )