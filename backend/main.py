from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ICU Early Warning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
VITAL_DIR = PROJECT_DIR / "vital"

PANEL_DATA_PATH = VITAL_DIR / "hospital_deterioration_hourly_panel.csv"

XGB_MODEL_PATH = VITAL_DIR / "xgb_deterioration_model.joblib"
XGB_THRESHOLD_PATH = VITAL_DIR / "xgb_best_threshold.txt"

GRU_MODEL_PATH = VITAL_DIR / "gru_24h_model.pt"
X_SEQ_PATH = VITAL_DIR / "X_seq_24h.npy"
Y_SEQ_PATH = VITAL_DIR / "y_seq_24h.npy"
PATIENT_SEQ_PATH = VITAL_DIR / "patient_seq_24h.npy"
HOUR_SEQ_PATH = VITAL_DIR / "hour_seq_24h.npy"

panel_df = None
sort_hour_col = None

xgb_pipeline = None
xgb_threshold = 0.5
xgb_expected_columns = []

gru_model = None
gru_threshold = 0.94
gru_input_dim = None
device = "cuda" if torch.cuda.is_available() else "cpu"

X_seq = None
y_seq = None
patient_seq = None
hour_seq = None

FUSION_XGB_WEIGHT = 0.4
FUSION_GRU_WEIGHT = 0.6

if PANEL_DATA_PATH.exists():
    panel_df = pd.read_csv(PANEL_DATA_PATH)

    for candidate in [
        "hour_from_admission",
        "hours_from_admission",
        "icu_hour",
        "chart_hour",
        "time_step",
        "hour",
    ]:
        if candidate in panel_df.columns:
            sort_hour_col = candidate
            break

if XGB_MODEL_PATH.exists():
    xgb_pipeline = joblib.load(XGB_MODEL_PATH)
    if hasattr(xgb_pipeline, "feature_names_in_"):
        xgb_expected_columns = list(xgb_pipeline.feature_names_in_)

if XGB_THRESHOLD_PATH.exists():
    try:
        xgb_threshold = float(XGB_THRESHOLD_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        xgb_threshold = 0.5


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


if X_SEQ_PATH.exists():
    X_seq = np.load(X_SEQ_PATH, mmap_mode="r")

if Y_SEQ_PATH.exists():
    y_seq = np.load(Y_SEQ_PATH, mmap_mode="r")

if PATIENT_SEQ_PATH.exists():
    patient_seq = np.load(PATIENT_SEQ_PATH, mmap_mode="r")

if HOUR_SEQ_PATH.exists():
    hour_seq = np.load(HOUR_SEQ_PATH, mmap_mode="r")

if X_seq is not None:
    gru_input_dim = int(X_seq.shape[2])

if GRU_MODEL_PATH.exists() and gru_input_dim is not None:
    ckpt = torch.load(GRU_MODEL_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    gru_model = GRUModel(input_dim=gru_input_dim).to(device)
    gru_model.load_state_dict(state, strict=False)
    gru_model.eval()


class PatientData(BaseModel):
    patient_id: int


def find_latest_patient_row(patient_id_value: int) -> pd.Series:
    if panel_df is None:
        raise RuntimeError("Panel dataset not loaded")

    if "patient_id" not in panel_df.columns:
        raise RuntimeError("patient_id column not found in panel dataset")

    patient_rows = panel_df[panel_df["patient_id"] == patient_id_value].copy()

    if patient_rows.empty:
        raise ValueError(f"Patient {patient_id_value} not found in panel dataset")

    if sort_hour_col is not None:
        patient_rows = patient_rows.sort_values(sort_hour_col)
    else:
        patient_rows = patient_rows.reset_index(drop=True)

    return patient_rows.iloc[-1]


def build_xgb_input_from_latest_row(latest_row: pd.Series) -> pd.DataFrame:
    if not xgb_expected_columns:
        raise RuntimeError("XGBoost expected columns are unavailable")

    row = {col: None for col in xgb_expected_columns}

    for col in xgb_expected_columns:
        if col in latest_row.index:
            value = latest_row[col]
            if pd.isna(value):
                value = None
            row[col] = value

    return pd.DataFrame([row], columns=xgb_expected_columns)


def predict_xgb_from_patient(patient_id_value: int):
    if xgb_pipeline is None:
        raise RuntimeError("XGBoost model not loaded")

    latest_row = find_latest_patient_row(patient_id_value)
    x_input = build_xgb_input_from_latest_row(latest_row)
    prob = float(xgb_pipeline.predict_proba(x_input)[0, 1])
    return prob, latest_row


def get_latest_gru_sequence_for_patient(patient_id_value: int):
    if X_seq is None or patient_seq is None or hour_seq is None:
        raise RuntimeError("GRU sequence files are not loaded")

    patient_indices = np.where(patient_seq[:] == patient_id_value)[0]

    if len(patient_indices) == 0:
        raise ValueError(f"Patient {patient_id_value} not found in GRU sequence files")

    patient_hours = hour_seq[patient_indices]
    latest_idx = patient_indices[int(np.argmax(patient_hours))]

    seq = np.asarray(X_seq[latest_idx], dtype=np.float32)
    latest_hour = int(hour_seq[latest_idx])

    true_label = None
    if y_seq is not None:
        true_label = int(y_seq[latest_idx])

    return seq, latest_hour, true_label


def predict_gru_score(patient_id_value: int):
    if gru_model is None:
        raise RuntimeError("GRU model not loaded")

    seq, latest_hour, true_label = get_latest_gru_sequence_for_patient(patient_id_value)

    xb = torch.from_numpy(seq).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logit = gru_model(xb)
        prob = torch.sigmoid(logit).cpu().item()

    return float(prob), latest_hour, true_label


def risk_level_from_score(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    elif score >= 0.5 * threshold:
        return "Moderate"
    return "Low"


def build_alert_banner(level: str) -> str:
    if level.lower() == "high":
        return "ALERT: HIGH RISK DETERIORATION DETECTED"
    if level.lower() == "moderate":
        return "WARNING: MODERATE RISK — CLOSE MONITORING REQUIRED"
    return "STATUS: LOW RISK — PATIENT CURRENTLY STABLE"


def build_recommendation(level: str) -> str:
    if level.lower() == "high":
        return "Immediate clinical review recommended."
    if level.lower() == "moderate":
        return "Close monitoring and reassessment recommended."
    return "Continue routine monitoring."


def extract_latest_vitals(latest_row: pd.Series) -> dict:
    aliases = {
        "heart_rate": ["heart_rate", "HeartRate", "HR", "hr", "pulse", "heart_rate_mean"],
        "spo2": ["spo2", "SpO2", "SaO2", "spo2_mean", "oxygen_saturation"],
        "respiratory_rate": ["respiratory_rate", "RR", "rr", "RespRate", "resp_rate"],
        "systolic_bp": ["systolic_bp", "SBP", "sbp", "SysBP", "systolic_blood_pressure"],
        "lactate": ["lactate", "Lactate", "lactate_level"],
        "oxygen_device": ["oxygen_device"],
        "gender": ["gender", "sex"],
        "admission_type": ["admission_type"],
    }

    vitals = {}
    for logical_name, possible_cols in aliases.items():
        vitals[logical_name] = None
        for col in possible_cols:
            if col in latest_row.index:
                value = latest_row[col]
                if pd.isna(value):
                    value = None
                vitals[logical_name] = value
                break

    vitals["latest_hour_from_panel"] = None
    if sort_hour_col is not None and sort_hour_col in latest_row.index:
        value = latest_row[sort_hour_col]
        if not pd.isna(value):
            try:
                vitals["latest_hour_from_panel"] = int(value)
            except Exception:
                vitals["latest_hour_from_panel"] = None

    return vitals


def predict_patient_summary(patient_id_value: int) -> dict:
    xgb_prob, latest_row = predict_xgb_from_patient(patient_id_value)
    gru_prob, latest_hour, true_label = predict_gru_score(patient_id_value)

    fused_prob = (FUSION_XGB_WEIGHT * xgb_prob) + (FUSION_GRU_WEIGHT * gru_prob)
    fusion_threshold = (FUSION_XGB_WEIGHT * xgb_threshold) + (FUSION_GRU_WEIGHT * gru_threshold)
    level = risk_level_from_score(fused_prob, fusion_threshold)

    latest_vitals = extract_latest_vitals(latest_row)

    return {
        "patient_id": patient_id_value,
        "risk_score": round(float(fused_prob), 6),
        "risk_level": level,
        "lead_time_hours": 6.0,
        "threshold_used": round(float(fusion_threshold), 6),
        "latest_hour_from_admission": latest_hour,
        "true_label_if_available": true_label,
        "alert_banner": build_alert_banner(level),
        "recommendation": build_recommendation(level),
        "latest_vitals": latest_vitals,
    }


def get_candidate_patient_ids(limit: int = 12) -> list[int]:
    if panel_df is None or "patient_id" not in panel_df.columns:
        return []

    if sort_hour_col is not None:
        latest_per_patient = (
            panel_df[["patient_id", sort_hour_col]]
            .dropna(subset=["patient_id"])
            .groupby("patient_id", as_index=False)[sort_hour_col]
            .max()
            .sort_values(sort_hour_col, ascending=False)
        )
        return latest_per_patient["patient_id"].astype(int).head(max(limit * 3, 20)).tolist()

    return panel_df["patient_id"].dropna().astype(int).unique().tolist()[: max(limit * 3, 20)]


def get_patient_timeseries(patient_id_value: int, max_points: int = 24) -> dict:
    if panel_df is None:
        raise RuntimeError("Panel dataset not loaded")

    patient_rows = panel_df[panel_df["patient_id"] == patient_id_value].copy()

    if patient_rows.empty:
        raise ValueError(f"Patient {patient_id_value} not found in panel dataset")

    if sort_hour_col is not None:
        patient_rows = patient_rows.sort_values(sort_hour_col)

    patient_rows = patient_rows.tail(max_points)

    def pick_series(possible_cols: list[str]):
        for col in possible_cols:
            if col in patient_rows.columns:
                values = []
                for v in patient_rows[col].tolist():
                    if pd.isna(v):
                        values.append(None)
                    else:
                        try:
                            values.append(float(v))
                        except Exception:
                            values.append(None)
                return values
        return [None for _ in range(len(patient_rows))]

    if sort_hour_col is not None:
        hours = []
        for v in patient_rows[sort_hour_col].tolist():
            if pd.isna(v):
                hours.append(None)
            else:
                try:
                    hours.append(int(v))
                except Exception:
                    hours.append(None)
    else:
        hours = list(range(len(patient_rows)))

    return {
        "patient_id": patient_id_value,
        "hours": hours,
        "heart_rate": pick_series(["heart_rate", "HeartRate", "HR", "hr", "pulse", "heart_rate_mean"]),
        "respiratory_rate": pick_series(["respiratory_rate", "RR", "rr", "RespRate", "resp_rate"]),
        "systolic_bp": pick_series(["systolic_bp", "SBP", "sbp", "SysBP", "systolic_blood_pressure"]),
        "spo2": pick_series(["spo2", "SpO2", "SaO2", "spo2_mean", "oxygen_saturation"]),
        "lactate": pick_series(["lactate", "Lactate", "lactate_level"]),
    }


@app.get("/")
def root():
    return {
        "message": "FastAPI is running",
        "panel_data_loaded": panel_df is not None,
        "panel_data_path": str(PANEL_DATA_PATH),
        "xgb_model_loaded": xgb_pipeline is not None,
        "xgb_expected_columns_count": len(xgb_expected_columns),
        "xgb_threshold": xgb_threshold,
        "gru_model_loaded": gru_model is not None,
        "gru_threshold": gru_threshold,
        "gru_input_dim": gru_input_dim,
        "seq_files_loaded": X_seq is not None and patient_seq is not None and hour_seq is not None,
        "device": device,
        "fusion_xgb_weight": FUSION_XGB_WEIGHT,
        "fusion_gru_weight": FUSION_GRU_WEIGHT,
    }


@app.get("/model-info")
def model_info():
    return {
        "panel_data_loaded": panel_df is not None,
        "panel_rows": int(len(panel_df)) if panel_df is not None else 0,
        "xgb_model_loaded": xgb_pipeline is not None,
        "xgb_expected_columns": xgb_expected_columns,
        "xgb_threshold": xgb_threshold,
        "gru_model_loaded": gru_model is not None,
        "gru_threshold": gru_threshold,
        "gru_input_dim": gru_input_dim,
        "x_seq_shape": list(X_seq.shape) if X_seq is not None else None,
        "patient_seq_shape": list(patient_seq.shape) if patient_seq is not None else None,
        "hour_seq_shape": list(hour_seq.shape) if hour_seq is not None else None,
        "device": device,
        "fusion_xgb_weight": FUSION_XGB_WEIGHT,
        "fusion_gru_weight": FUSION_GRU_WEIGHT,
    }


@app.post("/predict")
def predict(data: PatientData):
    try:
        return predict_patient_summary(data.patient_id)
    except Exception as e:
        return {
            "risk_score": 0.0,
            "risk_level": "Error",
            "lead_time_hours": 0.0,
            "patient_id": data.patient_id,
            "message": str(e),
        }


@app.get("/dashboard/alerts")
def dashboard_alerts(limit: int = Query(default=8, ge=1, le=20)):
    candidates = get_candidate_patient_ids(limit=limit)

    patients = []
    for patient_id_value in candidates:
        try:
            patients.append(predict_patient_summary(int(patient_id_value)))
        except Exception:
            continue

    patients = sorted(patients, key=lambda x: x["risk_score"], reverse=True)[:limit]

    high_count = sum(1 for p in patients if p["risk_level"] == "High")
    moderate_count = sum(1 for p in patients if p["risk_level"] == "Moderate")
    low_count = sum(1 for p in patients if p["risk_level"] == "Low")

    return {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "total_patients": len(patients),
        "high_risk_count": high_count,
        "moderate_risk_count": moderate_count,
        "low_risk_count": low_count,
        "active_alerts": high_count,
        "patients": patients,
    }


@app.get("/patient/{patient_id}/timeseries")
def patient_timeseries(patient_id: int, max_points: int = Query(default=24, ge=8, le=72)):
    return get_patient_timeseries(patient_id, max_points=max_points)