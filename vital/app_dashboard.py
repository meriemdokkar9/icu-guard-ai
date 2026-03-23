import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED_PATH = HERE / "dashboard_predictions_gru.csv"
ALERTS_PATH = HERE / "dashboard_alerts_only.csv"
TIMELINES_DIR = HERE / "timelines_alert_patients"

# =========================
# CONFIG PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# إذا ملفك داخل vital/ (مثل ما هو عندك)
VITAL_DIR = os.path.join(BASE_DIR, "vital")

PRED_PATH = os.path.join(VITAL_DIR, "dashboard_predictions_gru.csv")
ALERTS_ONLY_PATH = os.path.join(VITAL_DIR, "dashboard_alerts_only.csv")
TIMELINES_DIR = os.path.join(VITAL_DIR, "timelines_alert_patients")
RAW_PANEL_PATH = os.path.join(VITAL_DIR, "hospital_deterioration_hourly_panel.csv")

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="ICU Deterioration Dashboard (GRU)", layout="wide")

# =========================
# HELPERS
# =========================
def load_predictions():
    if not os.path.exists(PRED_PATH):
        st.error(f"Missing file: {PRED_PATH}")
        st.stop()
    df = pd.read_csv(PRED_PATH)

    # safety: ensure columns exist
    needed = ["patient_id", "hour_from_admission", "true_label", "risk_score", "threshold", "alert"]
    for c in needed:
        if c not in df.columns:
            st.error(f"Column missing in predictions CSV: {c}")
            st.stop()

    # risk_level if not present
    if "risk_level" not in df.columns:
        df["risk_level"] = df["risk_score"].apply(risk_level_from_score)

    return df

def risk_level_from_score(s):
    # simple bins – تقدري تغيريهم
    if s >= 0.94:
        return "HIGH"
    elif s >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"

def color_risk(level):
    if level == "HIGH":
        return "background-color: #ffcccc; color: #7a0000;"
    if level == "MEDIUM":
        return "background-color: #fff2cc; color: #7a5a00;"
    return "background-color: #ccffcc; color: #006100;"

def load_patient_timeline_from_folder(patient_id):
    # نحاول نجيب تايملاين محفوظ
    f = os.path.join(TIMELINES_DIR, f"patient_{patient_id}_timeline.csv")
    if os.path.exists(f):
        return pd.read_csv(f)

    # لو ما لقي، نجرب أي ملف يشبهه
    pattern = os.path.join(TIMELINES_DIR, f"patient_{patient_id}_timeline*.csv")
    matches = glob.glob(pattern)
    if matches:
        return pd.read_csv(matches[0])

    return None

def build_timeline_from_predictions(df_pred, patient_id):
    p = df_pred[df_pred["patient_id"] == patient_id].copy()
    p = p.sort_values("hour_from_admission")
    return p

def load_raw_patient_signals(patient_id):
    # optional: load HR/RR/SpO2 from raw panel
    if not os.path.exists(RAW_PANEL_PATH):
        return None

    df = pd.read_csv(RAW_PANEL_PATH)
    if "patient_id" not in df.columns:
        return None

    p = df[df["patient_id"] == patient_id].copy()
    if len(p) == 0:
        return None

    p = p.sort_values("hour_from_admission")
    cols = ["hour_from_admission"]
    for c in ["heart_rate", "respiratory_rate", "spo2_pct"]:
        if c in p.columns:
            cols.append(c)
    return p[cols]

def plot_risk_curve(df_tl):
    fig = plt.figure()
    plt.plot(df_tl["hour_from_admission"], df_tl["risk_score"])
    plt.xlabel("Hour from admission")
    plt.ylabel("Risk score")
    plt.title("GRU Risk score over time")
    plt.grid(True)
    st.pyplot(fig)
    plt.close(fig)

def plot_signals(df_sig):
    # separate plots for clarity
    if df_sig is None:
        st.info("Raw vitals panel not loaded (optional).")
        return

    if "heart_rate" in df_sig.columns:
        fig = plt.figure()
        plt.plot(df_sig["hour_from_admission"], df_sig["heart_rate"])
        plt.xlabel("Hour from admission"); plt.ylabel("Heart Rate")
        plt.title("Heart Rate over time"); plt.grid(True)
        st.pyplot(fig); plt.close(fig)

    if "respiratory_rate" in df_sig.columns:
        fig = plt.figure()
        plt.plot(df_sig["hour_from_admission"], df_sig["respiratory_rate"])
        plt.xlabel("Hour from admission"); plt.ylabel("Respiratory Rate")
        plt.title("Respiratory Rate over time"); plt.grid(True)
        st.pyplot(fig); plt.close(fig)

    if "spo2_pct" in df_sig.columns:
        fig = plt.figure()
        plt.plot(df_sig["hour_from_admission"], df_sig["spo2_pct"])
        plt.xlabel("Hour from admission"); plt.ylabel("SpO2 (%)")
        plt.title("SpO2 over time"); plt.grid(True)
        st.pyplot(fig); plt.close(fig)

def safe_int(x, default=None):
    try:
        return int(x)
    except:
        return default

# =========================
# LOAD DATA
# =========================
df_pred = load_predictions()

# =========================
# SIDEBAR (NO ABOUT/NO NEXT STEPS)
# =========================
st.sidebar.title("ICU Dashboard")
page = st.sidebar.radio("Page", ["Doctor Dashboard", "Patient Page"])

# Search
st.sidebar.markdown("---")
st.sidebar.subheader("Search patient")
patient_input = st.sidebar.text_input("patient_id", value="")

threshold = st.sidebar.number_input("Alert threshold", min_value=0.0, max_value=1.0, value=0.94, step=0.01)

# =========================
# DOCTOR DASHBOARD
# =========================
if page == "Doctor Dashboard":
    st.title("Doctor Dashboard (GRU Early Warning)")

    # KPIs
    n_patients = df_pred["patient_id"].nunique()
    n_alerts = int((df_pred["alert"] == 1).sum())
    max_risk = float(df_pred["risk_score"].max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total patients", f"{n_patients}")
    c2.metric("Total alerts", f"{n_alerts}")
    c3.metric("Max risk score", f"{max_risk:.3f}")

    st.markdown("---")

    # Top 10 risky patients (by highest risk_score)
    st.subheader("Top 10 highest risk patients")

    top10 = df_pred.sort_values("risk_score", ascending=False).head(10).copy()
    top10["risk_level"] = top10["risk_score"].apply(risk_level_from_score)

    show_cols = ["patient_id", "hour_from_admission", "risk_score", "risk_level", "alert", "true_label", "threshold"]
    top10 = top10[show_cols]

    def style_row(row):
        return [color_risk(row["risk_level"])] * len(row)

    st.dataframe(top10.style.apply(style_row, axis=1), use_container_width=True)

    st.markdown("---")

    # Alerts table
    st.subheader("Alerts list (filterable)")

    only_alerts = df_pred[df_pred["risk_score"] >= threshold].copy()
    only_alerts["alert"] = (only_alerts["risk_score"] >= threshold).astype(int)
    only_alerts["risk_level"] = only_alerts["risk_score"].apply(risk_level_from_score)
    only_alerts = only_alerts.sort_values("risk_score", ascending=False)

    st.write(f"Patients above threshold={threshold:.2f}: **{len(only_alerts)}**")

    st.dataframe(
        only_alerts[show_cols].head(200).style.apply(style_row, axis=1),
        width="stretch"

    )

    st.info("Tip: go to Patient Page and paste patient_id to view timeline.")

# =========================
# PATIENT PAGE
# =========================
else:
    st.title("Patient Page")

    pid = safe_int(patient_input, None)

    if pid is None:
        st.warning("Enter a valid patient_id from sidebar.")
        st.stop()

    # timeline
    tl = load_patient_timeline_from_folder(pid)
    if tl is None:
        tl = build_timeline_from_predictions(df_pred, pid)

    if tl is None or len(tl) == 0:
        st.error(f"No data found for patient_id={pid}")
        st.stop()

    tl = tl.sort_values("hour_from_admission").reset_index(drop=True)
    tl["risk_level"] = tl["risk_score"].apply(risk_level_from_score)
    tl["alert"] = (tl["risk_score"] >= threshold).astype(int)

    last = tl.iloc[-1]
    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient", f"{pid}")
    c2.metric("Last hour", f"{int(last['hour_from_admission'])}")
    c3.metric("Last risk", f"{float(last['risk_score']):.3f}")
    c4.metric("Alert", "YES" if int(last["alert"]) == 1 else "NO")

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Risk score over time")
        plot_risk_curve(tl)

    with colB:
        st.subheader("Timeline table")
        show_cols = ["hour_from_admission", "risk_score", "risk_level", "alert", "true_label"]
        st.dataframe(tl[show_cols].tail(30).style.apply(
            lambda r: [color_risk(r["risk_level"])] * len(r), axis=1
        ), use_container_width=True)

    st.markdown("---")

    st.subheader("Vitals (optional)")
    df_sig = load_raw_patient_signals(pid)
    plot_signals(df_sig)

    st.markdown("---")
    st.subheader("Compare alert vs true label (last 30)")
    comp = tl[["hour_from_admission", "alert", "true_label", "risk_score"]].tail(30)
    st.dataframe(comp, width="stretch")

