import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ICU Early Deterioration Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRED_FILE = os.path.join(BASE_DIR, "dashboard_predictions_gru.csv")
ALERTS_FILE = os.path.join(BASE_DIR, "dashboard_alerts_only.csv")  # optional
RAW_FILE = os.path.join(BASE_DIR, "hospital_deterioration_hourly_panel.csv")

TIMELINES_DIR = os.path.join(BASE_DIR, "timelines_alert_patients")  # optional (alert timelines)

PID_COL = "patient_id"
TIME_COL = "hour_from_admission"
LABEL_COL = "deterioration_next_12h"

SIG_COLS = ["heart_rate", "respiratory_rate", "spo2_pct"]  # HR/RR/SpO2

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_predictions():
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(PRED_FILE)
    dfp = pd.read_csv(PRED_FILE)
    # safety
    for c in ["risk_score", "alert", "threshold"]:
        if c not in dfp.columns:
            raise ValueError(f"Missing column '{c}' in {PRED_FILE}")
    return dfp


@st.cache_data(show_spinner=False)
def load_raw_minimal():
    """Load minimal columns from raw hourly panel."""
    if not os.path.exists(RAW_FILE):
        return None

    usecols = [PID_COL, TIME_COL, LABEL_COL] + [c for c in SIG_COLS if c]
    df = pd.read_csv(RAW_FILE, usecols=[c for c in usecols if c is not None])
    # ensure numeric for plotting
    for c in [TIME_COL] + SIG_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values([PID_COL, TIME_COL]).reset_index(drop=True)
    return df


def risk_level(score: float):
    if score >= 0.94:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    return "LOW"


def risk_color(level: str):
    return {"HIGH": "🟥", "MEDIUM": "🟨", "LOW": "🟩"}.get(level, "⬜")


def safe_int(x, default=None):
    try:
        return int(x)
    except:
        return default


# =========================
# LOAD DATA
# =========================
try:
    df_pred = load_predictions()
except Exception as e:
    st.error(f"Predictions file error: {e}")
    st.stop()

df_raw = load_raw_minimal()  # can be None

# add risk_level if missing
if "risk_level" not in df_pred.columns:
    df_pred["risk_level"] = df_pred["risk_score"].apply(risk_level)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio("Page", ["Hospital Dashboard", "Patient Dashboard"])

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("View", ["All patients", "Alerts only"])
search_pid = st.sidebar.text_input("Search patient_id (optional)")
thr = float(df_pred["threshold"].iloc[0]) if "threshold" in df_pred.columns else 0.94
st.sidebar.markdown(f"**Model threshold:** {thr}")

# =========================
# FILTER PREDICTIONS
# =========================
df_view = df_pred.copy()

if view_mode == "Alerts only":
    df_view = df_view[df_view["alert"] == 1].copy()

if search_pid.strip():
    pid = safe_int(search_pid.strip())
    if pid is not None:
        df_view = df_view[df_view[PID_COL] == pid].copy()

# =========================
# HEADER
# =========================
st.title("🩺 ICU Early Deterioration Dashboard (GRU 24h)")
st.caption("Hospital-level monitoring + patient-level signals (HR/RR/SpO₂)")

# =========================
# HOSPITAL DASHBOARD
# =========================
if page == "Hospital Dashboard":
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total patients", len(df_pred))
    c2.metric("Rows in view", len(df_view))
    c3.metric("Total alerts", int(df_pred["alert"].sum()))
    c4.metric("Max risk", float(np.round(df_pred["risk_score"].max(), 6)))

    st.markdown("---")

    # Top 10 high risk
    st.subheader("🔥 Top 10 Highest Risk Patients")
    top10 = df_pred.sort_values("risk_score", ascending=False).head(10).copy()
    top10["flag"] = top10["risk_level"].apply(risk_color)

    cols_show = ["flag", PID_COL, TIME_COL, "risk_score", "risk_level", "alert", "true_label"]
    cols_show = [c for c in cols_show if c in top10.columns]
    st.dataframe(top10[cols_show], width="stretch")

    st.markdown("---")

    # Alerts table
    st.subheader("📋 Patients Table")
    df_table = df_view.sort_values("risk_score", ascending=False).copy()
    df_table["flag"] = df_table["risk_level"].apply(risk_color)

    cols_show2 = ["flag", PID_COL, TIME_COL, "risk_score", "risk_level", "alert", "true_label"]
    cols_show2 = [c for c in cols_show2 if c in df_table.columns]
    st.dataframe(df_table[cols_show2], width="stretch")

    st.markdown("---")

    # Risk distribution
    st.subheader("📊 Risk Score Distribution")
    fig = px.histogram(df_pred, x="risk_score", nbins=50, title="Risk Score Histogram")
    st.plotly_chart(fig, width="stretch")

# =========================
# PATIENT DASHBOARD
# =========================
else:
    st.subheader("👤 Patient Page")

    pid_list = df_pred[PID_COL].unique().tolist()
    default_pid = safe_int(search_pid.strip(), None)
    if default_pid is None or default_pid not in pid_list:
        default_pid = pid_list[0]

    selected_pid = st.selectbox("Select patient_id", pid_list, index=pid_list.index(default_pid))

    # show patient summary (from predictions)
    p_row = df_pred[df_pred[PID_COL] == selected_pid].iloc[0]
    lev = p_row["risk_level"]
    st.markdown(
        f"### Summary\n"
        f"- **Patient:** `{selected_pid}`\n"
        f"- **Latest hour:** `{int(p_row[TIME_COL])}`\n"
        f"- **Risk score:** `{float(p_row['risk_score']):.6f}`  {risk_color(lev)} **{lev}**\n"
        f"- **Alert:** `{int(p_row['alert'])}` (threshold `{thr}`)\n"
        f"- **True label (at that hour):** `{int(p_row.get('true_label', 0))}`\n"
    )

    st.markdown("---")

    # Load timeline for this patient (best: from raw signals + also compute risk curve if timeline csv exists)
    # 1) Risk timeline (from exported timelines if exist)
    tl_path = os.path.join(TIMELINES_DIR, f"patient_{selected_pid}_timeline.csv")
    tl_full_path = os.path.join(BASE_DIR, f"patient_{selected_pid}_timeline_full.csv")  # if you exported full
    df_tl = None

    if os.path.exists(tl_full_path):
        df_tl = pd.read_csv(tl_full_path)
    elif os.path.exists(tl_path):
        df_tl = pd.read_csv(tl_path)

    if df_tl is not None and len(df_tl) > 0 and "risk_score" in df_tl.columns:
        df_tl = df_tl.sort_values("hour_from_admission").reset_index(drop=True)

        st.subheader("📈 Risk score over time")
        fig_risk = px.line(df_tl, x="hour_from_admission", y="risk_score", title="Risk Score Timeline")
        st.plotly_chart(fig_risk, width="stretch")

        st.subheader("✅ Alert vs True Label (timeline)")
        show_cols = [c for c in ["hour_from_admission", "risk_score", "alert", "true_label"] if c in df_tl.columns]
        st.dataframe(df_tl[show_cols].tail(50), width="stretch")
    else:
        st.info("No risk timeline file found for this patient. (This is OK — you still have hospital-level predictions.)")

    st.markdown("---")

    # 2) Signals HR/RR/SpO2 from raw file
    if df_raw is None:
        st.warning("Raw hourly file not found, so HR/RR/SpO₂ plots cannot be displayed.")
    else:
        df_sig = df_raw[df_raw[PID_COL] == selected_pid].copy()
        if len(df_sig) == 0:
            st.warning("No raw signal rows found for this patient.")
        else:
            df_sig = df_sig.sort_values(TIME_COL).reset_index(drop=True)

            st.subheader("🫀 Vital signals over time (HR / RR / SpO₂)")

            # build long format for nicer plot
            cols_present = [c for c in SIG_COLS if c in df_sig.columns]
            if len(cols_present) == 0:
                st.warning("HR/RR/SpO₂ columns are missing in the raw dataset.")
            else:
                df_long = df_sig[[TIME_COL] + cols_present].melt(
                    id_vars=[TIME_COL], var_name="signal", value_name="value"
                )
                fig_sig = px.line(
                    df_long,
                    x=TIME_COL,
                    y="value",
                    color="signal",
                    title="Signals Timeline"
                )
                st.plotly_chart(fig_sig, width="stretch")

                st.subheader("📌 Last 10 hours values")
                st.dataframe(df_sig[[TIME_COL] + cols_present].tail(10), width="stretch")
