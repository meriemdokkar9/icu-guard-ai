import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HOURS_PER_DAY = 24

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def try_parse_datetime(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed
    return s

def to_hours_diff(a, b) -> float:
    diff = a - b
    if hasattr(diff, "total_seconds"):
        return float(diff.total_seconds() / 3600.0)
    return float(diff)

def compute_first_event_times(df: pd.DataFrame, patient_col: str) -> pd.Series:
    return df.loc[df["y"] == 1].groupby(patient_col)["t_end"].min()

def apply_suppression(df_in: pd.DataFrame, patient_col: str, theta: float, cooldown_hours: int, consecutive_k: int) -> pd.DataFrame:
    df = df_in.copy()
    df["raw_alert"] = (df["p"] >= theta).astype(int)
    df["alert"] = 0

    for pid, g in df.groupby(patient_col, sort=False):
        g = g.sort_values("t_end")
        last_alert_time = None
        consec = 0

        for ridx, row in g.iterrows():
            if row["raw_alert"] == 1:
                consec += 1
            else:
                consec = 0

            if consec >= consecutive_k:
                if last_alert_time is None:
                    df.loc[ridx, "alert"] = 1
                    last_alert_time = row["t_end"]
                    consec = 0
                else:
                    dt = row["t_end"] - last_alert_time
                    if hasattr(dt, "total_seconds"):
                        hours_since = dt.total_seconds() / 3600.0
                    else:
                        hours_since = float(dt)

                    if hours_since >= cooldown_hours:
                        df.loc[ridx, "alert"] = 1
                        last_alert_time = row["t_end"]
                        consec = 0
    return df

def evaluate_streaming(df_base: pd.DataFrame, patient_col: str, theta: float, cooldown: int, consecutive: int):
    df = apply_suppression(df_base, patient_col, theta, cooldown, consecutive)

    total_alerts = int(df["alert"].sum())
    total_steps = int(len(df))
    alerts_per_patient_day = (total_alerts / total_steps) * HOURS_PER_DAY if total_steps else 0.0

    flagged_patients = df.groupby(patient_col)["alert"].max()
    pct_patients_flagged = float(100.0 * flagged_patients.mean()) if len(flagged_patients) else 0.0

    first_event = compute_first_event_times(df, patient_col)
    patients_with_event = list(first_event.index)

    detected = 0
    missed = 0
    lead_times = []

    for pid in patients_with_event:
        t_event = first_event.loc[pid]
        g = df.loc[df[patient_col] == pid].sort_values("t_end")
        alerts_before = g.loc[(g["t_end"] < t_event) & (g["alert"] == 1)]
        if len(alerts_before) == 0:
            missed += 1
            continue
        detected += 1
        t_first_alert = alerts_before["t_end"].iloc[0]
        lead_times.append(to_hours_diff(t_event, t_first_alert))

    n_events = len(patients_with_event)
    event_recall = (detected / n_events) if n_events else float("nan")

    lead_arr = np.array(lead_times, dtype=float)
    median_lead = float(np.median(lead_arr)) if len(lead_arr) else float("nan")
    iqr_lead = float(np.percentile(lead_arr, 75) - float(np.percentile(lead_arr, 25))) if len(lead_arr) else float("nan")

    return {
        "theta": float(theta),
        "alerts_per_patient_day": float(alerts_per_patient_day),
        "pct_patients_flagged": float(pct_patients_flagged),
        "n_events": int(n_events),
        "event_recall": float(event_recall),
        "missed_events": int(missed),
        "median_lead_hours": float(median_lead),
        "iqr_lead_hours": float(iqr_lead),
    }

def save_table_png(df: pd.DataFrame, path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    df_show = df.copy()
    for c in df_show.columns:
        if c in ["theta"]:
            df_show[c] = df_show[c].map(lambda x: f"{x:.1f}")
        elif df_show[c].dtype.kind in "f":
            df_show[c] = df_show[c].map(lambda x: f"{x:.3f}")

    table = ax.table(cellText=df_show.values, colLabels=df_show.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_recall_vs_alerts(df: pd.DataFrame, out_path: str, title: str):
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(df["alerts_per_patient_day"], df["event_recall"], marker="o")
    for _, r in df.iterrows():
        ax.annotate(f"θ={r['theta']:.1f}", (r["alerts_per_patient_day"], r["event_recall"]), fontsize=8)
    ax.set_xlabel("Alerts per patient-day")
    ax.set_ylabel("Event recall")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True, help="CSV with GRU predictions")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--patient_col", type=str, default="patient_id")
    ap.add_argument("--time_col", type=str, default="hour_from_admission")
    ap.add_argument("--label_col", type=str, default="deterioration_next_12h")
    ap.add_argument("--pred_col", type=str, default="p", help="prediction probability column name")
    ap.add_argument("--thresholds", type=str, default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--cooldown", type=int, default=6)
    ap.add_argument("--consecutive", type=int, default=3)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.pred_csv)

    for c in [args.patient_col, args.time_col, args.label_col, args.pred_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Available: {list(df.columns)[:40]} ...")

    df[args.time_col] = try_parse_datetime(df[args.time_col])
    df["y"] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int)
    df["p"] = pd.to_numeric(df[args.pred_col], errors="coerce").fillna(0.0).astype(float)
    df["t_end"] = df[args.time_col]

    # IMPORTANT: sort
    df = df.sort_values([args.patient_col, "t_end"]).reset_index(drop=True)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    rows = []
    for th in thresholds:
        rows.append(evaluate_streaming(df[[args.patient_col, "t_end", "y", "p"]], args.patient_col, th, args.cooldown, args.consecutive))

    res = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "gru_realtime_results_suppressed.csv")
    res.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(res)

    save_table_png(res, os.path.join(args.outdir, "gru_realtime_table.png"),
                   title=f"GRU suppressed (cooldown={args.cooldown}h, consecutive={args.consecutive})")
    plot_recall_vs_alerts(res, os.path.join(args.outdir, "gru_realtime_recall_vs_alerts.png"),
                          title="GRU real-time: recall vs alert burden (suppressed)")
    print("Saved PNGs in outputs/: gru_realtime_table.png, gru_realtime_recall_vs_alerts.png")

if __name__ == "__main__":
    main()