import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_avg(x, w):
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()


def compute_alerts_for_patient(t, p, theta, cooldown, consecutive):
    """
    Given time t (ascending) and prob p, produce binary alerts with cooldown + consecutive logic.
    """
    raw = (p >= theta).astype(int)
    alerts = np.zeros_like(raw, dtype=int)

    last_alert_time = None
    consec = 0
    for i in range(len(raw)):
        consec = consec + 1 if raw[i] == 1 else 0
        if consec >= consecutive:
            if last_alert_time is None or (t[i] - last_alert_time) >= cooldown:
                alerts[i] = 1
                last_alert_time = t[i]
                consec = 0
    return alerts


def evaluate(df, theta, cooldown, consecutive):
    """
    Event-level recall: detect an event if at least one alert occurs before first event time.
    Lead time: first_event_time - first_alert_time (first alert before event).
    Alert burden: alerts per patient-day.
    """
    # per patient hours and alert counts
    per_pat_hours = df.groupby("patient_id")["hour_from_admission"].count()
    per_pat_alerts = df.groupby("patient_id")["alert"].sum()
    alerts_per_patient_day = float(((per_pat_alerts / per_pat_hours) * 24.0).mean())

    # patient flagged
    flagged = df.groupby("patient_id")["alert"].max()
    pct_patients_flagged = float(flagged.mean() * 100.0)

    # event-level
    first_event_time = df[df["true_label"] == 1].groupby("patient_id")["hour_from_admission"].min()
    n_events = int(len(first_event_time))

    detected = 0
    lead_times = []
    for pid, t_event in first_event_time.items():
        g = df[df["patient_id"] == pid]
        a = g[(g["alert"] == 1) & (g["hour_from_admission"] < t_event)]
        if len(a) > 0:
            detected += 1
            lead_times.append(float(t_event - a["hour_from_admission"].iloc[0]))

    event_recall = detected / n_events if n_events > 0 else np.nan
    missed = n_events - detected

    if len(lead_times) > 0:
        median_lead = float(np.median(lead_times))
        iqr_lead = float(np.percentile(lead_times, 75) - np.percentile(lead_times, 25))
    else:
        median_lead = np.nan
        iqr_lead = np.nan

    return {
        "theta": theta,
        "alerts_per_patient_day": alerts_per_patient_day,
        "pct_patients_flagged": pct_patients_flagged,
        "n_events": n_events,
        "event_recall": event_recall,
        "missed_events": missed,
        "median_lead_hours": median_lead,
        "iqr_lead_hours": iqr_lead,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--cooldown", type=int, default=6)
    ap.add_argument("--consecutive", type=int, default=1)

    # Innovation knobs
    ap.add_argument("--smooth", type=int, default=3, help="moving average window (hours)")
    ap.add_argument("--trend", type=int, default=3, help="trend window (hours) for slope")
    ap.add_argument("--trend_min", type=float, default=0.005, help="minimum increase required")
    ap.add_argument("--watch_theta", type=float, default=0.08, help="optional watch threshold")
    ap.add_argument("--use_watch", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    df = df[["patient_id", "hour_from_admission", "true_label", "risk_score"]].copy()
    df = df.sort_values(["patient_id", "hour_from_admission"]).reset_index(drop=True)

    # Apply smoothing + trend gating per patient
    df["risk_smooth"] = np.nan
    df["trend_ok"] = 0

    for pid, g in df.groupby("patient_id", sort=False):
        p = g["risk_score"].to_numpy(dtype=float)
        t = g["hour_from_admission"].to_numpy(dtype=int)

        ps = moving_avg(p, args.smooth)
        df.loc[g.index, "risk_smooth"] = ps

        # trend: compare current smooth vs smooth(trend_window hours ago)
        trend_ok = np.zeros_like(t, dtype=int)
        for i in range(len(t)):
            j = i - (args.trend - 1)
            if j >= 0 and not np.isnan(ps[i]) and not np.isnan(ps[j]):
                if (ps[i] - ps[j]) >= args.trend_min:
                    trend_ok[i] = 1
        df.loc[g.index, "trend_ok"] = trend_ok

    # replace NaNs from smoothing with original risk (early hours)
    df["risk_smooth"] = df["risk_smooth"].fillna(df["risk_score"])

    # Build final score for alerting:
    # require trend_ok=1 to reduce noisy spikes
    df["score"] = df["risk_smooth"] * df["trend_ok"]

    # ALERT logic
    df["alert"] = 0
    for pid, g in df.groupby("patient_id", sort=False):
        t = g["hour_from_admission"].to_numpy(dtype=int)
        p = g["score"].to_numpy(dtype=float)
        alerts = compute_alerts_for_patient(t, p, args.theta, args.cooldown, args.consecutive)
        df.loc[g.index, "alert"] = alerts

    # Optional: WATCH stage (early warning) — counts separately
    if args.use_watch:
        df["watch"] = 0
        for pid, g in df.groupby("patient_id", sort=False):
            t = g["hour_from_admission"].to_numpy(dtype=int)
            p = g["score"].to_numpy(dtype=float)
            watch = compute_alerts_for_patient(t, p, args.watch_theta, args.cooldown, args.consecutive)
            df.loc[g.index, "watch"] = watch

    # Evaluate and save table
    res = evaluate(df, args.theta, args.cooldown, args.consecutive)
    out_table = pd.DataFrame([res])
    out_csv = f"{args.outdir}/advanced_policy_results.csv"
    out_table.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(out_table)

    # PNGs
    # 1) quick table figure
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.axis("off")
    ax.set_title("Advanced Policy (smoothing + trend gating) - Results", pad=10)
    show = out_table.copy()
    for c in show.columns:
        if show[c].dtype.kind in "f":
            show[c] = show[c].map(lambda x: f"{x:.3f}")
    tbl = ax.table(cellText=show.values, colLabels=show.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)
    plt.tight_layout()
    fig.savefig(f"{args.outdir}/advanced_policy_table.png", dpi=200)
    plt.close(fig)

    # 2) lead-time distribution for detected events
    first_event_time = df[df["true_label"]==1].groupby("patient_id")["hour_from_admission"].min()
    lead_times = []
    for pid, t_event in first_event_time.items():
        g = df[df["patient_id"]==pid]
        a = g[(g["alert"]==1) & (g["hour_from_admission"] < t_event)]
        if len(a)>0:
            lead_times.append(float(t_event - a["hour_from_admission"].iloc[0]))

    plt.figure(figsize=(7,5))
    if len(lead_times)>0:
        plt.hist(lead_times, bins=30)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Count")
    plt.title("Lead time distribution (Advanced Policy)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/advanced_policy_leadtime.png", dpi=200)
    plt.close()

    print("Saved PNGs:", "advanced_policy_table.png", "advanced_policy_leadtime.png")


if __name__ == "__main__":
    main()