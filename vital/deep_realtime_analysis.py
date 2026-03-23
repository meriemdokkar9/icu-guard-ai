import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HOURS_PER_DAY = 24

def apply_suppression(df, patient_col, theta, cooldown_hours, consecutive_k):
    df = df.sort_values([patient_col, "t_end"]).copy()
    df["raw_alert"] = (df["p"] >= theta).astype(int)
    df["alert"] = 0

    for pid, g in df.groupby(patient_col, sort=False):
        last_alert_time = None
        consec = 0
        for idx, row in g.iterrows():
            consec = consec + 1 if row["raw_alert"] == 1 else 0
            if consec >= consecutive_k:
                if last_alert_time is None:
                    df.loc[idx, "alert"] = 1
                    last_alert_time = row["t_end"]
                    consec = 0
                else:
                    hours_since = float(row["t_end"] - last_alert_time)
                    if hours_since >= cooldown_hours:
                        df.loc[idx, "alert"] = 1
                        last_alert_time = row["t_end"]
                        consec = 0
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--time_col", default="hour_from_admission")
    ap.add_argument("--label_col", default="true_label")
    ap.add_argument("--pred_col", default="risk_score")
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--cooldown", type=int, default=6)
    ap.add_argument("--consecutive", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    df = df[[args.patient_col, args.time_col, args.label_col, args.pred_col]].copy()
    df.columns = ["pid","t_end","y","p"]
    df["y"] = df["y"].astype(int)
    df["p"] = df["p"].astype(float)
    df["t_end"] = df["t_end"].astype(int)

    # suppression
    df2 = apply_suppression(df, "pid", args.theta, args.cooldown, args.consecutive)

    # alerts per patient-day distribution
    per_pat_hours = df2.groupby("pid")["t_end"].count()
    per_pat_alerts = df2.groupby("pid")["alert"].sum()
    alerts_per_day_per_patient = (per_pat_alerts / per_pat_hours) * HOURS_PER_DAY

    # event patients vs non-event patients
    has_event = df2.groupby("pid")["y"].max()
    flagged = df2.groupby("pid")["alert"].max()

    non_event_flag_rate = float((flagged[has_event==0].mean())*100.0) if (has_event==0).any() else np.nan
    event_flag_rate = float((flagged[has_event==1].mean())*100.0) if (has_event==1).any() else np.nan

    # lead times (first alert before first event)
    first_event_time = df2[df2["y"]==1].groupby("pid")["t_end"].min()
    lead_times = []
    for pid, t_event in first_event_time.items():
        g = df2[df2["pid"]==pid]
        a = g[(g["alert"]==1) & (g["t_end"] < t_event)]
        if len(a)>0:
            lead_times.append(float(t_event - a["t_end"].iloc[0]))

    lead_times = np.array(lead_times, dtype=float)

    print("=== Deep Real-time Analysis ===")
    print("theta=", args.theta, "cooldown=", args.cooldown, "consecutive=", args.consecutive)
    print("Patients:", df2["pid"].nunique())
    print("Event patients:", int((has_event==1).sum()))
    print("Non-event patients:", int((has_event==0).sum()))
    print(f"Flag rate (event patients): {event_flag_rate:.2f}%")
    print(f"Flag rate (non-event patients): {non_event_flag_rate:.2f}%")
    print(f"Median alerts/day per patient: {np.median(alerts_per_day_per_patient):.3f}")
    print(f"90th percentile alerts/day: {np.percentile(alerts_per_day_per_patient, 90):.3f}")

    if len(lead_times)>0:
        print(f"Lead time median (h): {np.median(lead_times):.2f}")
        print(f"Lead time IQR (h): {np.percentile(lead_times,75)-np.percentile(lead_times,25):.2f}")

    # PNG 1: alerts/day distribution
    plt.figure(figsize=(7,5))
    plt.hist(alerts_per_day_per_patient, bins=30)
    plt.xlabel("Alerts per patient-day")
    plt.ylabel("Number of patients")
    plt.title("Alert burden distribution (patient-level)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/deep_alerts_per_patient_day.png", dpi=200)
    plt.close()

    # PNG 2: lead time distribution
    plt.figure(figsize=(7,5))
    if len(lead_times)>0:
        plt.hist(lead_times, bins=30)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Number of detected events")
    plt.title("Lead time distribution (first alert before event)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/deep_lead_time_distribution.png", dpi=200)
    plt.close()

    # PNG 3: bar flag rates
    plt.figure(figsize=(6,4))
    plt.bar(["Event patients","Non-event patients"], [event_flag_rate, non_event_flag_rate])
    plt.ylabel("Percent flagged (%)")
    plt.title("Who gets alerted?")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/deep_flag_rates.png", dpi=200)
    plt.close()

    print("Saved PNGs in outputs/: deep_alerts_per_patient_day.png, deep_lead_time_distribution.png, deep_flag_rates.png")

if __name__ == "__main__":
    main()