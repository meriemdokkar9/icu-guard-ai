import glob
import pandas as pd

def main():
    files = glob.glob(r"outputs\fused_a*_results.csv")
    if not files:
        raise SystemExit("No files found: outputs\\fused_a*_results.csv")

    best = None
    rows = []

    for f in files:
        df = pd.read_csv(f)
        df = df[df["alerts_per_patient_day"] <= 1.0].copy()
        if len(df) == 0:
            continue

        # pick best row per file
        df = df.sort_values(["event_recall", "median_lead_hours"], ascending=[False, False])
        top = df.iloc[0].to_dict()
        top["file"] = f
        rows.append(top)

    if not rows:
        raise SystemExit("No operating point with alerts/day <= 1 found.")

    allr = pd.DataFrame(rows)
    allr = allr.sort_values(["event_recall", "median_lead_hours"], ascending=[False, False])

    best = allr.iloc[0]
    print("=== Candidate best points (alerts/day <= 1) ===")
    print(allr[["file","theta","alerts_per_patient_day","event_recall","median_lead_hours","pct_patients_flagged"]].to_string(index=False))
    print("\n=== BEST CHOICE ===")
    print(best.to_string())

    allr.to_csv(r"outputs\best_points_summary.csv", index=False)
    print("\nSaved: outputs\\best_points_summary.csv")

if __name__ == "__main__":
    main()