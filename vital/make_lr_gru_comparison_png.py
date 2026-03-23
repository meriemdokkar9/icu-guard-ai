import pandas as pd
import matplotlib.pyplot as plt

def main():
    lr = pd.read_csv(r"outputs\lr_realtime_results_suppressed.csv")
    gru = pd.read_csv(r"outputs\gru_realtime_results_suppressed.csv")

    # 1) Recall vs Alerts curve
    plt.figure(figsize=(7,5))
    plt.plot(lr["alerts_per_patient_day"], lr["event_recall"], marker="o", label="LR baseline")
    plt.plot(gru["alerts_per_patient_day"], gru["event_recall"], marker="o", label="GRU (24h, 33F)")
    plt.xlabel("Alerts per patient-day")
    plt.ylabel("Event recall")
    plt.title("Real-time trade-off: recall vs alert burden")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"outputs\compare_lr_vs_gru_recall_vs_alerts.png", dpi=200)
    plt.close()

    # 2) Table PNG (best point by recall under alerts/day <= 1.0)
    def pick_best(df):
        cand = df[df["alerts_per_patient_day"] <= 1.0].copy()
        if len(cand) == 0:
            cand = df.copy()
        return cand.sort_values(["event_recall","median_lead_hours"], ascending=False).head(1)

    lr_best = pick_best(lr)
    gru_best = pick_best(gru)

    tbl = pd.concat([
        lr_best.assign(model="LR"),
        gru_best.assign(model="GRU")
    ], ignore_index=True)

    cols = ["model","theta","alerts_per_patient_day","pct_patients_flagged","n_events","event_recall","median_lead_hours","iqr_lead_hours"]
    tbl = tbl[cols]

    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    ax.set_title("Best operating points (constraint: alerts/day ≤ 1.0 if possible)", fontsize=12, pad=10)

    show = tbl.copy()
    for c in show.columns:
        if c in ["theta"]:
            show[c] = show[c].map(lambda x: f"{x:.2f}")
        elif show[c].dtype.kind in "f":
            show[c] = show[c].map(lambda x: f"{x:.3f}")

    table = ax.table(cellText=show.values, colLabels=show.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(r"outputs\compare_lr_vs_gru_best_points.png", dpi=200)
    plt.close(fig)

    print("Saved PNGs:")
    print(r"outputs\compare_lr_vs_gru_recall_vs_alerts.png")
    print(r"outputs\compare_lr_vs_gru_best_points.png")

if __name__ == "__main__":
    main()