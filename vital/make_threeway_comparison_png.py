import pandas as pd
import matplotlib.pyplot as plt

def main():

    gru = pd.read_csv(r"outputs\gru_realtime_results_suppressed.csv")
    f06 = pd.read_csv(r"outputs\fused_a06_results.csv")
    f04 = pd.read_csv(r"outputs\fused_a04_results.csv")

    plt.figure(figsize=(8,6))

    plt.plot(gru["alerts_per_patient_day"], gru["event_recall"], marker="o", label="GRU")
    plt.plot(f06["alerts_per_patient_day"], f06["event_recall"], marker="o", label="Fusion alpha=0.6")
    plt.plot(f04["alerts_per_patient_day"], f04["event_recall"], marker="o", label="Fusion alpha=0.4")

    # Highlight chosen operating point (alpha=0.6, theta=0.08)
    sel = f06.loc[(f06["theta"] - 0.08).abs().idxmin()]
    plt.scatter([sel["alerts_per_patient_day"]],
                [sel["event_recall"]],
                marker="*",
                s=250,
                label="Chosen Operating Point")

    plt.xlabel("Alerts per patient-day")
    plt.ylabel("Event Recall")
    plt.title("Real-time Trade-off: Recall vs Alert Burden")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"outputs\compare_gru_fusion.png", dpi=300)
    plt.close()

    print("Saved: outputs\\compare_gru_fusion.png")

if __name__ == "__main__":
    main()