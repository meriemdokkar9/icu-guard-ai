import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--title", default="Model Results")
    ap.add_argument("--out_prefix", default="outputs/results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Plot Recall vs Alerts
    plt.figure(figsize=(7,5))
    plt.plot(df["alerts_per_patient_day"], df["event_recall"], marker="o")
    plt.xlabel("Alerts per patient-day")
    plt.ylabel("Event Recall")
    plt.title(args.title + " - Recall vs Alert Burden")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_recall_vs_alerts.png", dpi=300)
    plt.close()

    # Plot Table as PNG
    fig, ax = plt.subplots(figsize=(12,2.5))
    ax.axis("off")
    ax.set_title(args.title, pad=10)

    show = df.copy()
    for c in show.columns:
        if show[c].dtype.kind in "f":
            show[c] = show[c].map(lambda x: f"{x:.3f}")

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,1.3)

    plt.tight_layout()
    plt.savefig(args.out_prefix + "_table.png", dpi=300)
    plt.close()

    print("Saved PNGs:")
    print(args.out_prefix + "_recall_vs_alerts.png")
    print(args.out_prefix + "_table.png")

if __name__ == "__main__":
    main()