import pandas as pd
import matplotlib.pyplot as plt

# الملف الذي حفظناه
CSV_FILE = "track_b50ea1e4.csv"

df = pd.read_csv(CSV_FILE)

# غالباً الملف عمودين: time و value (لكن أسماء الأعمدة قد تختلف)
# نأخذ أول عمود كزمن، وثاني عمود كقيمة
time_col = df.columns[0]
val_col = df.columns[1]

df = df[[time_col, val_col]].dropna()
df.columns = ["t", "spo2"]

# ترتيب حسب الزمن
df = df.sort_values("t")

print("Rows:", len(df))
print("Time range (sec):", df["t"].min(), "->", df["t"].max())
print("SpO2 range:", df["spo2"].min(), "->", df["spo2"].max())

# رسم أول 20 دقيقة (إذا موجودة)
max_t = df["t"].min() + 20 * 60
df_plot = df[df["t"] <= max_t].copy()

plt.figure()
plt.plot(df_plot["t"], df_plot["spo2"])
plt.xlabel("Time (sec)")
plt.ylabel("SpO2")
plt.title("SpO2 signal (first 20 min)")
plt.tight_layout()
plt.savefig("spo2_plot.png", dpi=150)
print("Saved plot: spo2_plot.png")


# --- اكتشاف تدهور تنفسي بسيط ---
# تعريف: تدهور = SpO2 < 90
events = df[df["spo2"] < 90].copy()
print("\nNumber of samples with SpO2 < 90:", len(events))

if len(events) > 0:
    first_event_time = events["t"].iloc[0]
    print("First time SpO2 < 90 at (sec):", first_event_time)

    # نحفظ أوقات التدهور
    events_out = "events_spo2_lt90.csv"
    events[["t", "spo2"]].to_csv(events_out, index=False)
    print("Saved:", events_out)
else:
    print("No deterioration (SpO2<90) found in this track.")

python step4_plot_and_events.py
