import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Input نتائجك (من اللي طلعتيها)
# -----------------------------
data = [
    {
        "Split": "Kaggle Style\n(Random Row Split)",
        "ROC-AUC": 0.9752491627,
        "PR-AUC": 0.8253782139,
        "Precision (Class 1)": 0.9127,
        "Recall (Class 1)": 0.5834,
        "F1 (Class 1)": 0.7119,
        "Note": "High but optimistic (possible leakage)"
    },
    {
        "Split": "Realistic\n(GroupSplit by patient_id)",
        "ROC-AUC": 0.9467927649,
        "PR-AUC": 0.7120012521,
        "Precision (Class 1)": 0.8595,
        "Recall (Class 1)": 0.4965,
        "F1 (Class 1)": 0.6294,
        "Note": "Clinically realistic (no patient overlap)"
    }
]

df = pd.DataFrame(data)

# -----------------------------
# فرق النتائج (Delta)
# -----------------------------
delta = df.copy()
delta.loc[0, "Split"] = "Difference\n(Kaggle - Realistic)"
for col in ["ROC-AUC", "PR-AUC", "Precision (Class 1)", "Recall (Class 1)", "F1 (Class 1)"]:
    delta.loc[0, col] = df.loc[0, col] - df.loc[1, col]
delta.loc[0, "Note"] = "Drop expected when leakage is removed"

df_all = pd.concat([df, delta.iloc[[0]]], ignore_index=True)

print("\n=== Comparison (Kaggle vs Realistic) ===\n")
print(df_all.to_string(index=False))

# -----------------------------
# حفظ CSV
# -----------------------------
df_all.to_csv("comparison_kaggle_vs_realistic.csv", index=False)
print("\nSaved: comparison_kaggle_vs_realistic.csv")

# -----------------------------
# 1) Plot ROC/PR (Bar)
# -----------------------------
plt.figure(figsize=(9, 5))
plt.bar(df["Split"], df["ROC-AUC"], label="ROC-AUC")
plt.bar(df["Split"], df["PR-AUC"], bottom=0, label="PR-AUC")  # overlay look
plt.title("Kaggle Random Split vs Realistic GroupSplit (AUCs)")
plt.ylabel("Score")
plt.ylim(0.5, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot_auc_kaggle_vs_realistic.png", dpi=200)
plt.show()

print("Saved: plot_auc_kaggle_vs_realistic.png")

# -----
