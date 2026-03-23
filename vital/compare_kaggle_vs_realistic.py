import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Results Data
# -----------------------------
results = {
    "Split Type": ["Kaggle Random Row Split", "Realistic GroupSplit (patient_id)"],
    "ROC-AUC": [0.9752491627, 0.9467927649],
    "PR-AUC": [0.8253782139, 0.7120012521],
    "Precision_Class1": [0.9127, 0.8595],
    "Recall_Class1": [0.5834, 0.4965],
    "F1_Class1": [0.7119, 0.6294]
}

df = pd.DataFrame(results)

# -----------------------------
# Print Table
# -----------------------------
print("\n=== Comparison Table (Kaggle vs Realistic) ===\n")
print(df.to_string(index=False))

# -----------------------------
# Save as CSV
# -----------------------------
df.to_csv("kaggle_vs_realistic_comparison.csv", index=False)
print("\nSaved: kaggle_vs_realistic_comparison.csv")

# -----------------------------
# Plot ROC-AUC and PR-AUC
# -----------------------------
plt.figure(figsize=(8, 5))
df.plot(x="Split Type", y=["ROC-AUC", "PR-AUC"], kind="bar")
plt.title("Kaggle vs Realistic Split (ROC-AUC & PR-AUC)")
plt.ylabel("Score")
plt.ylim(0.5, 1.0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("kaggle_vs_realistic_auc.png", dpi=200)
plt.show()

print("Saved: kaggle_vs_realistic_auc.png")

# -----------------------------
# Plot Precision / Recall / F1 for Class 1
# -----------------------------
plt.figure(figsize=(8, 5))
df.plot(x="Split Type", y=["Precision_Class1", "Recall_Class1", "F1_Class1"], kind="bar")
plt.title("Kaggle vs Realistic Split (Class 1 Metrics)")
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("kaggle_vs_realistic_class1_metrics.png", dpi=200)
plt.show()

print("Saved: kaggle_vs_realistic_class1_metrics.png")
