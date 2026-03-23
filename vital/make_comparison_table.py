import pandas as pd
import matplotlib.pyplot as plt

# البيانات
data = {
    "Split": [
        "Kaggle Random Split",
        "Realistic GroupSplit (patient_id)"
    ],
    "ROC-AUC": [0.975, 0.947],
    "PR-AUC": [0.825, 0.712],
    "Precision (Class 1)": [0.913, 0.860],
    "Recall (Class 1)": [0.583, 0.497],
    "F1 (Class 1)": [0.712, 0.629]
}

df = pd.DataFrame(data)

# رسم الجدول كصورة
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

plt.title("Kaggle vs Realistic Evaluation Comparison", pad=20)
plt.tight_layout()
plt.savefig("table_kaggle_vs_realistic.png", dpi=300)
plt.show()

print("Saved successfully: table_kaggle_vs_realistic.png")
