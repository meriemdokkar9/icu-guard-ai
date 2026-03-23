import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# =============================
# Load predictions (must be in same folder)
# =============================
df = pd.read_csv("dashboard_predictions_gru.csv")

# Detect columns automatically
pid_col = "patient_id"
risk_col = "risk_score"

label_col = None
for c in ["true_label", "label", "y_true", "deterioration_next_12h"]:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError("True label column not found.")

# Convert types
df[risk_col] = pd.to_numeric(df[risk_col], errors="coerce")
df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

# =============================
# Global metrics
# =============================
roc = roc_auc_score(df[label_col], df[risk_col])
pr = average_precision_score(df[label_col], df[risk_col])

# =============================
# Patient-level evaluation
# =============================
patient_max = df.groupby(pid_col)[risk_col].max().reset_index()
patient_y = df.groupby(pid_col)[label_col].max().reset_index()
pm = patient_max.merge(patient_y, on=pid_col)

threshold = 0.50
y_pred = (pm[risk_col] >= threshold).astype(int)
y_true = pm[label_col]

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

alert_patients = int(y_pred.sum())
total_patients = len(pm)

# =============================
# Create table data
# =============================
table_data = [
    ["ROC-AUC", f"{roc:.4f}"],
    ["PR-AUC", f"{pr:.4f}"],
    ["Precision @0.50", f"{prec:.4f}"],
    ["Recall @0.50", f"{rec:.4f}"],
    ["F1-score @0.50", f"{f1:.4f}"],
    ["Alert patients", f"{alert_patients} / {total_patients}"],
]

table_df = pd.DataFrame(table_data, columns=["Metric", "Value"])

# =============================
# Create single visual table
# =============================
plt.figure(figsize=(8, 4))
plt.axis("off")

table = plt.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Model Evaluation Summary (Patient-Level)", pad=20)
plt.tight_layout()
plt.savefig("model_evaluation_summary.png", dpi=300)
plt.show()

print("Saved: model_evaluation_summary.png")
