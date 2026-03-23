import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================
# Load Data
# ============================
df = pd.read_csv("hospital_deterioration_hourly_panel.csv")
TARGET = "deterioration_next_12h"

# patient_id required
assert "patient_id" in df.columns

# Drop non-features
drop_cols = [TARGET]
if "hour_from_admission" in df.columns:
    drop_cols.append("hour_from_admission")

X_full = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)
groups = df["patient_id"].astype(int)

# Remove patient_id from features
X = X_full.drop(columns=["patient_id"])

print("Loaded:", df.shape)
print("Features:", X.shape)
print("Label counts:", np.bincount(y))


# ============================
# Preprocessing (OneHot categorical)
# ============================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# ============================
# GroupKFold Evaluation
# ============================
gkf = GroupKFold(n_splits=5)

fold_results = []

fold_num = 1
for train_idx, test_idx in gkf.split(X, y, groups=groups):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    overlap = len(train_patients & test_patients)

    # scale_pos_weight per fold
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos

    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("xgb", model)
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    yhat = (proba >= 0.5).astype(int)
    rep = classification_report(y_test, yhat, output_dict=True, zero_division=0)

    fold_results.append({
        "Fold": fold_num,
        "Patient overlap": overlap,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Precision (Class 1)": rep["1"]["precision"],
        "Recall (Class 1)": rep["1"]["recall"],
        "F1 (Class 1)": rep["1"]["f1-score"],
        "scale_pos_weight": spw
    })

    print(f"\nFold {fold_num} done | ROC={roc:.4f} PR={pr:.4f} Overlap={overlap}")
    fold_num += 1


# ============================
# Results Summary
# ============================
df_res = pd.DataFrame(fold_results)

mean_row = {
    "Fold": "MEAN",
    "Patient overlap": df_res["Patient overlap"].mean(),
    "ROC-AUC": df_res["ROC-AUC"].mean(),
    "PR-AUC": df_res["PR-AUC"].mean(),
    "Precision (Class 1)": df_res["Precision (Class 1)"].mean(),
    "Recall (Class 1)": df_res["Recall (Class 1)"].mean(),
    "F1 (Class 1)": df_res["F1 (Class 1)"].mean(),
    "scale_pos_weight": df_res["scale_pos_weight"].mean()
}

std_row = {
    "Fold": "STD",
    "Patient overlap": df_res["Patient overlap"].std(),
    "ROC-AUC": df_res["ROC-AUC"].std(),
    "PR-AUC": df_res["PR-AUC"].std(),
    "Precision (Class 1)": df_res["Precision (Class 1)"].std(),
    "Recall (Class 1)": df_res["Recall (Class 1)"].std(),
    "F1 (Class 1)": df_res["F1 (Class 1)"].std(),
    "scale_pos_weight": df_res["scale_pos_weight"].std()
}

df_summary = pd.concat([df_res, pd.DataFrame([mean_row, std_row])], ignore_index=True)

print("\n=== GroupKFold (5-fold) Summary ===\n")
print(df_summary.to_string(index=False))

df_summary.to_csv("xgb_groupkfold_5fold_report.csv", index=False)
print("\nSaved: xgb_groupkfold_5fold_report.csv")


# ============================
# Save summary table as image
# ============================
disp = df_summary.copy()

for c in ["ROC-AUC","PR-AUC","Precision (Class 1)","Recall (Class 1)","F1 (Class 1)","scale_pos_weight"]:
    disp[c] = disp[c].apply(lambda v: "" if v == "" else f"{float(v):.4f}")

disp["Patient overlap"] = disp["Patient overlap"].apply(lambda v: "" if v == "" else str(int(float(v))))

fig, ax = plt.subplots(figsize=(14, 3.6))
ax.axis("off")

tbl = ax.table(
    cellText=disp.values,
    colLabels=disp.columns,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

plt.title("XGBoost Patient-Level Evaluation (5-Fold GroupKFold CV)", pad=15)
plt.tight_layout()
plt.savefig("xgb_groupkfold_5fold_table.png", dpi=300)
plt.show()

print("Saved: xgb_groupkfold_5fold_table.png")
