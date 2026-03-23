import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================
# 1) Load data
# ============================
CSV_PATH = "hospital_deterioration_hourly_panel.csv"
TARGET = "deterioration_next_12h"

df = pd.read_csv(CSV_PATH)
assert TARGET in df.columns, f"Target column '{TARGET}' not found."

# Keep patient_id for overlap proof
assert "patient_id" in df.columns, "patient_id column is required for overlap proof."

# Drop non-features
drop_cols = [TARGET]
if "hour_from_admission" in df.columns:
    drop_cols.append("hour_from_admission")

X_full = df.drop(columns=drop_cols)
y_full = df[TARGET].astype(int)
patient_ids = df["patient_id"].astype(int)

# Features only (remove patient_id from features)
X = X_full.drop(columns=["patient_id"])

print("Loaded:", df.shape)
print("X:", X.shape, "y counts:", np.bincount(y_full))


# ============================
# 2) Preprocess (OneHot for object columns)
# ============================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

# Helper to train + evaluate
def train_eval(split_name, X_train, X_test, y_train, y_test, train_pat_ids=None, test_pat_ids=None):
    # scale_pos_weight like Kaggle
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = (neg / pos) if pos > 0 else 1.0

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

    p1 = rep["1"]["precision"]
    r1 = rep["1"]["recall"]
    f1 = rep["1"]["f1-score"]

    overlap = None
    if train_pat_ids is not None and test_pat_ids is not None:
        overlap = len(set(train_pat_ids) & set(test_pat_ids))

    return {
        "Split": split_name,
        "Patient overlap": overlap if overlap is not None else "",
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Precision (Class 1)": p1,
        "Recall (Class 1)": r1,
        "F1 (Class 1)": f1,
        "scale_pos_weight": spw
    }


# ============================
# 3) Kaggle style split (Random Row + Stratify)
# ============================
X_tr_k, X_te_k, y_tr_k, y_te_k, pid_tr_k, pid_te_k = train_test_split(
    X, y_full, patient_ids,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

res_kaggle = train_eval(
    "Kaggle style (Random Row Split)",
    X_tr_k, X_te_k, y_tr_k, y_te_k,
    train_pat_ids=pid_tr_k, test_pat_ids=pid_te_k
)

print("\n[Kaggle style] patient overlap:", res_kaggle["Patient overlap"])


# ============================
# 4) Realistic split (GroupShuffleSplit by patient_id)
# ============================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_full, groups=patient_ids))

X_tr_g, X_te_g = X.iloc[train_idx], X.iloc[test_idx]
y_tr_g, y_te_g = y_full.iloc[train_idx], y_full.iloc[test_idx]
pid_tr_g, pid_te_g = patient_ids.iloc[train_idx], patient_ids.iloc[test_idx]

res_group = train_eval(
    "Realistic (GroupSplit by patient_id)",
    X_tr_g, X_te_g, y_tr_g, y_te_g,
    train_pat_ids=pid_tr_g, test_pat_ids=pid_te_g
)

print("[Group split] patient overlap:", res_group["Patient overlap"])


# ============================
# 5) Build proof table + save image
# ============================
results_df = pd.DataFrame([res_kaggle, res_group])

# Add delta row
delta = {
    "Split": "Difference (Kaggle - Realistic)",
    "Patient overlap": (res_kaggle["Patient overlap"] - res_group["Patient overlap"]),
}
for col in ["ROC-AUC","PR-AUC","Precision (Class 1)","Recall (Class 1)","F1 (Class 1)"]:
    delta[col] = res_kaggle[col] - res_group[col]
delta["scale_pos_weight"] = ""

results_df = pd.concat([results_df, pd.DataFrame([delta])], ignore_index=True)

# Round display
disp = results_df.copy()
for c in ["ROC-AUC","PR-AUC","Precision (Class 1)","Recall (Class 1)","F1 (Class 1)"]:
    disp[c] = disp[c].apply(lambda v: "" if v == "" else f"{float(v):.3f}")
disp["Patient overlap"] = disp["Patient overlap"].apply(lambda v: "" if v == "" else str(int(v)))

# Save CSV too
results_df.to_csv("proof_kaggle_vs_realistic.csv", index=False)

# Make table PNG
fig, ax = plt.subplots(figsize=(14, 2.8))
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

# Highlight overlap column cells for first 2 rows
# header row is 0 in table object, data starts at row 1
overlap_col_idx = list(disp.columns).index("Patient overlap")
for r in [1, 2]:
    cell = tbl[(r, overlap_col_idx)]
    cell.set_text_props(weight="bold")

# Bold delta row
delta_row = 3
for c in range(len(disp.columns)):
    tbl[(delta_row, c)].set_text_props(weight="bold")
ax.set_title("XGBoost Evaluation Comparison (Kaggle Split vs Patient-Level Split)", pad=14)
plt.tight_layout()
plt.savefig("proof_table_kaggle_vs_realistic.png", dpi=300)
plt.show()

print("\nSaved:")
print(" - proof_table_kaggle_vs_realistic.png")
print(" - proof_kaggle_vs_realistic.csv")
