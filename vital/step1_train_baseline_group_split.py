import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATA
# =========================
DATA_PATH = "hospital_deterioration_hourly_panel.csv"

df = pd.read_csv(DATA_PATH)

print("Loaded:", df.shape)
print("Label counts:\n", df["deterioration_next_12h"].value_counts())

# =========================
# TARGET + GROUP
# =========================
y = df["deterioration_next_12h"].astype(int).values
groups = df["patient_id"].values

# Drop columns we don't want as features
drop_cols = [
    "patient_id",
    "deterioration_next_12h",
    "deterioration_event",
    "deterioration_hour",
    "deterioration_within_12h_from_admission"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# =========================
# FEATURE TYPES
# =========================
cat_cols = ["oxygen_device", "gender", "admission_type"]
num_cols = [c for c in X.columns if c not in cat_cols]

# =========================
# PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# =========================
# MODEL
# =========================
model = LogisticRegression(max_iter=2000, class_weight="balanced")

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# =========================
# GROUP SPLIT
# =========================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train label counts:", np.bincount(y_train))
print("Test label counts:", np.bincount(y_test))

# =========================
# TRAIN
# =========================
clf.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("\nTest ROC-AUC:", auc)

y_pred = (y_prob >= 0.5).astype(int)
print("\nClassification report:\n", classification_report(y_test, y_pred))
