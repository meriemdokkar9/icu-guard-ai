import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("train_multicase_3sig.csv")

feature_cols = [
    "spo2_mean_5m", "spo2_min_5m", "spo2_std_5m", "spo2_slope_5m",
    "rr_mean_5m", "rr_min_5m", "rr_std_5m", "rr_slope_5m",
    "hr_mean_5m", "hr_min_5m", "hr_std_5m", "hr_slope_5m"
]

# --- Group split by caseid ---
caseids = df["caseid"].unique()
rng = np.random.RandomState(42)
rng.shuffle(caseids)

test_frac = 0.2
n_test = max(1, int(len(caseids) * test_frac))
test_caseids = set(caseids[:n_test])

train_df = df[~df["caseid"].isin(test_caseids)].copy()
test_df  = df[df["caseid"].isin(test_caseids)].copy()

X_train = train_df[feature_cols]
y_train = train_df["label_next_30min"]

X_test = test_df[feature_cols]
y_test = test_df["label_next_30min"]

print("Train cases:", train_df["caseid"].nunique(), "Test cases:", test_df["caseid"].nunique())
print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Test label distribution:\n", y_test.value_counts())

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Group-split Results (by caseid) ===\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", auc)

