import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# تحميل Dataset الجديد
df = pd.read_csv("train_multicase_3sig.csv")

# Features (SpO2 + RR + HR)
feature_cols = [
    "spo2_mean_5m", "spo2_min_5m", "spo2_std_5m", "spo2_slope_5m",
    "rr_mean_5m", "rr_min_5m", "rr_std_5m", "rr_slope_5m",
    "hr_mean_5m", "hr_min_5m", "hr_std_5m", "hr_slope_5m"
]

X = df[feature_cols]
y = df["label_next_30min"]

# تقسيم Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تدريب موديل قوي
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# توقع
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# تقييم
print("\n=== Multi-signal Model Results ===\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", auc)
