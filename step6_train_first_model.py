import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# تحميل جدول التدريب
df = pd.read_csv("train_table_one_track.csv")

# Features و Label
X = df[["feat_mean_5m", "feat_min_5m", "feat_std_5m", "feat_slope_5m"]]
y = df["label_next_30min"]

# تقسيم Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تدريب موديل بسيط
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# توقع
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# تقييم
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", auc)
