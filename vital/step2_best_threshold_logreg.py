import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "hospital_deterioration_hourly_panel.csv"
LABEL = "deterioration_next_12h"
GROUP = "patient_id"

def main():
    df = pd.read_csv(DATA_PATH)

    y = df[LABEL].astype(int).values
    groups = df[GROUP].values

    drop_cols = [
        GROUP, LABEL,
        "deterioration_event",
        "deterioration_hour",
        "deterioration_within_12h_from_admission"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = ["oxygen_device", "gender", "admission_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
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

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    )

    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print("Test ROC-AUC:", auc)

    prec, rec, thr = precision_recall_curve(y_te, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    best_i = np.argmax(f1)
    best_thr = thr[best_i] if best_i < len(thr) else 0.5

    print("\nBest threshold (max F1):", best_thr)
    print("Best F1:", f1[best_i])
    print("Precision:", prec[best_i], "Recall:", rec[best_i])

    y_pred = (y_prob >= best_thr).astype(int)
    print("\nClassification report (best threshold):\n", classification_report(y_te, y_pred))

if __name__ == "__main__":
    main()
