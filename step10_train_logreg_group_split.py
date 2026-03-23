import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

CSV_PATH = "train_multicase_3sig.csv"
LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"

def main():
    df = pd.read_csv(CSV_PATH)

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]
    X = df[feature_cols].values
    y = df[LABEL_COL].values
    groups = df[GROUP_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print("Test ROC-AUC:", auc)

    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
