import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = "train_multicase_3sig.csv"
LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"


def main():
    df = pd.read_csv(CSV_PATH)

    # Features (exclude label, caseid, sec)
    feat_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]

    X = df[feat_cols].values
    y = df[LABEL_COL].values.astype(int)
    groups = df[GROUP_COL].values.astype(int)

    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Label counts:", np.bincount(y))

    # Group split by caseid (avoid leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # RandomForest baseline (strong for tabular features)
    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print("\nTest ROC-AUC:", auc)

    y_pred = (y_prob >= 0.5).astype(int)
    print("\nClassification report:\n", classification_report(y_te, y_pred))

    # Feature importance (top 10)
    importances = clf.feature_importances_
    top = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i in top:
        print(f"{feat_cols[i]:>15s}  {importances[i]:.4f}")


if __name__ == "__main__":
    main()
