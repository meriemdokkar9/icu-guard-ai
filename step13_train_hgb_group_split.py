import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.ensemble import HistGradientBoostingClassifier


CSV_PATH = "train_multicase_3sig.csv"
LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"


def main():
    df = pd.read_csv(CSV_PATH)

    feat_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]

    X = df[feat_cols].values
    y = df[LABEL_COL].values.astype(int)
    groups = df[GROUP_COL].values.astype(int)

    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Label counts:", np.bincount(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )

    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print("\nTest ROC-AUC:", auc)

    # Find best threshold by F1
    prec, rec, thr = precision_recall_curve(y_te, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    best_i = np.argmax(f1)
    best_thr = thr[best_i] if best_i < len(thr) else 0.5

    print("\nBest threshold:", best_thr)
    print("Best F1:", f1[best_i])
    print("Precision:", prec[best_i], "Recall:", rec[best_i])

    y_pred = (y_prob >= best_thr).astype(int)
    print("\nClassification report:\n", classification_report(y_te, y_pred))


if __name__ == "__main__":
    main()
