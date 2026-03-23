import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


DATA_PATH = "hospital_deterioration_hourly_panel.csv"
LABEL = "deterioration_next_12h"
GROUP = "patient_id"

DROP_COLS = [
    GROUP, LABEL,
    "deterioration_event",
    "deterioration_hour",
    "deterioration_within_12h_from_admission"
]

CAT_COLS = ["oxygen_device", "gender", "admission_type"]


def best_threshold_by_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    i = int(np.argmax(f1))
    best_thr = thr[i] if i < len(thr) else 0.5
    return float(best_thr), float(f1[i]), float(prec[i]), float(rec[i])


def main():
    df = pd.read_csv(DATA_PATH)

    y = df[LABEL].astype(int).values
    groups = df[GROUP].values

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print("Loaded:", df.shape)
    print("Label counts:", np.bincount(y))

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ],
        remainder="drop"
    )

    # imbalance
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight
    )

    clf = Pipeline([("prep", preprocess), ("model", xgb)])
    clf.fit(X_tr, y_tr)

    # Evaluate
    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print("\nTest ROC-AUC:", auc)

    thr, f1, p, r = best_threshold_by_f1(y_te, y_prob)
    print("\nBest threshold (max F1):", thr)
    print("Best F1:", f1, "Precision:", p, "Recall:", r)

    y_pred = (y_prob >= thr).astype(int)
    print("\nClassification report (best threshold):\n", classification_report(y_te, y_pred))

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    # Get feature names after preprocessing (one-hot expands cat features)
    prep = clf.named_steps["prep"]
    model = clf.named_steps["model"]

    # numeric names
    num_names = num_cols

    # categorical onehot names
    cat_names = []
    if len(cat_cols) > 0:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()

    feat_names = num_names + cat_names
    importances = model.feature_importances_

    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop 20 important features:")
    print(fi.head(20).to_string(index=False))

    fi.to_csv("xgb_feature_importance.csv", index=False)
    print("\nSaved: xgb_feature_importance.csv")

    # =========================
    # SAVE MODEL + THRESHOLD
    # =========================
    joblib.dump(clf, "xgb_deterioration_model.joblib")
    with open("xgb_best_threshold.txt", "w", encoding="utf-8") as f:
        f.write(str(thr))

    print("Saved: xgb_deterioration_model.joblib")
    print("Saved: xgb_best_threshold.txt")


if __name__ == "__main__":
    main()
