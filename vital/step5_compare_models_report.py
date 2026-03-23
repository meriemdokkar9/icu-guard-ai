import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    best_thr = float(thr[i]) if i < len(thr) else 0.5

    return best_thr, float(f1[i]), float(prec[i]), float(rec[i])


def threshold_for_target_recall(y_true, y_prob, target_recall=0.80):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)

    idx = np.where(rec >= target_recall)[0]
    if len(idx) == 0:
        return None

    i = idx[-1]
    t = float(thr[i]) if i < len(thr) else 0.0

    return t, float(prec[i]), float(rec[i])


def build_preprocess(X):
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Logistic needs scaling
    num_pipe_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Trees do not need scaling
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess_scaled = ColumnTransformer([
        ("num", num_pipe_scaled, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    preprocess_tree = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocess_scaled, preprocess_tree


def eval_model(name, clf, X_te, y_te):
    y_prob = clf.predict_proba(X_te)[:, 1]

    roc_auc = roc_auc_score(y_te, y_prob)
    pr_auc = average_precision_score(y_te, y_prob)

    thr_f1, best_f1, p_f1, r_f1 = best_threshold_by_f1(y_te, y_prob)

    y_pred = (y_prob >= thr_f1).astype(int)

    print("\n" + "=" * 80)
    print(f"MODEL: {name}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"Best threshold (max F1): {thr_f1:.4f}")
    print(f"Best F1: {best_f1:.4f} | Precision: {p_f1:.4f} | Recall: {r_f1:.4f}")

    print("\nClassification report @ best F1 threshold:\n")
    print(classification_report(y_te, y_pred))

    target = threshold_for_target_recall(y_te, y_prob, target_recall=0.80)

    if target is None:
        print("No threshold achieves Recall >= 0.80")
        thr80 = None
        p80 = None
        r80 = None
    else:
        thr80, p80, r80 = target
        print(f"Operating point Recall>=0.80: threshold={thr80:.4f} | Precision={p80:.4f} | Recall={r80:.4f}")

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_thr_f1": thr_f1,
        "best_f1": best_f1,
        "precision_best_f1": p_f1,
        "recall_best_f1": r_f1,
        "thr_recall80": thr80,
        "precision_recall80": p80,
        "recall_recall80": r80
    }


def main():
    df = pd.read_csv(DATA_PATH)

    y = df[LABEL].astype(int).values
    groups = df[GROUP].values

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    print("Loaded:", df.shape)
    print("Label counts:", np.bincount(y))

    # =========================
    # GROUP SPLIT (patient split)
    # =========================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print("Train size:", X_tr.shape, "Test size:", X_te.shape)
    print("Train label:", np.bincount(y_tr), "Test label:", np.bincount(y_te))

    preprocess_scaled, preprocess_tree = build_preprocess(X_tr)

    # imbalance ratio for XGB
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    results = []

    # =========================
    # Logistic Regression
    # =========================
    logreg = Pipeline([
        ("prep", preprocess_scaled),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
    logreg.fit(X_tr, y_tr)
    results.append(eval_model("LogisticRegression", logreg, X_te, y_te))

    # =========================
    # Random Forest
    # =========================
    rf = Pipeline([
        ("prep", preprocess_tree),
        ("model", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        ))
    ])
    rf.fit(X_tr, y_tr)
    results.append(eval_model("RandomForest", rf, X_te, y_te))

    # =========================
    # XGBoost
    # =========================
    xgb = Pipeline([
        ("prep", preprocess_tree),
        ("model", XGBClassifier(
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
        ))
    ])
    xgb.fit(X_tr, y_tr)
    results.append(eval_model("XGBoost", xgb, X_te, y_te))

    report = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    report.to_csv("model_comparison_report.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print(report.to_string(index=False))

    print("\nSaved: model_comparison_report.csv")


if __name__ == "__main__":
    main()
