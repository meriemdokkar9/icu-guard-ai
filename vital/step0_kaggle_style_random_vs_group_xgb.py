import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

DATA_PATH = "hospital_deterioration_hourly_panel.csv"

PID = "patient_id"
TIME = "hour_from_admission"
LABEL = "deterioration_next_12h"

# columns that leak / not allowed
DROP_COLS = [
    PID,
    "deterioration_event",
    "deterioration_hour",
    "deterioration_within_12h_from_admission",
]

CAT_COLS = ["oxygen_device", "gender", "admission_type"]

def make_features(df: pd.DataFrame):
    df = df.copy()

    # keep label separately
    y = df[LABEL].astype(int).values

    # drop leakage columns + label itself later
    drop_exist = [c for c in DROP_COLS if c in df.columns]
    Xdf = df.drop(columns=drop_exist, errors="ignore")

    # remove label from features if exists
    Xdf = Xdf.drop(columns=[LABEL], errors="ignore")

    # one-hot categorical
    cat_exist = [c for c in CAT_COLS if c in Xdf.columns]
    if cat_exist:
        Xcat = pd.get_dummies(Xdf[cat_exist], dummy_na=True)
        Xnum = Xdf.drop(columns=cat_exist, errors="ignore")
        Xdf = pd.concat([Xnum.reset_index(drop=True), Xcat.reset_index(drop=True)], axis=1)

    # ensure numeric
    for c in Xdf.columns:
        if not pd.api.types.is_numeric_dtype(Xdf[c]):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # fill NaNs with median
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)

    return Xdf.values.astype(np.float32), y, list(Xdf.columns)

def train_eval(Xtr, ytr, Xte, yte, title=""):
    # (اختياري) scaling — يساعد موديلات linear أكثر، بس نخليه حتى يكون “مثل كاجل”
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(Xtr_s, ytr)

    prob = model.predict_proba(Xte_s)[:, 1]
    roc = roc_auc_score(yte, prob)
    pr = average_precision_score(yte, prob)

    print(f"\n=== {title} ===")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)

    # threshold default 0.5 just to show report
    pred = (prob >= 0.5).astype(int)
    print("\nReport @ thr=0.5:\n", classification_report(yte, pred, digits=4))
    return roc, pr

def main():
    df = pd.read_csv(DATA_PATH)

    # IMPORTANT: sort not required for ML, but clean
    if PID in df.columns and TIME in df.columns:
        df = df.sort_values([PID, TIME]).reset_index(drop=True)

    print("Loaded:", df.shape)
    print("Label counts:", np.bincount(df[LABEL].astype(int)))

    # build X,y
    X, y, feat_cols = make_features(df)
    print("Features:", len(feat_cols))

    # ---------------------------
    # 1) Kaggle-style Random split (ROW split)  ✅ (leakage risk)
    # ---------------------------
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_eval(Xtr, ytr, Xte, yte, title="KAGGLE STYLE (Random ROW split)")

    # ---------------------------
    # 2) Realistic split (Group split by patient_id) ✅
    # ---------------------------
    groups = df[PID].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    Xtr2, ytr2 = X[tr_idx], y[tr_idx]
    Xte2, yte2 = X[te_idx], y[te_idx]

    # show leakage proof for random split (how many patients overlap)
    # (في group split المفروض overlap = 0)
    pid_tr = set(df.iloc[tr_idx][PID].unique())
    pid_te = set(df.iloc[te_idx][PID].unique())
    overlap = len(pid_tr.intersection(pid_te))
    print("\n[Group split check] patient overlap:", overlap)

    train_eval(Xtr2, ytr2, Xte2, yte2, title="REALISTIC (Group split by patient_id)")

if __name__ == "__main__":
    main()
