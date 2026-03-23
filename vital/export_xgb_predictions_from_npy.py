import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

def try_train_xgb(Xtr, ytr, Xte):
    """
    Try XGBoost if installed; otherwise fallback to sklearn HistGradientBoostingClassifier.
    Returns: p_te, p_all_fn (callable to predict_proba on any X)
    """
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=4,
            random_state=42,
        )
        # scale_pos_weight for imbalance
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        if pos > 0:
            clf.set_params(scale_pos_weight=float(neg / pos))
        clf.fit(Xtr, ytr)
        p_te = clf.predict_proba(Xte)[:, 1]

        def p_all_fn(X):
            return clf.predict_proba(X)[:, 1]

        model_name = "xgboost"
        return p_te, p_all_fn, model_name

    except Exception as e:
        print("[WARN] xgboost not available or failed, fallback to sklearn HGB. Reason:", str(e))
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            random_state=42
        )
        clf.fit(Xtr, ytr)
        p_te = clf.predict_proba(Xte)[:, 1]

        def p_all_fn(X):
            return clf.predict_proba(X)[:, 1]

        model_name = "sklearn_hgb"
        return p_te, p_all_fn, model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", default="X_seq_24h.npy")
    ap.add_argument("--y", default="y_seq_24h.npy")
    ap.add_argument("--pid", default="patient_seq_24h.npy")
    ap.add_argument("--hour", default="hour_seq_24h.npy")
    ap.add_argument("--out_csv", default="xgb_hourly_predictions_from_npy.csv")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading NPY...")
    X = np.load(args.X).astype(np.float32)        # (N, 24, 33)
    y = np.load(args.y).astype(int).reshape(-1)   # (N,)
    pid = np.load(args.pid).reshape(-1)
    hour = np.load(args.hour).reshape(-1)

    print("Shapes:", X.shape, y.shape, pid.shape, hour.shape)

    # Simple strong baseline: last timestep features
    X_last = X[:, -1, :]

    # Group split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    idx = np.arange(len(y))
    train_idx, test_idx = next(gss.split(idx, y, groups=pid))

    Xtr, ytr = X_last[train_idx], y[train_idx]
    Xte, yte = X_last[test_idx], y[test_idx]

    print("Train patients:", len(np.unique(pid[train_idx])), "| Test patients:", len(np.unique(pid[test_idx])))

    p_te, p_all_fn, model_name = try_train_xgb(Xtr, ytr, Xte)

    auroc = roc_auc_score(yte, p_te)
    auprc = average_precision_score(yte, p_te)
    print(f"[{model_name} sanity] AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    p_all = p_all_fn(X_last).astype(np.float32)

    out = pd.DataFrame({
        "patient_id": pid.astype(int),
        "hour_from_admission": hour.astype(int),
        "true_label": y.astype(int),
        "risk_score": p_all
    })
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(out.head())


if __name__ == "__main__":
    main()