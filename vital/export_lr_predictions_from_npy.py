import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", default="X_seq_24h.npy")
    ap.add_argument("--y", default="y_seq_24h.npy")
    ap.add_argument("--pid", default="patient_seq_24h.npy")
    ap.add_argument("--hour", default="hour_seq_24h.npy")
    ap.add_argument("--out_csv", default="lr_hourly_predictions_from_npy.csv")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading NPY...")
    X = np.load(args.X).astype(np.float32)        # (N, 24, 33)
    y = np.load(args.y).astype(int).reshape(-1)   # (N,)
    pid = np.load(args.pid).reshape(-1)           # (N,)
    hour = np.load(args.hour).reshape(-1)         # (N,)

    print("Shapes:", X.shape, y.shape, pid.shape, hour.shape)

    # Baseline LR features: last timestep only (N, 33)
    X_last = X[:, -1, :]

    # Group split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    idx = np.arange(len(y))
    train_idx, test_idx = next(gss.split(idx, y, groups=pid))

    print("Train rows:", len(train_idx), "Test rows:", len(test_idx))
    print("Train patients:", len(np.unique(pid[train_idx])), "Test patients:", len(np.unique(pid[test_idx])))

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_last[train_idx])
    Xte = scaler.transform(X_last[test_idx])

    # class_weight to handle imbalance
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1
    )
    lr.fit(Xtr, y[train_idx])

    p_te = lr.predict_proba(Xte)[:, 1]

    auroc = roc_auc_score(y[test_idx], p_te)
    auprc = average_precision_score(y[test_idx], p_te)
    print(f"[Sanity on held-out patients] AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    # Produce predictions for ALL rows (train+test) for streaming simulation
    p_all = lr.predict_proba(scaler.transform(X_last))[:, 1].astype(np.float32)

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