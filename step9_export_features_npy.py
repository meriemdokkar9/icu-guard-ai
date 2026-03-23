import numpy as np
import pandas as pd

CSV_PATH = "train_multicase_3sig.csv"

OUT_X = "X_feat_3sig.npy"
OUT_Y = "y_feat_3sig.npy"
OUT_CASEID = "caseid_feat_3sig.npy"
OUT_SEC = "sec_feat_3sig.npy"

LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"

def main():
    df = pd.read_csv(CSV_PATH)

    # تأكد الأعمدة موجودة
    for c in [LABEL_COL, GROUP_COL, "sec"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]

    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)
    caseid = df[GROUP_COL].values.astype(np.int64)
    sec = df["sec"].values.astype(np.int64)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    np.save(OUT_CASEID, caseid)
    np.save(OUT_SEC, sec)

    print("Saved:")
    print("X:", OUT_X, X.shape)
    print("y:", OUT_Y, y.shape, "labels:", np.bincount(y))
    print("caseid:", OUT_CASEID, caseid.shape)
    print("sec:", OUT_SEC, sec.shape)

if __name__ == "__main__":
    main()
