import numpy as np
import pandas as pd

DATA_PATH = "hospital_deterioration_hourly_panel.csv"

# Outputs
OUT_X = "X_seq_24h.npy"
OUT_Y = "y_seq_24h.npy"
OUT_G = "patient_seq_24h.npy"
OUT_H = "hour_seq_24h.npy"

LABEL = "deterioration_next_12h"
PID = "patient_id"
TIME = "hour_from_admission"

WINDOW_HOURS = 24
CAT_COLS = ["oxygen_device", "gender", "admission_type"]

DROP_COLS = [
    PID, LABEL,
    "deterioration_event",
    "deterioration_hour",
    "deterioration_within_12h_from_admission"
]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values([PID, TIME]).reset_index(drop=True)

    print("Loaded:", df.shape)

    # One-hot encode categorical columns
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    if len(cat_cols) > 0:
        df_cat = pd.get_dummies(df[cat_cols], dummy_na=True)
        df_num = df.drop(columns=cat_cols)
    else:
        df_cat = None
        df_num = df.copy()

    # Drop leakage columns
    drop_cols_exist = [c for c in DROP_COLS if c in df_num.columns]
    X_base = df_num.drop(columns=drop_cols_exist)

    # Merge numeric + onehot
    if df_cat is not None:
        X_full = pd.concat([X_base.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    else:
        X_full = X_base

    feature_cols = X_full.columns.tolist()
    print("Total feature columns:", len(feature_cols))

    # Build final dataframe for windowing
    df_feat = pd.concat(
        [df[[PID, TIME, LABEL]].reset_index(drop=True), X_full.reset_index(drop=True)],
        axis=1
    )

    # (Optional safety) remove any duplicated columns
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

    # Fill missing values safely (column-by-column)
    num_cols = df_feat[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        m = df_feat[c].median()
        df_feat[c] = df_feat[c].fillna(m)

    # any remaining NaNs -> 0
    df_feat[feature_cols] = df_feat[feature_cols].fillna(0)

    X_list, y_list, g_list, h_list = [], [], [], []

    # Group by patient
    for pid, g in df_feat.groupby(PID):
        g = g.sort_values(TIME)

        Xmat = g[feature_cols].values.astype(np.float32)
        yvec = g[LABEL].values.astype(np.int64)
        hours = g[TIME].values.astype(np.int64)

        if len(g) < WINDOW_HOURS:
            continue

        for i in range(WINDOW_HOURS - 1, len(g)):
            X_win = Xmat[i - WINDOW_HOURS + 1:i + 1]  # (24, F)
            y_lab = yvec[i]
            hr = hours[i]

            X_list.append(X_win)
            y_list.append(y_lab)
            g_list.append(pid)
            h_list.append(hr)

    if len(X_list) == 0:
        print("No sequences built. Check window size or dataset.")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list, dtype=np.int64)
    h = np.array(h_list, dtype=np.int64)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    np.save(OUT_G, g)
    np.save(OUT_H, h)

    print("\nSaved sequences:")
    print("X:", X.shape)
    print("y:", y.shape, "label counts:", np.bincount(y))
    print("groups:", g.shape, "unique patients:", len(np.unique(g)))
    print("hour:", h.shape)


if __name__ == "__main__":
    main()


