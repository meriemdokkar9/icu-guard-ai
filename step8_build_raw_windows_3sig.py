import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
THRESH = 90
WINDOW_MIN = 5
HORIZON_MIN = 30
STEP_SEC = 15

# نفس اللي عندك بالملف القديم (تأكدي منهم!)
TRKS_URL = "https://api.vitaldb.net/trks"          # جدول التراكات
API_TRACK = "https://api.vitaldb.net/{}"
     # مسار تحميل track (gzip csv)

CASEIDS_CSV = "hypoxemia_caseids.csv"             # نفس اللي عندك بالقديم
OUT_X = "X_raw_3sig.npy"
OUT_Y = "y_raw_3sig.npy"
OUT_CASEID = "caseid_raw_3sig.npy"
OUT_SEC = "sec_raw_3sig.npy"


def download_track(tid):
    # vitaldb يعيد csv مضغوط غالبًا
    return pd.read_csv(API_TRACK.format(tid), compression="gzip")


def to_1hz(df):
    tcol, vcol = df.columns[0], df.columns[1]
    x = df[[tcol, vcol]].copy()
    x[tcol] = pd.to_numeric(x[tcol], errors="coerce")
    x[vcol] = pd.to_numeric(x[vcol], errors="coerce")
    x = x.dropna()
    if len(x) < 10:
        return None
    x["sec"] = x[tcol].round().astype(int)
    y = x.groupby("sec")[vcol].mean().reset_index().sort_values("sec")
    y[vcol] = y[vcol].ffill()
    return y


def pick_tid(case_trks, keywords):
    pat = "|".join(keywords)
    m = case_trks["tname"].astype(str).str.contains(pat, case=False, na=False)
    if not m.any():
        return None
    return case_trks.loc[m, "tid"].iloc[0]


def build_one_case_raw_windows(caseid, trks_case):
    # 3 إشارات: SpO2 + RR + HR
    tid_spo2 = pick_tid(trks_case, ["SPO2", "SpO2", "PLETH_SPO2"])
    tid_rr   = pick_tid(trks_case, ["RESP", "RR"])
    tid_hr   = pick_tid(trks_case, ["HR", "ECG_HR", "HeartRate"])

    if None in [tid_spo2, tid_rr, tid_hr]:
        return [], [], [], []

    spo2 = to_1hz(download_track(tid_spo2))
    rr   = to_1hz(download_track(tid_rr))
    hr   = to_1hz(download_track(tid_hr))
    if any(x is None for x in [spo2, rr, hr]):
        return [], [], [], []

    spo2 = spo2.rename(columns={spo2.columns[1]: "spo2"})
    rr   = rr.rename(columns={rr.columns[1]: "rr"})
    hr   = hr.rename(columns={hr.columns[1]: "hr"})

    data = spo2.merge(rr, on="sec", how="inner").merge(hr, on="sec", how="inner")
    if len(data) < (WINDOW_MIN * 60 + 10):
        return [], [], [], []

    # Label: هل سيحدث SpO2 < 90 خلال 30 دقيقة القادمة؟
    data["is_low"] = (data["spo2"] < THRESH).astype(int)
    horizon_sec = HORIZON_MIN * 60
    future_max = data["is_low"].shift(-1)[::-1].rolling(window=horizon_sec, min_periods=1).max()[::-1]
data["label_next_30min"] = future_max.fillna(0).astype(int)


    # Build raw windows
 win = WINDOW_MIN * 60
    step = STEP_SEC

    X_list, y_list, caseid_list, sec_list = [], [], [], []

    # إشارة: (T,C)=(300,3)
    for end in range(win, len(data), step):
        start = end - win

        w_spo2 = data["spo2"].iloc[start:end].values
        w_rr   = data["rr"].iloc[start:end].values
        w_hr   = data["hr"].iloc[start:end].values

        if len(w_spo2) != win or len(w_rr) != win or len(w_hr) != win:
            continue

        X = np.stack([w_spo2, w_rr, w_hr], axis=1).astype(np.float32)
        y = int(data["label_next_30min"].iloc[end - 1])
        sec_end = int(data["sec"].iloc[end - 1])

        X_list.append(X)
        y_list.append(y)
        caseid_list.append(int(caseid))
        sec_list.append(sec_end)

    return X_list, y_list, caseid_list, sec_list


def main():
    trks = pd.read_csv(TRKS_URL)
    caseids = pd.read_csv(CASEIDS_CSV)["caseid"].tolist()

    X_all, y_all, caseid_all, sec_all = [], [], [], []

    for i, cid in enumerate(caseids, 1):
        print(f"[{i}/{len(caseids)}] caseid={cid}")
        case_trks = trks[trks["caseid"] == cid]

        try:
            X_list, y_list, c_list, s_list = build_one_case_raw_windows(cid, case_trks)
            if len(X_list) > 0:
                print("  windows:", len(X_list), " label1:", int(np.sum(y_list)))
                X_all.extend(X_list)
                y_all.extend(y_list)
                caseid_all.extend(c_list)
                sec_all.extend(s_list)
            else:
                print("  skipped (missing/short/empty)")
        except Exception as e:
            print("  failed:", e)

    if len(X_all) == 0:
        raise SystemExit("No windows built. Check track keywords, API urls, and caseids file.")

    X_all = np.stack(X_all, axis=0)  # (N,300,3)
    y_all = np.array(y_all, dtype=np.int64)
    caseid_all = np.array(caseid_all, dtype=np.int64)
    sec_all = np.array(sec_all, dtype=np.int64)

    np.save(OUT_X, X_all)
    np.save(OUT_Y, y_all)
    np.save(OUT_CASEID, caseid_all)
    np.save(OUT_SEC, sec_all)

    print("\nSaved raw windows:")
    print("X:", X_all.shape, "y:", y_all.shape)
    print("Label counts:", np.bincount(y_all))


if __name__ == "__main__":
    main()


