import pandas as pd
import numpy as np

TRKS_URL = "https://api.vitaldb.net/trks"
API_TRACK = "https://api.vitaldb.net/{}"

FS = 1
WINDOW_MIN = 5
HORIZON_MIN = 30
THRESH = 90

def download_track(tid):
    url = API_TRACK.format(tid)
    return pd.read_csv(url, compression="gzip")

def to_grid(df):
    # أول عمود زمن، ثاني عمود قيمة
    tcol, vcol = df.columns[0], df.columns[1]
    df = df[[tcol, vcol]].dropna().copy()
    df.columns = ["t", "v"]
    df = df.sort_values("t")
    t0, t1 = float(df["t"].min()), float(df["t"].max())
    tgrid = np.arange(np.floor(t0), np.ceil(t1) + 1, 1.0/FS)
    s = pd.Series(df["v"].values, index=df["t"].values)
    s = s[~s.index.duplicated(keep="last")]
    sgrid = s.reindex(tgrid, method="ffill")
    out = pd.DataFrame({"t": tgrid, "v": sgrid.values}).dropna()
    return out

def pick_tid(case_trks, keywords):
    # يرجع أول tid يطابق الكلمات
    pat = "|".join(keywords)
    m = case_trks["tname"].astype(str).str.contains(pat, case=False, na=False)
    if not m.any():
        return None
    return case_trks.loc[m, "tid"].iloc[0]

def build_one_case(caseid, trks_case):
    # اختر tids
    tid_spo2 = pick_tid(trks_case, ["SPO2", "SpO2", "PLETH_SPO2"])
    tid_rr   = pick_tid(trks_case, ["RESP", "RR"])
    tid_etco2= pick_tid(trks_case, ["ETCO2", "CO2"])
    tid_hr   = pick_tid(trks_case, ["HR", "ECG_HR", "HeartRate"])

    if None in [tid_spo2, tid_rr, tid_etco2, tid_hr]:
        return None

    # نزّل و اعمل grid
    spo2 = to_grid(download_track(tid_spo2)).rename(columns={"v":"spo2"})
    rr   = to_grid(download_track(tid_rr)).rename(columns={"v":"rr"})
    etco2= to_grid(download_track(tid_etco2)).rename(columns={"v":"etco2"})
    hr   = to_grid(download_track(tid_hr)).rename(columns={"v":"hr"})

    # دمج على الزمن (inner join عشان يكون عندنا نفس اللحظات)
    data = spo2.merge(rr, on="t", how="inner").merge(etco2, on="t", how="inner").merge(hr, on="t", how="inner")
    data["caseid"] = caseid

    # label من spo2
    data["is_low"] = (data["spo2"] < THRESH).astype(int)
    horizon_sec = HORIZON_MIN * 60
    future_max = (data["is_low"][::-1].rolling(window=horizon_sec, min_periods=1).max()[::-1])
    data["label_next_30min"] = future_max.astype(int)

    # features (آخر 5 دقائق) لكل إشارة
    win_sec = WINDOW_MIN * 60
    for col in ["spo2","rr","etco2","hr"]:
        data[f"{col}_mean_5m"] = data[col].rolling(win_sec, min_periods=win_sec).mean()
        data[f"{col}_min_5m"]  = data[col].rolling(win_sec, min_periods=win_sec).min()
        data[f"{col}_std_5m"]  = data[col].rolling(win_sec, min_periods=win_sec).std()
        data[f"{col}_slope_5m"]= (data[col] - data[col].shift(win_sec)) / win_sec

    feat_cols = []
    for col in ["spo2","rr","etco2","hr"]:
        feat_cols += [f"{col}_mean_5m", f"{col}_min_5m", f"{col}_std_5m", f"{col}_slope_5m"]

    out = data.dropna(subset=feat_cols).copy()
    keep = ["caseid","t"] + feat_cols + ["label_next_30min"]
    return out[keep]

# --- main ---
trks = pd.read_csv(TRKS_URL)
eligible = pd.read_csv("eligible_caseids.csv")
caseids = eligible["caseid"].tolist()

all_rows = []
for i, cid in enumerate(caseids, 1):
    print(f"[{i}/{len(caseids)}] caseid={cid}")
    case_trks = trks[trks["caseid"] == cid]
    try:
        one = build_one_case(cid, case_trks)
        if one is not None and len(one) > 0:
            all_rows.append(one)
            print("  rows:", len(one), " label1:", int(one["label_next_30min"].sum()))
        else:
            print("  skipped (missing/empty)")
    except Exception as e:
        print("  failed:", e)

if not all_rows:
    raise SystemExit("No cases built. Try increasing head() in eligible list or relax keywords.")

final = pd.concat(all_rows, ignore_index=True)
final.to_csv("train_multicase.csv", index=False)

print("\nSaved: train_multicase.csv")
print("Total rows:", len(final))
print("Label distribution:\n", final["label_next_30min"].value_counts())
