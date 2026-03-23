import pandas as pd
import numpy as np

TRKS_URL = "https://api.vitaldb.net/trks"
API_TRACK = "https://api.vitaldb.net/{}"

WINDOW_MIN = 5
HORIZON_MIN = 10
THRESH = 90

def download_track(tid):
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

def build_one_case(caseid, trks_case):
    # ✅ فقط 3 إشارات: SpO2 + RR + HR
    tid_spo2 = pick_tid(trks_case, ["SPO2", "SpO2", "PLETH_SPO2"])
    tid_rr   = pick_tid(trks_case, ["RESP", "RR"])
    tid_hr   = pick_tid(trks_case, ["HR", "ECG_HR", "HeartRate"])

    if None in [tid_spo2, tid_rr, tid_hr]:
        return None

    spo2 = to_1hz(download_track(tid_spo2))
    rr   = to_1hz(download_track(tid_rr))
    hr   = to_1hz(download_track(tid_hr))
    if any(x is None for x in [spo2, rr, hr]):
        return None

    spo2 = spo2.rename(columns={spo2.columns[1]: "spo2"})
    rr   = rr.rename(columns={rr.columns[1]: "rr"})
    hr   = hr.rename(columns={hr.columns[1]: "hr"})

    data = spo2.merge(rr, on="sec", how="inner").merge(hr, on="sec", how="inner")
    if len(data) < (WINDOW_MIN * 60 + 10):
        return None

    data["caseid"] = caseid

    # Label من SpO2: هل سيحدث <90 خلال 30 دقيقة؟
    data["is_low"] = (data["spo2"] < THRESH).astype(int)
    horizon_sec = HORIZON_MIN * 60
    future_max = (data["is_low"][::-1].rolling(window=horizon_sec, min_periods=1).max()[::-1])
    data["label_next_30min"] = future_max.astype(int)

    # Features آخر 5 دقائق لكل إشارة
    win = WINDOW_MIN * 60
    for col in ["spo2", "rr", "hr"]:
        data[f"{col}_mean_5m"] = data[col].rolling(win, min_periods=win).mean()
        data[f"{col}_min_5m"]  = data[col].rolling(win, min_periods=win).min()
        data[f"{col}_std_5m"]  = data[col].rolling(win, min_periods=win).std()
        data[f"{col}_slope_5m"]= (data[col] - data[col].shift(win)) / win

    feat_cols = []
    for col in ["spo2", "rr", "hr"]:
        feat_cols += [f"{col}_mean_5m", f"{col}_min_5m", f"{col}_std_5m", f"{col}_slope_5m"]

    out = data.dropna(subset=feat_cols).copy()
    keep = ["caseid", "sec"] + feat_cols + ["label_next_30min"]
    return out[keep]

# --- MAIN ---
trks = pd.read_csv(TRKS_URL)
caseids = pd.read_csv("hypoxemia_caseids.csv")["caseid"].tolist()

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
            print("  skipped (missing/short/empty)")
    except Exception as e:
        print("  failed:", e)

if not all_rows:
    raise SystemExit("No cases built even with 3 signals. We'll expand search next.")

final = pd.concat(all_rows, ignore_index=True)
final.to_csv("train_multicase_3sig.csv", index=False)

print("\nSaved: train_multicase_3sig.csv")
print("Total rows:", len(final))
print("Label distribution:\n", final["label_next_30min"].value_counts())
