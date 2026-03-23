import pandas as pd

TRKS_URL = "https://api.vitaldb.net/trks"
trks = pd.read_csv(TRKS_URL)

# كلمات بحث للإشارات (أحياناً تختلف أسماء الأجهزة)
sig_map = {
    "spo2": ["SPO2", "SpO2", "PLETH_SPO2"],
    "rr": ["RESP", "RR"],
    "etco2": ["ETCO2", "CO2"],
    "hr": ["HR", "ECG_HR", "HeartRate"]
}

def has_any(series, keys):
    pat = "|".join(keys)
    return series.astype(str).str.contains(pat, case=False, na=False)

# لكل caseid: هل عنده كل إشارة؟
g = trks.groupby("caseid")["tname"]

flags = []
for caseid, names in g:
    names = names.astype(str)
    row = {"caseid": caseid}
    row["has_spo2"] = has_any(names, sig_map["spo2"]).any()
    row["has_rr"] = has_any(names, sig_map["rr"]).any()
    row["has_etco2"] = has_any(names, sig_map["etco2"]).any()
    row["has_hr"] = has_any(names, sig_map["hr"]).any()
    flags.append(row)

flags = pd.DataFrame(flags)
eligible = flags[flags["has_spo2"] & flags["has_rr"] & flags["has_etco2"] & flags["has_hr"]].copy()

print("Eligible cases (have SpO2+RR+ETCO2+HR):", len(eligible))

print("Eligible cases (have SpO2+RR+ETCO2+HR):", len(eligible))

# ناخذ عينة عشوائية من الحالات المؤهلة (أفضل من أول caseid)
N = 50
eligible_sample = eligible.sample(n=min(N, len(eligible)), random_state=42)

eligible_sample.to_csv("eligible_caseids.csv", index=False)
print("Saved: eligible_caseids.csv with", len(eligible_sample), "cases")
print(eligible_sample.head(10))
