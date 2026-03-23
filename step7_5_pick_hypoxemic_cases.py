import pandas as pd

TRKS_URL = "https://api.vitaldb.net/trks"
API_TRACK = "https://api.vitaldb.net/{}"

CHECK_N = 200   # عدد SpO2 tracks اللي نفحصهم بالبداية (نقدر نزيده)
THRESH = 90     # hypoxemia إذا SpO2 أقل من 90

trks = pd.read_csv(TRKS_URL)

# نجيب أول CHECK_N من تراكات SpO2
mask = trks["tname"].astype(str).str.contains("spo2", case=False, na=False)
spo2_trks = trks[mask].copy().head(CHECK_N)

hypo_caseids = []

for _, row in spo2_trks.iterrows():
    tid = row["tid"]
    caseid = row["caseid"]
    try:
        df = pd.read_csv(API_TRACK.format(tid), compression="gzip")
        # عادة أول عمود زمن وثاني عمود قيمة
        vcol = df.columns[1]
        v = pd.to_numeric(df[vcol], errors="coerce").dropna()
        if len(v) == 0:
            continue
        if v.min() < THRESH:
            hypo_caseids.append(caseid)
            print("Found hypoxemia caseid:", caseid, "min SpO2:", float(v.min()))
    except Exception:
        continue

hypo_caseids = sorted(set(hypo_caseids))
out = pd.DataFrame({"caseid": list(hypo_caseids)})

out.to_csv("hypoxemia_caseids.csv", index=False)

print("\nSaved: hypoxemia_caseids.csv")
print("Count hypoxemia cases:", len(out))
print(out.head(20))

