import pandas as pd

TRKS_URL = "https://api.vitaldb.net/trks"

trks = pd.read_csv(TRKS_URL)

# فلترة أي track يحتوي SpO2
mask = trks["tname"].astype(str).str.contains("spo2", case=False, na=False)
spo2_trks = trks[mask].copy()

print("All tracks:", trks.shape)
print("SpO2 tracks found:", spo2_trks.shape[0])

# نعرض أول 20 track
print(spo2_trks[["tid", "caseid", "tname"]].head(20))

# حفظ IDs في ملف
spo2_trks[["tid", "caseid", "tname"]].to_csv("spo2_track_ids.csv", index=False)
print("\nSaved: spo2_track_ids.csv")

